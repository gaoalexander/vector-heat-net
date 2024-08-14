import json
import os
import sys
import argparse
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import vector_heat_net
from vector_heat_net.utils import toNP
from vector_heat_net.data.retopo_dataset import RetopoCurvatureDataset

# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks",
                    default='debug')
args = parser.parse_args()

# system things
device = torch.device('cpu')
# device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 8

# model
input_features = args.input_features  # one of ['xyz', 'hks']
k_eig = 128

# training settings
train = not args.evaluate
n_epoch = 5000
lr = 1e-3
decay_every = 50
decay_rate = 0.9
augment_random_rotate = (input_features == 'xyz')
test_every = 10

# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
pretrain_path = os.path.join(base_path, "pretrained_models/test001_{}.pth".format(input_features))
model_save_path = os.path.join(base_path, "data/saved_models/test001_seg_{}.pth".format(input_features))
dataset_path = "/home/jovyan/git/vector-diffusion-net/experiments/quad_meshing/data/torus_small"

# === Load datasets

# Load the test dataset
test_dataset = RetopoCurvatureDataset(dataset_path, split='test', k_eig=k_eig, use_cache=False, op_cache_dir=None)
test_loader = DataLoader(test_dataset, batch_size=None)

C_in = {'xyz': 3, 'hks': 16, 'debug': 1}[input_features]  # dimension of input features

C_width = 1
diffusion_layer = vector_heat_net.layers_vector.LearnedTimeDiffusion(C_width, 'implicit_dense')


# Do an evaluation pass on the test dataset
@torch.no_grad()
def test(diffusion_time):
    diffusion_layer.eval()

    for data in tqdm(test_loader):
        # Get data
        verts, faces, frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY, vert2face_apply_transport, cotan_L, cotan_evals, cotan_evecs, targets, pd1, pd2, pv1, pv2 = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames_verts = frames_verts.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)

        # Construct vector features
        if input_features == 'xyz':
            scalar_features = verts

            # Compute gradients
            scalar_features_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

            # gradient after diffusion
            scalar_features_gradX = torch.mm(gradX, scalar_features)
            scalar_features_gradY = torch.mm(gradY, scalar_features)

            scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
            scalar_features_grad = torch.stack(scalar_features_grads, dim=0)
            # print(gradX.shape, scalar_features.shape, scalar_features_gradX.shape, scalar_features_grad.shape)

            vec_features = torch.transpose(scalar_features_grad, -1, -2)  # [B, V, 2, C]
        elif input_features == 'hks':
            scalar_features = vector_heat_net.geometry_vector.compute_hks_autoscale(evals, evecs, 16)
            vec_features = None
        else:
            vec_features = torch.zeros((1, verts.shape[0], 2))
            vec_features[:, 0, 1] = 0.25

            vec_features = torch.view_as_complex(vec_features)
            vec_features = vec_features.unsqueeze(-1)

        diffusion_layer.set_diffusion_time(diffusion_time)
        diffused = diffusion_layer(vec_features, L, mass, evals, evecs)

    return vec_features, diffused, frames_verts

if __name__ == "__main__":
    diffusion_time = 0.01
    input_vectors, diffused_result, frames_verts = test(diffusion_time)

    output_filepath = f"/home/jovyan/git/vector-diffusion-net/experiments/quad_meshing/output/torus_small_diffusion_time_{diffusion_time}.json"

    input_vectors, diffused_result, frames_verts = toNP(input_vectors), toNP(diffused_result), toNP(frames_verts)
    input_vectors = np.squeeze(input_vectors).real[:, None] * frames_verts[:, 0, :] + \
                    np.squeeze(input_vectors).imag[:, None] * frames_verts[:, 1, :]
    diffused_result = np.squeeze(diffused_result).real[:, None] * frames_verts[:, 0, :] + \
                      np.squeeze(diffused_result).imag[:, None] * frames_verts[:, 1, :]

    with open(output_filepath, "w") as f:
        json.dump(
            {
                "initial_vector_values": input_vectors.tolist(),
                "diffused_vector_values": diffused_result.tolist(),
                "axis_x": frames_verts[:, 0, :].tolist(),
                "axis_y": frames_verts[:, 1, :].tolist(),
                "axis_n": frames_verts[:, 2, :].tolist(),
            },
            f,
        )
