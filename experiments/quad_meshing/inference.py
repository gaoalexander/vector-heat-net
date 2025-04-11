import argparse
import datetime
import json
import numpy as np
import os
import shutil
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import vector_heat_net
from vector_heat_net.dataset.retopo_dataset import RetopoInferenceDataset
from vector_heat_net.utils import toNP
from vector_heat_net.loss import complex_mse_loss, complex_nmse_loss, complex_cosine_loss, size_loss
from vector_heat_net.utils import complex_to_interleaved

# === Options
# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_false", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks",
                    default='hks_grad')
parser.add_argument("--dataset_path",
                    type=str,
                    help="path to dataset directory containing train/test subdirectories",
                    default='experiments/quad_meshing/data/example_spot')
parser.add_argument("--pretrain_path", type=str, help="path to directory containing pretrained checkpoint")
args = parser.parse_args()

# system things
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# model
input_features = args.input_features  # one of ['xyz', 'hks_grad', 'hks_grad_cograd', 'random']
k_eig = 128

# dataset things
rotate_vector_field_operator = torch.exp((0 + 1j) * torch.tensor([torch.pi / 2])).to(device)

# training settings
train = not args.evaluate
n_epoch = 6000
lr = 1e-4
decay_every = 200
decay_rate = 0.85
augment_random_rotate = False  # (input_features == 'xyz')
test_every = 2
save_every = 20

# Important paths
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")

base_path = os.path.dirname(__file__)
pretrain_path = args.pretrain_path
experiment_dir = '/'.join(pretrain_path.split('/')[:-2])
DATASET = experiment_dir.split('/')[-2]
saved_models_path = os.path.join(experiment_dir, "saved_models")
scripts_path = os.path.join(experiment_dir, "scripts")
inference_path = os.path.join(experiment_dir, "inference")
dataset_path = args.dataset_path

os.makedirs(inference_path, exist_ok=True)

# === Load datasets
test_dataset = RetopoInferenceDataset(dataset_path, split='test', k_eig=k_eig, use_cache=True, op_cache_dir=None)
test_loader = DataLoader(test_dataset, batch_size=None)

# === Create the model
C_in = {
    'xyz_grad': 3,
    'hks_grad': 30,
    'mean_curvature': 2
}[args.input_features]  # dimension of input features
model = vector_heat_net.layers.VectorDiffusionNet(C_in=C_in * 2,
                                                              C_out=1,
                                                              C_width=256,
                                                              N_block=6,
                                                              last_activation=None,
                                                              outputs_at='vertices',
                                                              batchnorm=False,
                                                              diffusion_method='spectral')
model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

# Do an evaluation pass on the test dataset
@torch.no_grad()
def test():
    model.eval()
    with torch.no_grad():
        loss_array = []
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY, cotan_L, cotan_evals, \
            cotan_evecs = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames_verts = frames_verts.to(device)
            frames_faces = frames_faces.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            cotan_L = cotan_L.to(device)
            cotan_evals = cotan_evals.to(device)
            cotan_evecs = cotan_evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)

            # Construct input vector features
            if args.input_features == 'xyz_grad':
                scalar_features = verts

                scalar_features_grads = []
                scalar_features_gradX = torch.mm(gradX, scalar_features)
                scalar_features_gradY = torch.mm(gradY, scalar_features)

                scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
                scalar_features_grad = torch.stack(scalar_features_grads, dim=0)
                vec_features = torch.view_as_complex(scalar_features_grad)
            elif args.input_features == 'hks_grad':
                scalar_features = vector_heat_net.geometry.compute_hks_autoscale(cotan_evals, cotan_evecs, C_in)
                scalar_features = scalar_features.to(dtype=torch.float32)

                scalar_features_grads = []
                scalar_features_gradX = torch.mm(gradX, scalar_features)
                scalar_features_gradY = torch.mm(gradY, scalar_features)

                scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
                scalar_features_grad = torch.stack(scalar_features_grads, dim=0)
                scalar_features_grad = (scalar_features_grad /
                                        torch.std(torch.linalg.norm(scalar_features_grad, axis=3), dim=1)[:, None, :,
                                        None])

                grad_features = torch.view_as_complex(scalar_features_grad)
                cograd_features = rotate_vector_field_operator * grad_features
                vec_features = torch.cat((grad_features, cograd_features), dim=-1)
            elif args.input_features == 'mean_curvature':
                mean_curvature = (pv1 + pv2) / 2.0
                k_grads = []
                k_gradX = torch.mm(gradX, mean_curvature)
                k_gradY = torch.mm(gradY, mean_curvature)
                k_grads.append(torch.stack((k_gradX, k_gradY), dim=-1))
                k_grad = torch.stack(k_grads, dim=0)
                k_grad = k_grad / torch.linalg.norm(k_grad, dim=3).mean()
                k_grad = torch.view_as_complex(k_grad.to(dtype=torch.float32)).squeeze(0)
                k_grad_rotated = rotate_vector_field_operator * k_grad
                vec_features = torch.stack((k_grad, k_grad_rotated), dim=1).squeeze(2)[None, :, :]
            else:
                vec_features = None

            # Apply the model
            preds_verts = model(vec_features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                faces=faces)

    return preds_verts, frames_verts, frames_faces, grad_features


print("Running inference...")
test_preds, test_frames_verts, test_frames_faces, test_grad_features = test()

# save test predictions
test_preds, test_frames_verts, test_frames_faces = toNP(test_preds), toNP(test_frames_verts), toNP(test_frames_faces)
test_grad_features = toNP(test_grad_features)[0, :, 0][:, None]

test_preds_3d = np.squeeze(test_preds).real[:, None] * test_frames_verts[:, 0, :] + \
                np.squeeze(test_preds).imag[:, None] * test_frames_verts[:, 1, :]
test_grad_features_3d = np.squeeze(test_grad_features).real[:, None] * test_frames_verts[:, 0, :] + \
                        np.squeeze(test_grad_features).imag[:, None] * test_frames_verts[:, 1, :]

output_filepath = os.path.join(inference_path, f"output.json")

with open(output_filepath, "w") as f:
    json.dump(
        {
            "preds": test_preds_3d.tolist(),
            "preds_local": toNP(complex_to_interleaved(torch.tensor(test_preds)).squeeze(0)).tolist(),
            "axis_x_verts": test_frames_verts[:, 0, :].tolist(),
            "axis_y_verts": test_frames_verts[:, 1, :].tolist(),
            "axis_n_verts": test_frames_verts[:, 2, :].tolist(),
            "axis_x_faces": test_frames_faces[:, 0, :].tolist(),
            "axis_y_faces": test_frames_faces[:, 1, :].tolist(),
            "axis_n_faces": test_frames_faces[:, 2, :].tolist(),
            "grad_features": test_grad_features_3d.tolist()
        },
        f,
    )
