import os
import sys
import json
import shutil
import argparse
import torch
import datetime
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import igl

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import vector_heat_net
from vector_heat_net.data.retopo_dataset import RetopoCurvatureDataset
from vector_heat_net.utils import toNP
from vector_heat_net.loss import complex_mse_loss, complex_nmse_loss, complex_cosine_loss, size_loss
from vector_heat_net.layers_vector import complex_to_interleaved

from torch.utils.tensorboard import SummaryWriter

# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks",
                    default='hks_grad_cograd')
parser.add_argument("--target_type", type=str, help="what features to use as target", default='gt')
args = parser.parse_args()

# system things
# device = torch.device('cpu') 
device = torch.device('cuda:0')
dtype = torch.float32

# model
input_features = args.input_features  # one of ['xyz', 'hks_grad', 'hks_grad_cograd', 'random']
target_type = args.target_type
k_eig = 128

# dataset things
target_scalars = torch.rand((5002, 2)).to(device)
target_scalars = (target_scalars - 0.5).detach()
input_scalars = torch.rand((5002, 3)).to(device)
rotate_vector_field_operator = torch.exp((0 + 1j) * torch.tensor([torch.pi / 2])).to(device)
normalize_targets = False

# training settings
train = not args.evaluate
n_epoch = 4500
lr = 1e-4
decay_every = 150
decay_rate = 0.85
augment_random_rotate = False  # (input_features == 'xyz')
test_every = 2
save_every = 20

# # training settings
# train = not args.evaluate
# n_epoch = 400000
# lr = 3 * 1e-4
# decay_every = 20000
# decay_rate = 0.85
# augment_random_rotate = False #(input_features == 'xyz')
# test_every = 2
# save_every = 1000

# training settings
# train = not args.evaluate
# n_epoch = 600000
# lr = 5 * 1e-4
# decay_every = 1000
# decay_rate = 0.99
# augment_random_rotate = False #(input_features == 'xyz')
# test_every = 10
# save_every = 10000

# Important paths
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")

DATASET = "HEADS"
base_path = os.path.dirname(__file__)
pretrain_path = os.path.join(base_path, "output/{}/{}/saved_models/{}.pth".format(DATASET, "20240104_0755", "4480"))
experiment_directory_path = os.path.join(base_path, f"output/{formatted_datetime}/")
saved_models_path = os.path.join(experiment_directory_path, "saved_models")
scripts_path = os.path.join(experiment_directory_path, "scripts")
pred_json_path = os.path.join(experiment_directory_path, "pred_json")
op_cache_dir = os.path.join(base_path, "data", "op_cache")
model_save_path = os.path.join(saved_models_path, ".pth")
# 
# dataset_path = "/home/jovyan/git/vector-diffusion-net/experiments/quad_meshing/data/heads_verts_120823/preprocessed"
dataset_path = "/home/jovyan/git/vector-diffusion-net/experiments/quad_meshing/data/heads_test"
# dataset_path = "/home/jovyan/git/aas2/avatar-auto-setup/avatar_auto_setup/modules/data/heads_080123/preprocessed_10k"
# dataset_path = "/home/jovyan/git/vector-diffusion-net/experiments/quad_meshing/data/heads_test_03_easy"
# dataset_path = "/home/jovyan/git/vector-diffusion-net/experiments/vector_diffusion/data/heads_equivariance_test"

tb_path = os.path.join(base_path, f"output/tb/{formatted_datetime}")

# Setup tensorboard for logging
writer = SummaryWriter(tb_path)

# === Load datasets
if train:
    os.makedirs(experiment_directory_path, exist_ok=True)
    os.makedirs(saved_models_path, exist_ok=True)
    os.makedirs(scripts_path, exist_ok=True)
    os.makedirs(pred_json_path, exist_ok=True)

    # copy scripts for better reproducibility and change tracking
    shutil.copy2(os.path.join(base_path, "train_retopo_verts.py"), os.path.join(scripts_path, "train_retopo_verts.py"))
    shutil.copy2(os.path.join(base_path, "vector_diffusion_dataset.py"),
                 os.path.join(scripts_path, "vector_diffusion_dataset.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/geometry.py"),
                 os.path.join(scripts_path, "geometry.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/layers.py"),
                 os.path.join(scripts_path, "layers.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/loss.py"), os.path.join(scripts_path, "loss.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/utils.py"), os.path.join(scripts_path, "utils.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/nn/mlp.py"), os.path.join(scripts_path, "mlp.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/nn/nonlin.py"), os.path.join(scripts_path, "nonlin.py"))

    train_dataset = RetopoInferenceDataset(dataset_path, split='train', k_eig=k_eig, use_cache=True, op_cache_dir=None)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

test_dataset = RetopoInferenceDataset(dataset_path, split='test', k_eig=k_eig, use_cache=True, op_cache_dir=None)
test_loader = DataLoader(test_dataset, batch_size=None)

# === Create the model
C_in = {'xyz': 3, 'hks_grad': 15, 'hks_grad_cograd': 15, 'random': 3}[input_features]  # dimension of input features

if input_features == 'hks_grad_cograd':
    model = vector_heat_net.layers_vector.VectorDiffusionNet(C_in=C_in * 2,
                                                                  C_out=1,
                                                                  C_width=256,
                                                                  N_block=6,
                                                                  last_activation=None,
                                                                  outputs_at='vertices',
                                                                  dropout=True,
                                                                  batchnorm=False,
                                                                  diffusion_method='spectral')
else:
    model = vector_heat_net.layers_vector.VectorDiffusionNet(C_in=C_in,
                                                                  C_out=1,
                                                                  C_width=128,
                                                                  N_block=4,
                                                                  last_activation=None,
                                                                  outputs_at='vertices',
                                                                  dropout=True,
                                                                  diffusion_method='spectral')
model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)


def train_epoch(epoch, normalize_targets=False):
    # lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr
        lr *= decay_rate
        writer.add_scalar('learning rate', lr, epoch * len(train_loader))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    model.train()
    optimizer.zero_grad()

    loss_array = []
    for data in tqdm(train_loader):

        # Get data
        verts, faces, frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY, vert2face_apply_transport, cotan_L, cotan_evals, cotan_evecs, targets = data

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
        #         vert2face_apply_transport = vert2face_apply_transport.to(device)

        # Construct target features
        if target_type == 'grads_rotated':
            targets = grads_rotated
            targets = targets.to(device)
        elif target_type == 'eigenbasis':
            targets = evecs[:, 0][None, :, None]
            targets = targets.to(device)
        elif target_type == 'grad_random':
            target_scalars_grads = []
            target_scalars_gradX = torch.mm(gradX, target_scalars)
            target_scalars_gradY = torch.mm(gradY, target_scalars)

            target_scalars_grads.append(torch.stack((target_scalars_gradX, target_scalars_gradY), dim=-1))
            target_scalars_grad = torch.stack(target_scalars_grads, dim=0)
            targets = torch.view_as_complex(target_scalars_grad)
            targets = targets / torch.linalg.norm(targets.detach(), dim=-1).unsqueeze(-1)
            targets = (rotate_vector_field_operator * targets).detach()
            targets = targets.to(device)
        elif target_type == 'random':
            targets = torch.view_as_complex(target_scalars)
            targets = targets / torch.linalg.norm(targets[:, None], dim=-1)
            targets = targets.detach()
            targets = targets.to(device)
        elif target_type == 'gt':
            targets = targets.to(device)

        if normalize_targets:
            targets = targets / torch.linalg.norm(targets[:, None], dim=-1)
        #         else:
        #             targets *= 10

        # Construct input vector features
        if input_features == 'xyz':
            scalar_features = verts

            # Compute gradients
            scalar_features_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

            # gradient after diffusion
            scalar_features_gradX = torch.mm(gradX, scalar_features)
            scalar_features_gradY = torch.mm(gradY, scalar_features)

            scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
            scalar_features_grad = torch.stack(scalar_features_grads, dim=0)
            vec_features = torch.view_as_complex(scalar_features_grad)
        elif input_features == 'hks_grad':
            scalar_features = vector_heat_net.geometry_vector.compute_hks_autoscale(cotan_evals, cotan_evecs, C_in)
            scalar_features = scalar_features.to(dtype=torch.float32)

            # Compute gradients
            scalar_features_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

            # gradient after diffusion
            scalar_features_gradX = torch.mm(gradX, scalar_features)
            scalar_features_gradY = torch.mm(gradY, scalar_features)

            scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
            scalar_features_grad = torch.stack(scalar_features_grads, dim=0)

            vec_features = torch.view_as_complex(scalar_features_grad)
        elif input_features == 'hks_grad_cograd':
            scalar_features = vector_heat_net.geometry_vector.compute_hks_autoscale(cotan_evals, cotan_evecs, C_in)
            scalar_features = scalar_features.to(dtype=torch.float32)

            # Compute gradients
            scalar_features_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

            # gradient after diffusion
            scalar_features_gradX = torch.mm(gradX, scalar_features)
            scalar_features_gradY = torch.mm(gradY, scalar_features)

            scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
            scalar_features_grad = torch.stack(scalar_features_grads, dim=0)

            grad_features = torch.view_as_complex(scalar_features_grad)
            cograd_features = rotate_vector_field_operator * grad_features
            vec_features = torch.cat((grad_features, cograd_features), dim=-1)
        elif input_features == 'random':
            scalar_features = input_scalars

            # Compute gradients
            scalar_features_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

            # gradient after diffusion
            scalar_features_gradX = torch.mm(gradX, scalar_features)
            scalar_features_gradY = torch.mm(gradY, scalar_features)

            scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
            scalar_features_grad = torch.stack(scalar_features_grads, dim=0)

            vec_features = torch.view_as_complex(scalar_features_grad)
        else:
            vec_features = None

        # Apply the model
        preds_verts = model(vec_features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        #         """
        #         Parallel transport and average vertex-based predictions to get face-based predictions
        #         preds_verts: [V, 1 (complex)]
        #         faces: [F, 3]
        #         """
        #         preds_faces = preds_verts.squeeze()[faces] # [F, 3 (complex)]
        #         preds_faces = vert2face_apply_transport * preds_faces
        #         preds_faces = preds_faces.mean(dim=-1)

        #         preds_faces *= 10
        #         targets *= 10

        n_vectors = 4
        eps = torch.Tensor([1e-8]).to(device)
        #             per_vertex_loss = complex_mse_loss(preds_verts_exp4, targets_exp4)
        targets_normalized = targets / torch.maximum(eps, torch.linalg.norm(targets[:, None], dim=-1))
        per_vertex_loss = size_loss(targets, preds_verts, eps=eps) + complex_cosine_loss(
            targets_normalized ** n_vectors, preds_verts ** n_vectors, eps)

        #         print(f"\n\nTrain Preds {preds_verts.shape}: \n{torch.squeeze(preds_verts)}\n\n")
        #         print(f"\n\nTrain Targets {targets.shape}: \n{torch.squeeze(targets)}\n\n")

        loss = per_vertex_loss.mean()
        #         loss = (per_vertex_loss * mass.real).sum() / mass.real.sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_array.append(loss.detach().cpu().numpy())
    return per_vertex_loss, loss, preds_verts, targets, frames_verts, frames_faces, vert2face_apply_transport, loss_array, grad_features


# Do an evaluation pass on the test dataset
@torch.no_grad()
def test(normalize_targets=False):
    model.eval()
    with torch.no_grad():
        loss_array = []
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY, vert2face_apply_transport, cotan_L, cotan_evals, cotan_evecs, targets = data

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
            #             vert2face_apply_transport = vert2face_apply_transport.to(device)

            # Construct target features
            if target_type == 'grads_rotated':
                targets = grads_rotated
                targets = targets.to(device)
            elif target_type == 'eigenbasis':
                targets = evecs[:, 0][None, :, None]
                targets = targets.to(device)
            elif target_type == 'grad_random':
                target_scalars_grads = []
                target_scalars_gradX = torch.mm(gradX, target_scalars)
                target_scalars_gradY = torch.mm(gradY, target_scalars)

                target_scalars_grads.append(torch.stack((target_scalars_gradX, target_scalars_gradY), dim=-1))
                target_scalars_grad = torch.stack(target_scalars_grads, dim=0)
                targets = torch.view_as_complex(target_scalars_grad)
                targets = targets / torch.linalg.norm(targets.detach(), dim=-1).unsqueeze(-1)
                targets = (rotate_vector_field_operator * targets).detach()
                targets = targets.to(device)
            elif target_type == 'random':
                targets = torch.view_as_complex(target_scalars)
                targets = targets / torch.linalg.norm(targets[:, None], dim=-1)
                targets = targets.detach()
                targets = targets.to(device)
            elif target_type == 'gt':
                targets = targets.to(device)

            if normalize_targets:
                targets = targets / torch.linalg.norm(targets[:, None], dim=-1)
            #             else:
            #                 targets *= 10

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

                vec_features = torch.view_as_complex(scalar_features_grad)
            elif input_features == 'hks_grad':
                scalar_features = vector_heat_net.geometry_vector.compute_hks_autoscale(cotan_evals, cotan_evecs,
                                                                                             C_in)
                scalar_features = scalar_features.to(dtype=torch.float32)

                # Compute gradients
                scalar_features_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

                # gradient after diffusion
                scalar_features_gradX = torch.mm(gradX, scalar_features)
                scalar_features_gradY = torch.mm(gradY, scalar_features)

                scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
                scalar_features_grad = torch.stack(scalar_features_grads, dim=0)

                vec_features = torch.view_as_complex(scalar_features_grad)
            elif input_features == 'hks_grad_cograd':
                scalar_features = vector_heat_net.geometry_vector.compute_hks_autoscale(cotan_evals, cotan_evecs,
                                                                                             C_in)
                scalar_features = scalar_features.to(dtype=torch.float32)

                # Compute gradients
                scalar_features_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

                # gradient after diffusion
                scalar_features_gradX = torch.mm(gradX, scalar_features)
                scalar_features_gradY = torch.mm(gradY, scalar_features)

                scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
                scalar_features_grad = torch.stack(scalar_features_grads, dim=0)

                grad_features = torch.view_as_complex(scalar_features_grad)
                cograd_features = rotate_vector_field_operator * grad_features
                vec_features = torch.cat((grad_features, cograd_features), dim=-1)
            elif input_features == 'random':
                scalar_features = input_scalars

                # Compute gradients
                scalar_features_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

                # gradient after diffusion
                scalar_features_gradX = torch.mm(gradX, scalar_features)
                scalar_features_gradY = torch.mm(gradY, scalar_features)

                scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
                scalar_features_grad = torch.stack(scalar_features_grads, dim=0)

                vec_features = torch.view_as_complex(scalar_features_grad)
            else:
                vec_features = None

            # Apply the model
            preds_verts = model(vec_features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                faces=faces)

            n_vectors = 4
            eps = torch.Tensor([1e-8]).to(device)
            #             per_vertex_loss = complex_mse_loss(preds_verts_exp4, targets_exp4)
            targets_normalized = targets / torch.maximum(eps, torch.linalg.norm(targets[:, None], dim=-1))
            per_vertex_loss = size_loss(targets, preds_verts, eps=eps) + complex_cosine_loss(
                targets_normalized ** n_vectors, preds_verts ** n_vectors, eps)

            loss = per_vertex_loss.mean()
            #             loss = (per_vertex_loss * mass.real).sum() / mass.real.sum()

            loss_array.append(loss.detach().cpu().numpy())
    return per_vertex_loss, loss, preds_verts, targets, frames_verts, frames_faces, vert2face_apply_transport, loss_array, grad_features


if train:
    print("Training...")

    writer.add_scalar('learning rate', lr, 0)

    for epoch in range(n_epoch):
        per_vertex_train_loss, train_loss, train_preds, train_targets, train_frames_verts, train_frames_faces, vert2face_apply_transport, train_loss_array, train_grad_features = train_epoch(
            epoch, normalize_targets)

        writer.add_scalar('training loss', np.log(np.array(train_loss_array).mean()), epoch * len(train_loader))

        if epoch % test_every == 0:
            per_vertex_test_loss, test_loss, test_preds, test_targets, test_frames_verts, test_frames_faces, vert2face_apply_transport, test_loss_array, test_grad_features = test(
                normalize_targets)
            writer.add_scalar('test loss', np.array(test_loss_array).mean(), epoch * len(train_loader))
            print("Epoch {} - Train overall: {}  Test overall: {}".format(epoch, np.array(train_loss_array).mean(),
                                                                          np.array(test_loss_array).mean()))
        else:
            print("Epoch {} - Train overall: {}".format(epoch, np.array(train_loss_array).mean()))

        if epoch > 0 and epoch % save_every == 0:
            # save model
            print(" ==> saving last model to " + model_save_path.replace(".pth", f"{epoch}.pth"))
            torch.save(model.state_dict(), model_save_path.replace(".pth", f"{epoch}.pth"))

            # save train predictions
            train_preds, train_targets, train_frames_verts, train_frames_faces = toNP(train_preds), toNP(
                train_targets), toNP(train_frames_verts), toNP(train_frames_faces)
            train_grad_features = toNP(train_grad_features)[0, :, 0][:, None]

            #             vert2face_apply_transport = toNP(vert2face_apply_transport)
            train_preds_3d = np.squeeze(train_preds).real[:, None] * train_frames_verts[:, 0, :] + \
                             np.squeeze(train_preds).imag[:, None] * train_frames_verts[:, 1, :]
            train_targets_3d = np.squeeze(train_targets).real[:, None] * train_frames_verts[:, 0, :] + \
                               np.squeeze(train_targets).imag[:, None] * train_frames_verts[:, 1, :]
            train_grad_features_3d = np.squeeze(train_grad_features).real[:, None] * train_frames_verts[:, 0, :] + \
                                     np.squeeze(train_grad_features).imag[:, None] * train_frames_verts[:, 1, :]

            output_filepath = os.path.join(base_path,
                                           f"output/{formatted_datetime}/pred_json/output_train_{epoch}.json")
            with open(output_filepath, "w") as f:
                json.dump(
                    {
                        "preds": train_preds_3d.tolist(),
                        "preds_local": toNP(complex_to_interleaved(torch.tensor(train_preds)).squeeze(0)).tolist(),
                        "targets": train_targets_3d.tolist(),
                        "axis_x_verts": train_frames_verts[:, 0, :].tolist(),
                        "axis_y_verts": train_frames_verts[:, 1, :].tolist(),
                        "axis_n_verts": train_frames_verts[:, 2, :].tolist(),
                        "axis_x_faces": train_frames_faces[:, 0, :].tolist(),
                        "axis_y_faces": train_frames_faces[:, 1, :].tolist(),
                        "axis_n_faces": train_frames_faces[:, 2, :].tolist(),
                        #                         "vert2face_apply_transport": np.log(vert2face_apply_transport).imag.tolist(),
                        "per_vertex_loss": per_vertex_train_loss.tolist(),
                        "grad_features": train_grad_features_3d.tolist()
                    },
                    f,
                )

            # save test predictions
            test_preds, test_targets, test_frames_verts, test_frames_faces = toNP(test_preds), toNP(test_targets), toNP(
                test_frames_verts), toNP(test_frames_faces)
            test_grad_features = toNP(test_grad_features)[0, :, 0][:, None]

            test_preds_3d = np.squeeze(test_preds).real[:, None] * test_frames_verts[:, 0, :] + \
                            np.squeeze(test_preds).imag[:, None] * test_frames_verts[:, 1, :]
            test_targets_3d = np.squeeze(test_targets).real[:, None] * test_frames_verts[:, 0, :] + \
                              np.squeeze(test_targets).imag[:, None] * test_frames_verts[:, 1, :]
            test_grad_features_3d = np.squeeze(test_grad_features).real[:, None] * test_frames_verts[:, 0, :] + \
                                    np.squeeze(test_grad_features).imag[:, None] * test_frames_verts[:, 1, :]

            output_filepath = os.path.join(base_path, f"output/{formatted_datetime}/pred_json/output_test_{epoch}.json")
            with open(output_filepath, "w") as f:
                json.dump(
                    {
                        "preds": test_preds_3d.tolist(),
                        "preds_local": toNP(complex_to_interleaved(torch.tensor(test_preds)).squeeze(0)).tolist(),
                        "targets": test_targets_3d.tolist(),
                        "axis_x_verts": test_frames_verts[:, 0, :].tolist(),
                        "axis_y_verts": test_frames_verts[:, 1, :].tolist(),
                        "axis_n_verts": test_frames_verts[:, 2, :].tolist(),
                        "axis_x_faces": test_frames_faces[:, 0, :].tolist(),
                        "axis_y_faces": test_frames_faces[:, 1, :].tolist(),
                        "axis_n_faces": test_frames_faces[:, 2, :].tolist(),
                        "per_vertex_loss": per_vertex_test_loss.tolist(),
                        "grad_features": test_grad_features_3d.tolist()
                    },
                    f,
                )
else:
    print("Running inference...")

    per_vertex_test_loss, test_loss, test_preds, test_targets, test_frames_verts, test_frames_faces, vert2face_apply_transport, test_loss_array, test_grad_features = test(
        normalize_targets)
    print("Test overall: {}".format(np.array(test_loss_array).mean()))

    # save test predictions
    test_preds, test_targets, test_frames_verts, test_frames_faces = toNP(test_preds), toNP(test_targets), toNP(
        test_frames_verts), toNP(test_frames_faces)
    test_grad_features = toNP(test_grad_features)[0, :, 0][:, None]

    test_preds_3d = np.squeeze(test_preds).real[:, None] * test_frames_verts[:, 0, :] + \
                    np.squeeze(test_preds).imag[:, None] * test_frames_verts[:, 1, :]
    test_targets_3d = np.squeeze(test_targets).real[:, None] * test_frames_verts[:, 0, :] + \
                      np.squeeze(test_targets).imag[:, None] * test_frames_verts[:, 1, :]
    test_grad_features_3d = np.squeeze(test_grad_features).real[:, None] * test_frames_verts[:, 0, :] + \
                            np.squeeze(test_grad_features).imag[:, None] * test_frames_verts[:, 1, :]

    output_filepath = os.path.join(base_path, f"output/latest_inference.json")
    with open(output_filepath, "w") as f:
        json.dump(
            {
                "preds": test_preds_3d.tolist(),
                "preds_local": toNP(complex_to_interleaved(torch.tensor(test_preds)).squeeze(0)).tolist(),
                "targets": test_targets_3d.tolist(),
                "axis_x_verts": test_frames_verts[:, 0, :].tolist(),
                "axis_y_verts": test_frames_verts[:, 1, :].tolist(),
                "axis_n_verts": test_frames_verts[:, 2, :].tolist(),
                "axis_x_faces": test_frames_faces[:, 0, :].tolist(),
                "axis_y_faces": test_frames_faces[:, 1, :].tolist(),
                "axis_n_faces": test_frames_faces[:, 2, :].tolist(),
                "per_vertex_loss": per_vertex_test_loss.tolist(),
                "grad_features": test_grad_features_3d.tolist()
            },
            f,
        )

# Test
# test_loss = test()
# print("Overall test loss: {}".format(test_loss))

writer.close()
