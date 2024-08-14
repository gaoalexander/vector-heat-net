import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import vector_heat_net
from vector_diffusion_dataset_equivariance_test import VectorDiffusionDataset
from vector_heat_net.utils import toNP
from vector_heat_net.layers_vector import complex_to_interleaved

from torch.utils.tensorboard import SummaryWriter

# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default='hks_grad_cograd')
parser.add_argument("--target_type", type=str, help="what features to use as target", default='grads_rotated')
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
rotate_vector_field_operator = torch.exp((0+1j) * torch.tensor([torch.pi / 2])).to(device)
normalize_targets = True

# training settings
train = not args.evaluate
n_epoch = 800000
lr = 1e-4
decay_every = 1000
decay_rate = 0.99
augment_random_rotate = False #(input_features == 'xyz')
test_every = 1
save_every = 500

# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
pretrain_path = os.path.join(base_path, "pretrained_models/test001_{}.pth".format(input_features))
model_save_path = os.path.join(base_path, "data/saved_models/test001_seg_{}.pth".format(input_features))
# dataset_path = "/home/jovyan/git/aas2/avatar-auto-setup/avatar_auto_setup/modules/data/heads_080123/preprocessed_10k"
dataset_path = "/home/jovyan/git/vector-diffusion-net/experiments/vector_diffusion/data/test001/meshes"
tb_path = os.path.join(base_path, "data/tb")

# Setup tensorboard for logging
writer = SummaryWriter(tb_path)

# === Load datasets
if train:
    train_dataset = VectorDiffusionDataset(dataset_path, train=True, k_eig=k_eig, use_cache=False, op_cache_dir=None)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

test_dataset = VectorDiffusionDataset(dataset_path, train=False, k_eig=k_eig, use_cache=False, op_cache_dir=None)
test_loader = DataLoader(test_dataset, batch_size=None)

# for data1 in train_loader:
#     verts, faces, frames_verts, frames_faces, mass, L1, evals1, evecs1, gradX, gradY, vert2face_apply_transport, cotan_L1, cotan_evals1, cotan_evecs1 = data1

# for data2 in test_loader:
#     verts, faces, frames_verts, frames_faces, mass, L2, evals2, evecs2, gradX, gradY, vert2face_apply_transport, cotan_L2, cotan_evals2, cotan_evecs2 = data2

# L_diff = toNP(L1.to_dense()) - toNP(L2.to_dense())
# rows, cols = np.nonzero(L_diff)
# print(L_diff.shape)
# print("L: ", L_diff[rows, cols])

# cotan_L_diff = toNP(cotan_L1.to_dense()) - toNP(cotan_L2.to_dense())
# rows, cols = np.nonzero(cotan_L_diff)
# print(cotan_L_diff.shape)
# print("cotan_L: ", cotan_L_diff[rows, cols])

# # print("evals: ", (toNP(evals1.to_dense()) - toNP(evals2.to_dense())))
# # print("evecs: ", (toNP(evecs2.to_dense()) - toNP(evecs2.to_dense())))

# exit()
    
# === Create the model
C_in = {'xyz': 3, 'hks_grad': 15, 'hks_grad_cograd': 15, 'random': 3}[input_features]  # dimension of input features

if input_features == 'hks_grad_cograd':
    model = vector_heat_net.layers_vector.VectorDiffusionNet(C_in=C_in * 2,
                                                                  C_out=1,
                                                                  C_width=128,
                                                                  N_block=4,
                                                                  last_activation=None,
                                                                  outputs_at='vertices',
                                                                  dropout=False,
                                                                  diffusion_method='spectral')
else:
    model = vector_heat_net.layers_vector.VectorDiffusionNet(C_in=C_in,
                                                                  C_out=1,
                                                                  C_width=128,
                                                                  N_block=4,
                                                                  last_activation=None,
                                                                  outputs_at='vertices',
                                                                  dropout=False,
                                                                  diffusion_method='spectral')
model = model.to(device)

for data in test_loader:
    verts, faces, frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY, vert2face_apply_transport, cotan_L, cotan_evals, cotan_evecs = data

    # Move to device
    verts = verts.to(device)
    faces = faces.to(device)
    frames_verts = frames_verts.to(device)
    frames_faces = frames_faces.to(device)
    mass = mass.to(device)
    L = L.to(device)
    evals = evals.to(device)
    evecs = evecs.to(device)
    gradX = gradX.to(device)
    gradY = gradY.to(device)

    scalar_features = verts[:, 0][:, None]

    # Compute gradients
    scalar_features_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

    # gradient after diffusion
    scalar_features_gradX = torch.mm(gradX, scalar_features)
    scalar_features_gradY = torch.mm(gradY, scalar_features)

    scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
    scalar_features_grad = torch.stack(scalar_features_grads, dim=0)
    grads_rotated = torch.view_as_complex(scalar_features_grad).squeeze(0)
    print(grads_rotated)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def complex_mse_loss(output, target):    
    return (torch.abs(output.squeeze() - target.squeeze()) ** 2)

def train_epoch(epoch, normalize_targets=False):
    # lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr
        lr *= decay_rate
        writer.add_scalar('learning rate', lr, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    model.train()
    optimizer.zero_grad()

    for data in tqdm(train_loader):

        # Get data
        verts, faces, frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY, vert2face_apply_transport, cotan_L, cotan_evals, cotan_evecs = data
        
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
        vert2face_apply_transport = vert2face_apply_transport.to(device)
        
        # Construct target features
        if target_type == 'grads_rotated':
            targets = grads_rotated
            targets = targets / torch.linalg.norm(targets.detach(), dim=1)[:, None]
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

        # Construct input vector features
        if input_features == 'xyz':
            scalar_features = verts
            
            # Compute gradients
            scalar_features_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

            # gradient after diffusion
            scalar_features_gradX = torch.mm(gradX, scalar_features)
            scalar_features_gradY = torch.mm(gradY, scalar_features)

            scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
            scalar_features_grad = torch.stack(scalar_features_grads, dim=0)
            vec_features = torch.view_as_complex(scalar_features_grad)
        elif input_features == 'hks_grad':
            scalar_features = vector_heat_net.geometry_vector.compute_hks_autoscale(evals, evecs, C_in)
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
            scalar_features_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

            # gradient after diffusion
            scalar_features_gradX = torch.mm(gradX, scalar_features)
            scalar_features_gradY = torch.mm(gradY, scalar_features)

            scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
            scalar_features_grad = torch.stack(scalar_features_grads, dim=0)
            
            vec_features = torch.view_as_complex(scalar_features_grad)
        else:
            vec_features = None
        
        # Apply the model
        preds_verts = model(vec_features , mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        
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
            
#         n_vectors = 4
#         preds_faces = preds_faces ** n_vectors
#         targets = targets ** n_vectors
#         loss = complex_mse_loss(preds_faces, targets)


#         print(f"\n\nTrain Preds {preds_verts.shape}: \n{torch.squeeze(preds_verts)}\n\n")
#         print(f"\n\nTrain Targets {targets.shape}: \n{torch.squeeze(targets)}\n\n")

        per_vertex_loss = complex_mse_loss(preds_verts, targets)
        loss = per_vertex_loss.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return per_vertex_loss, loss, preds_verts, targets, frames_verts, frames_faces, vert2face_apply_transport, grad_features


# Do an evaluation pass on the test dataset
@torch.no_grad()
def test(normalize_targets=False):
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY, vert2face_apply_transport, cotan_L, cotan_evals, cotan_evecs = data

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
            vert2face_apply_transport = vert2face_apply_transport.to(device)

            # Construct target features
            if target_type == 'grads_rotated':
                targets = grads_rotated
                targets = targets / torch.linalg.norm(targets.detach(), dim=1)[:, None]
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
                scalar_features = vector_heat_net.geometry_vector.compute_hks_autoscale(evals, evecs, C_in)
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
                scalar_features_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching

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

            per_vertex_loss = complex_mse_loss(preds_verts, targets)
            loss = per_vertex_loss.mean()
            
    return per_vertex_loss, loss, preds_verts, targets, frames_verts, frames_faces, vert2face_apply_transport, grad_features


if train:
    print("Training...")

    writer.add_scalar('learning rate', lr, 0)

    for epoch in range(n_epoch):
        per_vertex_train_loss, train_loss, train_preds, train_targets, train_frames_verts, train_frames_faces, vert2face_apply_transport, train_grad_features = train_epoch(epoch, normalize_targets)
        
        writer.add_scalar('training loss', torch.log(train_loss), epoch)

        if epoch % test_every == 0:
            per_vertex_test_loss, test_loss, test_preds, test_targets, test_frames_verts, test_frames_faces, vert2face_apply_transport, test_grad_features = test(normalize_targets)
            writer.add_scalar('test loss', torch.log(test_loss), epoch)
            print("Epoch {} - Train overall: {}  Test overall: {}".format(epoch, train_loss, test_loss))
        else:
            print("Epoch {} - Train overall: {}".format(epoch, train_loss))
        
        if epoch > 0 and epoch % save_every == 0:
            # save model
            print(" ==> saving last model to " + model_save_path)
            torch.save(model.state_dict(), model_save_path)

            # save train predictions
            train_preds, train_targets, train_frames_verts, train_frames_faces = toNP(train_preds), toNP(train_targets), toNP(train_frames_verts), toNP(train_frames_faces)
            train_grad_features = toNP(train_grad_features)[0, :, 0][:, None]
            vert2face_apply_transport = toNP(vert2face_apply_transport)
            train_preds_3d = np.squeeze(train_preds).real[:, None] * train_frames_verts[:, 0, :] + \
                             np.squeeze(train_preds).imag[:, None] * train_frames_verts[:, 1, :]
            train_targets_3d = np.squeeze(train_targets).real[:, None] * train_frames_verts[:, 0, :] + \
                               np.squeeze(train_targets).imag[:, None] * train_frames_verts[:, 1, :]
            train_grad_features_3d = np.squeeze(train_grad_features).real[:, None] * train_frames_verts[:, 0, :] + \
                                       np.squeeze(train_grad_features).imag[:, None] * train_frames_verts[:, 1, :]
            
            output_filepath = os.path.join(base_path, f"data/outputs/output_train_{epoch}.json")
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
                        "vert2face_apply_transport": np.log(vert2face_apply_transport).imag.tolist(),
                        "per_vertex_loss": per_vertex_train_loss.tolist(),
                        "grad_features": train_grad_features_3d.tolist()
                    },
                    f,
                )

            # save test predictions
            test_preds, test_targets, test_frames_verts, test_frames_faces = toNP(test_preds), toNP(test_targets), toNP(test_frames_verts), toNP(test_frames_faces)
            test_grad_features = toNP(test_grad_features)[0, :, 0][:, None]
            test_preds_3d = np.squeeze(test_preds).real[:, None] * test_frames_verts[:, 0, :] + \
                            np.squeeze(test_preds).imag[:, None] * test_frames_verts[:, 1, :]
            test_targets_3d = np.squeeze(test_targets).real[:, None] * test_frames_verts[:, 0, :] + \
                              np.squeeze(test_targets).imag[:, None] * test_frames_verts[:, 1, :]
            test_grad_features_3d = np.squeeze(test_grad_features).real[:, None] * test_frames_verts[:, 0, :] + \
                           np.squeeze(test_grad_features).imag[:, None] * test_frames_verts[:, 1, :]

            output_filepath = os.path.join(base_path, f"data/outputs/output_test_{epoch}.json")
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
test_loss = test()
print("Overall test loss: {}".format(test_loss))

writer.close()