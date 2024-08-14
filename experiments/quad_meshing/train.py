import argparse
import datetime
import json

import igl
import numpy as np
import os
import shutil
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import vector_heat_net
from vector_heat_net.dataset.retopo_dataset import RetopoCurvatureDataset
from vector_heat_net.utils import toNP, convert_intrinsic2d_to_extrinsic3d, save_predictions
from vector_heat_net.loss import complex_mse_loss, complex_nmse_loss, complex_cosine_loss, size_loss
from vector_heat_net.layers import complex_to_interleaved
from torch.utils.tensorboard import SummaryWriter


def save_code_checkpoint(base_path, scripts_path):
    os.makedirs(experiment_directory_path, exist_ok=True)
    os.makedirs(saved_models_path, exist_ok=True)
    os.makedirs(scripts_path, exist_ok=True)
    os.makedirs(pred_json_path, exist_ok=True)

    # copy scripts for better reproducibility and change tracking
    shutil.copy2(os.path.join(base_path, "train.py"),
                 os.path.join(scripts_path, "train.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/dataset/retopo_dataset.py"),
                 os.path.join(scripts_path, "retopo_dataset.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/geometry.py"),
                 os.path.join(scripts_path, "geometry.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/layers.py"),
                 os.path.join(scripts_path, "layers.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/loss.py"),
                 os.path.join(scripts_path, "loss.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/utils.py"),
                 os.path.join(scripts_path, "utils.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/nn/mlp.py"),
                 os.path.join(scripts_path, "mlp.py"))
    shutil.copy2(os.path.join("src/vector_heat_net/nn/nonlin.py"),
                 os.path.join(scripts_path, "nonlin.py"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="evaluate using the pretrained model")
    parser.add_argument("--dropout",
                        action="store_true",
                        help="use dropout during model training")
    parser.add_argument("--normalize_targets",
                        action="store_true",
                        help="normalize target direction vectors to unit length (default false)")
    parser.add_argument("--input_features",
                        type=str,
                        help="what features to use as input ('xyz_grad' or 'hks_grad') default: hks_grad",
                        default='hks_grad')
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to dataset directory containing train/test subdirectories",
                        default='experiments/quad_meshing/data/example_spot')
    parser.add_argument("--pretrain_path",
                        type=str,
                        help="path to saved run directory",
                        default='')
    parser.add_argument("--k_eig",
                        type=int,
                        help="number of eigenvectors to use in spectral approximation",
                        default=128)
    parser.add_argument("--n_epoch",
                        type=int,
                        help="number of training epochs",
                        default=60000)
    parser.add_argument("--lr",
                        type=float,
                        help="learning rate",
                        default=0.0001)
    parser.add_argument("--decay_every",
                        type=int,
                        help="number of epochs between learning rate decay",
                        default=400)
    parser.add_argument("--decay_rate",
                        type=float,
                        help="amount of learning rate decay",
                        default=0.95)
    parser.add_argument("--val_every",
                        type=int,
                        help="test every n epochs",
                        default=20)
    parser.add_argument("--save_every",
                        type=int,
                        help="save every n epochs",
                        default=400)
    return parser.parse_args()


args = get_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
rotate_vector_field_operator = torch.exp((0 + 1j) * torch.tensor([torch.pi / 2])).to(device)
train = not args.evaluate
lr = args.lr

# Important paths
DATASET = args.dataset_path.split('/')[-1]
formatted_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
base_path = os.path.dirname(__file__)
experiment_directory_path = os.path.join(base_path, f"output/{DATASET}/{formatted_datetime}/")
saved_models_path = os.path.join(experiment_directory_path, "saved_models")
scripts_path = os.path.join(experiment_directory_path, "scripts")
pred_json_path = os.path.join(experiment_directory_path, "pred_json")
op_cache_dir = os.path.join(base_path, "data", "op_cache")
model_save_path = os.path.join(saved_models_path, ".pth")

writer = SummaryWriter(os.path.join(base_path, f"output/tb/{formatted_datetime}"))  # Setup tensorboard for logging

if train:
    train_dataset = RetopoCurvatureDataset(args.dataset_path, split='train', k_eig=args.k_eig, use_cache=True, op_cache_dir=None, device=device)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    save_code_checkpoint(base_path, scripts_path)
test_dataset = RetopoCurvatureDataset(args.dataset_path, split='test', k_eig=args.k_eig, use_cache=True, op_cache_dir=None, device=device)
test_loader = DataLoader(test_dataset, batch_size=None)

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
                                                              dropout=args.dropout,
                                                              batchnorm=False,
                                                              diffusion_method='spectral')
model = model.to(device)

if not train:
    print("Loading pretrained model from: " + str(args.pretrain_path))
    model.load_state_dict(torch.load(args.pretrain_path))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)


def train_epoch(epoch, normalize_targets=False):
    if epoch > 0 and epoch % args.decay_every == 0:  # lr decay
        global lr
        lr *= args.decay_rate
        writer.add_scalar('learning rate', lr, epoch * len(train_loader))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    model.train()
    optimizer.zero_grad()

    loss_array = []
    for data in tqdm(train_loader):
        (verts, faces, frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY,
         cotan_L, cotan_evals, cotan_evecs, targets, pd1, pd2, pv1, pv2) = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames_verts = frames_verts.to(device)
        frames_faces = frames_faces.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        cotan_evals = cotan_evals.to(device)
        cotan_evecs = cotan_evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        targets = targets.to(device)

        if normalize_targets:
            targets = targets / torch.linalg.norm(targets[:, None], dim=-1)

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
                                    torch.std(torch.linalg.norm(scalar_features_grad, axis=3), dim=1)[:, None, :, None])

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

        # apply model
        preds_verts = model(vec_features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        # compute loss
        n_vectors = 4
        target_scale_factor = 10
        eps = torch.Tensor([1e-8]).to(device)

        # normalize targets solely to make cross field direction loss more stable
        targets_normalized = targets / torch.maximum(eps, torch.linalg.norm(targets[:, None], dim=-1))
        per_vertex_loss = (size_loss(targets * target_scale_factor, preds_verts, eps=eps) +
                           complex_cosine_loss(targets_normalized ** n_vectors, preds_verts ** n_vectors, eps))
        loss = per_vertex_loss.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_array.append(loss.detach().cpu().numpy())
    return {
        "per_vertex_loss": per_vertex_loss, 
        "loss": loss, 
        "preds_verts": preds_verts, 
        "targets": targets, 
        "frames_verts": frames_verts, 
        "frames_faces": frames_faces, 
        "loss_array": loss_array
    }


# Do an evaluation pass on the test dataset
@torch.no_grad()
def test(normalize_targets=False):
    model.eval()
    with torch.no_grad():
        loss_array = []
        for data in tqdm(test_loader):
            (verts, faces, frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY, 
             cotan_L, cotan_evals, cotan_evecs, targets, pd1, pd2, pv1, pv2) = data

            verts = verts.to(device)
            faces = faces.to(device)
            frames_verts = frames_verts.to(device)
            frames_faces = frames_faces.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            cotan_evals = cotan_evals.to(device)
            cotan_evecs = cotan_evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            targets = targets.to(device)

            if normalize_targets:
                targets = targets / torch.linalg.norm(targets[:, None], dim=-1)

            # Construct vector features
            if args.input_features == 'xyz_grad':
                scalar_features = verts

                scalar_features_grads = []
                scalar_features_gradX = torch.mm(gradX, scalar_features)
                scalar_features_gradY = torch.mm(gradY, scalar_features)

                scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
                scalar_features_grad = torch.stack(scalar_features_grads, dim=0)

                vec_features = torch.view_as_complex(scalar_features_grad)
            elif args.input_features == 'hks_grad':
                scalar_features = vector_heat_net.geometry.compute_hks_autoscale(cotan_evals, cotan_evecs,
                                                                                             C_in)
                scalar_features = scalar_features.to(dtype=torch.float32)

                # Compute gradients
                scalar_features_grads = []

                # gradient after diffusion
                scalar_features_gradX = torch.mm(gradX, scalar_features)
                scalar_features_gradY = torch.mm(gradY, scalar_features)

                scalar_features_grads.append(torch.stack((scalar_features_gradX, scalar_features_gradY), dim=-1))
                scalar_features_grad = torch.stack(scalar_features_grads, dim=0)

                scalar_features_grad = scalar_features_grad / torch.std(torch.linalg.norm(scalar_features_grad, axis=3),
                                                                        dim=1)[:, None, :, None]

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

            # apply model
            preds_verts = model(vec_features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            # compute loss
            n_vectors = 4
            target_scale_factor = 10
            eps = torch.Tensor([1e-8]).to(device)

            targets_normalized = targets / torch.maximum(eps, torch.linalg.norm(targets[:, None], dim=-1))
            per_vertex_loss = (size_loss(targets * target_scale_factor, preds_verts, eps=eps) +
                               complex_cosine_loss(targets_normalized ** n_vectors, preds_verts ** n_vectors, eps))
            loss = per_vertex_loss.mean()
            loss_array.append(loss.detach().cpu().numpy())
    return {
        "per_vertex_loss": per_vertex_loss, 
        "loss": loss, 
        "preds_verts": preds_verts, 
        "targets": targets, 
        "frames_verts": frames_verts, 
        "frames_faces": frames_faces, 
        "loss_array": loss_array
    }


if train:
    print("Training...")
    writer.add_scalar('learning rate', lr, 0)

    for epoch in range(args.n_epoch):
        train_out = train_epoch(epoch, args.normalize_targets)
        writer.add_scalar('training loss', np.log(np.array(train_out["loss_array"]).mean()), epoch * len(train_loader))

        if epoch % args.val_every != 0:
            print("Epoch {} - Train overall: {}".format(epoch, np.array(train_out["loss_array"]).mean()))
        else:
            val_out = test(args.normalize_targets)
            writer.add_scalar('val loss', np.array(val_out["loss_array"]).mean(), epoch * len(train_loader))
            print("Epoch {} - Train overall: {}  Val overall: {}".format(epoch, np.array(train_out["loss_array"]).mean(), np.array(val_out["loss_array"]).mean()))

        if epoch > 0 and epoch % args.save_every == 0:
            print(" ==> saving last model to " + model_save_path.replace(".pth", f"{epoch}.pth"))
            torch.save(model.state_dict(), model_save_path.replace(".pth", f"{epoch}.pth"))

            save_predictions(
                train_out["preds_verts"],
                train_out["targets"],
                train_out["frames_verts"],
                train_out["frames_faces"],
                train_out["per_vertex_loss"],
                output_filepath=os.path.join(base_path, f"output/{DATASET}/{formatted_datetime}/pred_json/output_train_{epoch}.json")
            )

            save_predictions(
                val_out["preds_verts"],
                val_out["targets"],
                val_out["frames_verts"],
                val_out["frames_faces"],
                val_out["per_vertex_loss"],
                output_filepath=os.path.join(base_path, f"output/{DATASET}/{formatted_datetime}/pred_json/output_val_{epoch}.json")
            )
else:
    print("Running inference...")
    test_out = test(args.normalize_targets)
    print("Test overall: {}".format(np.array(test_out["loss_array"]).mean()))
    
    save_predictions(
        test_out["preds_verts"],
        test_out["targets"],
        test_out["frames_verts"],
        test_out["frames_faces"],
        test_out["per_vertex_loss"],
        output_filepath=os.path.join(base_path, f"output/latest_inference.json")
    )

writer.close()
