import json
import shutil
import os
import sys
import random
import h5py
import igl
import numpy as np

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import vector_heat_net
from vector_heat_net.utils import toNP
from vector_heat_net.geometry import global_to_local, global_to_local_batch
from pathlib import Path
from tqdm import tqdm


class RetopoDataset(Dataset):
    def __init__(self,
                 root_dir,
                 split='train',
                 k_eig=128,
                 use_cache=True,
                 op_cache_dir=None):
        self.train = (split == 'train')  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.input_vecs_list = []
        self.targets_list = []  # per-face vector targets

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list, self.targets_list = torch.load(
                    load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        target_files = []

        # Train test split
        if split == 'train':
            mesh_dirpath = os.path.join(self.root_dir, "train")
            target_dirpath = os.path.join(self.root_dir, "train")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)
        else:
            mesh_dirpath = os.path.join(self.root_dir, "test")
            target_dirpath = os.path.join(self.root_dir, "test")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)

        print("loading {} meshes".format(len(mesh_files)))
        print(mesh_files)

        # Load the actual files
        for mesh_file, target_file in tqdm(zip(mesh_files, target_files)):
            assert ".obj" in mesh_file
            assert ".h5" in target_file

            print("loading mesh " + str(mesh_file))
            verts, faces = pp3d.read_mesh(mesh_file)

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
#             verts, scale = vector_heat_net.geometry_vector.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            
            print("loading target " + str(target_file))
            
            target_directions = {}
            with h5py.File(target_file, "r") as h5_file:
                for key, dataset in h5_file["data"].items():
                    target_directions[key] = dataset[0][()]
            targets_u = np.array(target_directions["u"])
            targets_v = np.array(target_directions["v"])

            u_norm = np.linalg.norm(targets_u, axis=-1)[:, None]
            v_norm = np.linalg.norm(targets_v, axis=-1)[:, None]

            targets = np.where(u_norm > v_norm, targets_u, targets_v)
            self.targets_list.append(targets)

        # Precompute operators
        self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list = \
            vector_heat_net.geometry.get_all_operators(self.verts_list,
                                                                   self.faces_list,
                                                                   k_eig=self.k_eig,
                                                                   op_cache_dir=self.op_cache_dir)
        
        for iFile in range(len(mesh_files)):
            frames_verts = self.frames_verts_list[iFile]
            
#             # compute parallel transport operator from vertex tangent plane to face tangent planes
            frames_faces = self.frames_faces_list[iFile]
    
            frames_verts = torch.Tensor(frames_verts)
            frames_faces = torch.Tensor(frames_faces)

            targets_local = global_to_local_batch(frames_verts[:, 0, :], 
                                                  frames_verts[:, 1, :],
                                                  frames_verts[:, 2, :],
                                                  torch.Tensor(self.targets_list[iFile]))

            self.targets_list[iFile] = torch.view_as_complex(torch.tensor(targets_local)).to(dtype=torch.cfloat)            

        # save to cache
        if use_cache:
            vector_heat_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list,
                        self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list, self.targets_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_verts_list[idx], self.frames_faces_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], None, self.cotan_L_list[idx], self.cotan_evals_list[idx], self.cotan_evecs_list[idx], self.targets_list[idx]


class RetopoCurvatureDataset(Dataset):
    def __init__(self,
                 root_dir,
                 split='train',
                 k_eig=128,
                 use_cache=True,
                 op_cache_dir=None,
                 alternate_basis=False,
                 device='cuda'):
        self.train = (split == 'train')  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.input_vecs_list = []
        self.targets_list = []  # per-face vector targets
        self.pd1_list = []
        self.pd2_list = []
        self.pv1_list = []
        self.pv2_list = []

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            other_cache = os.path.join(self.cache_dir, "other.pt")
            if self.train:
                load_cache = train_cache
            elif split == 'test':
                load_cache = test_cache
            elif split == 'other':
                load_cache == other_cache
            
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list, self.targets_list, self.pd1_list, self.pd2_list, self.pv1_list, self.pv2_list = torch.load(
                    load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        target_files = []

        # Train test split
        if split == 'train':
            mesh_dirpath = os.path.join(self.root_dir, "train")
            target_dirpath = os.path.join(self.root_dir, "train")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)
        elif split == 'test':
            mesh_dirpath = os.path.join(self.root_dir, "test")
            target_dirpath = os.path.join(self.root_dir, "test")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)
        elif split == 'other':
            mesh_dirpath = os.path.join(self.root_dir, "other")
            target_dirpath = os.path.join(self.root_dir, "other")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)

        print("loading {} meshes".format(len(mesh_files)))
        print(mesh_files)

        # Load the actual files
        for mesh_file, target_file in tqdm(zip(mesh_files, target_files)):
            assert ".obj" in mesh_file
            assert ".h5" in target_file

            print("loading mesh " + str(mesh_file))
            
            verts, faces = pp3d.read_mesh(mesh_file)

            # SKIP COMPUTING PCD since it is unstable
#             pd1, pd2, pv1, pv2 = igl.principal_curvature(verts, faces)

#             pd1 = torch.Tensor(pd1)
#             pd2 = torch.Tensor(pd2)
#             pv1 = torch.Tensor(pv1)[:, None]
#             pv2 = torch.Tensor(pv2)[:, None]
            
#             self.pd1_list.append(pd1)
#             self.pd2_list.append(pd2)
#             self.pv1_list.append(pv1)
#             self.pv2_list.append(pv2)

            self.pd1_list.append(None)
            self.pd2_list.append(None)
            self.pv1_list.append(None)
            self.pv2_list.append(None)
            
            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
#             verts, scale = vector_heat_net.geometry_vector.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            
            print("loading target " + str(target_file))
            
            target_directions = {}
            with h5py.File(target_file, "r") as h5_file:
                for key, dataset in h5_file["data"].items():
                    target_directions[key] = dataset[0][()]
            targets_u = np.array(target_directions["u"])
            targets_v = np.array(target_directions["v"])

            u_norm = np.linalg.norm(targets_u, axis=-1)[:, None]
            v_norm = np.linalg.norm(targets_v, axis=-1)[:, None]

            targets = np.where(u_norm > v_norm, targets_u, targets_v)
            self.targets_list.append(targets)

        # Precompute operators
        self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list = \
            vector_heat_net.geometry.get_all_operators(self.verts_list,
                                                                   self.faces_list,
                                                                   k_eig=self.k_eig,
                                                                   op_cache_dir=self.op_cache_dir,
                                                                   alternate_basis=alternate_basis)
        for iFile in range(len(mesh_files)):
            frames_verts = self.frames_verts_list[iFile]
            
#             # compute parallel transport operator from vertex tangent plane to face tangent planes
            frames_faces = self.frames_faces_list[iFile]
    
            frames_verts = torch.Tensor(frames_verts)
            frames_faces = torch.Tensor(frames_faces)

            targets_local = global_to_local_batch(frames_verts[:, 0, :], 
                                                  frames_verts[:, 1, :],
                                                  frames_verts[:, 2, :],
                                                  torch.Tensor(self.targets_list[iFile]),
                                                  device=device)

            self.targets_list[iFile] = torch.view_as_complex(torch.tensor(targets_local)).to(dtype=torch.cfloat)            
#             self.pd1_list[iFile] = global_to_local_batch(frames_verts[:, 0, :], frames_verts[:, 1, :], frames_verts[:, 2, :], self.pd1_list[iFile])
#             self.pd2_list[iFile] = global_to_local_batch(frames_verts[:, 0, :], frames_verts[:, 1, :], frames_verts[:, 2, :], self.pd2_list[iFile])

        # save to cache
        if use_cache:
            vector_heat_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list,
                        self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list, self.targets_list, self.pd1_list, self.pd2_list, self.pv1_list, self.pv2_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_verts_list[idx], self.frames_faces_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.cotan_L_list[idx], self.cotan_evals_list[idx], self.cotan_evecs_list[idx], self.targets_list[idx], self.pd1_list[idx], self.pd2_list[idx], self.pv1_list[idx], self.pv2_list[idx]


class RetopoDatasetV2(Dataset):
    def __init__(self,
                 root_dir,
                 split='train',
                 k_eig=128,
                 use_cache=True,
                 op_cache_dir=None,
                 alternate_basis=False):
        self.train = (split == 'train')  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.input_vecs_list = []
        self.targets_list = []  # per-face vector targets
        self.pd1_list = []
        self.pd2_list = []
        self.pv1_list = []
        self.pv2_list = []

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            other_cache = os.path.join(self.cache_dir, "other.pt")
            if self.train:
                load_cache = train_cache
            elif split == 'test':
                load_cache = test_cache
            elif split == 'other':
                load_cache == other_cache
            
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list, self.targets_list, self.pd1_list, self.pd2_list, self.pv1_list, self.pv2_list = torch.load(
                    load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        target_files = []

        # Train test split
        if split == 'train':
            mesh_dirpath = os.path.join(self.root_dir, "train")
            target_dirpath = os.path.join(self.root_dir, "train")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)
        elif split == 'test':
            mesh_dirpath = os.path.join(self.root_dir, "test")
            target_dirpath = os.path.join(self.root_dir, "test")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)
        elif split == 'other':
            mesh_dirpath = os.path.join(self.root_dir, "other")
            target_dirpath = os.path.join(self.root_dir, "other")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)

        print("loading {} meshes".format(len(mesh_files)))
        print(mesh_files)

        # Load the actual files
        for mesh_file, target_file in tqdm(zip(mesh_files, target_files)):
            assert ".obj" in mesh_file
            assert ".h5" in target_file

            print("loading mesh " + str(mesh_file))
            
            verts, faces = pp3d.read_mesh(mesh_file)

            # SKIP COMPUTING PCD since it is unstable
#             pd1, pd2, pv1, pv2 = igl.principal_curvature(verts, faces)

#             pd1 = torch.Tensor(pd1)
#             pd2 = torch.Tensor(pd2)
#             pv1 = torch.Tensor(pv1)[:, None]
#             pv2 = torch.Tensor(pv2)[:, None]
            
#             self.pd1_list.append(pd1)
#             self.pd2_list.append(pd2)
#             self.pv1_list.append(pv1)
#             self.pv2_list.append(pv2)

            self.pd1_list.append(None)
            self.pd2_list.append(None)
            self.pv1_list.append(None)
            self.pv2_list.append(None)
            
            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
#             verts, scale = vector_heat_net.geometry_vector.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            
            print("loading target " + str(target_file))
            
            target_directions = {}
            with h5py.File(target_file, "r") as h5_file:
                for key, dataset in h5_file["data"].items():
                    target_directions[key] = dataset[()]
            targets = np.array(target_directions["u"])
#             targets_v = np.array(target_directions["v"])

#             u_norm = np.linalg.norm(targets_u, axis=-1)[:, None]
#             v_norm = np.linalg.norm(targets_v, axis=-1)[:, None]
#             targets = np.where(u_norm > v_norm, targets_u, targets_v)
            self.targets_list.append(targets)

        # Precompute operators
        self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list = \
            vector_heat_net.geometry.get_all_operators(self.verts_list,
                                                                   self.faces_list,
                                                                   k_eig=self.k_eig,
                                                                   op_cache_dir=self.op_cache_dir,
                                                                   alternate_basis=alternate_basis)
        for iFile in range(len(self.frames_verts_list)):
            frames_verts = self.frames_verts_list[iFile]
            
#             # compute parallel transport operator from vertex tangent plane to face tangent planes
            frames_faces = self.frames_faces_list[iFile]
    
            frames_verts = torch.Tensor(frames_verts)
            frames_faces = torch.Tensor(frames_faces)

            targets_local = global_to_local_batch(frames_verts[:, 0, :], 
                                                  frames_verts[:, 1, :],
                                                  frames_verts[:, 2, :],
                                                  torch.Tensor(self.targets_list[iFile]))
        
            self.targets_list[iFile] = torch.view_as_complex(torch.tensor(targets_local)).to(dtype=torch.cfloat)            
#             self.pd1_list[iFile] = global_to_local_batch(frames_verts[:, 0, :], frames_verts[:, 1, :], frames_verts[:, 2, :], self.pd1_list[iFile])
#             self.pd2_list[iFile] = global_to_local_batch(frames_verts[:, 0, :], frames_verts[:, 1, :], frames_verts[:, 2, :], self.pd2_list[iFile])

        # save to cache
        if use_cache:
            vector_heat_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list,
                        self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list, self.targets_list, self.pd1_list, self.pd2_list, self.pv1_list, self.pv2_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_verts_list[idx], self.frames_faces_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.cotan_L_list[idx], self.cotan_evals_list[idx], self.cotan_evecs_list[idx], self.targets_list[idx], self.pd1_list[idx], self.pd2_list[idx], self.pv1_list[idx], self.pv2_list[idx]

    
class RetopoCurvatureFFDataset(Dataset):
    def __init__(self,
                 root_dir,
                 split='train',
                 k_eig=128,
                 use_cache=True,
                 op_cache_dir=None,
                 alternate_basis=False):
        self.train = (split == 'train')  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.input_vecs_list = []
        self.targets_u_list = []  # per-face vector targets
        self.targets_v_list = []  # per-face vector targets
        self.targets_list = []  # per-face vector targets
        self.pd1_list = []
        self.pd2_list = []
        self.pv1_list = []
        self.pv2_list = []

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            other_cache = os.path.join(self.cache_dir, "other.pt")
            if self.train:
                load_cache = train_cache
            elif split == 'test':
                load_cache = test_cache
            elif split == 'other':
                load_cache == other_cache
            
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list, self.targets_list, self.pd1_list, self.pd2_list, self.pv1_list, self.pv2_list = torch.load(
                    load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        target_files = []

        # Train test split
        if split == 'train':
            mesh_dirpath = os.path.join(self.root_dir, "train")
            target_dirpath = os.path.join(self.root_dir, "train")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)
        elif split == 'test':
            mesh_dirpath = os.path.join(self.root_dir, "test")
            target_dirpath = os.path.join(self.root_dir, "test")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)
        elif split == 'other':
            mesh_dirpath = os.path.join(self.root_dir, "other")
            target_dirpath = os.path.join(self.root_dir, "other")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)

        print("loading {} meshes".format(len(mesh_files)))
        print(mesh_files)

        # Load the actual files
        for mesh_file, target_file in tqdm(zip(mesh_files, target_files)):
            assert ".obj" in mesh_file
            assert ".h5" in target_file

            print("loading mesh " + str(mesh_file))
            verts, faces = pp3d.read_mesh(mesh_file)

            pd1, pd2, pv1, pv2 = igl.principal_curvature(verts, faces)
            
            pd1 = torch.Tensor(pd1)
            pd2 = torch.Tensor(pd2)
            pv1 = torch.Tensor(pv1)[:, None]
            pv2 = torch.Tensor(pv2)[:, None]
            
            self.pd1_list.append(pd1)
            self.pd2_list.append(pd2)
            self.pv1_list.append(pv1)
            self.pv2_list.append(pv2)

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
#             verts, scale = vector_heat_net.geometry_vector.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            
            print("loading target " + str(target_file))
            
            target_directions = {}
            with h5py.File(target_file, "r") as h5_file:
                for key, dataset in h5_file["data"].items():
                    target_directions[key] = dataset[0][()]

            targets_u_candidate = np.array(target_directions["u"])
            targets_v_candidate = np.array(target_directions["v"])

            u_norm = np.linalg.norm(targets_u_candidate, axis=-1)[:, None]
            v_norm = np.linalg.norm(targets_v_candidate, axis=-1)[:, None]
            targets_u = np.where(u_norm > v_norm, targets_u_candidate, targets_v_candidate)
            targets_v = np.where(u_norm > v_norm, targets_v_candidate, targets_u_candidate)

            self.targets_u_list.append(targets_u)
            self.targets_v_list.append(targets_v) 
            self.targets_list.append(None)

        # Precompute operators
        self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list = \
            vector_heat_net.geometry.get_all_operators(self.verts_list,
                                                                   self.faces_list,
                                                                   k_eig=self.k_eig,
                                                                   op_cache_dir=self.op_cache_dir,
                                                                   alternate_basis=alternate_basis)
        
        for iFile in range(len(mesh_files)):
            frames_verts = self.frames_verts_list[iFile]
            
#             # compute parallel transport operator from vertex tangent plane to face tangent planes
            frames_faces = self.frames_faces_list[iFile]
    
            frames_verts = torch.Tensor(frames_verts)
            frames_faces = torch.Tensor(frames_faces)

            targets_u_local = global_to_local_batch(frames_verts[:, 0, :], 
                                                  frames_verts[:, 1, :],
                                                  frames_verts[:, 2, :],
                                                  torch.Tensor(self.targets_u_list[iFile]))
            
            targets_v_local = global_to_local_batch(frames_verts[:, 0, :], 
                                                  frames_verts[:, 1, :],
                                                  frames_verts[:, 2, :],
                                                  torch.Tensor(self.targets_v_list[iFile]))
            targets_local = torch.stack((targets_u_local, targets_v_local), dim=0)

            self.targets_list[iFile] = torch.view_as_complex(torch.tensor(targets_local)).to(dtype=torch.cfloat)            
            self.pd1_list[iFile] = global_to_local_batch(frames_verts[:, 0, :], frames_verts[:, 1, :], frames_verts[:, 2, :], self.pd1_list[iFile])
            self.pd2_list[iFile] = global_to_local_batch(frames_verts[:, 0, :], frames_verts[:, 1, :], frames_verts[:, 2, :], self.pd2_list[iFile])

        # save to cache
        if use_cache:
            vector_heat_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list,
                        self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list, self.targets_list, self.pd1_list, self.pd2_list, self.pv1_list, self.pv2_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_verts_list[idx], self.frames_faces_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.cotan_L_list[idx], self.cotan_evals_list[idx], self.cotan_evecs_list[idx], self.targets_list[idx], self.pd1_list[idx], self.pd2_list[idx], self.pv1_list[idx], self.pv2_list[idx]

    
class RetopoInferenceDataset(Dataset):
    def __init__(self,
                 root_dir,
                 split='train',
                 k_eig=128,
                 use_cache=True,
                 op_cache_dir=None):
        self.train = (split == 'train')  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.input_vecs_list = []
        self.targets_list = []  # per-face vector targets

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list = torch.load(
                    load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        target_files = []

        # Train test split
        if split == 'train':
            mesh_dirpath = os.path.join(self.root_dir, "train")
            target_dirpath = os.path.join(self.root_dir, "train")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)
        else:
            mesh_dirpath = os.path.join(self.root_dir, "test")
            target_dirpath = os.path.join(self.root_dir, "test")
            for fname in os.listdir(mesh_dirpath):
                if 'DS_Store' not in fname and ".obj" in fname:
                    sample_name = Path(fname).stem
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    target_fullpath = os.path.join(target_dirpath, sample_name + ".h5")
                    mesh_files.append(mesh_fullpath)
                    target_files.append(target_fullpath)

        print("loading {} meshes".format(len(mesh_files)))
        print(mesh_files)

        # Load the actual files
        for mesh_file, target_file in tqdm(zip(mesh_files, target_files)):
            assert ".obj" in mesh_file
            assert ".h5" in target_file

            print("loading mesh " + str(mesh_file))
            verts, faces = pp3d.read_mesh(mesh_file)

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
#             verts, scale = vector_heat_net.geometry_vector.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            
            print("loading target " + str(target_file))
            
#             target_directions = {}
#             with h5py.File(target_file, "r") as h5_file:
#                 for key, dataset in h5_file["data"].items():
#                     target_directions[key] = dataset[0][()]
#             targets_u = np.array(target_directions["u"])
#             targets_v = np.array(target_directions["v"])

#             u_norm = np.linalg.norm(targets_u, axis=-1)[:, None]
#             v_norm = np.linalg.norm(targets_v, axis=-1)[:, None]

#             targets = np.where(u_norm > v_norm, targets_u, targets_v)
#             self.targets_list.append(targets)
            self.targets_list.append([None])

        # Precompute operators
        self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list = \
            vector_heat_net.geometry.get_all_operators(self.verts_list,
                                                                   self.faces_list,
                                                                   k_eig=self.k_eig,
                                                                   op_cache_dir=self.op_cache_dir)
        
        for iFile in range(len(mesh_files)):
            frames_verts = self.frames_verts_list[iFile]
            
#             # compute parallel transport operator from vertex tangent plane to face tangent planes
            frames_faces = self.frames_faces_list[iFile]
    
            frames_verts = torch.Tensor(frames_verts)
            frames_faces = torch.Tensor(frames_faces)

#             targets_local = []
#             for i, target_direction in enumerate(self.targets_list[iFile]):   
#                 target_local = global_to_local(frames_verts[i, 0, :], 
#                                                frames_verts[i, 1, :],
#                                                frames_verts[i, 2, :], 
#                                                torch.Tensor(target_direction))
#                 target_local = target_local[:2]
#                 targets_local.append(target_local.detach().tolist())

#             self.targets_list[iFile] = torch.view_as_complex(torch.tensor(targets_local))

        # save to cache
        if use_cache:
            vector_heat_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_verts_list, self.frames_faces_list, self.massvec_list, self.L_list,
                        self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.cotan_L_list, self.cotan_evals_list, self.cotan_evecs_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_verts_list[idx], self.frames_faces_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.cotan_L_list[idx], self.cotan_evals_list[idx], self.cotan_evecs_list[idx]
