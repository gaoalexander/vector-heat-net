import sys

import os
import random

import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import torch
import torch.nn as nn

from .nn import VectorMLP
from .utils import toNP
from .geometry import to_basis, from_basis
from .utils import complex_to_interleaved, interleaved_to_complex


class LearnedTimeDiffusion(nn.Module):
    """
    Applies vector diffusion with learned per-channel t.

    In the spectral domain this becomes 
        f_out = e ^ (lambda_i t) f_in

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values 
    """

    def __init__(self, C_inout, method='spectral'):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method  # one of ['spectral', 'implicit_dense']

        nn.init.constant_(self.diffusion_time, 0.0)

    def set_diffusion_time(self, t):
        self.diffusion_time.data = torch.ones_like(self.diffusion_time) * t
        print(f"Diffusion time set to: {self.diffusion_time.data}")

    def forward(self, vec, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if vec.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    vec.shape, self.C_inout))

        if self.method == 'spectral':
            vec_spec = evecs.T.conj() @ (mass[:, None] * vec.squeeze())
            vec_diffuse_spec = torch.exp(-evals[:, None] @ self.diffusion_time[None, :]) * vec_spec
            vec_diffuse = (evecs @ vec_diffuse_spec)[None, :, :]

        elif self.method == 'implicit_dense':  # using complex representation
            # =========MULTI CHANNEL==========
            V = vec.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            A = L.to_dense().unsqueeze(0).unsqueeze(0).expand(-1, self.C_inout, V, V).clone()
            A *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            A += torch.diag_embed(mass).unsqueeze(0).unsqueeze(0)

            b = vec * mass.unsqueeze(-1)
            b = torch.transpose(b, 1, 2).unsqueeze(-1)

            # Factor the system and solve
            A_cholesky = torch.linalg.cholesky(A)
#             A_cholesky = A_cholesky.cpu()
#             b = b.cpu()
            vec_diffuse = torch.cholesky_solve(b, A_cholesky)
            vec_diffuse = vec_diffuse.cuda()
            vec_diffuse = torch.transpose(vec_diffuse, 1, 2).squeeze(-1)

        else:
            raise ValueError("unrecognized method")

        return vec_diffuse


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''

    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=0.2)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity (exclude last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )


class VectorDiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, vector_mlp_hidden_dims,
                 dropout=True,
                 batchnorm=False,
                 diffusion_method='spectral',
                 with_gradient_features=False,
                 with_gradient_rotations=True):
        super(VectorDiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.vector_mlp_hidden_dims = vector_mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.vector_diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)
        self.VECTOR_MLP_C = 2 * self.C_width

        # MLPs
        self.vector_mlp = VectorMLP([self.VECTOR_MLP_C] + self.vector_mlp_hidden_dims + [self.C_width],
                                    batchnorm=batchnorm, dropout=dropout)

    def forward(self, vec_in, mass, L, evals, evecs, gradX, gradY):
        # Manage dimensions
        B = vec_in.shape[0]  # batch dimension
        if vec_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    vec_in.shape, self.C_width))

        # Diffusion block
        vec_diffuse = self.vector_diffusion(vec_in, L, mass, evals, evecs)

        # Stack inputs to mlp
        feature_combined = torch.cat((vec_in, vec_diffuse), dim=-1)

        # Apply the mlp
        interleaved = complex_to_interleaved(feature_combined)
        mlp_out = self.vector_mlp(interleaved)
        vec_out = interleaved_to_complex(mlp_out)

        # Skip connection
        vec_out = vec_out + vec_in

        return vec_out


class VectorDiffusionNet(nn.Module):

    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, outputs_at='vertices',
                 vector_mlp_hidden_dims=None, dropout=True,
                 batchnorm=False, with_gradient_features=False, with_gradient_rotations=True,
                 diffusion_method='implicit_dense'):
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(VectorDiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError(
            "invalid setting for outputs_at")

        # MLP options
        if vector_mlp_hidden_dims == None:
            vector_mlp_hidden_dims = [C_width, C_width]
        self.vector_mlp_hidden_dims = vector_mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError(
            "invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width, bias=False)  # .to(dtype=torch.cfloat)
        self.last_lin = nn.Linear(C_width, C_out, bias=False)  # .to(dtype=torch.cfloat)

        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = VectorDiffusionNetBlock(C_width=C_width,
                                            vector_mlp_hidden_dims=vector_mlp_hidden_dims,
                                            dropout=dropout,
                                            batchnorm=batchnorm,
                                            diffusion_method=diffusion_method,
                                            with_gradient_features=with_gradient_features,
                                            with_gradient_rotations=with_gradient_rotations)

            self.blocks.append(block)
            self.add_module("block_" + str(i_block), self.blocks[-1])

    def forward(self, vec_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            vec_in (tensor):    Input features, dimension [N,2,C] or [B,N,2,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            vec_out (tensor):    Output with dimension [N,2,C_out] or [B,N,2,C_out]
        """
        interleaved = complex_to_interleaved(vec_in)
        hidden_first = self.first_lin(interleaved)
        complex_features = interleaved_to_complex(hidden_first)

        for b in self.blocks:
            complex_features = b(complex_features, mass, L, evals, evecs, gradX, gradY)

        interleaved = complex_to_interleaved(complex_features)
        hidden_last = self.last_lin(interleaved)
        vec_out = interleaved_to_complex(hidden_last)
        return vec_out
