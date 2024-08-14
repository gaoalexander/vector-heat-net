import scipy
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import os.path
import sys
import random
from multiprocessing import Pool

import gpytoolbox

import numpy as np
import scipy.spatial
import torch
from torch.distributions.categorical import Categorical
import sklearn.neighbors

import robust_laplacian
import potpourri3d as pp3d

import openmesh as om
import vector_heat_net.utils as utils
from vector_heat_net.connection_laplacian.connection_laplacian import connection_laplacian
from vector_heat_net.connection_laplacian.build_vertex_gradient_operator import build_vertex_gradient_operator
from vector_heat_net.geometry_utils.build_connection_laplacian_intrinsic import build_connection_laplacian_intrinsic
from vector_heat_net.geometry_utils.build_cotangent_laplacian import build_intrinsic_laplacian
from .utils import toNP

def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)


def norm2(x, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return dot(x, x)


def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)


def face_coords(verts, faces):
    coords = verts[faces]
    return coords


def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)


def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)


# Given (..., 3) vectors and normals, projects out any components of vecs
# which lies in the direction of normals. Normals are assumed to be unit.

def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)


def face_area(verts, faces):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)
    return 0.5 * norm(raw_normal)

def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal

def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:,2,:]
    return normal / np.linalg.norm(normal,axis=-1, keepdims=True)

def mesh_vertex_normals(verts, faces):
    # numpy in / out
    face_n = toNP(face_normals(torch.tensor(verts), torch.tensor(faces))) # ugly torch <---> numpy

    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:,i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals,axis=-1,keepdims=True)

    return vertex_normals


def vertex_normals(verts, faces, n_neighbors_cloud=30):
    verts_np = toNP(verts)

    if faces.numel() == 0: # point cloud
    
        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts_np[neigh_inds,:]
        neigh_points = neigh_points - verts_np[:,np.newaxis,:]
        normals = neighborhood_normal(neigh_points)

    else: # mesh

        normals = mesh_vertex_normals(verts_np, toNP(faces))

        # if any are NaN, wiggle slightly and recompute
        bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_normals_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5) * scale
            wiggle_verts = verts_np + bad_normals_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, toNP(faces))

        # if still NaN assign random normals (probably means unreferenced verts in mesh)
        bad_normals_mask = np.isnan(normals).any(axis=1)
        if bad_normals_mask.any():
            normals[bad_normals_mask,:] = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5)[bad_normals_mask,:]
            normals = normals / np.linalg.norm(normals, axis=-1)[:,np.newaxis]
            

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)
        
    if torch.any(torch.isnan(normals)): raise ValueError("NaN normals :(")

    return normals


def build_vert_tangent_frames(verts, faces, normals=None):

    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    if normals == None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals 

    # = find an orthogonal basis
    basis_cand1 = torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)
    
    basisX = torch.where((torch.abs(dot(vert_normals, basis_cand1))
                          < 0.9).unsqueeze(-1), basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)
    
    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames


def build_face_tangent_frames(verts, faces):
    """
    compute per-face reference coordinate frames
    """
    tri_edge_1 = verts[faces[:, 1], :] - verts[faces[:, 0], :]
    tri_edge_2 = verts[faces[:, 2], :] - verts[faces[:, 0], :]
    axis_x = tri_edge_1
    axis_n = np.cross(tri_edge_1, tri_edge_2)
    axis_y = np.cross(axis_n, axis_x)

    # normalize
    axis_x = axis_x / np.expand_dims(np.linalg.norm(axis_x, axis=1), axis=1)
    axis_y = axis_y / np.expand_dims(np.linalg.norm(axis_y, axis=1), axis=1)
    axis_n = axis_n / np.expand_dims(np.linalg.norm(axis_n, axis=1), axis=1)
    
    axis_x = torch.Tensor(axis_x)
    axis_y = torch.Tensor(axis_y)
    axis_n = torch.Tensor(axis_n)

    frames = torch.stack((axis_x, axis_y, axis_n), dim=-2)
    return frames

        
def build_grad_point_cloud(verts, frames, n_neighbors_cloud=30):
    verts_np = toNP(verts)
    frames_np = toNP(frames)

    _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
    neigh_points = verts_np[neigh_inds,:]
    neigh_vecs = neigh_points - verts_np[:,np.newaxis,:]

    # TODO this could easily be way faster. For instance we could avoid the weird edges format and the corresponding pure-python loop via some numpy broadcasting of the same logic. The way it works right now is just to share code with the mesh version. But its low priority since its preprocessing code.

    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
    edges = np.stack((edge_inds_from, neigh_inds.flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)
   
    return build_grad(verts_np, torch.tensor(edges), edge_tangent_vecs)


def global_to_local(axis_x, axis_y, axis_n, global_vector, device='cuda'):
    """
    dtype :: torch.tensor
    
    axis_x: [3,]
    axis_y: [3,]
    axis_n: [3,]
    global_vector: [3,]
    """
    # transform target directions into local reference frames
    axis_x = axis_x[:, None]
    axis_y = axis_y[:, None]
    axis_n = axis_n[:, None]

    local_to_world_transform = torch.cat((axis_x, axis_y, axis_n), axis=1)
    world_to_local_transform = torch.linalg.inv(local_to_world_transform)
    local_vector = world_to_local_transform.to(dtype=torch.float32).to(device) @ global_vector.to(dtype=torch.float32).to(device)

    # discard component along normal axis
    return local_vector[:2]


def global_to_local_batch(axis_x, axis_y, axis_n, global_vector, device='cuda'):
    """
    dtype :: torch.tensor
    
    axis_x: [N, 3]
    axis_y: [N, 3]
    axis_n: [N, 3]
    global_vector: [N, 3]
    """
    # transform target directions into local reference frames
    axis_x = axis_x[:, :, None]
    axis_y = axis_y[:, :, None]
    axis_n = axis_n[:, :, None]
    global_vector = global_vector[:, :, None]

    local_to_world_transform = torch.cat((axis_x, axis_y, axis_n), axis=2)
    world_to_local_transform = torch.linalg.inv(local_to_world_transform)
    local_vector = torch.bmm(world_to_local_transform.to(dtype=torch.float32).to(device), global_vector.to(dtype=torch.float32).to(device))

#     print("DEBUG LOCAL TO GLOBAL: ", local_vector[:, :, 0][:, 2])
    # discard component along normal axis
    return local_vector[:, :, 0][:, :2]


def local_to_global(axis_x, axis_y, axis_n, local_vector):
    """
    axis_x: [3,]
    axis_y: [3,]
    axis_n: [3,]
    global_vector: [3,]
    """
    # transform target directions into local reference frames
    axis_x = axis_x[:, None]
    axis_y = axis_y[:, None]
    axis_n = axis_n[:, None]

    local_to_world_transform = np.concatenate((axis_x, axis_y, axis_n), axis=1)
    global_vector = local_to_world_transform @ local_vector

    # discard component along normal axis
    return global_vector


def local_to_global_batch(axis_x, axis_y, axis_n, local_vector, device='cuda'):
    """
    axis_x: [N, 3]
    axis_y: [N, 3]
    axis_n: [N, 3]
    local_vector: [N, 1] (complex)
    """
    # transform target directions into local reference frames
    axis_x = axis_x[:, :, None]
    axis_y = axis_y[:, :, None]
    axis_n = axis_n[:, :, None]
    local_vector = local_vector[:, :, None]
    local_vector = torch.stack((local_vector.real, local_vector.imag, torch.zeros_like(local_vector.real)), dim=1).squeeze(3)

    local_to_world_transform = torch.cat((axis_x, axis_y, axis_n), axis=2)
    global_vector = torch.bmm(local_to_world_transform.to(dtype=torch.float32).to(device), local_vector.to(dtype=torch.float32).to(device))

    # discard component along normal axis
    return global_vector[:, :, 0]


def get_all_edges(verts, faces):
    """
    for every vertex, get its one-ring edges in cyclic order.
    the edges are expressed as unit-length vectors.
    """
    mesh = om.TriMesh(verts, faces)

    all_edges = []
    for vert_idx, one_ring_verts in enumerate(mesh.vertex_vertex_indices()):
        one_ring_verts = one_ring_verts[::-1]
        mask = np.where(one_ring_verts != -1)
        one_ring_verts = one_ring_verts[mask]

        edges = verts[one_ring_verts] - verts[vert_idx]
        normalized_edges = edges / np.linalg.norm(edges, axis=-1)[:, None]
        all_edges.append(normalized_edges)
    return all_edges


def get_all_one_ring_angles_v2(verts, faces):
    """
    returns:
        one_rings: (#V, #neighbors)
            for each vert in the mesh, the indices of its neighboring faces
        all_one_ring_angles: (#V, #neighbors)
            for each vert in the mesh, the angles of its neighboring faces (in CCW order)
            note that the number of neighbors varies based on topology
    """
    mesh = om.TriMesh(verts, faces)

    one_rings = []
    all_one_ring_angles = []
    for vert_idx, one_ring_verts in enumerate(mesh.vertex_vertex_indices()):
        one_ring_verts = one_ring_verts[::-1]
        mask = np.where(one_ring_verts != -1)
        one_ring_verts = one_ring_verts[mask]
        one_rings.append(one_ring_verts)

        edges = verts[one_ring_verts] - verts[vert_idx]
        normalized_edges = edges / np.linalg.norm(edges, axis=-1)[:, None]

        angles = []
        for j, edge in enumerate(normalized_edges):
            a = normalized_edges[j]
            b = normalized_edges[j - 1]
            theta = np.arccos(np.dot(a, b))
            angles.append(theta)
        all_one_ring_angles.append(angles)
    return one_rings, all_one_ring_angles


def get_transition_angles(verts, faces, axis_x_verts, axis_x_faces, axis_y_faces, axis_n_faces):
    # visualize angles
    one_rings, all_one_ring_angles = get_all_one_ring_angles_v2(verts, faces)
    all_edges = get_all_edges(verts, faces)

    axis_edge_indices = []
    total_interior_angles = []
    for i in range(len(one_rings)):
        total_interior_angles.append(np.array(all_one_ring_angles[i]).sum())
        dot_products = all_edges[i] @ axis_x_verts[i]
        argmax = np.argmax(dot_products)
        axis_edge_indices.append(argmax)

    axis_edge_indices = np.array(axis_edge_indices)

    total_interior_angles = np.array(total_interior_angles)

    transition_angles = np.zeros(faces.shape[0] * 3)
    for j, face in enumerate(faces):
        for i, vert_idx in enumerate(face):
            e_ij = verts[face[(i + 1) % 3]] - verts[face[i]]
            e_ij_normalized = e_ij / np.linalg.norm(e_ij)
            dot_products = all_edges[face[i]] @ e_ij_normalized
            e_ij_idx = np.argmax(dot_products)
            angles = all_one_ring_angles[face[i]]
            angles = np.roll(angles, -(axis_edge_indices[face[i]] + 1))

            e_ij_face_plane_local = global_to_local(torch.Tensor(axis_x_faces[j]),
                                                    torch.Tensor(axis_y_faces[j]),
                                                    torch.Tensor(axis_n_faces[j]),
                                                    torch.Tensor(e_ij_normalized)).cpu().numpy()

            vert_tangent_plane_angle = 0 + angles[0: (e_ij_idx - axis_edge_indices[face[i]]) % angles.shape[0]].sum()
            vert_tangent_plane_angle = vert_tangent_plane_angle * ((2 * np.pi) / total_interior_angles[face[i]])  # normalize angle
            face_tangent_plane_angle = np.arctan2(e_ij_face_plane_local[1], e_ij_face_plane_local[0])
            transition_angle = face_tangent_plane_angle - vert_tangent_plane_angle

            transition_angles[j * 3 + i] = transition_angle

    return transition_angles


def transport_verts2faces(faces, vector_field_verts, transition_angles, crossfield=False):
    transition_angles = torch.Tensor(transition_angles)
    preds_local = torch.Tensor(vector_field_verts)
    if not vector_field_verts.is_complex():
        preds_local = preds_local.view(int(preds_local.shape[0] / 2), 2) # [V, 2]
        preds_local = torch.view_as_complex(preds_local)
    preds_local = preds_local[faces.flatten()]
    transported = torch.exp((0 + 1j) * transition_angles) * preds_local
    transported = transported.view(faces.shape[0], 3)

    if crossfield:
        averaged = torch.mean(transported ** 4, dim=1)[:, None]
        averaged = averaged ** (1 / 4)
    else:
        averaged = torch.mean(transported, dim=1)[:, None]
    return averaged


def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)
    return edge_tangent

#     rescale_factor = (np.linalg.norm(edge_vecs, axis=1) / np.linalg.norm(edge_tangent, axis=1))[:, None]
#     rescale_factor = np.nan_to_num(rescale_factor)
#     return edge_tangent * rescale_factor


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient. All values pointwise.
    - edges: (2, E)
    """
    
    edges_np = toNP(edges)
    edge_tangent_vectors_np = toNP(edge_tangent_vectors)

    # TODO find a way to do this in pure numpy?

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges_np[1, iE]
            ind_lookup.append(jV)
    
            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(
            N, N)).tocsc()

    return mat


def construct_div_operator(V, F):
    grad = gpytoolbox.grad(V, F)
    double_area = gpytoolbox.doublearea(V, F)
    mass = np.kron(np.eye(3), np.diag(double_area))
    return -0.25 * grad.T @ mass


def div(vec, gradX, gradY):
    vec = vec.squeeze()
    return torch.mm(gradX, vec.real[:, None]) + torch.mm(gradY, vec.imag[:, None])


def curl(vec, J, gradX, gradY):
    return -div(J * vec, gradX, gradY)


def compute_operators(verts, faces, k_eig, alternate_basis=False, intrinsic_mollification=True):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.

    See get_operators() for a similar routine that wraps this one with a layer of caching.

    Torch in / torch out.

    Arguments:
      - vertices: (V,3) vertex positions
      - faces: (F,3) list of triangular faces. If empty, assumed to be a point cloud.
      - k_eig: number of eigenvectors to use

    Returns:
      - frames: (V,3,3) X/Y/Z coordinate frame at each vertex. Z coordinate is normal (e.g. [:,2,:] for normals)
      - massvec: (V) real diagonal of lumped mass matrix
      - L: (VxV) complex sparse matrix of (weak) Connection Laplacian
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian 
      - gradX: (VxV) sparse matrix which gives X-component of gradient in the local basis at the vertex
      - gradY: same as gradX but for Y-component of gradient

    PyTorch doesn't seem to like complex sparse matrices, so we store the "real" and "imaginary" (aka X and Y) gradient matrices separately, rather than as one complex sparse matrix.

    Note: for a generalized eigenvalue problem, the mass matrix matters! The eigenvectors are only othrthonormal with respect to the mass matrix, like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """
    device = verts.device
    dtype = verts.dtype
    eps = 1e-8
    verts_np = toNP(verts).astype(np.float64)
    faces_np = toNP(faces)

    # ===================================
    # compute mesh cotangent laplacian
    # ===================================
    cotan_L, M = robust_laplacian.mesh_laplacian(verts_np, faces_np)
#     cotan_L, _ = build_intrinsic_laplacian(verts_np, faces_np, intrinsic_mollification=intrinsic_mollification)

    # ===================================
    # compute mesh connection laplacian
    # ===================================
    alternate_basis_string = "ALTERNATE" if alternate_basis else "DEFAULT"
    print(f"Computing Connection Laplacian using {alternate_basis_string} local basis...")
    
    L, basisX, basisY, basisN, massvec_np = build_connection_laplacian_intrinsic(
        verts_np, faces_np,
        intrinsic_mollification=intrinsic_mollification,
        alternate_basis=alternate_basis
    )
    
    if(np.isnan(L.data).any()):
        print(f"Connection Laplacian contains {np.isnan(L.data).sum()} NaN values!")
        raise RuntimeError("NaN Laplace matrix")

    # ===================================
    # construct local tangent frames
    # ===================================
    basisX, basisY, basisN = torch.tensor(basisX), torch.tensor(basisY), torch.tensor(basisN) # [V, 3]
    frames_verts = torch.stack((basisX, basisY, basisN), dim=-2).to(device) # [V, 3, 3]
    frames_faces = build_face_tangent_frames(verts, faces)
    # ===================================
    # compute the eigenbases
    # ===================================
    if k_eig > 0:
        cotan_L_eigsh = (cotan_L + scipy.sparse.identity(cotan_L.shape[0]) * eps).tocsc()
        cotan_L_eigsh = cotan_L.tocsc()
        # eigendecomposition of connection laplacian
        np.random.seed(0)        
        evals_np, evecs_np = scipy.sparse.linalg.eigsh(L,
                                                       k=k_eig,
                                                       M=massvec_np,
                                                       sigma=eps,
                                                       v0=np.random.rand(min(L.shape)),
                                                       maxiter=k_eig * 50,
                                                       tol=1e-16)
        # eigendecomposition of cotangent laplacian
        np.random.seed(0)
        cotan_evals_np, cotan_evecs_np = scipy.sparse.linalg.eigsh(cotan_L_eigsh,
                                                                   k=k_eig,
                                                                   M=massvec_np,
                                                                   sigma=1e-8,
                                                                   v0=np.random.rand(min(cotan_L_eigsh.shape)),
                                                                   maxiter=k_eig * 50,
                                                                   tol=1e-16)

        # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
        evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))
        cotan_evals_np = np.clip(cotan_evals_np, a_min=0., a_max=float('inf'))
    else: #k_eig == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0],0))

    # ===================================
    # construct gradient matrices
    # ===================================
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # For meshes, we use the same edges as were used to build the Laplacian.
    edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
    edge_vecs = edge_tangent_vectors(verts, frames_verts, edges)
    grad_mat_np = build_grad(verts, edges, edge_vecs)

    # ALTERNATIVE IMPLEMENTATION: Construct fully INTRINSIC graident operator:
    # grad_mat_np, _, _ = build_vertex_gradient_operator(verts_np, faces_np)

    # Split complex gradient in to two real sparse mats (torch doesn't like complex sparse matrices)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # ===================================
    # convert back to torch
    # ===================================
    massvec = torch.from_numpy(massvec_np.diagonal()).to(device=device, dtype=torch.cfloat)
    L = utils.sparse_complex_np_to_torch(L).to(device=device, dtype=torch.cfloat)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=torch.float32)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=torch.cfloat)
    gradX = utils.sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = utils.sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)
    cotan_L = utils.sparse_np_to_torch(cotan_L).to(device=device, dtype=dtype)
    cotan_evals = torch.from_numpy(cotan_evals_np).to(device=device, dtype=dtype)
    cotan_evecs = torch.from_numpy(cotan_evecs_np).to(device=device, dtype=dtype)

    return frames_verts, frames_faces, massvec, L, evals, evecs, gradX, gradY, cotan_L, cotan_evals, cotan_evecs


def get_all_operators(verts_list, faces_list, k_eig, op_cache_dir=None, normals=None, alternate_basis=False):
    N = len(verts_list)

    frames_verts = []
    frames_faces = []
    massvec = []
    L = []
    evals = []
    evecs = []
    cotan_L = []
    cotan_evals = []
    cotan_evecs = []
    gradX = []
    gradY = []

    for i in range(N):
        print("get_all_operators() processing {} / {} {:.3f}%".format(i, N, i / N * 100))
        try:
            outputs = get_operators(verts_list[i], faces_list[i], k_eig, op_cache_dir, alternate_basis=alternate_basis)
        except:
            print("FAILED TO COMPUTE EIGENDECOMPOSITION - Skipping mesh...")
            continue
        frames_verts.append(outputs[0])
        frames_faces.append(outputs[1])       
        massvec.append(outputs[2])
        L.append(outputs[3])
        evals.append(outputs[4])
        evecs.append(outputs[5])
        gradX.append(outputs[6])
        gradY.append(outputs[7])
        cotan_L.append(outputs[8])
        cotan_evals.append(outputs[9])
        cotan_evecs.append(outputs[10])
    return frames_verts, frames_faces, massvec, L, evals, evecs, gradX, gradY, cotan_L, cotan_evals, cotan_evecs


def get_operators(verts, faces, k_eig=128, op_cache_dir=None, normals=None, overwrite_cache=False, alternate_basis=False):
    """
    See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk, and finally returned as a tensor with dtype/device matching the `verts` input.
    """
    device = verts.device
    dtype = verts.dtype
    verts_np = toNP(verts)
    faces_np = toNP(faces)

    if(np.isnan(verts_np).any()):
        raise RuntimeError("tried to construct operators from NaN verts")

    # Check the cache directory
    # Note 1: Collisions here are exceptionally unlikely, so we could probably just use the hash...
    #         but for good measure we check values nonetheless.
    # Note 2: There is a small possibility for race conditions to lead to bucket gaps or duplicate
    #         entries in this cache. The good news is that that is totally fine, and at most slightly
    #         slows performance with rare extra cache misses.
    found = False
    if op_cache_dir is not None:
        utils.ensure_dir_exists(op_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                op_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")
            
            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"].item()

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    continue

                # print("  cache hit!")

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if overwrite_cache: 
                    print("  overwriting cache by request")
                    os.remove(search_path)
                    break
                
                if cache_k_eig < k_eig:
                    print("  overwriting cache --- not enough eigenvalues")
                    os.remove(search_path)
                    break
                
                if "L_data" not in npzfile:
                    print("  overwriting cache --- entries are absent")
                    os.remove(search_path)
                    break


                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # This entry matches! Return it.
                frames = npzfile["frames"]
                mass = npzfile["mass"]
                L = read_sp_mat("L")
                evals = npzfile["evals"][:k_eig]
                evecs = npzfile["evecs"][:,:k_eig]
                gradX = read_sp_mat("gradX")
                gradY = read_sp_mat("gradY")

                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=torch.complex128)
                L = utils.sparse_complex_np_to_torch(L).to(device=device, dtype=torch.complex128)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=torch.complex128)
                gradX = utils.sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
                gradY = utils.sparse_np_to_torch(gradY).to(device=device, dtype=dtype)

                found = True
                
                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break
            
            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

    if not found:
        # No matching entry found; recompute.
        frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY, cotan_L, cotan_evals, cotan_evecs = (
            compute_operators(verts, faces, k_eig, alternate_basis=alternate_basis))
    return frames_verts, frames_faces, mass, L, evals, evecs, gradX, gradY, cotan_L, cotan_evals, cotan_evecs


def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,V,2,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = torch.conj(basis.transpose(-2, -1)).unsqueeze(0)
    return basisT @ (massvec * values.squeeze()).unsqueeze(0).unsqueeze(-1)


def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (K,D)
      - basis: (V,K)
    Outputs:
      - (V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        print(basis.data.dtype, values.data.dtype)
        return basis @ values
    else:
        return torch.matmul(basis, values)


def compute_hks(evals, evecs, scales):
    """
    Inputs:
      - evals: (K) eigenvalues
      - evecs: (V,K) values
      - scales: (S) times
    Outputs:
      - (V,S) hks values
    """

    # expand batch
    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    # TODO could be a matmul
    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1) # (B,1,S,K)    
    terms = power_coefs * (evecs * evecs).unsqueeze(2)  # (B,V,S,K)
    out = torch.sum(terms, dim=-1) # (B,V,S)

    if expand_batch:
        return out.squeeze(0)
    else:
        return out


def compute_hks_autoscale(evals, evecs, count):
    # these scales roughly approximate those suggested in the hks paper
    scales = torch.logspace(-2, 0., steps=count, device=evals.device, dtype=evals.dtype)
    return compute_hks(evals, evecs, scales)


def normalize_positions(pos, faces=None, method='mean', scale_method='max_rad'):
    # center and unit-scale positions

    if method == 'mean':
        # center using the average point position
        pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
    elif method == 'bbox': 
        # center via the middle of the axis-aligned bounding box
        bbox_min = torch.min(pos, dim=-2).values
        bbox_max = torch.max(pos, dim=-2).values
        center = (bbox_max + bbox_min) / 2.
        pos -= center.unsqueeze(-2)
    else:
        raise ValueError("unrecognized method")

    if scale_method == 'max_rad':
        scale = torch.max(norm(pos), dim=-1, keepdim=True).values.unsqueeze(-1)
        pos = pos / scale
    elif scale_method == 'area': 
        if faces is None:
            raise ValueError("must pass faces for area normalization")
        coords = pos[faces]
        vec_A = coords[:, 1, :] - coords[:, 0, :]
        vec_B = coords[:, 2, :] - coords[:, 0, :]
        face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
        total_area = torch.sum(face_areas)
        scale = (1. / torch.sqrt(total_area))
        pos = pos * scale
    else:
        raise ValueError("unrecognized scale method")
    return pos, scale


# Finds the k nearest neighbors of source on target.
# Return is two tensors (distances, indices). Returned points will be sorted in increasing order of distance.
def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(-1, points_target.shape[0], -1)
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(points_source.shape[0], -1, -1)

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result
    
    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k+1 if omit_diagonal else k 
        _, neighbors = kd_tree.query(points_source_np, k=k_search)
        
        if omit_diagonal: 
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1]-1))

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds
    
    else:
        raise ValueError("unrecognized method")


def farthest_point_sampling(points, n_sample):
    # Torch in, torch out. Returns a |V| mask with n_sample elements set to true.

    N = points.shape[0]
    if(n_sample > N): raise ValueError("not enough points to sample")

    chosen_mask = torch.zeros(N, dtype=torch.bool, device=points.device)
    min_dists = torch.ones(N, dtype=points.dtype, device=points.device) * float('inf')

    # pick the centermost first point
    points = normalize_positions(points)
    i = torch.min(norm2(points), dim=0).indices
    chosen_mask[i] = True

    for _ in range(n_sample-1):
        
        # update distance
        dists = norm2(points[i,:].unsqueeze(0) - points)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        i = torch.max(min_dists,dim=0).indices.item()
        chosen_mask[i] = True

    return chosen_mask


def geodesic_label_errors(target_verts, target_faces, pred_labels, gt_labels, normalization='diameter', geodesic_cache_dir=None):
    """
    Return a vector of distances between predicted and ground-truth lables (normalized by geodesic diameter or area)

    This method is SLOW when it needs to recompute geodesic distances.
    """

    # move all to numpy cpu
    target_verts = toNP(target_verts) 
    target_faces = toNP(target_faces) 

    pred_labels = toNP(pred_labels) 
    gt_labels = toNP(gt_labels) 

    dists = get_all_pairs_geodesic_distance(target_verts, target_faces, geodesic_cache_dir) 

    result_dists = dists[pred_labels, gt_labels]

    if normalization == 'diameter':
        geodesic_diameter = np.max(dists)
        normalized_result_dists = result_dists / geodesic_diameter
    elif normalization == 'area':
        total_area = torch.sum(face_area(torch.tensor(target_verts), torch.tensor(target_faces)))
        normalized_result_dists = result_dists / torch.sqrt(total_area)
    else:
        raise ValueError('unrecognized normalization')

    return normalized_result_dists


# This function and the helper class below are to support parallel computation of all-pairs geodesic distance
def all_pairs_geodesic_worker(verts, faces, i):
    import igl

    N = verts.shape[0]

    # TODO: this re-does a ton of work, since it is called independently each time. Some custom C++ code could surely make it faster.
    sources = np.array([i])[:,np.newaxis]
    targets = np.arange(N)[:,np.newaxis]
    dist_vec = igl.exact_geodesic(verts, faces, sources, targets)
    
    return dist_vec
        
class AllPairsGeodesicEngine(object):
    def __init__(self, verts, faces):
        self.verts = verts 
        self.faces = faces 
    def __call__(self, i):
        return all_pairs_geodesic_worker(self.verts, self.faces, i)


def get_all_pairs_geodesic_distance(verts_np, faces_np, geodesic_cache_dir=None):
    """
    Return a gigantic VxV dense matrix containing the all-pairs geodesic distance matrix. Internally caches, recomputing only if necessary.

    (numpy in, numpy out)
    """

    # need libigl for geodesic call
    try:
        import igl
    except ImportError as e:
        raise ImportError("Must have python libigl installed for all-pairs geodesics. `conda install -c conda-forge igl`")

    # Check the cache
    found = False 
    if geodesic_cache_dir is not None:
        utils.ensure_dir_exists(geodesic_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                geodesic_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts_np, cache_verts)) or (not np.array_equal(faces_np, cache_faces)):
                    i_cache_search += 1
                    continue

                # This entry matches! Return it.
                found = True
                result_dists = npzfile["dist"]
                break

            except FileNotFoundError:
                break

    if not found:
                
        print("Computing all-pairs geodesic distance (warning: SLOW!)")

        # Not found, compute from scratch
        # warning: slowwwwwww

        N = verts_np.shape[0]

        try:
            pool = Pool(None) # on 8 processors
            engine = AllPairsGeodesicEngine(verts_np, faces_np)
            outputs = pool.map(engine, range(N))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        result_dists = np.array(outputs)

        # replace any failed values with nan
        result_dists = np.nan_to_num(result_dists, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # we expect that this should be a symmetric matrix, but it might not be. Take the min of the symmetric values to make it symmetric
        result_dists = np.fmin(result_dists, np.transpose(result_dists))

        # on rare occaisions MMP fails, yielding nan/inf; set it to the largest non-failed value if so
        max_dist = np.nanmax(result_dists)
        result_dists = np.nan_to_num(result_dists, nan=max_dist, posinf=max_dist, neginf=max_dist)

        print("...finished computing all-pairs geodesic distance")

        # put it in the cache if possible
        if geodesic_cache_dir is not None:

            print("saving geodesic distances to cache: " + str(geodesic_cache_dir))

            # TODO we're potentially saving a double precision but only using a single
            # precision here; could save storage by always saving as floats
            np.savez(search_path,
                     verts=verts_np,
                     faces=faces_np,
                     dist=result_dists
                     )

    return result_dists
