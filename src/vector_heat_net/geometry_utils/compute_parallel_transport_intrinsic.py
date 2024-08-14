import time

import gpytoolbox
import igl
import numpy as np
import openmesh as om
from tqdm import tqdm

from .compute_edge_lengths import compute_edge_lengths
from .intrinsic_mollification_eps import intrinsic_mollification_eps
from .edge_length_map import construct_edge_length_map


def get_vertex_one_rings_intrinsic_vectorized(v, f, edge_lengths, is_boundary_vert, eps=0, alternate_basis=False):
    """
    l: np.array of size [#F, 3]
        list of edge lengths. each row of l represents the edges of face i, as [l_ij, l_jk, l_ki]
    returns:
        one_rings: (#V, #neighbors)
            for each vert in the mesh, the indices of its neighboring faces
        all_one_ring_angles: (#V, #neighbors)
            for each vert in the mesh, the angles of its neighboring faces (in CCW order)
            note that the number of neighbors varies based on topology
    """
    mesh = om.TriMesh(v, f)

    edge_length_map = construct_edge_length_map(v, f, edge_lengths)
    axis_edge_indices = [np.NaN for i in range(v.shape[0])]

    one_ring_verts = mesh.vertex_vertex_indices()[:, ::-1]
    n_one_ring_verts = np.sum(np.where(one_ring_verts == -1, 0, 1), axis=1)
    mask = (one_ring_verts != -1)
    max_one_ring_size = one_ring_verts.shape[-1]
    angles = np.empty(one_ring_verts.shape)

    if alternate_basis:
        for i in range(len(one_ring_verts)):
            one_ring = one_ring_verts[i][mask[i]]
            one_ring = np.roll(one_ring, shift=1)
            one_ring_verts[i][mask[i]] = one_ring

    for i in reversed(range(max_one_ring_size)):
        """
        one_ring_verts - np.array [N, max(len(one_rings))]:
            -1 -1    4 7 0 3   ==>  4 4   4 7 0 3
            -1 -1 -1   3 2 6   ==>  3 3 3   3 2 6
            -1 -1    7 9 8 1   ==>  7 7   7 9 8 1
            ...
        """
        one_ring_verts[:, i] = np.where(one_ring_verts[:, i] == -1,
                                        one_ring_verts[:, (i + 1) % max_one_ring_size],
                                        one_ring_verts[:, i])

    for i in range(max_one_ring_size):
        # periodic assumption for angles around manifold one ring
        a_len = np.maximum(
            edge_length_map[np.arange(v.shape[0]), one_ring_verts[:, i]],
            edge_length_map[one_ring_verts[:, i], np.arange(v.shape[0])]
        )
        b_len = np.maximum(
            edge_length_map[np.arange(v.shape[0]), one_ring_verts[:, (i + 1) % max_one_ring_size]],
            edge_length_map[one_ring_verts[:, (i + 1) % max_one_ring_size], np.arange(v.shape[0])]
        )
        c_len = np.maximum(
            edge_length_map[one_ring_verts[:, i], one_ring_verts[:, (i + 1) % max_one_ring_size]],
            edge_length_map[one_ring_verts[:, (i + 1) % max_one_ring_size], one_ring_verts[:, i]],
        )
        theta = np.arccos((a_len ** 2 + b_len ** 2 - c_len ** 2) / (2 * a_len * b_len))
        angles[:, i] = theta
        axis_edge_indices = np.where(a_len > eps, i, axis_edge_indices)
    axis_edge_indices = axis_edge_indices.astype(int)

    mask_boundary = mask.copy()
    mask_boundary[:, -1] = False
    boundary_vert_angles = np.where(mask_boundary, angles, 0)
    boundary_vert_angles[:, -1] = (2 * np.pi) - np.sum(np.where(mask_boundary, angles, 0), axis=1)

    all_one_ring_angles = np.where(is_boundary_vert[:, None],
                                   boundary_vert_angles,
                                   np.where(mask, angles, 0))
    total_interior_angles = np.sum(np.array(all_one_ring_angles), axis=1)

    axis_edge_vert_idx = one_ring_verts.flatten()[axis_edge_indices + np.arange(v.shape[0]) * one_ring_verts.shape[1]]
    axis_edge_vertices = v[axis_edge_vert_idx]
    one_ring_vertices = np.where(mask, one_ring_verts, -1)

    return (one_ring_vertices,
            all_one_ring_angles,
            total_interior_angles,
            axis_edge_indices,
            axis_edge_vertices,
            n_one_ring_verts)


def get_vertex_one_rings_intrinsic(v, f, edge_lengths, boundary_verts, eps=0):
    """
    l: np.array of size [#F, 3]
        list of edge lengths. each row of l represents the edges of face i, as [l_ij, l_jk, l_ki]
    returns:
        one_rings: (#V, #neighbors)
            for each vert in the mesh, the indices of its neighboring faces
        all_one_ring_angles: (#V, #neighbors)
            for each vert in the mesh, the angles of its neighboring faces (in CCW order)
            note that the number of neighbors varies based on topology
    """
    mesh = om.TriMesh(v, f)

    edge_length_map = construct_edge_length_map(v, f, edge_lengths)
    one_ring_vertices = []
    all_one_ring_angles = []
    total_interior_angles = []
    axis_edge_indices = [np.NaN for i in range(v.shape[0])]
    axis_edge_vertices = []
    for vert_idx, one_ring_verts in enumerate(mesh.vertex_vertex_indices()):
        one_ring_verts = one_ring_verts[::-1]
        mask = np.where(one_ring_verts != -1)
        one_ring_verts = one_ring_verts[mask]
        one_ring_vertices.append(one_ring_verts)

        """
        For a given triangle ijk, here we compute the angle at vertex i.
        primary edge (e1): ij
        secondary edge (e2): ik
        opposite edge (e3): jk
                c
           k *_____* j
             |    /
             |   /
           b |  /  a
             |θ/
             |/
             * i
        """
        angles = []
        for j, _ in enumerate(one_ring_verts):
            a_len = max(edge_length_map[vert_idx, one_ring_verts[j]],
                        edge_length_map[one_ring_verts[j], vert_idx])
            b_len = max(edge_length_map[vert_idx, one_ring_verts[(j + 1) % len(one_ring_verts)]],
                        edge_length_map[one_ring_verts[(j + 1) % len(one_ring_verts)], vert_idx])
            c_len = max(edge_length_map[one_ring_verts[j], one_ring_verts[(j + 1) % len(one_ring_verts)]],
                        edge_length_map[one_ring_verts[(j + 1) % len(one_ring_verts)], one_ring_verts[j]])

            if a_len + b_len > c_len and b_len + c_len > a_len and c_len + a_len > b_len:
                # using law of cosines
                theta = np.arccos((a_len ** 2 + b_len ** 2 - c_len ** 2) / (2 * a_len * b_len))
                if (vert_idx not in boundary_verts) or (vert_idx in boundary_verts and j < len(one_ring_verts) - 1):
                    angles.append(theta)
            if a_len > eps:
                axis_edge_indices[vert_idx] = j

        all_one_ring_angles.append(angles)
        total_interior_angles.append(np.array(angles).sum())
        axis_edge_vertices.append(v[one_ring_verts[axis_edge_indices[vert_idx]]])
    return one_ring_vertices, all_one_ring_angles, total_interior_angles, axis_edge_indices, axis_edge_vertices


def get_vertex_one_rings_intrinsic_no_boundary(v, f, edge_lengths, eps=0):
    """
    l: np.array of size [#F, 3]
        list of edge lengths. each row of l represents the edges of face i, as [l_ij, l_jk, l_ki]
    returns:
        one_rings: (#V, #neighbors)
            for each vert in the mesh, the indices of its neighboring faces
        all_one_ring_angles: (#V, #neighbors)
            for each vert in the mesh, the angles of its neighboring faces (in CCW order)
            note that the number of neighbors varies based on topology
    """
    mesh = om.TriMesh(v, f)

    edge_length_map = construct_edge_length_map(v, f, edge_lengths)
    one_ring_vertices = []
    all_one_ring_angles = []
    total_interior_angles = []
    axis_edge_indices = [np.NaN for i in range(v.shape[0])]
    axis_edge_vertices = []

    for vert_idx, one_ring_verts in enumerate(mesh.vertex_vertex_indices()):
        one_ring_verts = one_ring_verts[::-1]
        mask = np.where(one_ring_verts != -1)
        one_ring_verts = one_ring_verts[mask]
        one_ring_vertices.append(one_ring_verts)

        """
        For a given triangle ijk, here we compute the angle at vertex i.
        primary edge (e1): ij
        secondary edge (e2): ik
        opposite edge (e3): jk
                c
           k *_____* j
             |    /
             |   /
           b |  /  a
             |θ/
             |/
             * i
        """
        angles = []
        for j, _ in enumerate(one_ring_verts):
            a_len = edge_length_map[vert_idx, one_ring_verts[j]]
            b_len = edge_length_map[vert_idx, one_ring_verts[(j + 1) % len(one_ring_verts)]]
            c_len = edge_length_map[one_ring_verts[j], one_ring_verts[(j + 1) % len(one_ring_verts)]]

            assert a_len + b_len > c_len and b_len + c_len > a_len and c_len + a_len > b_len,\
                   f"triangle [{vert_idx, one_ring_verts[j], one_ring_verts[(j + 1) % len(one_ring_verts)]}] " \
                   f"does not satisfy triangle inequality: lengths [{a_len}, {b_len}, {c_len}]"
            # using law of cosines
            theta = np.arccos((a_len ** 2 + b_len ** 2 - c_len ** 2) / (2 * a_len * b_len))
            angles.append(theta)
            if a_len > eps:
                axis_edge_indices[vert_idx] = j

        all_one_ring_angles.append(angles)
        total_interior_angles.append(np.array(angles).sum())
        axis_edge_vertices.append(v[one_ring_verts[axis_edge_indices[vert_idx]]].tolist())
    return one_ring_vertices, all_one_ring_angles, total_interior_angles, axis_edge_indices, np.array(axis_edge_vertices)


def compute_parallel_transport_intrinsic(v, f, intrinsic_mollification=True, alternate_basis=False):
    """
    Parameters
    ----------
    l: np.array of size [#F, 3]
        list of edge lengths. each row of l represents the edges of face i, as [l_ij, l_jk, l_ki]
    intrinsic_mollification
        compute and add small constant eps to all intrinsic edge lengths to satisfy triangle inequality
    Returns
    -------

    """
    transition_angles = np.zeros_like(f).astype(np.csingle)  # ordering: [[i->j, j->k, k->i], ...]

    boundary_loops = igl.all_boundary_loop(f)
    boundary_verts = []
    for loop in boundary_loops:
        for i in range(len(loop)):
            boundary_verts.append((loop[i]))
    is_boundary_vert = np.isin(np.arange(v.shape[0]), boundary_verts)

    edge_lengths = compute_edge_lengths(v, f)
    if intrinsic_mollification:
        eps = intrinsic_mollification_eps(v, f, edge_lengths)
        edge_lengths = edge_lengths + eps
        one_rings, all_one_ring_angles, total_interior_angles, axis_edge_indices, axis_edge_vertices, n_one_ring_verts = \
            get_vertex_one_rings_intrinsic_vectorized(v, f, edge_lengths, is_boundary_vert, eps=eps, alternate_basis=alternate_basis)
    else:
        one_rings, all_one_ring_angles, total_interior_angles, axis_edge_indices, axis_edge_vertices, n_one_ring_verts = \
            get_vertex_one_rings_intrinsic_vectorized(v, f, edge_lengths, is_boundary_vert, eps=0, alternate_basis=alternate_basis)

    M = gpytoolbox.massmatrix_intrinsic(edge_lengths * edge_lengths, f)

    # compute local bases
    basisN = igl.per_vertex_normals(v, f, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)
    basisY = np.cross(basisN, axis_edge_vertices - v)
    basisX = np.cross(basisY, basisN)

    basisX = basisX / np.linalg.norm(basisX, axis=1)[:, None]
    basisY = basisY / np.linalg.norm(basisY, axis=1)[:, None]
    basisN = basisN / np.linalg.norm(basisN, axis=1)[:, None]

    for i in range(3):
        vi_idx = f[:, i]
        vj_idx = f[:, (i + 1) % 3]

        vi_is_boundary_vert = np.isin(vi_idx, boundary_verts)
        vj_is_boundary_vert = np.isin(vj_idx, boundary_verts)

        vi_e_ij_idx = np.where(one_rings[vi_idx] == vj_idx[:, None])
        vj_e_ij_idx = np.where(one_rings[vj_idx] == vi_idx[:, None])

        integrated_angles = np.zeros_like(all_one_ring_angles).astype(np.float32)
        integrated_angles[:, 0] = all_one_ring_angles[:, 0]
        for j in range(1, integrated_angles.shape[1]):
            integrated_angles[:, j] = integrated_angles[:, j - 1] + all_one_ring_angles[:, j - 1]

        vi_start = np.arange(f.shape[0]) * all_one_ring_angles.shape[1] + axis_edge_indices[vi_idx]
        vj_start = np.arange(f.shape[0]) * all_one_ring_angles.shape[1] + axis_edge_indices[vj_idx]

        vi_end = np.arange(f.shape[0]) * all_one_ring_angles.shape[1] + vi_e_ij_idx[1]
        vj_end = np.arange(f.shape[0]) * all_one_ring_angles.shape[1] + vj_e_ij_idx[1]

        vi_e_ij_angle = integrated_angles[vi_idx].flatten()[vi_end] - integrated_angles[vi_idx].flatten()[vi_start]
        vj_e_ij_angle = integrated_angles[vj_idx].flatten()[vj_end] - integrated_angles[vj_idx].flatten()[vj_start]

        vi_e_ij_angle = np.where(vi_is_boundary_vert,
                                 vi_e_ij_angle,
                                 vi_e_ij_angle * ((2 * np.pi) / np.sum(all_one_ring_angles[vi_idx], axis=1)))  # normalize angle
        vj_e_ij_angle = np.where(vj_is_boundary_vert,
                                 vj_e_ij_angle,
                                 vj_e_ij_angle * ((2 * np.pi) / np.sum(all_one_ring_angles[vj_idx], axis=1)))  # normalize angle

        transition_angle = (np.pi + vj_e_ij_angle) - vi_e_ij_angle
        transition_angles[:, i] = transition_angle

    complex_rotations = np.exp((0 + 1j) * transition_angles)
    return complex_rotations, basisX, basisY, basisN, M.astype(np.cdouble)


def compute_parallel_transport_intrinsic_no_boundary(v, f, intrinsic_mollification=True):
    """
    Parameters
    ----------
    l: np.array of size [#F, 3]
        list of edge lengths. each row of l represents the edges of face i, as [l_ij, l_jk, l_ki]
    intrinsic_mollification
        compute and add small constant eps to all intrinsic edge lengths to satisfy triangle inequality
    Returns
    -------

    """
    # default_basis_edge_index = 0
    transition_angles = np.zeros_like(f).astype(np.csingle)  # ordering: [[i->j, j->k, k->i], ...]

    edge_lengths = compute_edge_lengths(v, f)
    if intrinsic_mollification:
        eps = intrinsic_mollification_eps(v, f, edge_lengths)
        edge_lengths = edge_lengths + eps
        one_rings, all_one_ring_angles, total_interior_angles, axis_edge_indices = get_vertex_one_rings_intrinsic_no_boundary(v, f, edge_lengths, eps=eps)
    else:
        one_rings, all_one_ring_angles, total_interior_angles, axis_edge_indices = get_vertex_one_rings_intrinsic_no_boundary(v, f, edge_lengths, eps=0)

    M = gpytoolbox.massmatrix_intrinsic(edge_lengths * edge_lengths, f)

    # compute local bases
    basisN = igl.per_vertex_normals(v, f, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)
    basisX = np.zeros(basisN.shape)
    basisY = np.zeros(basisN.shape)
    # for i in range(basisN.shape[0]):
    basisY = np.cross(basisN, v[one_rings[:, axis_edge_indices]] - v)
    basisX = np.cross(basisY, basisN)
    # basisY[i] = np.cross(basisN[i], v[one_rings[i][axis_edge_indices[i]]] - v[i])
    # basisX[i] = np.cross(basisY[i], basisN[i])

    basisX = basisX / np.linalg.norm(basisX, axis=1)[:, None]
    basisY = basisY / np.linalg.norm(basisY, axis=1)[:, None]
    basisN = basisN / np.linalg.norm(basisN, axis=1)[:, None]

    for j, face in tqdm(enumerate(f)):
        for i, vert_idx in enumerate(face):
            # v_i *----->* v_j
            vi_idx = face[i]
            vj_idx = face[(i + 1) % 3]

            vi_e_ij_idx = np.where(one_rings[vi_idx] == vj_idx)[0][0]
            vj_e_ij_idx = np.where(one_rings[vj_idx] == vi_idx)[0][0]

            vi_angles = np.roll(all_one_ring_angles[vi_idx], -(axis_edge_indices[vi_idx]))
            vj_angles = np.roll(all_one_ring_angles[vj_idx], -(axis_edge_indices[vj_idx]))

            vi_e_ij_angle = vi_angles[0: (vi_e_ij_idx - axis_edge_indices[vi_idx]) % vi_angles.shape[0]].sum()  # total angle from basis to e_ij in v_i local tangent plane
            vj_e_ij_angle = vj_angles[0: (vj_e_ij_idx - axis_edge_indices[vj_idx]) % vj_angles.shape[0]].sum() #+ total_interior_angles[vj_idx] / 2  # total angle from basis to e_ij in v_j local tangent plane

            vi_e_ij_angle = vi_e_ij_angle * ((2 * np.pi) / total_interior_angles[vi_idx])  # normalize angle
            vj_e_ij_angle = vj_e_ij_angle * ((2 * np.pi) / total_interior_angles[vj_idx])

            transition_angle = (np.pi + vj_e_ij_angle) - vi_e_ij_angle
            transition_angles[j, i] = transition_angle

    complex_rotations = np.exp((0 + 1j) * transition_angles)
    print(f"NaNs in parallel transport rotations: {np.count_nonzero(np.isnan(complex_rotations))}")
    return complex_rotations, basisX, basisY, basisN, M.astype(np.csingle)
