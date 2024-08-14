import igl
import numpy as np
import openmesh as om
from tqdm import tqdm


# def get_all_one_ring_edge_vectors(verts, faces):
#     """
#     for every vertex, get its one-ring edges in cyclic order.
#     the edges are expressed as unit-length vectors.
#     """
#     mesh = om.TriMesh(verts, faces)
#
#     all_edges = []
#     for vert_idx, one_ring_verts in enumerate(mesh.vertex_vertex_indices()):
#         one_ring_verts = one_ring_verts[::-1]
#         mask = np.where(one_ring_verts != -1)
#         one_ring_verts = one_ring_verts[mask]
#
#         edges = verts[one_ring_verts] - verts[vert_idx]
#         normalized_edges = edges / np.linalg.norm(edges, axis=-1)[:, None]
#         all_edges.append(normalized_edges)
#     return all_edges


def get_vertex_one_rings(verts, faces):
    """
    returns:
        one_rings: (#V, #neighbors)
            for each vert in the mesh, the indices of its neighboring faces
        all_one_ring_angles: (#V, #neighbors)
            for each vert in the mesh, the angles of its neighboring faces (in CCW order)
            note that the number of neighbors varies based on topology
    """
    mesh = om.TriMesh(verts, faces)

    one_ring_vertices = []
    all_one_ring_angles = []
    total_interior_angles = []
    one_ring_edge_vectors = []
    for vert_idx, one_ring_verts in enumerate(mesh.vertex_vertex_indices()):
        one_ring_verts = one_ring_verts[::-1]
        mask = np.where(one_ring_verts != -1)
        one_ring_verts = one_ring_verts[mask]
        one_ring_vertices.append(one_ring_verts)
        print(vert_idx, one_ring_verts)

        edges = verts[one_ring_verts] - verts[vert_idx]
        normalized_edges = edges / np.linalg.norm(edges, axis=-1)[:, None]

        angles = []
        for j, edge in enumerate(normalized_edges):
            a = normalized_edges[j]
            b = normalized_edges[j - 1]
            theta = np.arccos(np.dot(a, b))
            angles.append(theta)
        all_one_ring_angles.append(angles)
        total_interior_angles.append(np.array(angles).sum())
        one_ring_edge_vectors.append(edges)
    return one_ring_vertices, one_ring_edge_vectors, all_one_ring_angles, total_interior_angles


def compute_parallel_transport(v, f, intrinsic_mollification=True):
    one_rings, one_ring_edge_vectors, all_one_ring_angles, total_interior_angles = get_vertex_one_rings(v, f)

    print(all_one_ring_angles)
    default_basis_edge_index = 0
    transition_angles = np.zeros_like(f).astype(np.csingle)  # ordering: [[i->j, j->k, k->i], ...]

    # compute local bases
    basisN = igl.per_vertex_normals(v, f, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)
    basisX = np.zeros(basisN.shape)
    basisY = np.zeros(basisN.shape)
    for i in range(basisN.shape[0]):
        basisY[i] = np.cross(basisN[i], one_ring_edge_vectors[i][default_basis_edge_index])
        basisX[i] = np.cross(basisY[i], basisN[i])
    basisX = basisX / np.linalg.norm(basisX, axis=1)[:, None]
    basisY = basisY / np.linalg.norm(basisY, axis=1)[:, None]
    basisN = basisN / np.linalg.norm(basisN, axis=1)[:, None]

    # for all vertices, compute index of one ring edges that corresponds to the X-basis
    axis_edge_indices = []
    for i in range(len(one_rings)):
        dot_products_i = one_ring_edge_vectors[i] @ basisX[i]
        argmax = np.argmax(dot_products_i)
        axis_edge_indices.append(argmax)
    axis_edge_indices = np.array(axis_edge_indices)

    boundary_loops = igl.all_boundary_loop(f)
    boundary_edges = []
    for loop in boundary_loops:
        for i in range(len(loop)):
            boundary_edges.append((loop[i], loop[(i + 1) % len(loop)]))
    boundary_edges = set(boundary_edges)

    for j, face in tqdm(enumerate(f)):
        for i, vert_idx in enumerate(face):
            # v_i *----->* v_j
            vi_idx = face[i]
            vj_idx = face[(i + 1) % 3]

            is_boundary_edge = False
            if (vi_idx, vj_idx) in boundary_edges or (vj_idx, vi_idx) in boundary_edges:
                is_boundary_edge = True
            e_ij = v[vj_idx] - v[vi_idx]
            e_ij_normalized = e_ij / np.linalg.norm(e_ij)

            vi_dot_products = one_ring_edge_vectors[vi_idx] @ e_ij_normalized
            vj_dot_products = one_ring_edge_vectors[vj_idx] @ (-e_ij_normalized)

            vi_e_ij_idx = np.argmax(vi_dot_products)  # get the index in the one ring of a specific vertex, of edge_ij
            vj_e_ij_idx = np.argmax(vj_dot_products)

            vi_angles = np.roll(all_one_ring_angles[vi_idx], -(axis_edge_indices[vi_idx] + 1))
            vj_angles = np.roll(all_one_ring_angles[vj_idx], -(axis_edge_indices[vj_idx] + 1))

            if is_boundary_edge:
                vi_angles = np.concatenate((vi_angles, np.array([2 * np.pi - vi_angles.sum()])))
                vj_angles = np.concatenate((vj_angles, np.array([2 * np.pi - vj_angles.sum()])))

            vi_e_ij_angle = vi_angles[0: (vi_e_ij_idx - axis_edge_indices[vi_idx]) % vi_angles.shape[0]].sum()  # total angle from basis to e_ij in v_i local tangent plane
            vj_e_ij_angle = vj_angles[0: (vj_e_ij_idx - axis_edge_indices[vj_idx]) % vj_angles.shape[0]].sum() #+ total_interior_angles[vj_idx] / 2  # total angle from basis to e_ij in v_j local tangent plane

            if is_boundary_edge:
                vi_e_ij_angle = vi_e_ij_angle * ((2 * np.pi) / vi_angles.sum())  # normalize angle
                vj_e_ij_angle = vj_e_ij_angle * ((2 * np.pi) / vj_angles.sum())
            else:
                vi_e_ij_angle = vi_e_ij_angle * ((2 * np.pi) / total_interior_angles[vi_idx])  # normalize angle
                vj_e_ij_angle = vj_e_ij_angle * ((2 * np.pi) / total_interior_angles[vj_idx])

            transition_angle = (np.pi + vj_e_ij_angle) - vi_e_ij_angle
            transition_angles[j, i] = transition_angle

    complex_rotations = np.exp((0 + 1j) * transition_angles)
    print(f"NaNs in parallel transport rotations: {np.count_nonzero(np.isnan(complex_rotations))}")
    return complex_rotations, basisX, basisY, basisN
