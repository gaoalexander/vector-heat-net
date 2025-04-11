import igl
import numpy as np

from src.vector_heat_net.geometry_utils.compute_parallel_transport import get_vertex_one_rings


def angle_between_vectors_ccw(v1, v2, per_face_normals):
    """
    Calculates the counterclockwise angle between two 3D vectors.

    Args:
        v1 (numpy.ndarray): The first vector (3D).
        v2 (numpy.ndarray): The second vector (3D).
        per_face_normals (numpy.ndarray): Normal vector defining the plane orientation.

    Returns:
        float: The counterclockwise angle in radians.
    """
    v1 = v1 / np.linalg.norm(v1, axis=-1)[:, None]
    v2 = v2 / np.linalg.norm(v2, axis=-1)[:, None]
    angle = np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1), np.sum(v1 * v2, axis=-1))

    angle = np.where(np.sum(np.cross(v1, v2) * per_face_normals, axis=-1) < 0, 2 * np.pi - angle, angle)
    return angle


def compute_vertex_based_angle(axis_x_vert, e_ij, total_interior_angle, one_ring_edge_vectors, one_ring_angles):
    similarity_axis = -np.inf
    similarity_edge = -np.inf
    idx_axis = 0
    idx_edge = 0
    for i in range(len(one_ring_edge_vectors)):
        sim = np.dot(
            axis_x_vert / np.linalg.norm(axis_x_vert),
            one_ring_edge_vectors[i] / np.linalg.norm(one_ring_edge_vectors[i])
        )
        if sim > similarity_axis:
            similarity_axis = sim
            idx_axis = i

        sim = np.dot(
            e_ij / np.linalg.norm(e_ij),
            one_ring_edge_vectors[i] / np.linalg.norm(one_ring_edge_vectors[i])
        )
        if sim > similarity_edge:
            similarity_edge = sim
            idx_edge = i

    if idx_axis < idx_edge:
        angle = np.sum(one_ring_angles[idx_axis + 1: idx_edge + 1]) * (2 * np.pi / total_interior_angle)
    elif idx_axis > idx_edge:
        angle = (np.sum(one_ring_angles[idx_axis + 1:]) + np.sum(one_ring_angles[: idx_edge + 1])) * (2 * np.pi / total_interior_angle)
    else:
        return 0
    return angle


def v_to_f_operator(v, f, axis_x_verts, axis_x_faces):
    per_face_normals = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
    one_ring_vertices, one_ring_edge_vectors, all_one_ring_angles, total_interior_angles = get_vertex_one_rings(v, f)

    angular_differences = np.zeros((f.shape[0], 3))

    for i in range(3):
        e_ij = v[f[:, (i + 1) % 3]] - v[f[:, i]]
        theta_face = angle_between_vectors_ccw(axis_x_faces, e_ij, per_face_normals)

        theta_vert = np.zeros((f.shape[0]))
        for j in range(f.shape[0]):
            theta_vert[j] = compute_vertex_based_angle(
                axis_x_verts[f[j, i]],
                e_ij[j],
                total_interior_angles[f[j, i]],
                one_ring_edge_vectors[f[j, i]],
                all_one_ring_angles[f[j, i]]
            )
        angular_differences[:, i] = theta_face - theta_vert

    return np.exp(angular_differences * (0 + 1j))


def interpolate_vector_field_v_to_f(v, f, u_verts, axis_x_verts, axis_x_faces):
    T = v_to_f_operator(v, f, axis_x_verts, axis_x_faces)
    u_verts_complex = u_verts[:, 0] + u_verts[:, 1] * (0 + 1j)

    u_faces = T * u_verts_complex[f]
    u_faces = np.sum(u_faces, axis=-1) / 3
    return u_faces
