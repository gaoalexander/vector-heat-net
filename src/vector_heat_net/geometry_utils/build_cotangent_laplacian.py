import gpytoolbox
import numpy as np
from scipy.sparse import csr_array

from .avg_edge_length import avg_edge_length
from .compute_edge_lengths import compute_edge_lengths
from .intrinsic_cotan_weights import intrinsic_cotan_weights
from .intrinsic_mollification_eps import intrinsic_mollification_eps


def cotan_weights(v, f, eps=0.0):
    """
    Computes cotangent weights for a given mesh.

    args:
        v: Vertex positions (np.ndarray of shape (num_vertices, 3)).
        f: Face indices (np.ndarray of shape (num_faces, 3)).
        eps: Small value to prevent division by zero. Defaults to 0.0.
    returns:
        Cotangent weights (np.ndarray of shape (num_faces, 3)).
    """

    weights = np.zeros(f.shape)
    for i in range(3):
        vec1 = v[f[:, (i + 1) % 3]] - v[f[:, i]]
        vec2 = v[f[:, (i + 2) % 3]] - v[f[:, i]]
        dot = np.sum(vec1 * vec2, axis=1)
        cross_norm = np.linalg.norm(np.cross(vec1, vec2), axis=1)
        cot_theta = dot / (cross_norm + eps)
        weights[:, i] = 0.5 * cot_theta
    return weights


def build_cotangent_laplacian(v, f, eps=0.0):
    """
    Builds the cotangent Laplacian matrix.

    args:
        v: Vertex positions (np.ndarray of shape (num_vertices, 3)).
        f: Face indices (np.ndarray of shape (num_faces, 3)).
        eps: Small value to prevent division by zero. Defaults to 0.0.
    returns:
        The cotangent Laplacian matrix (csr_array of shape (num_vertices, num_vertices)).
    """

    row, col, data = [], [], []
    weights = cotan_weights(v, f, eps=eps)

    for i in range(3):
        row.append(f[:, i].tolist())
        col.append(f[:, (i + 1) % 3].tolist())
        data.append(-weights[:, (i + 2) % 3])

        row.append(f[:, (i + 1) % 3].tolist())
        col.append(f[:, i].tolist())
        data.append(-weights[:, (i + 2) % 3])

        row.append(f[:, i].tolist())
        col.append(f[:, i].tolist())
        data.append(weights[:, (i + 1) % 3])

        row.append(f[:, i].tolist())
        col.append(f[:, i].tolist())
        data.append(weights[:, (i + 2) % 3])
    row = np.array(row).flatten()
    col = np.array(col).flatten()
    data = np.array(data).flatten()
    return csr_array((data, (row, col)), shape=(v.shape[0], v.shape[0]))


def build_intrinsic_laplacian(v, f, intrinsic_mollification=True):
    """
    Builds the intrinsic Laplacian matrix.

    args:
        v: Vertex positions (np.ndarray of shape (num_vertices, 3)).
        f: Face indices (np.ndarray of shape (num_faces, 3)).
        intrinsic_mollification: If True, applies intrinsic mollification to edge lengths. Defaults to True.
    returns:
        A tuple containing:
            - The intrinsic Laplacian matrix (csr_array of shape (num_vertices, num_vertices)).
            - The mass matrix (np.ndarray of shape (num_vertices, num_vertices)).
    """

    row, col, data = [], [], []
    weights = intrinsic_cotan_weights(v, f, intrinsic_mollification=intrinsic_mollification)

    edge_lengths = compute_edge_lengths(v, f)
    if intrinsic_mollification:
        eps = intrinsic_mollification_eps(v, f, edge_lengths)
        edge_lengths = edge_lengths + eps
    M = gpytoolbox.massmatrix_intrinsic(edge_lengths * edge_lengths, f)

    for i in range(3):
        row.append(f[:, i].tolist())
        col.append(f[:, (i + 1) % 3].tolist())
        data.append(-weights[:, (i + 2) % 3])

        row.append(f[:, (i + 1) % 3].tolist())
        col.append(f[:, i].tolist())
        data.append(-weights[:, (i + 2) % 3])

        row.append(f[:, i].tolist())
        col.append(f[:, i].tolist())
        data.append(weights[:, (i + 1) % 3])

        row.append(f[:, i].tolist())
        col.append(f[:, i].tolist())
        data.append(weights[:, (i + 2) % 3])
    row = np.array(row).flatten()
    col = np.array(col).flatten()
    data = np.array(data).flatten()
    return csr_array((data, (row, col)), shape=(v.shape[0], v.shape[0])), M
