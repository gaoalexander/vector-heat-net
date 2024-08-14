import numpy as np


def compute_edge_lengths(v, f):
    """
    Computes edge lengths for a given mesh.

    args:
        v: Vertex positions (np.ndarray of shape (num_vertices, 3)).
        f: Face indices (np.ndarray of shape (num_faces, 3)).
    returns:
        Edge lengths for each face (np.ndarray of shape (num_faces, 3)).
    """
    edge_lengths = np.zeros(f.shape)
    for i in range(3):
        edge_lengths[:, i] = np.linalg.norm(v[f[:, (i + 1) % 3]] - v[f[:, i]], axis=1)
    return edge_lengths
