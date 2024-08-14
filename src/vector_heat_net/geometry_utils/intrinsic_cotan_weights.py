import numpy as np
from .compute_edge_lengths import compute_edge_lengths
from .intrinsic_mollification_eps import intrinsic_mollification_eps


def intrinsic_cotan_weights(v, f, intrinsic_mollification=True):
    """
    Computes cotangent Laplacian weights using intrinsic edge lengths only,
    instead of the traditional extrinsic cotangent formula.

    Args:
        v: Vertex positions (np.ndarray of shape (num_vertices, 3))
        f: Face indices (np.ndarray of shape (num_faces, 3))
        intrinsic_mollification: If True, applies intrinsic mollification to edge lengths. Defaults to True.

    Returns:
        weights: (np.ndarray of shape (num_faces, 3))
    """

    weights = np.zeros(f.shape)
    edge_lengths = compute_edge_lengths(v, f)

    if intrinsic_mollification:
        eps = intrinsic_mollification_eps(v, f, edge_lengths)
        edge_lengths = edge_lengths + eps

    semi_perimeters = np.sum(edge_lengths, axis=1) / 2
    areas = np.sqrt(semi_perimeters *
                    (semi_perimeters - edge_lengths[:, 0]) *
                    (semi_perimeters - edge_lengths[:, 1]) *
                    (semi_perimeters - edge_lengths[:, 2]))

    for i in range(3):
        weights[:, i] = (np.square(edge_lengths[:, i]) +
                         np.square(edge_lengths[:, (i + 2) % 3]) -
                         np.square(edge_lengths[:, (i + 1) % 3])) / (8 * areas)
    return weights
