from .compute_edge_lengths import compute_edge_lengths


def avg_edge_length(v, f):
    """
    Computes the average edge length of a mesh.

    args:
        v: Vertex positions (np.ndarray of shape (num_vertices, 3)).
        f: Face indices (np.ndarray of shape (num_faces, 3)).
    returns:
        The average edge length (float).
    """

    return compute_edge_lengths(v, f).mean()
