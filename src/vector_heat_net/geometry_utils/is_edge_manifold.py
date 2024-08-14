import numpy as np


def is_edge_manifold(v, f):
    """
    Checks if a mesh is edge-manifold. (An edge-manifold mesh has a maximum of two faces sharing an edge.)

    Args:
        v: Vertex positions (np.ndarray of shape (num_vertices, 3)).
        f: Face indices (np.ndarray of shape (num_faces, 3)).

    Returns:
        True if the mesh is edge-manifold, False otherwise.
    """

    edges = np.concatenate((np.concatenate((f[:, 0], f[:, 1], f[:, 2]), axis=0)[:, None],
                            np.concatenate((f[:, 1], f[:, 2], f[:, 0]), axis=0)[:, None]), axis=1)
    adjacency_list = np.zeros((v.shape[0], v.shape[0]))
    for edge in edges:
        sorted_edge = sorted(edge.tolist())
        adjacency_list[sorted_edge[0], sorted_edge[1]] += 1
        if adjacency_list[sorted_edge[0], sorted_edge[1]] > 2:
            print(f"\tEdge [{sorted_edge[0]}, {sorted_edge[1]}] is non-manifold.  Stopping...")
            return False
    if np.any(adjacency_list == 1):
        return False
    return True
