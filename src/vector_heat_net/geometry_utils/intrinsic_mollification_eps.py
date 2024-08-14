from .avg_edge_length import avg_edge_length


def intrinsic_mollification_eps(v, f, edge_lengths):
    """
    Computes the epsilon value used for intrinsic mollification of edge lengths in a mesh.
    Intrinsic mollification helps improve the stability of Laplacian computations on meshes with poor triangle quality.
    (Reference: https://nmwsharp.com/media/papers/int-tri-course/int_tri_course.pdf#page=51.88)

    args:
        v: Vertex positions (np.ndarray of shape (num_vertices, 3)).
        f: Face indices (np.ndarray of shape (num_faces, 3)).
        edge_lengths: A numpy array of shape (num_faces, 3) containing edge lengths for each triangle.
    returns:
        eps: epsilon value for intrinsic mollification (float)
    """

    delta = 1e-5 * avg_edge_length(v, f)  # default value as suggested in https://nmwsharp.com/media/papers/int-tri-course/int_tri_course.pdf#page=51.88
    eps = 0
    for i in range(3):
        triangle_inequality_residual = edge_lengths[:, i] + edge_lengths[:, (i + 1) % 3] - edge_lengths[:, (i + 2) % 3]
        curr_max = max(0, (delta - triangle_inequality_residual).max())
        eps = max(eps, curr_max)
    print(f"\tEPS (intrinsic mollification): {eps}")
    return eps
