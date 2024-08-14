import numpy as np
from scipy.sparse import csr_array

from .intrinsic_cotan_weights import intrinsic_cotan_weights
from .compute_parallel_transport_intrinsic import compute_parallel_transport_intrinsic


def build_connection_laplacian_intrinsic(v, f, intrinsic_mollification=True, alternate_basis=False):
    """
    Builds the intrinsic connection Laplacian matrix.

    args:
        v: Vertex positions (np.ndarray of shape (num_vertices, 3)).
        f: Face indices (np.ndarray of shape (num_faces, 3)).
        intrinsic_mollification: If True, applies intrinsic mollification to edge lengths. Defaults to True.
        alternate_basis: If True, uses an alternate basis for the Laplacian. Defaults to False.
    returns:
        A tuple containing:
            - The connection Laplacian matrix (csr_array of shape (num_vertices, num_vertices)).
            - Basis vectors X (np.ndarray of shape (num_vertices, 3)).
            - Basis vectors Y (np.ndarray of shape (num_vertices, 3)).
            - Basis vectors N (np.ndarray of shape (num_vertices, 3)).
            - Mass matrix M (np.ndarray of shape (num_vertices, num_vertices)).
    """

    row, col, data = [], [], []
    weights = intrinsic_cotan_weights(v, f, intrinsic_mollification=intrinsic_mollification)
    weights = weights.astype(np.csingle)
    parallel_transport_rotations, basisX, basisY, basisN, M = compute_parallel_transport_intrinsic(v, f, intrinsic_mollification=intrinsic_mollification, alternate_basis=alternate_basis)

    for i in range(3):
        r = parallel_transport_rotations[:, i]
        row.append(f[:, i].tolist())
        col.append(f[:, (i + 1) % 3].tolist())
        data.append(-weights[:, (i + 2) % 3] / r)
        # data.append(-weights[:, (i + 2) % 3])

        row.append(f[:, (i + 1) % 3].tolist())
        col.append(f[:, i].tolist())
        data.append(-weights[:, (i + 2) % 3] * r)
        # data.append(-weights[:, (i + 2) % 3])

        row.append(f[:, i].tolist())
        col.append(f[:, i].tolist())
        data.append(weights[:, (i + 1) % 3])

        row.append(f[:, i].tolist())
        col.append(f[:, i].tolist())
        data.append(weights[:, (i + 2) % 3])

    row = np.array(row).flatten()
    col = np.array(col).flatten()
    data = np.array(data).flatten()
    return csr_array((data, (row, col)), shape=(v.shape[0], v.shape[0]), dtype=np.csingle), basisX, basisY, basisN, M
