import numpy as np
from scipy.sparse import csr_array


def construct_edge_length_map(v, f, edge_lengths):
    """
    f: faces
    l: edge lengths
    Returns
    -------
    edge_length_map:
        a |V|x|V| sparse array, where edge_length_map[i, j] corresponds to the length of edge ij
    """
    row, col, data = [], [], []
    for i in range(3):
        row.append(f[:, i].tolist())
        col.append(f[:, (i + 1) % 3].tolist())
        data.append(edge_lengths[:, i].tolist())
        # row.append(f[:, (i + 1) % 3].tolist())
        # col.append(f[:, i].tolist())
        # data.append(edge_lengths[:, i].tolist())

    row = np.array(row).flatten()
    col = np.array(col).flatten()
    data = np.array(data).flatten()

    visited = set()
    for i in range(row.shape[0]):
        edge = f"{row[i]},{col[i]}"
        if edge in visited:
            print(f"Edge {edge} is a duplicate!")
            exit()
        else:
            visited.add(edge)

    return csr_array((data, (row, col)), shape=(v.shape[0], v.shape[0]))
