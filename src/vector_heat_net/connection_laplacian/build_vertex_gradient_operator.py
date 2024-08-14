import scipy

import numpy as np
from .build_implicit_half_edges import build_implicit_half_edges
from .half_edge_tangent_vectors import half_edge_tangent_vectors
from .vertex_one_ring_half_edges import vertex_one_ring_half_edges
from .tip_vertex import tip_vertex


def build_vertex_gradient_operator(V, F):
    print("Building INTRINSIC vertex-based gradient operator.")
    twins, tips = build_implicit_half_edges(F)
    hE_tangent_vectors, v2he, basis_x, basis_y, basis_n = half_edge_tangent_vectors(V, F, twins, tips, True)

    basis_x = basis_x / np.linalg.norm(basis_x, axis=1)[:, None]
    basis_y = basis_y / np.linalg.norm(basis_y, axis=1)[:, None]

    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5

    nV = V.shape[0]
    for v in range(nV):
        he_list = vertex_one_ring_half_edges(twins, v2he, v)
        # print(he_list)

        n_neigh = len(he_list)
        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [v]
        for ii in range(n_neigh):
            he = he_list[ii]
            he_vec = np.array([np.real(hE_tangent_vectors[he]), np.imag(hE_tangent_vectors[he])])
            w_e = 1.

            lhs_mat[ii][:] = w_e * he_vec
            rhs_mat[ii][0] = w_e * (-1)
            rhs_mat[ii][ii + 1] = w_e * 1

            v_tip = tip_vertex(tips, he)
            # v_tail = tail_vertex(tips, he)
            ind_lookup.append(v_tip)

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(v)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix((data_vals, (row_inds, col_inds)), shape=(nV, nV)).tocsc()
    return mat, basis_x, basis_y
