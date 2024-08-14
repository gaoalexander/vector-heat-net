import numpy as np
import scipy
from .half_edge_tangent_vectors import half_edge_tangent_vectors
from .tangent_space_rotations import tangent_space_rotations
from .cotangent_weights import cotangent_weights
from .twin import twin
from .next import next
# from .massmatrix import massmatrix
from .tip_vertex import tip_vertex
from .tail_vertex import tail_vertex
from .build_implicit_half_edges import build_implicit_half_edges

def connection_laplacian(V,F, twins=None, tips=None, return_basis_vectors=True, alternate_basis=False):
    """
    Build connection laplacian matrix

    Parameters
    ==========
    V: 2-d array
        (V,3) vertex locations
    F: 2-d array
        (F,3) face list
    twins: 1-d array
        (F*3,) array of twin half edge indices
    tips: 1-d array
        (F*3,) array of tip vertex indices of each half edgge
    return_basis_vectors: bool
        True/False of whether to return the tangent plane basis vectors in 3D (for visualization) 

    Returns
    =======
    Lc: 2-d scipy sparse complex csr array
        (V,V) array of connection laplacian (each entry is a complex number)
    v2he_x: 1-d array
        (V,) map from each vertex to the half-edge index representing basis x
    basis_x, basis_y: 2-d array
        (V,3) arrays representing tangent plane basis vectors in 3D (for visualization)
    """
    if twins is None or tips is None:
        twins, tips = build_implicit_half_edges(F)
    nV = V.shape[0]
    nF = F.shape[0]

    hE_tangent_vectors, v2he_x, basis_x, basis_y, basis_n = half_edge_tangent_vectors(V,F,twins,tips,True, alternate_basis=alternate_basis)
    rotation_along_halfedges = tangent_space_rotations(F, twins, hE_tangent_vectors)
    hE_cotangent_weights = cotangent_weights(V,F).flatten()

    rows = np.empty((nF*3*4,), dtype=int)
    cols = np.empty((nF*3*4,), dtype=int)
    vals = np.empty((nF*3*4,), dtype=np.csingle)
    for he in range(nF*3):
        he_twin = twin(twins, he)
        v_tail = tail_vertex(tips, he)
        v_tip = tip_vertex(tips, he)
        rot_he = rotation_along_halfedges[he] # TODO: power by number of symmetric rosy field
        rot_he_twin = rotation_along_halfedges[he_twin]
        cotan_he = hE_cotangent_weights[he]

        rows[he*4:(he+1)*4] = v_tail, v_tail, v_tip, v_tip
        cols[he*4:(he+1)*4] = v_tail, v_tip, v_tail, v_tip
        vals[he*4:(he+1)*4] = cotan_he, -cotan_he*rot_he, -cotan_he*rot_he_twin, cotan_he
    Lc = scipy.sparse.csr_array((vals, (rows, cols)), shape=(nV, nV), dtype=np.csingle)
    if return_basis_vectors:
        return Lc, v2he_x, basis_x, basis_y, basis_n
    else:
        return Lc