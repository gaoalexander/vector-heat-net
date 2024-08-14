import numpy as np
from .norm_row import norm_row

def half_edge_lengths(V,F):
    """
    compute the length of each half edge

    Parameters
    ==========
    V: 2-d array
        (V,3) vertex locations
    F: 2-d array
        (F,3) face list

    Returns
    =======
    hEl: 1-d array
        (F*3,) array of half edge's lengths, half edge index are computed by face_side_to_half_edge_index((f,s))
    """
    hE01 = norm_row(V[F[:,1],:] - V[F[:,0],:])
    hE12 = norm_row(V[F[:,2],:] - V[F[:,1],:])
    hE20 = norm_row(V[F[:,0],:] - V[F[:,2],:])
    return np.stack((hE01, hE12, hE20), axis=1).flatten()