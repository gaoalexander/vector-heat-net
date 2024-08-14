import numpy as np
from .twin import twin
from .global_variables import GHOST_HALF_EDGE

def tangent_space_rotations(F,twins,tangent_vectors):
    """
    compute rotation (as a complex number) between neighborhood tangent planes

    Parameters
    ==========
    F: 2-d array
        (F,3) face indices
    twins: 1-d array
        (F*3,) twin half edges
    tangent_vectors: 1-d complex array
        (F*3,) array of half-edges represented as tangent vectors (complex numbers) at their tail vertices

    Returns
    =======
    rotation_along_halfedges: 1-d array
        (F*3,) array of tangent space rotation (complex number) when transporting tangent vectors along the half edge
    """
    nF = F.shape[0]
    rotation_along_halfedges = np.zeros((nF*3,), dtype=np.csingle)
    hE_visited = np.full((nF*3,), False, dtype=bool)
    for he in range(nF*3):
        if hE_visited[he] == False:
            heA = he
            heB =  twin(twins, heA)

            if heB == GHOST_HALF_EDGE: # if boundary edge, then the rotation is pi
                rotation_along_halfedges[heA] = np.cos(np.pi/2) + np.sin(np.pi/2) * 1j
                hE_visited[heA] = True
            else:
                vecA = tangent_vectors[heA]
                vecB = tangent_vectors[heB]

                rot = -vecA / vecB # complex number division
                rot = rot / np.absolute(rot) # complex number normalization
                rot_inverse = np.conjugate(rot) # rot is unit vec, so conjugate == inverse
                rotation_along_halfedges[heA] = rot
                rotation_along_halfedges[heB] = rot_inverse

                hE_visited[heA] = True
                hE_visited[heB] = True
    return rotation_along_halfedges
