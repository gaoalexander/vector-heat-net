from .next import next
from .twin import twin
from .global_variables import GHOST_HALF_EDGE
import numpy as np


def vertex_one_ring_half_edges(twins, v2he, v):
    """
    get the vertex one-ring half-edges emitting from the vertex v

    Inputs
    twins: (he,) array of twin half edges
    v2he: (V,) array of emitting half edge indices
    v: vertex index

    Outputs
    one_ring: np array of one ring half-edges starting from v

    Warning
    if the vertex v is a boundary vertex, this will only return the valid half-edges in the ccw order. Thus the twin(twins,one_ring[0]) is GHOST_HALF_EDGE, the twin(twins,next(next(one_ring[-1]))) is also GHOST_HALF_EDGE.
    """
    he = v2he[v]

    he_start = he
    one_ring = [he]
    while True:
        # get counter clockwise half edge
        he = twin(twins, next(next(he)))
        if he == he_start:
            # this is an interior vertex, so it reaches the starting he
            break
        if he == GHOST_HALF_EDGE:  # hit boundary, then go clock wise
            he = he_start
            while True:
                if twin(twins, he) == GHOST_HALF_EDGE:
                    break
                he = next(twin(twins, he))
                one_ring.insert(0, he)
            break
        one_ring.append(he)
    return np.array(one_ring)