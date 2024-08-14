import numpy as np
from .next import next
from .global_variables import EPS

def opposite_corner_angle(hEl, he, verbose=True):
    """
    Computes triangle corner angle opposite the face-side fs.

    Parameters
    ==========
    hEl: 1-d array
        (|F|*3,) array of half edge's lengths
    he: int
        half edge index

    Outputs
    The corner angle opposite to (f,s), in radians
    """
    # Gather edge lengths
    l_a = hEl[he] 
    l_b = hEl[next(he)] 
    l_c = hEl[next(next(he))] 

    # uniformly scale the edge length to increase numerical robustness
    l_sum = l_a + l_b + l_c
    l_a = l_a / l_sum
    l_b = l_b / l_sum
    l_c = l_c / l_sum

    # Law of cosines (inverse)
    d = (l_b**2 + l_c**2 - l_a**2) / (2*l_b*l_c)

    if np.abs(d) <= 1: # satisfy triangle inequality
        return np.arccos(d)
    if np.abs(d) > 1 and np.abs(d) < 1+EPS: # slightly violate triangle inequality
        d = np.clip(d, -1, 1)
        return np.arccos(d)
    else:
        if verbose:
            print("d: ", d)
            print("l: ", l_a, l_b, l_c)
            print("In opposite_corner_angle.py, the triangle violates triangle inequality.")
        return np.NAN

