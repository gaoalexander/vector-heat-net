import numpy as np

def gram_schmidt_orthogonalization(v, u):
    """
    fix u and make v be orthogonal to u

    Parameters
    ==========
    v: 2-d array or 1-d array
        (n,dim) array or (dim)
    u: 2-d array or 1-d array
        (n,dim) array or (dim)

    Returns
    =======
    v_ortho: 2-d array or 1-d array
        (n,dim) or (dim) array such that np.sum(v_ortho*u, 1) == 0
    """
    if len(v.shape) == 2:
        return v - (np.sum(v*u,1)/np.sum(u*u,1))[:,None] * u
    elif len(v.shape) == 1:
        return v - (v.dot(u) / u.dot(u)) * u 