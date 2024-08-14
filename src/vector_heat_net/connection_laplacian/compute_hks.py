import numpy as np
import scipy
from scipy.sparse.linalg import eigsh


# def compute_hks(L, M, nt=100, tmin=0.3, tmax=4.0, eps=1e-8):
#     L = L.astype(np.float64)
#     M = np.diag(np.real(M).astype(np.float64))
#     cotan_L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()

#     print("\ncomputing eigenvalues...")
#     evals, evecs = eigsh(cotan_L_eigsh, k=200, M=M, sigma=eps)
#     # evals, evecs = eigsh(L, k=200, M=M)
#     print("finished computing eigenvalues.")
#     evecs_sqr = evecs ** 2
#     exp_m_evals = np.exp(-evals)

# #     t = np.logspace(-2, 0., num=nt)
#     t = np.exp(np.linspace(np.log(tmin), np.log(tmax), nt))
#     hks = evecs_sqr @ (exp_m_evals[:, None] * t[None, :])
#     return hks


def compute_hks(L, M, tmin = None, tmax = None, num_times = 100, num_eigen_basis=300, eps=1e-8):
    """
    compute heat kernal signature by Sun et al. 2009

    Inputs
        V: V by 3 vertex locations
        F: F by 3 face list
        tmin, tmax: min and max diffusion time
        num_times: number of time steps in the diffusion
        num_eigen_basis: number of eigenbasis in use

    Outputs
        hks: V by num_times heat kernal signature
    """
#     L = cotmatrix(V,F)
#     M = massmatrix(V,F)
    np.random.seed(0)
    
    L = L.astype(np.float64)
    M = np.diag(np.real(M).astype(np.float64))
    cotan_L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()

    print("\ncomputing eigenvalues...")
    evals, evecs = eigsh(cotan_L_eigsh, k=200, M=M, sigma=eps)
    # evals, evecs = eigsh(L, k=200, M=M)
    print("finished computing eigenvalues.")

#     evals, evecs = eigs(-L, M, num_eigen_basis)
    evecs_sqr = evecs**2

    if tmin is None:
        tmin = 4 * np.log(10.) / np.max(evals)
    if tmax is None:
        tmax = 4 * np.log(10.) / evals[1]

    t = np.logspace(tmin,tmax,num_times)

    # hks = evecs_sqr @ (exp_m_evals[:,None] * t[None,:])
    hks = evecs_sqr @ np.exp(-evals[:,None] * t[None,:])
    # MK = M.diagonal()[:,None] * hks
    return hks