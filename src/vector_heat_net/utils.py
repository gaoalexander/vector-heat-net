import hashlib
import json
import os

import numpy as np
import scipy
import torch


def complex_to_interleaved(complex_features):
    """Converts a tensor of complex scalars (representing tangent vectors) to interleaved real scalars.

    args:
        complex_features (torch.Tensor): A tensor of complex scalars with shape [1, V, C].
    returns:
        interleaved (torch.Tensor): A tensor of interleaved real scalars with shape [1, V * 2, C].
    """
    real = complex_features.real
    imag = complex_features.imag
    interleaved_features = torch.stack((real, imag), dim=2).view(complex_features.shape[0],
                                                                 complex_features.shape[1] * 2,
                                                                 complex_features.shape[2])
    return interleaved_features


def interleaved_to_complex(interleaved_features):
    """Converts a tensor of interleaved real scalars to complex scalars.

    Args:
        interleaved_features (torch.Tensor): A tensor of interleaved real scalars with shape [batch_size, V * 2, C].

    Returns:
        complex_features (torch.Tensor): A tensor of complex scalars with shape [batch_size, V, C].
    """

    # convert interleaved real scalars (which represent tangent vectors) into complex scalars
    real = interleaved_features[:, 0::2, :]
    imag = interleaved_features[:, 1::2, :] * (0 + 1j)
    complex_features = real + imag
    return complex_features


def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()

def label_smoothing_log_loss(pred, labels, smoothing=0.0):
    n_class = pred.shape[-1]
    one_hot = torch.zeros_like(pred)
    one_hot[labels] = 1.
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    loss = -(one_hot * pred).sum(dim=-1).mean()
    return loss


# Randomly rotate points.
# Torch in, torch out
# Note fornow, builds rotation matrix on CPU. 
def random_rotate_points(pts, randgen=None):
    R = random_rotation_matrix(randgen) 
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R) 

def random_rotate_points_y(pts):
    angles = torch.rand(1, device=pts.device, dtype=pts.dtype) * (2. * np.pi)
    rot_mats = torch.zeros(3, 3, device=pts.device, dtype=pts.dtype)
    rot_mats[0,0] = torch.cos(angles)
    rot_mats[0,2] = torch.sin(angles)
    rot_mats[2,0] = -torch.sin(angles)
    rot_mats[2,2] = torch.cos(angles)
    rot_mats[1,1] = 1.

    pts = torch.matmul(pts, rot_mats)
    return pts

# Numpy things

# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

def sparse_complex_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse_coo_tensor(torch.LongTensor(indices),
                                   torch.tensor(values, dtype=torch.complex64),
                                   torch.Size(shape)).coalesce()


# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat


# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()

def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randgen is None:
        randgen = np.random.RandomState()
        
    theta, phi, z = tuple(randgen.rand(3).tolist())
    
    theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

# Python string/file utilities
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


def convert_intrinsic2d_to_extrinsic3d(preds_2d, frames_verts):
    return (np.squeeze(preds_2d).real[:, None] * frames_verts[:, 0, :] +
            np.squeeze(preds_2d).imag[:, None] * frames_verts[:, 1, :])


def save_predictions(preds, targets, frames_verts, frames_faces, per_vertex_loss, output_filepath):
    preds = toNP(preds)
    targets = toNP(targets)
    frames_verts = toNP(frames_verts)
    frames_faces = toNP(frames_faces)

    # convert 2D (intrinsic) coordinates to 3D (extrinsic)
    preds_3d = convert_intrinsic2d_to_extrinsic3d(preds, frames_verts)
    targets_3d = convert_intrinsic2d_to_extrinsic3d(targets, frames_verts)

    with open(output_filepath, "w") as f:
        json.dump(
            {
                "preds": preds_3d.tolist(),
                "preds_local": toNP(complex_to_interleaved(torch.tensor(preds)).squeeze(0)).tolist(),
                "targets": targets_3d.tolist(),
                "axis_x_verts": frames_verts[:, 0, :].tolist(),
                "axis_y_verts": frames_verts[:, 1, :].tolist(),
                "axis_n_verts": frames_verts[:, 2, :].tolist(),
                "axis_x_faces": frames_faces[:, 0, :].tolist(),
                "axis_y_faces": frames_faces[:, 1, :].tolist(),
                "axis_n_faces": frames_faces[:, 2, :].tolist(),
                "per_vertex_loss": per_vertex_loss.tolist(),
            },
            f,
        )

