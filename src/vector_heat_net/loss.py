import torch
import torch.nn.functional as F
from vector_heat_net.layers import complex_to_interleaved


def complex_mse_loss(output, target):
    """Computes the complex mean squared error (MSE) loss.
    This function computes the mean squared error between two complex-valued tensors.

    Args:
        output: The predicted complex tensor.
        target: The ground truth complex tensor.
    Returns:
        A tensor representing the complex MSE loss.
    """
    return (torch.abs(output.squeeze() - target.squeeze()) ** 2)


def complex_nmse_loss(output, target, eps=1e-8):
    """
    Computes the normalized mean square error (NMSE) loss for complex tensors.

    This function computes the normalized mean squared error between two complex-valued tensors.
    The denominator is clamped to avoid division by zero.

    Args:
        output: The predicted complex tensor.
        target: The ground truth complex tensor.
    Returns:
        A tensor representing the normalized mean squared error loss.
    """
    return (torch.abs(output.squeeze() - target.squeeze()) ** 2) / torch.clamp(torch.abs(target.squeeze()), min=eps, max=1.0)


def size_loss(y, y_pred, relative=True, eps=None):
    """Computes the size loss between two tensors.

    Args:
        y: The ground truth tensor.
        y_pred: The predicted tensor.
        relative: If True, computes relative size loss. Defaults to True.
        eps: A small value to prevent division by zero. Defaults to None.
    Returns:
        A tensor representing the size loss.
    """

    y = y.squeeze()[:, None]
    y_pred = y_pred.squeeze()[:, None]

    y_norm = torch.linalg.norm(y, dim=1)
    y_pred_norm = torch.linalg.norm(y_pred, dim=1)

    if relative:
        return torch.abs((y_pred_norm - y_norm) / torch.maximum(eps, y_norm))
    else:
        return torch.abs(y_pred_norm - y_norm)


def complex_cosine_loss(y, y_pred, eps=torch.Tensor([1e-8])):
    """Computes the complex cosine loss between two complex tensors.

    Args:
        y: The ground truth complex tensor.
        y_pred: The predicted complex tensor.
        eps: A small value to prevent division by zero. Defaults to 1e-8.
    Returns:
        A tensor representing the complex cosine loss.
    """
    # view complex-valued data as real-valued
    y = torch.view_as_real(y.squeeze())
    y_pred = torch.view_as_real(y_pred.squeeze())

    # normalize per-vertex vectors to unit length
    y_norm = torch.linalg.norm(y, dim=1)[:, None]
    y_pred_norm = torch.linalg.norm(y_pred, dim=1)[:, None]

    y_normalized = y / torch.maximum(eps, y_norm)
    y_pred_normalized = y_pred / torch.maximum(eps, y_pred_norm)

    return 1.0 - torch.bmm(y_normalized[:, None, :], y_pred_normalized[:, :, None]).squeeze()


def von_mises_loss(y, y_pred, kappa=1.0):
    """Computes the von Mises loss between two tensors.

    Args:
        y: The ground truth tensor.
        y_pred: The predicted tensor.
        kappa: Concentration parameter for the von Mises distribution. Defaults to 1.0.
    Returns:
        A tensor representing the von Mises loss.
    """
    y = torch.view_as_real(y).squeeze()
    y_pred = torch.view_as_real(y_pred).squeeze()

    batchsize = y.shape[0]
    vec_len = y.shape[1]
    y = F.normalize(y, dim=1)
    y_pred = F.normalize(y_pred, dim=1)
    dot_product = torch.bmm(
        y.view(batchsize, 1, vec_len), y_pred.view(batchsize, vec_len, 1)
    ).reshape(-1)
#     return 1 - torch.exp(kappa * (8 * (dot_product ** 4 - dot_product ** 2))) # invariant to pi/2 rotations
#     return 1 - torch.exp(kappa * (dot_product - 1)) # no rotation invariance
    return 1 - torch.exp(kappa * (2 * (dot_product ** 2) - 2)) # invariant to pi rotations


def total_loss(u, v, u_pred, v_pred, size_relative=True, eps=None):
    """Computes the total loss between two pairs of tensors.

    Args:
        u: The ground truth tensor for the first component.
        v: The ground truth tensor for the second component.
        u_pred: The predicted tensor for the first component.
        v_pred: The predicted tensor for the second component.
        size_relative: If True, computes relative size loss. Defaults to True.
        eps: A small value to prevent division by zero. Defaults to None.
    Returns:
        A tensor representing the total loss.
    """
    direction_loss = von_mises_loss(u, u_pred) + von_mises_loss(v, v_pred)
    _size_loss = size_loss(u, u_pred, relative=size_relative, eps=eps) + size_loss(v, v_pred, relative=size_relative, eps=eps)
    return direction_loss + _size_loss
