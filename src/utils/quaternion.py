"""Quaternion helpers used by the relative-root training loss."""

import torch


def qnormalize(q):
    """Normalize quaternions along the last dimension.

    Args:
        q (torch.Tensor): Tensor with shape ``(..., 4)`` in ``wxyz`` order.

    Returns:
        torch.Tensor: Unit quaternions with the same shape as ``q``.
    """
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    # Keep dimensions so broadcasting remains valid for arbitrary batch shapes.
    return q / torch.norm(q, dim=-1, keepdim=True)


def qbetween(v0, v1):
    """Compute the quaternion rotating one vector direction to another.

    Args:
        v0 (torch.Tensor): Source vectors with shape ``(..., 3)``.
        v1 (torch.Tensor): Target vectors with shape ``(..., 3)``.

    Returns:
        torch.Tensor: Unit quaternions with shape ``(..., 4)``.
    """
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    # The cross product gives the rotation axis; the scalar term encodes angle.
    axis = torch.cross(v0, v1, dim=-1)
    scalar = torch.sqrt(
        (v0**2).sum(dim=-1, keepdim=True) * (v1**2).sum(dim=-1, keepdim=True)
    ) + (v0 * v1).sum(dim=-1, keepdim=True)
    return qnormalize(torch.cat([scalar, axis], dim=-1))
