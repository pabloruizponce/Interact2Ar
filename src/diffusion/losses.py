"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th

import utils.quaternion as quaternion
import utils.rotation_conversions as geometry


def seq_masked_mse(prediction, target, mask):
    """
    Calculates the masked mean squared error for a sequence.
    The loss is first averaged over the feature dimension, then a masked
    average is taken over the batch and time dimensions.
    """
    assert prediction.shape == target.shape, (
        f"Prediction shape {prediction.shape} does not match target shape {target.shape}"
    )
    # The original assertion was for a specific feature dimension.
    # It might be better to make this more general if your features can change.
    # assert prediction.shape[-1] == 672

    # Calculate squared error -> shape: (B, T, Feature_Dim)
    loss = (target - prediction) ** 2

    # 1. Average over the feature dimension first.
    loss = loss.mean(dim=-1, keepdim=True)  # -> shape: (B, T, 1)

    # 2. Apply the timestep mask and average over batch and time.
    # mask should have shape (B, T, 1)
    loss = (loss * mask).sum() / (mask.sum() + 1.0e-7)
    return loss


def mix_masked_mse(prediction, target, mask, timestep_mask, contact_mask=None):
    """
    Calculates a more complex masked MSE.
    It handles an optional contact_mask for features and averages
    over time and batch dimensions in separate, masked steps.
    """
    assert prediction.shape == target.shape, (
        f"Prediction shape {prediction.shape} does not match target shape {target.shape}"
    )

    # Initial squared error calculation
    loss = (target - prediction) ** 2

    # If prediction has 4 dimensions, average over the last one to get to 3D
    if loss.ndim == 4:
        # 1. Average over the feature dimension, using contact_mask if provided.
        if contact_mask is not None:
            # Masked average over the feature dimension
            loss = (loss * contact_mask).sum(dim=-1) / (
                contact_mask.sum(dim=-1) + 1.0e-7
            )
            loss = loss.mean(dim=-1, keepdim=True)
        else:
            # Simple average over the feature dimension
            loss = loss.mean(dim=-1)
            loss = loss.mean(dim=-1, keepdim=True)
    elif loss.ndim == 3:
        # 1. Average over the feature dimension, using contact_mask if provided.
        if contact_mask is not None:
            # Masked average over the feature dimension
            loss = (loss * contact_mask).sum(dim=-1, keepdim=True) / (
                contact_mask.sum(dim=-1, keepdim=True) + 1.0e-7
            )
        else:
            # Simple average over the feature dimension
            loss = loss.mean(dim=-1, keepdim=True)
    # At this point, loss has a consistent shape of (B, T, 1)

    # 2. Apply the primary mask and average over the time dimension.
    # The original sum over (-1, -2) is correct for a (B, T, 1) tensor.
    loss = (loss * mask).sum(dim=(-1, -2)) / (
        mask.sum(dim=(-1, -2)) + 1.0e-7
    )  # -> shape: (B,)

    # 3. Apply the timestep_mask and average over the batch dimension.
    loss = (loss * timestep_mask).sum(dim=0) / (
        timestep_mask.sum(dim=0) + 1.0e-7
    )  # -> shape: scalar
    return loss


def relative_root(p1, p2):
    """Compute the relative facing direction between two root orientations.

    Args:
        p1: Axis-angle root orientation for the first person.
        p2: Axis-angle root orientation for the second person.

    Returns:
        torch.Tensor: Two-channel quaternion slice used by the root-orientation loss.
    """
    p1_matrix = geometry.axis_angle_to_matrix(p1)
    p1_forward = p1_matrix[..., :, 2]

    p2_matrix = geometry.axis_angle_to_matrix(p2)
    p2_forward = p2_matrix[..., :, 2]

    rel_root = quaternion.qbetween(p1_forward, p2_forward)
    return rel_root[..., [0, 2]]


def distance_map(joints1, joints2):
    """Compute pairwise distances and a close-contact mask between two bodies.

    Args:
        joints1: Joint tensor for the first person.
        joints2: Joint tensor for the second person.

    Returns:
        tuple: Pairwise distance matrix and a binary mask for distances below one meter.
    """
    thresh = 1.0
    distance_matrix = th.cdist(joints1, joints2)
    mask = (distance_matrix < thresh).float()
    return distance_matrix, mask


def foot_contact(output_body, target_body, mask, timestep_mask):
    """Penalize predicted foot motion when the target foot is in contact.

    Args:
        output_body: Predicted world-space body joints with shape ``(B, T, J, 3)``.
        target_body: Target world-space body joints with shape ``(B, T, J, 3)``.
        mask: Temporal mask for valid autoregressive prediction frames.
        timestep_mask: Diffusion timestep mask used by the mixed loss schedule.

    Returns:
        torch.Tensor: Masked foot-contact velocity loss.
    """
    fids = [7, 10, 8, 11]

    feet_vel = target_body[:, 1:, fids, :] - target_body[:, :-1, fids, :]
    output_feet_vel = output_body[:, 1:, fids, :] - output_body[:, :-1, fids, :]
    feet_h = target_body[:, :-1, fids, 1]

    velfactor, heightfactor = (
        th.Tensor([0.0001, 0.0001, 0.0001, 0.0001]).to(feet_vel.device),
        th.Tensor([0.005, 0.005, 0.005, 0.005]).to(feet_vel.device),
    )

    feet_x = (feet_vel[..., 0]) ** 2
    feet_y = (feet_vel[..., 1]) ** 2
    feet_z = (feet_vel[..., 2]) ** 2

    contact = (
        ((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)
    ).float()

    return mix_masked_mse(
        output_feet_vel,
        th.zeros_like(output_feet_vel),
        mask[:, :-1],
        timestep_mask,
        contact.unsqueeze(-1),
    )


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """Compute discretized Gaussian log likelihoods.

    Args:
        x: Target values, historically image pixels rescaled to ``[-1, 1]``.
        means: Gaussian means with the same shape as ``x``.
        log_scales: Gaussian log standard deviations with the same shape as ``x``.

    Returns:
        torch.Tensor: Elementwise log probabilities in nats.
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
