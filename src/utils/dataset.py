import os

import numpy as np
import torch

import utils.rotation_conversions as geometry


def extract_motion_data(motions):
    """
    Extracts motion data from the batch.

    Args:
        batch: A batch of data containing motion information. (B,T,672)

    Returns:
        A tuple containing the motion data and its length.
    """

    B, T, _ = motions.shape

    poses = motions[:, :, : 55 * 12].reshape(
        B, T, 55, 12
    )  # Extracting the first 6 dimensions

    poses1 = poses[:, :, :, :6]  # Extracting the first 6 dimensions
    trans1 = motions[:, :, 55 * 12 : 55 * 12 + 6].reshape(
        B, T, 1, 6
    )  # Extracting the translation for the first person

    smpl_motion1 = {
        "root_orient": geometry.matrix_to_axis_angle(
            geometry.rotation_6d_to_matrix(poses1[:, :, 0, :])
        ),
        "pose_body": geometry.matrix_to_axis_angle(
            geometry.rotation_6d_to_matrix(poses1[:, :, 1:22, :])
        ).reshape(B, T, 63),
        "pose_lhand": geometry.matrix_to_axis_angle(
            geometry.rotation_6d_to_matrix(poses1[:, :, 25:40, :])
        ).reshape(B, T, 45),
        "pose_rhand": geometry.matrix_to_axis_angle(
            geometry.rotation_6d_to_matrix(poses1[:, :, 40:55, :])
        ).reshape(B, T, 45),
        "trans": trans1[:, :, 0, :3],
        "betas": torch.zeros((10)),
        "gender": "neutral",
    }

    poses2 = poses[:, :, :, 6:]
    trans2 = motions[:, :, 55 * 12 + 6 : 55 * 12 + 12].reshape(
        B, T, 1, 6
    )  # Extracting the translation for the second person

    smpl_motion2 = {
        "root_orient": geometry.matrix_to_axis_angle(
            geometry.rotation_6d_to_matrix(poses2[:, :, 0, :])
        ),
        "pose_body": geometry.matrix_to_axis_angle(
            geometry.rotation_6d_to_matrix(poses2[:, :, 1:22, :])
        ).reshape(B, T, 63),
        "pose_lhand": geometry.matrix_to_axis_angle(
            geometry.rotation_6d_to_matrix(poses2[:, :, 25:40, :])
        ).reshape(B, T, 45),
        "pose_rhand": geometry.matrix_to_axis_angle(
            geometry.rotation_6d_to_matrix(poses2[:, :, 40:55, :])
        ).reshape(B, T, 45),
        "trans": trans2[:, :, 0, :3],
        "betas": torch.zeros((10)),
        "gender": "neutral",
    }

    return smpl_motion1, smpl_motion2


class MotionNormalizerTorch:
    """Torch normalizer backed by stored Inter-X mean and standard deviation."""
    def __init__(self, representation="smpl"):
        """Load normalization statistics for one representation.

        Args:
            representation: Statistics prefix such as ``smpl`` or ``joints``.
        """
        mean = np.load(os.path.join("data", f"{representation}_mean.npy"))
        std = np.load(os.path.join("data", f"{representation}_std.npy"))
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()

    def forward(self, x):
        """Normalize a motion tensor.

        Args:
            x: Motion tensor whose last dimension matches the stored statistics.

        Returns:
            torch.Tensor: Normalized tensor on the same device as ``x``.
        """
        device = x.device
        x = x.clone()
        x = (x - self.mean.to(device)) / (self.std.to(device) + 1e-8)
        return x

    def backward(self, x):
        """Denormalize a motion tensor.

        Args:
            x: Normalized motion tensor.

        Returns:
            torch.Tensor: Tensor restored to the original representation scale.
        """
        device = x.device
        x = x.clone()
        x = x * self.std.to(device) + self.mean.to(device)
        return x
