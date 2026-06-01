import os

import numpy as np
import torch
from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import PinholeCamera
from scipy.ndimage import gaussian_filter1d

import utils.rotation_conversions as geometry
from visualize.visualize import (
    create_camera_midpoint,
    create_camera_top_down_view,
    get_interaction_smplx,
)

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
N_SMPLX_JOINTS = 55
POSE_FEATURES = N_SMPLX_JOINTS * 12
TRANS_FEATURES = 12


def _configure_headless_backend():
    """Configure aitviewer's ModernGL backend for headless containers.

    Aitviewer does not expose moderngl-window's ``backend`` argument, so Docker
    runs fall back to X11 unless we set the backend before the context is built.
    Set ``INTERACT2AR_RENDER_BACKEND=egl`` to force NVIDIA EGL rendering.
    """
    if not os.environ.get("INTERACT2AR_RENDER_BACKEND"):
        return

    from moderngl_window.context.headless.window import Window as HeadlessWindow

    if getattr(HeadlessWindow, "_interact2ar_backend_patch", False):
        return

    original_init_mgl_context = HeadlessWindow.init_mgl_context

    def init_mgl_context(self):
        """Initialize ModernGL with the backend requested by Interact2Ar."""
        # The backend is read lazily so tests or scripts can set it at runtime.
        if self._backend is None:
            self._backend = os.environ.get("INTERACT2AR_RENDER_BACKEND")
        return original_init_mgl_context(self)

    # This intentionally patches the headless window class used by aitviewer.
    HeadlessWindow.init_mgl_context = init_mgl_context
    HeadlessWindow._interact2ar_backend_patch = True


def create_renderer(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Create an aitviewer renderer only when a video is requested.

    Args:
        width: Output video width in pixels.
        height: Output video height in pixels.

    Returns:
        HeadlessRenderer: Renderer configured for 30 FPS playback.
    """
    # Renderer creation touches the OpenGL context, so keep it out of imports.
    _configure_headless_backend()
    renderer = HeadlessRenderer(size=(width, height))
    renderer.playback_fps = 30.0
    return renderer


def _axis_angle(array):
    """Convert 6D rotations to numpy axis-angle vectors.

    Args:
        array: Tensor containing one or more 6D rotations.

    Returns:
        np.ndarray: Axis-angle rotations detached on CPU.
    """
    return (
        geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(array))
        .cpu()
        .detach()
        .numpy()
    )


def _decode_person_motion(poses_6d, trans_6d):
    """Decode one person's flattened Interact2Ar motion into SMPL-X fields.

    Args:
        poses_6d: Tensor of shape ``(T, 55, 6)`` with SMPL-X joint rotations.
        trans_6d: Tensor of shape ``(T, 6)`` with global translation features.

    Returns:
        dict: Motion dictionary accepted by the aitviewer SMPL-X sequence helper.
    """
    # The model stores rotations in 6D; aitviewer expects axis-angle SMPL-X poses.
    return {
        "root_orient": _axis_angle(poses_6d[:, 0, :]),
        "pose_body": _axis_angle(poses_6d[:, 1:22, :]).reshape(-1, 63),
        "pose_lhand": _axis_angle(poses_6d[:, 25:40, :]).reshape(-1, 45),
        "pose_rhand": _axis_angle(poses_6d[:, 40:55, :]).reshape(-1, 45),
        "trans": trans_6d[:, :3].cpu().detach().numpy(),
        "betas": np.zeros(10, dtype=np.float32),
        "gender": "neutral",
    }


def extract_motion_data(motion):
    """Split a generated interaction tensor into two SMPL-X motion dictionaries.

    Args:
        motion: Tensor of shape ``(T, 672)`` in the Interact2Ar 6D representation.

    Returns:
        tuple[dict, dict]: SMPL-X motion dictionaries for person one and person two.
    """
    poses = motion[:, :POSE_FEATURES].reshape(-1, N_SMPLX_JOINTS, 12)
    translations = motion[:, POSE_FEATURES : POSE_FEATURES + TRANS_FEATURES].reshape(
        -1, 2, 6
    )

    motion1 = _decode_person_motion(poses[:, :, :6], translations[:, 0, :])
    motion2 = _decode_person_motion(poses[:, :, 6:], translations[:, 1, :])
    return motion1, motion2


def visualize_trajectory(trajectory, color=(0, 0, 1, 1)):
    """Create an animated root trajectory line.

    Args:
        trajectory: Array of shape ``(T, 3)`` with root positions.
        color: RGBA line color.

    Returns:
        Lines: aitviewer renderable that reveals the path over time.
    """
    from aitviewer.renderables.lines import Lines

    # Each frame contains the path observed up to that frame.
    num_frames = trajectory.shape[0]
    frame_indices = np.arange(num_frames)[:, np.newaxis]
    point_indices = np.arange(num_frames)[np.newaxis, :]
    animated_trajectory = trajectory[np.minimum(frame_indices, point_indices)]
    return Lines(animated_trajectory, color=color)


def visualize_motion(
    motion,
    motion_id="predicted",
    trajectory=True,
    top=False,
    camera_follow=True,
    quality="low",
):
    """Render one generated interaction with aitviewer.

    Args:
        motion: Generated motion tensor in the flattened 6D representation.
        motion_id: Output stem used by the renderer.
        trajectory: Whether to draw root trajectories for both people.
        top: Whether to use the top-down camera preset.
        camera_follow: Whether the camera follows the interaction midpoint.
        quality: Rendering quality preset passed to aitviewer.
    """
    smoothed_motion = gaussian_filter1d(
        motion.cpu().detach().numpy(), sigma=3, axis=0, mode="nearest"
    )
    smpl_motion1, smpl_motion2 = extract_motion_data(torch.as_tensor(smoothed_motion))

    sequence1, sequence2 = get_interaction_smplx(
        motion_id,
        body_shape=True,
        motion1=smpl_motion1,
        motion2=smpl_motion2,
        downsample=1,
    )

    if top:
        camera_positions, camera_targets = create_camera_top_down_view(
            sequence1, sequence2, 5.0
        )
    else:
        camera_positions, camera_targets = create_camera_midpoint(
            sequence1, sequence2, 2.0, 1.0, side=True
        )

    if not camera_follow:
        camera_positions = camera_positions.mean(axis=0, keepdims=True)
        camera_targets = camera_targets.mean(axis=0, keepdims=True)

    renderer = create_renderer()
    camera = PinholeCamera(
        position=camera_positions,
        target=camera_targets,
        cols=renderer.window_size[0],
        rows=renderer.window_size[1],
        viewer=renderer,
    )

    # Add camera and renderables explicitly so the cleanup path mirrors setup.
    renderer.scene.add(camera)
    renderer.set_temp_camera(camera)
    renderer.scene.add(sequence1)
    renderer.scene.add(sequence2)

    trajectory1 = trajectory2 = None
    if trajectory:
        trajectory1 = visualize_trajectory(smpl_motion1["trans"], color=(1, 0, 0, 1))
        trajectory2 = visualize_trajectory(smpl_motion2["trans"], color=(0, 0, 1, 1))
        renderer.scene.add(trajectory1)
        renderer.scene.add(trajectory2)

    renderer.save_video(
        video_dir=f"{motion_id}.mp4",
        output_fps=30,
        quality=quality,
        scale_factor=1.0,
    )

    renderer.scene.remove(sequence1)
    renderer.scene.remove(sequence2)
    renderer.scene.remove(camera)
    if trajectory1 is not None and trajectory2 is not None:
        renderer.scene.remove(trajectory1)
        renderer.scene.remove(trajectory2)
