import numpy as np
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.material import Material


def get_smpl_sequence(
    motion: dict, smpl_layer: SMPLLayer, body_shape: bool, downsample: int = 1
) -> SMPLSequence:
    """Create one aitviewer SMPL-X sequence from decoded motion parameters.

    Args:
        motion: Dictionary with SMPL-X pose, hand, translation, shape, and gender fields.
        smpl_layer: Shared SMPL-X layer used for both people in the interaction.
        body_shape: Whether to pass shape coefficients to aitviewer.
        downsample: Frame stride used when rendering long sequences.

    Returns:
        SMPLSequence: Renderable SMPL-X sequence for one person.
    """
    return SMPLSequence(
        poses_body=motion["pose_body"][::downsample],
        smpl_layer=smpl_layer,
        poses_root=motion["root_orient"][::downsample],
        betas=motion["betas"] if body_shape else None,
        trans=motion["trans"][::downsample],
        poses_left_hand=motion["pose_lhand"][::downsample],
        poses_right_hand=motion["pose_rhand"][::downsample],
    )


def get_interaction_smplx(
    motion_id: str,
    body_shape: bool,
    downsample: int = 1,
    motion1: dict | None = None,
    motion2: dict | None = None,
) -> tuple[SMPLSequence, SMPLSequence]:
    """Create renderable SMPL-X sequences for both people in an interaction.

    Args:
        motion_id: Motion identifier used only for error messages and output naming.
        body_shape: Whether to render with shape coefficients.
        downsample: Frame stride used when rendering long sequences.
        motion1: Decoded SMPL-X motion dictionary for the first person.
        motion2: Decoded SMPL-X motion dictionary for the second person.

    Returns:
        tuple[SMPLSequence, SMPLSequence]: Renderable sequences for both people.

    Raises:
        ValueError: If either decoded motion dictionary is missing.
    """
    if motion1 is None or motion2 is None:
        raise ValueError(f"Decoded motions are required to render {motion_id}.")

    # A single SMPL-X layer keeps both people in the same model coordinate frame.
    layer = SMPLLayer(model_type="smplx")
    sequence1 = get_smpl_sequence(motion1, layer, body_shape, downsample)
    sequence2 = get_smpl_sequence(motion2, layer, body_shape, downsample)

    # Use stable subject colors instead of textures so videos are easy to compare.
    sequence1.mesh_seq.material = Material(
        diffuse=0.75, ambient=0.15, color=(1.0, 0.23, 0.22, 1.0)
    )
    sequence2.mesh_seq.material = Material(
        diffuse=0.75, ambient=0.15, color=(0.23, 0.43, 1.0, 1.0)
    )

    return sequence1, sequence2


def _safe_normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector while avoiding division by zero.

    Args:
        vector: Vector to normalize.

    Returns:
        np.ndarray: Unit vector, or the original vector when its norm is near zero.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector
    return vector / norm


def create_camera_midpoint(
    sequence1: SMPLSequence,
    sequence2: SMPLSequence,
    distance: float,
    height: float,
    side: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a side camera that follows the midpoint between both people.

    Args:
        sequence1: Renderable sequence for the first person.
        sequence2: Renderable sequence for the second person.
        distance: Base camera distance from the interaction midpoint.
        height: Camera height above the pelvis midpoint.
        side: Whether to place the camera on one side or the opposite side.

    Returns:
        tuple[np.ndarray, np.ndarray]: Per-frame camera positions and targets.
    """
    pelvis1 = sequence1.joints[:, 0, :]
    pelvis2 = sequence2.joints[:, 0, :]
    midpoint = (pelvis1 + pelvis2) / 2

    positions = []
    targets = []
    for person1, person2, target in zip(pelvis1, pelvis2, midpoint, strict=True):
        direction = _safe_normalize(person2 - person1)
        perpendicular = _safe_normalize(np.cross(direction, np.array([0.0, 1.0, 0.0])))
        if not side:
            perpendicular = -perpendicular

        # Increase distance with subject separation so both people stay in frame.
        separation = np.linalg.norm(person2 - person1)
        camera_position = target + perpendicular * (distance + separation * 0.5)
        camera_position = camera_position.copy()
        camera_position[1] += height
        positions.append(camera_position)
        targets.append(target)

    return np.asarray(positions), np.asarray(targets)


def create_camera_top_down_view(
    sequence1: SMPLSequence, sequence2: SMPLSequence, height: float, top: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Create a top-down camera centered on both people.

    Args:
        sequence1: Renderable sequence for the first person.
        sequence2: Renderable sequence for the second person.
        height: Base camera height above the interaction midpoint.
        top: Whether to view from above or below the interaction plane.

    Returns:
        tuple[np.ndarray, np.ndarray]: Per-frame camera positions and targets.
    """
    pelvis1 = sequence1.joints[:, 0, :]
    pelvis2 = sequence2.joints[:, 0, :]
    midpoint = (pelvis1 + pelvis2) / 2
    separation = np.linalg.norm(pelvis2 - pelvis1, axis=1)

    height_offset = np.zeros_like(midpoint)
    signed_height = height + separation * 0.5
    height_offset[:, 1] = signed_height if top else -signed_height
    height_offset[:, 2] = 1e-6  # Avoid a perfectly vertical camera target direction.

    return midpoint + height_offset, midpoint
