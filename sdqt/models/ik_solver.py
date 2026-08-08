"""FABRIK IK solver for Daz2Supreme bone posing.

FABRIK (Forward And Backward Reaching Inverse Kinematics) iteratively
adjusts bone positions to reach a target, then converts solved positions
back to Euler rotations compatible with the existing FK pose system.

Predefined IK chains map common end effectors (hands, feet) to their
parent bone chains in the skeleton hierarchy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sdqt.models.scene3d import Bone, Figure, Skeleton


# ── Predefined IK chains for humanoid skeletons ─────────────────────────────

# (end_effector, chain_length) — chain walks up parent_index
_IK_CHAINS: dict[str, int] = {
    # Arms: hand → forearm → shoulder (3 bones)
    "lHand": 3,
    "rHand": 3,
    # Legs: foot → shin → thigh (3 bones)
    "lFoot": 3,
    "rFoot": 3,
    # Fingers: tip → mid → base (3 bones)
    "lThumb3": 3, "rThumb3": 3,
    "lIndex3": 3, "rIndex3": 3,
    "lMid3": 3, "rMid3": 3,
    "lRing3": 3, "rRing3": 3,
    "lPinky3": 3, "rPinky3": 3,
    # Head: head → neck (2 bones)
    "head": 2,
}


@dataclass
class IKChain:
    """An IK chain: ordered list of bone indices from root to end effector."""
    bone_indices: list[int] = field(default_factory=list)
    bone_names: list[str] = field(default_factory=list)


def find_ik_chain(skeleton: Skeleton, end_effector: str,
                  chain_length: int | None = None) -> IKChain | None:
    """Build an IK chain by walking up from end_effector through parents.

    If chain_length is None, uses the predefined length for known bones,
    or defaults to 3.
    """
    idx = skeleton.bone_index(end_effector)
    if idx < 0:
        return None

    if chain_length is None:
        chain_length = _IK_CHAINS.get(end_effector, 3)

    # Walk up parent chain
    indices = [idx]
    names = [skeleton.bones[idx].name]
    current = idx
    for _ in range(chain_length - 1):
        parent = skeleton.bones[current].parent_index
        if parent < 0:
            break
        indices.append(parent)
        names.append(skeleton.bones[parent].name)
        current = parent

    # Reverse so chain goes root → ... → end_effector
    indices.reverse()
    names.reverse()
    return IKChain(bone_indices=indices, bone_names=names)


def get_available_ik_chains(skeleton: Skeleton) -> dict[str, IKChain]:
    """Return all predefined IK chains that exist in the skeleton."""
    chains: dict[str, IKChain] = {}
    for effector, length in _IK_CHAINS.items():
        chain = find_ik_chain(skeleton, effector, length)
        if chain and len(chain.bone_indices) >= 2:
            chains[effector] = chain
    return chains


# ── FABRIK solver ────────────────────────────────────────────────────────────

def _get_chain_world_positions(
    figure: Figure,
    chain: IKChain,
    bone_poses: dict[str, tuple[float, float, float]],
) -> list[np.ndarray]:
    """Compute world positions for each bone in the IK chain."""
    from sdqt.widgets.gl_viewport_pbr import _translation_matrix, _rotation_matrix

    skel = figure.skeleton
    n = len(skel.bones)

    # Compute full world matrices (same as _compute_pose_matrices but
    # we only need positions, not skinning transforms)
    world_mats = [np.eye(4, dtype=np.float32) for _ in range(n)]
    for i, bone in enumerate(skel.bones):
        t = _translation_matrix(bone.rest_position)
        r = _rotation_matrix(bone.rest_rotation)
        pose_rot = bone_poses.get(bone.name)
        if pose_rot is not None:
            r = r @ _rotation_matrix(np.array(pose_rot, dtype=np.float32))
        local = t @ r
        if bone.parent_index >= 0:
            world_mats[i] = world_mats[bone.parent_index] @ local
        else:
            world_mats[i] = local

    return [world_mats[idx][:3, 3].copy() + figure.position
            for idx in chain.bone_indices]


def solve_fabrik(
    figure: Figure,
    chain: IKChain,
    target: np.ndarray,
    bone_poses: dict[str, tuple[float, float, float]],
    iterations: int = 10,
    tolerance: float = 0.001,
) -> dict[str, tuple[float, float, float]]:
    """Run FABRIK to solve the IK chain toward target.

    Returns updated bone_poses dict with new rotations for chain bones.
    """
    if len(chain.bone_indices) < 2:
        return bone_poses

    # Get current world positions
    positions = _get_chain_world_positions(figure, chain, bone_poses)
    n = len(positions)

    # Compute segment lengths
    lengths = []
    for i in range(n - 1):
        lengths.append(float(np.linalg.norm(positions[i + 1] - positions[i])))

    total_length = sum(lengths)
    target = np.array(target, dtype=np.float32)
    root = positions[0].copy()

    # Check if target is reachable
    dist_to_target = float(np.linalg.norm(target - root))
    if dist_to_target > total_length:
        # Stretch toward target
        direction = (target - root) / (dist_to_target + 1e-8)
        for i in range(n - 1):
            positions[i + 1] = positions[i] + direction * lengths[i]
    else:
        # FABRIK iterations
        for _iter in range(iterations):
            # Check convergence
            if np.linalg.norm(positions[-1] - target) < tolerance:
                break

            # ── Backward pass (end → root) ──
            positions[-1] = target.copy()
            for i in range(n - 2, -1, -1):
                direction = positions[i] - positions[i + 1]
                dist = np.linalg.norm(direction)
                if dist < 1e-8:
                    direction = np.array([0, 1, 0], dtype=np.float32)
                else:
                    direction /= dist
                positions[i] = positions[i + 1] + direction * lengths[i]

            # ── Forward pass (root → end) ──
            positions[0] = root
            for i in range(n - 1):
                direction = positions[i + 1] - positions[i]
                dist = np.linalg.norm(direction)
                if dist < 1e-8:
                    direction = np.array([0, 1, 0], dtype=np.float32)
                else:
                    direction /= dist
                positions[i + 1] = positions[i] + direction * lengths[i]

    # ── Convert solved positions back to Euler rotations ──
    result = dict(bone_poses)
    skel = figure.skeleton

    for ci in range(n - 1):
        bone_idx = chain.bone_indices[ci]
        child_idx = chain.bone_indices[ci + 1]
        bone = skel.bones[bone_idx]

        # Desired direction: from this bone to next in solved positions
        desired_dir = positions[ci + 1] - positions[ci]
        desired_len = np.linalg.norm(desired_dir)
        if desired_len < 1e-8:
            continue
        desired_dir /= desired_len

        # Rest direction: from this bone to child in rest pose
        child_bone = skel.bones[child_idx]
        rest_dir = child_bone.rest_position.copy()
        rest_len = np.linalg.norm(rest_dir)
        if rest_len < 1e-8:
            continue
        rest_dir /= rest_len

        # Compute rotation from rest_dir to desired_dir
        rot = _rotation_between_vectors(rest_dir, desired_dir)

        # Convert to Euler XYZ degrees
        euler = _matrix_to_euler_xyz(rot)

        # Subtract rest rotation to get pose-only rotation
        euler_deg = np.degrees(euler).astype(np.float32)
        pose_rot = euler_deg - bone.rest_rotation

        # Apply joint limits
        if bone.rotation_min is not None and bone.rotation_max is not None:
            pose_rot = np.clip(pose_rot, bone.rotation_min, bone.rotation_max)

        result[bone.name] = (float(pose_rot[0]), float(pose_rot[1]), float(pose_rot[2]))

    return result


# ── Math helpers ─────────────────────────────────────────────────────────────

def _rotation_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Compute 3x3 rotation matrix that rotates v_from to v_to."""
    v_from = v_from / (np.linalg.norm(v_from) + 1e-8)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-8)

    cross = np.cross(v_from, v_to)
    dot = float(np.dot(v_from, v_to))

    if dot > 0.9999:
        return np.eye(3, dtype=np.float32)
    if dot < -0.9999:
        # 180 degree rotation — find perpendicular axis
        perp = np.array([1, 0, 0], dtype=np.float32)
        if abs(v_from[0]) > 0.9:
            perp = np.array([0, 1, 0], dtype=np.float32)
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        # Rodrigues for 180 degrees
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ], dtype=np.float32)
        return (np.eye(3) + 2 * K @ K).astype(np.float32)

    # Rodrigues' rotation formula
    K = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + K + (K @ K) / (1.0 + dot)
    return R


def _matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Extract Euler XYZ angles (radians) from a 3x3 rotation matrix."""
    sy = float(R[0, 2])
    sy = max(-1.0, min(1.0, sy))

    if abs(sy) < 0.9999:
        x = math.atan2(-float(R[1, 2]), float(R[2, 2]))
        y = math.asin(sy)
        z = math.atan2(-float(R[0, 1]), float(R[0, 0]))
    else:
        # Gimbal lock
        x = math.atan2(float(R[2, 1]), float(R[1, 1]))
        y = math.asin(sy)
        z = 0.0

    return np.array([x, y, z], dtype=np.float32)


def screen_to_world_ray(
    mx: int, my: int, width: int, height: int,
    vp: np.ndarray, cam_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert screen pixel to world-space ray (origin, direction)."""
    ndc_x = (2.0 * mx / width) - 1.0
    ndc_y = 1.0 - (2.0 * my / height)

    inv_vp = np.linalg.inv(vp)
    near = inv_vp @ np.array([ndc_x, ndc_y, -1, 1], dtype=np.float32)
    far = inv_vp @ np.array([ndc_x, ndc_y, 1, 1], dtype=np.float32)
    near = near[:3] / near[3]
    far = far[:3] / far[3]

    direction = far - near
    direction /= np.linalg.norm(direction) + 1e-8
    return cam_pos.copy(), direction


def project_to_plane(
    ray_origin: np.ndarray, ray_dir: np.ndarray,
    plane_point: np.ndarray, plane_normal: np.ndarray,
) -> np.ndarray:
    """Ray-plane intersection — returns world point."""
    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-8:
        return plane_point  # parallel — return plane point
    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    return ray_origin + ray_dir * t
