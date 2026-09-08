"""Candidate trajectory feature encoding."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def encode_ego_local_features(
    world_trajectories: np.ndarray,
    *,
    ego_xy: np.ndarray,
    ego_heading: float | np.ndarray,
    position_scale_m: float,
    speed_scale_mps: float,
    lateral_normal_sign: float = 1.0,
) -> np.ndarray:
    """Encode `[x,y,yaw,speed]` samples as flattened ego-forward/left features."""
    trajectories = np.asarray(world_trajectories, dtype=np.float64)
    if trajectories.ndim < 3 or trajectories.shape[-1] != 4:
        raise ValueError(f"world_trajectories must have shape [..., K, H, 4], got {trajectories.shape}")
    if position_scale_m <= 0.0 or speed_scale_mps <= 0.0:
        raise ValueError("feature scales must be positive")

    ego_xy = np.asarray(ego_xy, dtype=np.float64)
    ego_heading = np.asarray(ego_heading, dtype=np.float64)
    batch_shape = trajectories.shape[:-3]
    ego_xy = np.broadcast_to(ego_xy, (*batch_shape, 2))
    ego_heading = np.broadcast_to(ego_heading, batch_shape)
    delta = trajectories[..., :2] - ego_xy[..., None, None, :]
    c = np.cos(ego_heading)[..., None, None]
    s = np.sin(ego_heading)[..., None, None]
    forward = c * delta[..., 0] + s * delta[..., 1]
    lateral = float(lateral_normal_sign) * (-s * delta[..., 0] + c * delta[..., 1])
    relative_heading = wrap_angle(trajectories[..., 2] - ego_heading[..., None, None])
    speed = trajectories[..., 3]
    point_features = np.stack(
        [
            np.clip(forward / position_scale_m, -1.0, 1.0),
            np.clip(lateral / position_scale_m, -1.0, 1.0),
            np.sin(relative_heading),
            np.cos(relative_heading),
            np.clip(speed / speed_scale_mps, -1.0, 1.0),
        ],
        axis=-1,
    )
    return point_features.reshape(*point_features.shape[:-2], -1).astype(np.float32)
