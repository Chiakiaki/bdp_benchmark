"""Independent timed Frenet geometry with a separate desired-speed command."""

import numpy as np

from .contracts import EgoState
from .frenet import FrenetState, ReferencePath, frenet_to_world, generate_frenet_trajectories
from .retiming import SpeedProfile, _extended_reference


def independent_frenet_motion(
    initial: FrenetState, lane_targets: np.ndarray, profile: SpeedProfile, *, tau: float, horizon_s: float,
) -> np.ndarray:
    """Return speed-major [K,H,(s,d,s_dot,d_dot,s_ddot,d_ddot)] descriptors.

    Only the longitudinal polynomial is replaced. Its initial velocity keeps the
    Frenet projection; the lateral quintic still runs on physical trajectory time.
    Desired scalar speed is supplied separately by profile.speed.
    """
    if not np.isfinite([tau, horizon_s]).all() or not 0 < tau <= horizon_s:
        raise ValueError("Independent Frenet motion requires 0 < tau <= horizon_s")
    lane_targets = np.asarray(lane_targets, dtype=np.float64)
    target_d = np.tile(lane_targets, len(profile.targets))
    targets = np.repeat(profile.targets, len(lane_targets))
    count = profile.speed.shape[-1]
    result = generate_frenet_trajectories(initial, target_d, targets, horizon_s=horizon_s, sample_count=count)
    times = np.linspace(0.0, horizon_s, count)
    ramp_time = np.minimum(times, tau)
    acceleration = (targets - initial.s_dot) / tau
    result[..., 0] = (initial.s + initial.s_dot * ramp_time + .5 * acceleration[:, None] * ramp_time**2
                      + targets[:, None] * np.maximum(times - tau, 0))
    result[..., 2] = np.where(times >= tau, targets[:, None], initial.s_dot + acceleration[:, None] * ramp_time)
    result[..., 4] = acceleration[:, None] * (times < tau)
    return result


def independent_frenet_to_world(
    reference: ReferencePath, origin: EgoState, frenet: np.ndarray, profile: SpeedProfile,
    *, initial_reference_heading: float,
) -> np.ndarray:
    """Convert geometric positions while keeping V(t), not geometric speed, in column 3."""
    reference = _extended_reference(reference, max(float(np.max(frenet[..., 0])), 1e-6))
    lane_count = frenet.shape[0] // len(profile.targets)
    desired_speed = np.repeat(profile.speed, lane_count, axis=0)
    world = frenet_to_world(
        reference, frenet[..., 0], frenet[..., 1], speed=desired_speed,
        longitudinal_speed=frenet[..., 2], lateral_speed=frenet[..., 3],
        initial_heading_rad=np.asarray(origin.heading),
    )
    # Keep signed initial motion behind the local reference origin, as in legacy v3.
    before_origin = frenet[..., 0] < 0
    signed_d = reference.lateral_normal_sign * frenet[..., 1]
    c, s = np.cos(initial_reference_heading), np.sin(initial_reference_heading)
    world[..., 0] = np.where(before_origin, reference.xy[0, 0] + frenet[..., 0] * c - signed_d * s, world[..., 0])
    world[..., 1] = np.where(before_origin, reference.xy[0, 1] + frenet[..., 0] * s + signed_d * c, world[..., 1])
    world[:, 0, :2] = origin.xy

    # Heading follows sampled XY geometry, independent of the desired-speed field.
    motion = np.gradient(world[..., :2], axis=1)
    moving = np.linalg.norm(motion, axis=-1) > 1e-9
    heading = np.arctan2(motion[..., 1], motion[..., 0])
    heading[:, 0] = origin.heading
    index = np.maximum.accumulate(np.where(moving, np.arange(world.shape[1]), 0), axis=1)
    world[..., 2] = np.take_along_axis(heading, index, axis=1)
    return world
