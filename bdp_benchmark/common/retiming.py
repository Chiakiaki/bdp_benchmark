"""V3 speed commands and distance-based timing of geometric candidate paths."""

from dataclasses import dataclass

import numpy as np

from .contracts import EgoState
from .frenet import FrenetState, ReferencePath, generate_frenet_trajectories


@dataclass(frozen=True)
class SpeedProfile:
    targets: np.ndarray
    distance: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray


def speed_command_profile(
    initial_speed: float, scales: np.ndarray, *, tau: float, rate: float,
    minimum_speed: float, maximum_speed: float, horizon_s: float, sample_count: int,
) -> SpeedProfile:
    values = [initial_speed, tau, rate, minimum_speed, maximum_speed, horizon_s]
    if not np.isfinite(values).all() or not (
        initial_speed >= 0 and 0 < tau <= horizon_s and rate > 0
        and 0 <= minimum_speed < maximum_speed and sample_count >= 2
    ):
        raise ValueError("Invalid v3 speed profile bounds, tau, rate or sampling")
    scales = np.asarray(scales, dtype=np.float64)
    if scales.ndim != 1 or not scales.size or not np.isfinite(scales).all() or np.any(np.abs(scales) > 1):
        raise ValueError("speed scales must be a finite vector in [-1, 1]")
    targets = np.clip(initial_speed + tau * rate * scales, minimum_speed, maximum_speed)
    times = np.linspace(0.0, horizon_s, sample_count)
    ramp_time = np.minimum(times, tau)
    effective_acceleration = (targets - initial_speed) / tau
    speed = np.where(times >= tau, targets[:, None], initial_speed + effective_acceleration[:, None] * ramp_time)
    distance = (initial_speed * ramp_time + 0.5 * effective_acceleration[:, None] * ramp_time**2
                + targets[:, None] * np.maximum(times - tau, 0.0))
    acceleration = effective_acceleration[:, None] * (times < tau)
    return SpeedProfile(targets, distance, speed, acceleration)


def _extended_reference(reference: ReferencePath, required_length: float) -> ReferencePath:
    """Continue the final reference tangent only when the supplied route ends."""
    if reference.length >= required_length:
        return reference
    heading = reference.segment_heading[-1]
    end = reference.xy[-1] + (required_length - reference.length) * np.array([np.cos(heading), np.sin(heading)])
    return ReferencePath.from_xy(np.vstack((reference.xy, end)), lateral_normal_sign=reference.lateral_normal_sign)


def lane_geometry_required_length(shape_length: float, maximum_distance: float) -> float:
    # Include lane-change geometry and a generous route-following tail for inner curves.
    return 2.0 * shape_length + 4.0 * maximum_distance + 1.0


def retimed_lane_trajectories(
    reference: ReferencePath, origin: EgoState, *, initial_d: float,
    lane_targets: np.ndarray, initial_reference_heading: float, shape_length: float,
    profile: SpeedProfile, geometry_sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return speed-major world [K,H,4] and Frenet [K,H,6] samples.

    The quintic lateral shape uses geometric progress, not physical time.
    Arc-length interpolation of the dense world polyline sets its actual timing.
    Frenet derivatives are exact within each polyline segment; knots are corners.
    """
    if not np.isfinite(shape_length) or shape_length <= 0 or geometry_sample_count < 3:
        raise ValueError("shape_length must be positive and geometry_sample_count >= 3")
    lane_targets = np.asarray(lane_targets, dtype=np.float64)
    relative_heading = origin.heading - initial_reference_heading
    initial = FrenetState(
        0.0, shape_length * np.cos(relative_heading), 0.0, initial_d,
        reference.lateral_normal_sign * shape_length * np.sin(relative_heading), 0.0,
    )
    geometric = generate_frenet_trajectories(
        initial, lane_targets, np.full_like(lane_targets, shape_length),
        horizon_s=1.0, sample_count=geometry_sample_count,
    )
    required = lane_geometry_required_length(shape_length, float(profile.distance.max()))
    reference = _extended_reference(reference, required)
    tail_s = np.linspace(geometric[0, -1, 0], required, geometry_sample_count)[1:]
    tail = np.stack((np.broadcast_to(tail_s, (len(lane_targets), len(tail_s))),
                     np.broadcast_to(lane_targets[:, None], (len(lane_targets), len(tail_s)))), axis=-1)
    sd = np.concatenate((geometric[..., :2], tail), axis=1)
    x, y, _ = reference.sample_center(sd[..., 0])
    # Continuous normals avoid lateral jumps at the centerline's polyline knots.
    h = reference.segment_heading
    node_heading = np.r_[h[0], (h[:-1] + h[1:]) / 2, h[-1]]
    node_heading[0] = h[0] + np.arctan2(np.sin(initial_reference_heading - h[0]),
                                       np.cos(initial_reference_heading - h[0]))
    heading = np.interp(sd[..., 0], reference.cumulative_s, node_heading)
    # The local reference starts at s=0. Preserve backward-facing initial tangents
    # through a local tangent continuation, rather than clamping negative s to zero.
    before_origin = sd[..., 0] < 0
    x = np.where(before_origin, reference.xy[0, 0] + sd[..., 0] * np.cos(initial_reference_heading), x)
    y = np.where(before_origin, reference.xy[0, 1] + sd[..., 0] * np.sin(initial_reference_heading), y)
    heading = np.where(before_origin, initial_reference_heading, heading)
    signed_d = reference.lateral_normal_sign * sd[..., 1]
    xy = np.stack((x - np.sin(heading) * signed_d, y + np.cos(heading) * signed_d), axis=-1)
    xy[:, 0] = origin.xy
    return _retime_polyline(xy, sd, profile, origin.heading)


def _retime_polyline(xy: np.ndarray, sd: np.ndarray, profile: SpeedProfile, initial_heading: float):
    delta = np.diff(xy, axis=1)
    lengths = np.linalg.norm(delta, axis=-1)
    arc = np.concatenate((np.zeros((len(xy), 1)), np.cumsum(lengths, axis=1)), axis=1)
    if np.any(arc[:, -1] < profile.distance.max() - 1e-8):
        raise ValueError("V3 lane geometry is too short for the speed profile")
    query = profile.distance[None, ...]
    # Batched lookup over [lane, speed, time], with no worker/horizon loop.
    index = np.sum(arc[:, None, None, :] <= query[..., None], axis=-1) - 1
    index = np.clip(index, 0, arc.shape[-1] - 2)
    row = np.arange(len(xy))[:, None, None]
    segment_length = np.maximum(lengths[row, index], 1e-12)
    fraction = np.clip((query - arc[row, index]) / segment_length, 0, 1)
    positions = xy[row, index] + fraction[..., None] * delta[row, index]
    tangent = np.arctan2(delta[row, index, 1], delta[row, index, 0])
    tangent = np.where(query <= 1e-12, initial_heading, tangent)
    sd_delta = np.diff(sd, axis=1)[row, index]
    coordinates = sd[row, index] + fraction[..., None] * sd_delta
    slope = sd_delta / segment_length[..., None]
    velocity = slope * profile.speed[None, ..., None]
    acceleration = slope * profile.acceleration[None, ..., None]
    frenet = np.concatenate((coordinates, velocity, acceleration), axis=-1)
    speed = np.broadcast_to(profile.speed, tangent.shape)
    world = np.concatenate((positions, tangent[..., None], speed[..., None]), axis=-1)
    # [lane, speed, H, D] -> speed-major candidate ordering, matching v1/v2.
    return (world.transpose(1, 0, 2, 3).reshape(-1, world.shape[2], 4),
            frenet.transpose(1, 0, 2, 3).reshape(-1, frenet.shape[2], 6))


def retimed_circular_trajectories(origin: EgoState, profile: SpeedProfile, curvatures: np.ndarray) -> np.ndarray:
    curvature = np.asarray(curvatures, dtype=np.float64)
    distance = profile.distance[:, None, :]
    turn = distance * curvature[None, :, None]
    chord = distance * np.sinc(turn / (2 * np.pi))
    direction = origin.heading + turn / 2
    speed = np.broadcast_to(profile.speed[:, None, :], turn.shape)
    world = np.stack((origin.x + chord * np.cos(direction), origin.y + chord * np.sin(direction),
                      origin.heading + turn, speed), axis=-1)
    return world.reshape(-1, world.shape[-2], 4)


def turn_then_straight_trajectories(
    origin: EgoState, profile: SpeedProfile, curvatures: np.ndarray, *, turn_duration_s: float,
) -> np.ndarray:
    """Follow a circular arc until the speed ramp ends, then its terminal tangent."""
    if not np.isfinite(turn_duration_s) or turn_duration_s <= 0:
        raise ValueError("turn_duration_s must be finite and positive")
    # The shared speed law ramps linearly until tau. Compute its integral at tau
    # analytically rather than snapping/interpolating the cutoff to a sample index.
    turn_distance = .5 * (profile.speed[:, 0] + profile.targets) * turn_duration_s
    arc_profile = SpeedProfile(profile.targets, np.minimum(profile.distance, turn_distance[:, None]),
                               profile.speed, profile.acceleration)
    world = retimed_circular_trajectories(origin, arc_profile, curvatures)
    remaining = np.repeat(np.maximum(profile.distance - turn_distance[:, None], 0), len(curvatures), axis=0)
    world[..., :2] += remaining[..., None] * np.stack((np.cos(world[..., 2]), np.sin(world[..., 2])), axis=-1)
    return world
