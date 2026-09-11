"""MetaDrive state extraction and action-conditioned Frenet descriptors."""

from __future__ import annotations

import numpy as np

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import CandidateSet, EgoState
from bdp_benchmark.common.features import encode_ego_local_features
from bdp_benchmark.common.frenet import FrenetState, ReferencePath, frenet_to_world, generate_frenet_trajectories


FRENET_PID_LATERAL_DIM = 3
FRENET_PID_SPEED_DIM = 5
FRENET_PID_ACTION_COUNT = FRENET_PID_LATERAL_DIM * FRENET_PID_SPEED_DIM


def decode_discrete_action_grid(*, steering_dim: int, throttle_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if steering_dim < 2 or throttle_dim < 2:
        raise ValueError("MetaDrive discrete steering and throttle dimensions must both be at least 2")
    action = np.arange(steering_dim * throttle_dim, dtype=np.int64)
    steering = (action % steering_dim) * (2.0 / (steering_dim - 1)) - 1.0
    throttle = (action // steering_dim) * (2.0 / (throttle_dim - 1)) - 1.0
    return steering.astype(np.float64), throttle.astype(np.float64)


def native_action_targets(
    *,
    current_d: float,
    current_speed: float,
    steering_dim: int,
    throttle_dim: int,
    lateral_span_m: float,
    speed_span_mps: float,
    minimum_speed_mps: float,
    maximum_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    steering, throttle = decode_discrete_action_grid(steering_dim=steering_dim, throttle_dim=throttle_dim)
    target_d = float(current_d) - steering * float(lateral_span_m)
    target_speed = np.clip(
        float(current_speed) + throttle * float(speed_span_mps),
        float(minimum_speed_mps),
        float(maximum_speed_mps),
    )
    return target_d, target_speed


def frenet_pid_target_grid(
    *,
    lane_centers: np.ndarray,
    current_speed: float,
    speed_delta_mps: float,
    minimum_speed_mps: float,
    maximum_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    lane_centers = np.asarray(lane_centers, dtype=np.float64)
    if lane_centers.shape != (FRENET_PID_LATERAL_DIM,):
        raise ValueError(f"lane_centers must have shape ({FRENET_PID_LATERAL_DIM},), got {lane_centers.shape}")
    speed_levels = np.linspace(-1.0, 1.0, FRENET_PID_SPEED_DIM, dtype=np.float64)
    speed_targets = np.clip(
        float(current_speed) + speed_levels * float(speed_delta_mps),
        float(minimum_speed_mps),
        float(maximum_speed_mps),
    )
    return np.tile(lane_centers, FRENET_PID_SPEED_DIM), np.repeat(speed_targets, FRENET_PID_LATERAL_DIM)


class MetaDriveAdapter:
    SEMANTIC_ACTION_COUNT = FRENET_PID_ACTION_COUNT

    def __init__(self, env, config: CandidateGenerationConfig) -> None:
        self.env = env
        self.config = config

    @property
    def vehicle(self):
        return self.env.agent

    @property
    def lane(self):
        lane = self.vehicle.navigation.current_lane
        if lane is None:
            raise RuntimeError("MetaDrive navigation has no current lane")
        return lane

    def ego_state(self) -> EgoState:
        return EgoState(
            x=float(self.vehicle.position[0]),
            y=float(self.vehicle.position[1]),
            heading=float(self.vehicle.heading_theta),
            speed=float(self.vehicle.speed),
        )

    def _lane_state(self, planning_state: EgoState | None = None) -> tuple[float, FrenetState, np.ndarray]:
        lane = self.lane
        position = self.vehicle.position if planning_state is None else planning_state.xy
        heading = float(self.vehicle.heading_theta) if planning_state is None else float(planning_state.heading)
        speed = float(self.vehicle.speed) if planning_state is None else float(planning_state.speed)
        longitudinal, lateral = lane.local_coordinates(position)
        lane_heading = float(lane.heading_theta_at(longitudinal))
        # MetaDrive's velocity can differ from chassis orientation while the
        # vehicle is steering. Actual-state descriptors should follow the direct
        # heading API; nominal states already carry their planned heading.
        velocity = speed * np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float64)
        lane_forward = np.asarray([np.cos(lane_heading), np.sin(lane_heading)])
        lateral_probe = np.asarray(lane.position(longitudinal, 1.0), dtype=np.float64) - np.asarray(
            lane.position(longitudinal, 0.0), dtype=np.float64
        )
        lateral_probe /= max(float(np.linalg.norm(lateral_probe)), 1e-9)
        return float(longitudinal), FrenetState(
            s=0.0,
            s_dot=max(float(np.dot(velocity, lane_forward)), 0.0),
            s_ddot=0.0,
            d=float(lateral),
            d_dot=float(np.dot(velocity, lateral_probe)),
            d_ddot=0.0,
        ), np.asarray(position, dtype=np.float64)

    def action_targets(
        self,
        execution_mode: str,
        planning_state: EgoState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, state, position = self._lane_state(planning_state)
        if execution_mode == "native_controller":
            return native_action_targets(
                current_d=float(state.d),
                current_speed=float(self.vehicle.speed) if planning_state is None else float(planning_state.speed),
                steering_dim=int(self.env.config["discrete_steering_dim"]),
                throttle_dim=int(self.env.config["discrete_throttle_dim"]),
                lateral_span_m=self.config.native_lateral_span_m,
                speed_span_mps=self.config.native_speed_span_mps,
                minimum_speed_mps=self.config.minimum_target_speed_mps,
                maximum_speed_mps=self.config.maximum_target_speed_mps,
            )
        speed = float(self.vehicle.speed) if planning_state is None else float(planning_state.speed)
        return frenet_pid_target_grid(
            lane_centers=self._frenet_pid_lane_centers(position),
            current_speed=speed,
            speed_delta_mps=self.config.speed_delta_mps,
            minimum_speed_mps=self.config.minimum_target_speed_mps,
            maximum_speed_mps=self.config.maximum_target_speed_mps,
        )

    def _frenet_pid_lane_centers(self, position: np.ndarray) -> np.ndarray:
        lane = self.lane
        navigation = self.vehicle.navigation
        reference_lanes = list(navigation.current_ref_lanes or [lane])
        try:
            current_index = reference_lanes.index(lane)
        except ValueError as exc:
            raise RuntimeError("MetaDrive current lane is not present in navigation.current_ref_lanes") from exc

        longitudinal, _ = lane.local_coordinates(position)
        lane_width = float(lane.width_at(longitudinal))

        def center_offset(target_lane) -> float:
            target_longitudinal, _ = target_lane.local_coordinates(position)
            target_center = target_lane.position(target_longitudinal, 0.0)
            return float(lane.local_coordinates(target_center)[1])

        left = center_offset(reference_lanes[current_index - 1]) if current_index > 0 else -lane_width
        right = (
            center_offset(reference_lanes[current_index + 1])
            if current_index + 1 < len(reference_lanes)
            else lane_width
        )
        return np.asarray([left, 0.0, right], dtype=np.float64)

    def _reference_path(self, start_longitudinal: float, required_length: float) -> ReferencePath:
        lane = self.lane
        sample_count = max(64, self.config.sample_count * 6)
        longitudinal = start_longitudinal + np.linspace(0.0, max(required_length, 1.0), sample_count)
        points = np.asarray([lane.position(float(s), 0.0) for s in longitudinal], dtype=np.float64)
        lane_heading = float(lane.heading_theta_at(start_longitudinal))
        standard_left = np.asarray([-np.sin(lane_heading), np.cos(lane_heading)])
        lane_lateral = np.asarray(lane.position(start_longitudinal, 1.0), dtype=np.float64) - np.asarray(
            lane.position(start_longitudinal, 0.0), dtype=np.float64
        )
        normal_sign = 1.0 if float(np.dot(standard_left, lane_lateral)) >= 0.0 else -1.0
        return ReferencePath.from_xy(points, lateral_normal_sign=normal_sign)

    def build_candidate_set(
        self,
        execution_mode: str,
        planning_state: EgoState | None = None,
    ) -> CandidateSet:
        start_longitudinal, initial, _ = self._lane_state(planning_state)
        target_d, target_speed = self.action_targets(execution_mode, planning_state)
        frenet = generate_frenet_trajectories(
            initial,
            target_d,
            target_speed,
            horizon_s=self.config.horizon_s,
            sample_count=self.config.sample_count,
        )
        reference = self._reference_path(start_longitudinal, float(np.max(frenet[..., 0])) + 1.0)
        feature_origin = self.ego_state() if planning_state is None else planning_state
        world = frenet_to_world(
            reference,
            frenet[..., 0],
            frenet[..., 1],
            speed=np.hypot(frenet[..., 2], frenet[..., 3]),
            longitudinal_speed=frenet[..., 2],
            lateral_speed=frenet[..., 3],
            initial_heading_rad=np.asarray(feature_origin.heading),
        )
        features = encode_ego_local_features(
            world,
            ego_xy=feature_origin.xy,
            ego_heading=feature_origin.heading,
            position_scale_m=self.config.position_scale_m,
            speed_scale_mps=self.config.speed_scale_mps,
            lateral_normal_sign=reference.lateral_normal_sign,
        )
        action_count = int(self.env.action_space.n)
        return CandidateSet(
            labels=np.arange(action_count, dtype=np.int64),
            mask=np.ones(action_count, dtype=np.float32),
            trajectories=world.astype(np.float32),
            features=features,
        )
