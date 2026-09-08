"""MetaDrive state extraction and action-conditioned Frenet descriptors."""

from __future__ import annotations

import numpy as np

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import CandidateSet, EgoState
from bdp_benchmark.common.features import encode_ego_local_features
from bdp_benchmark.common.frenet import FrenetState, ReferencePath, frenet_to_world, generate_frenet_trajectories


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


class MetaDriveAdapter:
    SEMANTIC_ACTION_COUNT = 5

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

    def _lane_state(self) -> tuple[float, FrenetState]:
        lane = self.lane
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
        lane_heading = float(lane.heading_theta_at(longitudinal))
        velocity = np.asarray(self.vehicle.velocity, dtype=np.float64)
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
        )

    def action_targets(self, execution_mode: str) -> tuple[np.ndarray, np.ndarray]:
        _, state = self._lane_state()
        if execution_mode == "native_controller":
            return native_action_targets(
                current_d=float(state.d),
                current_speed=float(self.vehicle.speed),
                steering_dim=int(self.env.config["discrete_steering_dim"]),
                throttle_dim=int(self.env.config["discrete_throttle_dim"]),
                lateral_span_m=self.config.native_lateral_span_m,
                speed_span_mps=self.config.native_speed_span_mps,
                minimum_speed_mps=self.config.minimum_target_speed_mps,
                maximum_speed_mps=self.config.maximum_target_speed_mps,
            )
        lane_width = float(self.lane.width_at(self.lane.local_coordinates(self.vehicle.position)[0]))
        lane_delta = lane_width * self.config.lane_change_width_scale
        speed = float(self.vehicle.speed)
        target_d = np.asarray(
            [float(state.d) - lane_delta, float(state.d), float(state.d) + lane_delta, float(state.d), float(state.d)],
            dtype=np.float64,
        )
        target_speed = np.asarray(
            [speed, speed, speed, speed + self.config.speed_delta_mps, speed - self.config.speed_delta_mps],
            dtype=np.float64,
        )
        return target_d, np.clip(
            target_speed,
            self.config.minimum_target_speed_mps,
            self.config.maximum_target_speed_mps,
        )

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

    def build_candidate_set(self, execution_mode: str) -> CandidateSet:
        start_longitudinal, initial = self._lane_state()
        target_d, target_speed = self.action_targets(execution_mode)
        frenet = generate_frenet_trajectories(
            initial,
            target_d,
            target_speed,
            horizon_s=self.config.horizon_s,
            sample_count=self.config.sample_count,
        )
        reference = self._reference_path(start_longitudinal, float(np.max(frenet[..., 0])) + 1.0)
        world = frenet_to_world(
            reference,
            frenet[..., 0],
            frenet[..., 1],
            speed=np.hypot(frenet[..., 2], frenet[..., 3]),
            longitudinal_speed=frenet[..., 2],
            lateral_speed=frenet[..., 3],
        )
        ego = self.ego_state()
        features = encode_ego_local_features(
            world,
            ego_xy=ego.xy,
            ego_heading=ego.heading,
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
