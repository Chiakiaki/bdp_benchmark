"""MetaDrive state extraction and action-conditioned Frenet descriptors."""

from __future__ import annotations

import numpy as np

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import CandidateSet, EgoState
from bdp_benchmark.common.features import encode_ego_local_features
from bdp_benchmark.common.frenet import (
    FrenetState, ReferencePath, frenet_to_world, generate_frenet_trajectories,
    generate_constant_curvature_trajectories,
)
from bdp_benchmark.metadrive.route_reference import (
    RoutePosition,
    build_route_reference,
    project_route_position,
    resolve_route_lanes,
)


FRENET_PID_LATERAL_DIM = 3
FRENET_PID_SPEED_DIM = 5
FRENET_PID_ACTION_COUNT = FRENET_PID_LATERAL_DIM * FRENET_PID_SPEED_DIM
FRENET_PID_CURVATURE_ACTION_COUNT = FRENET_PID_ACTION_COUNT + 2 * FRENET_PID_SPEED_DIM
FRENET_REFERENCE_MODES = ("lane_segment", "route_continuous")


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

    def __init__(
        self,
        env,
        config: CandidateGenerationConfig,
        *,
        reference_mode: str = "lane_segment",
        max_steering_rad: float | None = None,
    ) -> None:
        if reference_mode not in FRENET_REFERENCE_MODES:
            raise ValueError(f"reference_mode must be one of {FRENET_REFERENCE_MODES}, got {reference_mode!r}")
        self.env = env
        self.config = config
        self.reference_mode = reference_mode
        self.max_steering_rad = max_steering_rad
        self._route_lane_cache_key = None
        self._route_lane_cache: tuple | None = None

    def reset(self) -> None:
        self._route_lane_cache_key = None
        self._route_lane_cache = None

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

    def _route_lanes(self) -> tuple:
        navigation = self.vehicle.navigation
        current_lane = self.lane
        key = (
            id(navigation.map),
            tuple(navigation.checkpoints),
            tuple(current_lane.index),
        )
        if key != self._route_lane_cache_key:
            self._route_lane_cache = resolve_route_lanes(
                navigation,
                current_lane,
                planning_xy=np.asarray(self.vehicle.position, dtype=np.float64),
            )
            self._route_lane_cache_key = key
        assert self._route_lane_cache is not None
        return self._route_lane_cache

    def _route_state(
        self,
        planning_state: EgoState | None = None,
    ) -> tuple[tuple, RoutePosition, FrenetState, np.ndarray, float]:
        origin = self.ego_state() if planning_state is None else planning_state
        position = np.asarray(origin.xy, dtype=np.float64)
        speed = float(origin.speed)
        route_lanes = self._route_lanes()
        route_position = project_route_position(route_lanes, position)
        heading = float(origin.heading)
        velocity = speed * np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float64)
        lane_forward = np.asarray(
            [np.cos(route_position.start_heading), np.sin(route_position.start_heading)],
            dtype=np.float64,
        )
        standard_left = np.asarray([-lane_forward[1], lane_forward[0]], dtype=np.float64)
        lane_lateral = standard_left * route_position.lateral_normal_sign
        initial = FrenetState(
            s=0.0,
            s_dot=max(float(np.dot(velocity, lane_forward)), 0.0),
            s_ddot=0.0,
            d=route_position.start_lateral,
            d_dot=float(np.dot(velocity, lane_lateral)),
            d_ddot=0.0,
        )
        return route_lanes, route_position, initial, position, speed

    def _targets_from_state(
        self,
        execution_mode: str,
        *,
        state: FrenetState,
        position: np.ndarray,
        speed: float,
        lane,
    ) -> tuple[np.ndarray, np.ndarray]:
        if execution_mode == "native_controller":
            return native_action_targets(
                current_d=float(state.d),
                current_speed=speed,
                steering_dim=int(self.env.config["discrete_steering_dim"]),
                throttle_dim=int(self.env.config["discrete_throttle_dim"]),
                lateral_span_m=self.config.native_lateral_span_m,
                speed_span_mps=self.config.native_speed_span_mps,
                minimum_speed_mps=self.config.minimum_target_speed_mps,
                maximum_speed_mps=self.config.maximum_target_speed_mps,
            )
        return frenet_pid_target_grid(
            lane_centers=self._frenet_pid_lane_centers(position, lane=lane),
            current_speed=speed,
            speed_delta_mps=self.config.speed_delta_mps,
            minimum_speed_mps=self.config.minimum_target_speed_mps,
            maximum_speed_mps=self.config.maximum_target_speed_mps,
        )

    def action_targets(
        self,
        execution_mode: str,
        planning_state: EgoState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if execution_mode in ("frenet_pid_v3", "frenet_pid_v3_legacy"):
            origin = self._v3_origin(planning_state)
            _, position, _, xy, _ = self._route_state(origin)
            centers = self._frenet_pid_lane_centers(xy, lane=position.lane)
            targets = self._v3_speed_profile(origin.speed).targets
            return np.tile(centers, len(targets)), np.repeat(targets, len(centers))
        if self.reference_mode == "route_continuous":
            _route_lanes, route_position, state, position, speed = self._route_state(planning_state)
            return self._targets_from_state(
                execution_mode,
                state=state,
                position=position,
                speed=speed,
                lane=route_position.lane,
            )
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

    def _frenet_pid_lane_centers(self, position: np.ndarray, *, lane=None) -> np.ndarray:
        lane = self.lane if lane is None else lane
        navigation = self.vehicle.navigation
        longitudinal, _ = lane.local_coordinates(position)
        lane_width = float(lane.width_at(longitudinal))

        def center_offset(target_lane) -> float:
            target_longitudinal, _ = target_lane.local_coordinates(position)
            target_center = target_lane.position(target_longitudinal, 0.0)
            return float(lane.local_coordinates(target_center)[1])

        road_network = navigation.map.road_network
        lane_index = lane.index
        peer_lanes = None
        if isinstance(lane_index, tuple) and len(lane_index) >= 3:
            from_node, to_node = lane_index[:2]
            peer_lanes = road_network.graph.get(from_node, {}).get(to_node)
        if not peer_lanes and hasattr(road_network, "get_peer_lanes_from_index"):
            peer_lanes = road_network.get_peer_lanes_from_index(lane_index)
        offsets = [center_offset(peer_lane) for peer_lane in (peer_lanes or [lane])]
        left_offsets = [offset for offset in offsets if offset < -1e-6]
        right_offsets = [offset for offset in offsets if offset > 1e-6]
        left = max(left_offsets) if left_offsets else -lane_width
        right = min(right_offsets) if right_offsets else lane_width
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
        if execution_mode == "frenet_pid_v3":
            return self._build_v3_candidates(planning_state)
        if execution_mode == "frenet_pid_v3_legacy":
            return self._build_v3_legacy_candidates(planning_state)
        if self.config.include_curvature_candidates and execution_mode == "native_controller":
            raise ValueError("curvature candidates require MetaDrive Frenet-PID execution")
        route_lanes = None
        route_position = None
        if self.reference_mode == "route_continuous":
            route_lanes, route_position, initial, position, speed = self._route_state(planning_state)
            target_d, target_speed = self._targets_from_state(
                execution_mode,
                state=initial,
                position=position,
                speed=speed,
                lane=route_position.lane,
            )
        else:
            start_longitudinal, initial, _ = self._lane_state(planning_state)
            target_d, target_speed = self.action_targets(execution_mode, planning_state)
        frenet = generate_frenet_trajectories(
            initial,
            target_d,
            target_speed,
            horizon_s=self.config.horizon_s,
            sample_count=self.config.sample_count,
        )
        required_length = float(np.max(frenet[..., 0])) + 1.0
        if route_lanes is not None and route_position is not None:
            reference = build_route_reference(
                route_lanes,
                planning_xy=position,
                required_length=required_length,
                point_count=max(64, self.config.sample_count * 6),
                route_position=route_position,
            ).reference
        else:
            reference = self._reference_path(start_longitudinal, required_length)
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
        if self.config.include_curvature_candidates:
            # Keep the original 15 labels; append a left/right pair for each speed.
            speeds = np.repeat(target_speed.reshape(FRENET_PID_SPEED_DIM, FRENET_PID_LATERAL_DIM)[:, 0], 2)
            center_curvature = self._center_curvature()
            curves = generate_constant_curvature_trajectories(
                feature_origin, speeds, np.tile([center_curvature, -center_curvature], FRENET_PID_SPEED_DIM),
                horizon_s=self.config.horizon_s, sample_count=self.config.sample_count,
            )
            world = np.concatenate([world, curves], axis=0)
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

    def _center_curvature(self) -> float:
        vehicle_limit = np.deg2rad(float(self.vehicle.max_steering))
        limit = vehicle_limit if self.max_steering_rad is None else min(vehicle_limit, self.max_steering_rad)
        wheelbase = float(self.vehicle.FRONT_WHEELBASE + self.vehicle.REAR_WHEELBASE)
        rear_offset = float(self.vehicle.REAR_WHEELBASE)
        if not np.isfinite([limit, wheelbase, rear_offset]).all() or not (
            0 < limit < np.pi / 2 and wheelbase > 0 and rear_offset >= 0
        ):
            raise ValueError("curvature candidates require valid vehicle geometry and steering limits")
        rear_curvature = np.tan(self.config.curvature_steering_fraction * limit) / wheelbase
        return float(rear_curvature / np.hypot(1.0, rear_offset * rear_curvature))

    def _v3_origin(self, planning_state: EgoState | None) -> EgoState:
        actual = self.ego_state()
        pose = actual if planning_state is None else planning_state
        return EgoState(pose.x, pose.y, pose.heading, actual.speed)

    def _v3_speed_profile(self, actual_speed: float):
        from bdp_benchmark.common.retiming import speed_command_profile

        self.config.validate_v3()
        return speed_command_profile(
            actual_speed, np.asarray(self.config.speed_action_scales), tau=self.config.speed_command_time_s,
            rate=self.config.speed_delta_rate_mps2, minimum_speed=self.config.minimum_target_speed_mps,
            maximum_speed=self.config.maximum_target_speed_mps, horizon_s=self.config.horizon_s,
            sample_count=self.config.sample_count,
        )

    def _build_v3_legacy_candidates(self, planning_state: EgoState | None) -> CandidateSet:
        from bdp_benchmark.common.retiming import (
            lane_geometry_required_length, retimed_lane_trajectories, retimed_circular_trajectories,
        )

        origin = self._v3_origin(planning_state)
        profile = self._v3_speed_profile(origin.speed)
        lanes, position, _, xy, _ = self._route_state(origin)
        centers = self._frenet_pid_lane_centers(xy, lane=position.lane)
        width = float(position.lane.width_at(position.start_longitudinal))
        # A positive spatial lane-change shape is needed even when accelerating from rest.
        shape_length = max(origin.speed * self.config.horizon_s, 2 * width)
        required = lane_geometry_required_length(shape_length, float(profile.distance.max()))
        point_count = max(128, self.config.sample_count * 12)
        reference = build_route_reference(
            lanes, planning_xy=xy, required_length=required, point_count=point_count, route_position=position,
        ).reference
        world, _frenet = retimed_lane_trajectories(
            reference, origin, initial_d=position.start_lateral, lane_targets=centers,
            initial_reference_heading=position.start_heading, shape_length=shape_length,
            profile=profile, geometry_sample_count=point_count,
        )
        targets = np.repeat(profile.targets, len(centers))
        if self.config.include_curvature_candidates:
            curvature = self._center_curvature()
            curves = retimed_circular_trajectories(origin, profile, np.array([curvature, -curvature]))
            world = np.concatenate((world, curves))
            targets = np.concatenate((targets, np.repeat(profile.targets, 2)))
        features = encode_ego_local_features(
            world, ego_xy=origin.xy, ego_heading=origin.heading, position_scale_m=self.config.position_scale_m,
            speed_scale_mps=self.config.speed_scale_mps, lateral_normal_sign=reference.lateral_normal_sign,
        )
        count = int(self.env.action_space.n)
        return CandidateSet(np.arange(count), np.ones(count, dtype=np.float32), world.astype(np.float32),
                            features, speed_targets=targets)

    def _build_v3_candidates(self, planning_state: EgoState | None) -> CandidateSet:
        from dataclasses import replace
        from bdp_benchmark.common.independent_frenet import independent_frenet_motion, independent_frenet_to_world
        from bdp_benchmark.common.retiming import turn_then_straight_trajectories

        origin = self._v3_origin(planning_state)
        profile = self._v3_speed_profile(origin.speed)
        lanes, position, initial, xy, _ = self._route_state(origin)
        # Preserve the initial nominal direction, including backward-facing poses.
        initial = replace(initial, s_dot=origin.speed * np.cos(origin.heading - position.start_heading))
        centers = self._frenet_pid_lane_centers(xy, lane=position.lane)
        frenet = independent_frenet_motion(
            initial, centers, profile, tau=self.config.speed_command_time_s, horizon_s=self.config.horizon_s,
        )
        reference = build_route_reference(
            lanes, planning_xy=xy, required_length=max(float(np.max(frenet[..., 0])) + 1, 1),
            point_count=max(64, self.config.sample_count * 6), route_position=position,
        ).reference
        world = independent_frenet_to_world(
            reference, origin, frenet, profile, initial_reference_heading=position.start_heading,
        )
        if self.config.include_curvature_candidates:
            curvature = self._center_curvature()
            curves = turn_then_straight_trajectories(
                origin, profile, np.array([curvature, -curvature]), turn_duration_s=self.config.speed_command_time_s,
            )
            world = np.concatenate((world, curves))
        features = encode_ego_local_features(
            world, ego_xy=origin.xy, ego_heading=origin.heading, position_scale_m=self.config.position_scale_m,
            speed_scale_mps=self.config.speed_scale_mps, lateral_normal_sign=reference.lateral_normal_sign,
        )
        count = int(self.env.action_space.n)
        # No direct target metadata: the PID must preview the desired-speed column.
        return CandidateSet(np.arange(count), np.ones(count, dtype=np.float32), world.astype(np.float32), features)
