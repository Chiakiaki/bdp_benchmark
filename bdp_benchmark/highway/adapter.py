"""HighwayEnv state extraction and action-conditioned Frenet descriptors."""

from __future__ import annotations

import numpy as np

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import CandidateSet, EgoState
from bdp_benchmark.common.features import encode_ego_local_features
from bdp_benchmark.common.frenet import FrenetState, ReferencePath, frenet_to_world, generate_frenet_trajectories


class HighwayAdapter:
    def __init__(self, env, config: CandidateGenerationConfig) -> None:
        self.env = env
        self.config = config

    @property
    def raw(self):
        return self.env.unwrapped

    def ego_state(self) -> EgoState:
        vehicle = self.raw.vehicle
        return EgoState(
            x=float(vehicle.position[0]),
            y=float(vehicle.position[1]),
            heading=float(vehicle.heading),
            speed=float(vehicle.speed),
        )

    def _base_lane_state(self):
        vehicle = self.raw.vehicle
        lane_index = vehicle.lane_index
        lane = self.raw.road.network.get_lane(lane_index)
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        lane_heading = float(lane.heading_at(longitudinal))
        heading_error = float(vehicle.heading - lane_heading)
        state = FrenetState(
            s=0.0,
            s_dot=max(float(vehicle.speed * np.cos(heading_error)), 0.0),
            s_ddot=0.0,
            d=float(lateral),
            d_dot=float(vehicle.speed * np.sin(heading_error)),
            d_ddot=0.0,
        )
        return lane_index, lane, float(longitudinal), state

    def _shift_target_lane(self, target_lane_index, direction: int):
        from_node, to_node, lane_id = target_lane_index
        lane_count = len(self.raw.road.network.graph[from_node][to_node])
        shifted_id = int(np.clip(int(lane_id) + direction, 0, lane_count - 1))
        shifted = (from_node, to_node, shifted_id)
        if self.raw.road.network.get_lane(shifted).is_reachable_from(self.raw.vehicle.position):
            return shifted
        return target_lane_index

    def _target_lateral_in_base_lane(self, target_lane_index, base_lane, base_longitudinal: float) -> float:
        target_lane = self.raw.road.network.get_lane(target_lane_index)
        target_longitudinal, _ = target_lane.local_coordinates(self.raw.vehicle.position)
        target_center = target_lane.position(target_longitudinal, 0.0)
        _, target_lateral = base_lane.local_coordinates(target_center)
        return float(target_lateral)

    def action_targets(self, execution_mode: str) -> tuple[np.ndarray, np.ndarray]:
        del execution_mode
        vehicle = self.raw.vehicle
        _, base_lane, base_longitudinal, _ = self._base_lane_state()
        actions = self.raw.action_type.actions
        target_lane_index = getattr(vehicle, "target_lane_index", vehicle.lane_index)
        retained_speed = float(getattr(vehicle, "target_speed", vehicle.speed))
        target_d: list[float] = []
        target_speed: list[float] = []
        for action_idx in range(len(actions)):
            action_name = actions[action_idx]
            lane_index = target_lane_index
            speed = retained_speed
            if action_name == "LANE_LEFT":
                lane_index = self._shift_target_lane(target_lane_index, -1)
            elif action_name == "LANE_RIGHT":
                lane_index = self._shift_target_lane(target_lane_index, 1)
            elif action_name in ("FASTER", "SLOWER"):
                direction = 1 if action_name == "FASTER" else -1
                speed_index = int(vehicle.speed_to_index(vehicle.speed)) + direction
                speed_index = int(np.clip(speed_index, 0, len(vehicle.target_speeds) - 1))
                speed = float(vehicle.index_to_speed(speed_index))
            target_d.append(self._target_lateral_in_base_lane(lane_index, base_lane, base_longitudinal))
            target_speed.append(speed)
        return np.asarray(target_d, dtype=np.float64), np.clip(
            np.asarray(target_speed, dtype=np.float64),
            self.config.minimum_target_speed_mps,
            self.config.maximum_target_speed_mps,
        )

    def _reference_path(self, required_length: float) -> ReferencePath:
        vehicle = self.raw.vehicle
        lane_index, _, longitudinal, _ = self._base_lane_state()
        route = list(vehicle.route or [lane_index])
        if not route or route[0][:2] != lane_index[:2]:
            route.insert(0, lane_index)
        else:
            route[0] = lane_index
        sample_count = max(64, self.config.sample_count * 6)
        offsets = np.linspace(0.0, max(float(required_length), 1.0), sample_count)
        points = [
            self.raw.road.network.position_heading_along_route(route, longitudinal + offset, 0.0, lane_index)[0]
            for offset in offsets
        ]
        return ReferencePath.from_xy(np.asarray(points, dtype=np.float64))

    def build_candidate_set(self, execution_mode: str) -> CandidateSet:
        _, _, _, initial = self._base_lane_state()
        target_d, target_speed = self.action_targets(execution_mode)
        frenet = generate_frenet_trajectories(
            initial,
            target_d,
            target_speed,
            horizon_s=self.config.horizon_s,
            sample_count=self.config.sample_count,
        )
        reference = self._reference_path(float(np.max(frenet[..., 0])) + 1.0)
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
        action_count = int(self.raw.action_space.n)
        return CandidateSet(
            labels=np.arange(action_count, dtype=np.int64),
            mask=np.ones(action_count, dtype=np.float32),
            trajectories=world.astype(np.float32),
            features=features,
        )
