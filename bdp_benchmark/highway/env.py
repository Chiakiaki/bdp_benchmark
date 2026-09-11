"""Sidecar Gymnasium environment for HighwayEnv benchmark modes."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401  Register upstream environments.
from highway_env.vehicle.kinematics import Vehicle

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import CandidateSet
from bdp_benchmark.common.nominal import NominalTrajectoryState
from bdp_benchmark.common.tracking import TrackerConfig, TrajectoryPIDTracker

from .adapter import HighwayAdapter


class HighwayBenchmarkEnv(gym.Wrapper):
    def __init__(
        self,
        *,
        env_id: str,
        execution_mode: str,
        generation_config: CandidateGenerationConfig,
        tracker_config: TrackerConfig,
        env_config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        if execution_mode not in ("native_controller", "frenet_pid", "frenet_pid_v2"):
            raise ValueError(f"Unsupported HighwayEnv execution mode: {execution_mode}")
        kwargs: dict[str, Any] = {"config": dict(env_config or {})}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        super().__init__(gym.make(env_id, **kwargs))
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError(f"Highway benchmark requires a Discrete action space, got {self.action_space}")
        if execution_mode in ("frenet_pid", "frenet_pid_v2") and int(self.action_space.n) != 5:
            raise ValueError("Highway Frenet PID mode requires the five-action DiscreteMetaAction space")
        self.execution_mode = execution_mode
        self.environment_variant = env_id
        self.adapter = HighwayAdapter(self.env, generation_config)
        self.tracker = TrajectoryPIDTracker(tracker_config)
        self._latest_candidates: CandidateSet | None = None
        self._visualizer = None
        self._nominal_state = NominalTrajectoryState() if execution_mode == "frenet_pid_v2" else None

    def build_candidate_set(self) -> CandidateSet:
        planning_state = None
        if self._nominal_state is not None:
            planning_state = self._nominal_state.planning_state(self.adapter.ego_state())
        self._latest_candidates = self.adapter.build_candidate_set(self.execution_mode, planning_state)
        return self._latest_candidates

    def get_visual_candidate_set(self) -> CandidateSet:
        return self._latest_candidates or self.build_candidate_set()

    def preview_pid_target(self, candidate_index: int) -> np.ndarray | None:
        if self.execution_mode not in ("frenet_pid", "frenet_pid_v2"):
            return None
        candidates = self.get_visual_candidate_set()
        self.tracker.set_reference(candidates.trajectories[int(candidate_index)])
        return self.tracker.preview_target(self.adapter.ego_state())

    def set_visual_overlay(self, overlay) -> None:
        self._visual_overlay = overlay
        if self._visualizer is not None:
            self._visualizer.set_overlay(overlay)

    def enable_visualization(self) -> None:
        from .visualizer import HighwayVisualizer

        if self._visualizer is None:
            self._visualizer = HighwayVisualizer(self)

    def reset(self, **kwargs):
        self._latest_candidates = None
        self.tracker.reset()
        if self._nominal_state is not None:
            self._nominal_state.reset()
        result = self.env.reset(**kwargs)
        if self._nominal_state is not None:
            self._nominal_state.planning_state(self.adapter.ego_state())
        if self._visualizer is not None:
            self._visualizer.install()
        return result

    def step(self, action):
        action_idx = int(action)
        if self.execution_mode == "native_controller":
            self._latest_candidates = None
            return self._with_environment_variant(self.env.step(action_idx))
        candidate_set = self._latest_candidates or self.build_candidate_set()
        if action_idx < 0 or action_idx >= candidate_set.trajectories.shape[0]:
            raise ValueError(f"Invalid candidate action {action_idx}")
        if self._nominal_state is not None:
            self._nominal_state.commit(candidate_set.trajectories[action_idx])
        self.tracker.set_reference(candidate_set.trajectories[action_idx])
        self._latest_candidates = None
        return self._with_environment_variant(self._step_frenet_pid(action_idx))

    def _with_environment_variant(self, result):
        obs, reward, terminated, truncated, info = result
        info = dict(info)
        info["environment_variant"] = self.environment_variant
        return obs, reward, terminated, truncated, info

    def _step_frenet_pid(self, action: int):
        raw = self.env.unwrapped
        raw.time += 1.0 / raw.config["policy_frequency"]
        raw.action_type.act(action)
        frames = int(raw.config["simulation_frequency"] // raw.config["policy_frequency"])
        dt = 1.0 / raw.config["simulation_frequency"]
        ego_vehicle = raw.vehicle
        for frame in range(frames):
            for vehicle in raw.road.vehicles:
                if vehicle is not ego_vehicle:
                    vehicle.act()
            control = self.tracker.step(self.adapter.ego_state(), dt=dt)
            Vehicle.act(
                ego_vehicle,
                {"steering": control.steering_rad, "acceleration": control.acceleration_mps2},
            )
            raw.road.step(dt)
            raw.steps += 1
            if frame < frames - 1:
                raw._automatic_rendering()
        raw.enable_auto_render = False
        obs = raw.observation_type.observe()
        reward = float(raw._reward(action))
        terminated = bool(raw._is_terminated())
        truncated = bool(raw._is_truncated())
        info = raw._info(obs, action)
        if raw.render_mode == "human":
            raw.render()
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._visualizer is not None:
            self._visualizer.close()
        return self.env.close()
