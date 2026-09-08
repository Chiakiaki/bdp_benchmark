"""Sidecar Gymnasium environment for HighwayEnv benchmark modes."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

import highway_env  # noqa: F401  Register upstream environments.
from highway_env.vehicle.kinematics import Vehicle

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import CandidateSet
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
        if execution_mode not in ("native_controller", "frenet_pid"):
            raise ValueError(f"Unsupported HighwayEnv execution mode: {execution_mode}")
        kwargs: dict[str, Any] = {"config": dict(env_config or {})}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        super().__init__(gym.make(env_id, **kwargs))
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError(f"Highway benchmark requires a Discrete action space, got {self.action_space}")
        if execution_mode == "frenet_pid" and int(self.action_space.n) != 5:
            raise ValueError("Highway Frenet PID mode requires the five-action DiscreteMetaAction space")
        self.execution_mode = execution_mode
        self.adapter = HighwayAdapter(self.env, generation_config)
        self.tracker = TrajectoryPIDTracker(tracker_config)
        self._latest_candidates: CandidateSet | None = None

    def build_candidate_set(self) -> CandidateSet:
        self._latest_candidates = self.adapter.build_candidate_set(self.execution_mode)
        return self._latest_candidates

    def reset(self, **kwargs):
        self._latest_candidates = None
        self.tracker.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        action_idx = int(action)
        if self.execution_mode == "native_controller":
            self._latest_candidates = None
            return self.env.step(action_idx)
        candidate_set = self._latest_candidates or self.build_candidate_set()
        if action_idx < 0 or action_idx >= candidate_set.trajectories.shape[0]:
            raise ValueError(f"Invalid candidate action {action_idx}")
        self.tracker.set_reference(candidate_set.trajectories[action_idx])
        self._latest_candidates = None
        return self._step_frenet_pid(action_idx)

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
