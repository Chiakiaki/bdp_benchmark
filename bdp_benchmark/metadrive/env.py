"""Sidecar Gymnasium environment for MetaDrive benchmark modes."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import CandidateSet
from bdp_benchmark.common.nominal import NominalTrajectoryState
from bdp_benchmark.common.tracking import TrackerConfig

from .adapter import MetaDriveAdapter
from .adapter import FRENET_PID_ACTION_COUNT
from .policy import MetaDriveFrenetPIDPolicy


class MetaDriveBenchmarkEnv(gym.Wrapper):
    def __init__(
        self,
        *,
        env_id: str = "MetaDrive-v0",
        execution_mode: str,
        generation_config: CandidateGenerationConfig,
        tracker_config: TrackerConfig,
        env_config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        if execution_mode not in ("native_controller", "frenet_pid", "frenet_pid_v2"):
            raise ValueError(f"Unsupported MetaDrive execution mode: {execution_mode}")
        config = dict(env_config or {})
        config["use_render"] = render_mode == "human" or bool(config.get("use_render", False))
        if execution_mode == "native_controller":
            config["discrete_action"] = True
            config["use_multi_discrete"] = False
        else:
            config["agent_policy"] = MetaDriveFrenetPIDPolicy
            config["discrete_action"] = False
        env_classes = {
            "MetaDrive-v0": MetaDriveEnv,
            "metadrive-v0": MetaDriveEnv,
            "ScenarioEnv-v0": ScenarioEnv,
            "scenario-v0": ScenarioEnv,
        }
        if env_id not in env_classes:
            raise ValueError(f"Unsupported MetaDrive env_id: {env_id}. Available: {sorted(env_classes)}")
        super().__init__(env_classes[env_id](config))
        expected_actions = (
            int(self.env.config["discrete_steering_dim"]) * int(self.env.config["discrete_throttle_dim"])
            if execution_mode == "native_controller"
            else FRENET_PID_ACTION_COUNT
        )
        if not isinstance(self.action_space, gym.spaces.Discrete) or int(self.action_space.n) != expected_actions:
            raise RuntimeError(f"MetaDrive action space does not match expected Discrete({expected_actions}): {self.action_space}")
        self.execution_mode = execution_mode
        self.adapter = MetaDriveAdapter(self.env, generation_config)
        self.tracker_config = tracker_config
        self._latest_candidates: CandidateSet | None = None
        self._visualizer = None
        self._nominal_state = NominalTrajectoryState() if execution_mode == "frenet_pid_v2" else None

    def _pid_policy(self) -> MetaDriveFrenetPIDPolicy:
        policy = self.env.engine.get_policy(self.env.agent.name)
        if not isinstance(policy, MetaDriveFrenetPIDPolicy):
            raise RuntimeError(f"Expected MetaDriveFrenetPIDPolicy, got {type(policy).__name__}")
        return policy

    def reset(self, **kwargs):
        self._latest_candidates = None
        if self._nominal_state is not None:
            self._nominal_state.reset()
        if self._visualizer is not None:
            self._visualizer.clear()
        options = kwargs.pop("options", None)
        if options:
            raise ValueError("This MetaDrive version does not support non-empty Gymnasium reset options")
        if kwargs.get("seed") is not None:
            start = int(self.env.start_index)
            count = int(self.env.num_scenarios)
            kwargs["seed"] = start + (int(kwargs["seed"]) - start) % count
        result = self.env.reset(**kwargs)
        if self.execution_mode == "frenet_pid":
            self._pid_policy().configure_tracker(self.tracker_config)
        elif self.execution_mode == "frenet_pid_v2":
            self._pid_policy().configure_tracker(self.tracker_config)
        if self._nominal_state is not None:
            self._nominal_state.planning_state(self.adapter.ego_state())
        if self._visualizer is not None:
            self._visualizer.install()
        return result

    def build_candidate_set(self) -> CandidateSet:
        planning_state = None
        if self._nominal_state is not None:
            planning_state = self._nominal_state.planning_state(self.adapter.ego_state())
        self._latest_candidates = self.adapter.build_candidate_set(self.execution_mode, planning_state)
        return self._latest_candidates

    def get_visual_candidate_set(self) -> CandidateSet:
        return self._latest_candidates or self.build_candidate_set()

    def preview_pid_target(self, candidate_index: int):
        if self.execution_mode not in ("frenet_pid", "frenet_pid_v2"):
            return None
        candidates = self.get_visual_candidate_set()
        policy = self._pid_policy()
        policy.set_reference(candidates.trajectories[int(candidate_index)])
        return policy.tracker.preview_target(self.adapter.ego_state())

    def set_visual_overlay(self, overlay) -> None:
        self._visual_overlay = overlay
        if self._visualizer is not None:
            self._visualizer.set_overlay(overlay)

    def enable_visualization(self) -> None:
        from .visualizer import MetaDriveVisualizer

        if self._visualizer is None:
            self._visualizer = MetaDriveVisualizer(self)

    def step(self, action):
        action_idx = int(action)
        if self.execution_mode in ("frenet_pid", "frenet_pid_v2"):
            candidate_set = self._latest_candidates or self.build_candidate_set()
            if action_idx < 0 or action_idx >= candidate_set.trajectories.shape[0]:
                raise ValueError(f"Invalid candidate action {action_idx}")
            if self._nominal_state is not None:
                self._nominal_state.commit(candidate_set.trajectories[action_idx])
            self._pid_policy().set_reference(candidate_set.trajectories[action_idx])
        self._latest_candidates = None
        return self.env.step(action_idx)

    def close(self):
        if self._visualizer is not None:
            self._visualizer.close()
        return self.env.close()
