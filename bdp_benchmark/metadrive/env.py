"""Sidecar Gymnasium environment for MetaDrive benchmark modes."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import CandidateSet
from bdp_benchmark.common.tracking import TrackerConfig

from .adapter import MetaDriveAdapter
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
        if execution_mode not in ("native_controller", "frenet_pid"):
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
            else 5
        )
        if not isinstance(self.action_space, gym.spaces.Discrete) or int(self.action_space.n) != expected_actions:
            raise RuntimeError(f"MetaDrive action space does not match expected Discrete({expected_actions}): {self.action_space}")
        self.execution_mode = execution_mode
        self.adapter = MetaDriveAdapter(self.env, generation_config)
        self.tracker_config = tracker_config
        self._latest_candidates: CandidateSet | None = None

    def _pid_policy(self) -> MetaDriveFrenetPIDPolicy:
        policy = self.env.engine.get_policy(self.env.agent.name)
        if not isinstance(policy, MetaDriveFrenetPIDPolicy):
            raise RuntimeError(f"Expected MetaDriveFrenetPIDPolicy, got {type(policy).__name__}")
        return policy

    def reset(self, **kwargs):
        self._latest_candidates = None
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
        return result

    def build_candidate_set(self) -> CandidateSet:
        self._latest_candidates = self.adapter.build_candidate_set(self.execution_mode)
        return self._latest_candidates

    def step(self, action):
        action_idx = int(action)
        if self.execution_mode == "frenet_pid":
            candidate_set = self._latest_candidates or self.build_candidate_set()
            if action_idx < 0 or action_idx >= candidate_set.trajectories.shape[0]:
                raise ValueError(f"Invalid candidate action {action_idx}")
            self._pid_policy().set_reference(candidate_set.trajectories[action_idx])
        self._latest_candidates = None
        return self.env.step(action_idx)
