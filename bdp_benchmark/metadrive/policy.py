"""MetaDrive policy that tracks a selected benchmark Frenet trajectory."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from metadrive.policy.base_policy import BasePolicy

from bdp_benchmark.common.contracts import EgoState
from bdp_benchmark.common.tracking import TrackerConfig, TrajectoryPIDTracker
from bdp_benchmark.metadrive.adapter import FRENET_PID_ACTION_COUNT


class MetaDriveFrenetPIDPolicy(BasePolicy):
    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed, config)
        self.tracker = TrajectoryPIDTracker(TrackerConfig())

    @classmethod
    def get_input_space(cls):
        return spaces.Discrete(FRENET_PID_ACTION_COUNT)

    def configure_tracker(self, config: TrackerConfig) -> None:
        self.tracker = TrajectoryPIDTracker(config)

    def set_reference(self, trajectory: np.ndarray) -> None:
        self.tracker.set_reference(trajectory)

    def reset(self):
        super().reset()
        if hasattr(self, "tracker"):
            self.tracker.reset()

    def act(self, agent_id):
        del agent_id
        if self.tracker.reference is None:
            return [0.0, 0.0]
        vehicle = self.control_object
        ego = EgoState(
            x=float(vehicle.position[0]),
            y=float(vehicle.position[1]),
            heading=float(vehicle.heading_theta),
            speed=float(vehicle.speed),
        )
        dt = float(self.engine.global_config["physics_world_step_size"])
        control = self.tracker.step(ego, dt=dt)
        max_steering_degrees = max(float(vehicle.max_steering), 1e-6)
        steering = np.clip(np.rad2deg(control.steering_rad) / max_steering_degrees, -1.0, 1.0)
        if control.acceleration_mps2 >= 0.0:
            throttle = control.acceleration_mps2 / self.tracker.config.max_accel_mps2
        else:
            throttle = control.acceleration_mps2 / self.tracker.config.max_decel_mps2
        action = [float(steering), float(np.clip(throttle, -1.0, 1.0))]
        self.action_info["action"] = action
        return action
