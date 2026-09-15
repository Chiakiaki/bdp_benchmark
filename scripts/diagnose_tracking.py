#!/usr/bin/env python3
"""Replay a frozen selector on fixed seeds and write tracking CSV/JSON diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT.parent):
    sys.path.insert(0, str(path))

import numpy as np
from stable_baselines3 import PPO
from critic_based_rl.inference import predict_discrete_with_scores
from bdp_benchmark.config import parse_args, resolved_config
from bdp_benchmark.env_factory import make_single_env
from bdp_benchmark.tracking_diagnostics import TrackingRecorder, tracking_errors
from bdp_benchmark.common.tracking_geometry import project_trajectory, sample_trajectory
from bdp_benchmark.metadrive.route_reference import resolve_route_lanes, build_route_reference
from bdp_benchmark.visualization import make_overlay


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tracking_output", type=Path, required=True)
    parser.add_argument("--tracking_seeds", type=int, nargs="+", default=[5000, 5001, 5002, 5003, 5004])
    parser.add_argument("--tracking_steps", type=int, default=500)
    parser.add_argument("--tracking_center_action", action="store_true")
    parser.add_argument("--tracking_lane_speed_mps", type=float, default=None,
                        help="Track the fixed assigned lane route at this speed, without a selector.")
    options, rest = parser.parse_known_args()
    if options.tracking_steps <= 0:
        parser.error("--tracking_steps must be positive")
    if options.tracking_lane_speed_mps is not None and options.tracking_lane_speed_mps <= 0:
        parser.error("--tracking_lane_speed_mps must be positive")
    args = parse_args(rest)
    if args.simulator != 'metadrive' or args.trajectory_execution_mode not in ('frenet_pid', 'frenet_pid_v2'):
        parser.error("Tracking diagnostics require a MetaDrive Frenet-PID job")
    args.run_mode = "train"  # Rendering is explicitly opt-in for this diagnostic.
    args.n_envs = 1
    args.monitor_mode = "none"
    env = make_single_env(args)
    raw = env
    while not hasattr(raw, "adapter"):
        raw = raw.env
    output = options.tracking_output
    output.mkdir(parents=True, exist_ok=True)
    recorder = TrackingRecorder(output / "tracking.csv")
    values, episodes = [], []
    started = time.perf_counter()
    try:
        model = None if options.tracking_center_action or options.tracking_lane_speed_mps is not None else PPO.load(
            args.model_path, env=env, device=args.device)
        for seed in options.tracking_seeds:
            obs, _ = env.reset(seed=seed)
            lane_reference = None
            if options.tracking_lane_speed_mps is not None:
                vehicle = raw.env.agent
                route = resolve_route_lanes(vehicle.navigation, vehicle.navigation.current_lane)
                path = build_route_reference(route, planning_xy=vehicle.position, required_length=1e4,
                                            point_count=20001).reference
                x, y, heading = path.sample_center(path.cumulative_s)
                lane_reference = np.column_stack((x, y, heading, np.full(len(x), options.tracking_lane_speed_mps)))
                vehicle.set_velocity([np.cos(vehicle.heading_theta), np.sin(vehicle.heading_theta)],
                                     value=options.tracking_lane_speed_mps)
            distance = total_reward = 0.0
            for step in range(options.tracking_steps):
                before = raw.adapter.ego_state()
                if lane_reference is None:
                    candidates = raw.get_visual_candidate_set()
                    scores = None
                    if model is None:
                        action = 7
                    elif args.visual_check:
                        batched = {key: value[None] for key, value in obs.items()} if isinstance(obs, dict) else obs[None]
                        actions, scores = predict_discrete_with_scores(model, batched, deterministic=True)
                        action = int(np.asarray(actions).item())
                    else:
                        action = int(np.asarray(model.predict(obs, deterministic=True)[0]).item())
                    reference = candidates.trajectories[action].copy()
                    if args.visual_check:
                        raw.set_visual_overlay(make_overlay(
                            labels=candidates.labels, trajectories=candidates.trajectories, features=candidates.features,
                            scores=None if scores is None else np.asarray(scores).reshape(-1), selected_index=action,
                            pid_target_xy=raw.preview_pid_target(action)))
                        raw.render()
                    obs, reward, terminated, truncated, info = env.step(action)
                else:
                    action = -1
                    s, _, arc = project_trajectory(lane_reference, before.xy)
                    distances = s + options.tracking_lane_speed_mps * np.linspace(
                        0, args.trajectory_horizon_s, args.trajectory_sample_count)
                    reference = np.array([sample_trajectory(lane_reference, arc, d) for d in distances])
                    raw._pid_policy().set_reference(reference)
                    obs, reward, terminated, truncated, info = raw.env.step(0)
                actual = raw.adapter.ego_state()
                errors = tracking_errors(reference, actual)
                errors.update(steering=float(raw.env.agent.steering), throttle=float(raw.env.agent.throttle_brake),
                              steering_saturated=float(abs(raw.env.agent.steering) >= 0.99),
                              throttle_saturated=float(abs(raw.env.agent.throttle_brake) >= 0.99))
                errors.update(raw._pid_policy().tracker.last_diagnostics)
                row = {"seed": seed, "step": step, "action": action, **errors,
                       "terminated": terminated, "truncated": truncated,
                       "route_completion": float(info.get("route_completion", 0.0))}
                recorder.write(row)
                values.append(errors)
                distance += float(np.linalg.norm(actual.xy - before.xy))
                total_reward += reward
                if args.visual_check:
                    raw.render()
                    if seed == options.tracking_seeds[0] and step == min(50, options.tracking_steps - 1):
                        raw.env.engine.win.getScreenshot().write(str((output / "visual_check.png").resolve()))
                if terminated or truncated:
                    break
            episodes.append({"seed": seed, "steps": step + 1, "distance_m": distance,
                             "reward": total_reward, "success": bool(info.get("arrive_dest", False)),
                             "out_of_road": bool(info.get("out_of_road", False)),
                             "crash": bool(info.get("crash", False)),
                             "route_completion": float(info.get("route_completion", 0.0))})
    finally:
        recorder.close()
        env.close()
    summary = {key: {"mean": float(np.mean([v[key] for v in values])),
                     "p95": float(np.percentile([v[key] for v in values], 95))} for key in values[0]}
    summary.update(episodes=episodes, steps=len(values), seconds=time.perf_counter() - started,
                   config=resolved_config(args))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
