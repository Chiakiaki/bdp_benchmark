import numpy as np
import pytest

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import EgoState
from bdp_benchmark.common.tracking import TrackerConfig, TrajectoryPIDTracker
from bdp_benchmark.config import parse_args, resolved_config
from bdp_benchmark.env_factory import generation_config_from_args
from bdp_benchmark.metadrive.env import MetaDriveBenchmarkEnv


def test_v3_is_metadrive_only_and_tau_is_independent_of_steering():
    with pytest.raises(ValueError, match="only for MetaDrive"):
        parse_args(["--trajectory_execution_mode", "frenet_pid_v3"])
    args = parse_args(["--simulator", "metadrive", "--trajectory_execution_mode", "frenet_pid_v3",
                       "--frenet_speed_command_time_s", "0.4", "--pid_lookahead_time_s", "0.7"])
    assert args.frenet_speed_command_time_s < args.pid_lookahead_time_s


def test_old_short_horizon_is_not_rejected_by_inactive_v3_tau():
    args = parse_args(["--simulator", "metadrive", "--trajectory_execution_mode", "frenet_pid_v2",
                       "--trajectory_horizon_s", "0.5"])
    assert args.frenet_speed_command_time_s == 1


@pytest.mark.parametrize("horizon", ["nan", "inf"])
def test_v3_rejects_nonfinite_horizon_before_environment_creation(horizon):
    with pytest.raises(ValueError, match="trajectory_horizon_s"):
        parse_args(["--simulator", "metadrive", "--trajectory_execution_mode", "frenet_pid_v3",
                    "--trajectory_horizon_s", horizon])


def test_v3_cli_and_backup_roundtrip(tmp_path):
    import yaml
    args = parse_args([
        "--simulator", "metadrive", "--env_id", "MetaDrive-v0",
        "--trajectory_execution_mode", "frenet_pid_v3", "--pid_controller_mode", "adaptive_pursuit",
        "--frenet_speed_command_time_s", "0.8", "--frenet_speed_delta_rate_mps2", "7",
        "--frenet_speed_action_scales", "-1", "-0.2", "0", "0.2", "1",
    ])
    c = generation_config_from_args(args)
    assert c.speed_command_time_s == 0.8
    assert c.speed_delta_rate_mps2 == 7
    assert tuple(c.speed_action_scales) == (-1, -.2, 0, .2, 1)
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(resolved_config(args)))
    assert generation_config_from_args(parse_args(["--config", str(path)])) == c


@pytest.mark.parametrize("opts", [["--frenet_speed_command_time_s", "3"],
                                  ["--frenet_speed_action_scales", "-1", "0", "1"],
                                  ["--frenet_speed_action_scales", "-1", "0", "0", ".5", "1"]])
def test_v3_invalid_settings_fail_clearly(opts):
    with pytest.raises(ValueError):
        parse_args(["--simulator", "metadrive", "--trajectory_execution_mode", "frenet_pid_v3", *opts])


def test_direct_speed_setpoint_bypasses_preview_and_clears_on_reset_or_legacy_reference():
    tracker = TrajectoryPIDTracker(TrackerConfig(controller_mode="adaptive_pursuit", speed_kp=2, speed_ki=0))
    reference = np.column_stack((np.linspace(0, 24, 11), np.zeros(11), np.zeros(11), np.full(11, 12)))
    tracker.set_reference(reference, speed_target_mps=2)
    control = tracker.step(EgoState(0, 0, 0, 10), dt=.1)
    assert tracker.last_diagnostics["target_speed_mps"] == 2
    assert control.acceleration_mps2 == -8
    tracker.set_reference(reference)
    assert tracker.step(EgoState(0, 0, 0, 10), dt=.1).acceleration_mps2 > 0
    tracker.reset()
    assert tracker.speed_target_mps is None


@pytest.mark.parametrize("curves", [False, True])
@pytest.mark.parametrize("mode", ["frenet_pid_v3", "frenet_pid_v3_legacy"])
def test_v3_nominal_pose_actual_speed_and_selected_target_metadata(curves, mode):
    env = MetaDriveBenchmarkEnv(
        execution_mode=mode,
        generation_config=CandidateGenerationConfig(maximum_target_speed_mps=12, include_curvature_candidates=curves),
        tracker_config=TrackerConfig(controller_mode="adaptive_pursuit", speed_kp=2, speed_ki=0),
        reference_mode="route_continuous",
        env_config={"map": "S", "num_scenarios": 1, "traffic_density": 0, "log_level": 50},
    )
    try:
        obs, _ = env.reset(seed=0)
        vehicle = env.env.agent
        vehicle.set_velocity(vehicle.heading, 10)
        actual = env.adapter.ego_state()
        nominal = EgoState(actual.x + .3, actual.y, actual.heading, 25)
        env._nominal_state.commit(np.array([
            [nominal.x, nominal.y, nominal.heading, 25],
            [nominal.x + 10, nominal.y, nominal.heading, 25],
        ]))
        candidates = env.build_candidate_set()
        count = 25 if curves else 15
        assert candidates.features.shape == (count, 55)
        np.testing.assert_allclose(candidates.trajectories[:, 0, :2], np.tile(nominal.xy, (count, 1)), atol=1e-5)
        np.testing.assert_allclose(candidates.trajectories[:, 0, 3], actual.speed)
        if mode.endswith("legacy"):
            np.testing.assert_allclose(candidates.speed_targets[:15], np.repeat([2, 6, 10, 12, 12], 3), atol=1e-4)
        else:
            assert candidates.speed_targets is None
            np.testing.assert_allclose(candidates.trajectories[:15, -1, 3], np.repeat([2, 6, 10, 12, 12], 3), atol=1e-4)
        if curves:
            np.testing.assert_allclose(candidates.trajectories[15:, -1, 3], np.repeat([2, 6, 10, 12, 12], 2), atol=1e-4)
            np.testing.assert_allclose(candidates.trajectories[15, :, 3], candidates.trajectories[0, :, 3])
        env.preview_pid_target(0)
        if mode.endswith("legacy"):
            assert env._pid_policy().tracker.speed_target_mps == pytest.approx(2, abs=1e-4)
        else:
            assert env._pid_policy().tracker.speed_target_mps is None
        next_obs, reward, *_ = env.step(0)
        assert next_obs.shape == obs.shape
        assert np.isfinite(reward)
        if mode.endswith("legacy"):
            assert env._pid_policy().tracker.last_diagnostics["target_speed_mps"] == pytest.approx(2, abs=1e-4)
            assert vehicle.throttle_brake < -0.99
        else:
            assert env._pid_policy().tracker.last_diagnostics["target_speed_mps"] > 2.1
    finally:
        env.close()


@pytest.mark.parametrize("simulator", ["metadrive", "highway"])
def test_v3_legacy_config_is_explicit_and_metadrive_only(simulator):
    argv = ["--simulator", simulator, "--trajectory_execution_mode", "frenet_pid_v3_legacy"]
    if simulator == "highway":
        with pytest.raises(ValueError, match="only for MetaDrive"):
            parse_args(argv)
    else:
        args = parse_args(argv)
        assert resolved_config(args)["environment"]["trajectory_execution_mode"] == "frenet_pid_v3_legacy"
