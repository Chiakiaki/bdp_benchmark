import numpy as np
import pytest

from bdp_benchmark.common.contracts import EgoState
from bdp_benchmark.common.tracking import TrackerConfig, TrajectoryPIDTracker
from bdp_benchmark.tracking_diagnostics import tracking_errors


def reference(y=0.0, speed=8.0):
    return np.column_stack((np.linspace(0, 40, 11), np.full(11, y), np.zeros(11), np.full(11, speed)))


def test_diagnostics_distinguish_sample_spacing_from_cross_track_error():
    errors = tracking_errors(reference(), EgoState(2, 0, 0, 8))
    assert errors['nominal_error_m'] == 2
    assert errors['path_error_m'] == 0


def test_pursuit_is_symmetric_and_uses_interpolated_lookahead():
    controls = []
    for y in [-1, 1]:
        tracker = TrajectoryPIDTracker(TrackerConfig(controller_mode='adaptive_pursuit'))
        tracker.set_reference(reference(y))
        controls.append(tracker.step(EgoState(2, 0, 0, 8), dt=0.1))
    assert controls[0].steering_rad == pytest.approx(-controls[1].steering_rad)
    assert controls[1].steering_rad > 0


def test_speed_integrator_survives_reference_changes_and_rejects_windup():
    tracker = TrajectoryPIDTracker(TrackerConfig(controller_mode='adaptive_pursuit', speed_ki=1.0))
    tracker.set_reference(reference(speed=9))
    tracker.step(EgoState(2, 0, 0, 8), dt=0.1)
    integral = tracker._speed_pid.integral
    tracker.set_reference(reference(speed=9))
    assert tracker._speed_pid.integral == integral
    tracker.set_reference(reference(speed=100))
    for _ in range(100):
        tracker.step(EgoState(2, 0, 0, 0), dt=0.1)
    assert tracker._speed_pid.integral == integral
    tracker.reset()
    assert tracker._speed_pid.integral == 0


def test_stationary_reference_returns_finite_control():
    tracker = TrajectoryPIDTracker(TrackerConfig(controller_mode='adaptive_pursuit'))
    tracker.set_reference(np.zeros((11, 4)))
    control = tracker.step(EgoState(0, 0, 0, 0), dt=0.1)
    assert control.steering_rad == 0
    assert control.acceleration_mps2 == 0


def test_preview_uses_distance_interpolation_and_does_not_change_integrator():
    tracker = TrajectoryPIDTracker(TrackerConfig(controller_mode='adaptive_pursuit', lookahead_time_s=0.7))
    tracker.set_reference(reference())
    before = tracker._speed_pid.integral
    target = tracker.preview_target(EgoState(2, 0, 0, 8))
    np.testing.assert_allclose(target, [7.6, 0])
    assert tracker._speed_pid.integral == before


def test_circular_center_reference_has_correct_rear_axle_geometry():
    radius, rear, wheelbase = 10.0, 1.4166, 2.46894
    theta = np.linspace(-0.2, 1.5, 300)
    path = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), theta + np.pi / 2, np.ones(len(theta)) * 8))
    tracker = TrajectoryPIDTracker(TrackerConfig(controller_mode='adaptive_pursuit'),
                                   wheelbase=wheelbase, rear_axle_offset=rear)
    tracker.set_reference(path)
    ego = EgoState(radius, 0, np.pi / 2 - np.arcsin(rear / radius), 8)
    control = tracker.step(ego, dt=0.1)
    assert control.steering_rad == pytest.approx(np.arctan(wheelbase / np.sqrt(radius**2 - rear**2)), abs=0.003)


def test_positive_time_preview_accelerates_from_rest():
    t = np.linspace(0, 2, 11)
    path = np.column_stack((0.5*t**2, np.zeros(11), np.zeros(11), t))
    tracker = TrajectoryPIDTracker(TrackerConfig(controller_mode='adaptive_pursuit'))
    tracker.set_reference(path)
    assert tracker.step(EgoState(0, 0, 0, 0), dt=0.1).acceleration_mps2 > 0
    with pytest.raises(ValueError, match='speed_preview_s'):
        TrackerConfig(controller_mode='adaptive_pursuit', speed_preview_s=0)


def test_braking_windup_is_blocked_and_integrator_can_unwind():
    tracker = TrajectoryPIDTracker(TrackerConfig(controller_mode='adaptive_pursuit', speed_ki=1))
    tracker.set_reference(reference(speed=0))
    for _ in range(50):
        tracker.step(EgoState(2, 0, 0, 30), dt=0.1)
    assert tracker._speed_pid.integral == 0
    tracker._speed_pid.integral = 20
    tracker.set_reference(reference(speed=8))
    tracker.step(EgoState(2, 0, 0, 9), dt=0.1)
    assert tracker._speed_pid.integral < 20


def test_metadrive_adaptive_controller_uses_decision_dt_and_trajectory_dt():
    from bdp_benchmark.config import parse_args
    from bdp_benchmark.env_factory import make_single_env
    args = parse_args(['--simulator', 'metadrive', '--trajectory_execution_mode', 'frenet_pid_v2',
                       '--pid_controller_mode', 'adaptive_pursuit', '--tracking_diagnostics',
                       '--trajectory_horizon_s', '3', '--trajectory_sample_count', '11',
                       '--environment_config', '{"map":"S","traffic_density":0,"num_scenarios":1,"log_level":50}'])
    env = make_single_env(args)
    try:
        obs, _ = env.reset(seed=0)
        tracker = env._pid_policy().tracker
        assert tracker.sample_dt == 0.3
        dts = []
        original = tracker.step
        def capture(ego, *, dt):
            dts.append(dt)
            return original(ego, dt=dt)
        tracker.step = capture
        after, _, _, _, info = env.step(13)
        assert dts == [0.1]
        assert after.shape == obs.shape
        assert 'path_error_m' in info['tracking']
    finally:
        env.close()


def test_diagnostics_do_not_change_observation_reward_or_motion():
    from bdp_benchmark.config import parse_args
    from bdp_benchmark.env_factory import make_single_env
    outputs = []
    for diagnostics in ['--no-tracking_diagnostics', '--tracking_diagnostics']:
        args = parse_args(['--simulator', 'metadrive', '--trajectory_execution_mode', 'frenet_pid_v2',
                           '--pid_controller_mode', 'adaptive_pursuit', diagnostics,
                           '--environment_config', '{"map":"S","traffic_density":0,"num_scenarios":1,"log_level":50}'])
        env = make_single_env(args)
        try:
            env.reset(seed=0)
            transitions = []
            for action in [13, 13, 7, 1]:
                obs, reward, terminated, truncated, _ = env.step(action)
                transitions.append((obs, reward, terminated, truncated))
            outputs.append(transitions)
        finally:
            env.close()
    for off, on in zip(*outputs):
        np.testing.assert_array_equal(off[0], on[0])
        assert off[1:] == on[1:]
