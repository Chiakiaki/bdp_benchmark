from pathlib import Path

import numpy as np
import pytest

from bdp_benchmark.config import parse_args
from bdp_benchmark.env_factory import make_single_env
from bdp_benchmark.runner import run_training
from critic_based_rl.inference import predict_discrete_with_scores
from critic_based_rl.model import get_algorithm_class


@pytest.mark.parametrize("bundle", ["benchmark_more_algorithm", "benchmark_metadrive_12mps_more_algorithm"])
@pytest.mark.parametrize("mode", ["native", "frenet_pid_v3"])
@pytest.mark.parametrize("algorithm", ["A2C", "DQN"])
def test_builtin_metadrive_short_train_save_reload(bundle, mode, algorithm, tmp_path):
    root = Path(__file__).resolve().parents[1] / "scripts/job_scripts"
    args = parse_args([
        "--config", str(root / bundle / f"metadrive_{mode}_{algorithm.lower()}_builtin_benchmark.yaml"),
        "--n_envs", "1", "--vec_env", "dummy", "--num_timesteps", "16",
        "--trpo_timesteps_per_batch", "8", "--batch_size", "8",
        "--dqn_buffer_size", "128", "--dqn_learning_starts", "0", "--dqn_train_freq", "1",
        "--save_freq", "8", "--eval_freq", "0", "--device", "cpu", "--log_path", str(tmp_path),
    ])
    log = run_training(args)
    checkpoint = log / "models" / f"{algorithm}_builtin_final_model.zip"
    assert checkpoint.exists()
    assert list((log / "models").glob("step_*_steps.zip"))
    env = make_single_env(args)
    try:
        model = get_algorithm_class(algorithm).load(checkpoint, env=env, device="cpu")
        observation, _ = env.reset(seed=41)
        assert observation.shape == (275,)
        assert env.action_space.n == 25
        action, scores = predict_discrete_with_scores(model, observation[None], deterministic=True)
        assert scores.shape == (1, 25)
        assert np.isfinite(scores.numpy()).all()
        env.step(int(action[0]))
        assert model._n_updates > 0
    finally:
        env.close()
