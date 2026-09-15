import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from critic_based_rl.inference import predict_discrete_with_scores
from critic_based_rl.policies import BDPBoltzmannPolicy
from bdp_benchmark.config import parse_args
from bdp_benchmark import runner


class CandidateEnv(gym.Env):
    def __init__(self, count, state_dim=4, feature_dim=3):
        self.action_space = spaces.Discrete(count)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(-1, 1, (state_dim,), dtype=np.float32),
            "candidates": spaces.Box(-1, 1, (count, feature_dim), dtype=np.float32),
            "candidate_mask": spaces.Box(0, 1, (count,), dtype=np.float32),
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.observation_space.sample()
        obs["candidate_mask"][:] = 1
        return obs, {}

    def step(self, action):
        return self.reset()[0], 0.0, True, False, {}


@pytest.mark.parametrize("scorer", ["dot", "mlp"])
def test_evaluation_load_supports_new_candidate_count_without_changing_weights(tmp_path, scorer):
    old_env = DummyVecEnv([lambda: CandidateEnv(15)])
    new_env = DummyVecEnv([lambda: CandidateEnv(25)])
    try:
        model = PPO(BDPBoltzmannPolicy, old_env, n_steps=2, batch_size=2, device="cpu",
                    policy_kwargs={"candidate_scorer": scorer, "candidate_net_arch": [8], "value_net_arch": [8]})
        path = tmp_path / "old.zip"
        model.save(path)
        args = parse_args(["--policy_mode", "bdp", "--candidate_sampler", "one_hot",
                           "--model_path", str(path), "--device", "cpu", "--allow_bdp_candidate_count_change"])
        loaded = runner._load_evaluation_model(args, new_env)
        assert loaded.action_space.n == 25
        for key, value in model.policy.state_dict().items():
            torch.testing.assert_close(loaded.policy.state_dict()[key], value, rtol=0, atol=0)
        action, scores = predict_discrete_with_scores(loaded, new_env.reset(), deterministic=True)
        assert scores.shape == (1, 25)
        assert 0 <= int(action[0]) < 25
        args.allow_bdp_candidate_count_change = False
        with pytest.raises(ValueError, match="spaces do not match"):
            runner._load_evaluation_model(args, new_env)
    finally:
        old_env.close()
        new_env.close()


@pytest.mark.parametrize("dimensions", [{"state_dim": 5}, {"feature_dim": 4}])
def test_evaluation_count_override_does_not_allow_other_input_changes(tmp_path, dimensions):
    old_env = DummyVecEnv([lambda: CandidateEnv(15)])
    new_env = DummyVecEnv([lambda: CandidateEnv(25, **dimensions)])
    try:
        model = PPO(BDPBoltzmannPolicy, old_env, n_steps=2, batch_size=2, device="cpu",
                    policy_kwargs={"candidate_scorer": "dot", "candidate_net_arch": [8], "value_net_arch": [8]})
        path = tmp_path / "old.zip"
        model.save(path)
        args = parse_args(["--policy_mode", "bdp", "--model_path", str(path),
                           "--device", "cpu", "--allow_bdp_candidate_count_change"])
        with pytest.raises(ValueError, match="not compatible except for candidate count"):
            runner._load_evaluation_model(args, new_env)
    finally:
        old_env.close()
        new_env.close()
