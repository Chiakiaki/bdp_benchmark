from pathlib import Path
import os
import subprocess

import pytest
import yaml

from bdp_benchmark.config import parse_args, resolved_config

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "job_scripts"
BUNDLES = ("benchmark_more_algorithm", "benchmark_metadrive_12mps_more_algorithm")


@pytest.mark.parametrize("bundle", BUNDLES)
def test_renamed_bundle_preserves_trpo_and_adds_four_jobs(bundle):
    folder = ROOT / bundle
    assert folder.is_dir()
    assert not (ROOT / bundle.replace("_more_algorithm", "_trpo")).exists()
    files = sorted(folder.glob("*.yaml"))
    assert len(files) == 9
    names = set()
    counts = {"TRPO": 0, "A2C": 0, "DQN": 0}
    for path in files:
        config = yaml.safe_load(path.read_text())
        algorithm = config["algorithm"]
        name = algorithm["name"]
        counts[name] += 1
        names.add(config["runtime"]["run_name"])
        assert algorithm["learning_rate"] == 5e-5
        assert algorithm["num_timesteps"] == 2000000
        assert algorithm["n_envs"] == 8
        assert algorithm["gamma"] == 0.99
        if name == "TRPO":
            assert algorithm["n_critic_updates"] == 20 and algorithm["batch_size"] == 100
        else:
            assert config["model_architecture"]["policy_mode"] == "builtin"
            mode = "frenet_pid_v3" if config["environment"]["trajectory_execution_mode"] == "frenet_pid_v3" else "native"
            original = yaml.safe_load((ROOT / bundle.removesuffix("_more_algorithm") / f"metadrive_{mode}_builtin_benchmark.yaml").read_text())
            for section in ("environment", "frenet", "pid", "logging"):
                assert config.get(section) == original.get(section)
            assert "n_critic_updates" not in algorithm and "ppo_epochs" not in algorithm
            assert "target_kl" not in algorithm
            if name == "A2C":
                assert algorithm["trpo_timesteps_per_batch"] == 1000
                assert algorithm["gae_lambda"] == 0.95
                assert "batch_size" not in algorithm
                assert config["model_architecture"] == original["model_architecture"]
            else:
                assert algorithm["batch_size"] == 100
                assert "trpo_timesteps_per_batch" not in algorithm and "gae_lambda" not in algorithm
                assert algorithm["dqn_buffer_size"] == 1000000
                assert "value_layers" not in config["model_architecture"]
        args = parse_args(["--config", str(path)])
        assert args.sb3_algorithm == name
        effective = resolved_config(args)["algorithm"]
        for key, value in algorithm.items():
            assert effective["sb3_algorithm" if key == "name" else key] == value
    assert counts == {"TRPO": 5, "A2C": 2, "DQN": 2}
    assert len(names) == 9


@pytest.mark.parametrize("bundle", BUNDLES)
def test_more_algorithm_launcher_and_retained_trpo_launcher(bundle, tmp_path):
    folder = ROOT / bundle
    for name, count in (("run_metadrive_more_algorithm_comparison.sh", 9),
                        ("run_metadrive_trpo_comparison.sh", 5)):
        script = folder / name
        subprocess.run(["bash", "-n", str(script)], check=True)
        result = subprocess.run(["bash", str(script), "--num_timesteps", "32"],
                                cwd=tmp_path, check=True, text=True, capture_output=True,
                                env={**os.environ, "BDP_BENCHMARK_PYTHON": "/bin/echo"})
        lines = [line for line in result.stdout.splitlines() if line.startswith("scripts/train.py")]
        assert len(lines) == count
        for line in lines:
            assert line.endswith("--num_timesteps 32")
            assert Path(line.split("--config ")[1].split(" --")[0]).is_file()
        result = subprocess.run(["bash", str(script)], cwd=tmp_path, text=True, capture_output=True,
                                env={**os.environ, "BDP_BENCHMARK_PYTHON": "/bin/false"})
        assert result.returncode != 0 and "Starting" not in result.stdout


@pytest.mark.parametrize("algorithm", ["A2C", "DQN"])
def test_new_algorithm_config_roundtrip_with_cli_override(algorithm):
    path = ROOT / BUNDLES[0] / f"metadrive_native_{algorithm.lower()}_builtin_benchmark.yaml"
    args = parse_args(["--config", str(path), "--learning_rate", "0.0002"])
    assert resolved_config(args)["algorithm"]["learning_rate"] == 0.0002


@pytest.mark.parametrize("algorithm", ["A2C", "DQN"])
def test_evaluation_rejects_unsupported_bdp_algorithm_before_loading(algorithm):
    from types import SimpleNamespace
    from bdp_benchmark.runner import _load_evaluation_model
    args = SimpleNamespace(sb3_algorithm=algorithm, policy_mode="bdp")
    with pytest.raises(ValueError, match="builtin"):
        _load_evaluation_model(args, None)
