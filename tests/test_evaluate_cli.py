from types import SimpleNamespace

from scripts.evaluate import apply_visual_check_defaults


def test_visual_check_restores_deterministic_ten_episode_defaults() -> None:
    args = SimpleNamespace(visual_check=True, inference_mode="stochastic", evaluate_episodes=3)

    apply_visual_check_defaults(args, ["--visual_check"])

    assert args.inference_mode == "deterministic"
    assert args.evaluate_episodes == 10


def test_visual_check_preserves_explicit_inference_and_episode_overrides() -> None:
    args = SimpleNamespace(visual_check=True, inference_mode="stochastic", evaluate_episodes=3)

    apply_visual_check_defaults(
        args,
        ["--visual_check", "--inference_mode=stochastic", "--evaluate_episodes", "3"],
    )

    assert args.inference_mode == "stochastic"
    assert args.evaluate_episodes == 3


def test_non_visual_evaluation_is_unchanged() -> None:
    args = SimpleNamespace(visual_check=False, inference_mode="stochastic", evaluate_episodes=3)

    apply_visual_check_defaults(args, [])

    assert args.inference_mode == "stochastic"
    assert args.evaluate_episodes == 3
