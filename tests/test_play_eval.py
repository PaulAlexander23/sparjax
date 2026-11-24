from pathlib import Path

import pytest

from sparjax.agents.ppo.networks import init_actor_critic
from sparjax.cli import main
from sparjax.envs.registry import EnvSpec
from sparjax.eval.play import play_games
from sparjax.opponents import RandomOpponent


@pytest.mark.parametrize("greedy", [False, True])
def test_play_games_report(greedy: bool) -> None:
    graphdef, params = init_actor_critic(11, 2, (8,), seed=0)
    report = play_games(
        EnvSpec("openspiel:kuhn_poker"),
        graphdef,
        params,
        RandomOpponent(),
        200,
        num_envs=16,
        greedy=greedy,
    )
    assert report.games >= 200
    assert report.win_rate + report.draw_rate + report.loss_rate == pytest.approx(1.0)
    assert sum(report.games_by_seat) == report.games
    assert report.ci95 > 0
    metrics = report.as_metrics("play/random")
    assert metrics["play/random/games"] == report.games
    with pytest.raises(ValueError, match="num_games"):
        play_games(EnvSpec("openspiel:kuhn_poker"), graphdef, params, RandomOpponent(), 0)


def test_cli_train_with_play_eval_and_truncation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "train",
            "--env.id",
            "pettingzoo:classic/go-v5",
            "--env.kwargs",
            "board_size",
            "5",
            "--env.max-steps",
            "8",
            "--env.num-envs",
            "4",
            "--arch.num-actors",
            "1",
            "--opponent.kind",
            "go_light",
            "--ppo.conv-channels",
            "4",
            "--ppo.hidden",
            "8",
            "--ppo.rollout-len",
            "8",
            "--total-timesteps",
            "64",
            "--log-every",
            "1",
            "--eval.opponents",
            "random",
            "go_light",
            "--eval.games",
            "8",
            "--log-dir",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert "opponent=go_light" in out
    rows = (tmp_path / "metrics.jsonl").read_text()
    assert "play/random/win_rate" in rows
    assert "play/go_light/win_rate" in rows
