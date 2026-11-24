import json
from pathlib import Path

import numpy as np
import pytest

from sparjax.agents.ppo.loss import PPOHyperparams
from sparjax.agents.ppo.system import train
from sparjax.cli import main
from sparjax.config import ArchConfig, EnvConfig, OpponentConfig, PPOConfig, TrainConfig
from sparjax.envs.openspiel_aec import OpenSpielAECEnv
from sparjax.eval.exact import evaluate_exact, network_to_tabular
from sparjax.experiment import make_env_fn, make_opponent, make_provider
from sparjax.loggers import ConsoleLogger, JsonlLogger, MultiLogger, NullLogger
from sparjax.opponents import FixedProvider, RandomOpponent
from sparjax.opponents.cfr import solve_cfr

KUHN_CEILING_VS_RANDOM = 0.4583333


def small_cfg(**kw: object) -> TrainConfig:
    base: dict[str, object] = {
        "env": EnvConfig("kuhn_poker", num_envs=32),
        "arch": ArchConfig(num_actors=2),
        "ppo": PPOConfig(rollout_len=16, update=PPOHyperparams(num_minibatches=4)),
        "total_timesteps": 2 * 32 * 16 * 60,
        "log_every": 20,
    }
    base.update(kw)
    return TrainConfig(**base)  # pyright: ignore[reportArgumentType]


@pytest.mark.slow
def test_ppo_learns_to_exploit_random_kuhn_opponent() -> None:
    """CI smoke test: PPO approaches the exact best-response payoff vs a random player."""
    cfg = small_cfg()
    spec = make_opponent(cfg.opponent, cfg.env.game)
    result = train(cfg, make_env_fn(cfg), make_provider(spec))
    ev = evaluate_exact(
        network_to_tabular(result.graphdef, result.params, "kuhn_poker"), spec.table
    )
    assert ev.mean_payoff > 0.35, ev
    assert ev.mean_payoff <= KUHN_CEILING_VS_RANDOM + 1e-6
    assert len(result.history) == 3
    assert result.history[-1]["perf/param_lag"] <= 1
    assert len(result.episode_returns) > 1000


def test_strictly_on_policy_mode_has_zero_lag() -> None:
    cfg = small_cfg(
        arch=ArchConfig(num_actors=1, max_lag=0), total_timesteps=32 * 16 * 4, log_every=1
    )
    result = train(cfg, make_env_fn(cfg), FixedProvider(RandomOpponent()))
    assert all(h["perf/param_lag"] == 0 for h in result.history)


def test_actor_failure_is_raised_not_hung() -> None:
    calls = {"n": 0}

    def flaky_env() -> OpenSpielAECEnv:
        calls["n"] += 1
        if calls["n"] > 1:  # the probe env works, actor envs fail
            raise RuntimeError("env exploded")
        return OpenSpielAECEnv("kuhn_poker")

    with pytest.raises(RuntimeError, match="worker thread failed"):
        train(small_cfg(), flaky_env, FixedProvider(RandomOpponent()))


def test_on_update_hook_and_jsonl_logging(tmp_path: Path) -> None:
    cfg = small_cfg(total_timesteps=2 * 32 * 16 * 4, log_every=2)
    seen: list[int] = []

    def hook(update: int, graphdef: object, params: object) -> dict[str, float]:
        del graphdef, params
        seen.append(update)
        return {"custom/metric": 1.0}

    logger = MultiLogger([JsonlLogger(tmp_path / "m.jsonl"), NullLogger()])
    result = train(cfg, make_env_fn(cfg), FixedProvider(RandomOpponent()), logger, hook)
    assert seen == [2, 4]
    rows = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [2 * 1024, 4 * 1024]
    assert rows[0]["custom/metric"] == 1.0
    assert result.history[0]["custom/metric"] == 1.0


def test_config_properties() -> None:
    cfg = small_cfg()
    assert cfg.steps_per_update == 1024
    assert cfg.num_updates == 60
    assert TrainConfig(total_timesteps=1).num_updates == 1


def test_make_opponent_variants(tmp_path: Path) -> None:
    rand = make_opponent(OpponentConfig("random"), "kuhn_poker")
    assert rand.opponent_id == "random"
    cfr = make_opponent(OpponentConfig("cfr", cfr_iters=5), "kuhn_poker")
    assert cfr.opponent_id == "cfr_plus_5"
    path = solve_cfr("kuhn_poker", 3).save(tmp_path / "k.npz")
    loaded = make_opponent(OpponentConfig("policy", policy_path=path), "kuhn_poker")
    assert loaded.opponent_id == "k"
    assert len(loaded.table) == 12
    with pytest.raises(ValueError, match="policy_path"):
        make_opponent(OpponentConfig("policy"), "kuhn_poker")
    with pytest.raises(ValueError, match="not leduc_poker"):
        make_opponent(OpponentConfig("policy", policy_path=path), "leduc_poker")


def test_console_logger(capsys: pytest.CaptureFixture[str]) -> None:
    ConsoleLogger(["a", "missing"]).log(5, {"a": 1.5, "b": 2.0})
    ConsoleLogger().log(6, {"b": 2.0})
    out = capsys.readouterr().out
    assert "a=1.5" in out
    assert "b=2" in out
    assert "missing" not in out


def test_jsonl_logger_writes_null_for_nan(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "x" / "m.jsonl")
    logger.log(1, {"v": float("nan")})
    logger.close()
    assert json.loads((tmp_path / "x" / "m.jsonl").read_text())["v"] is None


def test_cli_train_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "train",
            "--env.num-envs",
            "16",
            "--arch.num-actors",
            "1",
            "--total-timesteps",
            str(16 * 16 * 4),
            "--log-every",
            "2",
            "--log-dir",
            str(tmp_path),
            "--opponent.kind",
            "cfr",
            "--opponent.cfr-iters",
            "5",
        ]
    )
    out = capsys.readouterr().out
    assert "ceiling" in out
    assert "eval/payoff" in out
    rows = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(rows) == 2
    assert "eval/exploitability" in json.loads(rows[-1])


def test_cli_train_quiet_without_eval(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "train",
            "--quiet",
            "--no-exact-eval",
            "--env.num-envs",
            "8",
            "--arch.num-actors",
            "1",
            "--total-timesteps",
            "128",
            "--log-every",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert "eval/payoff" not in out
    assert np.isfinite(1.0)
