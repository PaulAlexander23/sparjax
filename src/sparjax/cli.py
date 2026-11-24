"""Command-line interface: ``sparjax <command> ...``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import tyro

from sparjax.config import TrainConfig
from sparjax.loggers import ConsoleLogger, JsonlLogger, Logger, MultiLogger
from sparjax.opponents.tabular import TabularPolicy

if TYPE_CHECKING:
    from sparjax.agents.ppo.system import TrainResult

__all__ = ["CFRBestResponse", "CFRSolve", "Train", "main"]


@dataclass(frozen=True)
class CFRSolve:
    """Solve an OpenSpiel game with CFR and save the average policy (needs the openspiel extra)."""

    game: str = "leduc_poker"
    """OpenSpiel game short name."""
    iters: int = 1000
    """Number of CFR iterations. Fewer iterations give a more exploitable opponent."""
    algorithm: Literal["cfr", "cfr_plus"] = "cfr_plus"
    """CFR variant."""
    out: Path | None = None
    """Output .npz path (default: policies/<game>_<algorithm>_<iters>.npz)."""
    report: bool = True
    """Also print exact best-response values for the saved policy."""

    def run(self) -> Path:
        """Solve, save and optionally report. Returns the output path."""
        from sparjax.opponents.cfr import solve_cfr  # optional dependency

        policy = solve_cfr(self.game, self.iters, algorithm=self.algorithm)
        out = self.out or Path("policies") / f"{self.game}_{self.algorithm}_{self.iters}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        policy.save(out)
        seconds = policy.metadata["solve_seconds"]
        print(f"saved {len(policy)} information states to {out} ({seconds} s)")
        if self.report:
            CFRBestResponse(policy=out).run()
        return out


@dataclass(frozen=True)
class CFRBestResponse:
    """Print exact best-response values and exploitability of a saved tabular policy."""

    policy: Path = field(default=Path("policy.npz"))
    """A .npz file written by ``sparjax cfr-solve``."""

    def run(self) -> None:
        """Compute and print the report."""
        from sparjax.opponents.cfr import best_response_report

        loaded = TabularPolicy.load(self.policy)
        rep = best_response_report(loaded)
        br0, br1 = rep.best_response_value
        on0, on1 = rep.on_policy_value
        print(f"game:                {loaded.game_name}  ({dict(loaded.metadata)})")
        print(f"on-policy value:     seat0 {on0:+.5f}  seat1 {on1:+.5f}")
        print(f"best-response value: seat0 {br0:+.5f}  seat1 {br1:+.5f}")
        print(f"max payoff vs it (random seats) = exploitability: {rep.exploitability:.5f}")


@dataclass(frozen=True)
class Train:
    """Train a PPO agent against a fixed opponent on the Sebulba architecture."""

    config: Annotated[TrainConfig, tyro.conf.arg(name="")] = field(default_factory=TrainConfig)
    exact_eval: bool = True
    """At every log step, evaluate the network exactly (payoff vs opponent, exploitability).
    Only practical for small games such as Kuhn and Leduc."""
    quiet: bool = False
    """Do not print metrics to the console."""

    def run(self) -> TrainResult:
        """Build everything from the config and train."""
        from sparjax.agents.ppo.system import train  # heavy imports
        from sparjax.eval.exact import evaluate_exact, network_to_tabular
        from sparjax.eval.play import play_games
        from sparjax.experiment import (
            make_env_fn,
            make_opponent,
            make_provider,
            make_scripted_opponent,
        )
        from sparjax.opponents.cfr import best_response_report

        cfg = self.config
        game = cfg.env.spec.openspiel_game
        spec = make_opponent(cfg.opponent, game)
        table = spec.table
        exact = self.exact_eval and game is not None and table is not None
        ceiling = best_response_report(table).mean_best_response if exact and table else None
        header = [
            f"env={cfg.env.id}",
            f"opponent={spec.opponent_id}",
            f"ceiling (best response, random seats)={ceiling:.4f}" if ceiling is not None else "",
            f"updates={cfg.num_updates}",
            f"steps/update={cfg.steps_per_update}",
        ]
        print(" ".join(h for h in header if h))

        eval_opponents = {n: make_scripted_opponent(n) for n in cfg.eval.opponents}

        def on_update(update: int, graphdef: Any, params: Any) -> dict[str, float]:
            metrics: dict[str, float] = {}
            for name, opponent in eval_opponents.items():
                report = play_games(
                    make_env_fn(cfg),
                    graphdef,
                    params,
                    opponent,
                    cfg.eval.games,
                    greedy=cfg.eval.greedy,
                    seed=cfg.seed + update,
                )
                metrics.update(report.as_metrics(f"play/{name}"))
            if game is None or table is None or ceiling is None:
                return metrics
            learned = network_to_tabular(graphdef, params, game)
            ev = evaluate_exact(learned, table)
            return metrics | {
                "eval/payoff": ev.mean_payoff,
                "eval/payoff_seat0": ev.payoff_vs_opponent[0],
                "eval/payoff_seat1": ev.payoff_vs_opponent[1],
                "eval/ceiling": ceiling,
                "eval/gap_to_ceiling": ceiling - ev.mean_payoff,
                "eval/exploitability": ev.exploitability,
            }

        loggers: list[Logger] = []
        if not self.quiet:
            loggers.append(ConsoleLogger(_CONSOLE_KEYS))
        if cfg.log_dir is not None:
            loggers.append(JsonlLogger(cfg.log_dir / "metrics.jsonl"))
        hook = on_update if exact or eval_opponents else None
        return train(cfg, make_env_fn(cfg), make_provider(spec), MultiLogger(loggers), hook)


_CONSOLE_KEYS = [
    "train/return_mean",
    "train/win_rate",
    "play/random/win_rate",
    "play/go_light/win_rate",
    "eval/payoff",
    "eval/gap_to_ceiling",
    "eval/exploitability",
    "loss/entropy",
    "ppo/approx_kl",
    "perf/sps",
]


Command = (
    Annotated[CFRSolve, tyro.conf.subcommand("cfr-solve")]
    | Annotated[CFRBestResponse, tyro.conf.subcommand("cfr-br")]
    | Annotated[Train, tyro.conf.subcommand("train")]
)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``sparjax`` console script."""
    command = tyro.cli(Command, args=argv)
    command.run()


if __name__ == "__main__":  # pragma: no cover
    main()
