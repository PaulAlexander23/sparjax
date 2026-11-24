"""Command-line interface: ``sparjax <command> ...``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import tyro

from sparjax.opponents.tabular import TabularPolicy

__all__ = ["CFRBestResponse", "CFRSolve", "main"]


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


Command = (
    Annotated[CFRSolve, tyro.conf.subcommand("cfr-solve")]
    | Annotated[CFRBestResponse, tyro.conf.subcommand("cfr-br")]
)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``sparjax`` console script."""
    command = tyro.cli(Command, args=argv)
    command.run()
