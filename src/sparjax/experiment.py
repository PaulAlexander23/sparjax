"""Build environments and opponents from a :class:`~sparjax.config.TrainConfig`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sparjax.config import OpponentConfig, TrainConfig
from sparjax.opponents.base import Opponent, OpponentProvider
from sparjax.opponents.go import GoLightPlayoutOpponent
from sparjax.opponents.providers import FixedProvider
from sparjax.opponents.selfplay import SnapshotPoolProvider
from sparjax.opponents.simple import RandomOpponent
from sparjax.opponents.tabular import TabularPolicy, TabularPolicyOpponent

__all__ = [
    "OpponentSpec",
    "make_env_fn",
    "make_opponent",
    "make_provider",
    "make_scripted_opponent",
]


@dataclass(frozen=True)
class OpponentSpec:
    """A constructed opponent plus its tabular form when it has one (for exact evaluation)."""

    opponent_id: str
    opponent: Opponent | None
    """``None`` for self-play, whose opponents come from the snapshot pool."""
    table: TabularPolicy | None
    """Tabular form, available for OpenSpiel games (enables exact evaluation)."""


def make_env_fn(cfg: TrainConfig) -> Callable[[], Any]:
    """A picklable factory for one environment of ``cfg.env``."""
    return cfg.env.spec


def make_opponent(cfg: OpponentConfig, game: str | None) -> OpponentSpec:
    """Build the opponent described by ``cfg``.

    Args:
        cfg: Opponent configuration.
        game: OpenSpiel game name, or ``None`` for non-OpenSpiel envs (where only the
            random opponent is available and there is no tabular form).

    For OpenSpiel games every supported opponent is tabular (random play is CFR's first
    average policy), so the exact best-response ceiling is always computable.
    """
    if cfg.kind == "selfplay":
        return OpponentSpec("selfplay", None, None)
    if cfg.kind == "go_light":
        return OpponentSpec("go_light", GoLightPlayoutOpponent(), None)
    if game is None:
        if cfg.kind != "random":
            raise ValueError(f"opponent.kind={cfg.kind!r} needs an OpenSpiel env")
        return OpponentSpec("random", RandomOpponent(), None)
    if cfg.kind == "random":
        from sparjax.opponents.cfr import solve_cfr  # optional dependency

        # One CFR iteration's average policy is uniform random over legal actions.
        table = solve_cfr(game, 1)
        return OpponentSpec("random", RandomOpponent(), table)
    if cfg.kind == "cfr":
        from sparjax.opponents.cfr import solve_cfr

        table = solve_cfr(game, cfg.cfr_iters, algorithm=cfg.cfr_algorithm)
        name = f"{cfg.cfr_algorithm}_{cfg.cfr_iters}"
        return OpponentSpec(name, TabularPolicyOpponent(table), table)
    if cfg.policy_path is None:
        raise ValueError("opponent.kind='policy' needs opponent.policy_path")
    table = TabularPolicy.load(cfg.policy_path)
    if table.game_name != game:
        raise ValueError(f"policy is for {table.game_name}, not {game}")
    return OpponentSpec(cfg.policy_path.stem, TabularPolicyOpponent(table), table)


def make_scripted_opponent(name: str) -> Opponent:
    """A scripted opponent by name (``random`` or ``go_light``)."""
    if name == "random":
        return RandomOpponent()
    if name == "go_light":
        return GoLightPlayoutOpponent()
    raise ValueError(f"unknown scripted opponent {name!r}")


def make_provider(
    spec: OpponentSpec, cfg: OpponentConfig | None = None, seed: int = 0
) -> OpponentProvider:
    """The provider for ``spec``: a snapshot pool for self-play, else a fixed opponent."""
    if spec.opponent is None:
        sp = (cfg or OpponentConfig(kind="selfplay")).selfplay
        return SnapshotPoolProvider(
            sampler=sp.sampler,
            max_size=sp.pool_size,
            latest_prob=sp.latest_prob,
            pfsp_power=sp.pfsp_power,
            seed=seed,
        )
    return FixedProvider(spec.opponent, spec.opponent_id)
