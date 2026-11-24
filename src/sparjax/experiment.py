"""Build environments and opponents from a :class:`~sparjax.config.TrainConfig`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sparjax.config import OpponentConfig, TrainConfig
from sparjax.envs.openspiel_aec import OpenSpielAECEnv
from sparjax.opponents.base import Opponent
from sparjax.opponents.providers import FixedProvider
from sparjax.opponents.simple import RandomOpponent
from sparjax.opponents.tabular import TabularPolicy, TabularPolicyOpponent

__all__ = ["OpponentSpec", "make_env_fn", "make_opponent"]


@dataclass(frozen=True)
class OpponentSpec:
    """A constructed opponent plus its tabular form when it has one (for exact evaluation)."""

    opponent_id: str
    opponent: Opponent
    table: TabularPolicy


def make_env_fn(cfg: TrainConfig) -> Callable[[], Any]:
    """Factory for one environment of ``cfg.env.game``."""
    game = cfg.env.game
    return lambda: OpenSpielAECEnv(game)


def make_opponent(cfg: OpponentConfig, game: str) -> OpponentSpec:
    """Build the opponent described by ``cfg`` for ``game``.

    Every supported opponent is tabular (random play is the empty table, whose missing
    states default to uniform), so the exact best-response ceiling is always computable.
    """
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


def make_provider(spec: OpponentSpec) -> FixedProvider:
    """A provider that always serves ``spec``'s opponent."""
    return FixedProvider(spec.opponent, spec.opponent_id)
