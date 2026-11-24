"""Evaluate a policy network by playing games against fixed opponents.

This works for any environment, unlike :mod:`sparjax.eval.exact`, which needs OpenSpiel's
game tree. Each game is played in a random seat. The report gives win/draw/loss rates
and the mean payoff with a 95% confidence interval, overall and per seat.
"""
# flax.nnx and jax.jit wrappers are only partially typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from flax import nnx

from sparjax.agents.ppo.networks import ActorCritic, MaskedCategorical, Params, apply
from sparjax.envs.batch import InProcessAECBatch
from sparjax.envs.loop import BatchedOpponentLoop
from sparjax.opponents.base import Opponent
from sparjax.opponents.providers import FixedProvider

__all__ = ["PlayReport", "play_games"]

_Z95 = 1.96


@dataclass(frozen=True, slots=True)
class PlayReport:
    """Results of :func:`play_games` (learner's perspective)."""

    games: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    mean_return: float
    ci95: float
    """Half-width of the 95% confidence interval of ``mean_return``."""
    win_rate_by_seat: tuple[float, float]
    games_by_seat: tuple[int, int]

    def as_metrics(self, prefix: str) -> dict[str, float]:
        """Flatten into ``{prefix/win_rate: ..., ...}`` for logging."""
        return {
            f"{prefix}/win_rate": self.win_rate,
            f"{prefix}/draw_rate": self.draw_rate,
            f"{prefix}/loss_rate": self.loss_rate,
            f"{prefix}/return_mean": self.mean_return,
            f"{prefix}/return_ci95": self.ci95,
            f"{prefix}/win_rate_seat0": self.win_rate_by_seat[0],
            f"{prefix}/win_rate_seat1": self.win_rate_by_seat[1],
            f"{prefix}/games": float(self.games),
        }


def play_games(
    env_fn: Any,
    graphdef: nnx.GraphDef[ActorCritic],
    params: Params,
    opponent: Opponent,
    num_games: int,
    *,
    num_envs: int = 32,
    greedy: bool = False,
    seed: int = 0,
) -> PlayReport:
    """Play at least ``num_games`` games of the network against ``opponent``.

    Args:
        env_fn: Factory for one environment.
        graphdef: Network structure.
        params: Network parameters.
        opponent: The fixed opponent.
        num_games: Minimum number of finished games to report on.
        num_envs: Games played in parallel.
        greedy: Pick the most likely action instead of sampling.
        seed: Seeds seats, deals and sampling.
    """
    if num_games < 1:
        raise ValueError("num_games must be >= 1")
    cpu = jax.devices("cpu")[0]
    params = jax.device_put(params, cpu)

    @jax.jit
    def act(p: Params, obs: jax.Array, mask: jax.Array, key: jax.Array) -> jax.Array:
        logits, _ = apply(graphdef, p, obs, mask)
        if greedy:
            return logits.argmax(axis=-1)
        return MaskedCategorical(logits, mask).sample(key)

    batch = InProcessAECBatch(env_fn, num_envs)
    loop = BatchedOpponentLoop(batch, FixedProvider(opponent, "eval"), seed=seed)
    key = jax.device_put(jax.random.key(seed), cpu)
    step = loop.reset()
    returns: list[float] = []
    seats: list[int] = []
    while len(returns) < num_games:
        key, sub = jax.random.split(key)
        obs = jax.device_put(step.observations, cpu)
        mask = jax.device_put(step.action_masks.astype(bool), cpu)
        step = loop.step(np.asarray(act(params, obs, mask, sub)))
        for ep in step.episodes:
            if ep.learner_decisions > 0:
                returns.append(ep.learner_return)
                seats.append(ep.learner_seat)
    batch.close()

    r = np.asarray(returns)
    s = np.asarray(seats)
    sign = np.sign(r)
    stderr = float(np.std(r, ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("inf")
    by_seat = tuple(
        float(np.mean(sign[s == k] > 0)) if (s == k).any() else float("nan") for k in (0, 1)
    )
    return PlayReport(
        games=len(r),
        win_rate=float(np.mean(sign > 0)),
        draw_rate=float(np.mean(sign == 0)),
        loss_rate=float(np.mean(sign < 0)),
        mean_return=float(np.mean(r)),
        ci95=_Z95 * stderr,
        win_rate_by_seat=(by_seat[0], by_seat[1]),
        games_by_seat=(int((s == 0).sum()), int((s == 1).sum())),
    )
