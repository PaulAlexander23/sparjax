"""Solve games with CFR and evaluate tabular policies exactly, via OpenSpiel.

Requires the ``openspiel`` extra. ``pyspiel`` ships without type stubs, so the
unknown-type checks are relaxed for this file only.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportMissingTypeArgument=false

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pyspiel
from open_spiel.python import policy as os_policy
from open_spiel.python.algorithms import exploitability as os_exploitability

from sparjax.opponents.tabular import TabularPolicy

__all__ = [
    "BestResponseReport",
    "CFRAlgorithm",
    "best_response_report",
    "solve_cfr",
    "to_openspiel_policy",
]

CFRAlgorithm = Literal["cfr", "cfr_plus"]

_SOLVERS: dict[str, Any] = {
    "cfr": pyspiel.CFRSolver,
    "cfr_plus": pyspiel.CFRPlusSolver,
}


def _load_game(game_name: str, game_params: Mapping[str, Any] | None) -> Any:
    return pyspiel.load_game(game_name, dict(game_params or {}))


def solve_cfr(
    game_name: str,
    iterations: int,
    algorithm: CFRAlgorithm = "cfr_plus",
    game_params: Mapping[str, Any] | None = None,
) -> TabularPolicy:
    """Run CFR on an OpenSpiel game and return its **average** policy as a table.

    The average policy is the one that converges to a Nash equilibrium. Fewer iterations
    give a more exploitable policy, which is useful as a training opponent.

    Args:
        game_name: OpenSpiel short name, e.g. ``"leduc_poker"``.
        iterations: Number of CFR iterations (``>= 1``).
        algorithm: ``"cfr"`` (vanilla) or ``"cfr_plus"``.
        game_params: Extra OpenSpiel game parameters.

    Returns:
        A policy covering every information state of every player.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    game = _load_game(game_name, game_params)
    solver = _SOLVERS[algorithm](game)
    start = time.perf_counter()
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    elapsed = time.perf_counter() - start
    average = solver.average_policy()

    num_actions = int(game.num_distinct_actions())
    template = os_policy.TabularPolicy(game)
    probs: dict[str, np.ndarray] = {}
    for key, state in zip(template.state_lookup, template.states, strict=True):
        row = np.zeros(num_actions, dtype=np.float64)
        for action, p in average.get_state_policy(state):
            row[int(action)] = float(p)
        probs[str(key)] = row / row.sum()
    return TabularPolicy(
        game_name=game_name,
        num_actions=num_actions,
        probs=probs,
        metadata={
            "solver": f"open_spiel.{algorithm}",
            "iterations": iterations,
            "game_params": dict(game_params or {}),
            "solve_seconds": round(elapsed, 3),
        },
    )


def to_openspiel_policy(policy: TabularPolicy, game: Any | None = None) -> Any:
    """Convert a :class:`TabularPolicy` into an OpenSpiel ``TabularPolicy``.

    Information states missing from ``policy`` get the uniform distribution over legal
    actions (this is what lets partially-specified policies be evaluated).
    """
    if game is None:
        game = _load_game(policy.game_name, policy.metadata.get("game_params"))
    out = os_policy.TabularPolicy(game)
    for key, idx in out.state_lookup.items():
        row = policy.probs.get(str(key))
        if row is not None:
            out.action_probability_array[idx] = row
    return out


@dataclass(frozen=True, slots=True)
class BestResponseReport:
    """Exact best-response analysis of a fixed policy in a two-player zero-sum game.

    ``best_response_value[p]`` is the most a player in seat ``p`` can win per game against
    the policy in the other seat. ``on_policy_value[p]`` is what seat ``p`` gets when the
    policy plays itself.

    With seats assigned uniformly at random, the best achievable mean payoff against the
    policy is ``mean(best_response_value)``, which equals its exploitability.
    """

    best_response_value: tuple[float, float]
    on_policy_value: tuple[float, float]
    exploitability: float
    nash_conv: float

    @property
    def mean_best_response(self) -> float:
        """Best achievable payoff with random seat assignment."""
        return float(np.mean(self.best_response_value))


def best_response_report(policy: TabularPolicy) -> BestResponseReport:
    """Compute exact best-response values and exploitability of ``policy``."""
    game = _load_game(policy.game_name, policy.metadata.get("game_params"))
    if game.num_players() != 2:
        raise ValueError("best_response_report supports two-player games only")
    os_pol = to_openspiel_policy(policy, game)
    br: list[dict[str, Any]] = [os_exploitability.best_response(game, os_pol, p) for p in range(2)]
    nash_conv = float(cast(Any, os_exploitability.nash_conv(game, os_pol)))
    return BestResponseReport(
        best_response_value=(
            float(br[0]["best_response_value"]),
            float(br[1]["best_response_value"]),
        ),
        on_policy_value=(float(br[0]["on_policy_value"]), float(br[1]["on_policy_value"])),
        exploitability=nash_conv / 2,
        nash_conv=nash_conv,
    )
