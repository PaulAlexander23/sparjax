"""Opponent and opponent-provider protocols.

An :class:`Opponent` chooses actions for a batch of games in which it is to move. An
:class:`OpponentProvider` decides *which* opponent an actor plays next and collects match
results (needed later for e.g. prioritised fictitious self-play).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from jaxtyping import Float, Int, Int8

__all__ = ["Opponent", "OpponentBatch", "OpponentId", "OpponentProvider", "sample_actions"]

OpponentId = str


@dataclass(frozen=True, slots=True)
class OpponentBatch:
    """Everything an opponent may look at for ``B`` games where it is to move.

    Attributes:
        observations: Observation tensors from the opponent's perspective.
        action_masks: 1 for legal actions, 0 otherwise.
        info_states: Optional per-game information-state keys (used by tabular policies).
    """

    observations: Float[np.ndarray, "batch obs_dim"]
    action_masks: Int8[np.ndarray, "batch num_actions"]
    info_states: Sequence[str] | None = None

    def __post_init__(self) -> None:
        """Validate that all fields describe the same number of games."""
        n = self.action_masks.shape[0]
        if self.observations.shape[0] != n:
            raise ValueError("observations and action_masks have different batch sizes")
        if self.info_states is not None and len(self.info_states) != n:
            raise ValueError("info_states and action_masks have different batch sizes")
        if n and not self.action_masks.any(axis=1).all():
            raise ValueError("every game in the batch needs at least one legal action")

    @property
    def size(self) -> int:
        """Number of games in the batch."""
        return int(self.action_masks.shape[0])


class Opponent(Protocol):
    """A policy that picks actions for the games in which it is to move."""

    def act(self, batch: OpponentBatch, rng: np.random.Generator) -> Int[np.ndarray, " batch"]:
        """Return one legal action per game in ``batch``."""
        ...


class OpponentProvider(Protocol):
    """Chooses the next opponent for an actor and records match outcomes."""

    def next_opponent(self, actor_id: int) -> tuple[OpponentId, Opponent]:
        """Return the opponent (and its id) that ``actor_id`` should play next."""
        ...

    def report(self, opponent_id: OpponentId, outcome: float) -> None:
        """Record the learner's payoff from one finished game against ``opponent_id``."""
        ...


def sample_actions(
    probs: Float[np.ndarray, "batch num_actions"],
    action_masks: Int8[np.ndarray, "batch num_actions"],
    rng: np.random.Generator,
) -> Int[np.ndarray, " batch"]:
    """Sample one action per row from ``probs`` restricted to legal actions.

    Probabilities on illegal actions are zeroed and each row is renormalised. A row whose
    legal actions all have zero probability falls back to uniform over legal actions.

    Args:
        probs: Non-negative (unnormalised is fine) action weights.
        action_masks: 1 for legal actions, 0 otherwise.
        rng: Random generator.

    Returns:
        Integer actions, shape ``(batch,)``.
    """
    masked = np.where(action_masks.astype(bool), probs, 0.0).astype(np.float64)
    totals = masked.sum(axis=1, keepdims=True)
    uniform = action_masks.astype(np.float64)
    masked = np.where(totals > 0, masked, uniform)
    masked /= masked.sum(axis=1, keepdims=True)
    cdf = np.cumsum(masked, axis=1)
    u = rng.random((masked.shape[0], 1))
    actions = (u >= cdf).sum(axis=1)
    # Guard against floating-point round-off at the top of the CDF.
    last_legal = masked.shape[1] - 1 - np.argmax(masked[:, ::-1] > 0, axis=1)
    return np.minimum(actions, last_legal).astype(np.int64)
