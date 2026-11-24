"""Opponent providers: who does the learner play next?"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sparjax.opponents.base import Opponent, OpponentId

__all__ = ["FixedProvider", "MixtureEntry", "MixtureProvider", "OutcomeStats", "StatsMixin"]


@dataclass(slots=True)
class OutcomeStats:
    """Running payoff statistics against one opponent (learner's perspective)."""

    games: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    wins: int = 0
    """Games with a strictly positive payoff for the learner."""

    @property
    def win_rate(self) -> float:
        """Fraction of games the learner won (0.5 if no games yet)."""
        return self.wins / self.games if self.games else 0.5

    @property
    def mean(self) -> float:
        """Mean payoff per game (0 if no games)."""
        return self.total / self.games if self.games else 0.0

    @property
    def stderr(self) -> float:
        """Standard error of the mean payoff (``inf`` with fewer than two games)."""
        if self.games < 2:
            return float("inf")
        var = (self.total_sq - self.games * self.mean**2) / (self.games - 1)
        return float(np.sqrt(max(var, 0.0) / self.games))

    def add(self, outcome: float) -> None:
        """Record one game's payoff."""
        self.games += 1
        self.total += outcome
        self.total_sq += outcome * outcome
        self.wins += int(outcome > 0)


class StatsMixin:
    """Thread-safe outcome bookkeeping shared by providers."""

    def __init__(self) -> None:
        """Start with no recorded games."""
        self._lock: threading.Lock = threading.Lock()
        self._stats: dict[OpponentId, OutcomeStats] = {}

    def report(self, opponent_id: OpponentId, outcome: float) -> None:
        """Record the learner's payoff from one game against ``opponent_id``."""
        with self._lock:
            self._stats.setdefault(opponent_id, OutcomeStats()).add(outcome)

    def stats(self) -> dict[OpponentId, OutcomeStats]:
        """Snapshot of the per-opponent statistics."""
        with self._lock:
            return {
                k: OutcomeStats(v.games, v.total, v.total_sq, v.wins)
                for k, v in self._stats.items()
            }


class FixedProvider(StatsMixin):
    """Always returns the same opponent."""

    def __init__(self, opponent: Opponent, opponent_id: OpponentId = "fixed") -> None:
        """Create a provider for a single opponent."""
        super().__init__()
        self._opponent: Opponent = opponent
        self._id: OpponentId = opponent_id

    def next_opponent(self, actor_id: int) -> tuple[OpponentId, Opponent]:
        """Return the fixed opponent."""
        del actor_id
        return self._id, self._opponent


@dataclass(frozen=True, slots=True)
class MixtureEntry:
    """One component of a :class:`MixtureProvider`."""

    opponent_id: OpponentId
    opponent: Opponent
    weight: float


class MixtureProvider(StatsMixin):
    """Samples an opponent from a fixed weighted list each time it is asked.

    Args:
        entries: Opponents and their (unnormalised, positive) weights.
        seed: Seed for the provider's own random generator.
    """

    def __init__(self, entries: Sequence[MixtureEntry], seed: int | None = None) -> None:
        """Create the provider."""
        super().__init__()
        if not entries:
            raise ValueError("MixtureProvider needs at least one entry")
        ids = [e.opponent_id for e in entries]
        if len(set(ids)) != len(ids):
            raise ValueError("opponent ids must be unique")
        weights = np.array([e.weight for e in entries], dtype=np.float64)
        if (weights <= 0).any():
            raise ValueError("weights must be positive")
        self._entries: list[MixtureEntry] = list(entries)
        self._p: np.ndarray = weights / weights.sum()
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def next_opponent(self, actor_id: int) -> tuple[OpponentId, Opponent]:
        """Sample an opponent according to the weights."""
        del actor_id
        with self._lock:
            idx = int(self._rng.choice(len(self._entries), p=self._p))
        entry = self._entries[idx]
        return entry.opponent_id, entry.opponent
