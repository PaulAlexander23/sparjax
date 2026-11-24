"""Self-play: opponents drawn from a pool of the learner's own past snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, override, runtime_checkable

import numpy as np
from flax import nnx

from sparjax.agents.ppo.networks import ActorCritic, Params
from sparjax.opponents.base import Opponent, OpponentId
from sparjax.opponents.neural import NeuralOpponent, NeuralPolicy
from sparjax.opponents.providers import StatsMixin

__all__ = ["Sampler", "SnapshotPoolProvider", "SnapshotSink"]

Sampler = Literal["latest", "uniform", "pfsp"]


@runtime_checkable
class SnapshotSink(Protocol):
    """Something the learner can hand parameter snapshots to (e.g. a snapshot pool)."""

    def add_snapshot(
        self, update: int, graphdef: nnx.GraphDef[ActorCritic], params: Params
    ) -> None:
        """Store the learner's parameters after ``update`` updates."""
        ...


@dataclass(frozen=True, slots=True)
class _Snapshot:
    opponent_id: OpponentId
    update: int
    opponent: NeuralOpponent


class SnapshotPoolProvider(StatsMixin):
    """Serves frozen past versions of the learner as opponents.

    Samplers:

    * ``latest``: always the newest snapshot (naive self-play).
    * ``uniform``: uniform over the pool, i.e. **fictitious self-play**: the learner best
      responds to the average of its past strategies.
    * ``pfsp``: prioritised FSP (AlphaStar). Snapshot ``i`` is drawn with weight
      ``(1 - w_i) ** power``, where ``w_i`` is the learner's win rate against it, so
      opponents the learner still loses to are played more.

    With ``latest_prob > 0``, the newest snapshot is used with that probability before
    falling back to the sampler (as in OpenAI Five's 80/20 mix).

    Args:
        sampler: How to choose among snapshots.
        max_size: Pool capacity; the oldest snapshot is evicted beyond it.
        latest_prob: Probability of playing the newest snapshot regardless of sampler.
        pfsp_power: Exponent of the PFSP weighting.
        greedy: Snapshots play their most likely action instead of sampling.
        seed: Seed for the provider's sampling.
    """

    def __init__(
        self,
        sampler: Sampler = "uniform",
        max_size: int = 1000,
        latest_prob: float = 0.0,
        pfsp_power: float = 2.0,
        greedy: bool = False,
        seed: int = 0,
    ) -> None:
        """Create an empty pool. The learner fills it via :meth:`add_snapshot`."""
        super().__init__()
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if not 0.0 <= latest_prob <= 1.0:
            raise ValueError("latest_prob must be in [0, 1]")
        self.sampler: Sampler = sampler
        self.max_size: int = max_size
        self.latest_prob: float = latest_prob
        self.pfsp_power: float = pfsp_power
        self.greedy: bool = greedy
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._pool: list[_Snapshot] = []
        self._policy: NeuralPolicy | None = None
        self._picks: dict[OpponentId, int] = {}

    # ------------------------------------------------------------------ learner side
    def add_snapshot(
        self, update: int, graphdef: nnx.GraphDef[ActorCritic], params: Params
    ) -> None:
        """Freeze ``params`` as a new opponent (evicting the oldest beyond capacity)."""
        with self._lock:
            if self._policy is None:
                self._policy = NeuralPolicy(graphdef, greedy=self.greedy)
            snapshot = _Snapshot(
                f"snapshot_{update:06d}", update, NeuralOpponent(self._policy, params)
            )
            self._pool.append(snapshot)
            if len(self._pool) > self.max_size:
                self._pool.pop(0)

    # ------------------------------------------------------------------ actor side
    def next_opponent(self, actor_id: int) -> tuple[OpponentId, Opponent]:
        """Sample a snapshot to play next."""
        del actor_id
        with self._lock:
            if not self._pool:
                raise RuntimeError("snapshot pool is empty; the learner adds the first one")
            if self.sampler == "latest" or self._rng.random() < self.latest_prob:
                chosen = self._pool[-1]
            elif self.sampler == "uniform":
                chosen = self._pool[int(self._rng.integers(len(self._pool)))]
            else:
                win = np.array(
                    [self._win_rate_locked(s.opponent_id) for s in self._pool], dtype=np.float64
                )
                weights = (1.0 - win) ** self.pfsp_power + 1e-3
                chosen = self._pool[
                    int(self._rng.choice(len(self._pool), p=weights / weights.sum()))
                ]
            self._picks[chosen.opponent_id] = self._picks.get(chosen.opponent_id, 0) + 1
            return chosen.opponent_id, chosen.opponent

    def _win_rate_locked(self, opponent_id: OpponentId) -> float:
        stats = self._stats.get(opponent_id)
        return stats.win_rate if stats is not None else 0.5

    # ------------------------------------------------------------------ introspection
    @property
    def size(self) -> int:
        """Number of snapshots in the pool."""
        with self._lock:
            return len(self._pool)

    def metrics(self) -> dict[str, float]:
        """Pool statistics for logging: size, newest snapshot, mean opponent age."""
        with self._lock:
            if not self._pool:
                return {"selfplay/pool_size": 0.0}
            newest = self._pool[-1].update
            updates = {s.opponent_id: s.update for s in self._pool}
            picked = [(updates[k], n) for k, n in self._picks.items() if k in updates]
            total = sum(n for _, n in picked)
            age = sum((newest - u) * n for u, n in picked) / total if total else 0.0
            self._picks.clear()
            return {
                "selfplay/pool_size": float(len(self._pool)),
                "selfplay/newest_update": float(newest),
                "selfplay/mean_opponent_age": float(age),
            }

    def snapshots(self) -> list[tuple[OpponentId, int, Params]]:
        """``(id, update, params)`` of every snapshot in the pool, oldest first."""
        with self._lock:
            return [(s.opponent_id, s.update, s.opponent.params) for s in self._pool]

    def snapshot_ids(self) -> list[OpponentId]:
        """Ids of the snapshots currently in the pool, oldest first."""
        with self._lock:
            return [s.opponent_id for s in self._pool]

    @override
    def __repr__(self) -> str:
        """Short description."""
        return f"SnapshotPoolProvider(sampler={self.sampler!r}, size={self.size})"
