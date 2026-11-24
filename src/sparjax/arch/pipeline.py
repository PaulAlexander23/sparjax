"""Rollout hand-off from actor threads to the learner.

Follows Stoix's ``OnPolicyPipeline`` (Apache-2.0, see ``NOTICE``): one bounded queue per
actor, and the learner waits for one rollout from *every* actor before an update. That
keeps each update's batch balanced across actors and stops a fast actor from flooding
the learner.
"""

from __future__ import annotations

import queue
from typing import Any

from sparjax.arch.lifetime import Lifetime

__all__ = ["OnPolicyPipeline"]


class OnPolicyPipeline:
    """Per-actor bounded queues of rollouts."""

    def __init__(self, num_actors: int, lifetime: Lifetime, maxsize: int = 1) -> None:
        """Create ``num_actors`` queues holding up to ``maxsize`` rollouts each."""
        if num_actors < 1:
            raise ValueError("num_actors must be >= 1")
        self._queues: list[queue.Queue[Any]] = [
            queue.Queue(maxsize=maxsize) for _ in range(num_actors)
        ]
        self._lifetime: Lifetime = lifetime

    @property
    def num_actors(self) -> int:
        """Number of actor queues."""
        return len(self._queues)

    def send(self, actor_id: int, rollout: Any, poll: float = 0.1) -> None:
        """Hand a rollout to the learner, blocking while this actor's queue is full.

        Raises:
            StopRequestedError: If shutdown is requested while waiting.
        """
        q = self._queues[actor_id]
        while True:
            self._lifetime.check()
            try:
                q.put(rollout, timeout=poll)
            except queue.Full:
                continue
            return

    def collect(self, poll: float = 0.1) -> list[Any]:
        """Return one rollout from each actor (in actor order), blocking until all arrive.

        Raises:
            StopRequestedError: If shutdown is requested while waiting.
        """
        out: list[Any] = []
        for q in self._queues:
            while True:
                self._lifetime.check()
                try:
                    out.append(q.get(timeout=poll))
                except queue.Empty:
                    continue
                break
        return out
