"""Simple opponents: uniform random and user-scripted."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from jaxtyping import Float, Int

from sparjax.opponents.base import OpponentBatch, sample_actions

__all__ = ["RandomOpponent", "ScriptedOpponent"]


class RandomOpponent:
    """Picks uniformly among legal actions."""

    def act(self, batch: OpponentBatch, rng: np.random.Generator) -> Int[np.ndarray, " batch"]:
        """Return a uniformly random legal action for each game."""
        return sample_actions(batch.action_masks.astype(np.float64), batch.action_masks, rng)


ScriptFn = Callable[[OpponentBatch], Float[np.ndarray, "batch num_actions"]]


class ScriptedOpponent:
    """Wraps a function that maps a batch to action weights.

    The weights are masked to legal actions and sampled, so scripts can be written as
    simple preferences (e.g. "always weight action 1") without worrying about legality.

    Example:
        ```python
        always_call = ScriptedOpponent(lambda b: np.tile([0.0, 1.0, 0.0], (b.size, 1)))
        ```
    """

    def __init__(self, script: ScriptFn) -> None:
        """Create an opponent from ``script``."""
        self._script: ScriptFn = script

    def act(self, batch: OpponentBatch, rng: np.random.Generator) -> Int[np.ndarray, " batch"]:
        """Evaluate the script and sample a legal action per game."""
        weights = np.asarray(self._script(batch), dtype=np.float64)
        if weights.shape != batch.action_masks.shape:
            msg = f"script returned shape {weights.shape}, expected {batch.action_masks.shape}"
            raise ValueError(msg)
        if (weights < 0).any():
            raise ValueError("script returned negative action weights")
        return sample_actions(weights, batch.action_masks, rng)
