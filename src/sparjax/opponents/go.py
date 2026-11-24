"""Scripted opponents for PettingZoo's Go (``classic/go-v5``)."""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Int

from sparjax.opponents.base import OpponentBatch, sample_actions

__all__ = ["GoLightPlayoutOpponent", "own_eye_points"]

# In PettingZoo's Go observation, from the observer's point of view, plane 0 holds the
# stones of the player who just moved (the opponent) and plane 1 the observer's own.
_OPPONENT_PLANE = 0
_OWN_PLANE = 1


def own_eye_points(
    own: Bool[np.ndarray, "batch n n"], opp: Bool[np.ndarray, "batch n n"]
) -> Bool[np.ndarray, "batch n n"]:
    """Empty points whose on-board orthogonal neighbours are all the observer's stones."""
    empty = ~(own | opp)
    padded = np.pad(own, ((0, 0), (1, 1), (1, 1)), constant_values=True)  # edges count as own
    neighbours_own = (
        padded[:, :-2, 1:-1] & padded[:, 2:, 1:-1] & padded[:, 1:-1, :-2] & padded[:, 1:-1, 2:]
    )
    return empty & neighbours_own


class GoLightPlayoutOpponent:
    """The classic "light playout" policy used inside Go engines' simulations.

    Plays uniformly at random among legal moves, except it never fills its own
    single-point eyes and never passes unless no other move is legal. Uniform random
    play fills its own eyes and loses its groups, so this is a much stronger baseline.
    """

    def act(self, batch: OpponentBatch, rng: np.random.Generator) -> Int[np.ndarray, " batch"]:
        """Pick one move per game."""
        obs = np.asarray(batch.observations).astype(bool)
        if obs.ndim != 4 or obs.shape[1] != obs.shape[2]:
            raise ValueError(f"expected Go observations [batch, n, n, planes], got {obs.shape}")
        n = obs.shape[1]
        masks = batch.action_masks.astype(bool)
        eyes = own_eye_points(obs[..., _OWN_PLANE], obs[..., _OPPONENT_PLANE]).reshape(-1, n * n)
        preferred = masks.copy()
        preferred[:, : n * n] &= ~eyes
        preferred[:, n * n :] = False  # pass
        has_move = preferred.any(axis=1, keepdims=True)
        weights = np.where(has_move, preferred, masks).astype(np.float64)
        return sample_actions(weights, batch.action_masks, rng)
