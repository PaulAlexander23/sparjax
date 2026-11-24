"""Opponents that play a frozen copy of a policy network."""
# flax.nnx and jax.jit wrappers are only partially typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from flax import nnx
from jaxtyping import Int

from sparjax.agents.ppo.networks import ActorCritic, MaskedCategorical, Params, apply
from sparjax.opponents.base import OpponentBatch

__all__ = ["NeuralOpponent", "NeuralPolicy"]


def _bucket(n: int) -> int:
    """Smallest power of two >= n (bounds the number of distinct jit shapes)."""
    return 1 << max(0, (n - 1).bit_length())


class NeuralPolicy:
    """A jitted act function for one network structure, shared by many snapshots.

    Opponent batches vary in size from call to call, and ``jax.jit`` compiles once per
    input shape. Batches are therefore padded to the next power of two, so at most
    ``log2(num_envs) + 1`` shapes are ever compiled.

    Args:
        graphdef: Network structure.
        device: Where inference runs (default: the first CPU device).
        greedy: Pick the most likely action instead of sampling.
    """

    def __init__(
        self, graphdef: nnx.GraphDef[ActorCritic], device: Any | None = None, greedy: bool = False
    ) -> None:
        """Compile-on-demand act function for ``graphdef``."""
        self.device: Any = device if device is not None else jax.devices("cpu")[0]
        self.greedy: bool = greedy

        def act(p: Params, obs: jax.Array, mask: jax.Array, key: jax.Array) -> jax.Array:
            logits, _ = apply(graphdef, p, obs, mask)
            if greedy:
                return logits.argmax(axis=-1)
            return MaskedCategorical(logits, mask).sample(key)

        self._act: Any = jax.jit(act)

    def place(self, params: Params) -> Params:
        """Put ``params`` on this policy's device."""
        return jax.device_put(params, self.device)

    def act(
        self, params: Params, batch: OpponentBatch, rng: np.random.Generator
    ) -> Int[np.ndarray, " batch"]:
        """Choose actions for ``batch`` with ``params``."""
        n = batch.size
        size = _bucket(n)
        obs = np.zeros((size, *batch.observations.shape[1:]), batch.observations.dtype)
        mask = np.ones((size, batch.action_masks.shape[1]), bool)
        obs[:n] = batch.observations
        mask[:n] = batch.action_masks.astype(bool)
        key = jax.random.key(int(rng.integers(0, 2**31 - 1)))
        out = self._act(
            params,
            jax.device_put(obs, self.device),
            jax.device_put(mask, self.device),
            jax.device_put(key, self.device),
        )
        return np.asarray(out)[:n].astype(np.int64)


class NeuralOpponent:
    """Plays fixed network parameters through a shared :class:`NeuralPolicy`."""

    def __init__(self, policy: NeuralPolicy, params: Params) -> None:
        """Freeze ``params`` (copied to the policy's device)."""
        self.policy: NeuralPolicy = policy
        self.params: Params = policy.place(params)

    def act(self, batch: OpponentBatch, rng: np.random.Generator) -> Int[np.ndarray, " batch"]:
        """Choose actions with the frozen parameters."""
        return self.policy.act(self.params, batch, rng)
