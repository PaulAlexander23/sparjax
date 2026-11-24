"""Masked actor-critic network (flax.nnx) and pure functions over its parameters.

Actors and the learner never hold the nnx module. They hold its parameter
:class:`~flax.nnx.State` (a pytree) and call :func:`apply` with the static
:class:`~flax.nnx.GraphDef`. That makes parameters easy to ship between threads and devices.
"""
# flax.nnx, optax and jax.lax are only partially typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false, reportMissingTypeStubs=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int

__all__ = ["ActorCritic", "MaskedCategorical", "Params", "apply", "init_actor_critic"]

Params = nnx.State[Any, Any]
"""Network parameters: the ``nnx.State`` pytree of an :class:`ActorCritic`."""

_NEG_INF = -1e9


def _mlp(
    in_dim: int, hidden: Sequence[int], out_dim: int, rngs: nnx.Rngs, out_scale: float
) -> nnx.Sequential:
    layers: list[Any] = []
    dim = in_dim
    for h in hidden:
        layers += [
            nnx.Linear(dim, h, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)), rngs=rngs),
            jax.nn.tanh,
        ]
        dim = h
    layers.append(
        nnx.Linear(dim, out_dim, kernel_init=nnx.initializers.orthogonal(out_scale), rngs=rngs)
    )
    return nnx.Sequential(*layers)


class ActorCritic(nnx.Module):
    """Separate MLP torsos for the policy logits and the state value."""

    def __init__(
        self, obs_dim: int, num_actions: int, hidden: Sequence[int], rngs: nnx.Rngs
    ) -> None:
        """Build the actor (small output init) and critic MLPs."""
        self.actor: nnx.Sequential = _mlp(obs_dim, hidden, num_actions, rngs, out_scale=0.01)
        self.critic: nnx.Sequential = _mlp(obs_dim, hidden, 1, rngs, out_scale=1.0)

    @override
    def __call__(
        self, obs: Float[Array, "*batch obs_dim"], mask: Bool[Array, "*batch num_actions"]
    ) -> tuple[Float[Array, "*batch num_actions"], Float[Array, "*batch"]]:
        """Return masked logits (illegal actions at -1e9) and values."""
        logits = jnp.where(mask, self.actor(obs), _NEG_INF)
        value = self.critic(obs)[..., 0]
        return logits, value


def init_actor_critic(
    obs_dim: int, num_actions: int, hidden: Sequence[int], seed: int
) -> tuple[nnx.GraphDef[ActorCritic], Params]:
    """Create a network and split it into ``(graphdef, params)``."""
    model = ActorCritic(obs_dim, num_actions, hidden, nnx.Rngs(seed))
    graphdef, params = nnx.split(model)
    return graphdef, params


def apply(
    graphdef: nnx.GraphDef[ActorCritic],
    params: Params,
    obs: Float[Array, "*batch obs_dim"],
    mask: Bool[Array, "*batch num_actions"],
) -> tuple[Float[Array, "*batch num_actions"], Float[Array, "*batch"]]:
    """Pure forward pass: ``(masked_logits, values)``."""
    model = nnx.merge(graphdef, params)
    return model(obs, mask)


class MaskedCategorical:
    """Categorical distribution over masked logits (illegal actions get zero mass)."""

    def __init__(
        self, logits: Float[Array, "*batch num_actions"], mask: Bool[Array, "*batch num_actions"]
    ) -> None:
        """Wrap already-masked logits and the legal-action mask."""
        self.mask: Bool[Array, "*batch num_actions"] = mask
        self.log_probs_all: Float[Array, "*batch num_actions"] = jax.nn.log_softmax(logits, axis=-1)

    def sample(self, key: jax.Array) -> Int[Array, "*batch"]:
        """Draw one action per row."""
        return jax.random.categorical(key, self.log_probs_all, axis=-1)

    def log_prob(self, actions: Int[Array, "*batch"]) -> Float[Array, "*batch"]:
        """Log-probability of ``actions``."""
        return jnp.take_along_axis(self.log_probs_all, actions[..., None], axis=-1)[..., 0]

    def entropy(self) -> Float[Array, "*batch"]:
        """Entropy over legal actions only."""
        p = jnp.exp(self.log_probs_all)
        return -jnp.sum(jnp.where(self.mask, p * self.log_probs_all, 0.0), axis=-1)
