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
import numpy as np
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
    """Actor-critic with an optional shared CNN torso and separate MLP heads.

    * Without ``conv_channels``: the observation is flattened and fed to two separate
      MLPs (policy and value), as in M4.
    * With ``conv_channels``: a shared stack of 3x3 ``SAME`` convolutions (ReLU) reads an
      ``HxWxC`` observation, and its flattened features feed the two MLP heads.
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        num_actions: int,
        hidden: Sequence[int],
        rngs: nnx.Rngs,
        conv_channels: Sequence[int] = (),
    ) -> None:
        """Build the torso and heads (small output init for the policy head)."""
        self.obs_shape: tuple[int, ...] = tuple(int(d) for d in obs_shape)
        conv: nnx.Sequential | None = None
        if conv_channels:
            if len(self.obs_shape) != 3:
                raise ValueError(f"a CNN torso needs H x W x C observations, got {self.obs_shape}")
            layers: list[Any] = []
            channels = self.obs_shape[-1]
            for c in conv_channels:
                layers += [
                    nnx.Conv(
                        channels,
                        c,
                        kernel_size=(3, 3),
                        padding="SAME",
                        kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                        rngs=rngs,
                    ),
                    jax.nn.relu,
                ]
                channels = c
            conv = nnx.Sequential(*layers)
            features = self.obs_shape[0] * self.obs_shape[1] * channels
        else:
            features = int(np.prod(self.obs_shape))
        # Assign once, explicitly as data, so nnx treats it as a sub-module either way.
        self.conv: nnx.Sequential | None = nnx.data(conv)
        self.actor: nnx.Sequential = _mlp(features, hidden, num_actions, rngs, out_scale=0.01)
        self.critic: nnx.Sequential = _mlp(features, hidden, 1, rngs, out_scale=1.0)

    @override
    def __call__(
        self, obs: Float[Array, "*batch ..."], mask: Bool[Array, "*batch num_actions"]
    ) -> tuple[Float[Array, "*batch num_actions"], Float[Array, "*batch"]]:
        """Return masked logits (illegal actions at -1e9) and values."""
        batch = obs.shape[: obs.ndim - len(self.obs_shape)]
        x = obs.astype(jnp.float32)
        if self.conv is not None:
            x = self.conv(x.reshape(-1, *self.obs_shape))
        x = x.reshape(*batch, -1)
        logits = jnp.where(mask, self.actor(x), _NEG_INF)
        value = self.critic(x)[..., 0]
        return logits, value


def init_actor_critic(
    obs_shape: int | Sequence[int],
    num_actions: int,
    hidden: Sequence[int],
    seed: int,
    conv_channels: Sequence[int] = (),
) -> tuple[nnx.GraphDef[ActorCritic], Params]:
    """Create a network and split it into ``(graphdef, params)``."""
    shape = (obs_shape,) if isinstance(obs_shape, int) else tuple(obs_shape)
    model = ActorCritic(shape, num_actions, hidden, nnx.Rngs(seed), conv_channels)
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
