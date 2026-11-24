"""GAE, the clipped PPO loss and the jitted learner update."""
# flax.nnx, optax and jax.lax are only partially typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false, reportMissingTypeStubs=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jaxtyping import Array, Bool, Float, Int

from sparjax.agents.ppo.networks import ActorCritic, MaskedCategorical, Params, apply

AnyArray = jax.Array | np.ndarray

__all__ = ["AnyArray", "Batch", "PPOHyperparams", "compute_gae", "make_update_fn", "ppo_loss"]


@dataclass(frozen=True)
class PPOHyperparams:
    """Hyper-parameters of the PPO update (static under ``jit``)."""

    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 4
    num_minibatches: int = 4
    normalize_advantages: bool = True


class Batch(NamedTuple):
    """A rollout batch, time-major: arrays are ``[T, N, ...]`` (``last_value`` is ``[N]``)."""

    obs: Float[AnyArray, "T N obs_dim"]
    mask: Bool[AnyArray, "T N num_actions"]
    actions: Int[AnyArray, "T N"]
    log_probs: Float[AnyArray, "T N"]
    values: Float[AnyArray, "T N"]
    rewards: Float[AnyArray, "T N"]
    dones: Bool[AnyArray, "T N"]
    last_value: Float[AnyArray, " N"]


def compute_gae(
    rewards: Float[AnyArray, "T N"],
    values: Float[AnyArray, "T N"],
    dones: Bool[AnyArray, "T N"],
    last_value: Float[AnyArray, " N"],
    gamma: float,
    gae_lambda: float,
) -> tuple[Float[Array, "T N"], Float[Array, "T N"]]:
    """Generalised advantage estimation.

    ``dones[t]`` means the game ended after action ``t``, so nothing is bootstrapped past
    it. Truncations are currently treated as terminations (the poker envs never truncate).

    Returns:
        ``(advantages, returns)``, both ``[T, N]``.
    """
    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)
    rewards, values, last_value = jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(last_value)
    not_done = 1.0 - jnp.asarray(dones).astype(values.dtype)
    deltas = rewards + gamma * next_values * not_done - values

    def body(carry: Float[Array, " N"], x: tuple[Array, Array]) -> tuple[Array, Array]:
        delta, nd = x
        adv = delta + gamma * gae_lambda * nd * carry
        return adv, adv

    _, advantages = jax.lax.scan(body, jnp.zeros_like(last_value), (deltas, not_done), reverse=True)
    return advantages, advantages + values


def ppo_loss(
    graphdef: nnx.GraphDef[ActorCritic],
    params: Params,
    obs: Float[Array, "B obs_dim"],
    mask: Bool[Array, "B num_actions"],
    actions: Int[Array, " B"],
    old_log_probs: Float[Array, " B"],
    advantages: Float[Array, " B"],
    returns: Float[Array, " B"],
    hp: PPOHyperparams,
    ent_coef: float | Float[Array, ""] | None = None,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Clipped-surrogate PPO loss with value and entropy terms.

    ``ent_coef`` overrides ``hp.ent_coef`` (used for entropy schedules).
    """
    ent_coef = hp.ent_coef if ent_coef is None else ent_coef
    logits, values = apply(graphdef, params, obs, mask)
    dist = MaskedCategorical(logits, mask)
    log_probs = dist.log_prob(actions)
    log_ratio = log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    if hp.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = -jnp.minimum(
        ratio * advantages, jnp.clip(ratio, 1.0 - hp.clip_eps, 1.0 + hp.clip_eps) * advantages
    ).mean()
    v_loss = 0.5 * jnp.square(values - returns).mean()
    entropy = dist.entropy().mean()
    total = pg_loss + hp.vf_coef * v_loss - ent_coef * entropy
    metrics = {
        "loss/policy": pg_loss,
        "loss/value": v_loss,
        "loss/entropy": entropy,
        "loss/total": total,
        "ppo/approx_kl": ((ratio - 1.0) - log_ratio).mean(),
        "ppo/clip_frac": (jnp.abs(ratio - 1.0) > hp.clip_eps).mean(),
    }
    return total, metrics


def make_update_fn(
    graphdef: nnx.GraphDef[ActorCritic], optimizer: optax.GradientTransformation, hp: PPOHyperparams
) -> Any:
    """Build ``update(params, opt_state, batch, key, ent_coef) -> (params, opt_state, metrics)``.

    ``ent_coef`` is a traced scalar, so an entropy schedule does not trigger recompilation.

    One call runs ``epochs`` passes over the flattened ``T * N`` samples in
    ``num_minibatches`` shuffled minibatches. The returned function is not yet jitted.
    """

    def update(
        params: Params,
        opt_state: Any,
        batch: Batch,
        key: jax.Array,
        ent_coef: float | jax.Array = hp.ent_coef,
    ) -> tuple[Params, Any, dict[str, Array]]:
        advantages, returns = compute_gae(
            batch.rewards, batch.values, batch.dones, batch.last_value, hp.gamma, hp.gae_lambda
        )
        flat = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            (batch.obs, batch.mask, batch.actions, batch.log_probs, advantages, returns),
        )
        n = flat[0].shape[0]
        if n % hp.num_minibatches:
            raise ValueError(
                f"batch size {n} not divisible by num_minibatches {hp.num_minibatches}"
            )
        mb = n // hp.num_minibatches

        def minibatch_step(carry: tuple[Params, Any], idx: Array) -> Any:
            p, o = carry
            mb_data = jax.tree.map(lambda x: x[idx], flat)
            grads, metrics = jax.grad(ppo_loss, argnums=1, has_aux=True)(
                graphdef, p, *mb_data, hp, ent_coef
            )
            updates, o = optimizer.update(grads, o, p)
            p = optax.apply_updates(p, updates)
            metrics["opt/grad_norm"] = optax.tree.norm(grads)
            return (p, o), metrics

        def epoch_step(
            carry: tuple[Params, Any], k: Array
        ) -> tuple[tuple[Params, Any], dict[str, Array]]:
            perm = jax.random.permutation(k, n).reshape(hp.num_minibatches, mb)
            return jax.lax.scan(minibatch_step, carry, perm)

        (params, opt_state), metrics = jax.lax.scan(
            epoch_step, (params, opt_state), jax.random.split(key, hp.epochs)
        )
        metrics = jax.tree.map(jnp.mean, metrics)
        metrics["value/explained_variance"] = 1.0 - jnp.var(returns - batch.values) / (
            jnp.var(returns) + 1e-8
        )
        metrics["value/mean"] = batch.values.mean()
        metrics["ppo/ent_coef"] = jnp.asarray(ent_coef, jnp.float32)
        return params, opt_state, metrics

    return update
