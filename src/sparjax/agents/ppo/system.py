"""Sebulba PPO: actor threads play against opponents, one learner thread runs PPO.

Data flow::

    ParameterServer ──params──► Actor threads (BatchedOpponentLoop + jitted policy)
          ▲                              │ rollouts
          │                              ▼
          └──── publish ◄──── Learner ◄── OnPolicyPipeline

The learner runs in the calling thread. Each actor thread owns its own batch of AEC
environments and a :class:`~sparjax.envs.loop.BatchedOpponentLoop`, and asks the opponent
provider for a new opponent at the start of each rollout.
"""
# flax.nnx, optax and jax.lax are only partially typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false, reportMissingTypeStubs=false

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
import optax
from flax import nnx

from sparjax.agents.ppo.loss import Batch, make_update_fn
from sparjax.agents.ppo.networks import (
    ActorCritic,
    MaskedCategorical,
    Params,
    apply,
    init_actor_critic,
)
from sparjax.arch import (
    Lifetime,
    OnPolicyPipeline,
    ParameterServer,
    StopRequestedError,
    resolve_device,
)
from sparjax.config import TrainConfig
from sparjax.envs.batch import InProcessAECBatch
from sparjax.envs.loop import BatchedOpponentLoop
from sparjax.loggers import Logger, NullLogger
from sparjax.opponents.base import OpponentProvider

__all__ = ["Rollout", "TrainResult", "train"]

EnvFn = Callable[[], Any]


@dataclass(frozen=True)
class Rollout:
    """What an actor sends to the learner for one update."""

    actor_id: int
    param_version: int
    batch: Batch  # numpy arrays, time-major
    episode_returns: list[float]
    episode_opponents: list[str]


@dataclass
class TrainResult:
    """Outcome of :func:`train`."""

    graphdef: nnx.GraphDef[ActorCritic]
    params: Params
    history: list[dict[str, float]] = field(default_factory=list[dict[str, float]])
    episode_returns: list[float] = field(default_factory=list[float])
    """Every finished game's learner payoff, in learner-update order."""


def _make_act_fn(graphdef: nnx.GraphDef[ActorCritic]) -> Any:
    def act(
        params: Params, obs: jax.Array, mask: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        logits, values = apply(graphdef, params, obs, mask)
        dist = MaskedCategorical(logits, mask)
        actions = dist.sample(key)
        return actions, dist.log_prob(actions), values

    return jax.jit(act)


def _actor(
    actor_id: int,
    cfg: TrainConfig,
    env_fn: EnvFn,
    provider: OpponentProvider,
    act_fn: Any,
    server: ParameterServer,
    pipeline: OnPolicyPipeline,
    lifetime: Lifetime,
) -> None:
    device = resolve_device(cfg.arch.actor_device)
    batch_env = InProcessAECBatch(env_fn, cfg.env.num_envs)
    loop = BatchedOpponentLoop(
        batch_env, provider, actor_id=actor_id, seed=cfg.seed * 1000 + actor_id
    )
    key = jax.device_put(jax.random.key(cfg.seed * 1000 + actor_id + 1), device)
    loop.next_opponent()
    step = loop.reset()
    T, N = cfg.ppo.rollout_len, cfg.env.num_envs
    obs_shape, A = batch_env.obs_shape, batch_env.num_actions

    for r in range(cfg.num_updates):
        params, version = server.get(device, min_version=max(0, r - cfg.arch.max_lag))
        loop.next_opponent()
        obs = np.zeros((T, N, *obs_shape), np.float32)
        mask = np.zeros((T, N, A), bool)
        actions = np.zeros((T, N), np.int32)
        log_probs = np.zeros((T, N), np.float32)
        values = np.zeros((T, N), np.float32)
        rewards = np.zeros((T, N), np.float32)
        dones = np.zeros((T, N), bool)
        ep_returns: list[float] = []
        ep_opps: list[str] = []
        for t in range(T):
            lifetime.check()
            obs[t], mask[t] = step.observations, step.action_masks.astype(bool)
            key, sub = jax.random.split(key)
            a, lp, v = act_fn(
                params, jax.device_put(obs[t], device), jax.device_put(mask[t], device), sub
            )
            actions[t], log_probs[t], values[t] = np.asarray(a), np.asarray(lp), np.asarray(v)
            step = loop.step(actions[t])
            rewards[t], dones[t] = step.rewards, step.dones
            for ep in step.episodes:
                if ep.learner_decisions > 0:
                    ep_returns.append(ep.learner_return)
                    ep_opps.append(ep.opponent_id)
        key, sub = jax.random.split(key)
        _, _, last_v = act_fn(
            params,
            jax.device_put(step.observations, device),
            jax.device_put(step.action_masks.astype(bool), device),
            sub,
        )
        rollout = Rollout(
            actor_id=actor_id,
            param_version=version,
            batch=Batch(obs, mask, actions, log_probs, values, rewards, dones, np.asarray(last_v)),
            episode_returns=ep_returns,
            episode_opponents=ep_opps,
        )
        pipeline.send(actor_id, rollout)
    batch_env.close()


def train(
    cfg: TrainConfig,
    env_fn: EnvFn,
    provider: OpponentProvider,
    logger: Logger | None = None,
    on_update: Callable[[int, nnx.GraphDef[ActorCritic], Params], dict[str, float] | None]
    | None = None,
) -> TrainResult:
    """Train a PPO agent against ``provider``'s opponents on a Sebulba layout.

    Args:
        cfg: Configuration.
        env_fn: Factory for one two-player AEC environment.
        provider: Supplies opponents to the actors.
        logger: Receives metrics every ``cfg.log_every`` updates.
        on_update: Optional hook called on logging updates with ``(update, graphdef,
            params)``; any
            metrics it returns are logged too (used for evaluation).

    Returns:
        Final parameters and the logged metric history.
    """
    logger = logger or NullLogger()
    probe = InProcessAECBatch(env_fn, 1)
    obs_dim, num_actions = int(np.prod(probe.obs_shape)), probe.num_actions
    probe.close()

    graphdef, params = init_actor_critic(obs_dim, num_actions, cfg.ppo.hidden, cfg.seed)
    learner_device = resolve_device(cfg.arch.learner_device)
    params = jax.device_put(params, learner_device)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm), optax.adam(cfg.ppo.lr, eps=1e-5)
    )
    opt_state = optimizer.init(params)
    update_fn = jax.jit(make_update_fn(graphdef, optimizer, cfg.ppo.update))
    act_fn = _make_act_fn(graphdef)
    key = jax.device_put(jax.random.key(cfg.seed), learner_device)

    lifetime = Lifetime()
    server = ParameterServer([resolve_device(cfg.arch.actor_device)], lifetime)
    pipeline = OnPolicyPipeline(cfg.arch.num_actors, lifetime)
    server.publish(params, version=0)
    for i in range(cfg.arch.num_actors):
        lifetime.spawn(
            _actor, f"actor-{i}", i, cfg, env_fn, provider, act_fn, server, pipeline, lifetime
        )

    result = TrainResult(graphdef=graphdef, params=params)
    window: list[float] = []
    start = last_log = time.perf_counter()
    last_logged_update = 0
    try:
        for u in range(cfg.num_updates):
            rollouts: list[Rollout] = pipeline.collect()
            batch = Batch(
                *(
                    np.concatenate(xs, axis=1 if np.ndim(xs[0]) >= 2 else 0)
                    for xs in zip(*(r.batch for r in rollouts), strict=True)
                )
            )
            batch = jax.device_put(batch, learner_device)
            key, sub = jax.random.split(key)
            params, opt_state, metrics = update_fn(params, opt_state, batch, sub)
            server.publish(params, version=u + 1)
            for r in rollouts:
                window += r.episode_returns
                result.episode_returns += r.episode_returns

            if (u + 1) % cfg.log_every == 0 or u + 1 == cfg.num_updates:
                now = time.perf_counter()
                row = {k: float(v) for k, v in jax.device_get(metrics).items()}
                updates_since = u + 1 - last_logged_update
                row["perf/sps"] = cfg.steps_per_update * updates_since / max(now - last_log, 1e-9)
                row["perf/elapsed_s"] = now - start
                row["perf/param_lag"] = float(np.mean([u - r.param_version for r in rollouts]))
                row["train/episodes"] = float(len(window))
                row["train/return_mean"] = float(np.mean(window)) if window else float("nan")
                row["train/return_stderr"] = (
                    float(np.std(window, ddof=1) / np.sqrt(len(window)))
                    if len(window) > 1
                    else float("nan")
                )
                if on_update is not None:
                    row.update(on_update(u + 1, graphdef, params) or {})
                step = (u + 1) * cfg.steps_per_update
                logger.log(step, row)
                result.history.append({"step": float(step), **row})
                window, last_log, last_logged_update = [], time.perf_counter(), u + 1
    except StopRequestedError:
        pass
    finally:
        stuck = lifetime.join()
        logger.close()
    lifetime.raise_if_failed()
    if stuck:
        raise RuntimeError(f"threads did not shut down: {stuck}")
    result.params = jax.device_get(params)
    return result
