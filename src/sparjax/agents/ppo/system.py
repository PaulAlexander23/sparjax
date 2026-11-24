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

import json
import threading
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
from sparjax.arch.evaluator import AsyncEvaluator
from sparjax.checkpoint import config_to_dict, save_checkpoint
from sparjax.config import TrainConfig
from sparjax.envs.batch import AECBatch, InProcessAECBatch
from sparjax.envs.loop import BatchedOpponentLoop
from sparjax.envs.subprocess import SubprocessAECBatch
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
    param_wait_s: float = 0.0
    """Time the actor spent blocked waiting for fresh enough parameters."""
    collect_s: float = 0.0
    """Time the actor spent collecting this rollout (acting + env stepping)."""


@dataclass
class TrainResult:
    """Outcome of :func:`train`."""

    graphdef: nnx.GraphDef[ActorCritic]
    params: Params
    history: list[dict[str, float]] = field(default_factory=list[dict[str, float]])
    episode_returns: list[float] = field(default_factory=list[float])
    """Every finished game's learner payoff, in learner-update order."""


def _lr_schedule(cfg: TrainConfig) -> float | optax.Schedule:
    """Constant learning rate, or linear decay to zero over all optimiser steps."""
    if not cfg.ppo.anneal_lr:
        return cfg.ppo.lr
    steps = cfg.num_updates * cfg.ppo.update.epochs * cfg.ppo.update.num_minibatches
    return optax.linear_schedule(cfg.ppo.lr, 0.0, steps)


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
    batch_env: AECBatch = (
        SubprocessAECBatch(env_fn, cfg.env.num_envs, cfg.env.num_workers)
        if cfg.env.num_workers > 0
        else InProcessAECBatch(env_fn, cfg.env.num_envs)
    )
    loop = BatchedOpponentLoop(
        batch_env, provider, actor_id=actor_id, seed=cfg.seed * 1000 + actor_id
    )
    key = jax.device_put(jax.random.key(cfg.seed * 1000 + actor_id + 1), device)
    loop.next_opponent()
    step = loop.reset()
    T, N = cfg.ppo.rollout_len, cfg.env.num_envs
    obs_shape, A = batch_env.obs_shape, batch_env.num_actions

    for r in range(cfg.num_updates):
        t0 = time.perf_counter()
        params, version = server.get(device, min_version=max(0, r - cfg.arch.max_lag))
        t1 = time.perf_counter()
        loop.next_opponent()
        obs = np.zeros((T, N, *obs_shape), batch_env.obs_dtype)
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
            if step.final_observations:
                # Truncated games: bootstrap with the value of the cut-off position so GAE
                # can treat the step as terminal without biasing towards zero.
                ids = np.fromiter(step.final_observations, dtype=np.int64)
                finals = np.stack([step.final_observations[int(i)] for i in ids])
                key, sub = jax.random.split(key)
                _, _, v_final = act_fn(
                    params,
                    jax.device_put(finals, device),
                    jax.device_put(np.ones((len(ids), A), bool), device),
                    sub,
                )
                rewards[t, ids] += cfg.ppo.update.gamma * np.asarray(v_final)
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
            param_wait_s=t1 - t0,
            collect_s=time.perf_counter() - t1,
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
    obs_shape, num_actions = probe.obs_shape, probe.num_actions
    probe.close()

    graphdef, params = init_actor_critic(
        obs_shape, num_actions, cfg.ppo.hidden, cfg.seed, cfg.ppo.conv_channels
    )
    learner_device = resolve_device(cfg.arch.learner_device)
    params = jax.device_put(params, learner_device)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm), optax.adam(_lr_schedule(cfg), eps=1e-5)
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
    log_lock = threading.Lock()

    def emit(step: int, row: dict[str, float]) -> None:
        with log_lock:
            logger.log(step, row)
            result.history.append({"step": float(step), **row})

    evaluator: AsyncEvaluator | None = None
    if on_update is not None and cfg.arch.async_eval:
        hook = on_update

        def eval_fn(update: int, p: Params) -> dict[str, float]:
            return hook(update, graphdef, p) or {}

        def eval_sink(update: int, step: int, metrics: dict[str, float]) -> None:
            emit(step, {"eval/update": float(update), **metrics})

        evaluator = AsyncEvaluator(eval_fn, eval_sink, lifetime)

    def checkpoint(update: int) -> None:
        if cfg.log_dir is None:
            return
        save_checkpoint(
            cfg.log_dir / "checkpoints" / f"update_{update:06d}.npz",
            params,
            obs_shape=obs_shape,
            num_actions=num_actions,
            hidden=cfg.ppo.hidden,
            conv_channels=cfg.ppo.conv_channels,
            meta={"update": update, "step": update * cfg.steps_per_update, "env": cfg.env.id},
        )

    if cfg.log_dir is not None:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        (cfg.log_dir / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=2))
    window: list[float] = []
    start = last_log = time.perf_counter()
    last_logged_update = 0
    timing = dict.fromkeys(
        ["learner_wait_s", "learner_update_s", "actor_param_wait_s", "actor_collect_s"], 0.0
    )
    try:
        for u in range(cfg.num_updates):
            t_wait = time.perf_counter()
            rollouts: list[Rollout] = pipeline.collect()
            t_update = time.perf_counter()
            timing["learner_wait_s"] += t_update - t_wait
            batch = Batch(
                *(
                    np.concatenate(xs, axis=1 if np.ndim(xs[0]) >= 2 else 0)
                    for xs in zip(*(r.batch for r in rollouts), strict=True)
                )
            )
            batch = jax.device_put(batch, learner_device)
            key, sub = jax.random.split(key)
            ent_coef = cfg.ppo.ent_coef_at(u / cfg.num_updates)
            params, opt_state, metrics = update_fn(params, opt_state, batch, sub, ent_coef)
            jax.block_until_ready(params)  # so the timing below is real, not async dispatch
            timing["learner_update_s"] += time.perf_counter() - t_update
            for r in rollouts:
                timing["actor_param_wait_s"] += r.param_wait_s / len(rollouts)
                timing["actor_collect_s"] += r.collect_s / len(rollouts)
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
                for name, total in timing.items():
                    row[f"perf/{name}"] = total / updates_since  # mean seconds per update
                timing = dict.fromkeys(timing, 0.0)
                row["train/episodes"] = float(len(window))
                row["train/return_mean"] = float(np.mean(window)) if window else float("nan")
                outcomes = np.sign(np.asarray(window, dtype=np.float64))
                row["train/win_rate"] = float(np.mean(outcomes > 0)) if window else float("nan")
                row["train/draw_rate"] = float(np.mean(outcomes == 0)) if window else float("nan")
                row["train/return_stderr"] = (
                    float(np.std(window, ddof=1) / np.sqrt(len(window)))
                    if len(window) > 1
                    else float("nan")
                )
                step = (u + 1) * cfg.steps_per_update
                if evaluator is not None:
                    evaluator.submit(u + 1, step, params)
                elif on_update is not None:
                    row.update(on_update(u + 1, graphdef, params) or {})
                emit(step, row)
                window, last_log, last_logged_update = [], time.perf_counter(), u + 1
            if cfg.checkpoint_every and (u + 1) % cfg.checkpoint_every == 0:
                checkpoint(u + 1)
        checkpoint(cfg.num_updates)
        if evaluator is not None:
            evaluator.finish()
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
