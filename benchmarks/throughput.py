"""Throughput benchmarks for Sebulba PPO on Go 9x9.

Runs short training jobs and reports steady-state learner decisions per second (the
first logging window, which includes JIT compilation, is discarded) together with where
the time goes:

* ``learner_wait``: learner blocked waiting for rollouts (actors are the bottleneck)
* ``learner_update``: learner busy in the PPO update
* ``actor_param_wait``: actors blocked waiting for fresh parameters (learner is the bottleneck)
* ``actor_collect``: actors collecting a rollout (acting + env stepping)

Usage::

    uv run python benchmarks/throughput.py workers  --out runs/bench_workers.jsonl
    uv run python benchmarks/throughput.py lag      --out runs/bench_lag.jsonl
    uv run python benchmarks/throughput.py actors   --out runs/bench_actors.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from sparjax.agents.ppo.loss import PPOHyperparams
from sparjax.agents.ppo.system import train
from sparjax.config import ArchConfig, EnvConfig, PPOConfig, TrainConfig
from sparjax.opponents import FixedProvider, RandomOpponent

BASE = TrainConfig(
    env=EnvConfig("pettingzoo:classic/go-v5", {"board_size": 9}, num_envs=128, num_workers=4),
    arch=ArchConfig(num_actors=2, actor_device="default", learner_device="default", max_lag=1),
    ppo=PPOConfig(
        rollout_len=32,
        hidden=(256,),
        conv_channels=(32, 64, 64),
        update=PPOHyperparams(epochs=4, num_minibatches=8),
    ),
    log_every=5,
)


def run(cfg: TrainConfig, updates: int) -> dict[str, float]:
    """Train for ``updates`` updates and summarise steady-state performance."""
    cfg = replace(cfg, total_timesteps=updates * cfg.steps_per_update)
    start = time.perf_counter()
    result = train(cfg, cfg.env.spec, FixedProvider(RandomOpponent()))
    rows = [h for h in result.history if "perf/sps" in h][1:]  # drop the compile window
    keys = ["sps", "learner_wait_s", "learner_update_s", "actor_param_wait_s", "actor_collect_s"]
    out = {k: float(np.mean([r[f"perf/{k}"] for r in rows])) for k in keys}
    out["param_lag"] = float(np.mean([r["perf/param_lag"] for r in rows]))
    out["wall_s"] = time.perf_counter() - start
    return out


def sweep(name: str) -> list[tuple[dict[str, object], TrainConfig]]:
    """The configurations of one benchmark."""
    cases: list[tuple[dict[str, object], TrainConfig]] = []
    if name == "workers":
        for k in (0, 1, 2, 4, 8):
            cfg = replace(BASE, env=replace(BASE.env, num_workers=k))
            cases.append(({"workers_per_actor": k}, cfg))
    elif name == "lag":
        for epochs in (4, 16):
            for lag in (0, 1):
                ppo = replace(BASE.ppo, update=replace(BASE.ppo.update, epochs=epochs))
                cfg = replace(BASE, ppo=ppo, arch=replace(BASE.arch, max_lag=lag))
                cases.append(({"epochs": epochs, "max_lag": lag}, cfg))
    elif name == "actors":
        for actors, workers in ((1, 8), (2, 4), (4, 2)):
            for lag in (0, 1):
                cfg = replace(
                    BASE,
                    env=replace(BASE.env, num_workers=workers),
                    arch=replace(BASE.arch, num_actors=actors, max_lag=lag),
                )
                cases.append(
                    ({"actors": actors, "workers_per_actor": workers, "max_lag": lag}, cfg)
                )
    else:
        raise SystemExit(f"unknown benchmark {name!r}")
    return cases


def main() -> None:
    """Parse arguments, run one benchmark sweep, print and save the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["workers", "lag", "actors"])
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--actor-device", choices=["cpu", "default"], default="default", help="where actors infer"
    )
    args = parser.parse_args()
    for params, cfg in sweep(args.benchmark):
        cfg = replace(cfg, arch=replace(cfg.arch, actor_device=args.actor_device))
        params = {**params, "actor_device": args.actor_device}
        res = {**params, **run(cfg, args.updates)}
        print(
            json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}),
            flush=True,
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
