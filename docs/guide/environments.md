# Environments, workers and GPUs

## Choosing an environment

Environments are named by id, plus optional constructor arguments:

| id | what it builds |
|---|---|
| `openspiel:<game>` | OpenSpiel's two-player zero-sum sequential games (e.g. `openspiel:kuhn_poker`, `openspiel:leduc_poker`) behind a PettingZoo AEC wrapper, with exact evaluation and CFR opponents |
| `pettingzoo:<registered id>` | Any env in PettingZoo's AEC registry, built with `pettingzoo.make("aec", ...)`, e.g. `pettingzoo:classic/go-v5` or `pettingzoo:classic/connect_four-v3` |

```bash
uv run sparjax train --env.id pettingzoo:classic/go-v5 --env.kwargs board_size 9 \
    --ppo.conv-channels 32 64 64 --ppo.hidden 256
```

!!! note
    `openspiel:leduc_poker` is **OpenSpiel's** Leduc poker, not PettingZoo's
    `classic/leduc_holdem-v4` (which wraps RLCard and uses different rules and
    observations).

Any two-player AEC env with a discrete action space works. The action mask may be in
`obs["action_mask"]` (the PettingZoo classic convention) or in `info["action_mask"]`. For
image-like `H×W×C` observations (Go, Connect Four), add a CNN torso with
`--ppo.conv-channels`. Go needs the `go` extra (`pygame-ce`).

## Subprocess env workers

Python env stepping holds the GIL, so one actor thread can step only one env at a time.
`--env.num-workers K` moves each actor's envs into `K` worker processes
(`SubprocessAECBatch`), which step in parallel. With the same seeds, trajectories are
bit-identical to stepping in-process.

```bash
uv run sparjax train --env.id pettingzoo:classic/go-v5 --env.kwargs board_size 9 \
    --env.num-envs 128 --env.num-workers 4 --arch.num-actors 2 ...
```

Workers are started with `spawn` (forking a process that already runs JAX is unsafe), so:

- the env factory must be picklable; the CLI's `EnvSpec` is, but lambdas are not;
- **your own training scripts need an `if __name__ == "__main__":` guard.**

## Where to run what

Keep **actors on the CPU** (`--arch.actor-device cpu`, the default) and the **learner on
the GPU** (`--arch.learner-device default`). If actors run inference on the same GPU as
the learner, their small inference calls queue behind the learner's update and acting
stalls during training. That removes most of the benefit of running them concurrently
(see the [Go benchmarks](../experiments/go-sebulba.md)).

## GPU installs

```bash
uv sync --extra cuda12          # CUDA 12, current cuDNN
uv sync --extra cuda12-pascal   # GTX 10xx / compute capability 6.x
```

cuDNN 9.11 and later can no longer run convolutions on Pascal GPUs (you get
`CUDNN_STATUS_EXECUTION_FAILED` / status 5003 during autotuning). The `cuda12-pascal`
extra pins `nvidia-cudnn-cu12==9.10.2.21`, the last version that works. MLP-only models
are unaffected either way.
