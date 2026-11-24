# Sebulba speed-ups on Go 9×9 (M6)

These benchmarks show the two performance features of the architecture on an
environment that is expensive to step: PettingZoo's `classic/go-v5` on a 9×9 board
(about 160 µs per env step, about 128 moves per game).

1. **Subprocess env workers**: step many environments in parallel despite the GIL.
2. **Concurrent acting and learning**: with `max_lag = 1`, actors collect the next
   rollout while the learner trains on the previous one; with `max_lag = 0` they take
   turns.

**Setup.** Learner on a GTX 1080 (cuDNN 9.10, the `cuda12-pascal` extra), actors on the
CPU, a 16-thread desktop CPU. The opponent plays uniformly at random. CNN torso 32-64-64
(3×3) with a 256-unit head. 2 actors × 128 envs × 32-step rollouts (8,192 decisions per
update), 4 epochs × 8 minibatches unless stated. Throughput is steady state: the first
logging window, which includes JIT compilation, is dropped.

Reproduce with:

```bash
uv sync --extra cuda12-pascal            # or --extra cuda12 on newer GPUs
uv run python benchmarks/throughput.py workers --actor-device cpu
uv run python benchmarks/throughput.py lag     --actor-device cpu
uv run python benchmarks/throughput.py actors  --actor-device cpu
```

The columns are mean seconds per update: **L-wait** is the learner waiting for rollouts,
**L-update** the learner updating, **A-wait** actors waiting for parameters, **A-collect**
actors collecting.

## 1. Subprocess workers

| workers per actor | decisions/s | speed-up | L-wait | L-update | A-collect |
|---:|---:|---:|---:|---:|---:|
| 0 (in-process) | 2,296 | 1.0× | 3.08 | 0.43 | 3.57 |
| 1 | 4,539 | 2.0× | 1.41 | 0.39 | 1.80 |
| 2 | 6,942 | 3.0× | 0.78 | 0.39 | 1.18 |
| 4 | 8,795 | 3.8× | 0.53 | 0.39 | 0.93 |
| 8 | 10,207 | **4.4×** | 0.40 | 0.39 | 0.80 |

A single worker already doubles throughput, because env stepping moves off the actor
thread and overlaps with policy inference. Gains flatten at about 8 workers per actor
(16 processes on 16 hardware threads).

## 2. Concurrent acting and learning

| update | `max_lag` | decisions/s | L-wait | L-update | A-wait | A-collect |
|---|---:|---:|---:|---:|---:|---:|
| 4 epochs | 0 (sync) | 6,229 | 0.92 | 0.39 | 0.40 | 0.92 |
| 4 epochs | 1 (async) | **8,839** (+42%) | 0.53 | 0.39 | 0.00 | 0.93 |
| 16 epochs | 0 (sync) | 3,324 | 0.92 | 1.53 | 1.55 | 0.92 |
| 16 epochs | 1 (async) | **5,301** (+60%) | 0.00 | 1.54 | 0.61 | 0.93 |

In sync mode the time per update is about *collect + update*; in async mode it is about
*max(collect, update)*. The measured speed-ups match that model almost exactly: the
predicted values are +42% and +61%. With 16 epochs the async learner never waits
(L-wait = 0), so it is the bottleneck and the actors idle instead.

### Actor scaling (sync vs async)

| actors × workers | sync | async | speed-up |
|---|---:|---:|---:|
| 1 × 8 | 5,292 | 7,241 | +37% |
| 2 × 4 | 6,189 | 8,754 | +41% |
| 4 × 2 | 7,257 | **10,968** | +51% |

## 3. Wall-clock learning

Two full training runs that differ only in `max_lag` (3M decisions each):

| | wall time | time to 80% / 90% / 95% win rate | decisions to 90% | final win rate |
|---|---:|---:|---:|---:|
| sync (`max_lag 0`) | 492 s | 34 / 60 / 93 s | 327k | 93.9% |
| async (`max_lag 1`) | **353 s** | 36 / **50** / **78** s | 368k | 97.8% |

Async finishes the same budget **1.39× faster**. Learning from one-update-old parameters
costs slightly more decisions to reach 90% (368k vs 327k), but the throughput gain more
than pays for it.

## Finding: keep actors off the learner's GPU

With actor inference on the same GPU as the learner (`--arch.actor-device default`),
async gained only **+4%** (8,204 vs 7,853 decisions/s). The timings show why: actor
collection time rose from 0.65 s to 1.00 s per update (and from 0.64 s to 2.10 s with 16
epochs) whenever the learner was updating. XLA runs the learner's large update program
and the actors' small inference calls one after another on the device, so acting
stalled during training. CPU actors avoid this, and they are faster overall here too
(8,839 vs 8,204).

## 4. Against a stronger scripted opponent (M7a)

Trained against `go_light` (the light-playout policy, which itself beats random 90.5% on
9×9). Every 10 updates the network is evaluated by playing 200 games against each of
`random` and `go_light`. Settings: 4 actors × 2 workers × 128 envs, async, 15M decisions,
otherwise as above.

```bash
uv run sparjax train --env.id pettingzoo:classic/go-v5 --env.kwargs board_size 9 \
    --env.num-envs 128 --env.num-workers 2 --arch.num-actors 4 --opponent.kind go_light \
    --ppo.conv-channels 32 64 64 --ppo.hidden 256 --ppo.rollout-len 32 \
    --ppo.update.num-minibatches 8 --total-timesteps 15000000 \
    --eval.opponents random go_light --eval.games 200
```

| decisions | vs random | vs go_light (95% CI) |
|---:|---:|---:|
| 0.2M | 51.5% | 17.9% ± 5.3% |
| 1.8M | 99.0% | **97.0%** ± 2.4% |
| 3.3M | 100% | 98.5% ± 1.7% |
| 9.2M | 100% | **99.5%** ± 1.0% |
| 15.0M | 100% | 98.0% ± 1.9% |

PPO overtakes the scripted opponent within about 2M decisions (about 4 minutes) and then
stays at 98–99.5%, in both seats. Throughput was 7.5k decisions/s, below the 11k benchmark,
because the 400 evaluation games per check compete with the actors for CPU. The evaluator
skipped 19 of 92 snapshots while busy, as designed: it always evaluates the newest one.
This opponent is now too weak to measure further progress; self-play (M7b) is the
natural next opponent.
