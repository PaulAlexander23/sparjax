# sparjax

**Train PPO agents against the sparring partners of your choice.**

sparjax trains PPO agents in two-player, turn-based games that use the
[PettingZoo AEC](https://pettingzoo.farama.org/api/aec/) API. It runs on a JAX
**Sebulba** actor–learner architecture. The opponent is pluggable: a CFR policy, a
scripted bot, a frozen network, or a pool of the agent's own past snapshots.

!!! note "Status: research prototype"
    Milestones M1–M7b are complete and tested. Start with
    [Findings and future directions](findings.md) for what works and what we learned, or
    the [project brief](design/brief.md) for the design rationale.

## Installation (development)

```bash
git clone https://github.com/PaulAlexander23/sparjax
cd sparjax
uv sync                  # CPU
uv sync --extra cuda12   # CUDA 12
```

## Quick start

```bash
# Solve Leduc with CFR+ and report how exploitable the result is
uv run sparjax cfr-solve --game leduc_poker --algorithm cfr_plus --iters 20

# Train PPO against a CFR+(10) opponent, with exact evaluation as it trains
uv run sparjax train --env.id openspiel:leduc_poker --opponent.kind cfr \
    --opponent.cfr-iters 10 --total-timesteps 2000000

# Fictitious self-play against a pool of the learner's own snapshots
uv run sparjax train --env.id openspiel:leduc_poker --opponent.kind selfplay \
    --ppo.update.ent-coef 0.05 --total-timesteps 2000000

# Go 9x9 against a light-playout bot, envs stepped in 8 subprocess workers
uv run --extra go sparjax train --env.id pettingzoo:classic/go-v5 --env.kwargs board_size 9 \
    --env.num-workers 8 --env.max-steps 200 --opponent.kind go_light --no-exact-eval
```

`uv run sparjax train --help` lists every option. The experiment pages give the exact
settings behind each reported result.
