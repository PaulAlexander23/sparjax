# sparjax

**Train PPO agents against the sparring partners of your choice.**

sparjax trains PPO agents in two-player, turn-based games that use the
[PettingZoo AEC](https://pettingzoo.farama.org/api/aec/) API. It runs on a JAX
**Sebulba** actor–learner architecture. The opponent is pluggable: a CFR policy, a
scripted bot, a frozen network, or a pool of the agent's own past snapshots.

!!! warning "Pre-alpha"
    Nothing here is usable yet. Start with the [project brief](design/brief.md) and the
    [implementation plan](design/plan.md).

## Installation (development)

```bash
git clone https://github.com/PaulAlexander23/sparjax
cd sparjax
uv sync                  # CPU
uv sync --extra cuda12   # CUDA 12
```
