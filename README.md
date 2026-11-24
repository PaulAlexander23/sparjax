# sparjax

[![CI](https://github.com/PaulAlexander23/sparjax/actions/workflows/ci.yml/badge.svg)](https://github.com/PaulAlexander23/sparjax/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://paulalexander23.github.io/sparjax/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Train PPO agents against the sparring partners of your choice.**

sparjax trains PPO agents in two-player, turn-based games that use the
[PettingZoo AEC](https://pettingzoo.farama.org/api/aec/) API. It runs on a JAX
**Sebulba** actor–learner architecture. The opponent is pluggable: a CFR policy, a
scripted bot, a frozen network, or a pool of the agent's own past snapshots.

> **Status: pre-alpha.** See the [project brief](docs/design/brief.md) and the
> [implementation plan](docs/design/plan.md).

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
uv sync                      # CPU JAX + dev tools
uv sync --extra cuda12       # CUDA 12 JAX instead
```

| Task | Command |
|---|---|
| Lint | `uv run ruff check` |
| Format | `uv run ruff format` |
| Type check | `uv run basedpyright` |
| Tests + coverage | `uv run pytest` |
| Docs (live) | `uv run --group docs mkdocs serve` |
| Pre-commit hooks | `uv run pre-commit install` |

## License

Apache-2.0. Parts of the Sebulba architecture are adapted from
[Stoix](https://github.com/EdanToledo/Stoix) and [Mava](https://github.com/instadeepai/Mava).
See [NOTICE](NOTICE).
