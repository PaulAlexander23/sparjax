# sparjax: Implementation Plan

> Companion to [the brief](brief.md) (the *why*). This file is the *what and
> in which order*. Every milestone ends with CI green and something demonstrable.
> Status as of 2026-11-24: **M1–M3 complete (PRs open); M4 next.**

## Guiding rules

- **Logic separate from threads.** Every component (adapter, opponent loop, pipeline, param
  server, provider) is a plain class that can be tested synchronously. Threads only wire them
  together.
- **The env batch sits behind an interface from day one.** Subprocess workers are a planned
  requirement (M6), so nothing above the interface may touch an `AECEnv` directly.
- **One game definition.** The env, the CFR opponent and the evaluator all use the same
  `pyspiel` game.
- **Typed public API.** basedpyright must pass with no ignores in `src/` except at the
  `pyspiel` boundary, which has no stubs.
- **Small PRs**, roughly one per numbered task below.

## Key interfaces (to be fixed in M2/M3)

```python
# envs/batch.py: the boundary that subprocess workers will implement later
class AECBatch(Protocol):
    num_envs: int

    def reset(self, env_ids: IntArray, seeds: IntArray) -> BatchView: ...
    def step(self, env_ids: IntArray, actions: IntArray) -> BatchView: ...

    # One AEC step for each selected env. The return covers only those envs:
    #   to_move[i]          seat whose turn it is now
    #   obs[i], mask[i]     observation and action mask for that seat
    #   rewards[i, seat]    reward accrued since the previous call, for each seat
    #   done[i]             game over (termination or truncation)
    #   info_state[i]       optional str key (used by tabular opponents)


# Implementations: InProcessAECBatch (M3), SubprocessAECBatch (M6).
# Each call costs one message per worker, so it works over pipes too.


# opponents/base.py
class Opponent(Protocol):
    def act(self, batch: OpponentBatch, rng: np.random.Generator) -> IntArray: ...


class OpponentProvider(Protocol):
    def next_opponent(self, actor_id: int) -> tuple[OpponentId, Opponent]: ...
    def report(self, opponent_id: OpponentId, outcome: float) -> None: ...
```

---

## M1: Project skeleton

**Goal:** an installable, empty `sparjax` package with all tooling and CI green.

1. `git init`, `.gitignore`, `LICENSE` (Apache-2.0), `NOTICE` (Stoix + Mava attribution placeholder), `README.md`.
2. `pyproject.toml`:
   - `name = "sparjax"`, `requires-python = ">=3.12"`, `src/` layout, `uv_build` backend; version is static in `pyproject.toml` (uv_build has no dynamic versions) and read through `importlib.metadata`.
   - Runtime deps (kept minimal for now): `jax`, `numpy`. Others are added in the milestone that first needs them.
   - Extras: `cuda12 = ["jax[cuda12]"]`. Later: `openspiel`, `wandb`.
   - Dependency groups: `dev` (ruff, basedpyright, pytest, pytest-cov, pre-commit) and `docs` (mkdocs-material, mkdocstrings[python]).
3. Tool config in `pyproject.toml`:
   - **ruff**: line length 100, rules `E,F,W,I,UP,B,SIM,RUF,NPY,PT,D` (Google docstrings), tests exempt from `D`.
   - **basedpyright**: `typeCheckingMode = "recommended"` on `src/` and `tests/`, tuned for JAX noise (e.g. `reportAny` relaxed). Revisit strictness at M4.
   - **pytest**: `--cov=sparjax --cov-report=term-missing --cov-fail-under=90`, `--strict-markers`, markers `slow` and `gpu`.
4. `src/sparjax/__init__.py` (version), `src/sparjax/py.typed`, one smoke test (`import sparjax`, version and JAX device listing).
5. Docs: `mkdocs.yml` (Material, mkdocstrings, nav: Home, Design → Brief/Plan, API). Move `PROJECT_BRIEF.md` and `PLAN.md` to `docs/design/` so they are published.
6. `.pre-commit-config.yaml`: ruff check/format, basedpyright, end-of-file and whitespace fixers.
7. GitHub Actions (`astral-sh/setup-uv`, cached):
   - `ci.yml` on PR and push: `lint` (ruff check + format --check), `typecheck` (basedpyright), `test` (pytest-cov, matrix Python 3.12/3.13/3.14, CPU JAX), `docs` (`mkdocs build --strict`).
   - `docs.yml` on push to `main`: deploy to GitHub Pages.
   - `dependabot.yml` for GitHub Actions.
8. Local check that the GPU works: `uv sync --extra cuda12` and `jax.devices()`.
   ⚠️ The dev machine has a **GTX 1080 (Pascal, sm_61)**, and current JAX CUDA wheels may no
   longer support Pascal. If they don't, we develop on CPU (fine for poker-sized nets) and
   record the limitation in the README.

**Done when:** `uv sync && uv run ruff check && uv run ruff format --check && uv run basedpyright && uv run pytest && uv run mkdocs build --strict` all pass locally. The CI workflow files are in place.
*Creating the GitHub repo and pushing is a separate step that needs your go-ahead.*

---

## M2: Poker env + opponents (no RL)

**Goal:** play Kuhn/Leduc through our PettingZoo AEC wrapper against CFR, with exact best-response numbers.

1. Add `open_spiel` (extra `openspiel`, plus the dev group) and `pettingzoo`.
2. `envs/openspiel_aec.py`: `OpenSpielAECEnv(game_name, **params)`, a genuine `pettingzoo.AECEnv`.
   - Obs = `information_state_tensor`; `action_mask` in the obs dict (PettingZoo classic convention); `info["info_state"]` = `information_state_string`.
   - Chance nodes are resolved internally with a seeded RNG. Env IDs read `openspiel/kuhn_poker` and `openspiel/leduc_poker`, and the docs say plainly that this is **not** PettingZoo's `leduc_holdem`.
   - All `pyspiel` access is isolated in this module, the only place type ignores are allowed.
3. Tests: PettingZoo `api_test`, seeding determinism, and payoff parity with `pyspiel` on random playouts.
4. `opponents/`: `Opponent`/`OpponentProvider` protocols, `RandomOpponent`, `ScriptedOpponent`, `TabularPolicyOpponent` (+ npz/JSON save/load), `FixedProvider`, `MixtureProvider`.
5. `opponents/cfr.py` + CLI `sparjax cfr-solve --game <game> --iters K --out file`: run CFR+ and export the average policy. Also `sparjax cfr-br --policy <file>`, which prints the best-response value for each seat and the exploitability via OpenSpiel. *(Shipped with flat `cfr-solve`/`cfr-br` subcommands; tyro was pulled forward from M4 for this.)*
6. Tests: CFR on Kuhn converges near the known game value (−1/18 for player 0); a tabular round-trip keeps the policy identical.

**Done when:** `sparjax cfr-solve --game leduc_poker --iters 1000` writes a policy, and `cfr-br` reports its BR values. A tabular opponent loaded from that file plays legally in the AEC env.

---

## M3: Batched opponent loop

**Goal:** fixed-shape learner trajectories from N AEC envs against any opponent, with no RL yet.

1. `envs/batch.py`: the `AECBatch` protocol and `InProcessAECBatch`.
2. `envs/adapter.py`: maps AEC rewards, masks and seats to the learner's perspective; seat randomisation on reset.
3. `envs/loop.py`: `BatchedOpponentLoop.step(learner_actions) -> LearnerStep` (obs, mask, reward, done, and episode stats), following brief §4.2. It also fast-forwards through opponent moves after resets.
4. Tests (Hypothesis property tests): the sum of learner rewards per episode equals the game's return; the learner is never asked to act off-turn; shapes are constant; seats are balanced across resets.

**Done when:** a random learner against `RandomOpponent` on Leduc runs for 10k steps with all invariants holding.

*Outcome:* done. The reward-accounting property test caught a real bug: a reward received in a
freshly reset game, before the learner's first move, was credited to the previous game's final
transition. It is fixed with a per-env carry that goes into the new game's first transition.
Throughput on one core (Leduc, random learner): about 14k learner steps/s (7k games/s) at
N = 64–256, about 10.6k at N = 16, and the same against a CFR tabular opponent.

---

## M4: Sebulba PPO

**Goal:** end-to-end training on Kuhn against a fixed opponent.

1. Add `flax` (nnx), `optax`, `jaxtyping`, `tyro`.
2. `arch/`: `ParameterServer`, `OnPolicyPipeline`, `ThreadLifetime`, device selection, adapted from Stoix/Mava with attribution in `NOTICE` and file headers.
3. `agents/ppo/`: a masked actor-critic MLP (nnx), GAE, a clipped PPO loss with masked entropy, and a learner step via `jit(shard_map)` (falling back to a single device).
4. `agents/ppo/system.py`: actor threads (loop + jitted policy on the actor device), a learner thread, and clean shutdown.
5. `config.py` (frozen dataclasses) and `cli.py` (tyro): `sparjax train ppo --env openspiel/kuhn_poker --opponent cfr:path`.
6. `logging/`: a `Logger` protocol with console and JSONL backends.
7. Tests: unit tests for GAE, the loss and the mask; threaded components tested with a fake env; a **CI smoke test** where Kuhn against `RandomOpponent` improves its average payoff in < 60 s on CPU.

**Done when:** PPO on Kuhn against a uniform-random opponent beats it clearly, and against CFR(k) gets close to its best-response value.

---

## M5: Leduc against CFR + evaluation

1. `eval/`: payoff with confidence intervals for each seat; the gap to the best response; exploitability of the PPO policy (query the network at every info state to build an OpenSpiel `TabularPolicy`).
2. An async evaluator thread, plus checkpointing (orbax or plain npz; decided here).
3. An experiment script and a docs page with curves: PPO payoff against CFR(k) for 2–3 values of k, next to the exact BR value.

**Done when:** the v1 success criterion from the brief (≥ ~80% of the BR value for several k) is met and documented.

---

## M6: Subprocess env workers

1. `SubprocessAECBatch`: K worker processes, each owning a slice of envs and speaking the `AECBatch` protocol over pipes with shared-memory obs buffers.
2. Spawn-safe env factories (pickled constructor + kwargs). The opponent stays in the actor process.
3. Tests: in-process and subprocess batches produce identical trajectories under the same seeds; workers shut down cleanly when killed.
4. A benchmark doc comparing steps/s for threads only and for subprocess workers.

---

## M7: Extensions (ordered by interest, decided later)

- `NeuralOpponent` + `SnapshotPoolProvider` (latest / uniform FSP / PFSP): self-play as an opponent provider.
- More PettingZoo envs through the adapter (Connect Four, Texas Hold'em) and an "add your own env" guide.
- Recurrent policy; W&B logger backend; multiple learner devices.

---

## Risks

| Risk | Mitigation |
|---|---|
| ~~The GTX 1080 isn't supported by current JAX CUDA wheels~~ | **Resolved in M1:** `jax[cuda12]` 0.12.1 runs on the GTX 1080 |
| Material for MkDocs reaches end of life on 2027-05-05 (its successor, Zensical, reads `mkdocs.yml`) | Keep MkDocs for now; trial Zensical before EOL |
| The GIL caps actor-thread throughput | Expected; M6 subprocess workers |
| No `pyspiel` type stubs | Confine it to two modules; type ignores allowed only there |
| JAX API churn (`shard_map` moved from `jax.experimental`) | Pin minimum versions; one `arch/devices.py` compat module |
| Opponent loop off-by-one bugs (reward credit, resets) | Property tests in M3 before any RL |
