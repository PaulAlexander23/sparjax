# Project Brief: training PPO against arbitrary opponents on a Sebulba architecture

> Status: **v2, decisions settled (2026-11-24).** The project is named **sparjax**. See §12
> for the outcome of each decision and [PLAN.md](PLAN.md) for the implementation plan.
>
> **Changes since v1**
> - The focus is now **training against configurable opponents** (fixed CFR bots, scripted
>   bots, past snapshots). Self-play is one supported opponent source, not the goal.
> - Environments use the **PettingZoo AEC** interface.
> - The first target is **Leduc poker against a CFR opponent**, with exact evaluation
>   through OpenSpiel.
> - **Mava** was compared with Stoix as a possible base (§3).

---

## 1. The pitch

A small, typed JAX library. It trains a **PPO agent in two-player turn-based games** that
expose the **PettingZoo AEC** API, on a **Sebulba** actor–learner architecture.

The opponent is a plug-in. It can be:
- a **CFR policy** (Nash-approximating, tabular),
- a scripted bot (random, always-call),
- a frozen neural snapshot,
- a pool of past selves sampled by fictitious self-play.

**Can it be done?** Yes. Sebulba only needs actors that produce fixed-shape trajectory batches
and a learner that consumes them. If the actor resolves every opponent move before handing data
to the learner, the learner sees an ordinary single-agent MDP, so the PPO update stays
unchanged. All the new work is in the **actor loop** and the **opponent abstraction** (§4–5).

---

## 2. Background: Anakin vs Sebulba

| | **Anakin** | **Sebulba** |
|---|---|---|
| Env runs on | Accelerator (env in JAX) | CPU (any Python/C++ env) |
| Acting + learning | One fused `jit`/`pmap` program | Actor threads and a learner on separate devices |
| Fits | Jumanji, pgx, gymnax, brax | PettingZoo, OpenSpiel, Gymnasium, envpool |

PettingZoo AEC envs are ordinary Python objects, so **Sebulba is the right architecture.**

> ⚠️ **Honest note on speed.** Leduc is tiny. Throughput will be bound by Python env stepping
> and the GIL, not by the GPU. A JAX Leduc under Anakin would be much faster. The point of this
> project is the **general, env-agnostic setup**; Leduc is the first test case.

---

## 3. Choosing a base: Stoix vs Mava vs standalone

Both libraries were checked on 2026-11-24 (Stoix `main`, Mava `develop`).

| | **Stoix** | **Mava** |
|---|---|---|
| Focus | Single-agent RL | Cooperative MARL |
| Sebulba systems | PPO (`stoix/systems/ppo/sebulba/ff_ppo.py`) | IPPO/MAPPO ff+rec (`mava/systems/ppo/sebulba/`), plus IQL and Sable |
| Sebulba plumbing | `ParameterServer`, `OnPolicyPipeline`, `ThreadLifetime` (`stoix/utils/sebulba_utils.py`) | `ParamsSource`, a ticket-ordered `Pipeline` thread (`mava/utils/sebulba/`) |
| Learner | `pmap` + `pmean` | `jit(shard_map)` over a device mesh (more modern) |
| Env path into Sebulba | envpool / Gymnasium factories (`EnvFactory(num_envs)`) | `gymnasium.AsyncVectorEnv` of RWARE / LBF / SMAClite only |
| Turn-based / AEC / PettingZoo | ❌ | ❌ (only simultaneous actions, where every agent acts every step) |
| Competitive / fixed opponents | ❌ ("2-player self-play" on the roadmap) | ❌ (rewards are summed and shared by default) |
| JAX pin | `jax>=0.4.25,<0.6` | `jax==0.5.3` exactly (JAX 0.6 work in progress) |
| Other pins | optax and `stoa` from git main | `numpy==1.26.4`, `gymnasium==1.0.0`, `flax<0.10.4`, `python<3.13`; brax and mujoco pinned; 4 git-only deps (jaxmarl fork, gigastep, smaclite, marl-eval) |
| Packaging | uv + setuptools | uv + setuptools; PyPI/releases are stale (0.1.3, TF era); `main` last changed Dec 2023 |
| Maintenance | Active | `develop` is active but sparse (commits Aug–Sep 2026 after a quiet gap) |
| License | Apache-2.0 | Apache-2.0 |

**Verdict: pivoting to Mava as a dependency does not solve the fragility problem. Mava is
*more* pinned than Stoix.** Its multi-agent support is also aimed at *cooperative,
simultaneous-move* games, which is the opposite of our case. Fitting a turn-based competitive
game into either library means hiding the opponent inside a single-agent env. That workaround
exists for exactly that reason.

**🟡 DECISION 1, base.** *Recommendation:* **standalone and lean, adapting the best of both**
(Apache-2.0, with credit in `NOTICE`):
- From **Mava**: the `shard_map` learner and the `Pipeline` design.
- From **Stoix**: per-actor queues, the `ParameterServer` API, `ThreadLifetime` shutdown, and
  the `EnvFactory(num_envs)` idea.

The Sebulba core is an estimated ~500–800 LOC. Owning it lets us use current JAX/numpy/Python
and pass basedpyright cleanly.

---

## 4. Environment interface: PettingZoo AEC

### 4.1 What AEC gives us

```python
env.reset(seed=42)
for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    action = None if termination or truncation else policy(obs, mask)
    env.step(action)
```

- `agent_selection` says whose turn it is; `rewards`, `terminations`, `truncations` and
  `infos` are dicts keyed by agent.
- **Action masks** are not standardised. PettingZoo classic puts them in
  `obs["action_mask"]`; Shimmy/OpenSpiel puts them in `info["action_mask"]`.
  We normalise this in one adapter.
- **No official vectorisation for AEC.** SuperSuit is semi-deprecated (Aug 2025 notice) and
  only vectorises parallel envs. We write our own batching (§4.2).
- PettingZoo 1.27.0 (Aug 2026) supports Python 3.10–3.14.

### 4.2 How AEC plugs into Sebulba: batched "opponent-in-the-loop"

Each actor thread owns **N independent AEC envs** and one opponent. Each actor step:

```
1. Every env is at a learner decision point (or was just reset to one).
2. Learner policy acts on all N envs in one jitted batch call.   ──► recorded in the trajectory
3. Step each env with its learner action.
4. Loop: collect envs where it's now the OPPONENT's turn
         → one batched opponent.act(...) call → step those envs
         until every env is back at a learner turn or finished.
5. Finished envs: credit the final reward to the learner's transition,
   report (opponent_id, outcome) if needed, reset, randomise seat,
   and fast-forward through opponent moves if the opponent moves first.
6. Learner reward for the step = learner's cumulative AEC reward since its last action.
```

Why this design:
- The learner trajectory always has a fixed shape (`rollout_len × N`). PPO, GAE and the
  `ParameterServer` stay as they are in single-agent Sebulba.
- The opponent runs **outside the env, in batches**, so a single interface covers a CFR table
  lookup (numpy) and a neural opponent (a jitted call on the actor device).
- Seat randomisation is built in. Poker position matters.

Implementation: a `TwoPlayerAECAdapter` (mask and obs normalisation, reward accounting, which
seat is the learner) plus a `BatchedOpponentLoop` (steps 2–6), both with property tests.

**🟡 DECISION 2, parallelism in the actor.** Python env stepping holds the GIL, so many actor
threads won't scale linearly. Options:
1. Threads only, as in Stoix. Simplest, and fine for cheap envs like Leduc.
2. Subprocess workers per actor, as in Mava's `AsyncVectorEnv`. They step envs and exchange
   obs/actions over pipes. That scales better, but opponent calls need a round trip.

*Recommendation:* (1) first, keeping the env batch behind an interface so (2) can be added later.

---

## 5. Opponents

### 5.1 The abstraction

```python
class Opponent(Protocol):
    def act(self, batch: OpponentBatch, rng: np.random.Generator) -> np.ndarray: ...
    # OpponentBatch: obs tensors, action masks, and optional per-env info (e.g. info-state keys)

class OpponentProvider(Protocol):          # decides WHO we play next
    def next_opponent(self, actor_id: int) -> tuple[OpponentId, Opponent]: ...
    def report(self, opponent_id: OpponentId, outcome: float) -> None: ...
```

| Opponent | How it acts | v1? |
|---|---|---|
| `RandomOpponent` | uniform over legal actions | ✅ |
| `ScriptedOpponent` | user function (e.g. always-call, raise-with-K) | ✅ |
| **`TabularPolicyOpponent`** | lookup `info_state → probs` (CFR output) | ✅ |
| `NeuralOpponent` | frozen params, jitted on the actor device | later |

| Provider | Behaviour | v1? |
|---|---|---|
| `FixedProvider` | always the same opponent | ✅ |
| `MixtureProvider` | sample from a fixed weighted list (e.g. 70% CFR, 30% random) | ✅ |
| `SnapshotPoolProvider` | past learner snapshots; latest / uniform FSP / PFSP sampling | later |

Self-play then needs only `NeuralOpponent` + `SnapshotPoolProvider` plus a learner hook that
adds snapshots every K updates. No core changes, which shows the design is general enough.

### 5.2 The CFR opponent

- Compute it **offline** with OpenSpiel: `CFRPlusSolver(pyspiel.load_game("leduc_poker"))`,
  running `evaluate_and_update_policy()` for k iterations, then take `average_policy()`.
- Export it to a plain file (`info_state_string → action probs`; JSON or npz) so training
  doesn't need to run CFR. A small CLI does this: `<pkg> cfr solve leduc --iters 10000`.
- At play time it is a dict lookup. The env adapter puts the opponent's
  `information_state_string` in its info.

### 5.3 Is a CFR opponent a good training signal?

- Against an **exact** Nash opponent in two-player zero-sum poker, no strategy does better than
  the game value. PPO would get almost no gradient towards anything interesting.
- Against **CFR after k iterations**, the opponent is slightly exploitable. The best possible
  score is **exactly computable**: OpenSpiel's `best_response(game, cfr_policy, seat)` returns it.
- That gives us a **ground-truth benchmark**: PPO's average payoff against CFR(k) should rise
  towards the best-response value of CFR(k). The gap shows how much PPO and the infrastructure
  leave on the table.
- Training against a **mixture** (CFR + random + scripted) is useful when the goal is a robust
  agent rather than an exploiter.

---

## 6. Poker environment stack

| Option | AEC env | CFR | Exact evaluation | Risks |
|---|---|---|---|---|
| **A. OpenSpiel + our own thin AEC wrapper** | ~150 LOC wrapper around `pyspiel` (`leduc_poker`, `kuhn_poker`) | OpenSpiel CFR / CFR+ / MCCFR | `exploitability`, `nash_conv`, `best_response` | We own the wrapper (small, easy to test) |
| B. OpenSpiel + Shimmy `OpenSpielCompatibilityV0` | Ready-made AEC (mask in `info`) | OpenSpiel | Same as A | Need the underlying `pyspiel` state for info-state keys (wrapper access unverified); Shimmy releases are infrequent (2.0.1 in 2026, about 18 months after the previous one) |
| C. PettingZoo `leduc_holdem-v4` + RLCard `CFRAgent` | Native PettingZoo classic | RLCard (Leduc only) | ❌ none | RLCard has had no PyPI release since 2023, classifiers stop at Python 3.11, and it uses deprecated numpy `tostring`; no Kuhn |

> **Note: OpenSpiel's `leduc_poker` and PettingZoo's `leduc_holdem` are different games.**
> PettingZoo's version wraps RLCard, which uses different betting rules, action IDs and
> observation encoding. Our env is *OpenSpiel's* Leduc exposed through the PettingZoo AEC
> *API*, not PettingZoo's `leduc_holdem`. Our numbers therefore can't be compared with
> RLCard/PettingZoo Leduc results, and the docs and env IDs must say so
> (e.g. `openspiel/leduc_poker`). If we later want a CFR opponent for a PettingZoo-native game,
> CFR has to be solved on that game's own tree.

**🟡 DECISION 3.** *Recommendation:* **A.** We get one game definition shared by the env, the
CFR opponent and the evaluator, so there is no risk of encodings disagreeing. We get
**Kuhn** (good for fast tests) and **Leduc**, and exact exploitability. The wrapper is a true
PettingZoo `AECEnv`, so it passes PettingZoo's `api_test`, and any other PettingZoo env
(Connect Four, Texas Hold'em) plugs into the same pipeline later through the same adapter.

---

## 7. Evaluation

- **Payoff against each opponent** (mean chips per hand, with confidence intervals) for each
  seat and overall.
- **Gap to best response** against fixed tabular opponents (§5.3). This is the headline metric.
- **Exploitability of the learned PPO policy.** Query the network at every Leduc info state to
  build an OpenSpiel `TabularPolicy`, then call `exploitability(game, policy)`. Leduc has few
  enough info states for this to be cheap, so it can run in the async evaluator thread.
- Later, for self-play: Elo/TrueSkill against the pool.

---

## 8. Tooling and engineering

| Concern | Tool | Notes |
|---|---|---|
| Env & packaging | **uv** | `src/` layout, `uv.lock` committed; extras `[cuda12]`, `[openspiel]` |
| Lint + format | **ruff** | `E,F,I,UP,B,SIM,RUF,NPY,PT,D` |
| Types | **basedpyright** | Start at `standard`, move to `strict` per package; **jaxtyping** for shapes. pyspiel has no stubs, so it is isolated behind a typed adapter module |
| Tests | **pytest + pytest-cov** | Unit (adapter, opponent loop, CFR export, GAE, masking), PettingZoo `api_test` on our wrappers, end-to-end CPU smoke run on Kuhn (< 60 s); coverage gate |
| Docs | **mkdocs** (Material + mkdocstrings) | Architecture, "add your own env", "add your own opponent", API reference |
| CI | **GitHub Actions** | ruff → basedpyright → pytest-cov (Python 3.12–3.14, CPU JAX) → docs build; Pages deploy on `main`; pre-commit mirrors CI locally |

Threaded code is tested by separating logic from threads. The opponent loop, adapter, pipeline
and param server are plain classes tested synchronously. A few real-thread smoke tests run with
timeouts.

**🟡 DECISION 4, config:** dataclasses + `tyro` (typed) vs Hydra (what Stoix and Mava use).
*Rec:* tyro.

**🟡 DECISION 5, logging:** pluggable, with console/JSONL by default and W&B optional.

**🟡 DECISION 6, Python:** `>=3.11`.

**🟡 DECISION 7, NN library:** `flax.nnx` (fallback linen).

---

## 9. Proposed layout

```
<name>/
├── src/<name>/
│   ├── arch/                 # Sebulba core (adapted from Stoix/Mava, attributed)
│   │   ├── param_server.py  pipeline.py  lifetime.py  devices.py
│   ├── envs/
│   │   ├── adapter.py        # TwoPlayerAECAdapter: masks, rewards, learner seat
│   │   ├── loop.py           # BatchedOpponentLoop (§4.2)
│   │   └── openspiel_aec.py  # thin PettingZoo AECEnv over pyspiel
│   ├── opponents/
│   │   ├── base.py           # Opponent / OpponentProvider protocols
│   │   ├── simple.py         # Random, Scripted
│   │   ├── tabular.py        # TabularPolicyOpponent (+ load/save)
│   │   ├── providers.py      # Fixed, Mixture (SnapshotPool later)
│   │   └── cfr.py            # offline solve + export via OpenSpiel
│   ├── agents/ppo/           # masked actor-critic, loss, actor/learner threads
│   ├── eval/                 # payoff, BR gap, exploitability, async evaluator
│   ├── config.py  cli.py
├── tests/  docs/  .github/workflows/
├── LICENSE (Apache-2.0)  NOTICE
```

---

## 10. Milestones

1. **Skeleton**: uv, ruff, basedpyright, pytest-cov, mkdocs, CI all green.
2. **Poker env + opponents, no RL yet**: OpenSpiel AEC wrapper (passes `api_test`), adapter,
   Random/Tabular opponents, CFR solve and export CLI, best-response values computed.
3. **Batched opponent loop**: N envs against a fixed opponent with a random learner. Property
   tests check that the trajectory shape and reward totals match the game outcomes.
4. **Sebulba PPO** (actor threads, pipeline, param server, `shard_map` learner) on **Kuhn**
   against a fixed opponent. Smoke test in CI.
5. **Leduc against CFR(k)**: plot payoff vs best-response value; exploitability of the PPO policy.
6. **Extensions**: `MixtureProvider`, `NeuralOpponent` + `SnapshotPoolProvider` (self-play/FSP),
   other PettingZoo envs (Connect Four, Texas Hold'em), subprocess env workers.

**Success for v1:** on Leduc, PPO's average payoff against CFR(k) reaches a set fraction
(e.g. ≥ 80%) of the exact best-response value. It should do that for a couple of different k,
with all CI checks green.

---

## 11. Naming

The project is about **training against opponents of your choosing**. Availability below was
checked on PyPI on 2026-11-24.

| Name | PyPI | Rationale |
|---|---|---|
| **sparring** | ✅ free | A sparring partner is whoever you choose to train against: a CFR bot, a script, or yourself |
| **sparjax** | ✅ free | sparring + JAX; very searchable |
| **duelax** | ✅ free | Two-player focus + JAX |
| **rivax** | ✅ free | rival + JAX |
| **gauntlet-rl** | ✅ free | Run the gauntlet of opponents |
| mirrormatch | ✅ free | Now less fitting, since self-play isn't the focus |

*Recommendation:* **`sparring`**, or **`sparjax`**. Check GitHub and trademarks before choosing.

**🟡 DECISION 8.** Pick a name.

---

## 12. Decision checklist

| # | Decision | Outcome |
|---|---|---|
| 1 | Base library | ✅ Standalone; adapt Stoix + Mava Sebulba pieces (Apache-2.0) |
| 2 | Actor parallelism | ✅ Threads first. **Subprocess env workers are a planned requirement** (needed to run heavier PettingZoo envs), so the env-batch interface is designed for them from day one |
| 3 | Poker stack | ✅ OpenSpiel + our own thin PettingZoo AEC wrapper; Kuhn + Leduc (OpenSpiel rules, not PettingZoo/RLCard Leduc) |
| 4 | Config | ✅ dataclasses + tyro |
| 5 | Logging | ✅ Pluggable; console/JSONL default, W&B optional |
| 6 | Python | ✅ **≥ 3.12** (raised from 3.11: current JAX 0.12, flax 0.12 and numpy 2.5 all require 3.12) |
| 7 | NN library | ✅ flax.nnx |
| 8 | Name | ✅ **`sparjax`** |

---

### References

- Stoix: https://github.com/EdanToledo/Stoix (`stoix/systems/ppo/sebulba/ff_ppo.py`)
- Mava: https://github.com/instadeepai/Mava (`develop`, `mava/systems/ppo/sebulba/ff_ippo.py`)
- PettingZoo AEC API: https://pettingzoo.farama.org/api/aec/
- OpenSpiel: CFR (`open_spiel/python/algorithms/cfr.py`), exploitability (`.../exploitability.py`)
- Shimmy OpenSpiel wrapper: https://shimmy.farama.org/environments/open_spiel/
- Hessel et al., 2021, *Podracer architectures for scalable RL*
- Zinkevich et al., 2007, *Regret Minimization in Games with Incomplete Information* (CFR)
- Tammelin, 2014, *Solving Large Imperfect Information Games Using CFR+*
- Heinrich, Lanctot & Silver, 2015, *Fictitious Self-Play in Extensive-Form Games*
