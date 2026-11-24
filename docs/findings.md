# Findings and future directions

A summary of what sparjax can do as of 2026-11-24 (milestones M1–M7b), what we learned
building it, and where to go next. Each result links to its experiment page, which has
the exact commands, curves and caveats.

## The question

Can a Sebulba actor–learner architecture, built for single-agent RL in Stoix, train PPO
in two-player turn-based games against an opponent of our choice, including the agent's
own past snapshots? **Yes.** sparjax is a lean, standalone implementation of that idea.

## What exists

- **Environments.** Any PettingZoo AEC env (`pettingzoo:<id>`), plus our own AEC wrapper
  around OpenSpiel (`openspiel:<game>`), so the env, the CFR opponent and the exact
  evaluator share one game definition. Envs can run in the actor thread or spread over
  spawn-started subprocess workers. `--env.max-steps` truncates long games and bootstraps
  their values.
- **Opponent loop.** `BatchedOpponentLoop` resolves opponent moves so the learner sees a
  fixed-shape, single-agent stream, with random seats. Reward credit is covered by
  property tests.
- **Opponents.** Random, scripted (Go light playout), tabular CFR/CFR+ (OpenSpiel's C++
  solvers), frozen networks, and a self-play snapshot pool (latest / uniform FSP / PFSP).
- **Sebulba.** Actor threads, a versioned parameter server with an explicit `max_lag`
  (0 = synchronous, 1 = act on the previous parameters while the learner updates),
  per-actor queues, an asynchronous evaluator, checkpoints and pluggable loggers.
- **Evaluation.** Exact payoff, best response and exploitability for OpenSpiel games
  (including a reach-weighted pool average); win/draw/loss with 95% intervals by playing
  games for any env.
- **Engineering.** 178 tests at 99% branch coverage, basedpyright in recommended mode,
  ruff, CI on Python 3.12–3.14, and these docs deployed to GitHub Pages.

## Key results

| Experiment | Result | Page |
|---|---|---|
| CFR+ on Leduc | Exploitability 2.37 → 0.61 → 0.03 at 1, 10 and 50 iterations; CFR+(5–50) give a ladder of opponents of known strength | [CFR on Leduc](experiments/cfr-leduc.md) |
| PPO vs fixed opponents, Kuhn | Matches the exact best response against random and CFR(2); 99.9% of it against CFR(20) with an entropy schedule | [Kuhn](experiments/kuhn-ppo.md) |
| PPO vs CFR+, Leduc | ≥80% of the best-response value against random, CFR+(5), (10), (20) and (50) within 20M decisions | [Leduc](experiments/leduc-ppo.md) |
| Subprocess workers, Go 9x9 | 4.4× env throughput (2.3k → 10.2k steps/s) | [Go](experiments/go-sebulba.md) |
| Async acting and learning, Go 9x9 | +37–60% throughput with CPU actors, matching the `collect + update` vs `max(collect, update)` model; a 3M-step run finishes 1.39× sooner | [Go](experiments/go-sebulba.md) |
| PPO on Go 9x9 | 98% vs random in about 4 minutes; 97% (peak 99.5%) vs light playout, itself 90.5% vs random | [Go](experiments/go-sebulba.md) |
| Self-play, Leduc | FSP with entropy 0.05 cuts pool exploitability from 2.15 to 0.89 in 10M decisions, still falling; low-entropy variants stall | [Self-play](experiments/selfplay-leduc.md) |

## Lessons

**RL**

- **Entropy is the main lever when exploiting a fixed opponent.** Against strong CFR
  opponents PPO collapses onto a mediocre deterministic policy. Starting entropy at
  0.1–0.2 and decaying it to zero over most of training fixed every case we tried.
- **Self-play wants the opposite schedule: constant, higher entropy.** Decaying entropy
  makes the learner over-specialise against the pool; its own exploitability rises and
  drags the pool average back up.
- **PPO is a slow best-response oracle.** It needs 5–20M decisions to approach a best
  response to a *fixed* Leduc opponent, while the self-play pool changes every 41k
  decisions. This, not the self-play machinery, limits the self-play results.
- **Naive (latest-snapshot) self-play cycles**, as theory predicts. PFSP gave no benefit
  over uniform FSP on Leduc.

**Systems**

- **Keep actors off the learner's GPU.** With actors and learner sharing one GPU, async
  gained only about 4%; putting actors on the CPU gave 37–60%. Small-batch inference is
  cheap on the CPU and the GPU stays free for updates.
- **Python env stepping, not inference, is the bottleneck for real games.** The GIL makes
  actor threads alone useless for Go; subprocess workers are what scale.
- **Test the plumbing before any RL.** A Hypothesis property test caught rewards from a
  new game's opening move leaking into the previous game's last transition, a bug that
  would only have shown up as slightly worse learning.
- **Import every module in a fresh interpreter in CI.** A circular import passed the
  test suite because of import order; `tests/test_imports.py` now guards it.

**Environment quirks**

- GTX 10xx (Pascal) GPUs need cuDNN ≤ 9.10 for convolutions: use the `cuda12-pascal`
  extra, and `uv run --no-sync` so a plain `uv run` doesn't resync it away.
- pygame has no Python 3.14 wheel; the `go` extra uses `pygame-ce`.
- PettingZoo's Leduc differs from OpenSpiel's; we use OpenSpiel's through our own wrapper
  so exact evaluation applies.
- Subprocess workers use `spawn`, so env factories must be picklable (`EnvSpec`) and
  scripts need an `if __name__ == "__main__":` guard.

## Limitations

- Two-player, zero-sum, turn-based, discrete-action games only.
- One learner device; no recurrent policies (fine so far: Kuhn, Leduc and Go observations
  contain everything a policy needs).
- Exact evaluation only for small OpenSpiel games; everything else uses sampled play.
- Self-play on Leduc gets between CFR+(5) and CFR+(10) in exploitability, far from Nash.

## Future directions

Roughly in order of interest.

1. **Stronger self-play.** The open research question.
    - Snapshot much less often, so each snapshot is closer to a converged best response.
    - Sweep constant entropy (0.03–0.1) and longer runs: the best run was still improving.
    - NFSP-style split: keep a best-response learner, and train a separate average policy by
      supervised learning on its actions; evaluate the average policy, not the pool.
    - PSRO: solve the meta-game over snapshots instead of sampling uniformly.
    - Go self-play, scored against light playout and earlier checkpoints.
2. **More environments.** Connect Four and Texas Hold'em through the PettingZoo path, and
   an "add your own environment" guide.
3. **Recurrent policies** for games whose observations don't contain the history.
4. **W&B logger backend** behind the existing logger protocol.
5. **Multiple learner devices** with `shard_map` (only testable here with simulated CPU
   devices).
6. **Docs on Zensical** before Material for MkDocs reaches end of life on 2027-05-05.
