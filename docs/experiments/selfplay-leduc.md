# Self-play on Leduc poker (M7b)

Self-play is just another opponent provider. `--opponent.kind selfplay` serves frozen
snapshots of the learner from a pool, and the learner adds a snapshot every
`--opponent.selfplay.snapshot-every` updates. Leduc lets us measure progress *exactly*:

* `eval/exploitability`: exploitability of the learner's current policy;
* `eval/pool_exploitability`: exploitability of the uniform *mixture* of all snapshots.
  This is the average strategy that fictitious self-play drives towards Nash. It is
  computed with reach weights (OpenSpiel's `PolicyAggregator`), not by averaging action
  probabilities state by state.

For scale: uniform random play is 2.37, CFR+(5) is 1.37, CFR+(10) is 0.61, and Nash is 0
(see [CFR opponents on Leduc](cfr-leduc.md)).

```bash
uv run sparjax train --env.id openspiel:leduc_poker --env.num-envs 128 --ppo.hidden 128 128 \
    --opponent.kind selfplay --opponent.selfplay.sampler uniform \
    --opponent.selfplay.snapshot-every 10 --ppo.update.ent-coef 0.05 \
    --total-timesteps 10000000
```

**Setup.** 2 actors × 128 envs × 16-step rollouts, MLP 128-128, a snapshot every 10
updates (about 41k decisions), 10M decisions per run, five runs in parallel on a CPU.

## Results

| sampler | entropy | pool exploitability, start → end | best | current policy, end |
|---|---|---|---:|---:|
| latest (naive self-play) | 0.01 | 2.15 → 1.29 | 1.29 | 1.51 (swings 1.0–1.7) |
| uniform (FSP) | 0.01 | 2.18 → 1.68 | 1.44 | 2.54 |
| PFSP | 0.01 | 2.18 → 1.64 | 1.46 | 2.41 |
| uniform (FSP) | **0.05 constant** | 2.15 → **0.89** | **0.89** | 1.60 |
| uniform (FSP) | 0.1 → 0.01 over 50% | 2.11 → 1.25 | 0.90 | 1.85 |

Pool exploitability over training (uniform FSP, entropy 0.05): 2.15 → 1.22 → 1.07 → 1.03 →
0.98 → 0.95 → 0.94 → 0.92 → 0.89, still falling at the end.

## Findings

- **The machinery works.** Snapshots, samplers, opponent statistics and exact pool
  exploitability all behave as designed (and are unit-tested), and every variant improves
  on random play.
- **Only FSP with constant, fairly high entropy keeps improving.** It reaches 0.89, between
  CFR+(5) and CFR+(10), and was still falling after 10M decisions.
- **With low entropy, the learner gets *more* exploitable over time.** At 0.01, the current
  policy's exploitability rises to 2.4–2.5. It over-specialises against the pool and stops
  adapting, the pool fills with near-copies of that policy, and the pool average drifts
  back up. Decaying the entropy shows the same pattern once it gets low: best at 3.8M,
  then worse.
- **Naive self-play cycles, as theory predicts.** The `latest` learner swings between 1.0
  and 1.7. Its pool average still falls slowly because the pool keeps the history.
- **PFSP gives no benefit here.** Against a single fixed learner, favouring hard opponents
  behaves like uniform sampling at the same entropy.
- **PPO is a weak best-response oracle at this snapshot rate.** Against fixed opponents
  (M5) PPO needed 5–20M decisions to get close to a best response, but here the target
  moves every 41k decisions. Snapshotting less often (best responses that are closer to
  converged), or the NFSP/PSRO-style split into a best-response learner and an average
  policy, are the obvious next steps.
