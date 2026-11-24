# PPO against fixed opponents in Kuhn poker (M4)

The first end-to-end runs of Sebulba PPO. The learner plays Kuhn poker against a fixed
opponent, in a random seat each game. After every 10 updates the network is queried at
all 12 information states and evaluated **exactly** with OpenSpiel. The **ceiling** is
the exact best-response value against that opponent, averaged over seats: no policy can
do better on average.

Reproduce a row with:

```bash
uv run sparjax train --env.id openspiel:kuhn_poker --opponent.kind cfr --opponent.cfr-iters 5 \
    --opponent.cfr-algorithm cfr --total-timesteps 500000
```

Default settings: 2 actor threads × 64 envs × 16-step rollouts (2,048 decisions per
update), MLP 64-64, Adam 3e-4, 4 epochs × 4 minibatches, entropy 0.01, `max_lag = 1`.
Everything ran on a desktop CPU. Two seeds per row.

| opponent | ceiling | final payoff | gap | steps to 90% of ceiling | wall time |
|---|---:|---:|---:|---:|---:|
| random | 0.4583 | 0.4582 | 0.0% | 61k | 60 s |
| CFR(2) | 0.2708 | 0.2708 | 0.0% | 61k | 57 s |
| CFR(5) | 0.1214 | 0.1192 | 1.8% | 82k | 53 s |
| CFR(20) | 0.0407 | 0.0146 | **64%** | — | 52 s |

Throughput is about 10–12k learner decisions per second, including exact evaluation
every 10 updates.

## Findings

- **The pipeline is correct end to end.** Against exploitable opponents PPO converges to
  the exact best response (random, CFR(2)) or within 2% of it (CFR(5)).
- **Small edges are hard: PPO collapses to a deterministic policy too early.** Against
  CFR(20) it plays seat 1 optimally (+0.0826 vs a BR value of +0.0827), but in seat 0 it
  locks in two mistakes:
    - folding the Queen after check–bet, where calling is better (+0.033 per game in
      seat 0 on its own);
    - betting the King first, where checking to trap is better.

  A 4× longer run (2M steps) changes nothing: entropy is ≈ 0 and the payoff is frozen at
  0.0146. Removing advantage normalisation and the entropy bonus doesn't help either.
- **More entropy regularisation unsticks it.** It is slow, but keeps improving: at
  `ent_coef = 0.03`, seat 0 goes −0.058 → −0.024 in 600k steps (BR −0.0014); at 0.1 it
  goes −0.072 → −0.027. This is an exploration problem, not a pipeline bug. It matters for
  Leduc (M5), where useful CFR opponents leave edges of 0.03–0.6 chips per hand against a
  payoff standard deviation of about 4.

## Follow-up (M5): entropy schedules fix it

M5 added a linear entropy-coefficient schedule (`--ppo.ent-coef-end`,
`--ppo.ent-decay-frac`) and optional learning-rate annealing (`--ppo.anneal-lr`). Results
against CFR(20), 2M steps each:

| entropy schedule | other | final payoff | % of ceiling |
|---|---|---:|---:|
| 0.1 → 0 over the first 80% | — | 0.0406 | **99.9%** |
| 0.03 constant | LR annealed to 0 | 0.0309 | 76% |
| 0.05 → 0 over 100% | — | 0.0146 | 36% |
| 0.05 → 0 over 100% | LR 1e-3, annealed | 0.0146 | 36% |

Exploration has to be strong *early*. Starting at 0.05 already lets the policy lock into
the same local optimum as the constant 0.01 default. Starting at 0.1 and decaying finds
the exact best response.

```bash
uv run sparjax train --env.id openspiel:kuhn_poker --opponent.kind cfr --opponent.cfr-iters 20 \
    --opponent.cfr-algorithm cfr --total-timesteps 2000000 \
    --ppo.update.ent-coef 0.1 --ppo.ent-coef-end 0.0 --ppo.ent-decay-frac 0.8
```
