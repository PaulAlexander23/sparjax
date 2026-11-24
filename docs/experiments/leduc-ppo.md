# PPO against CFR opponents in Leduc poker (M5)

M5's success criterion is **≥ 80% of the exact best-response payoff against several
CFR(k) opponents**. This page records the runs that test it.

**Setup.** The learner plays OpenSpiel's Leduc poker against a fixed CFR+ average policy
after *k* iterations, in a random seat each game. The **ceiling** is the exact
best-response value against that opponent, averaged over seats (see
[CFR opponents on Leduc](cfr-leduc.md)). Every 20–50 updates a background thread queries
the network at all 936 information states and evaluates it exactly with OpenSpiel.

Common settings: 2 actor threads × 128 envs × 16-step rollouts (4,096 decisions per
update), MLP 128-128, Adam 3e-4, 4 epochs × 4 minibatches. The entropy coefficient decays
linearly to 0 over the first 80% of training. Everything ran on a desktop CPU at about
9–10k learner decisions per second per run, with four or five runs at a time.

```bash
uv run sparjax train --env.id openspiel:leduc_poker --env.num-envs 128 --ppo.hidden 128 128 \
    --opponent.kind cfr --opponent.cfr-iters 20 \
    --ppo.update.ent-coef 0.2 --ppo.ent-coef-end 0.0 --ppo.ent-decay-frac 0.8 \
    --total-timesteps 20000000 --log-dir runs/leduc_cfr20
```

## Results

| opponent | ceiling | steps | start entropy | final payoff | % of ceiling | ≥ 80%? |
|---|---:|---:|---:|---:|---:|:-:|
| random (= CFR 1) | 2.3736 | 5M | 0.1 | 2.3585 | 99.4% | ✅ |
| CFR+(5) | 1.3712 | 5M | 0.1 | 1.1364 | 82.9% | ✅ |
| CFR+(10) | 0.6104 | 5M | 0.1 | 0.4603 | 75.4% | ❌ |
| CFR+(10) | 0.6104 | 20M | 0.1 | 0.4706 | 77.1% | ❌ |
| CFR+(10) | 0.6104 | 20M | 0.2 | 0.6071 | **99.5%** | ✅ |
| CFR+(20) | 0.1737 | 5M | 0.1 | 0.1221 | 70.3% | ❌ |
| CFR+(20) | 0.1737 | 20M | 0.1 | 0.1439 | 82.8% | ✅ |
| CFR+(20) | 0.1737 | 20M | 0.2 | 0.1591 | **91.6%** | ✅ |
| CFR+(50) | 0.0341 | 5M | 0.1 | 0.0088 | 25.7% | ❌ |
| CFR+(50) | 0.0341 | 20M | 0.1 | 0.0289 | 84.6% | ✅ |

Learning curves, as % of ceiling at evenly spaced evaluations:

```text
CFR+(10), 20M, ent 0.2 :  -42% →  33% → 54% → 69% → 81% → 91% → 98% → 99% → 99%
CFR+(20), 20M, ent 0.2 : -219% → -51% → -2% → 26% → 49% → 69% → 85% → 91% → 90%
CFR+(20), 20M, ent 0.1 : -106% →  14% → 42% → 56% → 68% → 76% → 81% → 83% → 83%
CFR+(50), 20M, ent 0.1 : -688% → -172% → -71% → -16% → 22% → 61% → 76% → 81% → 84%
CFR+(10), 20M, ent 0.1 :    1% →  59% → 65% → 69% → 72% → 74% → 76% → 77% → 77%
```

## Findings

- **The criterion is met against every opponent tested:** random, CFR+(5), CFR+(10),
  CFR+(20) and CFR+(50). The harder opponents need roughly 15–20M learner decisions
  (about 30–35 minutes on a CPU).
- **Starting with more entropy is slower at first but ends higher.** For CFR+(20), starting
  at 0.2 instead of 0.1 sits below zero for the first quarter of training (it is still
  close to uniform play) and then overtakes: 91.6% vs 82.8% of the ceiling.
- **Starting entropy matters more than the step budget.** Against CFR+(10), starting at
  0.1 plateaus at 77% (59% after only 2.6M steps, then barely moving), while starting at
  0.2 reaches **99.5%**. The plateau was a local optimum caused by too little early
  exploration, not a capacity or budget limit. *Recommended schedule for Leduc:* start at
  0.2 and decay to 0 over the first 80% of 20M steps.
- **When a run falls short, the shortfall is mostly in the first seat.** For CFR+(20) at
  entropy 0.2, seat 0 gets +0.042 against a BR value of +0.065, while seat 1 gets +0.276
  against +0.282. Kuhn showed the same pattern. In the best run (CFR+(10), entropy 0.2)
  both seats are essentially at their BR values (+0.342 / +0.872 vs +0.343 / +0.878).
- **The learned policies are highly exploitable themselves** (exploitability 0.6–1.3).
  That is expected: they are counter-strategies to one opponent, not equilibrium strategies.
