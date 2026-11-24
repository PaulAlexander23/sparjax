# CFR opponents on Leduc poker

How exploitable is OpenSpiel's CFR average policy after *k* iterations? This sets how
much a learner can possibly win against a CFR(k) opponent, and therefore which *k* make
useful training opponents.

Reproduce one row with:

```bash
uv run sparjax cfr-solve --game leduc_poker --algorithm cfr_plus --iters 20
```

The values are exact (OpenSpiel best response), in chips per hand. "BR seat *p*" is the
most a best-responding learner in seat *p* wins per hand against CFR(k) in the other seat.
With seats randomised, the ceiling is the mean of the two BR values, which equals the
exploitability. A Nash-vs-Nash game is worth about −0.0856 to seat 0 and +0.0856 to seat 1.

| algorithm | k | exploitability | BR seat 0 | BR seat 1 | solve (s) |
|---|---:|---:|---:|---:|---:|
| cfr | 1 | 2.3736 | +2.0875 | +2.6597 | 0.01 |
| cfr | 5 | 1.4713 | +0.6790 | +2.2637 | 0.05 |
| cfr | 10 | 0.8886 | +0.3264 | +1.4508 | 0.10 |
| cfr | 20 | 0.4531 | +0.2149 | +0.6914 | 0.21 |
| cfr | 50 | 0.1891 | +0.0575 | +0.3206 | 0.55 |
| cfr | 100 | 0.0957 | −0.0159 | +0.2073 | 1.02 |
| cfr | 1000 | 0.0118 | −0.0770 | +0.1006 | 10.91 |
| cfr_plus | 1 | 2.3736 | +2.0875 | +2.6597 | 0.01 |
| cfr_plus | 5 | 1.3712 | +0.4728 | +2.2697 | 0.06 |
| cfr_plus | 10 | 0.6104 | +0.3429 | +0.8780 | 0.12 |
| cfr_plus | 20 | 0.1737 | +0.0653 | +0.2820 | 0.24 |
| cfr_plus | 50 | 0.0341 | −0.0582 | +0.1264 | 0.63 |
| cfr_plus | 100 | 0.0134 | −0.0759 | +0.1028 | 1.27 |
| cfr_plus | 1000 | 0.0003 | −0.0855 | +0.0860 | 12.07 |

Measured on 2026-11-24 with open_spiel 2.0.2 on a desktop CPU. An exact best-response
evaluation takes about 0.45 s per policy.

## Takeaways

- **CFR after one iteration is the uniform random policy.** Its exploitability, 2.37, is
  the "beat a random player" ceiling.
- **CFR+ converges far faster.** By k = 100 it is within 0.013 of Nash, and the possible
  gain is too small to measure in a reasonable number of hands. Leduc payoffs have a
  standard deviation of about 4 chips per hand (measured: 4.5 for CFR(1) self-play, 4.0
  for CFR+(20)). Resolving a 0.01 edge at 2 standard errors takes about 640k hands.
- **Useful training opponents are roughly CFR+(10–50).** Their exploitability is 0.03–0.6
  chips per hand: clearly beatable, but not trivially. M5 will compare PPO's payoff against
  these opponents with the BR ceilings in this table.
