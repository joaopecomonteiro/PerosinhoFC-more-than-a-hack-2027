# Final check `final16`: possession model × dead-ball model (other games only)

spellF1, mean per source then over sources; Δ paired per game × noise seed: mean (se) wins/n. A row passes when its Δ > 2 se and it wins more than half the runs in EVERY column (#38).

| possession × dead ball | bar/exact | bar/fuzzy | bar/random | atd/fuzzy | atd/exact | atd_flip/fuzzy | mean |
|---|---|---|---|---|---|---|---|
| who_v1 × dead_v1 (old FINAL) | 0.550 | 0.547 | 0.536 | 0.545 | 0.545 | 0.506 | 0.538 |
| who_v1 × asf (mqf.alive, 3 seeds) | 0.598 | 0.599 | 0.588 | 0.582 | 0.576 | 0.539 | 0.580 |
| who_v1 × v8 P(dead) in our HMM | 0.626 | 0.606 | 0.580 | 0.603 | 0.611 | 0.565 | 0.599 |
| who_v1 × v8 + its semi-Markov decoder (prior 0.5) | 0.627 | 0.611 | 0.585 | 0.606 | 0.613 | 0.568 | 0.602 |
| who_v1 × mean(asf, v8) in our HMM | 0.618 | 0.614 | 0.596 | 0.599 | 0.599 | 0.558 | 0.597 |
| tcn × dead_v1 | 0.608 | 0.605 | 0.596 | 0.593 | 0.590 | 0.560 | 0.592 |
| tcn × asf (mqf.alive, 3 seeds) **(new FINAL)** | 0.658 | 0.662 | 0.656 | 0.631 | 0.622 | 0.594 | 0.637 |
| tcn × v8 P(dead) in our HMM | 0.687 | 0.669 | 0.647 | 0.651 | 0.658 | 0.616 | 0.655 |
| tcn × v8 + its semi-Markov decoder (prior 0.5) | 0.689 | 0.674 | 0.652 | 0.655 | 0.661 | 0.625 | 0.659 |
| tcn × mean(asf, v8) in our HMM | 0.680 | 0.676 | 0.666 | 0.648 | 0.645 | 0.613 | 0.654 |
| who_v1 × true alive mask (ceiling) | 0.670 | 0.661 | 0.646 | 0.657 | 0.661 | 0.610 | 0.651 |

| comparison (spellF1) | bar/exact | bar/fuzzy | bar/random | atd/fuzzy | atd/exact | atd_flip/fuzzy | passes |
|---|---|---|---|---|---|---|---|
| **new FINAL − old FINAL** | +0.094 (0.008) 11/11 | +0.099 (0.008) 11/11 | +0.105 (0.008) 11/11 | +0.090 (0.004) 22/22 | +0.082 (0.003) 22/22 | +0.090 (0.009) 11/11 | PASS |
| possession only (tcn × dead_v1) − old | +0.048 (0.008) 11/11 | +0.049 (0.008) 11/11 | +0.048 (0.008) 11/11 | +0.053 (0.003) 22/22 | +0.052 (0.004) 22/22 | +0.058 (0.009) 11/11 | PASS |
| alive only (who_v1 × asf) − old | +0.043 (0.005) 11/11 | +0.046 (0.005) 11/11 | +0.049 (0.004) 11/11 | +0.037 (0.002) 22/22 | +0.031 (0.002) 22/22 | +0.030 (0.003) 11/11 | PASS |
| new FINAL − tcn × dead_v1 (the gains add up?) | +0.046 (0.005) 11/11 | +0.050 (0.005) 11/11 | +0.056 (0.004) 11/11 | +0.037 (0.002) 22/22 | +0.030 (0.002) 22/22 | +0.032 (0.003) 11/11 | PASS |
| who_v1: v8 P(dead) in our HMM − asf | +0.012 (0.006) 8/11 | -0.000 (0.006) 5/11 | -0.014 (0.006) 2/11 | +0.009 (0.004) 16/22 | +0.021 (0.004) 20/22 | +0.016 (0.004) 11/11 | no |
| who_v1: v8 + its semi-Markov decoder (prior 0.5) − asf | +0.011 (0.007) 7/11 | +0.001 (0.006) 5/11 | -0.010 (0.006) 2/11 | +0.010 (0.004) 16/22 | +0.021 (0.004) 21/22 | +0.016 (0.005) 9/11 | no |
| who_v1: mean(asf, v8) in our HMM − asf | +0.013 (0.003) 10/11 | +0.008 (0.003) 9/11 | +0.004 (0.003) 7/11 | +0.012 (0.002) 20/22 | +0.016 (0.002) 21/22 | +0.014 (0.002) 11/11 | no |
| who_v1: v8 + its decoder − dead_v1 | +0.053 (0.008) 11/11 | +0.048 (0.006) 11/11 | +0.038 (0.005) 11/11 | +0.046 (0.004) 22/22 | +0.052 (0.004) 22/22 | +0.046 (0.006) 11/11 | PASS |
| tcn: v8 P(dead) in our HMM − asf | +0.012 (0.006) 8/11 | +0.002 (0.004) 5/11 | -0.015 (0.005) 1/11 | +0.008 (0.004) 14/22 | +0.021 (0.004) 21/22 | +0.013 (0.004) 10/11 | no |
| tcn: v8 + its semi-Markov decoder (prior 0.5) − asf | +0.011 (0.007) 8/11 | +0.004 (0.005) 6/11 | -0.012 (0.004) 2/11 | +0.009 (0.004) 15/22 | +0.022 (0.004) 21/22 | +0.017 (0.005) 10/11 | no |
| tcn: mean(asf, v8) in our HMM − asf | +0.014 (0.003) 11/11 | +0.011 (0.002) 11/11 | +0.005 (0.002) 9/11 | +0.011 (0.002) 20/22 | +0.017 (0.002) 22/22 | +0.013 (0.003) 10/11 | PASS |
| tcn: v8 + its decoder − dead_v1 | +0.057 (0.009) 11/11 | +0.054 (0.007) 11/11 | +0.044 (0.005) 11/11 | +0.046 (0.004) 22/22 | +0.052 (0.005) 22/22 | +0.049 (0.005) 11/11 | PASS |
| new FINAL − old FINAL (macroF1) | +0.042 (0.003) 11/11 | +0.046 (0.002) 11/11 | +0.049 (0.003) 11/11 | +0.034 (0.002) 22/22 | +0.029 (0.002) 22/22 | +0.040 (0.003) 11/11 | |

**PASS**: the new FINAL beats the old one in every column.
Better than asf under the new possession model, in every column: tcn: mean(asf, v8) in our HMM − asf -- consider it for FINAL.
