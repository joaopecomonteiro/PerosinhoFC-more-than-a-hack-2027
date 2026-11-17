# Final check `fusion16`: possession model × dead-ball model (other games only)

spellF1, mean per source then over sources; Δ paired per game × noise seed: mean (se) wins/n. A row passes when its Δ > 2 se and it wins more than half the runs in EVERY column (#38).

| possession × dead ball | bar/exact | bar/fuzzy | bar/random | atd/fuzzy | atd/exact | atd_flip/fuzzy | mean |
|---|---|---|---|---|---|---|---|
| tcn × dead_v1 | 0.608 | 0.605 | 0.596 | 0.593 | 0.590 | 0.560 | 0.592 |
| tcn × asf (mqf.alive, 3 seeds) | 0.658 | 0.662 | 0.656 | 0.631 | 0.622 | 0.594 | 0.637 |
| tcn × v8 P(dead) in our HMM | 0.687 | 0.669 | 0.647 | 0.651 | 0.658 | 0.616 | 0.655 |
| tcn × v8 + its semi-Markov decoder (prior 0.5) | 0.689 | 0.674 | 0.652 | 0.655 | 0.661 | 0.625 | 0.659 |
| tcn × mean(asf, v8) in our HMM **(new FINAL)** | 0.680 | 0.676 | 0.666 | 0.648 | 0.645 | 0.613 | 0.654 |

| comparison (spellF1) | bar/exact | bar/fuzzy | bar/random | atd/fuzzy | atd/exact | atd_flip/fuzzy | passes |
|---|---|---|---|---|---|---|---|
| new FINAL − tcn × dead_v1 (the gains add up?) | +0.060 (0.006) 11/11 | +0.061 (0.005) 11/11 | +0.062 (0.004) 11/11 | +0.048 (0.002) 22/22 | +0.047 (0.003) 22/22 | +0.046 (0.004) 11/11 | PASS |
| tcn: v8 P(dead) in our HMM − asf | +0.012 (0.006) 8/11 | +0.002 (0.004) 5/11 | -0.015 (0.005) 1/11 | +0.008 (0.004) 14/22 | +0.021 (0.004) 21/22 | +0.013 (0.004) 10/11 | no |
| tcn: v8 + its semi-Markov decoder (prior 0.5) − asf | +0.011 (0.007) 8/11 | +0.004 (0.005) 6/11 | -0.012 (0.004) 2/11 | +0.009 (0.004) 15/22 | +0.022 (0.004) 21/22 | +0.017 (0.005) 10/11 | no |
| tcn: mean(asf, v8) in our HMM − asf | +0.014 (0.003) 11/11 | +0.011 (0.002) 11/11 | +0.005 (0.002) 9/11 | +0.011 (0.002) 20/22 | +0.017 (0.002) 22/22 | +0.013 (0.003) 10/11 | PASS |
| tcn: v8 + its decoder − dead_v1 | +0.057 (0.009) 11/11 | +0.054 (0.007) 11/11 | +0.044 (0.005) 11/11 | +0.046 (0.004) 22/22 | +0.052 (0.005) 22/22 | +0.049 (0.005) 11/11 | PASS |
| new FINAL − old FINAL (macroF1) | — | — | — | — | — | — | |

Old FINAL not in this run (--whos): no old-vs-new verdict.
Better than asf under the new possession model, in every column: tcn: mean(asf, v8) in our HMM − asf -- consider it for FINAL.
