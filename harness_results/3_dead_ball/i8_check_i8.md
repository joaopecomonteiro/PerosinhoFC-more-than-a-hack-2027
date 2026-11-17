# Italo's inputs in our models `i8` (possession: FINAL's owner TCN; other games only)

spellF1, mean per source then over sources; Δ paired per game × noise seed: mean (se) wins/n; PASS = Δ > 2 se and more than half the runs won in EVERY column (#38).

| dead ball | bar/exact | bar/fuzzy | bar/random | atd/fuzzy | atd/exact | atd_flip/fuzzy | mean |
|---|---|---|---|---|---|---|---|
| gbm_full_impff | 0.631 | 0.644 | 0.634 | 0.599 | 0.584 | 0.574 | 0.611 |
| gbm on full + i_ (new) | 0.651 | 0.652 | 0.642 | 0.614 | 0.606 | 0.583 | 0.625 |
| asf (FINAL's alive models) | 0.658 | 0.662 | 0.656 | 0.631 | 0.622 | 0.594 | 0.637 |
| asf on full + i_ (new) | 0.667 | 0.668 | 0.656 | 0.640 | 0.631 | 0.602 | 0.644 |
| v8 in our HMM | 0.687 | 0.669 | 0.647 | 0.651 | 0.658 | 0.616 | 0.655 |
| mean(asf, v8) = FINAL today | 0.680 | 0.676 | 0.666 | 0.648 | 0.645 | 0.613 | 0.654 |
| mean(asf_i8, v8) | 0.685 | 0.678 | 0.666 | 0.652 | 0.650 | 0.619 | 0.658 |

| comparison (spellF1) | bar/exact | bar/fuzzy | bar/random | atd/fuzzy | atd/exact | atd_flip/fuzzy | passes |
|---|---|---|---|---|---|---|---|
| gbm_i8 − gbm_full (do the inputs carry signal?) | +0.010 (0.004) 8/11 | +0.005 (0.002) 8/11 | +0.005 (0.003) 7/11 | +0.009 (0.002) 18/22 | +0.012 (0.002) 20/22 | +0.006 (0.002) 9/11 | no |
| asf_i8 − asf | +0.006 (0.003) 8/11 | +0.006 (0.003) 9/11 | +0.000 (0.003) 5/11 | +0.007 (0.001) 20/22 | +0.007 (0.001) 21/22 | +0.004 (0.002) 8/11 | no |
| **mean(asf_i8, v8) − FINAL today** | +0.005 (0.002) 8/11 | +0.003 (0.002) 8/11 | +0.001 (0.002) 5/11 | +0.004 (0.001) 19/22 | +0.004 (0.001) 16/22 | +0.004 (0.002) 7/11 | no |
| asf_i8 alone − FINAL today | -0.008 (0.002) 1/11 | -0.004 (0.003) 3/11 | -0.005 (0.003) 3/11 | -0.004 (0.002) 6/22 | -0.010 (0.002) 1/22 | -0.010 (0.002) 1/11 | no |

Nothing beats FINAL in every column: keep FINAL as it is.
