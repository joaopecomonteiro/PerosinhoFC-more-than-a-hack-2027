# Fusion suite `stoppage`

- Inputs `harness_results/fusion_preds`: 11 games (idsse, metrica, pff), columns atd_exact, atd_flip_fuzzy, atd_fuzzy, bar_exact, bar_fuzzy, bar_random, seeds 0, 1; streams: dead pair True, TCN seeds True, boundary True, dead boundary True, who_v1 False, FINAL labels True; fps [25.0, 29.97]
- Every number is HELD OUT: per game, the config (or for PIPELINE the decoder AND config) best on the other games (spellF1 per column, mean per source then over sources, then over columns; tie macroF1) -- final_check's averaging
- Baseline `final`; official metric `vendor/mtah2027/evaluation/possession_metrics.py`; 2.4 min, numba=True

## Sanity: labels identical to the exported FINAL labels

- `argmax`@0: 0/88 files
- `minrun`@0: 0/88 files
- `prod`: 88/88 files
- `stoppage`@0: 0/88 files

## Held-out spellF1 per column (mean per source, then over sources)

| decoder | atd_exact | atd_flip_fuzzy | atd_fuzzy | bar_exact | bar_fuzzy | bar_random | **mean** | macroF1 | adF1 | count err | folds agree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| final | 0.645 | 0.613 | 0.648 | 0.680 | 0.676 | 0.666 | **0.654** | 0.906 | 0.966 | -94 | 1.00 |
| prod | 0.645 | 0.613 | 0.648 | 0.680 | 0.676 | 0.666 | **0.654** | 0.906 | 0.966 | -94 | 1.00 |
| stoppage | 0.642 | 0.615 | 0.649 | 0.677 | 0.665 | 0.665 | **0.652** | 0.905 | 0.966 | -48 | 1.00 |
| minrun | 0.641 | 0.611 | 0.648 | 0.675 | 0.662 | 0.663 | **0.650** | 0.904 | 0.965 | -48 | 0.91 |
| PIPELINE | 0.640 | 0.611 | 0.647 | 0.675 | 0.663 | 0.662 | **0.650** | 0.904 | 0.965 | -47 | 0.91 |
| stoppage@0 | 0.625 | 0.600 | 0.628 | 0.659 | 0.648 | 0.653 | **0.635** | 0.904 | 0.965 | -16 | 1.00 |
| minrun@0 | 0.620 | 0.597 | 0.626 | 0.654 | 0.644 | 0.647 | **0.631** | 0.904 | 0.965 | -10 | 1.00 |
| argmax | 0.582 | 0.553 | 0.583 | 0.622 | 0.611 | 0.613 | **0.594** | 0.904 | 0.965 | +42 | 1.00 |
| argmax@0 | 0.582 | 0.553 | 0.583 | 0.622 | 0.611 | 0.613 | **0.594** | 0.904 | 0.965 | +42 | 1.00 |

## Paired vs `final` (held out): Δ spellF1 mean (se) wins/n over game × seed runs; PASS = Δ > 2 se and wins > n/2 in EVERY column (#38)

| decoder | atd_exact | atd_flip_fuzzy | atd_fuzzy | bar_exact | bar_fuzzy | bar_random | passes |
|---|---|---|---|---|---|---|---|
| prod | +0.000 (0.000) 0/22 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/22 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/11 | no |
| stoppage | -0.005 (0.002) 10/22 | +0.002 (0.004) 7/11 | -0.002 (0.003) 10/22 | +0.001 (0.003) 6/11 | -0.004 (0.004) 5/11 | +0.002 (0.003) 8/11 | no |
| minrun | -0.005 (0.002) 10/22 | +0.002 (0.004) 6/11 | -0.001 (0.003) 10/22 | +0.000 (0.003) 6/11 | -0.004 (0.004) 5/11 | +0.002 (0.003) 8/11 | no |
| PIPELINE | -0.005 (0.002) 9/22 | +0.001 (0.004) 6/11 | -0.002 (0.003) 10/22 | +0.000 (0.003) 6/11 | -0.004 (0.004) 5/11 | +0.001 (0.003) 8/11 | no |
| stoppage@0 | -0.023 (0.003) 0/22 | -0.012 (0.004) 2/11 | -0.021 (0.003) 3/22 | -0.018 (0.003) 1/11 | -0.021 (0.004) 1/11 | -0.013 (0.003) 1/11 | no |
| minrun@0 | -0.026 (0.003) 0/22 | -0.016 (0.004) 2/11 | -0.023 (0.003) 1/22 | -0.022 (0.003) 0/11 | -0.025 (0.004) 0/11 | -0.017 (0.004) 1/11 | no |
| argmax | -0.063 (0.003) 0/22 | -0.061 (0.005) 0/11 | -0.064 (0.004) 0/22 | -0.056 (0.004) 0/11 | -0.056 (0.006) 0/11 | -0.049 (0.005) 0/11 | no |
| argmax@0 | -0.063 (0.003) 0/22 | -0.061 (0.005) 0/11 | -0.064 (0.004) 0/22 | -0.056 (0.004) 0/11 | -0.056 (0.006) 0/11 | -0.049 (0.005) 0/11 | no |

`PIPELINE` = the whole suite as a procedure: inside each fold the decoder AND its config are chosen on the other games. It is the honest estimate of 'ship the suite's winner'; a single decoder's row is optimistic by the choice among decoders. `@0` = config 0 untuned (hmm3_bnd@0 = FINAL). `count err` = predicted − true spells (negative = merging). Low `folds agree` = flat optimum: prefer config 0 or round values.

## Match rate at IoU 0.5 by true possession length (final configs, pooled)

| decoder | 0-2 s | 2-5 s | 5-15 s | 15-inf s |
|---|---|---|---|---|
| final | 0.143 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| prod | 0.143 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| stoppage | 0.188 (8616) | 0.514 (9440) | 0.776 (8984) | 0.909 (6832) |
| minrun | 0.182 (8616) | 0.517 (9440) | 0.777 (8984) | 0.908 (6832) |
| stoppage@0 | 0.210 (8616) | 0.514 (9440) | 0.781 (8984) | 0.909 (6832) |
| minrun@0 | 0.211 (8616) | 0.515 (9440) | 0.781 (8984) | 0.907 (6832) |
| argmax | 0.209 (8616) | 0.518 (9440) | 0.783 (8984) | 0.908 (6832) |
| argmax@0 | 0.209 (8616) | 0.518 (9440) | 0.783 (8984) | 0.908 (6832) |

## Final configs (chosen on all games -- what would ship)

- `final` → `final` cfg 0: `{}`
- `prod` → `prod` cfg 0: `{"hook": "mqf.decode:fusion_baseline"}`
- `stoppage` → `stoppage` cfg 2: `{"min_between_s": 0.6136, "min_dead_s": 4.5371, "min_s": 1.384}`
- `minrun` → `minrun` cfg 4: `{"min_s": 1.4955}`
- `PIPELINE` → `stoppage` cfg 2: `{"min_between_s": 0.6136, "min_dead_s": 4.5371, "min_s": 1.384}`
- `stoppage@0` → `stoppage` cfg 0: `{"min_between_s": 2.0, "min_dead_s": 2.0, "min_s": 0.5}`
- `minrun@0` → `minrun` cfg 0: `{"min_s": 0.5}`
- `argmax` → `argmax` cfg 0: `{}`
- `argmax@0` → `argmax` cfg 0: `{}`
