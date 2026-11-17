# Fusion suite `f162229_deep`

- Inputs `harness_results/fusion_preds`: 11 games (idsse, metrica, pff), columns atd_exact, atd_flip_fuzzy, atd_fuzzy, bar_exact, bar_fuzzy, bar_random, seeds 0, 1; streams: dead pair True, TCN seeds True, boundary True, dead boundary True, who_v1 False, FINAL labels True; fps [25.0, 29.97]
- Every number is HELD OUT: per game, the config (or for PIPELINE the decoder AND config) best on the other games (spellF1 per column, mean per source then over sources, then over columns; tie macroF1) -- final_check's averaging
- Baseline `final`; official metric `C:\Users\pc\Documents\GitHub\PerosinhoFC-more-than-a-hack-2027\JS\vendor\mtah2027\evaluation\possession_metrics.py`; 10.9 min, numba=True

## Sanity: labels identical to the exported FINAL labels

- `fb_hsmm`@0: 0/88 files
- `hsmm_emp`@0: 0/88 files
- `hsmm_soft`@0: 6/88 files
- `prod`: 88/88 files

## Held-out spellF1 per column (mean per source, then over sources)

| decoder | atd_exact | atd_flip_fuzzy | atd_fuzzy | bar_exact | bar_fuzzy | bar_random | **mean** | macroF1 | adF1 | count err | folds agree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hsmm_soft | 0.664 | 0.634 | 0.664 | 0.692 | 0.674 | 0.669 | **0.666** | 0.914 | 0.970 | -89 | 1.00 |
| PIPELINE | 0.664 | 0.634 | 0.664 | 0.692 | 0.674 | 0.669 | **0.666** | 0.914 | 0.970 | -89 | 1.00 |
| hsmm_emp | 0.663 | 0.631 | 0.663 | 0.691 | 0.675 | 0.669 | **0.665** | 0.914 | 0.970 | -90 | 1.00 |
| fb_hsmm | 0.662 | 0.633 | 0.662 | 0.690 | 0.674 | 0.666 | **0.664** | 0.913 | 0.970 | -86 | 1.00 |
| fb_hsmm@0 | 0.647 | 0.615 | 0.649 | 0.680 | 0.676 | 0.671 | **0.656** | 0.907 | 0.966 | -86 | 1.00 |
| hsmm_soft@0 | 0.646 | 0.613 | 0.648 | 0.680 | 0.676 | 0.666 | **0.655** | 0.906 | 0.966 | -95 | 1.00 |
| hsmm_emp@0 | 0.646 | 0.612 | 0.648 | 0.680 | 0.676 | 0.665 | **0.655** | 0.906 | 0.966 | -95 | 1.00 |
| final | 0.645 | 0.613 | 0.648 | 0.680 | 0.676 | 0.666 | **0.654** | 0.906 | 0.966 | -94 | 1.00 |
| prod | 0.645 | 0.613 | 0.648 | 0.680 | 0.676 | 0.666 | **0.654** | 0.906 | 0.966 | -94 | 1.00 |

## Paired vs `final` (held out): Δ spellF1 mean (se) wins/n over game × seed runs; PASS = Δ > 2 se and wins > n/2 in EVERY column (#38)

| decoder | atd_exact | atd_flip_fuzzy | atd_fuzzy | bar_exact | bar_fuzzy | bar_random | passes |
|---|---|---|---|---|---|---|---|
| hsmm_soft | +0.014 (0.002) 21/22 | +0.020 (0.004) 10/11 | +0.014 (0.002) 20/22 | +0.009 (0.002) 10/11 | +0.003 (0.003) 8/11 | +0.005 (0.003) 7/11 | no |
| PIPELINE | +0.014 (0.002) 21/22 | +0.020 (0.004) 10/11 | +0.014 (0.002) 20/22 | +0.009 (0.002) 10/11 | +0.003 (0.003) 8/11 | +0.005 (0.003) 7/11 | no |
| hsmm_emp | +0.013 (0.002) 21/22 | +0.017 (0.004) 10/11 | +0.012 (0.002) 20/22 | +0.007 (0.002) 10/11 | +0.004 (0.002) 9/11 | +0.005 (0.003) 8/11 | no |
| fb_hsmm | +0.011 (0.002) 19/22 | +0.017 (0.004) 10/11 | +0.011 (0.002) 19/22 | +0.005 (0.002) 10/11 | +0.001 (0.002) 7/11 | +0.001 (0.003) 6/11 | no |
| fb_hsmm@0 | +0.002 (0.001) 15/22 | +0.004 (0.002) 7/11 | +0.000 (0.001) 13/22 | +0.000 (0.001) 5/11 | +0.001 (0.001) 8/11 | +0.006 (0.002) 10/11 | no |
| hsmm_soft@0 | +0.000 (0.000) 8/22 | +0.000 (0.000) 5/11 | +0.000 (0.000) 9/22 | +0.001 (0.000) 9/11 | +0.001 (0.000) 6/11 | +0.000 (0.000) 5/11 | no |
| hsmm_emp@0 | +0.000 (0.000) 12/22 | -0.001 (0.001) 4/11 | -0.000 (0.000) 10/22 | +0.001 (0.000) 10/11 | +0.001 (0.001) 8/11 | +0.000 (0.000) 5/11 | no |
| prod | +0.000 (0.000) 0/22 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/22 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/11 | no |

`PIPELINE` = the whole suite as a procedure: inside each fold the decoder AND its config are chosen on the other games. It is the honest estimate of 'ship the suite's winner'; a single decoder's row is optimistic by the choice among decoders. `@0` = config 0 untuned (hmm3_bnd@0 = FINAL). `count err` = predicted − true spells (negative = merging). Low `folds agree` = flat optimum: prefer config 0 or round values.

## Match rate at IoU 0.5 by true possession length (final configs, pooled)

| decoder | 0-2 s | 2-5 s | 5-15 s | 15-inf s |
|---|---|---|---|---|
| hsmm_soft | 0.151 (8616) | 0.467 (9440) | 0.767 (8984) | 0.902 (6832) |
| hsmm_emp | 0.149 (8616) | 0.464 (9440) | 0.764 (8984) | 0.901 (6832) |
| fb_hsmm | 0.152 (8616) | 0.467 (9440) | 0.768 (8984) | 0.901 (6832) |
| fb_hsmm@0 | 0.152 (8616) | 0.459 (9440) | 0.756 (8984) | 0.907 (6832) |
| hsmm_soft@0 | 0.142 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| hsmm_emp@0 | 0.141 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| final | 0.143 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| prod | 0.143 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |

## Final configs (chosen on all games -- what would ship)

- `hsmm_soft` → `hsmm_soft` cfg 17: `{"alpha": 0.5631, "alpha_d": 0.0, "bd": 0.3224, "c_ad": 7.4112, "c_da": 7.0822, "c_ha": 18.3424, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 5.0447, "lam_spell": 3.6561, "m_dead": 2.7881, "m_spell": 1.4082, "power": 1.0, "v8_share": 0.4876, "w_filled": 0.4107, "wd": 0.5, "wh": 1.6725, "who_clip": 0.0137}`
- `PIPELINE` → `hsmm_soft` cfg 17: `{"alpha": 0.5631, "alpha_d": 0.0, "bd": 0.3224, "c_ad": 7.4112, "c_da": 7.0822, "c_ha": 18.3424, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 5.0447, "lam_spell": 3.6561, "m_dead": 2.7881, "m_spell": 1.4082, "power": 1.0, "v8_share": 0.4876, "w_filled": 0.4107, "wd": 0.5, "wh": 1.6725, "who_clip": 0.0137}`
- `hsmm_emp` → `hsmm_emp` cfg 53: `{"alpha": 0.6364, "alpha_d": 0.0, "bd": 0.3818, "c_ad": 6.1329, "c_da": 11.7419, "c_ha": 17.2229, "cap_dead": 5.6032, "cap_spell": 3.8602, "dead_mix": "prob_mean", "fps_scale": true, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 1.2743, "lam_spell": 2.1787, "v8_share": 0.5378, "w_filled": 0.4272, "wd": 0.6617, "wh": 1.3605, "who_clip": 0.0459}`
- `fb_hsmm` → `fb_hsmm` cfg 11: `{"alpha": 0.4035, "alpha_d": 0.0, "bd": 0.4622, "c2": 2.2693, "c_ad": 14.672, "c_da": 9.208, "c_ha": 11.4028, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "logit_mean", "kappa": 1.0, "lam_dead": 7.3812, "lam_spell": 10.3873, "m_dead": 3.5166, "m_spell": 1.4719, "power": 1.0, "v8_share": 0.3104, "w_filled": 0.4077, "wd": 0.6492, "wh": 1.2139, "who_clip": 0.0319, "wpost": 1.0755}`
- `fb_hsmm@0` → `fb_hsmm` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c2": 2.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 8.0, "lam_spell": 8.0, "m_dead": 2.0, "m_spell": 1.0, "power": 1.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02, "wpost": 1.0}`
- `hsmm_soft@0` → `hsmm_soft` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 8.0, "lam_spell": 8.0, "m_dead": 2.0, "m_spell": 1.0, "power": 1.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `hsmm_emp@0` → `hsmm_emp` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "cap_dead": 6.0, "cap_spell": 4.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 1.0, "lam_spell": 1.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `final` → `final` cfg 0: `{}`
- `prod` → `prod` cfg 0: `{"hook": "mqf.decode:fusion_baseline"}`
