# Fusion suite `f162229_quick`

- Inputs `harness_results/fusion_preds`: 11 games (idsse, metrica, pff), columns atd_exact, atd_flip_fuzzy, atd_fuzzy, bar_exact, bar_fuzzy, bar_random, seeds 0, 1; streams: dead pair True, TCN seeds True, boundary True, dead boundary True, who_v1 False, FINAL labels True; fps [25.0, 29.97]
- Every number is HELD OUT: per game, the config (or for PIPELINE the decoder AND config) best on the other games (spellF1 per column, mean per source then over sources, then over columns; tie macroF1) -- final_check's averaging
- Baseline `final`; official metric `C:\Users\pc\Documents\GitHub\PerosinhoFC-more-than-a-hack-2027\JS\vendor\mtah2027\evaluation\possession_metrics.py`; 1.9 min, numba=True

## Sanity: labels identical to the exported FINAL labels

- `argmax`@0: 0/88 files
- `asrf`@0: 0/88 files
- `fb_hsmm`@0: 0/88 files
- `hier`@0: 0/88 files
- `hmm3`@0: 0/88 files
- `hmm3_bnd`@0: 88/88 files
- `hsmm_emp`@0: 0/88 files
- `hsmm_soft`@0: 6/88 files
- `prod`: 88/88 files

## Held-out spellF1 per column (mean per source, then over sources)

| decoder | atd_exact | atd_flip_fuzzy | atd_fuzzy | bar_exact | bar_fuzzy | bar_random | **mean** | macroF1 | adF1 | count err | folds agree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hsmm_emp | 0.661 | 0.631 | 0.662 | 0.691 | 0.671 | 0.670 | **0.664** | 0.914 | 0.970 | -79 | 1.00 |
| fb_hsmm | 0.662 | 0.633 | 0.662 | 0.690 | 0.674 | 0.666 | **0.664** | 0.913 | 0.970 | -86 | 1.00 |
| hsmm_soft | 0.660 | 0.628 | 0.662 | 0.689 | 0.674 | 0.674 | **0.664** | 0.913 | 0.970 | -84 | 1.00 |
| PIPELINE | 0.660 | 0.629 | 0.660 | 0.688 | 0.672 | 0.670 | **0.663** | 0.913 | 0.970 | -83 | 0.45 |
| hier | 0.659 | 0.623 | 0.658 | 0.686 | 0.676 | 0.660 | **0.660** | 0.913 | 0.970 | -97 | 1.00 |
| hmm3 | 0.653 | 0.621 | 0.654 | 0.685 | 0.672 | 0.667 | **0.659** | 0.911 | 0.969 | -82 | 0.82 |
| hmm3_bnd | 0.655 | 0.627 | 0.654 | 0.685 | 0.667 | 0.660 | **0.658** | 0.910 | 0.966 | -82 | 0.91 |
| fb_hsmm@0 | 0.647 | 0.615 | 0.649 | 0.680 | 0.676 | 0.671 | **0.656** | 0.907 | 0.966 | -86 | 1.00 |
| hsmm_soft@0 | 0.646 | 0.613 | 0.648 | 0.680 | 0.676 | 0.666 | **0.655** | 0.906 | 0.966 | -95 | 1.00 |
| hsmm_emp@0 | 0.646 | 0.612 | 0.648 | 0.680 | 0.676 | 0.665 | **0.655** | 0.906 | 0.966 | -95 | 1.00 |
| final | 0.645 | 0.613 | 0.648 | 0.680 | 0.676 | 0.666 | **0.654** | 0.906 | 0.966 | -94 | 1.00 |
| hmm3_bnd@0 | 0.645 | 0.613 | 0.648 | 0.680 | 0.676 | 0.666 | **0.654** | 0.906 | 0.966 | -94 | 1.00 |
| prod | 0.645 | 0.613 | 0.648 | 0.680 | 0.676 | 0.666 | **0.654** | 0.906 | 0.966 | -94 | 1.00 |
| hier@0 | 0.642 | 0.609 | 0.645 | 0.678 | 0.673 | 0.664 | **0.652** | 0.905 | 0.965 | -92 | 1.00 |
| asrf | 0.647 | 0.619 | 0.650 | 0.676 | 0.658 | 0.652 | **0.650** | 0.910 | 0.966 | -79 | 0.91 |
| hmm3@0 | 0.641 | 0.604 | 0.642 | 0.670 | 0.666 | 0.654 | **0.646** | 0.907 | 0.967 | -116 | 1.00 |
| asrf@0 | 0.633 | 0.604 | 0.637 | 0.669 | 0.657 | 0.660 | **0.643** | 0.904 | 0.966 | -52 | 1.00 |
| argmax | 0.582 | 0.553 | 0.583 | 0.622 | 0.611 | 0.613 | **0.594** | 0.904 | 0.965 | +42 | 1.00 |
| argmax@0 | 0.582 | 0.553 | 0.583 | 0.622 | 0.611 | 0.613 | **0.594** | 0.904 | 0.965 | +42 | 1.00 |

## Paired vs `final` (held out): Δ spellF1 mean (se) wins/n over game × seed runs; PASS = Δ > 2 se and wins > n/2 in EVERY column (#38)

| decoder | atd_exact | atd_flip_fuzzy | atd_fuzzy | bar_exact | bar_fuzzy | bar_random | passes |
|---|---|---|---|---|---|---|---|
| hsmm_emp | +0.011 (0.003) 19/22 | +0.016 (0.005) 9/11 | +0.012 (0.002) 20/22 | +0.006 (0.002) 9/11 | +0.001 (0.003) 7/11 | +0.004 (0.003) 8/11 | no |
| fb_hsmm | +0.011 (0.002) 19/22 | +0.017 (0.004) 10/11 | +0.011 (0.002) 19/22 | +0.005 (0.002) 10/11 | +0.001 (0.002) 7/11 | +0.001 (0.003) 6/11 | no |
| hsmm_soft | +0.010 (0.002) 19/22 | +0.016 (0.004) 10/11 | +0.011 (0.002) 20/22 | +0.005 (0.002) 8/11 | +0.003 (0.002) 9/11 | +0.007 (0.002) 8/11 | no |
| PIPELINE | +0.009 (0.002) 19/22 | +0.016 (0.004) 10/11 | +0.009 (0.002) 19/22 | +0.004 (0.002) 9/11 | +0.001 (0.003) 7/11 | +0.002 (0.002) 6/11 | no |
| hier | +0.009 (0.001) 21/22 | +0.009 (0.002) 10/11 | +0.007 (0.002) 19/22 | +0.002 (0.002) 6/11 | +0.001 (0.001) 7/11 | -0.004 (0.002) 2/11 | no |
| hmm3 | +0.005 (0.002) 17/22 | +0.011 (0.003) 9/11 | +0.005 (0.001) 15/22 | +0.001 (0.002) 5/11 | -0.001 (0.003) 4/11 | +0.001 (0.003) 5/11 | no |
| hmm3_bnd | +0.006 (0.003) 17/22 | +0.014 (0.004) 9/11 | +0.004 (0.002) 14/22 | +0.001 (0.004) 5/11 | -0.005 (0.003) 3/11 | -0.006 (0.004) 4/11 | no |
| fb_hsmm@0 | +0.002 (0.001) 15/22 | +0.004 (0.002) 7/11 | +0.000 (0.001) 13/22 | +0.000 (0.001) 5/11 | +0.001 (0.001) 8/11 | +0.006 (0.002) 10/11 | no |
| hsmm_soft@0 | +0.000 (0.000) 8/22 | +0.000 (0.000) 5/11 | +0.000 (0.000) 9/22 | +0.001 (0.000) 9/11 | +0.001 (0.000) 6/11 | +0.000 (0.000) 5/11 | no |
| hsmm_emp@0 | +0.000 (0.000) 12/22 | -0.001 (0.001) 4/11 | -0.000 (0.000) 10/22 | +0.001 (0.000) 10/11 | +0.001 (0.001) 8/11 | +0.000 (0.000) 5/11 | no |
| hmm3_bnd@0 | +0.000 (0.000) 0/22 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/22 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/11 | no |
| prod | +0.000 (0.000) 0/22 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/22 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/11 | +0.000 (0.000) 0/11 | no |
| hier@0 | -0.002 (0.000) 5/22 | -0.002 (0.001) 4/11 | -0.002 (0.001) 5/22 | -0.002 (0.001) 3/11 | -0.002 (0.001) 1/11 | -0.001 (0.001) 3/11 | no |
| asrf | -0.000 (0.003) 11/22 | +0.006 (0.005) 8/11 | +0.001 (0.003) 9/22 | -0.008 (0.003) 2/11 | -0.014 (0.003) 1/11 | -0.012 (0.004) 1/11 | no |
| hmm3@0 | -0.007 (0.002) 3/22 | -0.011 (0.003) 2/11 | -0.008 (0.002) 4/22 | -0.012 (0.004) 1/11 | -0.011 (0.003) 1/11 | -0.010 (0.003) 2/11 | no |
| asrf@0 | -0.017 (0.003) 1/22 | -0.012 (0.003) 1/11 | -0.015 (0.003) 3/22 | -0.012 (0.003) 1/11 | -0.016 (0.003) 1/11 | -0.008 (0.002) 2/11 | no |
| argmax | -0.063 (0.003) 0/22 | -0.061 (0.005) 0/11 | -0.064 (0.004) 0/22 | -0.056 (0.004) 0/11 | -0.056 (0.006) 0/11 | -0.049 (0.005) 0/11 | no |
| argmax@0 | -0.063 (0.003) 0/22 | -0.061 (0.005) 0/11 | -0.064 (0.004) 0/22 | -0.056 (0.004) 0/11 | -0.056 (0.006) 0/11 | -0.049 (0.005) 0/11 | no |

`PIPELINE` = the whole suite as a procedure: inside each fold the decoder AND its config are chosen on the other games. It is the honest estimate of 'ship the suite's winner'; a single decoder's row is optimistic by the choice among decoders. `@0` = config 0 untuned (hmm3_bnd@0 = FINAL). `count err` = predicted − true spells (negative = merging). Low `folds agree` = flat optimum: prefer config 0 or round values.

## Match rate at IoU 0.5 by true possession length (final configs, pooled)

| decoder | 0-2 s | 2-5 s | 5-15 s | 15-inf s |
|---|---|---|---|---|
| hsmm_emp | 0.158 (8616) | 0.479 (9440) | 0.772 (8984) | 0.901 (6832) |
| fb_hsmm | 0.152 (8616) | 0.467 (9440) | 0.768 (8984) | 0.901 (6832) |
| hsmm_soft | 0.158 (8616) | 0.472 (9440) | 0.768 (8984) | 0.904 (6832) |
| hier | 0.138 (8616) | 0.444 (9440) | 0.756 (8984) | 0.900 (6832) |
| hmm3 | 0.151 (8616) | 0.465 (9440) | 0.767 (8984) | 0.900 (6832) |
| hmm3_bnd | 0.162 (8616) | 0.459 (9440) | 0.764 (8984) | 0.892 (6832) |
| fb_hsmm@0 | 0.152 (8616) | 0.459 (9440) | 0.756 (8984) | 0.907 (6832) |
| hsmm_soft@0 | 0.142 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| hsmm_emp@0 | 0.141 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| final | 0.143 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| hmm3_bnd@0 | 0.143 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| prod | 0.143 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| hier@0 | 0.145 (8616) | 0.443 (9440) | 0.748 (8984) | 0.904 (6832) |
| asrf | 0.146 (8616) | 0.448 (9440) | 0.758 (8984) | 0.894 (6832) |
| hmm3@0 | 0.108 (8616) | 0.402 (9440) | 0.730 (8984) | 0.897 (6832) |
| asrf@0 | 0.185 (8616) | 0.481 (9440) | 0.757 (8984) | 0.904 (6832) |
| argmax | 0.209 (8616) | 0.518 (9440) | 0.783 (8984) | 0.908 (6832) |
| argmax@0 | 0.209 (8616) | 0.518 (9440) | 0.783 (8984) | 0.908 (6832) |

## Final configs (chosen on all games -- what would ship)

- `hsmm_emp` → `hsmm_emp` cfg 4: `{"alpha": 0.0815, "alpha_d": 0.5432, "bd": 0.3777, "c_ad": 6.1921, "c_da": 12.96, "c_ha": 6.5142, "cap_dead": 2.6363, "cap_spell": 1.7569, "dead_mix": "prob_mean", "fps_scale": true, "home_mix": "prob_mean", "kappa": 0.25, "lam_dead": 0.3535, "lam_spell": 0.6313, "v8_share": 0.4871, "w_filled": 0.9075, "wd": 0.5183, "wh": 1.1371, "who_clip": 0.007}`
- `fb_hsmm` → `fb_hsmm` cfg 11: `{"alpha": 0.4035, "alpha_d": 0.0, "bd": 0.4622, "c2": 2.2693, "c_ad": 14.672, "c_da": 9.208, "c_ha": 11.4028, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "logit_mean", "kappa": 1.0, "lam_dead": 7.3812, "lam_spell": 10.3873, "m_dead": 3.5166, "m_spell": 1.4719, "power": 1.0, "v8_share": 0.3104, "w_filled": 0.4077, "wd": 0.6492, "wh": 1.2139, "who_clip": 0.0319, "wpost": 1.0755}`
- `hsmm_soft` → `hsmm_soft` cfg 5: `{"alpha": 0.6468, "alpha_d": 0.0, "bd": 0.4275, "c_ad": 11.8077, "c_da": 8.6115, "c_ha": 15.0024, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 7.2965, "lam_spell": 6.8511, "m_dead": 2.0278, "m_spell": 1.4643, "power": 1.0, "v8_share": 0.6054, "w_filled": 0.5136, "wd": 1.0064, "wh": 1.0375, "who_clip": 0.0204}`
- `PIPELINE` → `hsmm_emp` cfg 4: `{"alpha": 0.0815, "alpha_d": 0.5432, "bd": 0.3777, "c_ad": 6.1921, "c_da": 12.96, "c_ha": 6.5142, "cap_dead": 2.6363, "cap_spell": 1.7569, "dead_mix": "prob_mean", "fps_scale": true, "home_mix": "prob_mean", "kappa": 0.25, "lam_dead": 0.3535, "lam_spell": 0.6313, "v8_share": 0.4871, "w_filled": 0.9075, "wd": 0.5183, "wh": 1.1371, "who_clip": 0.007}`
- `hier` → `hier` cfg 7: `{"alpha": 0.4384, "alpha_d": 0.0, "bd": 0.3959, "c_ad": 10.29, "c_da": 15.3621, "c_ha": 20.0784, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_alive": 7.4713, "lam_dead": 6.5171, "lam_spell": 11.0098, "m_alive": 0.4, "m_dead": 1.1613, "m_spell": 1.0049, "power": 1.0, "v8_share": 0.4592, "w_filled": 0.3563, "wd": 0.6744, "wh": 1.3862, "who_clip": 0.0397}`
- `hmm3` → `hmm3` cfg 10: `{"alpha_d": 0.712, "bd": 0.8337, "c_ad": 6.4042, "c_da": 9.5047, "c_ha": 10.5039, "dead_mix": "prob_mean", "fps_scale": true, "home_mix": "prob_mean", "kappa": 0.5, "v8_share": 0.3699, "w_filled": 0.523, "wd": 1.5397, "wh": 1.6779, "who_clip": 0.0216}`
- `hmm3_bnd` → `hmm3_bnd` cfg 4: `{"alpha": 0.9451, "alpha_d": 0.8923, "bd": 0.8009, "c_ad": 4.4184, "c_da": 29.8193, "c_ha": 19.8506, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "logit_mean", "kappa": 1.0, "v8_share": 0.4714, "w_filled": 0.9663, "wd": 0.6247, "wh": 0.6419, "who_clip": 0.0103}`
- `fb_hsmm@0` → `fb_hsmm` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c2": 2.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 8.0, "lam_spell": 8.0, "m_dead": 2.0, "m_spell": 1.0, "power": 1.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02, "wpost": 1.0}`
- `hsmm_soft@0` → `hsmm_soft` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 8.0, "lam_spell": 8.0, "m_dead": 2.0, "m_spell": 1.0, "power": 1.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `hsmm_emp@0` → `hsmm_emp` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "cap_dead": 6.0, "cap_spell": 4.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 1.0, "lam_spell": 1.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `final` → `final` cfg 0: `{}`
- `hmm3_bnd@0` → `hmm3_bnd` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `prod` → `prod` cfg 0: `{"hook": "mqf.decode:fusion_baseline"}`
- `hier@0` → `hier` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_alive": 8.0, "lam_dead": 8.0, "lam_spell": 8.0, "m_alive": 1.0, "m_dead": 2.0, "m_spell": 1.0, "power": 1.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `asrf` → `asrf` cfg 6: `{"bd": 0.8804, "c_ad": 9.3743, "c_da": 21.0137, "c_ha": 6.3608, "cuts": "union", "dead_mix": "prob_mean", "fps_scale": true, "home_mix": "logit_mean", "kappa": 0.0, "min_gap_s": 1.3075, "smooth_s": 0.0593, "theta": 0.7611, "v8_share": 0.1429, "w_filled": 0.3726, "wd": 0.6204, "wh": 0.7595, "who_clip": 0.0381}`
- `hmm3@0` → `hmm3` cfg 0: `{"alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `asrf@0` → `asrf` cfg 0: `{"bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 2.0, "cuts": "peaks", "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "min_gap_s": 0.5, "smooth_s": 0.2, "theta": 0.5, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `argmax` → `argmax` cfg 0: `{}`
- `argmax@0` → `argmax` cfg 0: `{}`
