# Fusion suite `synth_f161918_quick`

- Inputs `harness_results\fusion_synth`: 6 games (idsse, metrica, pff), columns clean_exact, noisy_fuzzy, seeds 0, 1; streams: dead pair True, TCN seeds True, boundary True, dead boundary True, who_v1 True, FINAL labels True; fps [25.0, 29.97]
- Every number is HELD OUT: per game, the config (or for PIPELINE the decoder AND config) best on the other games (spellF1 per column, mean per source then over sources, then over columns; tie macroF1) -- final_check's averaging
- Baseline `final`; official metric `C:\Users\pc\Documents\GitHub\PerosinhoFC-more-than-a-hack-2027\JS\vendor\mtah2027\evaluation\possession_metrics.py`; 10.5 min, numba=False

## Sanity: labels identical to the exported FINAL labels

- `argmax`@0: 0/24 files
- `asrf`@0: 0/24 files
- `fb_hsmm`@0: 0/24 files
- `hier`@0: 0/24 files
- `hmm3`@0: 0/24 files
- `hmm3_bnd`@0: 24/24 files
- `hsmm_emp`@0: 0/24 files
- `hsmm_soft`@0: 0/24 files
- `prod`: 24/24 files

## Held-out spellF1 per column (mean per source, then over sources)

| decoder | clean_exact | noisy_fuzzy | **mean** | macroF1 | adF1 | count err | folds agree |
|---|---|---|---|---|---|---|---|
| hsmm_soft | 0.781 | 0.667 | **0.724** | 0.893 | 0.959 | -3 | 1.00 |
| PIPELINE | 0.781 | 0.667 | **0.724** | 0.893 | 0.959 | -3 | 1.00 |
| hsmm_emp | 0.772 | 0.643 | **0.708** | 0.894 | 0.962 | +16 | 1.00 |
| hmm3_bnd | 0.754 | 0.641 | **0.698** | 0.882 | 0.953 | -20 | 1.00 |
| fb_hsmm | 0.741 | 0.646 | **0.693** | 0.887 | 0.952 | -6 | 1.00 |
| hmm3 | 0.739 | 0.639 | **0.689** | 0.884 | 0.950 | +2 | 1.00 |
| hier | 0.751 | 0.624 | **0.688** | 0.886 | 0.955 | +20 | 1.00 |
| asrf | 0.734 | 0.626 | **0.680** | 0.884 | 0.951 | +7 | 1.00 |
| hsmm_emp@0 | 0.730 | 0.608 | **0.669** | 0.874 | 0.941 | +9 | 1.00 |
| hsmm_soft@0 | 0.728 | 0.606 | **0.667** | 0.874 | 0.941 | +10 | 1.00 |
| hier@0 | 0.725 | 0.602 | **0.663** | 0.876 | 0.944 | +15 | 1.00 |
| final | 0.724 | 0.602 | **0.663** | 0.874 | 0.941 | +13 | 1.00 |
| hmm3_bnd@0 | 0.724 | 0.602 | **0.663** | 0.874 | 0.941 | +13 | 1.00 |
| prod | 0.724 | 0.602 | **0.663** | 0.874 | 0.941 | +13 | 1.00 |
| hmm3@0 | 0.722 | 0.598 | **0.660** | 0.874 | 0.941 | +11 | 1.00 |
| fb_hsmm@0 | 0.723 | 0.590 | **0.656** | 0.873 | 0.940 | +17 | 1.00 |
| asrf@0 | 0.567 | 0.513 | **0.540** | 0.837 | 0.943 | -60 | 1.00 |
| argmax | 0.179 | 0.026 | **0.103** | 0.848 | 0.938 | +2649 | 1.00 |
| argmax@0 | 0.179 | 0.026 | **0.103** | 0.848 | 0.938 | +2649 | 1.00 |

## Paired vs `final` (held out): Δ spellF1 mean (se) wins/n over game × seed runs; PASS = Δ > 2 se and wins > n/2 in EVERY column (#38)

| decoder | clean_exact | noisy_fuzzy | passes |
|---|---|---|---|
| hsmm_soft | +0.057 (0.005) 12/12 | +0.065 (0.005) 12/12 | PASS |
| PIPELINE | +0.057 (0.005) 12/12 | +0.065 (0.005) 12/12 | PASS |
| hsmm_emp | +0.048 (0.005) 12/12 | +0.041 (0.004) 12/12 | PASS |
| hmm3_bnd | +0.030 (0.008) 11/12 | +0.039 (0.007) 11/12 | PASS |
| fb_hsmm | +0.017 (0.004) 10/12 | +0.044 (0.005) 12/12 | PASS |
| hmm3 | +0.015 (0.002) 12/12 | +0.037 (0.003) 12/12 | PASS |
| hier | +0.028 (0.005) 12/12 | +0.022 (0.005) 10/12 | PASS |
| asrf | +0.010 (0.003) 10/12 | +0.023 (0.003) 12/12 | PASS |
| hsmm_emp@0 | +0.006 (0.001) 11/12 | +0.006 (0.002) 9/12 | PASS |
| hsmm_soft@0 | +0.004 (0.001) 11/12 | +0.004 (0.001) 10/12 | PASS |
| hier@0 | +0.001 (0.001) 7/12 | -0.000 (0.002) 6/12 | no |
| hmm3_bnd@0 | +0.000 (0.000) 0/12 | +0.000 (0.000) 0/12 | no |
| prod | +0.000 (0.000) 0/12 | +0.000 (0.000) 0/12 | no |
| hmm3@0 | -0.001 (0.001) 3/12 | -0.004 (0.002) 4/12 | no |
| fb_hsmm@0 | -0.001 (0.001) 2/12 | -0.012 (0.002) 0/12 | no |
| asrf@0 | -0.157 (0.009) 0/12 | -0.089 (0.015) 0/12 | no |
| argmax | -0.545 (0.006) 0/12 | -0.576 (0.009) 0/12 | no |
| argmax@0 | -0.545 (0.006) 0/12 | -0.576 (0.009) 0/12 | no |

`PIPELINE` = the whole suite as a procedure: inside each fold the decoder AND its config are chosen on the other games. It is the honest estimate of 'ship the suite's winner'; a single decoder's row is optimistic by the choice among decoders. `@0` = config 0 untuned (hmm3_bnd@0 = FINAL). `count err` = predicted − true spells (negative = merging). Low `folds agree` = flat optimum: prefer config 0 or round values.

## Match rate at IoU 0.5 by true possession length (final configs, pooled)

| decoder | 0-2 s | 2-5 s | 5-15 s | 15-inf s |
|---|---|---|---|---|
| hsmm_soft | 0.148 (661) | 0.705 (1537) | 0.909 (1885) | 0.943 (611) |
| hsmm_emp | 0.171 (661) | 0.730 (1537) | 0.920 (1885) | 0.936 (611) |
| hmm3_bnd | 0.115 (661) | 0.612 (1537) | 0.860 (1885) | 0.944 (611) |
| fb_hsmm | 0.145 (661) | 0.654 (1537) | 0.874 (1885) | 0.923 (611) |
| hmm3 | 0.145 (661) | 0.669 (1537) | 0.879 (1885) | 0.920 (611) |
| hier | 0.177 (661) | 0.722 (1537) | 0.905 (1885) | 0.925 (611) |
| asrf | 0.162 (661) | 0.675 (1537) | 0.873 (1885) | 0.920 (611) |
| hsmm_emp@0 | 0.163 (661) | 0.679 (1537) | 0.862 (1885) | 0.871 (611) |
| hsmm_soft@0 | 0.163 (661) | 0.680 (1537) | 0.860 (1885) | 0.871 (611) |
| hier@0 | 0.168 (661) | 0.690 (1537) | 0.863 (1885) | 0.872 (611) |
| final | 0.160 (661) | 0.682 (1537) | 0.860 (1885) | 0.869 (611) |
| hmm3_bnd@0 | 0.160 (661) | 0.682 (1537) | 0.860 (1885) | 0.869 (611) |
| prod | 0.160 (661) | 0.682 (1537) | 0.860 (1885) | 0.869 (611) |
| hmm3@0 | 0.153 (661) | 0.671 (1537) | 0.857 (1885) | 0.869 (611) |
| fb_hsmm@0 | 0.165 (661) | 0.681 (1537) | 0.858 (1885) | 0.863 (611) |
| asrf@0 | 0.132 (661) | 0.353 (1537) | 0.566 (1885) | 0.781 (611) |
| argmax | 0.194 (661) | 0.552 (1537) | 0.534 (1885) | 0.316 (611) |
| argmax@0 | 0.194 (661) | 0.552 (1537) | 0.534 (1885) | 0.316 (611) |

## Final configs (chosen on all games -- what would ship)

- `hsmm_soft` → `hsmm_soft` cfg 10: `{"alpha": 0.1259, "alpha_d": 0.8938, "bd": -1.1364, "c_ad": 10.1483, "c_da": 21.5929, "c_ha": 37.5258, "dead_mix": "loglin", "fps_scale": true, "home_mix": "logit_mean", "kappa": 0.25, "lam_dead": 11.0725, "lam_spell": 27.593, "m_dead": 3.9953, "m_spell": 1.733, "power": 2.0, "v1_share": 0.0041, "v8_share": 0.7173, "w_filled": 0.5901, "wd": 0.7753, "wh": 1.8847, "who_clip": 0.0165}`
- `PIPELINE` → `hsmm_soft` cfg 10: `{"alpha": 0.1259, "alpha_d": 0.8938, "bd": -1.1364, "c_ad": 10.1483, "c_da": 21.5929, "c_ha": 37.5258, "dead_mix": "loglin", "fps_scale": true, "home_mix": "logit_mean", "kappa": 0.25, "lam_dead": 11.0725, "lam_spell": 27.593, "m_dead": 3.9953, "m_spell": 1.733, "power": 2.0, "v1_share": 0.0041, "v8_share": 0.7173, "w_filled": 0.5901, "wd": 0.7753, "wh": 1.8847, "who_clip": 0.0165}`
- `hsmm_emp` → `hsmm_emp` cfg 9: `{"alpha": 0.1758, "alpha_d": 0.2643, "bd": -0.8272, "c_ad": 9.6038, "c_da": 29.1306, "c_ha": 20.9039, "cap_dead": 6.3361, "cap_spell": 3.4331, "dead_mix": "loglin", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 0.8893, "lam_spell": 1.1862, "v1_share": 0.0, "v8_share": 0.3456, "w_filled": 0.4952, "wd": 0.6297, "wh": 1.5893, "who_clip": 0.0177}`
- `hmm3_bnd` → `hmm3_bnd` cfg 2: `{"alpha": 0.1965, "alpha_d": 0.1298, "bd": -1.2745, "c_ad": 15.2891, "c_da": 14.7893, "c_ha": 25.2298, "dead_mix": "loglin", "fps_scale": false, "home_mix": "prob_mean", "kappa": 1.0, "v1_share": 0.3707, "v8_share": 0.1661, "w_filled": 0.9041, "wd": 0.9052, "wh": 0.7151, "who_clip": 0.0086}`
- `fb_hsmm` → `fb_hsmm` cfg 5: `{"alpha": 0.3731, "alpha_d": 0.1143, "bd": -0.3719, "c2": 1.5412, "c_ad": 8.8056, "c_da": 10.1264, "c_ha": 20.4743, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 7.3784, "lam_spell": 6.7699, "m_dead": 2.5906, "m_spell": 1.2775, "power": 1.0, "v1_share": 0.0, "v8_share": 0.7372, "w_filled": 0.1, "wd": 0.7014, "wh": 0.6814, "who_clip": 0.0152, "wpost": 1.6849}`
- `hmm3` → `hmm3` cfg 9: `{"alpha_d": 0.2107, "bd": -0.2478, "c_ad": 6.7997, "c_da": 9.298, "c_ha": 27.229, "dead_mix": "loglin", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "v1_share": 0.0, "v8_share": 0.5039, "w_filled": 0.4984, "wd": 0.5153, "wh": 0.7843, "who_clip": 0.028}`
- `hier` → `hier` cfg 10: `{"alpha": 0.4846, "alpha_d": 0.1104, "bd": -0.9732, "c_ad": 13.2061, "c_da": 7.1653, "c_ha": 18.1017, "dead_mix": "loglin", "fps_scale": false, "home_mix": "logit_mean", "kappa": 1.0, "lam_alive": 23.9066, "lam_dead": 4.6575, "lam_spell": 10.8244, "m_alive": 2.9611, "m_dead": 3.3448, "m_spell": 2.1768, "power": 2.0, "v1_share": 0.0549, "v8_share": 0.3129, "w_filled": 0.6587, "wd": 1.0184, "wh": 1.3629, "who_clip": 0.0245}`
- `asrf` → `asrf` cfg 10: `{"bd": -0.7817, "c_ad": 5.6957, "c_da": 14.1775, "c_ha": 17.5935, "cuts": "union", "dead_mix": "loglin", "fps_scale": false, "home_mix": "logit_mean", "kappa": 0.5, "min_gap_s": 0.3959, "smooth_s": 0.4952, "theta": 0.3864, "v1_share": 0.3221, "v8_share": 0.5979, "w_filled": 0.9593, "wd": 1.1899, "wh": 0.7003, "who_clip": 0.0105}`
- `hsmm_emp@0` → `hsmm_emp` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "cap_dead": 6.0, "cap_spell": 4.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 1.0, "lam_spell": 1.0, "v1_share": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `hsmm_soft@0` → `hsmm_soft` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 8.0, "lam_spell": 8.0, "m_dead": 2.0, "m_spell": 1.0, "power": 1.0, "v1_share": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `hier@0` → `hier` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_alive": 8.0, "lam_dead": 8.0, "lam_spell": 8.0, "m_alive": 1.0, "m_dead": 2.0, "m_spell": 1.0, "power": 1.0, "v1_share": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `final` → `final` cfg 0: `{}`
- `hmm3_bnd@0` → `hmm3_bnd` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "v1_share": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `prod` → `prod` cfg 0: `{"hook": "mqf.decode:fusion_baseline"}`
- `hmm3@0` → `hmm3` cfg 0: `{"alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "v1_share": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `fb_hsmm@0` → `fb_hsmm` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c2": 2.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 8.0, "lam_spell": 8.0, "m_dead": 2.0, "m_spell": 1.0, "power": 1.0, "v1_share": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02, "wpost": 1.0}`
- `asrf@0` → `asrf` cfg 0: `{"bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 2.0, "cuts": "peaks", "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "min_gap_s": 0.5, "smooth_s": 0.2, "theta": 0.5, "v1_share": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `argmax` → `argmax` cfg 0: `{}`
- `argmax@0` → `argmax` cfg 0: `{}`
