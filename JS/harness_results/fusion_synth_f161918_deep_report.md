# Fusion suite `synth_f161918_deep`

- Inputs `harness_results\fusion_synth`: 6 games (idsse, metrica, pff), columns clean_exact, noisy_fuzzy, seeds 0, 1; streams: dead pair True, TCN seeds True, boundary True, dead boundary True, who_v1 True, FINAL labels True; fps [25.0, 29.97]
- Every number is HELD OUT: per game, the config (or for PIPELINE the decoder AND config) best on the other games (spellF1 per column, mean per source then over sources, then over columns; tie macroF1) -- final_check's averaging
- Baseline `final`; official metric `C:\Users\pc\Documents\GitHub\PerosinhoFC-more-than-a-hack-2027\JS\vendor\mtah2027\evaluation\possession_metrics.py`; 35.1 min, numba=False

## Sanity: labels identical to the exported FINAL labels

- `hmm3_bnd`@0: 24/24 files
- `hsmm_emp`@0: 0/24 files
- `hsmm_soft`@0: 0/24 files
- `prod`: 24/24 files

## Held-out spellF1 per column (mean per source, then over sources)

| decoder | clean_exact | noisy_fuzzy | **mean** | macroF1 | adF1 | count err | folds agree |
|---|---|---|---|---|---|---|---|
| hsmm_emp | 0.779 | 0.683 | **0.731** | 0.896 | 0.960 | -9 | 1.00 |
| hsmm_soft | 0.777 | 0.681 | **0.729** | 0.895 | 0.961 | -16 | 1.00 |
| PIPELINE | 0.776 | 0.681 | **0.729** | 0.896 | 0.960 | -12 | 0.67 |
| hmm3_bnd | 0.766 | 0.673 | **0.720** | 0.891 | 0.957 | -9 | 1.00 |
| hsmm_emp@0 | 0.730 | 0.608 | **0.669** | 0.874 | 0.941 | +9 | 1.00 |
| hsmm_soft@0 | 0.728 | 0.606 | **0.667** | 0.874 | 0.941 | +10 | 1.00 |
| final | 0.724 | 0.602 | **0.663** | 0.874 | 0.941 | +13 | 1.00 |
| hmm3_bnd@0 | 0.724 | 0.602 | **0.663** | 0.874 | 0.941 | +13 | 1.00 |
| prod | 0.724 | 0.602 | **0.663** | 0.874 | 0.941 | +13 | 1.00 |

## Paired vs `final` (held out): Δ spellF1 mean (se) wins/n over game × seed runs; PASS = Δ > 2 se and wins > n/2 in EVERY column (#38)

| decoder | clean_exact | noisy_fuzzy | passes |
|---|---|---|---|
| hsmm_emp | +0.055 (0.005) 12/12 | +0.081 (0.004) 12/12 | PASS |
| hsmm_soft | +0.053 (0.005) 12/12 | +0.079 (0.004) 12/12 | PASS |
| PIPELINE | +0.052 (0.005) 12/12 | +0.079 (0.004) 12/12 | PASS |
| hmm3_bnd | +0.042 (0.005) 12/12 | +0.071 (0.005) 12/12 | PASS |
| hsmm_emp@0 | +0.006 (0.001) 11/12 | +0.006 (0.002) 9/12 | PASS |
| hsmm_soft@0 | +0.004 (0.001) 11/12 | +0.004 (0.001) 10/12 | PASS |
| hmm3_bnd@0 | +0.000 (0.000) 0/12 | +0.000 (0.000) 0/12 | no |
| prod | +0.000 (0.000) 0/12 | +0.000 (0.000) 0/12 | no |

`PIPELINE` = the whole suite as a procedure: inside each fold the decoder AND its config are chosen on the other games. It is the honest estimate of 'ship the suite's winner'; a single decoder's row is optimistic by the choice among decoders. `@0` = config 0 untuned (hmm3_bnd@0 = FINAL). `count err` = predicted − true spells (negative = merging). Low `folds agree` = flat optimum: prefer config 0 or round values.

## Match rate at IoU 0.5 by true possession length (final configs, pooled)

| decoder | 0-2 s | 2-5 s | 5-15 s | 15-inf s |
|---|---|---|---|---|
| hsmm_emp | 0.162 (661) | 0.691 (1537) | 0.905 (1885) | 0.951 (611) |
| hsmm_soft | 0.138 (661) | 0.688 (1537) | 0.889 (1885) | 0.939 (611) |
| hmm3_bnd | 0.156 (661) | 0.687 (1537) | 0.891 (1885) | 0.953 (611) |
| hsmm_emp@0 | 0.163 (661) | 0.679 (1537) | 0.862 (1885) | 0.871 (611) |
| hsmm_soft@0 | 0.163 (661) | 0.680 (1537) | 0.860 (1885) | 0.871 (611) |
| final | 0.160 (661) | 0.682 (1537) | 0.860 (1885) | 0.869 (611) |
| hmm3_bnd@0 | 0.160 (661) | 0.682 (1537) | 0.860 (1885) | 0.869 (611) |
| prod | 0.160 (661) | 0.682 (1537) | 0.860 (1885) | 0.869 (611) |

## Final configs (chosen on all games -- what would ship)

- `hsmm_emp` → `hsmm_emp` cfg 111: `{"alpha": 0.5672, "alpha_d": 0.0, "bd": -0.5552, "c_ad": 17.618, "c_da": 6.063, "c_ha": 24.1085, "cap_dead": 4.5798, "cap_spell": 4.5322, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 0.6656, "lam_spell": 1.2338, "v1_share": 0.0, "v8_share": 0.2242, "w_filled": 0.4412, "wd": 0.5, "wh": 0.9067, "who_clip": 0.0184}`
- `hsmm_soft` → `hsmm_soft` cfg 82: `{"alpha": 0.1155, "alpha_d": 0.7663, "bd": -0.2505, "c_ad": 21.4387, "c_da": 26.1117, "c_ha": 11.9186, "dead_mix": "prob_mean", "fps_scale": true, "home_mix": "logit_mean", "kappa": 0.5, "lam_dead": 7.2127, "lam_spell": 12.0237, "m_dead": 2.3423, "m_spell": 2.8517, "power": 1.0, "v1_share": 0.478, "v8_share": 0.4166, "w_filled": 0.3513, "wd": 0.6214, "wh": 0.8071, "who_clip": 0.0287}`
- `PIPELINE` → `hsmm_emp` cfg 111: `{"alpha": 0.5672, "alpha_d": 0.0, "bd": -0.5552, "c_ad": 17.618, "c_da": 6.063, "c_ha": 24.1085, "cap_dead": 4.5798, "cap_spell": 4.5322, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 0.6656, "lam_spell": 1.2338, "v1_share": 0.0, "v8_share": 0.2242, "w_filled": 0.4412, "wd": 0.5, "wh": 0.9067, "who_clip": 0.0184}`
- `hmm3_bnd` → `hmm3_bnd` cfg 104: `{"alpha": 0.7753, "alpha_d": 0.3081, "bd": -1.1817, "c_ad": 8.8255, "c_da": 24.3572, "c_ha": 18.1289, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "logit_mean", "kappa": 0.25, "v1_share": 0.1955, "v8_share": 0.9264, "w_filled": 0.4125, "wd": 0.9848, "wh": 0.5707, "who_clip": 0.0312}`
- `hsmm_emp@0` → `hsmm_emp` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "cap_dead": 6.0, "cap_spell": 4.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 1.0, "lam_spell": 1.0, "v1_share": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `hsmm_soft@0` → `hsmm_soft` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "lam_dead": 8.0, "lam_spell": 8.0, "m_dead": 2.0, "m_spell": 1.0, "power": 1.0, "v1_share": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `final` → `final` cfg 0: `{}`
- `hmm3_bnd@0` → `hmm3_bnd` cfg 0: `{"alpha": 0.5, "alpha_d": 0.0, "bd": 0.0, "c_ad": 10.0, "c_da": 10.0, "c_ha": 18.0, "dead_mix": "prob_mean", "fps_scale": false, "home_mix": "prob_mean", "kappa": 0.0, "v1_share": 0.0, "v8_share": 0.5, "w_filled": 0.4, "wd": 1.0, "wh": 1.0, "who_clip": 0.02}`
- `prod` → `prod` cfg 0: `{"hook": "mqf.decode:fusion_baseline"}`
