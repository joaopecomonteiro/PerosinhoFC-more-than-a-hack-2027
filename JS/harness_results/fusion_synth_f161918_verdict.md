# Fusion verdict `synth_f161918`

- inputs `harness_results\fusion_synth`: 6 games × 2 columns (clean_exact, noisy_fuzzy); baseline = FINAL's exported labels; every number held out (nested leave-one-game-out)
- PIPELINE = decoder AND config chosen inside each fold on the other games: the honest estimate of shipping this suite's winner

| row | clean_exact | noisy_fuzzy | mean | vs FINAL per column: Δ (se) wins/n | passes |
|---|---|---|---|---|---|
| final | 0.724 | 0.602 | 0.663 |  |  |
| PIPELINE | 0.776 | 0.681 | 0.729 | +0.052 (0.005) 12/12; +0.079 (0.004) 12/12 | PASS |
| hmm3_bnd@0 | 0.724 | 0.602 | 0.663 | +0.000 (0.000) 0/12; +0.000 (0.000) 0/12 |  |
| hsmm_emp | 0.779 | 0.683 | 0.731 | +0.055 (0.005) 12/12; +0.081 (0.004) 12/12 | PASS |
| hsmm_soft | 0.777 | 0.681 | 0.729 | +0.053 (0.005) 12/12; +0.079 (0.004) 12/12 | PASS |
| hmm3_bnd | 0.766 | 0.673 | 0.720 | +0.042 (0.005) 12/12; +0.071 (0.005) 12/12 | PASS |
| fb_hsmm | 0.741 | 0.646 | 0.693 | +0.017 (0.004) 10/12; +0.044 (0.005) 12/12 | PASS |
| hmm3 | 0.739 | 0.639 | 0.689 | +0.015 (0.002) 12/12; +0.037 (0.003) 12/12 | PASS |
| hier | 0.751 | 0.624 | 0.688 | +0.028 (0.005) 12/12; +0.022 (0.005) 10/12 | PASS |
| asrf | 0.734 | 0.626 | 0.680 | +0.010 (0.003) 10/12; +0.023 (0.003) 12/12 | PASS |
| argmax | 0.179 | 0.026 | 0.103 | -0.545 (0.006) 0/12; -0.576 (0.009) 0/12 |  |

PIPELINE picks per fold: hsmm_emp#111, hsmm_soft#82, hsmm_emp#111, hsmm_soft#82, hsmm_emp#111, hsmm_emp#111 (agreement 0.67)

**Adopt `hsmm_emp`** (PIPELINE passes in every column; the decoder's own row passes; 67% of folds agree). Rounded config loses +0.0003 in-sample → shipping the rounded one: `harness_results\fusion_synth_f161918_adopt.json`

Build it next to FINAL and compare (aggregates only, never to choose):
```
python -m mqf.make_submission --team Perosinho_FC_fusion --clean-tracks --ball-fill --who-model models/owner_tcn_full_impff_s0.pt models/owner_tcn_full_impff_s1.pt models/owner_tcn_full_impff_s2.pt --who-bnd-alpha 0.5 --alive-model models/alive_asf_full_impff_s0.pt models/alive_asf_full_impff_s1.pt models/alive_asf_full_impff_s2.pt --deadball-model models/deadball_v8.joblib --deadball-decoder hmm --params dead_who=max --fusion-decoder hsmm_emp --fusion-config harness_results\fusion_synth_f161918_adopt.json
python -m mqf.compare results/Perosinho_FC.csv results/Perosinho_FC_fusion.csv
```
Then freeze it: FINAL['options'] += {'fusion_decoder': ..., 'fusion_config': <the JSON's config>}, make reproduce, EXPERIMENTS #47, TECHNICAL.md.

Reports: fusion_synth_f161918_quick_report.md, fusion_synth_f161918_deep_report.md
