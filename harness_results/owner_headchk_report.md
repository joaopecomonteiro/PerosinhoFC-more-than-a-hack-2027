# Heading check `headchk` (other games only)

Pass-direction columns (feats.HEADING_COLS, ported from who/who_lab.py) through the #38 protocol. Paired Δ spellF1 = a − b per game × noise seed; cells: mean (se) wins · per source. Rule (#38): layer dead_v1, mean > 2 se in every condition and wins on more than half the runs.


## GBM arm — #38's conditions

Sanity: `bar` who_v1 0.516 (expect 0.516, #38), nearest 0.309 (expect 0.309); `atd` who_v1 0.504 (expect 0.504, #38)


### Layer `dead_v1`

| comparison | bar | atd |
|---|---|---|
| gbm_full_im vs who_v1 | +0.0009 (0.0055) 6/11 · idsse +0.005 metrica +0.040 pff -0.005 | -0.0055 (0.0025) 7/22 · idsse -0.006 metrica +0.009 pff -0.007 |
| gbm_stack_im vs who_v1 | +0.0034 (0.0077) 6/11 · idsse +0.012 metrica +0.048 pff -0.004 | -0.0003 (0.0028) 11/22 · idsse -0.004 metrica +0.022 pff -0.002 |
| gbm_full_heading_im vs who_v1 | +0.0030 (0.0066) 4/11 · idsse +0.015 metrica +0.047 pff -0.005 | -0.0063 (0.0032) 8/22 · idsse +0.002 metrica +0.012 pff -0.011 |
| gbm_stack_heading_im vs who_v1 | +0.0019 (0.0074) 6/11 · idsse +0.012 metrica +0.043 pff -0.006 | -0.0044 (0.0026) 8/22 · idsse -0.002 metrica +0.015 pff -0.007 |
| blend vs who_v1 | -0.1347 (0.0111) 0/11 · idsse -0.113 metrica -0.204 pff -0.131 | -0.1279 (0.0066) 0/22 · idsse -0.103 metrica -0.171 pff -0.129 |
| nearest vs who_v1 | -0.2072 (0.0169) 0/11 · idsse -0.221 metrica -0.324 pff -0.189 | -0.1783 (0.0079) 0/22 · idsse -0.170 metrica -0.253 pff -0.171 |
| heading effect: gbm_full_heading_im - gbm_full_im | +0.0021 (0.0026) 7/11 · idsse +0.010 metrica +0.008 pff -0.000 | -0.0008 (0.0019) 11/22 · idsse +0.008 metrica +0.003 pff -0.003 |
| heading effect: gbm_stack_heading_im - gbm_stack_im | -0.0014 (0.0015) 4/11 · idsse -0.000 metrica -0.005 pff -0.001 | -0.0041 (0.0016) 6/22 · idsse +0.002 metrica -0.007 pff -0.005 |
| blend - nearest (who_lab's comparison) | +0.0725 (0.0098) 11/11 · idsse +0.108 metrica +0.119 pff +0.058 | +0.0505 (0.0042) 22/22 · idsse +0.067 metrica +0.083 pff +0.042 |

### Layer `oracle`

| comparison | bar | atd |
|---|---|---|
| gbm_full_im vs who_v1 | -0.0102 (0.0061) 3/11 · idsse +0.004 metrica +0.032 pff -0.019 | -0.0039 (0.0027) 8/22 · idsse -0.006 metrica +0.016 pff -0.006 |
| gbm_stack_im vs who_v1 | +0.0007 (0.0059) 5/11 · idsse +0.009 metrica +0.034 pff -0.005 | +0.0060 (0.0025) 16/22 · idsse -0.000 metrica +0.020 pff +0.006 |
| gbm_full_heading_im vs who_v1 | -0.0053 (0.0083) 5/11 · idsse +0.016 metrica +0.045 pff -0.017 | -0.0048 (0.0029) 9/22 · idsse +0.004 metrica +0.012 pff -0.009 |
| gbm_stack_heading_im vs who_v1 | -0.0015 (0.0067) 5/11 · idsse +0.014 metrica +0.037 pff -0.010 | -0.0003 (0.0030) 8/22 · idsse +0.007 metrica +0.017 pff -0.004 |
| blend vs who_v1 | -0.1874 (0.0153) 0/11 · idsse -0.140 metrica -0.242 pff -0.192 | -0.1788 (0.0083) 0/22 · idsse -0.154 metrica -0.218 pff -0.180 |
| nearest vs who_v1 | -0.2789 (0.0190) 0/11 · idsse -0.278 metrica -0.394 pff -0.265 | -0.2285 (0.0094) 0/22 · idsse -0.220 metrica -0.300 pff -0.222 |
| heading effect: gbm_full_heading_im - gbm_full_im | +0.0048 (0.0032) 7/11 · idsse +0.012 metrica +0.013 pff +0.002 | -0.0009 (0.0023) 10/22 · idsse +0.010 metrica -0.004 pff -0.003 |
| heading effect: gbm_stack_heading_im - gbm_stack_im | -0.0023 (0.0026) 3/11 · idsse +0.005 metrica +0.003 pff -0.005 | -0.0063 (0.0022) 7/22 · idsse +0.007 metrica -0.003 pff -0.010 |
| blend - nearest (who_lab's comparison) | +0.0916 (0.0128) 11/11 · idsse +0.138 metrica +0.152 pff +0.072 | +0.0496 (0.0040) 22/22 · idsse +0.066 metrica +0.081 pff +0.042 |

## TCN arm — #40's conditions (--clean-tracks)

Sanity: `bar` who_v1 0.518 (expect 0.518), ens_impff+bnd − who_v1 +0.048 (expect +0.048, #40); `atd` who_v1 0.507 (expect 0.507), ens_impff+bnd − who_v1 +0.050 (expect +0.050, #40); `atd_flip` who_v1 0.472 (expect 0.472), ens_impff+bnd − who_v1 +0.054 (expect +0.054, #40)


### Layer `dead_v1`

| comparison | bar | atd | atd_flip |
|---|---|---|---|
| ens_impff_hd + own boundary vs who_v1 | +0.0543 (0.0108) 11/11 · idsse +0.034 metrica +0.125 pff +0.051 | +0.0474 (0.0029) 33/33 · idsse +0.040 metrica +0.061 pff +0.048 | +0.0520 (0.0051) 22/22 · idsse +0.049 metrica +0.058 pff +0.052 |
| ens_impff + own boundary vs who_v1 (#40) | +0.0476 (0.0081) 11/11 · idsse +0.037 metrica +0.094 pff +0.044 | +0.0498 (0.0032) 33/33 · idsse +0.050 metrica +0.038 pff +0.051 | +0.0542 (0.0050) 22/22 · idsse +0.059 metrica +0.036 pff +0.055 |
| heading effect: ens_impff_hd - ens_impff (both + own boundary) | +0.0067 (0.0036) 8/11 · idsse -0.004 metrica +0.031 pff +0.006 | -0.0024 (0.0024) 14/33 · idsse -0.011 metrica +0.023 pff -0.003 | -0.0022 (0.0031) 11/22 · idsse -0.010 metrica +0.022 pff -0.003 |
| heading effect: ens_impff_hd - ens_impff (no boundary) | +0.0097 (0.0047) 9/11 · idsse +0.005 metrica +0.007 pff +0.011 | +0.0009 (0.0020) 15/33 · idsse -0.006 metrica +0.017 pff +0.001 | -0.0011 (0.0028) 10/22 · idsse -0.003 metrica +0.021 pff -0.003 |
| heading effect per seed: TCN_hd - TCN (+ own boundary) | +0.0070 [+0.0050, +0.0098] over 3 seeds | +0.0004 [-0.0003, +0.0014] over 3 seeds | -0.0005 [-0.0047, +0.0048] over 3 seeds |

### Layer `oracle`

| comparison | bar | atd | atd_flip |
|---|---|---|---|
| ens_impff_hd + own boundary vs who_v1 | +0.0658 (0.0096) 11/11 · idsse +0.042 metrica +0.134 pff +0.063 | +0.0623 (0.0039) 33/33 · idsse +0.038 metrica +0.070 pff +0.068 | +0.0735 (0.0058) 22/22 · idsse +0.055 metrica +0.069 pff +0.079 |
| ens_impff + own boundary vs who_v1 (#40) | +0.0589 (0.0064) 11/11 · idsse +0.046 metrica +0.092 pff +0.058 | +0.0600 (0.0038) 33/33 · idsse +0.041 metrica +0.056 pff +0.065 | +0.0753 (0.0059) 22/22 · idsse +0.066 metrica +0.046 pff +0.081 |
| heading effect: ens_impff_hd - ens_impff (both + own boundary) | +0.0069 (0.0051) 7/11 · idsse -0.004 metrica +0.042 pff +0.005 | +0.0023 (0.0022) 20/33 · idsse -0.003 metrica +0.015 pff +0.002 | -0.0018 (0.0028) 13/22 · idsse -0.011 metrica +0.023 pff -0.003 |
| heading effect: ens_impff_hd - ens_impff (no boundary) | +0.0053 (0.0030) 7/11 · idsse +0.004 metrica +0.017 pff +0.004 | +0.0008 (0.0024) 20/33 · idsse -0.011 metrica +0.016 pff +0.002 | -0.0030 (0.0025) 9/22 · idsse -0.011 metrica +0.021 pff -0.004 |
| heading effect per seed: TCN_hd - TCN (+ own boundary) | +0.0079 [+0.0027, +0.0160] over 3 seeds | +0.0028 [-0.0001, +0.0079] over 3 seeds | +0.0001 [-0.0035, +0.0046] over 3 seeds |

## Verdict (layer dead_v1, every condition)

- fail  gbm_full_heading_im vs who_v1  → replaces who_v1 (#38 rule)
- fail  gbm_stack_heading_im vs who_v1  → replaces who_v1 (#38 rule)
- fail  blend vs who_v1  → replaces who_v1 (#38 rule)
- fail  heading effect: gbm_full_heading_im - gbm_full_im  → the columns add something to the trees
- fail  heading effect: gbm_stack_heading_im - gbm_stack_im  → the columns add something to the trees
- fail  heading effect: ens_impff_hd - ens_impff (both + own boundary)  → heading goes into #40's TCN candidate
- PASS  ens_impff_hd + own boundary vs who_v1  → replaces who_v1 (#38 rule)

No heading GBM passes against who_v1: the #38 decision stands for the trees. The TCN arm decides whether the columns go into the #40 candidate.
