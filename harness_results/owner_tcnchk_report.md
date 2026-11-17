# TCN check `tcnchk` (other games only)

Paired Δ spellF1 vs `who_v1` per game × noise seed, tracks cleaned (unless --no-clean), α = 0.5. Single-method cells: mean (se) wins · per source. Seed families: mean of the per-seed means [worst, best seed]. Rule (#38): mean > 2 se in every condition, layer dead_v1, wins on more than half.

Sanity: `bar` who_v1 0.518 (11 games, seeds [0]); `atd` who_v1 0.507 (11 games, seeds [0, 1, 2]); seeds 0 1: 0.507 (expect 0.507 = #39 `whoclean_on_nf`); `atd_flip` who_v1 0.472 (11 games, seeds [0, 1]); seeds 0 1: 0.472 (expect 0.472 = #39 `whoclean_on`)


## Layer `dead_v1`

| candidate | bar | atd | atd_flip |
|---|---|---|---|
| who_v1 + TCN_im boundary (1 seed) | +0.0061 [+0.0058, +0.0065] over 3 seeds | +0.0088 [+0.0083, +0.0091] over 3 seeds | +0.0036 [+0.0033, +0.0040] over 3 seeds |
| who_v1 + ens_im boundary | +0.0071 (0.0022) 10/11 · idsse +0.006 metrica +0.024 pff +0.005 | +0.0098 (0.0017) 29/33 · idsse +0.017 metrica +0.024 pff +0.006 | +0.0046 (0.0021) 15/22 · idsse +0.013 metrica +0.017 pff +0.001 |
| TCN_im (1 seed) | +0.0099 [+0.0015, +0.0159] over 3 seeds | +0.0095 [+0.0061, +0.0127] over 3 seeds | +0.0069 [+0.0003, +0.0110] over 3 seeds |
| TCN_im + own boundary (1 seed) | +0.0095 [-0.0001, +0.0164] over 3 seeds | +0.0087 [+0.0054, +0.0114] over 3 seeds | +0.0056 [-0.0006, +0.0097] over 3 seeds |
| ens_im | +0.0129 (0.0062) 8/11 · idsse +0.015 metrica +0.048 pff +0.008 | +0.0161 (0.0022) 31/33 · idsse +0.022 metrica +0.017 pff +0.015 | +0.0156 (0.0031) 18/22 · idsse +0.021 metrica +0.010 pff +0.015 |
| ens_im + own boundary | +0.0167 (0.0065) 9/11 · idsse +0.020 metrica +0.054 pff +0.011 | +0.0147 (0.0023) 29/33 · idsse +0.026 metrica +0.012 pff +0.012 | +0.0152 (0.0035) 19/22 · idsse +0.019 metrica +0.017 pff +0.014 |
| who_v1 + TCN_impff boundary (1 seed) | +0.0090 [+0.0071, +0.0103] over 3 seeds | +0.0125 [+0.0121, +0.0130] over 3 seeds | +0.0059 [+0.0053, +0.0067] over 3 seeds |
| who_v1 + ens_impff boundary | +0.0096 (0.0025) 11/11 · idsse +0.006 metrica +0.032 pff +0.008 | +0.0128 (0.0017) 29/33 · idsse +0.020 metrica +0.023 pff +0.010 | +0.0055 (0.0022) 14/22 · idsse +0.015 metrica +0.018 pff +0.002 |
| TCN_impff (1 seed) | +0.0364 [+0.0283, +0.0439] over 3 seeds | +0.0399 [+0.0366, +0.0448] over 3 seeds | +0.0451 [+0.0431, +0.0486] over 3 seeds |
| TCN_impff + own boundary (1 seed) | +0.0442 [+0.0393, +0.0487] over 3 seeds | +0.0461 [+0.0442, +0.0491] over 3 seeds | +0.0522 [+0.0494, +0.0576] over 3 seeds |
| ens_impff | +0.0350 (0.0095) 10/11 · idsse +0.026 metrica +0.091 pff +0.030 | +0.0409 (0.0032) 33/33 · idsse +0.035 metrica +0.037 pff +0.043 | +0.0455 (0.0059) 21/22 · idsse +0.041 metrica +0.031 pff +0.048 |
| ens_impff + own boundary | +0.0476 (0.0081) 11/11 · idsse +0.037 metrica +0.094 pff +0.044 | +0.0498 (0.0032) 33/33 · idsse +0.050 metrica +0.038 pff +0.051 | +0.0542 (0.0050) 22/22 · idsse +0.059 metrica +0.036 pff +0.055 |

## Layer `oracle`

| candidate | bar | atd | atd_flip |
|---|---|---|---|
| who_v1 + TCN_im boundary (1 seed) | +0.0001 [-0.0009, +0.0015] over 3 seeds | -0.0041 [-0.0045, -0.0036] over 3 seeds | -0.0007 [-0.0011, -0.0005] over 3 seeds |
| who_v1 + ens_im boundary | +0.0017 (0.0042) 4/11 · idsse +0.012 metrica +0.022 pff -0.003 | -0.0033 (0.0021) 10/33 · idsse +0.005 metrica +0.021 pff -0.009 | -0.0001 (0.0023) 10/22 · idsse +0.008 metrica +0.019 pff -0.004 |
| TCN_im (1 seed) | +0.0056 [-0.0074, +0.0141] over 3 seeds | +0.0017 [-0.0014, +0.0046] over 3 seeds | +0.0044 [-0.0039, +0.0114] over 3 seeds |
| TCN_im + own boundary (1 seed) | +0.0035 [-0.0081, +0.0113] over 3 seeds | +0.0005 [-0.0036, +0.0032] over 3 seeds | +0.0023 [-0.0055, +0.0079] over 3 seeds |
| ens_im | +0.0148 (0.0054) 9/11 · idsse +0.010 metrica +0.050 pff +0.012 | +0.0103 (0.0031) 25/33 · idsse +0.007 metrica +0.026 pff +0.009 | +0.0197 (0.0027) 21/22 · idsse +0.021 metrica +0.018 pff +0.020 |
| ens_im + own boundary | +0.0157 (0.0068) 8/11 · idsse +0.018 metrica +0.059 pff +0.010 | +0.0089 (0.0031) 22/33 · idsse +0.011 metrica +0.027 pff +0.006 | +0.0166 (0.0027) 20/22 · idsse +0.021 metrica +0.019 pff +0.015 |
| who_v1 + TCN_impff boundary (1 seed) | +0.0017 [+0.0015, +0.0020] over 3 seeds | -0.0056 [-0.0063, -0.0047] over 3 seeds | -0.0020 [-0.0027, -0.0013] over 3 seeds |
| who_v1 + ens_impff boundary | +0.0018 (0.0043) 4/11 · idsse +0.012 metrica +0.027 pff -0.004 | -0.0044 (0.0022) 12/33 · idsse +0.005 metrica +0.020 pff -0.010 | -0.0020 (0.0025) 8/22 · idsse +0.004 metrica +0.023 pff -0.007 |
| TCN_impff (1 seed) | +0.0501 [+0.0437, +0.0535] over 3 seeds | +0.0541 [+0.0500, +0.0604] over 3 seeds | +0.0678 [+0.0647, +0.0702] over 3 seeds |
| TCN_impff + own boundary (1 seed) | +0.0541 [+0.0500, +0.0574] over 3 seeds | +0.0534 [+0.0490, +0.0567] over 3 seeds | +0.0681 [+0.0649, +0.0716] over 3 seeds |
| ens_impff | +0.0550 (0.0074) 11/11 · idsse +0.028 metrica +0.083 pff +0.058 | +0.0598 (0.0039) 33/33 · idsse +0.040 metrica +0.050 pff +0.066 | +0.0732 (0.0064) 22/22 · idsse +0.059 metrica +0.036 pff +0.081 |
| ens_impff + own boundary | +0.0589 (0.0064) 11/11 · idsse +0.046 metrica +0.092 pff +0.058 | +0.0600 (0.0038) 33/33 · idsse +0.041 metrica +0.056 pff +0.065 | +0.0753 (0.0059) 22/22 · idsse +0.066 metrica +0.046 pff +0.081 |

## Verdict (layer dead_v1, every condition)

- PASS  ens_impff + own boundary  (worst-condition mean +0.0476)
- PASS  ens_impff  (worst-condition mean +0.0350)
- PASS  ens_im + own boundary  (worst-condition mean +0.0147)
- PASS  ens_im  (worst-condition mean +0.0129)
- PASS  who_v1 + ens_impff boundary  (worst-condition mean +0.0055)
- PASS  who_v1 + ens_im boundary  (worst-condition mean +0.0046)

Best passing: **ens_impff + own boundary**. Before adopting it: it must run in make_submission on the CPU, deterministically (float64, rounded outputs), and beat who_v1 by more than the cost of the extra moving part.
