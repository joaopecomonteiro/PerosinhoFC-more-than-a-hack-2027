# alive experiment `final16_atd_flip_who_v1`

```
{
 "tag": "final16_atd_flip_who_v1",
 "condition": "atd",
 "ball_modes": [
  "fuzzy"
 ],
 "seeds": [
  0
 ],
 "noise": [
  "team_flip_p=0.05",
  "gk_team_flip_p=0.5"
 ],
 "clean_tracks": true,
 "who": "who_v1",
 "who_bnd_alpha": 0.5,
 "models": {
  "asf_s0": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_asf_full_impff_s0.pt",
  "asf_s1": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_asf_full_impff_s1.pt",
  "asf_s2": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_asf_full_impff_s2.pt"
 },
 "dead_model": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\dead_v1.joblib",
 "ensemble": [
  "asf=asf_s0,asf_s1,asf_s2",
  "asf_v8=asf,v8"
 ],
 "deadball": [
  "v8=C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\deadball_v8.joblib:0.5"
 ],
 "hybrid_base": [],
 "variants": [
  "hmm"
 ],
 "variant_methods": null,
 "bnd_from": null,
 "tags": null,
 "params": {
  "tau_m": 1.5,
  "loose_m": 4.0,
  "still_speed": 1.3,
  "still_width": 0.35,
  "w_seen": 1.0,
  "w_unseen": 0.15,
  "w_filled": 0.4,
  "w_spread": 0.15,
  "w_tags": 0.4,
  "switch_cost": 18.0,
  "to_dead_cost": 10.0,
  "from_dead_cost": 10.0,
  "w_dead": 1.0,
  "dead_who": "max",
  "w_who_model": 1.0,
  "who_clip": 0.02
 },
 "rule_cfg": {
  "out_margin_m": 0.5,
  "spot_still_s": 1.0,
  "spot_line_m": 1.5,
  "spot_corner_m": 2.0,
  "spot_centre_m": 1.5,
  "slow_median_ms": 2.0,
  "hold_s": 0.6,
  "corner_m": 2.0,
  "corner_box": 6,
  "gk_box_max": 3,
  "kickoff_own_half": 0.95,
  "kickoff_centre_m": 2.0,
  "still_med_ms": 1.0,
  "still_p95_ms": 3.0,
  "still_min_s": 2.0,
  "long_break_s": 20.0,
  "burst_ms": 4.0,
  "burst_n": 3,
  "p_dead_on": 0.85,
  "p_dead_off": 0.15,
  "p_out": 0.97,
  "p_kick": 0.05
 },
 "ball_fill": true,
 "fuzzy": {
  "keep_p": 0.25,
  "start": [
   -1.0,
   3.0
  ],
  "end": [
   -2.0,
   1.0
  ]
 },
 "split": "val",
 "sources": [
  "idsse",
  "metrica",
  "pff"
 ],
 "jobs": 3,
 "minutes": 6.0
}
```

## atd / ball fuzzy -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 11    0.610    0.920          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.860          0.567            0.710        0.553     0.096 0.006 11/11
v8_hsmm    hsmm 11    0.568    0.888          0.968     0.499      0.832       0.967           1.637        0.397      0.417      -0.036         0.366         0.376        0.769          0.536            0.670        0.496     0.046 0.006 11/11
     v8     hmm 11    0.565    0.888          0.968     0.601      0.845       0.969           8.308        0.418      0.426      -0.020         0.371         0.376        0.783          0.534            0.662        0.498     0.046 0.005 11/11
 asf_v8     hmm 11    0.558    0.878          0.962     0.473      0.720       0.915           4.075        0.406      0.451      -0.065         0.349         0.376        0.768          0.531            0.644        0.499     0.044 0.004 11/11
    asf     hmm 11    0.539    0.853          0.943     0.427      0.653       0.841           4.439        0.570      0.483      -0.058         0.325         0.376        0.792          0.509            0.621        0.488     0.030 0.003 11/11
 asf_s2     hmm 11    0.536    0.851          0.941     0.412      0.636       0.871           5.576        0.624      0.555      -0.051         0.326         0.376        0.802          0.506            0.616        0.484     0.027 0.003 11/11
 asf_s0     hmm 11    0.530    0.847          0.938     0.388      0.636       0.830           6.045        0.605      0.578      -0.099         0.326         0.376        0.800          0.503            0.613        0.475     0.019 0.003 11/11
 asf_s1     hmm 11    0.526    0.846          0.938     0.482      0.618       0.821           5.647        0.618      0.505      -0.074         0.321         0.376        0.803          0.500            0.601        0.476     0.018 0.003 11/11
dead_v1     hmm 11    0.506    0.834          0.927     0.331      0.553       0.802          12.944        1.340      1.037      -0.242         0.325         0.376        0.819          0.478            0.580        0.459       NaN   NaN   NaN
     v0     hmm 11    0.472    0.758          0.880     0.122      0.245       0.472           9.746        3.349      1.080      -0.289         0.231         0.376        0.842          0.447            0.555        0.415    -0.040 0.004  0/11
  rules     hmm 11    0.388    0.742          0.843     0.487      0.599       0.769          71.342        2.817      3.031       2.364         0.385         0.376        0.992          0.421            0.385        0.357    -0.102 0.012  0/11
```
