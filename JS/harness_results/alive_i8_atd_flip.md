# alive experiment `i8_atd_flip`

```
{
 "tag": "i8_atd_flip",
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
 "who": "tcn=C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s0.pt,C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s1.pt,C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s2.pt",
 "who_bnd_alpha": 0.5,
 "models": {
  "asf_s0": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_asf_full_impff_s0.pt",
  "asf_s1": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_asf_full_impff_s1.pt",
  "asf_s2": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_asf_full_impff_s2.pt",
  "gbm_full": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_gbm_full_impff.joblib",
  "gbm_i8": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_gbm_full_i8_impff.joblib",
  "asfi_s0": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_asf_full_i8_impff_s0.pt",
  "asfi_s1": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_asf_full_i8_impff_s1.pt",
  "asfi_s2": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\alive_asf_full_i8_impff_s2.pt"
 },
 "dead_model": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\dead_v1.joblib",
 "ensemble": [
  "asf=asf_s0,asf_s1,asf_s2",
  "asfi=asfi_s0,asfi_s1,asfi_s2",
  "fin=asf,v8",
  "fin_i8=asfi,v8"
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
 "minutes": 7.5
}
```

## atd / ball fuzzy -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
  method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
  oracle       - 11    0.678    0.939          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.803          0.633            0.763        0.637     0.116 0.005 11/11
 v8_hsmm    hsmm 11    0.625    0.905          0.968     0.499      0.832       0.967           1.637        0.397      0.417      -0.036         0.366         0.376        0.752          0.593            0.721        0.560     0.049 0.005 11/11
  fin_i8     hmm 11    0.619    0.896          0.963     0.488      0.723       0.945           3.989        0.391      0.366      -0.069         0.349         0.376        0.759          0.587            0.705        0.564     0.049 0.004 11/11
      v8     hmm 11    0.616    0.904          0.968     0.601      0.848       0.968           8.167        0.424      0.419      -0.032         0.370         0.376        0.769          0.585            0.704        0.559     0.045 0.004 11/11
     fin     hmm 11    0.613    0.892          0.960     0.468      0.704       0.923           3.770        0.425      0.424      -0.072         0.346         0.376        0.759          0.586            0.691        0.561     0.046 0.004 11/11
    asfi     hmm 11    0.602    0.882          0.951     0.444      0.679       0.892           5.353        0.545      0.403      -0.093         0.335         0.376        0.776          0.575            0.679        0.552     0.036 0.004 11/11
 asfi_s2     hmm 11    0.598    0.880          0.950     0.502      0.668       0.877           5.835        0.556      0.434      -0.041         0.338         0.376        0.781          0.572            0.676        0.546     0.031 0.003 11/11
 asfi_s0     hmm 11    0.597    0.877          0.947     0.526      0.659       0.885           6.230        0.579      0.458      -0.070         0.335         0.376        0.789          0.567            0.678        0.547     0.031 0.005 11/11
     asf     hmm 11    0.594    0.869          0.943     0.427      0.653       0.836           4.255        0.568      0.473      -0.058         0.325         0.376        0.779          0.562            0.668        0.551     0.032 0.003 11/11
  asf_s2     hmm 11    0.592    0.868          0.941     0.412      0.636       0.871           5.539        0.606      0.555      -0.072         0.325         0.376        0.790          0.565            0.663        0.549     0.030 0.004 11/11
 asfi_s1     hmm 11    0.590    0.869          0.941     0.429      0.649       0.858           7.048        0.604      0.436      -0.131         0.325         0.376        0.790          0.560            0.665        0.543     0.026 0.003 11/11
  asf_s0     hmm 11    0.586    0.864          0.938     0.388      0.629       0.830           5.971        0.612      0.577      -0.100         0.326         0.376        0.781          0.561            0.656        0.540     0.023 0.003 11/11
  gbm_i8     hmm 11    0.583    0.866          0.940     0.466      0.669       0.823           7.764        0.631      0.872      -0.394         0.326         0.376        0.783          0.555            0.657        0.537     0.020 0.002 11/11
  asf_s1     hmm 11    0.580    0.863          0.939     0.482      0.615       0.821           5.645        0.635      0.510      -0.084         0.321         0.376        0.790          0.554            0.649        0.538     0.019 0.003 11/11
gbm_full     hmm 11    0.574    0.857          0.934     0.453      0.644       0.821           7.535        0.805      1.083      -0.575         0.318         0.376        0.797          0.545            0.642        0.534     0.014 0.002 11/11
 dead_v1     hmm 11    0.560    0.849          0.926     0.300      0.549       0.784          11.168        1.349      1.037      -0.216         0.323         0.376        0.807          0.533            0.628        0.520       NaN   NaN   NaN
      v0     hmm 11    0.508    0.739          0.862     0.057      0.186       0.348           6.075        5.344      1.086      -0.296         0.195         0.376        0.838          0.493            0.586        0.445    -0.065 0.008  0/11
   rules     hmm 11    0.427    0.757          0.843     0.487      0.599       0.769          70.719        2.817      3.031       2.364         0.385         0.376        0.985          0.457            0.427        0.397    -0.121 0.012  0/11
```
