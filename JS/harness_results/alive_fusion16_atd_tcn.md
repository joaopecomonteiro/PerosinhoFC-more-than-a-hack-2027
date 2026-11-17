# alive experiment `fusion16_atd_tcn`

```
{
 "tag": "fusion16_atd_tcn",
 "condition": "atd",
 "ball_modes": [
  "fuzzy",
  "exact"
 ],
 "seeds": [
  0,
  1
 ],
 "noise": [],
 "clean_tracks": true,
 "who": "tcn=C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s0.pt,C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s1.pt,C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s2.pt",
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
 "minutes": 43.8
}
```

## atd / ball fuzzy -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 22    0.710    0.945          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.816          0.664            0.799        0.666     0.116 0.002 22/22
v8_hsmm    hsmm 22    0.655    0.911          0.969     0.583      0.850       0.972           2.356        0.418      0.442      -0.034         0.369         0.376        0.777          0.619            0.759        0.586     0.046 0.004 22/22
     v8     hmm 22    0.651    0.911          0.969     0.656      0.852       0.973           6.967        0.415      0.426      -0.021         0.372         0.376        0.785          0.619            0.749        0.586     0.045 0.003 22/22
 asf_v8     hmm 22    0.648    0.900          0.962     0.479      0.760       0.929           3.454        0.446      0.402      -0.056         0.349         0.376        0.780          0.619            0.731        0.592     0.048 0.002 22/22
    asf     hmm 22    0.631    0.881          0.948     0.443      0.676       0.878           4.899        0.612      0.476      -0.071         0.332         0.376        0.796          0.607            0.701        0.584     0.037 0.002 22/22
 asf_s2     hmm 22    0.624    0.877          0.945     0.432      0.656       0.879           5.710        0.648      0.563      -0.099         0.330         0.376        0.805          0.602            0.694        0.577     0.031 0.002 22/22
 asf_s0     hmm 22    0.622    0.877          0.944     0.491      0.699       0.872           6.269        0.652      0.535      -0.090         0.334         0.376        0.804          0.601            0.691        0.575     0.029 0.003 22/22
 asf_s1     hmm 22    0.622    0.876          0.944     0.465      0.668       0.866           5.969        0.681      0.489      -0.085         0.329         0.376        0.811          0.599            0.692        0.576     0.029 0.002 22/22
dead_v1     hmm 22    0.593    0.861          0.931     0.344      0.583       0.829          10.361        1.268      1.072      -0.198         0.330         0.376        0.819          0.561            0.672        0.548       NaN   NaN   NaN
     v0     hmm 22    0.535    0.744          0.862     0.046      0.179       0.342           5.871        4.377      1.022      -0.285         0.195         0.376        0.858          0.514            0.616        0.474    -0.068 0.005  0/22
  rules     hmm 22    0.441    0.761          0.843     0.487      0.619       0.783          73.620        2.851      2.933       2.231         0.388         0.376        1.015          0.477            0.437        0.410    -0.137 0.009  0/22
```

## atd / ball exact -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 22    0.709    0.945          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.820          0.662            0.805        0.662     0.116 0.003 22/22
v8_hsmm    hsmm 22    0.661    0.914          0.971     0.648      0.862       0.977           2.235        0.338      0.382      -0.009         0.372         0.376        0.772          0.626            0.768        0.588     0.052 0.005 22/22
     v8     hmm 22    0.658    0.914          0.971     0.721      0.869       0.976           6.800        0.335      0.340       0.005         0.374         0.376        0.780          0.625            0.762        0.587     0.051 0.004 22/22
 asf_v8     hmm 22    0.645    0.899          0.960     0.462      0.734       0.926           3.475        0.469      0.377      -0.054         0.347         0.376        0.779          0.616            0.733        0.587     0.047 0.003 22/22
    asf     hmm 22    0.622    0.875          0.943     0.358      0.645       0.868           4.844        0.721      0.516      -0.165         0.325         0.376        0.800          0.591            0.703        0.574     0.030 0.002 22/22
 asf_s2     hmm 22    0.618    0.871          0.939     0.347      0.632       0.870           5.892        0.748      0.627      -0.189         0.324         0.376        0.806          0.590            0.697        0.567     0.025 0.002 22/22
 asf_s1     hmm 22    0.613    0.869          0.939     0.377      0.631       0.832           6.191        0.786      0.517      -0.137         0.321         0.376        0.812          0.586            0.689        0.564     0.021 0.002 22/22
 asf_s0     hmm 22    0.612    0.869          0.938     0.443      0.644       0.853           6.382        0.767      0.596      -0.205         0.326         0.376        0.807          0.582            0.687        0.566     0.022 0.002 21/22
dead_v1     hmm 22    0.590    0.858          0.929     0.323      0.563       0.825          12.430        1.339      1.112      -0.243         0.328         0.376        0.815          0.553            0.670        0.546       NaN   NaN   NaN
     v0     hmm 22    0.524    0.733          0.856     0.044      0.122       0.305           9.252        5.722      1.061      -0.404         0.185         0.376        0.840          0.487            0.621        0.465    -0.075 0.005  0/22
  rules     hmm 22    0.435    0.754          0.837     0.409      0.583       0.741          73.138        3.440      3.085       2.321         0.382         0.376        1.018          0.470            0.431        0.405    -0.139 0.010  0/22
```
