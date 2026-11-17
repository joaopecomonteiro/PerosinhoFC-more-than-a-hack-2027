# alive experiment `final16_bar_tcn`

```
{
 "tag": "final16_bar_tcn",
 "condition": "bar",
 "ball_modes": [
  "exact",
  "fuzzy",
  "random"
 ],
 "seeds": [
  0
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
 "minutes": 17.6
}
```

## bar / ball exact -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 11    0.735    0.950          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.824          0.721            0.815        0.670     0.127 0.004 11/11
v8_hsmm    hsmm 11    0.689    0.924          0.976     0.768      0.872       0.993           2.539        0.279      0.344       0.012         0.374         0.376        0.796          0.695            0.787        0.584     0.057 0.009 11/11
     v8     hmm 11    0.687    0.924          0.976     0.805      0.882       0.993           7.391        0.253      0.309       0.022         0.376         0.376        0.805          0.695            0.779        0.586     0.058 0.008 11/11
 asf_v8     hmm 11    0.680    0.917          0.971     0.496      0.777       0.978           4.617        0.361      0.276      -0.012         0.362         0.376        0.794          0.689            0.756        0.593     0.060 0.006 11/11
    asf     hmm 11    0.658    0.902          0.960     0.326      0.723       0.946           5.043        0.680      0.463      -0.129         0.351         0.376        0.807          0.666            0.725        0.583     0.046 0.005 11/11
 asf_s1     hmm 11    0.652    0.897          0.956     0.319      0.683       0.946           6.419        0.770      0.515      -0.145         0.347         0.376        0.820          0.659            0.723        0.573     0.037 0.005 11/11
 asf_s0     hmm 11    0.650    0.898          0.956     0.422      0.727       0.947           7.072        0.725      0.559      -0.093         0.355         0.376        0.813          0.655            0.719        0.575     0.038 0.005 11/11
 asf_s2     hmm 11    0.649    0.898          0.956     0.261      0.709       0.946           7.329        0.721      0.578      -0.147         0.350         0.376        0.821          0.653            0.716        0.577     0.038 0.004 11/11
dead_v1     hmm 11    0.608    0.872          0.933     0.381      0.655       0.897          15.391        1.298      1.172       0.130         0.359         0.376        0.818          0.598            0.684        0.542       NaN   NaN   NaN
     v0     hmm 11    0.540    0.820          0.895     0.258      0.449       0.784          45.878        2.075      1.109       0.556         0.331         0.376        0.918          0.557            0.627        0.437    -0.089 0.010  0/11
  rules     hmm 11    0.428    0.764          0.834     0.536      0.660       0.846          94.352        2.590      3.531       2.935         0.434         0.376        1.027          0.487            0.409        0.388    -0.157 0.015  0/11
```

## bar / ball fuzzy -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 11    0.725    0.947          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.822          0.710            0.793        0.672     0.123 0.005 11/11
 asf_v8     hmm 11    0.676    0.916          0.972     0.516      0.848       0.981           3.713        0.347      0.317      -0.014         0.369         0.376        0.778          0.684            0.743        0.599     0.061 0.005 11/11
v8_hsmm    hsmm 11    0.674    0.915          0.970     0.667      0.867       0.988           3.766        0.356      0.366       0.027         0.379         0.376        0.779          0.674            0.757        0.590     0.054 0.007 11/11
     v8     hmm 11    0.669    0.914          0.969     0.734      0.860       0.988           9.030        0.336      0.362       0.026         0.381         0.376        0.787          0.670            0.747        0.590     0.052 0.006 11/11
    asf     hmm 11    0.662    0.907          0.965     0.431      0.811       0.976           5.330        0.507      0.358      -0.009         0.363         0.376        0.793          0.678            0.718        0.589     0.050 0.005 11/11
 asf_s2     hmm 11    0.657    0.904          0.962     0.433      0.789       0.973           6.505        0.518      0.452      -0.009         0.362         0.376        0.807          0.671            0.716        0.583     0.044 0.006 11/11
 asf_s1     hmm 11    0.655    0.903          0.963     0.468      0.778       0.973           6.067        0.575      0.416      -0.030         0.359         0.376        0.802          0.668            0.714        0.582     0.043 0.005 11/11
 asf_s0     hmm 11    0.654    0.901          0.960     0.517      0.809       0.949           6.863        0.565      0.496       0.030         0.364         0.376        0.798          0.671            0.707        0.585     0.045 0.006 11/11
dead_v1     hmm 11    0.605    0.872          0.935     0.427      0.672       0.902          12.873        1.257      1.139       0.159         0.361         0.376        0.808          0.597            0.672        0.546       NaN   NaN   NaN
     v0     hmm 11    0.542    0.822          0.898     0.264      0.496       0.799          40.122        2.006      1.204       0.584         0.336         0.376        0.907          0.562            0.625        0.441    -0.087 0.011  0/11
  rules     hmm 11    0.434    0.766          0.836     0.562      0.700       0.834          92.886        2.121      2.961       2.313         0.438         0.376        1.014          0.489            0.420        0.393    -0.154 0.014  0/11
```

## bar / ball random -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 11    0.719    0.946          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.800          0.678            0.803        0.675     0.126 0.005 11/11
 asf_v8     hmm 11    0.666    0.915          0.972     0.610      0.834       0.992           4.059        0.280      0.254      -0.004         0.375         0.376        0.750          0.646            0.749        0.602     0.062 0.004 11/11
    asf     hmm 11    0.656    0.910          0.969     0.517      0.838       0.985           5.246        0.462      0.378      -0.004         0.370         0.376        0.769          0.645            0.725        0.598     0.056 0.004 11/11
v8_hsmm    hsmm 11    0.652    0.905          0.962     0.757      0.828       0.983           6.179        0.363      0.364       0.062         0.388         0.376        0.752          0.623            0.750        0.583     0.044 0.005 11/11
 asf_s2     hmm 11    0.651    0.906          0.966     0.531      0.812       0.981           6.452        0.447      0.446       0.010         0.368         0.376        0.770          0.641            0.722        0.588     0.049 0.005 11/11
 asf_s0     hmm 11    0.650    0.904          0.965     0.549      0.836       0.966           6.821        0.465      0.438       0.012         0.371         0.376        0.774          0.640            0.719        0.590     0.049 0.005 11/11
     v8     hmm 11    0.647    0.904          0.961     0.781      0.838       0.985          12.252        0.354      0.335       0.029         0.390         0.376        0.769          0.614            0.745        0.582     0.041 0.005 11/11
 asf_s1     hmm 11    0.645    0.904          0.965     0.534      0.806       0.987           7.374        0.516      0.413      -0.017         0.368         0.376        0.782          0.639            0.710        0.586     0.045 0.005 11/11
dead_v1     hmm 11    0.596    0.872          0.936     0.403      0.686       0.904          12.718        1.215      1.134       0.140         0.362         0.376        0.784          0.570            0.671        0.546       NaN   NaN   NaN
     v0     hmm 11    0.530    0.824          0.900     0.272      0.511       0.799          39.394        1.867      1.266       0.530         0.336         0.376        0.897          0.535            0.619        0.436    -0.091 0.011  0/11
  rules     hmm 11    0.427    0.776          0.844     0.662      0.773       0.872          93.319        1.955      2.871       2.129         0.452         0.376        0.986          0.466            0.418        0.395    -0.151 0.014  0/11
```
