# alive experiment `final16_bar_who_v1`

```
{
 "tag": "final16_bar_who_v1",
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
 "minutes": 12.8
}
```

## bar / ball exact -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 11    0.670    0.938          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.789          0.676            0.723        0.612     0.116 0.003 11/11
v8_hsmm    hsmm 11    0.627    0.912          0.976     0.768      0.872       0.993           2.539        0.279      0.344       0.012         0.374         0.376        0.746          0.650            0.696        0.535     0.053 0.008 11/11
     v8     hmm 11    0.626    0.912          0.976     0.805      0.887       0.993           7.419        0.247      0.302       0.029         0.377         0.376        0.756          0.650            0.691        0.538     0.055 0.008 11/11
 asf_v8     hmm 11    0.618    0.905          0.971     0.496      0.781       0.978           4.302        0.366      0.282      -0.008         0.363         0.376        0.739          0.642            0.668        0.544     0.055 0.006 11/11
    asf     hmm 11    0.598    0.890          0.960     0.326      0.725       0.946           4.792        0.644      0.463      -0.125         0.352         0.376        0.749          0.620            0.639        0.535     0.043 0.005 11/11
 asf_s2     hmm 11    0.589    0.886          0.956     0.261      0.709       0.946           6.855        0.703      0.574      -0.143         0.350         0.376        0.760          0.609            0.628        0.530     0.036 0.004 11/11
 asf_s1     hmm 11    0.589    0.885          0.956     0.319      0.684       0.946           6.103        0.774      0.514      -0.145         0.347         0.376        0.765          0.612            0.630        0.524     0.032 0.005 11/11
 asf_s0     hmm 11    0.586    0.885          0.956     0.422      0.727       0.947           6.756        0.744      0.561      -0.091         0.355         0.376        0.760          0.610            0.625        0.525     0.032 0.004 11/11
dead_v1     hmm 11    0.550    0.860          0.933     0.384      0.656       0.899          16.970        1.313      1.173       0.195         0.360         0.376        0.765          0.560            0.591        0.498       NaN   NaN   NaN
     v0     hmm 11    0.493    0.814          0.897     0.282      0.518       0.792          51.931        1.894      1.232       0.661         0.344         0.376        0.851          0.522            0.548        0.408    -0.076 0.008  0/11
  rules     hmm 11    0.385    0.754          0.834     0.536      0.660       0.846          94.412        2.587      3.528       2.932         0.434         0.376        0.975          0.452            0.346        0.357    -0.144 0.013  0/11
```

## bar / ball fuzzy -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 11    0.661    0.936          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.792          0.669            0.697        0.616     0.115 0.003 11/11
 asf_v8     hmm 11    0.614    0.905          0.972     0.516      0.852       0.981           4.027        0.337      0.322      -0.003         0.370         0.376        0.729          0.638            0.658        0.546     0.055 0.006 11/11
v8_hsmm    hsmm 11    0.611    0.904          0.970     0.667      0.867       0.988           3.766        0.356      0.366       0.027         0.379         0.376        0.734          0.630            0.666        0.537     0.048 0.006 11/11
     v8     hmm 11    0.606    0.903          0.969     0.734      0.863       0.988           9.402        0.346      0.364       0.033         0.382         0.376        0.743          0.620            0.661        0.538     0.046 0.005 11/11
    asf     hmm 11    0.599    0.896          0.965     0.431      0.813       0.977           5.682        0.505      0.365      -0.005         0.363         0.376        0.745          0.632            0.624        0.540     0.046 0.005 11/11
 asf_s2     hmm 11    0.596    0.893          0.962     0.433      0.789       0.973           6.820        0.519      0.449      -0.009         0.362         0.376        0.753          0.625            0.629        0.535     0.042 0.005 11/11
 asf_s1     hmm 11    0.591    0.892          0.962     0.468      0.778       0.973           6.418        0.577      0.417      -0.028         0.359         0.376        0.753          0.626            0.619        0.528     0.036 0.005 11/11
 asf_s0     hmm 11    0.590    0.890          0.960     0.517      0.811       0.949           7.216        0.559      0.488       0.033         0.364         0.376        0.749          0.627            0.612        0.530     0.037 0.006 11/11
dead_v1     hmm 11    0.547    0.861          0.934     0.432      0.672       0.899          14.030        1.296      1.144       0.185         0.362         0.376        0.762          0.560            0.580        0.500       NaN   NaN   NaN
     v0     hmm 11    0.494    0.816          0.899     0.356      0.560       0.810          44.953        1.875      1.239       0.670         0.349         0.376        0.848          0.525            0.549        0.409    -0.075 0.010  0/11
  rules     hmm 11    0.389    0.756          0.836     0.562      0.700       0.834          92.945        2.121      2.961       2.313         0.438         0.376        0.968          0.454            0.353        0.361    -0.141 0.012  0/11
```

## bar / ball random -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 11    0.646    0.933          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.787          0.625            0.699        0.616     0.112 0.004 11/11
 asf_v8     hmm 11    0.596    0.903          0.972     0.614      0.838       0.992           3.780        0.301      0.262       0.000         0.375         0.376        0.711          0.587            0.649        0.550     0.053 0.004 11/11
    asf     hmm 11    0.588    0.897          0.969     0.517      0.842       0.985           5.052        0.461      0.376      -0.003         0.370         0.376        0.724          0.588            0.627        0.547     0.049 0.004 11/11
v8_hsmm    hsmm 11    0.585    0.892          0.962     0.757      0.828       0.983           6.179        0.363      0.364       0.062         0.388         0.376        0.717          0.568            0.653        0.535     0.038 0.005 11/11
 asf_s2     hmm 11    0.583    0.893          0.966     0.531      0.812       0.981           6.173        0.456      0.446       0.012         0.368         0.376        0.727          0.585            0.624        0.541     0.043 0.004 11/11
     v8     hmm 11    0.580    0.892          0.961     0.781      0.844       0.985          12.622        0.355      0.339       0.041         0.391         0.376        0.735          0.559            0.649        0.532     0.035 0.005 11/11
 asf_s0     hmm 11    0.580    0.892          0.965     0.549      0.836       0.966           6.506        0.459      0.437       0.016         0.371         0.376        0.733          0.582            0.618        0.540     0.041 0.004 11/11
 asf_s1     hmm 11    0.579    0.892          0.965     0.534      0.806       0.987           6.744        0.511      0.413      -0.017         0.368         0.376        0.735          0.586            0.615        0.536     0.039 0.005 11/11
dead_v1     hmm 11    0.536    0.859          0.936     0.412      0.686       0.904          13.875        1.234      1.132       0.193         0.364         0.376        0.746          0.523            0.582        0.502       NaN   NaN   NaN
     v0     hmm 11    0.477    0.816          0.901     0.315      0.567       0.800          44.220        1.845      1.303       0.582         0.347         0.376        0.849          0.489            0.540        0.402    -0.083 0.010  0/11
  rules     hmm 11    0.384    0.764          0.844     0.662      0.773       0.872          93.378        1.955      2.871       2.129         0.452         0.376        0.954          0.436            0.353        0.363    -0.138 0.013  0/11
```
