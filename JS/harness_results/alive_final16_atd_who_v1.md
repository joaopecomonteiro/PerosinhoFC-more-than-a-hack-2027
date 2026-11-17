# alive experiment `final16_atd_who_v1`

```
{
 "tag": "final16_atd_who_v1",
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
 "minutes": 18.2
}
```

## atd / ball fuzzy -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 22    0.657    0.935          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.795          0.614            0.756        0.601     0.108 0.003 22/22
v8_hsmm    hsmm 22    0.606    0.902          0.969     0.583      0.850       0.972           2.356        0.418      0.442      -0.034         0.369         0.376        0.715          0.566            0.720        0.532     0.046 0.004 22/22
     v8     hmm 22    0.603    0.902          0.969     0.656      0.857       0.979           8.025        0.432      0.429      -0.015         0.373         0.376        0.723          0.565            0.713        0.532     0.046 0.003 22/22
 asf_v8     hmm 22    0.599    0.892          0.962     0.494      0.765       0.934           3.860        0.430      0.405      -0.043         0.351         0.376        0.716          0.570            0.691        0.538     0.048 0.002 22/22
    asf     hmm 22    0.582    0.872          0.948     0.446      0.680       0.878           5.092        0.611      0.479      -0.062         0.333         0.376        0.733          0.554            0.662        0.529     0.037 0.002 22/22
 asf_s0     hmm 22    0.575    0.868          0.944     0.491      0.699       0.872           6.602        0.654      0.551      -0.083         0.334         0.376        0.743          0.550            0.655        0.520     0.029 0.002 22/22
 asf_s2     hmm 22    0.575    0.868          0.945     0.432      0.657       0.879           6.061        0.635      0.565      -0.097         0.330         0.376        0.743          0.549            0.652        0.523     0.031 0.002 22/22
 asf_s1     hmm 22    0.573    0.867          0.944     0.465      0.669       0.866           6.065        0.658      0.493      -0.083         0.329         0.376        0.745          0.547            0.650        0.521     0.029 0.003 22/22
dead_v1     hmm 22    0.545    0.852          0.931     0.347      0.585       0.830          11.712        1.219      1.078      -0.155         0.332         0.376        0.757          0.511            0.632        0.493       NaN   NaN   NaN
     v0     hmm 22    0.501    0.764          0.877     0.099      0.240       0.439           9.222        3.095      1.066      -0.187         0.225         0.376        0.786          0.471            0.593        0.439    -0.050 0.003  0/22
  rules     hmm 22    0.410    0.753          0.843     0.487      0.619       0.783          73.573        2.851      2.926       2.231         0.388         0.376        0.955          0.441            0.412        0.376    -0.118 0.009  0/22
```

## atd / ball exact -- mean per source, then over sources; d_vs_ref paired vs `dead_v1` `hmm`

```
 method variant  n  spellF1  macroF1  alive_dead_f1  rec_0-5s  rec_5-12s  rec_12-30s  spurious_per90  start_med_s  end_med_s  end_bias_s  d_share_pred  d_share_true  spell_ratio  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
 oracle       - 22    0.661    0.936          1.000     1.000      1.000       1.000           0.000        0.000      0.000       0.000         0.376         0.376        0.796          0.624            0.761        0.598     0.111 0.003 22/22
v8_hsmm    hsmm 22    0.613    0.905          0.971     0.648      0.862       0.977           2.235        0.338      0.382      -0.009         0.372         0.376        0.712          0.575            0.730        0.534     0.052 0.004 22/22
     v8     hmm 22    0.611    0.905          0.971     0.723      0.869       0.985           7.736        0.342      0.341       0.010         0.375         0.376        0.720          0.575            0.724        0.534     0.052 0.004 22/22
 asf_v8     hmm 22    0.599    0.890          0.961     0.464      0.746       0.933           3.820        0.466      0.381      -0.047         0.349         0.376        0.715          0.568            0.698        0.533     0.047 0.003 22/22
    asf     hmm 22    0.576    0.866          0.943     0.365      0.647       0.866           5.136        0.720      0.516      -0.158         0.326         0.376        0.737          0.542            0.666        0.521     0.031 0.002 22/22
 asf_s2     hmm 22    0.570    0.862          0.940     0.345      0.636       0.869           6.400        0.738      0.633      -0.187         0.324         0.376        0.745          0.541            0.656        0.514     0.025 0.002 22/22
 asf_s1     hmm 22    0.568    0.860          0.939     0.377      0.633       0.833           6.250        0.785      0.516      -0.135         0.322         0.376        0.750          0.537            0.655        0.511     0.022 0.003 19/22
 asf_s0     hmm 22    0.566    0.860          0.938     0.443      0.644       0.853           7.204        0.764      0.600      -0.200         0.326         0.376        0.745          0.538            0.650        0.511     0.022 0.002 22/22
dead_v1     hmm 22    0.545    0.850          0.929     0.333      0.571       0.826          15.056        1.326      1.096      -0.186         0.330         0.376        0.753          0.505            0.637        0.491       NaN   NaN   NaN
     v0     hmm 22    0.496    0.760          0.874     0.106      0.205       0.426          13.541        3.020      1.066      -0.226         0.220         0.376        0.760          0.461            0.593        0.433    -0.055 0.004  0/22
  rules     hmm 22    0.405    0.746          0.837     0.409      0.583       0.741          73.249        3.440      3.085       2.321         0.382         0.376        0.959          0.434            0.408        0.372    -0.120 0.009  0/22
```
