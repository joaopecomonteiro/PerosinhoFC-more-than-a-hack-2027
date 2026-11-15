# owner experiment `headchk_tcn_atd_flip`

```
{
 "tag": "headchk_tcn_atd_flip",
 "condition": "atd",
 "seeds": [
  0,
  1
 ],
 "layers": [
  "dead_v1",
  "oracle"
 ],
 "decoders": [
  "hmm"
 ],
 "models": {
  "who_v1": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\who_v1.joblib",
  "tcn_full_impff_s0": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s0.pt",
  "tcn_full_impff_s1": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s1.pt",
  "tcn_full_impff_s2": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s2.pt",
  "tcn_full_heading_impff_s0": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_heading_impff_s0.pt",
  "tcn_full_heading_impff_s1": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_heading_impff_s1.pt",
  "tcn_full_heading_impff_s2": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_heading_impff_s2.pt"
 },
 "hybrid_base": [],
 "ref": "who_v1",
 "external": [],
 "boundary_from": [
  "tcn_full_impff_s0",
  "tcn_full_impff_s1",
  "tcn_full_impff_s2",
  "tcn_full_heading_impff_s0",
  "tcn_full_heading_impff_s1",
  "tcn_full_heading_impff_s2"
 ],
 "bnd_alpha": [
  0.5
 ],
 "ensemble": [
  "ens_impff=tcn_full_impff_s0,tcn_full_impff_s1,tcn_full_impff_s2",
  "ens_impff_hd=tcn_full_heading_impff_s0,tcn_full_heading_impff_s1,tcn_full_heading_impff_s2"
 ],
 "clean_tracks": true,
 "noise": [
  "team_flip_p=0.05",
  "gk_team_flip_p=0.5"
 ],
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
 "switch_cost_2state": 18.0,
 "rule_cfg": {
  "r_touch_m": 1.5,
  "r_free_m": 3.0,
  "free_from": "opponents",
  "v_match_ms": 2.5,
  "dwell_s": 0.4,
  "gap_tol_s": 0.2,
  "pass_chain": 2,
  "chain_gap_s": 3.0,
  "vel_half_s": 0.2,
  "filled_can_control": false,
  "tau_s": 4.0,
  "p_ctrl": 0.97,
  "p_hold": 0.9,
  "p_floor": 0.65,
  "p_contest": 0.6
 },
 "ball_fill": true,
 "dead_model": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\dead_v1.joblib",
 "split": "val",
 "sources": [
  "idsse",
  "metrica",
  "pff"
 ],
 "minutes": 25.1
}
```

## Per alive frame (mean over games per source, then over sources)

```
                           acc_all  acc_seen  acc_missing  auc_all  logloss_all  brier_all  ece_all
method                                                                                             
ens_impff_hd                 0.911     0.935        0.861    0.967        0.230      0.066    0.030
ens_impff                    0.910     0.934        0.862    0.967        0.230      0.067    0.031
tcn_full_heading_impff_s0    0.909     0.934        0.856    0.966        0.240      0.069    0.037
tcn_full_impff_s1            0.908     0.933        0.859    0.965        0.243      0.069    0.039
tcn_full_heading_impff_s1    0.907     0.932        0.858    0.964        0.248      0.071    0.038
tcn_full_heading_impff_s2    0.907     0.932        0.856    0.964        0.245      0.070    0.038
tcn_full_impff_s0            0.906     0.931        0.857    0.964        0.247      0.071    0.039
tcn_full_impff_s2            0.906     0.930        0.856    0.964        0.249      0.071    0.040
who_v1                       0.876     0.905        0.816    0.948        0.291      0.089    0.033
rule_hard                    0.808     0.845        0.729    0.804        0.768      0.185    0.172
rule                         0.808     0.845        0.729    0.846        0.481      0.151    0.058
smooth50                     0.767     0.803        0.690    0.779        0.857      0.211    0.190
blend                        0.764     0.801        0.689    0.808        0.567      0.180    0.084
hold                         0.761     0.802        0.674    0.758        0.950      0.239    0.239
nearest                      0.703     0.735        0.640    0.770        0.646      0.205    0.094
```

## Layer `dead_v1` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
                                method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
      tcn_full_heading_impff_s0+bnd0.5     hmm    0.562    0.846     0.914             0.833       0.186        0.598 22          0.525            0.649        0.511     0.054 0.006 22/22
                   ens_impff_hd+bnd0.5     hmm    0.559    0.846     0.914             0.806       0.178        0.580 22          0.520            0.647        0.509     0.052 0.005 22/22
              tcn_full_impff_s1+bnd0.5     hmm    0.557    0.846     0.913             0.838       0.190        0.583 22          0.532            0.624        0.517     0.058 0.005 22/22
      tcn_full_heading_impff_s1+bnd0.5     hmm    0.557    0.846     0.913             0.836       0.188        0.585 22          0.517            0.640        0.512     0.053 0.006 22/22
                      ens_impff+bnd0.5     hmm    0.556    0.846     0.914             0.813       0.175        0.580 22          0.530            0.625        0.512     0.054 0.005 22/22
             tcn_full_heading_impff_s0     hmm    0.552    0.846     0.914             0.770       0.150        0.567 22          0.516            0.639        0.501     0.045 0.005 22/22
      tcn_full_heading_impff_s2+bnd0.5     hmm    0.552    0.844     0.911             0.825       0.185        0.569 22          0.516            0.634        0.506     0.048 0.005 21/22
              tcn_full_impff_s0+bnd0.5     hmm    0.552    0.844     0.911             0.830       0.185        0.573 22          0.522            0.625        0.508     0.050 0.005 22/22
             tcn_full_heading_impff_s1     hmm    0.551    0.846     0.913             0.777       0.155        0.561 22          0.510            0.636        0.506     0.047 0.005 22/22
                          ens_impff_hd     hmm    0.551    0.846     0.914             0.753       0.149        0.551 22          0.509            0.641        0.502     0.044 0.005 22/22
              tcn_full_impff_s2+bnd0.5     hmm    0.551    0.844     0.910             0.838       0.177        0.581 22          0.522            0.622        0.508     0.049 0.005 22/22
                     tcn_full_impff_s1     hmm    0.550    0.846     0.913             0.781       0.157        0.561 22          0.518            0.625        0.508     0.049 0.005 22/22
                             ens_impff     hmm    0.546    0.846     0.913             0.761       0.147        0.550 22          0.512            0.620        0.505     0.045 0.006 21/22
                     tcn_full_impff_s0     hmm    0.546    0.844     0.911             0.777       0.158        0.550 22          0.511            0.623        0.503     0.044 0.006 21/22
             tcn_full_heading_impff_s2     hmm    0.545    0.844     0.911             0.766       0.155        0.542 22          0.503            0.631        0.502     0.042 0.006 20/22
                     tcn_full_impff_s2     hmm    0.545    0.844     0.910             0.768       0.146        0.549 22          0.507            0.624        0.503     0.043 0.006 21/22
                who_v1+ens_impffbnd0.5     hmm    0.517    0.832     0.891             0.882       0.185        0.539 22          0.486            0.607        0.459     0.006 0.002 14/22
        who_v1+tcn_full_impff_s1bnd0.5     hmm    0.517    0.832     0.891             0.888       0.186        0.541 22          0.485            0.605        0.461     0.007 0.002 16/22
        who_v1+tcn_full_impff_s2bnd0.5     hmm    0.517    0.832     0.891             0.885       0.184        0.542 22          0.486            0.605        0.459     0.006 0.002 14/22
        who_v1+tcn_full_impff_s0bnd0.5     hmm    0.516    0.832     0.891             0.887       0.188        0.536 22          0.485            0.605        0.459     0.005 0.002 14/22
who_v1+tcn_full_heading_impff_s0bnd0.5     hmm    0.516    0.832     0.891             0.889       0.187        0.540 22          0.484            0.606        0.459     0.005 0.002 14/22
who_v1+tcn_full_heading_impff_s2bnd0.5     hmm    0.516    0.832     0.891             0.887       0.188        0.539 22          0.487            0.604        0.458     0.005 0.002 14/22
             who_v1+ens_impff_hdbnd0.5     hmm    0.516    0.832     0.891             0.883       0.185        0.539 22          0.485            0.605        0.459     0.006 0.002 17/22
who_v1+tcn_full_heading_impff_s1bnd0.5     hmm    0.515    0.832     0.890             0.884       0.185        0.536 22          0.484            0.602        0.459     0.005 0.002 15/22
                                who_v1     hmm    0.506    0.831     0.889             0.814       0.140        0.510 22          0.471            0.589        0.457       NaN   NaN   NaN
                              rule_seg     hmm    0.380    0.785     0.814             0.817       0.075        0.347 22          0.339            0.451        0.349    -0.115 0.006  0/22
                                  rule     hmm    0.376    0.782     0.812             0.816       0.073        0.340 22          0.336            0.449        0.343    -0.120 0.006  0/22
                             rule_hard     hmm    0.376    0.782     0.812             0.830       0.076        0.343 22          0.336            0.449        0.342    -0.121 0.006  0/22
                                 blend     hmm    0.353    0.776     0.802             1.101       0.116        0.389 22          0.349            0.397        0.313    -0.145 0.007  0/22
                              smooth50     hmm    0.308    0.761     0.777             1.135       0.068        0.349 22          0.302            0.326        0.296    -0.172 0.010  0/22
                               nearest     hmm    0.299    0.767     0.786             1.132       0.076        0.326 22          0.283            0.331        0.282    -0.185 0.009  0/22
                                  hold     hmm    0.242    0.754     0.768             2.130       0.206        0.415 22          0.265            0.242        0.220    -0.242 0.012  0/22
```

## Layer `oracle` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
                                method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
                   ens_impff_hd+bnd0.5     hmm    0.684    0.940     0.912             0.810       0.319        0.711 22          0.632            0.782        0.637     0.073 0.006 22/22
                      ens_impff+bnd0.5     hmm    0.680    0.939     0.911             0.810       0.315        0.706 22          0.643            0.759        0.640     0.075 0.006 22/22
      tcn_full_heading_impff_s0+bnd0.5     hmm    0.680    0.939     0.910             0.839       0.326        0.724 22          0.634            0.774        0.632     0.070 0.006 22/22
                          ens_impff_hd     hmm    0.677    0.940     0.911             0.769       0.286        0.685 22          0.625            0.769        0.636     0.070 0.006 22/22
      tcn_full_heading_impff_s1+bnd0.5     hmm    0.677    0.938     0.909             0.836       0.327        0.708 22          0.627            0.770        0.633     0.068 0.005 22/22
              tcn_full_impff_s1+bnd0.5     hmm    0.677    0.938     0.909             0.839       0.333        0.706 22          0.635            0.757        0.637     0.072 0.006 22/22
             tcn_full_heading_impff_s1     hmm    0.676    0.938     0.910             0.795       0.298        0.699 22          0.616            0.778        0.636     0.069 0.007 22/22
                     tcn_full_impff_s1     hmm    0.676    0.939     0.910             0.793       0.301        0.694 22          0.636            0.758        0.635     0.070 0.006 22/22
             tcn_full_heading_impff_s0     hmm    0.676    0.939     0.910             0.792       0.290        0.706 22          0.631            0.763        0.634     0.069 0.006 22/22
                             ens_impff     hmm    0.675    0.939     0.911             0.775       0.288        0.690 22          0.636            0.749        0.640     0.073 0.006 22/22
      tcn_full_heading_impff_s2+bnd0.5     hmm    0.675    0.937     0.908             0.835       0.336        0.697 22          0.628            0.764        0.632     0.067 0.005 22/22
              tcn_full_impff_s2+bnd0.5     hmm    0.672    0.937     0.907             0.842       0.317        0.712 22          0.632            0.751        0.633     0.068 0.007 22/22
              tcn_full_impff_s0+bnd0.5     hmm    0.672    0.937     0.908             0.835       0.324        0.701 22          0.635            0.753        0.628     0.065 0.005 22/22
                     tcn_full_impff_s0     hmm    0.671    0.937     0.908             0.793       0.298        0.691 22          0.631            0.753        0.629     0.065 0.006 22/22
                     tcn_full_impff_s2     hmm    0.670    0.937     0.907             0.794       0.291        0.689 22          0.626            0.748        0.636     0.069 0.007 22/22
             tcn_full_heading_impff_s2     hmm    0.669    0.937     0.908             0.786       0.298        0.672 22          0.619            0.756        0.631     0.064 0.006 22/22
        who_v1+tcn_full_impff_s2bnd0.5     hmm    0.624    0.923     0.887             0.925       0.332        0.669 22          0.581            0.739        0.553    -0.001 0.003  9/22
who_v1+tcn_full_heading_impff_s0bnd0.5     hmm    0.624    0.923     0.887             0.927       0.334        0.669 22          0.582            0.737        0.553    -0.001 0.003  8/22
             who_v1+ens_impff_hdbnd0.5     hmm    0.623    0.923     0.887             0.921       0.331        0.666 22          0.577            0.739        0.553    -0.002 0.003  9/22
who_v1+tcn_full_heading_impff_s1bnd0.5     hmm    0.623    0.923     0.887             0.921       0.328        0.666 22          0.581            0.734        0.553    -0.001 0.003  6/22
                who_v1+ens_impffbnd0.5     hmm    0.623    0.923     0.887             0.923       0.329        0.667 22          0.580            0.736        0.552    -0.002 0.003  8/22
        who_v1+tcn_full_impff_s1bnd0.5     hmm    0.623    0.923     0.887             0.925       0.330        0.668 22          0.581            0.735        0.552    -0.002 0.002  9/22
        who_v1+tcn_full_impff_s0bnd0.5     hmm    0.622    0.923     0.887             0.926       0.330        0.666 22          0.577            0.737        0.552    -0.003 0.003  7/22
who_v1+tcn_full_heading_impff_s2bnd0.5     hmm    0.621    0.923     0.886             0.920       0.325        0.665 22          0.578            0.733        0.552    -0.003 0.002  8/22
                                who_v1     hmm    0.616    0.922     0.885             0.863       0.280        0.650 22          0.576            0.713        0.559       NaN   NaN   NaN
                              rule_seg     hmm    0.471    0.872     0.813             0.805       0.153        0.451 22          0.418            0.557        0.439    -0.130 0.007  0/22
                                  rule     hmm    0.464    0.869     0.808             0.834       0.157        0.449 22          0.420            0.543        0.430    -0.138 0.006  0/22
                             rule_hard     hmm    0.464    0.869     0.808             0.844       0.161        0.451 22          0.419            0.545        0.427    -0.139 0.006  0/22
                                 blend     hmm    0.418    0.859     0.793             1.259       0.254        0.510 22          0.407            0.480        0.368    -0.191 0.008  0/22
                              smooth50     hmm    0.361    0.846     0.773             1.220       0.136        0.447 22          0.340            0.384        0.357    -0.219 0.012  0/22
                               nearest     hmm    0.356    0.843     0.768             1.273       0.172        0.440 22          0.335            0.406        0.328    -0.240 0.010  0/22
                                  hold     hmm    0.279    0.839     0.763             2.345       0.357        0.492 22          0.295            0.293        0.250    -0.314 0.013  0/22
```
