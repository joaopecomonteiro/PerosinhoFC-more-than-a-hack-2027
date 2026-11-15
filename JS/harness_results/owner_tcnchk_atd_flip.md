# owner experiment `tcnchk_atd_flip`

```
{
 "tag": "tcnchk_atd_flip",
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
  "tcn_full_im_s0": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_im_s0.pt",
  "tcn_full_im_s1": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_im_s1.pt",
  "tcn_full_im_s2": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_im_s2.pt",
  "tcn_full_impff_s0": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s0.pt",
  "tcn_full_impff_s1": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s1.pt",
  "tcn_full_impff_s2": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_full_impff_s2.pt"
 },
 "hybrid_base": [],
 "ref": "who_v1",
 "external": [],
 "boundary_from": [
  "tcn_full_im_s0",
  "tcn_full_im_s1",
  "tcn_full_im_s2",
  "tcn_full_impff_s0",
  "tcn_full_impff_s1",
  "tcn_full_impff_s2"
 ],
 "bnd_alpha": [
  0.5
 ],
 "ensemble": [
  "ens_im=tcn_full_im_s0,tcn_full_im_s1,tcn_full_im_s2",
  "ens_impff=tcn_full_impff_s0,tcn_full_impff_s1,tcn_full_impff_s2"
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
 "minutes": 24.8
}
```

## Per alive frame (mean over games per source, then over sources)

```
                   acc_all  acc_seen  acc_missing  auc_all  logloss_all  brier_all  ece_all
method                                                                                     
ens_impff            0.910     0.934        0.862    0.967        0.230      0.067    0.031
tcn_full_impff_s1    0.908     0.933        0.859    0.965        0.243      0.069    0.039
tcn_full_impff_s0    0.906     0.931        0.857    0.964        0.247      0.071    0.039
tcn_full_impff_s2    0.906     0.930        0.856    0.964        0.249      0.071    0.040
ens_im               0.889     0.916        0.834    0.953        0.300      0.086    0.056
tcn_full_im_s2       0.888     0.915        0.833    0.946        0.340      0.093    0.073
tcn_full_im_s1       0.886     0.913        0.832    0.945        0.344      0.094    0.074
tcn_full_im_s0       0.881     0.909        0.826    0.944        0.353      0.097    0.075
who_v1               0.876     0.905        0.816    0.948        0.291      0.089    0.033
rule_hard            0.808     0.845        0.729    0.804        0.768      0.185    0.172
rule                 0.808     0.845        0.729    0.846        0.481      0.151    0.058
smooth50             0.767     0.803        0.690    0.779        0.857      0.211    0.190
hold                 0.761     0.802        0.674    0.758        0.950      0.239    0.239
nearest              0.703     0.735        0.640    0.770        0.646      0.205    0.094
```

## Layer `dead_v1` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
                        method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
      tcn_full_impff_s1+bnd0.5     hmm    0.557    0.846     0.913             0.838       0.190        0.583 22          0.532            0.624        0.517     0.058 0.005 22/22
              ens_impff+bnd0.5     hmm    0.556    0.846     0.914             0.813       0.175        0.580 22          0.530            0.625        0.512     0.054 0.005 22/22
      tcn_full_impff_s0+bnd0.5     hmm    0.552    0.844     0.911             0.830       0.185        0.573 22          0.522            0.625        0.508     0.050 0.005 22/22
      tcn_full_impff_s2+bnd0.5     hmm    0.551    0.844     0.910             0.838       0.177        0.581 22          0.522            0.622        0.508     0.049 0.005 22/22
             tcn_full_impff_s1     hmm    0.550    0.846     0.913             0.781       0.157        0.561 22          0.518            0.625        0.508     0.049 0.005 22/22
                     ens_impff     hmm    0.546    0.846     0.913             0.761       0.147        0.550 22          0.512            0.620        0.505     0.045 0.006 21/22
             tcn_full_impff_s0     hmm    0.546    0.844     0.911             0.777       0.158        0.550 22          0.511            0.623        0.503     0.044 0.006 21/22
             tcn_full_impff_s2     hmm    0.545    0.844     0.910             0.768       0.146        0.549 22          0.507            0.624        0.503     0.043 0.006 21/22
                 ens_im+bnd0.5     hmm    0.522    0.836     0.897             0.899       0.175        0.567 22          0.490            0.606        0.471     0.015 0.003 19/22
                        ens_im     hmm    0.521    0.836     0.897             0.858       0.151        0.556 22          0.492            0.599        0.472     0.016 0.003 18/22
        who_v1+ens_impffbnd0.5     hmm    0.517    0.832     0.891             0.882       0.185        0.539 22          0.486            0.607        0.459     0.006 0.002 14/22
who_v1+tcn_full_impff_s1bnd0.5     hmm    0.517    0.832     0.891             0.888       0.186        0.541 22          0.485            0.605        0.461     0.007 0.002 16/22
who_v1+tcn_full_impff_s2bnd0.5     hmm    0.517    0.832     0.891             0.885       0.184        0.542 22          0.486            0.605        0.459     0.006 0.002 14/22
who_v1+tcn_full_impff_s0bnd0.5     hmm    0.516    0.832     0.891             0.887       0.188        0.536 22          0.485            0.605        0.459     0.005 0.002 14/22
           who_v1+ens_imbnd0.5     hmm    0.516    0.832     0.890             0.869       0.177        0.535 22          0.483            0.606        0.458     0.005 0.002 15/22
   who_v1+tcn_full_im_s2bnd0.5     hmm    0.516    0.832     0.890             0.873       0.177        0.538 22          0.484            0.607        0.456     0.003 0.002 15/22
   who_v1+tcn_full_im_s1bnd0.5     hmm    0.515    0.832     0.890             0.869       0.178        0.532 22          0.484            0.605        0.457     0.004 0.002 14/22
   who_v1+tcn_full_im_s0bnd0.5     hmm    0.515    0.832     0.890             0.871       0.179        0.534 22          0.481            0.606        0.457     0.004 0.002 14/22
         tcn_full_im_s1+bnd0.5     hmm    0.515    0.834     0.894             0.953       0.189        0.572 22          0.482            0.595        0.467     0.010 0.003 17/22
                tcn_full_im_s2     hmm    0.514    0.835     0.895             0.892       0.156        0.558 22          0.488            0.588        0.466     0.009 0.004 15/22
         tcn_full_im_s2+bnd0.5     hmm    0.513    0.834     0.894             0.937       0.183        0.562 22          0.492            0.584        0.463     0.008 0.004 14/22
                tcn_full_im_s1     hmm    0.512    0.834     0.894             0.908       0.161        0.558 22          0.480            0.586        0.470     0.011 0.003 18/22
         tcn_full_im_s0+bnd0.5     hmm    0.510    0.832     0.890             0.950       0.183        0.569 22          0.482            0.595        0.453    -0.001 0.004 12/22
                tcn_full_im_s0     hmm    0.509    0.832     0.890             0.908       0.165        0.555 22          0.475            0.598        0.456     0.000 0.004 13/22
                        who_v1     hmm    0.506    0.831     0.889             0.814       0.140        0.510 22          0.471            0.589        0.457       NaN   NaN   NaN
                      rule_seg     hmm    0.380    0.785     0.814             0.817       0.075        0.347 22          0.339            0.451        0.349    -0.115 0.006  0/22
                          rule     hmm    0.376    0.782     0.812             0.816       0.073        0.340 22          0.336            0.449        0.343    -0.120 0.006  0/22
                     rule_hard     hmm    0.376    0.782     0.812             0.830       0.076        0.343 22          0.336            0.449        0.342    -0.121 0.006  0/22
                      smooth50     hmm    0.308    0.761     0.777             1.135       0.068        0.349 22          0.302            0.326        0.296    -0.172 0.010  0/22
                       nearest     hmm    0.299    0.767     0.786             1.132       0.076        0.326 22          0.283            0.331        0.282    -0.185 0.009  0/22
                          hold     hmm    0.242    0.754     0.768             2.130       0.206        0.415 22          0.265            0.242        0.220    -0.242 0.012  0/22
```

## Layer `oracle` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
                        method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
              ens_impff+bnd0.5     hmm    0.680    0.939     0.911             0.810       0.315        0.706 22          0.643            0.759        0.640     0.075 0.006 22/22
      tcn_full_impff_s1+bnd0.5     hmm    0.677    0.938     0.909             0.839       0.333        0.706 22          0.635            0.757        0.637     0.072 0.006 22/22
             tcn_full_impff_s1     hmm    0.676    0.939     0.910             0.793       0.301        0.694 22          0.636            0.758        0.635     0.070 0.006 22/22
                     ens_impff     hmm    0.675    0.939     0.911             0.775       0.288        0.690 22          0.636            0.749        0.640     0.073 0.006 22/22
      tcn_full_impff_s2+bnd0.5     hmm    0.672    0.937     0.907             0.842       0.317        0.712 22          0.632            0.751        0.633     0.068 0.007 22/22
      tcn_full_impff_s0+bnd0.5     hmm    0.672    0.937     0.908             0.835       0.324        0.701 22          0.635            0.753        0.628     0.065 0.005 22/22
             tcn_full_impff_s0     hmm    0.671    0.937     0.908             0.793       0.298        0.691 22          0.631            0.753        0.629     0.065 0.006 22/22
             tcn_full_impff_s2     hmm    0.670    0.937     0.907             0.794       0.291        0.689 22          0.626            0.748        0.636     0.069 0.007 22/22
                        ens_im     hmm    0.635    0.926     0.892             0.866       0.297        0.680 22          0.597            0.731        0.579     0.020 0.003 21/22
                 ens_im+bnd0.5     hmm    0.635    0.926     0.892             0.897       0.319        0.683 22          0.598            0.732        0.574     0.017 0.003 20/22
                tcn_full_im_s1     hmm    0.626    0.925     0.890             0.907       0.303        0.683 22          0.595            0.712        0.570     0.011 0.003 19/22
who_v1+tcn_full_impff_s2bnd0.5     hmm    0.624    0.923     0.887             0.925       0.332        0.669 22          0.581            0.739        0.553    -0.001 0.003  9/22
         tcn_full_im_s1+bnd0.5     hmm    0.624    0.924     0.889             0.941       0.324        0.687 22          0.593            0.713        0.566     0.008 0.003 17/22
   who_v1+tcn_full_im_s1bnd0.5     hmm    0.624    0.923     0.887             0.908       0.321        0.664 22          0.582            0.735        0.554    -0.001 0.002 10/22
           who_v1+ens_imbnd0.5     hmm    0.624    0.923     0.887             0.905       0.318        0.664 22          0.584            0.732        0.554    -0.000 0.002 10/22
   who_v1+tcn_full_im_s2bnd0.5     hmm    0.623    0.923     0.887             0.910       0.319        0.667 22          0.583            0.733        0.554    -0.001 0.002 11/22
   who_v1+tcn_full_im_s0bnd0.5     hmm    0.623    0.923     0.886             0.908       0.323        0.662 22          0.580            0.735        0.554    -0.001 0.002  8/22
        who_v1+ens_impffbnd0.5     hmm    0.623    0.923     0.887             0.923       0.329        0.667 22          0.580            0.736        0.552    -0.002 0.003  8/22
who_v1+tcn_full_impff_s1bnd0.5     hmm    0.623    0.923     0.887             0.925       0.330        0.668 22          0.581            0.735        0.552    -0.002 0.002  9/22
who_v1+tcn_full_impff_s0bnd0.5     hmm    0.622    0.923     0.887             0.926       0.330        0.666 22          0.577            0.737        0.552    -0.003 0.003  7/22
                tcn_full_im_s2     hmm    0.622    0.925     0.890             0.894       0.299        0.669 22          0.593            0.710        0.563     0.006 0.003 14/22
                tcn_full_im_s0     hmm    0.620    0.922     0.885             0.917       0.311        0.677 22          0.583            0.728        0.550    -0.004 0.004 10/22
         tcn_full_im_s2+bnd0.5     hmm    0.619    0.925     0.890             0.923       0.315        0.670 22          0.593            0.702        0.562     0.004 0.003 12/22
         tcn_full_im_s0+bnd0.5     hmm    0.619    0.922     0.886             0.949       0.329        0.683 22          0.584            0.726        0.548    -0.006 0.004 11/22
                        who_v1     hmm    0.616    0.922     0.885             0.863       0.280        0.650 22          0.576            0.713        0.559       NaN   NaN   NaN
                      rule_seg     hmm    0.471    0.872     0.813             0.805       0.153        0.451 22          0.418            0.557        0.439    -0.130 0.007  0/22
                          rule     hmm    0.464    0.869     0.808             0.834       0.157        0.449 22          0.420            0.543        0.430    -0.138 0.006  0/22
                     rule_hard     hmm    0.464    0.869     0.808             0.844       0.161        0.451 22          0.419            0.545        0.427    -0.139 0.006  0/22
                      smooth50     hmm    0.361    0.846     0.773             1.220       0.136        0.447 22          0.340            0.384        0.357    -0.219 0.012  0/22
                       nearest     hmm    0.356    0.843     0.768             1.273       0.172        0.440 22          0.335            0.406        0.328    -0.240 0.010  0/22
                          hold     hmm    0.279    0.839     0.763             2.345       0.357        0.492 22          0.295            0.293        0.250    -0.314 0.013  0/22
```
