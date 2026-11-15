# owner experiment `headchk_tcn_bar`

```
{
 "tag": "headchk_tcn_bar",
 "condition": "bar",
 "seeds": [
  0
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
 "noise": [],
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
 "minutes": 10.4
}
```

## Per alive frame (mean over games per source, then over sources)

```
                           acc_all  acc_seen  acc_missing  auc_all  logloss_all  brier_all  ece_all
method                                                                                             
ens_impff_hd                 0.930     0.943        0.759    0.976        0.191      0.053    0.026
tcn_full_heading_impff_s0    0.929     0.942        0.787    0.975        0.199      0.055    0.030
tcn_full_heading_impff_s1    0.928     0.941        0.744    0.974        0.200      0.055    0.029
tcn_full_heading_impff_s2    0.927     0.941        0.758    0.974        0.203      0.056    0.030
ens_impff                    0.926     0.939        0.761    0.975        0.197      0.055    0.027
tcn_full_impff_s1            0.925     0.938        0.753    0.973        0.206      0.057    0.030
tcn_full_impff_s0            0.924     0.938        0.756    0.973        0.208      0.058    0.030
tcn_full_impff_s2            0.923     0.936        0.774    0.973        0.209      0.059    0.031
who_v1                       0.904     0.919        0.707    0.965        0.236      0.070    0.024
rule_hard                    0.841     0.863        0.637    0.837        0.638      0.153    0.139
rule                         0.841     0.863        0.637    0.889        0.403      0.123    0.034
hold                         0.798     0.822        0.649    0.796        0.805      0.202    0.202
blend                        0.797     0.818        0.638    0.842        0.504      0.160    0.085
smooth50                     0.758     0.780        0.641    0.765        0.918      0.225    0.208
nearest                      0.704     0.723        0.519    0.785        0.619      0.197    0.097
```

## Layer `dead_v1` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
                                method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
                   ens_impff_hd+bnd0.5     hmm    0.619    0.874     0.934             0.825       0.255        0.655 11          0.594            0.716        0.548     0.054 0.011 11/11
      tcn_full_heading_impff_s0+bnd0.5     hmm    0.615    0.873     0.932             0.835       0.258        0.652 11          0.599            0.705        0.541     0.049 0.010 11/11
      tcn_full_heading_impff_s2+bnd0.5     hmm    0.615    0.873     0.932             0.837       0.255        0.652 11          0.601            0.701        0.543     0.051 0.010 11/11
      tcn_full_heading_impff_s1+bnd0.5     hmm    0.614    0.872     0.932             0.852       0.262        0.659 11          0.605            0.688        0.548     0.054 0.009 11/11
              tcn_full_impff_s1+bnd0.5     hmm    0.609    0.872     0.930             0.839       0.239        0.658 11          0.598            0.686        0.543     0.049 0.008 11/11
                      ens_impff+bnd0.5     hmm    0.608    0.872     0.931             0.818       0.233        0.644 11          0.598            0.684        0.542     0.048 0.008 11/11
             tcn_full_heading_impff_s2     hmm    0.608    0.873     0.932             0.772       0.201        0.638 11          0.593            0.696        0.534     0.042 0.011 11/11
             tcn_full_heading_impff_s0     hmm    0.607    0.873     0.932             0.782       0.208        0.637 11          0.598            0.697        0.528     0.038 0.009 11/11
                          ens_impff_hd     hmm    0.607    0.873     0.933             0.761       0.199        0.625 11          0.592            0.689        0.539     0.045 0.010 11/11
             tcn_full_heading_impff_s1     hmm    0.605    0.872     0.931             0.793       0.215        0.636 11          0.591            0.683        0.541     0.046 0.009 11/11
                     tcn_full_impff_s1     hmm    0.605    0.872     0.930             0.786       0.195        0.646 11          0.594            0.682        0.538     0.044 0.008 11/11
              tcn_full_impff_s0+bnd0.5     hmm    0.601    0.870     0.927             0.829       0.239        0.630 11          0.600            0.673        0.532     0.039 0.008 11/11
              tcn_full_impff_s2+bnd0.5     hmm    0.599    0.870     0.928             0.829       0.225        0.635 11          0.593            0.661        0.542     0.045 0.008 11/11
                             ens_impff     hmm    0.599    0.871     0.930             0.761       0.182        0.624 11          0.587            0.682        0.528     0.035 0.010 10/11
                     tcn_full_impff_s2     hmm    0.594    0.869     0.926             0.777       0.186        0.622 11          0.581            0.667        0.534     0.037 0.008 10/11
                     tcn_full_impff_s0     hmm    0.590    0.869     0.926             0.770       0.184        0.611 11          0.581            0.666        0.522     0.028 0.009  8/11
who_v1+tcn_full_heading_impff_s1bnd0.5     hmm    0.566    0.862     0.915             0.832       0.222        0.576 11          0.569            0.628        0.502     0.008 0.003  9/11
                who_v1+ens_impffbnd0.5     hmm    0.565    0.862     0.914             0.829       0.219        0.571 11          0.567            0.622        0.505     0.010 0.003 11/11
        who_v1+tcn_full_impff_s2bnd0.5     hmm    0.565    0.862     0.915             0.831       0.220        0.572 11          0.567            0.622        0.505     0.010 0.003 11/11
        who_v1+tcn_full_impff_s1bnd0.5     hmm    0.564    0.862     0.914             0.834       0.220        0.573 11          0.569            0.618        0.506     0.010 0.002 10/11
             who_v1+ens_impff_hdbnd0.5     hmm    0.564    0.862     0.914             0.830       0.219        0.572 11          0.564            0.624        0.505     0.009 0.003 10/11
who_v1+tcn_full_heading_impff_s2bnd0.5     hmm    0.563    0.862     0.914             0.832       0.220        0.570 11          0.566            0.621        0.502     0.007 0.003  8/11
who_v1+tcn_full_heading_impff_s0bnd0.5     hmm    0.563    0.862     0.914             0.832       0.219        0.572 11          0.565            0.620        0.504     0.008 0.003 10/11
        who_v1+tcn_full_impff_s0bnd0.5     hmm    0.563    0.862     0.915             0.832       0.218        0.571 11          0.564            0.621        0.503     0.007 0.003 11/11
                                who_v1     hmm    0.550    0.860     0.912             0.765       0.168        0.537 11          0.560            0.591        0.498       NaN   NaN   NaN
                              rule_seg     hmm    0.436    0.820     0.849             0.704       0.076        0.378 11          0.450            0.470        0.388    -0.111 0.009  0/11
                                  rule     hmm    0.431    0.818     0.847             0.707       0.075        0.373 11          0.448            0.465        0.381    -0.117 0.008  0/11
                             rule_hard     hmm    0.431    0.818     0.847             0.718       0.076        0.377 11          0.447            0.467        0.378    -0.119 0.008  0/11
                                 blend     hmm    0.398    0.807     0.830             1.097       0.166        0.439 11          0.444            0.392        0.359    -0.140 0.010  0/11
                               nearest     hmm    0.305    0.779     0.786             1.197       0.083        0.360 11          0.336            0.277        0.301    -0.212 0.015  0/11
                              smooth50     hmm    0.295    0.767     0.765             1.174       0.063        0.348 11          0.345            0.249        0.291    -0.221 0.016  0/11
                                  hold     hmm    0.293    0.789     0.803             1.855       0.234        0.443 11          0.322            0.293        0.263    -0.241 0.016  0/11
```

## Layer `oracle` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
                                method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
      tcn_full_heading_impff_s0+bnd0.5     hmm    0.750    0.952     0.930             0.846       0.430        0.784 11          0.725            0.851        0.674     0.066 0.009 11/11
                   ens_impff_hd+bnd0.5     hmm    0.750    0.953     0.931             0.831       0.415        0.782 11          0.718            0.857        0.675     0.066 0.010 11/11
      tcn_full_heading_impff_s2+bnd0.5     hmm    0.742    0.951     0.928             0.849       0.418        0.775 11          0.721            0.836        0.668     0.060 0.009 11/11
      tcn_full_heading_impff_s1+bnd0.5     hmm    0.740    0.951     0.928             0.861       0.418        0.788 11          0.720            0.829        0.670     0.060 0.007 11/11
             tcn_full_heading_impff_s0     hmm    0.736    0.951     0.929             0.799       0.361        0.770 11          0.710            0.828        0.669     0.058 0.009 11/11
              tcn_full_impff_s1+bnd0.5     hmm    0.736    0.950     0.927             0.854       0.406        0.779 11          0.714            0.824        0.668     0.057 0.008 11/11
                      ens_impff+bnd0.5     hmm    0.735    0.950     0.927             0.824       0.386        0.763 11          0.721            0.815        0.670     0.059 0.006 11/11
                          ens_impff_hd     hmm    0.735    0.952     0.930             0.779       0.348        0.759 11          0.708            0.823        0.674     0.060 0.008 11/11
             tcn_full_heading_impff_s2     hmm    0.734    0.951     0.929             0.792       0.356        0.759 11          0.711            0.824        0.667     0.056 0.009 11/11
             tcn_full_heading_impff_s1     hmm    0.732    0.951     0.928             0.813       0.366        0.771 11          0.709            0.814        0.672     0.058 0.008 11/11
              tcn_full_impff_s0+bnd0.5     hmm    0.729    0.949     0.925             0.838       0.396        0.760 11          0.722            0.808        0.658     0.050 0.007 11/11
                             ens_impff     hmm    0.727    0.950     0.927             0.776       0.333        0.747 11          0.704            0.806        0.670     0.055 0.007 11/11
                     tcn_full_impff_s1     hmm    0.726    0.949     0.926             0.802       0.341        0.765 11          0.705            0.805        0.667     0.053 0.008 11/11
              tcn_full_impff_s2+bnd0.5     hmm    0.724    0.949     0.925             0.837       0.375        0.757 11          0.715            0.789        0.669     0.055 0.006 11/11
                     tcn_full_impff_s2     hmm    0.722    0.948     0.924             0.798       0.340        0.751 11          0.703            0.794        0.669     0.054 0.007 11/11
                     tcn_full_impff_s0     hmm    0.719    0.948     0.924             0.796       0.339        0.748 11          0.699            0.803        0.656     0.044 0.008 11/11
who_v1+tcn_full_heading_impff_s1bnd0.5     hmm    0.684    0.940     0.911             0.860       0.371        0.704 11          0.689            0.756        0.606     0.001 0.005  5/11
who_v1+tcn_full_heading_impff_s2bnd0.5     hmm    0.683    0.939     0.911             0.861       0.370        0.704 11          0.687            0.750        0.613     0.005 0.003  5/11
who_v1+tcn_full_heading_impff_s0bnd0.5     hmm    0.683    0.939     0.911             0.859       0.367        0.705 11          0.687            0.750        0.610     0.004 0.004  5/11
             who_v1+ens_impff_hdbnd0.5     hmm    0.682    0.939     0.911             0.859       0.368        0.703 11          0.686            0.751        0.610     0.004 0.004  5/11
        who_v1+tcn_full_impff_s1bnd0.5     hmm    0.682    0.939     0.911             0.863       0.368        0.704 11          0.691            0.749        0.607     0.002 0.005  4/11
                who_v1+ens_impffbnd0.5     hmm    0.682    0.939     0.911             0.860       0.368        0.701 11          0.688            0.750        0.608     0.002 0.004  4/11
        who_v1+tcn_full_impff_s2bnd0.5     hmm    0.682    0.939     0.911             0.863       0.370        0.702 11          0.687            0.750        0.608     0.002 0.004  4/11
        who_v1+tcn_full_impff_s0bnd0.5     hmm    0.680    0.939     0.911             0.858       0.365        0.701 11          0.682            0.749        0.609     0.002 0.004  4/11
                                who_v1     hmm    0.670    0.938     0.910             0.789       0.301        0.675 11          0.676            0.723        0.612       NaN   NaN   NaN
                              rule_seg     hmm    0.535    0.897     0.849             0.697       0.155        0.482 11          0.536            0.589        0.481    -0.133 0.010  0/11
                                  rule     hmm    0.516    0.892     0.841             0.728       0.152        0.469 11          0.526            0.557        0.466    -0.148 0.009  0/11
                             rule_hard     hmm    0.514    0.891     0.841             0.735       0.152        0.470 11          0.525            0.555        0.463    -0.151 0.009  0/11
                                 blend     hmm    0.481    0.880     0.824             1.187       0.304        0.554 11          0.534            0.491        0.417    -0.189 0.014  0/11
                               nearest     hmm    0.361    0.848     0.776             1.303       0.169        0.455 11          0.395            0.335        0.352    -0.275 0.018  0/11
                              smooth50     hmm    0.359    0.838     0.761             1.218       0.139        0.444 11          0.409            0.315        0.354    -0.273 0.019  0/11
                                  hold     hmm    0.349    0.863     0.799             2.008       0.384        0.543 11          0.381            0.365        0.300    -0.313 0.018  0/11
```
