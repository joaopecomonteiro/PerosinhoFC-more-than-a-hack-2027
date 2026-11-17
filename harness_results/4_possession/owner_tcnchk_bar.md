# owner experiment `tcnchk_bar`

```
{
 "tag": "tcnchk_bar",
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
 "minutes": 10.2
}
```

## Per alive frame (mean over games per source, then over sources)

```
                   acc_all  acc_seen  acc_missing  auc_all  logloss_all  brier_all  ece_all
method                                                                                     
ens_impff            0.926     0.939        0.761    0.975        0.197      0.055    0.027
tcn_full_impff_s1    0.925     0.938        0.753    0.973        0.206      0.057    0.030
tcn_full_impff_s0    0.924     0.938        0.756    0.973        0.208      0.058    0.030
tcn_full_impff_s2    0.923     0.936        0.774    0.973        0.209      0.059    0.031
ens_im               0.912     0.927        0.731    0.965        0.248      0.069    0.045
tcn_full_im_s1       0.911     0.926        0.721    0.960        0.275      0.074    0.058
tcn_full_im_s2       0.910     0.925        0.741    0.961        0.276      0.074    0.057
tcn_full_im_s0       0.906     0.922        0.719    0.959        0.283      0.077    0.058
who_v1               0.904     0.919        0.707    0.965        0.236      0.070    0.024
rule_hard            0.841     0.863        0.637    0.837        0.638      0.153    0.139
rule                 0.841     0.863        0.637    0.889        0.403      0.123    0.034
hold                 0.798     0.822        0.649    0.796        0.805      0.202    0.202
smooth50             0.758     0.780        0.641    0.765        0.918      0.225    0.208
nearest              0.704     0.723        0.519    0.785        0.619      0.197    0.097
```

## Layer `dead_v1` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
                        method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
      tcn_full_impff_s1+bnd0.5     hmm    0.609    0.872     0.930             0.839       0.239        0.658 11          0.598            0.686        0.543     0.049 0.008 11/11
              ens_impff+bnd0.5     hmm    0.608    0.872     0.931             0.818       0.233        0.644 11          0.598            0.684        0.542     0.048 0.008 11/11
             tcn_full_impff_s1     hmm    0.605    0.872     0.930             0.786       0.195        0.646 11          0.594            0.682        0.538     0.044 0.008 11/11
      tcn_full_impff_s0+bnd0.5     hmm    0.601    0.870     0.927             0.829       0.239        0.630 11          0.600            0.673        0.532     0.039 0.008 11/11
      tcn_full_impff_s2+bnd0.5     hmm    0.599    0.870     0.928             0.829       0.225        0.635 11          0.593            0.661        0.542     0.045 0.008 11/11
                     ens_impff     hmm    0.599    0.871     0.930             0.761       0.182        0.624 11          0.587            0.682        0.528     0.035 0.010 10/11
             tcn_full_impff_s2     hmm    0.594    0.869     0.926             0.777       0.186        0.622 11          0.581            0.667        0.534     0.037 0.008 10/11
             tcn_full_impff_s0     hmm    0.590    0.869     0.926             0.770       0.184        0.611 11          0.581            0.666        0.522     0.028 0.009  8/11
         tcn_full_im_s1+bnd0.5     hmm    0.579    0.864     0.917             0.864       0.215        0.620 11          0.580            0.647        0.508     0.016 0.006  8/11
                 ens_im+bnd0.5     hmm    0.578    0.864     0.918             0.825       0.200        0.602 11          0.580            0.645        0.509     0.017 0.006  9/11
         tcn_full_im_s2+bnd0.5     hmm    0.575    0.863     0.915             0.851       0.209        0.609 11          0.583            0.640        0.503     0.012 0.006  8/11
                tcn_full_im_s1     hmm    0.575    0.863     0.917             0.828       0.191        0.606 11          0.575            0.640        0.510     0.016 0.006  9/11
                tcn_full_im_s2     hmm    0.574    0.863     0.915             0.822       0.190        0.602 11          0.580            0.639        0.503     0.012 0.007  8/11
                        ens_im     hmm    0.573    0.863     0.917             0.799       0.183        0.591 11          0.575            0.639        0.506     0.013 0.006  8/11
        who_v1+ens_impffbnd0.5     hmm    0.565    0.862     0.914             0.829       0.219        0.571 11          0.567            0.622        0.505     0.010 0.003 11/11
who_v1+tcn_full_impff_s2bnd0.5     hmm    0.565    0.862     0.915             0.831       0.220        0.572 11          0.567            0.622        0.505     0.010 0.003 11/11
who_v1+tcn_full_impff_s1bnd0.5     hmm    0.564    0.862     0.914             0.834       0.220        0.573 11          0.569            0.618        0.506     0.010 0.002 10/11
who_v1+tcn_full_impff_s0bnd0.5     hmm    0.563    0.862     0.915             0.832       0.218        0.571 11          0.564            0.621        0.503     0.007 0.003 11/11
                tcn_full_im_s0     hmm    0.562    0.861     0.912             0.834       0.184        0.590 11          0.557            0.635        0.495     0.002 0.008  5/11
   who_v1+tcn_full_im_s1bnd0.5     hmm    0.562    0.862     0.914             0.820       0.212        0.565 11          0.565            0.618        0.501     0.006 0.003  7/11
           who_v1+ens_imbnd0.5     hmm    0.561    0.862     0.914             0.819       0.210        0.564 11          0.567            0.615        0.503     0.007 0.002 10/11
   who_v1+tcn_full_im_s2bnd0.5     hmm    0.561    0.861     0.914             0.819       0.210        0.563 11          0.565            0.615        0.501     0.006 0.002  9/11
   who_v1+tcn_full_im_s0bnd0.5     hmm    0.560    0.861     0.914             0.823       0.211        0.565 11          0.565            0.614        0.502     0.006 0.002 10/11
         tcn_full_im_s0+bnd0.5     hmm    0.558    0.861     0.912             0.866       0.194        0.596 11          0.554            0.625        0.495    -0.000 0.007  5/11
                        who_v1     hmm    0.550    0.860     0.912             0.765       0.168        0.537 11          0.560            0.591        0.498       NaN   NaN   NaN
                      rule_seg     hmm    0.436    0.820     0.849             0.704       0.076        0.378 11          0.450            0.470        0.388    -0.111 0.009  0/11
                          rule     hmm    0.431    0.818     0.847             0.707       0.075        0.373 11          0.448            0.465        0.381    -0.117 0.008  0/11
                     rule_hard     hmm    0.431    0.818     0.847             0.718       0.076        0.377 11          0.447            0.467        0.378    -0.119 0.008  0/11
                       nearest     hmm    0.305    0.779     0.786             1.197       0.083        0.360 11          0.336            0.277        0.301    -0.212 0.015  0/11
                      smooth50     hmm    0.295    0.767     0.765             1.174       0.063        0.348 11          0.345            0.249        0.291    -0.221 0.016  0/11
                          hold     hmm    0.293    0.789     0.803             1.855       0.234        0.443 11          0.322            0.293        0.263    -0.241 0.016  0/11
```

## Layer `oracle` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
                        method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se  wins
      tcn_full_impff_s1+bnd0.5     hmm    0.736    0.950     0.927             0.854       0.406        0.779 11          0.714            0.824        0.668     0.057 0.008 11/11
              ens_impff+bnd0.5     hmm    0.735    0.950     0.927             0.824       0.386        0.763 11          0.721            0.815        0.670     0.059 0.006 11/11
      tcn_full_impff_s0+bnd0.5     hmm    0.729    0.949     0.925             0.838       0.396        0.760 11          0.722            0.808        0.658     0.050 0.007 11/11
                     ens_impff     hmm    0.727    0.950     0.927             0.776       0.333        0.747 11          0.704            0.806        0.670     0.055 0.007 11/11
             tcn_full_impff_s1     hmm    0.726    0.949     0.926             0.802       0.341        0.765 11          0.705            0.805        0.667     0.053 0.008 11/11
      tcn_full_impff_s2+bnd0.5     hmm    0.724    0.949     0.925             0.837       0.375        0.757 11          0.715            0.789        0.669     0.055 0.006 11/11
             tcn_full_impff_s2     hmm    0.722    0.948     0.924             0.798       0.340        0.751 11          0.703            0.794        0.669     0.054 0.007 11/11
             tcn_full_impff_s0     hmm    0.719    0.948     0.924             0.796       0.339        0.748 11          0.699            0.803        0.656     0.044 0.008 11/11
                 ens_im+bnd0.5     hmm    0.699    0.942     0.915             0.839       0.353        0.726 11          0.694            0.783        0.621     0.016 0.007  8/11
         tcn_full_im_s1+bnd0.5     hmm    0.697    0.941     0.914             0.882       0.375        0.739 11          0.702            0.776        0.614     0.011 0.006  7/11
                tcn_full_im_s1     hmm    0.697    0.940     0.913             0.860       0.361        0.730 11          0.696            0.775        0.619     0.014 0.006  8/11
                        ens_im     hmm    0.694    0.941     0.913             0.820       0.330        0.719 11          0.686            0.773        0.623     0.015 0.005  9/11
                tcn_full_im_s2     hmm    0.691    0.939     0.911             0.843       0.343        0.724 11          0.689            0.769        0.616     0.010 0.006  8/11
         tcn_full_im_s2+bnd0.5     hmm    0.691    0.940     0.912             0.867       0.360        0.728 11          0.694            0.768        0.612     0.007 0.006  7/11
who_v1+tcn_full_impff_s1bnd0.5     hmm    0.682    0.939     0.911             0.863       0.368        0.704 11          0.691            0.749        0.607     0.002 0.005  4/11
        who_v1+ens_impffbnd0.5     hmm    0.682    0.939     0.911             0.860       0.368        0.701 11          0.688            0.750        0.608     0.002 0.004  4/11
who_v1+tcn_full_impff_s2bnd0.5     hmm    0.682    0.939     0.911             0.863       0.370        0.702 11          0.687            0.750        0.608     0.002 0.004  4/11
   who_v1+tcn_full_im_s1bnd0.5     hmm    0.681    0.939     0.911             0.846       0.360        0.694 11          0.688            0.748        0.607     0.001 0.004  5/11
           who_v1+ens_imbnd0.5     hmm    0.680    0.939     0.911             0.848       0.359        0.694 11          0.688            0.745        0.608     0.002 0.004  4/11
who_v1+tcn_full_impff_s0bnd0.5     hmm    0.680    0.939     0.911             0.858       0.365        0.701 11          0.682            0.749        0.609     0.002 0.004  4/11
   who_v1+tcn_full_im_s2bnd0.5     hmm    0.679    0.939     0.910             0.846       0.357        0.691 11          0.686            0.745        0.606    -0.000 0.004  4/11
   who_v1+tcn_full_im_s0bnd0.5     hmm    0.677    0.939     0.910             0.849       0.353        0.693 11          0.684            0.740        0.606    -0.001 0.004  4/11
                tcn_full_im_s0     hmm    0.676    0.937     0.907             0.849       0.325        0.713 11          0.659            0.769        0.600    -0.007 0.009  5/11
         tcn_full_im_s0+bnd0.5     hmm    0.673    0.937     0.908             0.877       0.342        0.716 11          0.661            0.760        0.599    -0.008 0.008  6/11
                        who_v1     hmm    0.670    0.938     0.910             0.789       0.301        0.675 11          0.676            0.723        0.612       NaN   NaN   NaN
                      rule_seg     hmm    0.535    0.897     0.849             0.697       0.155        0.482 11          0.536            0.589        0.481    -0.133 0.010  0/11
                          rule     hmm    0.516    0.892     0.841             0.728       0.152        0.469 11          0.526            0.557        0.466    -0.148 0.009  0/11
                     rule_hard     hmm    0.514    0.891     0.841             0.735       0.152        0.470 11          0.525            0.555        0.463    -0.151 0.009  0/11
                       nearest     hmm    0.361    0.848     0.776             1.303       0.169        0.455 11          0.395            0.335        0.352    -0.275 0.018  0/11
                      smooth50     hmm    0.359    0.838     0.761             1.218       0.139        0.444 11          0.409            0.315        0.354    -0.273 0.019  0/11
                          hold     hmm    0.349    0.863     0.799             2.008       0.384        0.543 11          0.381            0.365        0.300    -0.313 0.018  0/11
```
