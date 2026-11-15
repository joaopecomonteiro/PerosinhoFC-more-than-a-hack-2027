# owner experiment `headchk_bar`

```
{
 "tag": "headchk_bar",
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
  "gbm_full_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_full_im.joblib",
  "gbm_stack_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_stack_im.joblib",
  "gbm_full_heading_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_full_heading_im.joblib",
  "gbm_stack_heading_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_stack_heading_im.joblib"
 },
 "hybrid_base": [],
 "ref": "who_v1",
 "external": [],
 "boundary_from": null,
 "bnd_alpha": [
  0.5,
  0.9
 ],
 "ensemble": [],
 "clean_tracks": false,
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
 "minutes": 3.1
}
```

## Per alive frame (mean over games per source, then over sources)

```
                      acc_all  acc_seen  acc_missing  auc_all  logloss_all  brier_all  ece_all
method                                                                                        
gbm_stack_heading_im    0.909     0.924        0.716    0.967        0.227      0.067    0.023
gbm_stack_im            0.908     0.923        0.700    0.967        0.227      0.067    0.023
gbm_full_heading_im     0.908     0.923        0.712    0.966        0.230      0.068    0.025
gbm_full_im             0.908     0.923        0.705    0.966        0.232      0.068    0.023
who_v1                  0.905     0.920        0.704    0.965        0.235      0.070    0.024
rule_hard               0.839     0.863        0.621    0.835        0.647      0.155    0.141
rule                    0.839     0.863        0.621    0.889        0.404      0.124    0.036
blend                   0.798     0.819        0.639    0.844        0.501      0.159    0.084
hold                    0.798     0.822        0.645    0.796        0.807      0.202    0.202
smooth50                0.757     0.779        0.641    0.764        0.924      0.226    0.209
nearest                 0.704     0.723        0.516    0.785        0.617      0.197    0.096
```

## Layer `dead_v1` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
              method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se wins
 gbm_full_heading_im     hmm    0.571    0.862     0.916             0.783       0.199        0.563 11          0.573            0.650        0.489     0.003 0.007 4/11
        gbm_stack_im     hmm    0.570    0.862     0.916             0.754       0.183        0.557 11          0.571            0.650        0.490     0.003 0.008 6/11
gbm_stack_heading_im     hmm    0.568    0.862     0.916             0.773       0.188        0.559 11          0.571            0.645        0.489     0.002 0.007 6/11
         gbm_full_im     hmm    0.565    0.860     0.914             0.767       0.182        0.554 11          0.564            0.642        0.489     0.001 0.006 6/11
              who_v1     hmm    0.552    0.860     0.913             0.771       0.163        0.542 11          0.559            0.602        0.494       NaN   NaN  NaN
            rule_seg     hmm    0.437    0.819     0.849             0.695       0.078        0.377 11          0.455            0.479        0.378    -0.115 0.007 0/11
                rule     hmm    0.432    0.817     0.845             0.696       0.072        0.373 11          0.447            0.474        0.374    -0.120 0.008 0/11
           rule_hard     hmm    0.430    0.817     0.845             0.708       0.074        0.375 11          0.443            0.475        0.372    -0.121 0.008 0/11
               blend     hmm    0.402    0.807     0.831             1.099       0.171        0.445 11          0.445            0.398        0.363    -0.135 0.011 0/11
             nearest     hmm    0.307    0.779     0.786             1.185       0.083        0.360 11          0.338            0.279        0.305    -0.207 0.017 0/11
            smooth50     hmm    0.296    0.766     0.764             1.172       0.062        0.353 11          0.346            0.251        0.291    -0.218 0.017 0/11
                hold     hmm    0.291    0.788     0.802             1.870       0.233        0.442 11          0.321            0.296        0.257    -0.244 0.016 0/11
```

## Layer `oracle` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
              method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se wins
 gbm_full_heading_im     hmm    0.688    0.940     0.912             0.813       0.349        0.688 11          0.688            0.779        0.598    -0.005 0.008 5/11
gbm_stack_heading_im     hmm    0.687    0.941     0.913             0.800       0.332        0.687 11          0.686            0.771        0.605    -0.002 0.007 5/11
        gbm_stack_im     hmm    0.686    0.940     0.913             0.783       0.317        0.686 11          0.681            0.768        0.610     0.001 0.006 5/11
         gbm_full_im     hmm    0.679    0.939     0.910             0.795       0.320        0.680 11          0.675            0.766        0.597    -0.010 0.006 3/11
              who_v1     hmm    0.674    0.939     0.911             0.792       0.304        0.677 11          0.672            0.734        0.615       NaN   NaN  NaN
            rule_seg     hmm    0.535    0.896     0.848             0.685       0.154        0.476 11          0.533            0.600        0.472    -0.142 0.010 0/11
                rule     hmm    0.514    0.890     0.839             0.709       0.145        0.460 11          0.519            0.567        0.457    -0.158 0.010 0/11
           rule_hard     hmm    0.513    0.890     0.839             0.716       0.146        0.460 11          0.518            0.565        0.455    -0.160 0.011 0/11
               blend     hmm    0.482    0.881     0.824             1.184       0.304        0.558 11          0.532            0.492        0.423    -0.187 0.015 0/11
             nearest     hmm    0.362    0.848     0.776             1.303       0.173        0.453 11          0.394            0.340        0.351    -0.279 0.019 0/11
            smooth50     hmm    0.358    0.837     0.760             1.212       0.134        0.443 11          0.406            0.313        0.356    -0.275 0.020 0/11
                hold     hmm    0.349    0.863     0.798             2.016       0.389        0.541 11          0.382            0.369        0.297    -0.317 0.019 0/11
```
