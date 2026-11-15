# owner experiment `headchkq_bar`

```
{
 "tag": "headchkq_bar",
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
  "gbm_full_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_q_full_im.joblib",
  "gbm_stack_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_q_stack_im.joblib",
  "gbm_full_heading_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_q_full_heading_im.joblib",
  "gbm_stack_heading_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_q_stack_heading_im.joblib"
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
 "minutes": 0.6
}
```

## Per alive frame (mean over games per source, then over sources)

```
                      acc_all  acc_seen  acc_missing  auc_all  logloss_all  brier_all  ece_all
method                                                                                        
who_v1                  0.917     0.930        0.729    0.976        0.203      0.060    0.014
gbm_stack_heading_im    0.906     0.919        0.768    0.966        0.294      0.079    0.093
gbm_stack_im            0.906     0.918        0.755    0.966        0.296      0.080    0.095
gbm_full_heading_im     0.904     0.919        0.734    0.964        0.296      0.080    0.091
gbm_full_im             0.903     0.917        0.730    0.963        0.299      0.081    0.093
rule_hard               0.856     0.876        0.658    0.851        0.580      0.139    0.124
rule                    0.856     0.876        0.658    0.905        0.371      0.112    0.019
blend                   0.816     0.841        0.654    0.863        0.462      0.146    0.070
hold                    0.809     0.833        0.661    0.807        0.763      0.191    0.191
smooth50                0.778     0.805        0.656    0.784        0.843      0.206    0.189
nearest                 0.724     0.747        0.527    0.807        0.572      0.183    0.080
```

## Layer `dead_v1` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
              method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se wins
              who_v1     hmm    0.555    0.870     0.926             0.726       0.151        0.572  3          0.533            0.602        0.529       NaN   NaN  NaN
 gbm_full_heading_im     hmm    0.524    0.862     0.912             0.705       0.123        0.522  3          0.498            0.579        0.495    -0.031 0.004  0/3
         gbm_full_im     hmm    0.519    0.862     0.912             0.666       0.100        0.509  3          0.505            0.568        0.485    -0.035 0.005  0/3
gbm_stack_heading_im     hmm    0.508    0.860     0.910             0.666       0.089        0.503  3          0.493            0.571        0.462    -0.046 0.010  0/3
        gbm_stack_im     hmm    0.507    0.861     0.911             0.637       0.090        0.482  3          0.483            0.564        0.473    -0.048 0.005  0/3
            rule_seg     hmm    0.433    0.829     0.863             0.660       0.069        0.383  3          0.434            0.479        0.385    -0.122 0.013  0/3
                rule     hmm    0.430    0.828     0.860             0.662       0.068        0.381  3          0.432            0.474        0.383    -0.125 0.013  0/3
           rule_hard     hmm    0.429    0.828     0.860             0.671       0.069        0.383  3          0.432            0.475        0.380    -0.126 0.014  0/3
               blend     hmm    0.404    0.819     0.847             1.030       0.156        0.454  3          0.427            0.398        0.386    -0.151 0.028  0/3
             nearest     hmm    0.315    0.795     0.808             1.072       0.072        0.363  3          0.333            0.279        0.334    -0.240 0.042  0/3
            smooth50     hmm    0.300    0.783     0.788             1.071       0.049        0.348  3          0.338            0.251        0.311    -0.255 0.049  0/3
                hold     hmm    0.285    0.797     0.813             1.907       0.236        0.452  3          0.329            0.296        0.232    -0.269 0.032  0/3
```

## Layer `oracle` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
              method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se wins
              who_v1     hmm    0.674    0.948     0.923             0.733       0.282        0.696  3          0.637            0.734        0.651       NaN   NaN  NaN
 gbm_full_heading_im     hmm    0.632    0.938     0.909             0.743       0.247        0.641  3          0.609            0.690        0.597    -0.042 0.008  0/3
         gbm_full_im     hmm    0.631    0.938     0.909             0.702       0.226        0.626  3          0.607            0.679        0.606    -0.044 0.007  0/3
gbm_stack_heading_im     hmm    0.617    0.938     0.908             0.689       0.196        0.616  3          0.594            0.676        0.581    -0.058 0.008  0/3
        gbm_stack_im     hmm    0.614    0.937     0.908             0.659       0.182        0.608  3          0.575            0.667        0.599    -0.060 0.004  0/3
            rule_seg     hmm    0.528    0.907     0.864             0.655       0.143        0.472  3          0.507            0.600        0.475    -0.147 0.015  0/3
                rule     hmm    0.511    0.902     0.856             0.673       0.133        0.468  3          0.496            0.567        0.470    -0.163 0.012  0/3
           rule_hard     hmm    0.509    0.902     0.856             0.680       0.133        0.467  3          0.495            0.565        0.466    -0.166 0.013  0/3
               blend     hmm    0.483    0.891     0.840             1.127       0.292        0.554  3          0.540            0.492        0.417    -0.192 0.047  0/3
            smooth50     hmm    0.362    0.852     0.781             1.128       0.125        0.427  3          0.399            0.313        0.374    -0.312 0.056  0/3
             nearest     hmm    0.362    0.861     0.796             1.192       0.158        0.437  3          0.390            0.340        0.355    -0.313 0.043  0/3
                hold     hmm    0.342    0.871     0.810             2.074       0.401        0.541  3          0.394            0.369        0.265    -0.332 0.044  0/3
```
