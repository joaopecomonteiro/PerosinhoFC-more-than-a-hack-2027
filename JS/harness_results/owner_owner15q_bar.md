# owner experiment `owner15q_bar`

```
{
 "tag": "owner15q_bar",
 "condition": "bar",
 "seeds": [
  0
 ],
 "layers": [
  "oracle",
  "dead_v1"
 ],
 "decoders": [
  "hmm"
 ],
 "models": {
  "who_v1": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\who_v1.joblib",
  "gbm_who_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_q_who_im.joblib",
  "gbm_full_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_q_full_im.joblib",
  "gbm_stack_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_gbm_q_stack_im.joblib",
  "mlp_full_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_mlp_q_full_im.pt",
  "tcn_full_im": "C:\\Users\\pc\\Documents\\GitHub\\PerosinhoFC-more-than-a-hack-2027\\JS\\models\\owner_tcn_q_full_im.pt"
 },
 "hybrid_base": [
  "gbm_full_im",
  "tcn_full_im"
 ],
 "ref": "who_v1",
 "external": [],
 "boundary_from": "tcn_full_im",
 "bnd_alpha": [
  0.5,
  0.9
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
 "minutes": 1.3
}
```

## Per alive frame (mean over games per source, then over sources)

```
                      acc_all  acc_seen  acc_missing  auc_all  logloss_all  brier_all  ece_all
method                                                                                        
tcn_full_im             0.919     0.931        0.722    0.976        0.202      0.059    0.015
tcn_full_im_hard        0.919     0.931        0.722    0.918        0.336      0.078    0.061
tcn_full_im+override    0.918     0.931        0.722    0.972        0.211      0.061    0.013
who_v1                  0.917     0.930        0.729    0.976        0.203      0.060    0.014
who_v1_hard             0.917     0.930        0.729    0.915        0.341      0.080    0.063
tcn_full_im+prod0.5     0.915     0.927        0.701    0.974        0.218      0.063    0.031
mlp_full_im             0.913     0.926        0.718    0.971        0.219      0.064    0.016
mlp_full_im_hard        0.913     0.926        0.718    0.910        0.359      0.084    0.067
gbm_stack_im_hard       0.911     0.923        0.731    0.907        0.368      0.086    0.069
gbm_stack_im            0.911     0.923        0.731    0.970        0.229      0.066    0.028
gbm_full_im             0.910     0.923        0.734    0.970        0.232      0.067    0.029
gbm_full_im_hard        0.910     0.923        0.734    0.907        0.372      0.087    0.070
gbm_full_im+override    0.909     0.923        0.734    0.964        0.238      0.068    0.024
tcn_full_im+prod        0.906     0.919        0.679    0.971        0.257      0.073    0.052
gbm_who_im              0.906     0.920        0.736    0.968        0.239      0.069    0.028
gbm_full_im+prod0.5     0.904     0.919        0.714    0.967        0.243      0.072    0.023
gbm_full_im+prod        0.891     0.907        0.706    0.961        0.288      0.083    0.058
rule_hard               0.856     0.876        0.658    0.851        0.580      0.139    0.124
rule                    0.856     0.876        0.658    0.905        0.371      0.112    0.019
hold                    0.809     0.833        0.661    0.807        0.763      0.191    0.191
smooth50                0.778     0.805        0.656    0.784        0.843      0.206    0.189
nearest                 0.724     0.747        0.527    0.807        0.572      0.183    0.080
```

## Layer `oracle` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
                       method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se wins
           tcn_full_im+bnd0.9     hmm    0.678    0.947     0.922             0.827       0.349        0.710  3          0.664            0.750        0.621     0.004 0.017  2/3
     who_v1+tcn_full_imbnd0.5     hmm    0.678    0.948     0.923             0.818       0.351        0.709  3          0.662            0.750        0.621     0.003 0.017  2/3
           tcn_full_im+bnd0.5     hmm    0.676    0.946     0.921             0.760       0.306        0.688  3          0.635            0.754        0.638     0.001 0.010  1/3
                       who_v1     hmm    0.674    0.948     0.923             0.733       0.282        0.696  3          0.637            0.734        0.651       NaN   NaN  NaN
     who_v1+tcn_full_imbnd0.9     hmm    0.670    0.947     0.923             0.902       0.394        0.723  3          0.668            0.740        0.602    -0.004 0.024  2/3
                  tcn_full_im     hmm    0.663    0.946     0.920             0.719       0.265        0.670  3          0.613            0.744        0.633    -0.011 0.011  1/3
         tcn_full_im+override     hmm    0.655    0.945     0.918             0.742       0.266        0.674  3          0.617            0.714        0.634    -0.019 0.001  0/3
gbm_full_im+tcn_full_imbnd0.5     hmm    0.652    0.941     0.914             0.795       0.308        0.673  3          0.639            0.718        0.601    -0.022 0.015  1/3
gbm_full_im+tcn_full_imbnd0.9     hmm    0.652    0.942     0.915             0.904       0.368        0.700  3          0.664            0.716        0.577    -0.022 0.029  1/3
                  gbm_full_im     hmm    0.650    0.941     0.913             0.736       0.264        0.660  3          0.625            0.715        0.609    -0.025 0.009  0/3
         gbm_full_im+override     hmm    0.647    0.940     0.912             0.751       0.271        0.660  3          0.628            0.702        0.611    -0.027 0.009  0/3
                 gbm_stack_im     hmm    0.647    0.941     0.913             0.715       0.243        0.650  3          0.634            0.702        0.604    -0.028 0.013  0/3
          tcn_full_im+prod0.5     hmm    0.642    0.942     0.914             0.732       0.248        0.658  3          0.587            0.724        0.615    -0.032 0.012  0/3
                  mlp_full_im     hmm    0.641    0.943     0.916             0.682       0.216        0.643  3          0.588            0.718        0.616    -0.034 0.010  0/3
                   gbm_who_im     hmm    0.640    0.939     0.911             0.746       0.257        0.654  3          0.615            0.675        0.629    -0.035 0.012  0/3
             tcn_full_im_hard     hmm    0.638    0.945     0.919             0.993       0.380        0.724  3          0.652            0.705        0.558    -0.036 0.031  1/3
             tcn_full_im+veto     hmm    0.631    0.942     0.914             0.655       0.230        0.591  3          0.572            0.690        0.632    -0.043 0.014  0/3
          gbm_full_im+prod0.5     hmm    0.620    0.935     0.905             0.729       0.238        0.610  3          0.604            0.680        0.577    -0.054 0.012  0/3
             gbm_full_im+veto     hmm    0.617    0.938     0.908             0.654       0.210        0.589  3          0.572            0.679        0.600    -0.058 0.004  0/3
                  who_v1_hard     hmm    0.612    0.945     0.919             1.147       0.425        0.733  3          0.655            0.659        0.523    -0.062 0.043  1/3
             tcn_full_im+prod     hmm    0.612    0.935     0.905             0.725       0.218        0.614  3          0.564            0.691        0.580    -0.063 0.010  0/3
             mlp_full_im_hard     hmm    0.605    0.941     0.914             1.045       0.344        0.713  3          0.628            0.667        0.519    -0.070 0.036  0/3
             gbm_full_im_hard     hmm    0.601    0.939     0.911             1.132       0.418        0.698  3          0.643            0.659        0.500    -0.074 0.045  1/3
            gbm_stack_im_hard     hmm    0.593    0.939     0.911             1.105       0.378        0.694  3          0.638            0.648        0.494    -0.081 0.046  1/3
             gbm_full_im+prod     hmm    0.583    0.926     0.892             0.745       0.213        0.574  3          0.564            0.653        0.532    -0.091 0.014  0/3
                     rule_seg     hmm    0.528    0.907     0.864             0.655       0.143        0.472  3          0.507            0.600        0.475    -0.147 0.015  0/3
                         rule     hmm    0.511    0.902     0.856             0.673       0.133        0.468  3          0.496            0.567        0.470    -0.163 0.012  0/3
                    rule_hard     hmm    0.509    0.902     0.856             0.680       0.133        0.467  3          0.495            0.565        0.466    -0.166 0.013  0/3
                     smooth50     hmm    0.362    0.852     0.781             1.128       0.125        0.427  3          0.399            0.313        0.374    -0.312 0.056  0/3
                      nearest     hmm    0.362    0.861     0.796             1.192       0.158        0.437  3          0.390            0.340        0.355    -0.313 0.043  0/3
                         hold     hmm    0.342    0.871     0.810             2.074       0.401        0.541  3          0.394            0.369        0.265    -0.332 0.044  0/3
```

## Layer `dead_v1` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `who_v1`

```
                       method decoder  spellF1  macroF1  team_acc  spells_pred/true  match_0-3s  match_3-10s  n  spellF1_idsse  spellF1_metrica  spellF1_pff  d_vs_ref    se wins
     who_v1+tcn_full_imbnd0.5     hmm    0.561    0.870     0.926             0.790       0.202        0.591  3          0.540            0.620        0.524     0.006 0.007  2/3
           tcn_full_im+bnd0.9     hmm    0.560    0.870     0.925             0.831       0.208        0.597  3          0.543            0.616        0.522     0.006 0.006  2/3
           tcn_full_im+bnd0.5     hmm    0.560    0.870     0.925             0.760       0.174        0.583  3          0.526            0.627        0.528     0.005 0.010  1/3
                       who_v1     hmm    0.555    0.870     0.926             0.726       0.151        0.572  3          0.533            0.602        0.529       NaN   NaN  NaN
     who_v1+tcn_full_imbnd0.9     hmm    0.553    0.870     0.925             0.882       0.236        0.610  3          0.545            0.614        0.501    -0.001 0.013  2/3
gbm_full_im+tcn_full_imbnd0.9     hmm    0.544    0.865     0.917             0.872       0.224        0.585  3          0.545            0.586        0.502    -0.011 0.012  1/3
gbm_full_im+tcn_full_imbnd0.5     hmm    0.543    0.865     0.918             0.767       0.184        0.547  3          0.528            0.598        0.505    -0.011 0.006  0/3
                  tcn_full_im     hmm    0.541    0.868     0.923             0.697       0.119        0.552  3          0.504            0.618        0.501    -0.014 0.015  1/3
             tcn_full_im_hard     hmm    0.540    0.869     0.923             0.941       0.216        0.619  3          0.549            0.585        0.486    -0.015 0.017  1/3
         tcn_full_im+override     hmm    0.534    0.867     0.921             0.721       0.119        0.558  3          0.508            0.592        0.501    -0.021 0.006  0/3
                  gbm_full_im     hmm    0.531    0.864     0.916             0.704       0.126        0.532  3          0.514            0.590        0.487    -0.024 0.009  0/3
                 gbm_stack_im     hmm    0.530    0.864     0.916             0.686       0.121        0.525  3          0.517            0.589        0.485    -0.025 0.010  0/3
                  who_v1_hard     hmm    0.528    0.868     0.924             1.032       0.256        0.628  3          0.532            0.562        0.491    -0.027 0.013  0/3
         gbm_full_im+override     hmm    0.526    0.863     0.915             0.714       0.127        0.531  3          0.514            0.580        0.485    -0.029 0.008  0/3
          tcn_full_im+prod0.5     hmm    0.526    0.865     0.918             0.716       0.119        0.540  3          0.487            0.589        0.502    -0.029 0.010  0/3
                  mlp_full_im     hmm    0.525    0.866     0.919             0.669       0.102        0.517  3          0.484            0.589        0.503    -0.030 0.011  0/3
                   gbm_who_im     hmm    0.521    0.862     0.913             0.722       0.128        0.523  3          0.501            0.557        0.505    -0.034 0.006  0/3
             mlp_full_im_hard     hmm    0.517    0.865     0.919             0.969       0.202        0.603  3          0.521            0.558        0.471    -0.038 0.013  0/3
             gbm_full_im_hard     hmm    0.516    0.863     0.915             1.038       0.251        0.597  3          0.531            0.549        0.468    -0.039 0.018  0/3
            gbm_stack_im_hard     hmm    0.515    0.863     0.915             0.986       0.228        0.588  3          0.535            0.548        0.462    -0.040 0.021  1/3
          gbm_full_im+prod0.5     hmm    0.513    0.859     0.909             0.704       0.123        0.497  3          0.507            0.566        0.465    -0.042 0.011  0/3
             tcn_full_im+veto     hmm    0.511    0.865     0.917             0.636       0.095        0.478  3          0.474            0.560        0.500    -0.043 0.009  0/3
             gbm_full_im+veto     hmm    0.502    0.861     0.912             0.630       0.086        0.471  3          0.476            0.555        0.476    -0.053 0.003  0/3
             tcn_full_im+prod     hmm    0.501    0.859     0.909             0.717       0.111        0.493  3          0.477            0.566        0.461    -0.054 0.009  0/3
             gbm_full_im+prod     hmm    0.480    0.850     0.896             0.730       0.106        0.465  3          0.476            0.538        0.426    -0.075 0.014  0/3
                     rule_seg     hmm    0.433    0.829     0.863             0.660       0.069        0.383  3          0.434            0.479        0.385    -0.122 0.013  0/3
                         rule     hmm    0.430    0.828     0.860             0.662       0.068        0.381  3          0.432            0.474        0.383    -0.125 0.013  0/3
                    rule_hard     hmm    0.429    0.828     0.860             0.671       0.069        0.383  3          0.432            0.475        0.380    -0.126 0.014  0/3
                      nearest     hmm    0.315    0.795     0.808             1.072       0.072        0.363  3          0.333            0.279        0.334    -0.240 0.042  0/3
                     smooth50     hmm    0.300    0.783     0.788             1.071       0.049        0.348  3          0.338            0.251        0.311    -0.255 0.049  0/3
                         hold     hmm    0.285    0.797     0.813             1.907       0.236        0.452  3          0.329            0.296        0.232    -0.269 0.032  0/3
```
