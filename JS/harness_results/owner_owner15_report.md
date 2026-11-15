# Owner experiments `owner15` (other games only)

Paired Δ spellF1 = a − b per game × seed; verdict needs |mean| > 2 se. hmm decoder unless noted.

Sanity first: in condition `bar`, layer `dead_v1`, `who_v1` and `nearest` must reproduce EXPERIMENTS #35 on the 11 val games (spellF1 0.516 and 0.309) -- same models, same ball fill, same decoder. If they do not, stop and check the models / bundles before reading anything else.


## Condition `bar` (11 games, seeds [0])

Sanity vs #35 (plain mean over games; expect who_v1 0.516, nearest 0.309 on the 11 val games): `nearest` 0.309, `who_v1` 0.516 (11 games)


### Layer `oracle`

Top 8 by spellF1 (mean per source, then over sources): `tcn_full_all` 0.696, `gbm_full_im+tcn_full_imbnd0.5` 0.689, `gbm_full_im+tcn_full_imbnd0.9` 0.689, `tcn_full_im+bnd0.5` 0.688, `gbm_stack_im` 0.686, `tcn_full_im+bnd0.9` 0.686, `tcn_full_im` 0.684, `mlp_full_im` 0.681

```
                        question                             a                 b  a_spellF1  b_spellF1   mean    se  wins  verdict  d_idsse  d_metrica  d_pff
              Q1 rules vs who_v1                          rule            who_v1      0.479      0.636 -0.158 0.010  0/11 b better   -0.153     -0.167 -0.158
              Q1 rules vs who_v1                     rule_hard            who_v1      0.476      0.636 -0.160 0.011  0/11 b better   -0.154     -0.169 -0.160
              Q1 rules vs who_v1                      rule_seg            who_v1      0.495      0.636 -0.142 0.010  0/11 b better   -0.138     -0.134 -0.144
              Q1 rules vs who_v1                          hold            who_v1      0.319      0.636 -0.317 0.019  0/11 b better   -0.290     -0.365 -0.318
              Q1 rules vs who_v1                      smooth50            who_v1      0.361      0.636 -0.275 0.020  0/11 b better   -0.266     -0.421 -0.259
              Q1 rules vs who_v1                       nearest            who_v1      0.358      0.636 -0.279 0.019  0/11 b better   -0.278     -0.394 -0.265
                     Q2 features                    gbm_who_im            who_v1      0.632      0.636 -0.005 0.004  5/11   no gap   -0.000     -0.002 -0.006
                     Q2 features                   gbm_full_im        gbm_who_im      0.626      0.632 -0.005 0.006  4/11   no gap    0.004      0.034 -0.013
                     Q2 features                   gbm_full_im            who_v1      0.626      0.636 -0.010 0.006  3/11   no gap    0.004      0.032 -0.019
                     Q2 features                  gbm_stack_im       gbm_full_im      0.637      0.626  0.011 0.004  9/11 a better    0.005      0.002  0.013
  Q3 training sources (gbm full)                gbm_full_imskc       gbm_full_im      0.568      0.626 -0.058 0.009  0/11 b better   -0.111     -0.041 -0.047
  Q3 training sources (gbm full)                gbm_full_impff       gbm_full_im      0.623      0.626 -0.003 0.006  7/11   no gap   -0.029     -0.030  0.007
  Q3 training sources (gbm full)                  gbm_full_all       gbm_full_im      0.582      0.626 -0.044 0.013  1/11 b better   -0.118     -0.075 -0.022
                  Q4 NN vs trees                   tcn_full_im       gbm_full_im      0.631      0.626  0.005 0.004  8/11   no gap   -0.004      0.013  0.006
                  Q4 NN vs trees                   tcn_full_im            who_v1      0.631      0.636 -0.005 0.008  3/11   no gap   -0.001      0.045 -0.012
                  Q4 NN vs trees                   mlp_full_im       gbm_full_im      0.642      0.626  0.016 0.006  9/11 a better   -0.007     -0.013  0.025
                  Q4 NN vs trees                  tcn_full_all       tcn_full_im      0.659      0.631  0.028 0.012  9/11 a better   -0.002     -0.001  0.039
                  Q4 NN vs trees                  mlp_full_all       mlp_full_im      0.599      0.642 -0.043 0.012  1/11 b better   -0.077     -0.068 -0.032
                  Q4 NN vs trees                  tcn_full_all       gbm_full_im      0.659      0.626  0.033 0.012 10/11 a better   -0.007      0.012  0.046
 Q5 probabilities vs hard labels                        who_v1       who_v1_hard      0.636      0.534  0.103 0.016 10/11 a better    0.016      0.075  0.128
 Q5 probabilities vs hard labels                   gbm_full_im  gbm_full_im_hard      0.626      0.548  0.079 0.013 10/11 a better    0.001      0.058  0.101
 Q5 probabilities vs hard labels                  gbm_stack_im gbm_stack_im_hard      0.637      0.546  0.091 0.015 10/11 a better    0.010      0.055  0.115
 Q5 probabilities vs hard labels                   mlp_full_im  mlp_full_im_hard      0.642      0.538  0.104 0.016 10/11 a better    0.018      0.076  0.129
 Q5 probabilities vs hard labels                   tcn_full_im  tcn_full_im_hard      0.631      0.600  0.032 0.005 11/11 a better    0.002      0.033  0.039
 Q5 probabilities vs hard labels                          rule         rule_hard      0.479      0.476  0.002 0.001  9/11 a better    0.001      0.002  0.002
         Q6 rule x model hybrids              gbm_full_im+prod       gbm_full_im      0.576      0.626 -0.050 0.006  0/11 b better   -0.055     -0.080 -0.045
         Q6 rule x model hybrids           gbm_full_im+prod0.5       gbm_full_im      0.609      0.626 -0.017 0.004  2/11 b better   -0.011     -0.038 -0.016
         Q6 rule x model hybrids          gbm_full_im+override       gbm_full_im      0.622      0.626 -0.004 0.003  5/11   no gap   -0.001     -0.033 -0.001
         Q6 rule x model hybrids              gbm_full_im+veto       gbm_full_im      0.609      0.626 -0.017 0.006  1/11 b better   -0.042     -0.053 -0.007
         Q6 rule x model hybrids              tcn_full_im+prod       tcn_full_im      0.612      0.631 -0.019 0.004  1/11 b better   -0.003     -0.023 -0.023
         Q6 rule x model hybrids           tcn_full_im+prod0.5       tcn_full_im      0.623      0.631 -0.008 0.003  3/11 b better    0.001     -0.010 -0.010
         Q6 rule x model hybrids          tcn_full_im+override       tcn_full_im      0.624      0.631 -0.008 0.004  0/11   no gap   -0.002     -0.048 -0.004
         Q6 rule x model hybrids              tcn_full_im+veto       tcn_full_im      0.611      0.631 -0.020 0.008  1/11 b better   -0.060     -0.044 -0.007
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.5       tcn_full_im      0.632      0.631  0.001 0.003  6/11   no gap    0.012      0.003 -0.003
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.9       tcn_full_im      0.628      0.631 -0.003 0.005  4/11   no gap    0.022     -0.009 -0.009
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.5            who_v1      0.635      0.636 -0.002 0.003  3/11   no gap    0.007      0.013 -0.006
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.9            who_v1      0.624      0.636 -0.013 0.007  2/11   no gap    0.013      0.016 -0.023
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.5       gbm_full_im      0.630      0.626  0.004 0.003  8/11   no gap    0.010      0.021 -0.000
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.9       gbm_full_im      0.623      0.626 -0.003 0.007  5/11   no gap    0.024      0.016 -0.012
```


### Layer `smooth50`

Top 8 by spellF1 (mean per source, then over sources): `tcn_full_all` 0.577, `gbm_full_im+tcn_full_imbnd0.9` 0.570, `gbm_full_im+tcn_full_imbnd0.5` 0.568, `gbm_stack_im` 0.565, `tcn_full_im+bnd0.5` 0.564, `tcn_full_im+bnd0.9` 0.562, `tcn_full_im` 0.561, `gbm_full_im` 0.559

```
                        question                             a                 b  a_spellF1  b_spellF1   mean    se  wins  verdict  d_idsse  d_metrica  d_pff
              Q1 rules vs who_v1                          rule            who_v1      0.408      0.508 -0.100 0.008  0/11 b better   -0.107     -0.141 -0.094
              Q1 rules vs who_v1                     rule_hard            who_v1      0.406      0.508 -0.102 0.008  0/11 b better   -0.109     -0.142 -0.096
              Q1 rules vs who_v1                      rule_seg            who_v1      0.425      0.508 -0.084 0.008  0/11 b better   -0.092     -0.118 -0.077
              Q1 rules vs who_v1                          hold            who_v1      0.256      0.508 -0.252 0.016  0/11 b better   -0.229     -0.324 -0.249
              Q1 rules vs who_v1                      smooth50            who_v1      0.298      0.508 -0.210 0.020  0/11 b better   -0.217     -0.370 -0.189
              Q1 rules vs who_v1                       nearest            who_v1      0.289      0.508 -0.219 0.017  0/11 b better   -0.215     -0.343 -0.205
                     Q2 features                    gbm_who_im            who_v1      0.506      0.508 -0.002 0.001  4/11   no gap    0.002     -0.002 -0.004
                     Q2 features                   gbm_full_im        gbm_who_im      0.509      0.506  0.004 0.004  8/11   no gap    0.003      0.032  0.000
                     Q2 features                   gbm_full_im            who_v1      0.509      0.508  0.001 0.004  6/11   no gap    0.006      0.029 -0.003
                     Q2 features                  gbm_stack_im       gbm_full_im      0.515      0.509  0.005 0.002  8/11 a better    0.006      0.006  0.005
  Q3 training sources (gbm full)                gbm_full_imskc       gbm_full_im      0.481      0.509 -0.028 0.007  1/11 b better   -0.073     -0.031 -0.016
  Q3 training sources (gbm full)                gbm_full_impff       gbm_full_im      0.507      0.509 -0.003 0.005  4/11   no gap   -0.016     -0.030  0.004
  Q3 training sources (gbm full)                  gbm_full_all       gbm_full_im      0.485      0.509 -0.024 0.011  3/11 b better   -0.081     -0.068 -0.004
                  Q4 NN vs trees                   tcn_full_im       gbm_full_im      0.508      0.509 -0.001 0.003  3/11   no gap    0.002      0.005 -0.003
                  Q4 NN vs trees                   tcn_full_im            who_v1      0.508      0.508 -0.000 0.006  5/11   no gap    0.008      0.034 -0.006
                  Q4 NN vs trees                   mlp_full_im       gbm_full_im      0.512      0.509  0.003 0.003  6/11   no gap   -0.002     -0.013  0.006
                  Q4 NN vs trees                  tcn_full_all       tcn_full_im      0.533      0.508  0.025 0.009 10/11 a better    0.007      0.011  0.031
                  Q4 NN vs trees                  mlp_full_all       mlp_full_im      0.496      0.512 -0.016 0.010  3/11   no gap   -0.052     -0.061 -0.001
                  Q4 NN vs trees                  tcn_full_all       gbm_full_im      0.533      0.509  0.023 0.007  9/11 a better    0.009      0.016  0.028
 Q5 probabilities vs hard labels                        who_v1       who_v1_hard      0.508      0.430  0.078 0.012 10/11 a better    0.013      0.070  0.095
 Q5 probabilities vs hard labels                   gbm_full_im  gbm_full_im_hard      0.509      0.445  0.065 0.011 10/11 a better    0.001      0.049  0.083
 Q5 probabilities vs hard labels                  gbm_stack_im gbm_stack_im_hard      0.515      0.443  0.071 0.012 10/11 a better    0.008      0.049  0.090
 Q5 probabilities vs hard labels                   mlp_full_im  mlp_full_im_hard      0.512      0.435  0.076 0.012 10/11 a better    0.009      0.069  0.094
 Q5 probabilities vs hard labels                   tcn_full_im  tcn_full_im_hard      0.508      0.476  0.032 0.005 11/11 a better    0.003      0.041  0.038
 Q5 probabilities vs hard labels                          rule         rule_hard      0.408      0.406  0.002 0.001 11/11 a better    0.002      0.002  0.002
         Q6 rule x model hybrids              gbm_full_im+prod       gbm_full_im      0.470      0.509 -0.040 0.007  0/11 b better   -0.040     -0.078 -0.035
         Q6 rule x model hybrids           gbm_full_im+prod0.5       gbm_full_im      0.494      0.509 -0.016 0.004  1/11 b better   -0.003     -0.029 -0.017
         Q6 rule x model hybrids          gbm_full_im+override       gbm_full_im      0.506      0.509 -0.003 0.003  3/11   no gap   -0.000     -0.029 -0.000
         Q6 rule x model hybrids              gbm_full_im+veto       gbm_full_im      0.488      0.509 -0.021 0.008  1/11 b better   -0.048     -0.066 -0.008
         Q6 rule x model hybrids              tcn_full_im+prod       tcn_full_im      0.492      0.508 -0.016 0.004  2/11 b better   -0.001     -0.027 -0.019
         Q6 rule x model hybrids           tcn_full_im+prod0.5       tcn_full_im      0.501      0.508 -0.007 0.003  2/11 b better    0.001     -0.016 -0.008
         Q6 rule x model hybrids          tcn_full_im+override       tcn_full_im      0.501      0.508 -0.007 0.004  1/11   no gap   -0.002     -0.041 -0.004
         Q6 rule x model hybrids              tcn_full_im+veto       tcn_full_im      0.485      0.508 -0.023 0.008  1/11 b better   -0.060     -0.054 -0.009
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.5       tcn_full_im      0.508      0.508 -0.000 0.002  4/11   no gap    0.009      0.003 -0.003
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.9       tcn_full_im      0.505      0.508 -0.003 0.003  2/11   no gap    0.016     -0.004 -0.008
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.5            who_v1      0.507      0.508 -0.001 0.003  4/11   no gap    0.011      0.013 -0.006
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.9            who_v1      0.499      0.508 -0.009 0.005  4/11   no gap    0.012      0.014 -0.017
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.5       gbm_full_im      0.509      0.509  0.000 0.003  6/11   no gap    0.007      0.022 -0.004
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.9       gbm_full_im      0.508      0.509 -0.001 0.005  5/11   no gap    0.023      0.018 -0.010
```


### Layer `dead_v1`

Top 8 by spellF1 (mean per source, then over sources): `tcn_full_all` 0.580, `gbm_full_im+tcn_full_imbnd0.9` 0.575, `gbm_full_im+tcn_full_imbnd0.5` 0.574, `tcn_full_im+bnd0.5` 0.573, `tcn_full_im+bnd0.9` 0.571, `gbm_stack_im` 0.570, `tcn_full_im` 0.569, `tcn_full_im+prod0.5` 0.565

```
                        question                             a                 b  a_spellF1  b_spellF1   mean    se  wins  verdict  d_idsse  d_metrica  d_pff
              Q1 rules vs who_v1                          rule            who_v1      0.396      0.516 -0.120 0.008  0/11 b better   -0.112     -0.128 -0.120
              Q1 rules vs who_v1                     rule_hard            who_v1      0.395      0.516 -0.121 0.008  0/11 b better   -0.116     -0.128 -0.122
              Q1 rules vs who_v1                      rule_seg            who_v1      0.401      0.516 -0.115 0.007  0/11 b better   -0.104     -0.123 -0.116
              Q1 rules vs who_v1                          hold            who_v1      0.272      0.516 -0.244 0.016  0/11 b better   -0.238     -0.306 -0.237
              Q1 rules vs who_v1                      smooth50            who_v1      0.297      0.516 -0.218 0.017  0/11 b better   -0.213     -0.351 -0.203
              Q1 rules vs who_v1                       nearest            who_v1      0.309      0.516 -0.207 0.017  0/11 b better   -0.221     -0.324 -0.189
                     Q2 features                    gbm_who_im            who_v1      0.516      0.516 -0.000 0.002  5/11   no gap    0.001      0.002 -0.001
                     Q2 features                   gbm_full_im        gbm_who_im      0.517      0.516  0.001 0.005  7/11   no gap    0.004      0.038 -0.004
                     Q2 features                   gbm_full_im            who_v1      0.517      0.516  0.001 0.006  6/11   no gap    0.005      0.040 -0.005
                     Q2 features                  gbm_stack_im       gbm_full_im      0.519      0.517  0.002 0.004  7/11   no gap    0.007      0.008  0.001
  Q3 training sources (gbm full)                gbm_full_imskc       gbm_full_im      0.456      0.517 -0.061 0.005  0/11 b better   -0.085     -0.041 -0.057
  Q3 training sources (gbm full)                gbm_full_impff       gbm_full_im      0.506      0.517 -0.010 0.006  4/11   no gap   -0.017     -0.039 -0.005
  Q3 training sources (gbm full)                  gbm_full_all       gbm_full_im      0.463      0.517 -0.054 0.007  0/11 b better   -0.090     -0.070 -0.043
                  Q4 NN vs trees                   tcn_full_im       gbm_full_im      0.524      0.517  0.007 0.006  8/11   no gap   -0.005      0.008  0.010
                  Q4 NN vs trees                   tcn_full_im            who_v1      0.524      0.516  0.008 0.006  7/11   no gap    0.000      0.047  0.005
                  Q4 NN vs trees                   mlp_full_im       gbm_full_im      0.524      0.517  0.007 0.004  8/11 a better   -0.000     -0.017  0.012
                  Q4 NN vs trees                  tcn_full_all       tcn_full_im      0.532      0.524  0.008 0.008  6/11   no gap    0.009      0.015  0.007
                  Q4 NN vs trees                  mlp_full_all       mlp_full_im      0.478      0.524 -0.046 0.008  1/11 b better   -0.066     -0.062 -0.040
                  Q4 NN vs trees                  tcn_full_all       gbm_full_im      0.532      0.517  0.016 0.007  8/11 a better    0.004      0.022  0.018
 Q5 probabilities vs hard labels                        who_v1       who_v1_hard      0.516      0.485  0.031 0.005 11/11 a better    0.004      0.041  0.036
 Q5 probabilities vs hard labels                   gbm_full_im  gbm_full_im_hard      0.517      0.494  0.022 0.006  9/11 a better   -0.008      0.042  0.028
 Q5 probabilities vs hard labels                  gbm_stack_im gbm_stack_im_hard      0.519      0.496  0.024 0.009  9/11 a better   -0.001      0.044  0.027
 Q5 probabilities vs hard labels                   mlp_full_im  mlp_full_im_hard      0.524      0.486  0.038 0.008 10/11 a better    0.004      0.060  0.044
 Q5 probabilities vs hard labels                   tcn_full_im  tcn_full_im_hard      0.524      0.505  0.019 0.004  9/11 a better   -0.005      0.029  0.024
 Q5 probabilities vs hard labels                          rule         rule_hard      0.396      0.395  0.002 0.001  9/11 a better    0.004     -0.001  0.001
         Q6 rule x model hybrids              gbm_full_im+prod       gbm_full_im      0.472      0.517 -0.045 0.007  0/11 b better   -0.029     -0.079 -0.044
         Q6 rule x model hybrids           gbm_full_im+prod0.5       gbm_full_im      0.501      0.517 -0.016 0.005  1/11 b better    0.000     -0.043 -0.017
         Q6 rule x model hybrids          gbm_full_im+override       gbm_full_im      0.513      0.517 -0.004 0.003  6/11   no gap    0.000     -0.031 -0.002
         Q6 rule x model hybrids              gbm_full_im+veto       gbm_full_im      0.497      0.517 -0.020 0.006  1/11 b better   -0.043     -0.048 -0.011
         Q6 rule x model hybrids              tcn_full_im+prod       tcn_full_im      0.510      0.524 -0.014 0.005  3/11 b better    0.003     -0.019 -0.018
         Q6 rule x model hybrids           tcn_full_im+prod0.5       tcn_full_im      0.518      0.524 -0.006 0.004  3/11   no gap    0.005     -0.009 -0.009
         Q6 rule x model hybrids          tcn_full_im+override       tcn_full_im      0.519      0.524 -0.005 0.004  4/11   no gap    0.003     -0.039 -0.003
         Q6 rule x model hybrids              tcn_full_im+veto       tcn_full_im      0.503      0.524 -0.021 0.007  1/11 b better   -0.054     -0.046 -0.009
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.5       tcn_full_im      0.524      0.524 -0.000 0.002  6/11   no gap    0.011      0.003 -0.003
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.9       tcn_full_im      0.520      0.524 -0.004 0.004  3/11   no gap    0.016     -0.003 -0.009
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.5            who_v1      0.524      0.516  0.008 0.001 11/11 a better    0.007      0.016  0.007
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.9            who_v1      0.521      0.516  0.005 0.003  6/11   no gap    0.009      0.019  0.003
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.5       gbm_full_im      0.523      0.517  0.007 0.002 11/11 a better    0.009      0.013  0.005
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.9       gbm_full_im      0.520      0.517  0.003 0.004  6/11   no gap    0.019      0.013 -0.003
```


### Decoder matters: hmm − raw (oracle layer)

```
              method   hmm   raw  hmm-raw
        tcn_full_all 0.659 0.602    0.057
         mlp_full_im 0.642 0.360    0.283
        gbm_stack_im 0.637 0.324    0.313
              who_v1 0.636 0.310    0.326
          gbm_who_im 0.632 0.308    0.324
         tcn_full_im 0.631 0.590    0.041
         gbm_full_im 0.626 0.334    0.292
tcn_full_im+override 0.624 0.554    0.070
      gbm_full_impff 0.623 0.361    0.262
 tcn_full_im+prod0.5 0.623 0.553    0.070
gbm_full_im+override 0.622 0.329    0.293
    tcn_full_im+prod 0.612 0.535    0.077
 gbm_full_im+prod0.5 0.609 0.342    0.267
    tcn_full_im_hard 0.600 0.590    0.010
        mlp_full_all 0.599 0.373    0.225
        gbm_full_all 0.582 0.353    0.229
    gbm_full_im+prod 0.576 0.349    0.227
      gbm_full_imskc 0.568 0.347    0.222
    gbm_full_im_hard 0.548 0.334    0.214
   gbm_stack_im_hard 0.546 0.324    0.222
    mlp_full_im_hard 0.538 0.360    0.178
         who_v1_hard 0.534 0.310    0.223
            rule_seg 0.495 0.493    0.002
                rule 0.479 0.475    0.003
           rule_hard 0.476 0.475    0.001
            smooth50 0.361 0.330    0.031
             nearest 0.358 0.080    0.278
                hold 0.319 0.265    0.054
```


### Per alive frame (calibration)

```
                        acc    auc  logloss  brier    ece
method                                                   
tcn_full_all          0.918  0.970    0.219  0.063  0.032
gbm_stack_im          0.908  0.967    0.227  0.067  0.023
gbm_full_im           0.908  0.966    0.232  0.068  0.023
mlp_full_all          0.906  0.966    0.232  0.068  0.027
gbm_full_impff        0.907  0.966    0.233  0.069  0.024
who_v1                0.905  0.965    0.235  0.070  0.024
mlp_full_im           0.906  0.965    0.236  0.069  0.025
gbm_who_im            0.902  0.964    0.238  0.070  0.026
gbm_full_all          0.903  0.964    0.239  0.071  0.029
gbm_full_im+override  0.906  0.962    0.243  0.071  0.022
gbm_full_imskc        0.902  0.961    0.248  0.073  0.037
gbm_full_im+prod0.5   0.901  0.963    0.256  0.075  0.042
tcn_full_im           0.910  0.961    0.280  0.075  0.058
tcn_full_im+prod0.5   0.908  0.961    0.285  0.076  0.063
tcn_full_im+override  0.908  0.950    0.289  0.077  0.056
tcn_full_im+prod      0.905  0.959    0.301  0.081  0.072
gbm_full_im+prod      0.889  0.959    0.302  0.087  0.066
tcn_full_im_hard      0.910  0.909    0.370  0.087  0.070
gbm_stack_im_hard     0.908  0.906    0.377  0.088  0.072
gbm_full_im_hard      0.908  0.905    0.380  0.089  0.072
mlp_full_im_hard      0.906  0.904    0.387  0.091  0.074
who_v1_hard           0.905  0.903    0.391  0.092  0.075
rule                  0.839  0.889    0.404  0.124  0.036
nearest               0.704  0.785    0.617  0.197  0.096
rule_hard             0.839  0.835    0.647  0.155  0.141
hold                  0.798  0.796    0.807  0.202  0.202
smooth50              0.757  0.764    0.924  0.226  0.209
```

Accuracy where the ball is NOT tracked: `tcn_full_all` 0.769, `mlp_full_all` 0.744, `tcn_full_im` 0.720, `tcn_full_im+override` 0.720, `tcn_full_im_hard` 0.720, `gbm_full_all` 0.719, `tcn_full_im+prod0.5` 0.715, `gbm_who_im` 0.711, `tcn_full_im+prod` 0.707, `gbm_full_im+override` 0.705


## Condition `atd` (11 games, seeds [0, 1])


### Layer `oracle`

Top 8 by spellF1 (mean per source, then over sources): `tcn_full_all` 0.683, `who_v1+tcn_full_imbnd0.5` 0.664, `gbm_stack_im` 0.664, `mlp_full_im` 0.663, `tcn_full_im+bnd0.5` 0.662, `tcn_full_im` 0.661, `gbm_full_im+tcn_full_imbnd0.5` 0.660, `tcn_full_im+override` 0.658

```
                        question                             a                 b  a_spellF1  b_spellF1   mean    se  wins  verdict  d_idsse  d_metrica  d_pff
              Q1 rules vs who_v1                          rule            who_v1      0.481      0.611 -0.130 0.006  0/22 b better   -0.148     -0.144 -0.124
              Q1 rules vs who_v1                     rule_hard            who_v1      0.478      0.611 -0.133 0.006  0/22 b better   -0.150     -0.143 -0.128
              Q1 rules vs who_v1                      rule_seg            who_v1      0.492      0.611 -0.120 0.006  0/22 b better   -0.143     -0.125 -0.113
              Q1 rules vs who_v1                          hold            who_v1      0.280      0.611 -0.331 0.014  0/22 b better   -0.299     -0.447 -0.325
              Q1 rules vs who_v1                      smooth50            who_v1      0.404      0.611 -0.208 0.013  0/22 b better   -0.229     -0.321 -0.188
              Q1 rules vs who_v1                       nearest            who_v1      0.383      0.611 -0.228 0.009  0/22 b better   -0.220     -0.300 -0.222
                     Q2 features                    gbm_who_im            who_v1      0.611      0.611 -0.000 0.002  9/22   no gap    0.003      0.004 -0.002
                     Q2 features                   gbm_full_im        gbm_who_im      0.608      0.611 -0.004 0.002  7/22   no gap   -0.010      0.012 -0.004
                     Q2 features                   gbm_full_im            who_v1      0.608      0.611 -0.004 0.003  8/22   no gap   -0.006      0.016 -0.006
                     Q2 features                  gbm_stack_im       gbm_full_im      0.617      0.608  0.010 0.002 17/22 a better    0.006      0.004  0.012
  Q3 training sources (gbm full)                gbm_full_imskc       gbm_full_im      0.561      0.608 -0.047 0.007  1/22 b better   -0.081     -0.091 -0.032
  Q3 training sources (gbm full)                gbm_full_impff       gbm_full_im      0.607      0.608 -0.001 0.004 11/22   no gap   -0.016     -0.026  0.007
  Q3 training sources (gbm full)                  gbm_full_all       gbm_full_im      0.573      0.608 -0.035 0.008  4/22 b better   -0.078     -0.095 -0.016
                  Q4 NN vs trees                   tcn_full_im       gbm_full_im      0.611      0.608  0.004 0.005 15/22   no gap    0.005      0.004  0.003
                  Q4 NN vs trees                   tcn_full_im            who_v1      0.611      0.611 -0.000 0.005 12/22   no gap   -0.001      0.020 -0.003
                  Q4 NN vs trees                   mlp_full_im       gbm_full_im      0.618      0.608  0.010 0.003 15/22 a better    0.008     -0.001  0.012
                  Q4 NN vs trees                  tcn_full_all       tcn_full_im      0.652      0.611  0.041 0.007 19/22 a better    0.010      0.003  0.053
                  Q4 NN vs trees                  mlp_full_all       mlp_full_im      0.588      0.618 -0.029 0.008  2/22 b better   -0.063     -0.066 -0.016
                  Q4 NN vs trees                  tcn_full_all       gbm_full_im      0.652      0.608  0.044 0.007 19/22 a better    0.015      0.007  0.056
 Q5 probabilities vs hard labels                        who_v1       who_v1_hard      0.611      0.502  0.110 0.009 22/22 a better    0.061      0.054  0.129
 Q5 probabilities vs hard labels                   gbm_full_im  gbm_full_im_hard      0.608      0.505  0.103 0.009 22/22 a better    0.048      0.056  0.122
 Q5 probabilities vs hard labels                  gbm_stack_im gbm_stack_im_hard      0.617      0.510  0.107 0.009 22/22 a better    0.049      0.056  0.128
 Q5 probabilities vs hard labels                   mlp_full_im  mlp_full_im_hard      0.618      0.501  0.116 0.008 22/22 a better    0.064      0.070  0.135
 Q5 probabilities vs hard labels                   tcn_full_im  tcn_full_im_hard      0.611      0.579  0.032 0.004 20/22 a better    0.005      0.015  0.041
 Q5 probabilities vs hard labels                          rule         rule_hard      0.481      0.478  0.003 0.001 18/22 a better    0.001     -0.001  0.003
         Q6 rule x model hybrids              gbm_full_im+prod       gbm_full_im      0.562      0.608 -0.046 0.004  0/22 b better   -0.055     -0.050 -0.043
         Q6 rule x model hybrids           gbm_full_im+prod0.5       gbm_full_im      0.591      0.608 -0.017 0.004  4/22 b better   -0.018     -0.018 -0.017
         Q6 rule x model hybrids          gbm_full_im+override       gbm_full_im      0.605      0.608 -0.003 0.001  3/22 b better   -0.001     -0.005 -0.003
         Q6 rule x model hybrids              gbm_full_im+veto       gbm_full_im      0.597      0.608 -0.011 0.003  4/22 b better   -0.015     -0.032 -0.007
         Q6 rule x model hybrids              tcn_full_im+prod       tcn_full_im      0.593      0.611 -0.018 0.003  2/22 b better   -0.019     -0.019 -0.018
         Q6 rule x model hybrids           tcn_full_im+prod0.5       tcn_full_im      0.604      0.611 -0.007 0.002  4/22 b better   -0.002     -0.004 -0.009
         Q6 rule x model hybrids          tcn_full_im+override       tcn_full_im      0.609      0.611 -0.002 0.001  4/22 b better   -0.000     -0.005 -0.003
         Q6 rule x model hybrids              tcn_full_im+veto       tcn_full_im      0.601      0.611 -0.010 0.003  6/22 b better   -0.024     -0.027 -0.005
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.5       tcn_full_im      0.609      0.611 -0.002 0.002 11/22   no gap    0.003      0.005 -0.004
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.9       tcn_full_im      0.603      0.611 -0.009 0.003  4/22 b better    0.008     -0.003 -0.013
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.5            who_v1      0.611      0.611 -0.001 0.003 10/22   no gap    0.011      0.022 -0.007
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.9            who_v1      0.588      0.611 -0.023 0.006  4/22 b better    0.003      0.030 -0.037
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.5       gbm_full_im      0.604      0.608 -0.003 0.002  9/22   no gap    0.007      0.011 -0.007
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.9       gbm_full_im      0.589      0.608 -0.019 0.004  5/22 b better    0.005      0.006 -0.028
```


### Layer `smooth50`

Top 8 by spellF1 (mean per source, then over sources): `tcn_full_all` 0.547, `who_v1+tcn_full_imbnd0.5` 0.536, `gbm_stack_im` 0.536, `gbm_full_im+tcn_full_imbnd0.5` 0.534, `who_v1+tcn_full_imbnd0.9` 0.532, `tcn_full_im+bnd0.5` 0.530, `gbm_full_im` 0.530, `gbm_full_im+tcn_full_imbnd0.9` 0.530

```
                        question                             a                 b  a_spellF1  b_spellF1   mean    se  wins  verdict  d_idsse  d_metrica  d_pff
              Q1 rules vs who_v1                          rule            who_v1      0.403      0.493 -0.090 0.005  0/22 b better   -0.110     -0.114 -0.082
              Q1 rules vs who_v1                     rule_hard            who_v1      0.400      0.493 -0.093 0.005  0/22 b better   -0.111     -0.115 -0.085
              Q1 rules vs who_v1                      rule_seg            who_v1      0.421      0.493 -0.072 0.005  0/22 b better   -0.098     -0.104 -0.062
              Q1 rules vs who_v1                          hold            who_v1      0.226      0.493 -0.267 0.013  0/22 b better   -0.232     -0.379 -0.262
              Q1 rules vs who_v1                      smooth50            who_v1      0.333      0.493 -0.160 0.011  0/22 b better   -0.187     -0.276 -0.138
              Q1 rules vs who_v1                       nearest            who_v1      0.311      0.493 -0.182 0.008  0/22 b better   -0.168     -0.255 -0.176
                     Q2 features                    gbm_who_im            who_v1      0.490      0.493 -0.003 0.001  5/22 b better   -0.002     -0.001 -0.004
                     Q2 features                   gbm_full_im        gbm_who_im      0.490      0.490 -0.000 0.002  8/22   no gap   -0.004      0.016 -0.001
                     Q2 features                   gbm_full_im            who_v1      0.490      0.493 -0.003 0.002  7/22   no gap   -0.006      0.015 -0.005
                     Q2 features                  gbm_stack_im       gbm_full_im      0.496      0.490  0.006 0.002 18/22 a better    0.002      0.009  0.007
  Q3 training sources (gbm full)                gbm_full_imskc       gbm_full_im      0.469      0.490 -0.021 0.007  5/22 b better   -0.049     -0.079 -0.007
  Q3 training sources (gbm full)                gbm_full_impff       gbm_full_im      0.491      0.490  0.001 0.004 13/22   no gap   -0.016     -0.024  0.008
  Q3 training sources (gbm full)                  gbm_full_all       gbm_full_im      0.476      0.490 -0.014 0.008  9/22   no gap   -0.049     -0.082  0.004
                  Q4 NN vs trees                   tcn_full_im       gbm_full_im      0.489      0.490 -0.001 0.003 10/22   no gap    0.001     -0.003 -0.001
                  Q4 NN vs trees                   tcn_full_im            who_v1      0.489      0.493 -0.004 0.003  9/22   no gap   -0.005      0.012 -0.005
                  Q4 NN vs trees                   mlp_full_im       gbm_full_im      0.493      0.490  0.003 0.002 12/22   no gap    0.004     -0.009  0.004
                  Q4 NN vs trees                  tcn_full_all       tcn_full_im      0.521      0.489  0.031 0.005 20/22 a better    0.009      0.005  0.040
                  Q4 NN vs trees                  mlp_full_all       mlp_full_im      0.486      0.493 -0.007 0.006  7/22   no gap   -0.037     -0.041  0.005
                  Q4 NN vs trees                  tcn_full_all       gbm_full_im      0.521      0.490  0.031 0.006 18/22 a better    0.011      0.002  0.040
 Q5 probabilities vs hard labels                        who_v1       who_v1_hard      0.493      0.412  0.080 0.006 22/22 a better    0.043      0.049  0.094
 Q5 probabilities vs hard labels                   gbm_full_im  gbm_full_im_hard      0.490      0.416  0.074 0.006 22/22 a better    0.035      0.050  0.087
 Q5 probabilities vs hard labels                  gbm_stack_im gbm_stack_im_hard      0.496      0.420  0.076 0.006 22/22 a better    0.035      0.058  0.088
 Q5 probabilities vs hard labels                   mlp_full_im  mlp_full_im_hard      0.493      0.414  0.079 0.006 22/22 a better    0.036      0.056  0.092
 Q5 probabilities vs hard labels                   tcn_full_im  tcn_full_im_hard      0.489      0.463  0.026 0.003 22/22 a better    0.009      0.007  0.033
 Q5 probabilities vs hard labels                          rule         rule_hard      0.403      0.400  0.003 0.001 20/22 a better    0.001      0.000  0.004
         Q6 rule x model hybrids              gbm_full_im+prod       gbm_full_im      0.460      0.490 -0.030 0.003  0/22 b better   -0.044     -0.033 -0.026
         Q6 rule x model hybrids           gbm_full_im+prod0.5       gbm_full_im      0.480      0.490 -0.010 0.003  4/22 b better   -0.014     -0.010 -0.009
         Q6 rule x model hybrids          gbm_full_im+override       gbm_full_im      0.488      0.490 -0.002 0.001  5/22 b better    0.001     -0.002 -0.002
         Q6 rule x model hybrids              gbm_full_im+veto       gbm_full_im      0.478      0.490 -0.012 0.003  3/22 b better   -0.017     -0.038 -0.008
         Q6 rule x model hybrids              tcn_full_im+prod       tcn_full_im      0.477      0.489 -0.012 0.002  4/22 b better   -0.013     -0.006 -0.013
         Q6 rule x model hybrids           tcn_full_im+prod0.5       tcn_full_im      0.485      0.489 -0.005 0.002  7/22 b better   -0.002      0.000 -0.006
         Q6 rule x model hybrids          tcn_full_im+override       tcn_full_im      0.488      0.489 -0.002 0.001  7/22 b better    0.000     -0.003 -0.002
         Q6 rule x model hybrids              tcn_full_im+veto       tcn_full_im      0.477      0.489 -0.012 0.003  3/22 b better   -0.025     -0.033 -0.006
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.5       tcn_full_im      0.487      0.489 -0.002 0.001  6/22   no gap    0.001      0.008 -0.003
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.9       tcn_full_im      0.482      0.489 -0.007 0.002  5/22 b better    0.002      0.006 -0.011
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.5            who_v1      0.492      0.493 -0.001 0.002  7/22   no gap    0.005      0.023 -0.006
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.9            who_v1      0.480      0.493 -0.013 0.004  4/22 b better   -0.002      0.033 -0.021
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.5       gbm_full_im      0.489      0.490 -0.001 0.002 11/22   no gap    0.004      0.014 -0.004
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.9       gbm_full_im      0.481      0.490 -0.009 0.002  4/22 b better    0.002      0.011 -0.014
```


### Layer `dead_v1`

Top 8 by spellF1 (mean per source, then over sources): `who_v1+tcn_full_imbnd0.5` 0.553, `tcn_full_all` 0.553, `who_v1+tcn_full_imbnd0.9` 0.552, `tcn_full_im+bnd0.5` 0.552, `tcn_full_im` 0.550, `tcn_full_im+prod0.5` 0.550, `gbm_full_im+tcn_full_imbnd0.5` 0.549, `tcn_full_im+override` 0.549

```
                        question                             a                 b  a_spellF1  b_spellF1   mean    se  wins  verdict  d_idsse  d_metrica  d_pff
              Q1 rules vs who_v1                          rule            who_v1      0.392      0.504 -0.112 0.005  0/22 b better   -0.124     -0.126 -0.107
              Q1 rules vs who_v1                     rule_hard            who_v1      0.390      0.504 -0.113 0.005  0/22 b better   -0.125     -0.124 -0.109
              Q1 rules vs who_v1                      rule_seg            who_v1      0.397      0.504 -0.107 0.005  0/22 b better   -0.115     -0.117 -0.103
              Q1 rules vs who_v1                          hold            who_v1      0.249      0.504 -0.254 0.013  0/22 b better   -0.216     -0.365 -0.250
              Q1 rules vs who_v1                      smooth50            who_v1      0.346      0.504 -0.157 0.010  0/22 b better   -0.159     -0.246 -0.146
              Q1 rules vs who_v1                       nearest            who_v1      0.325      0.504 -0.178 0.008  0/22 b better   -0.170     -0.253 -0.171
                     Q2 features                    gbm_who_im            who_v1      0.503      0.504 -0.001 0.002 12/22   no gap    0.005     -0.003 -0.002
                     Q2 features                   gbm_full_im        gbm_who_im      0.498      0.503 -0.005 0.002  5/22 b better   -0.011      0.012 -0.005
                     Q2 features                   gbm_full_im            who_v1      0.498      0.504 -0.006 0.002  7/22 b better   -0.006      0.009 -0.007
                     Q2 features                  gbm_stack_im       gbm_full_im      0.503      0.498  0.005 0.002 14/22 a better    0.002      0.013  0.005
  Q3 training sources (gbm full)                gbm_full_imskc       gbm_full_im      0.436      0.498 -0.062 0.006  0/22 b better   -0.082     -0.098 -0.052
  Q3 training sources (gbm full)                gbm_full_impff       gbm_full_im      0.490      0.498 -0.009 0.003  7/22 b better   -0.020     -0.028 -0.003
  Q3 training sources (gbm full)                  gbm_full_all       gbm_full_im      0.447      0.498 -0.051 0.006  0/22 b better   -0.081     -0.093 -0.038
                  Q4 NN vs trees                   tcn_full_im       gbm_full_im      0.511      0.498  0.013 0.004 17/22 a better    0.020      0.006  0.012
                  Q4 NN vs trees                   tcn_full_im            who_v1      0.511      0.504  0.007 0.003 16/22 a better    0.014      0.015  0.005
                  Q4 NN vs trees                   mlp_full_im       gbm_full_im      0.507      0.498  0.009 0.002 18/22 a better    0.002      0.004  0.011
                  Q4 NN vs trees                  tcn_full_all       tcn_full_im      0.522      0.511  0.011 0.004 16/22 a better   -0.008     -0.002  0.017
                  Q4 NN vs trees                  mlp_full_all       mlp_full_im      0.461      0.507 -0.046 0.005  1/22 b better   -0.059     -0.072 -0.040
                  Q4 NN vs trees                  tcn_full_all       gbm_full_im      0.522      0.498  0.024 0.006 16/22 a better    0.012      0.004  0.029
 Q5 probabilities vs hard labels                        who_v1       who_v1_hard      0.504      0.478  0.026 0.005 17/22 a better   -0.004      0.014  0.035
 Q5 probabilities vs hard labels                   gbm_full_im  gbm_full_im_hard      0.498      0.476  0.022 0.005 18/22 a better   -0.011      0.014  0.032
 Q5 probabilities vs hard labels                  gbm_stack_im gbm_stack_im_hard      0.503      0.483  0.020 0.005 17/22 a better   -0.011      0.020  0.028
 Q5 probabilities vs hard labels                   mlp_full_im  mlp_full_im_hard      0.507      0.477  0.030 0.006 17/22 a better   -0.009      0.024  0.041
 Q5 probabilities vs hard labels                   tcn_full_im  tcn_full_im_hard      0.511      0.493  0.018 0.004 18/22 a better    0.000     -0.003  0.025
 Q5 probabilities vs hard labels                          rule         rule_hard      0.392      0.390  0.001 0.001 13/22   no gap    0.001     -0.002  0.002
         Q6 rule x model hybrids              gbm_full_im+prod       gbm_full_im      0.461      0.498 -0.037 0.004  0/22 b better   -0.046     -0.038 -0.035
         Q6 rule x model hybrids           gbm_full_im+prod0.5       gbm_full_im      0.488      0.498 -0.011 0.003  4/22 b better   -0.010     -0.009 -0.011
         Q6 rule x model hybrids          gbm_full_im+override       gbm_full_im      0.497      0.498 -0.001 0.001  5/22   no gap    0.000      0.000 -0.001
         Q6 rule x model hybrids              gbm_full_im+veto       gbm_full_im      0.485      0.498 -0.014 0.002  3/22 b better   -0.017     -0.034 -0.010
         Q6 rule x model hybrids              tcn_full_im+prod       tcn_full_im      0.497      0.511 -0.014 0.002  2/22 b better   -0.012     -0.008 -0.015
         Q6 rule x model hybrids           tcn_full_im+prod0.5       tcn_full_im      0.508      0.511 -0.003 0.001  9/22 b better   -0.001      0.005 -0.005
         Q6 rule x model hybrids          tcn_full_im+override       tcn_full_im      0.510      0.511 -0.001 0.000  6/22 b better    0.001     -0.003 -0.002
         Q6 rule x model hybrids              tcn_full_im+veto       tcn_full_im      0.498      0.511 -0.013 0.003  3/22 b better   -0.027     -0.027 -0.008
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.5       tcn_full_im      0.509      0.511 -0.002 0.002  8/22   no gap   -0.003      0.011 -0.004
Q7 TCN boundaries in the decoder            tcn_full_im+bnd0.9       tcn_full_im      0.501      0.511 -0.010 0.003  5/22 b better   -0.005      0.009 -0.014
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.5            who_v1      0.510      0.504  0.006 0.002 14/22 a better    0.016      0.024  0.001
Q7 TCN boundaries in the decoder      who_v1+tcn_full_imbnd0.9            who_v1      0.501      0.504 -0.003 0.004  7/22   no gap    0.016      0.035 -0.012
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.5       gbm_full_im      0.504      0.498  0.006 0.002 14/22 a better    0.010      0.023  0.003
Q7 TCN boundaries in the decoder gbm_full_im+tcn_full_imbnd0.9       gbm_full_im      0.498      0.498  0.000 0.003  9/22   no gap    0.017      0.019 -0.006
```


### Decoder matters: hmm − raw (oracle layer)

```
              method   hmm   raw  hmm-raw
        tcn_full_all 0.652 0.598    0.053
         mlp_full_im 0.618 0.257    0.360
        gbm_stack_im 0.617 0.239    0.379
              who_v1 0.611 0.228    0.384
          gbm_who_im 0.611 0.229    0.382
         tcn_full_im 0.611 0.565    0.046
tcn_full_im+override 0.609 0.534    0.075
         gbm_full_im 0.608 0.240    0.367
      gbm_full_impff 0.607 0.281    0.326
gbm_full_im+override 0.605 0.237    0.368
 tcn_full_im+prod0.5 0.604 0.529    0.075
    tcn_full_im+prod 0.593 0.511    0.082
 gbm_full_im+prod0.5 0.591 0.263    0.327
        mlp_full_all 0.588 0.285    0.303
    tcn_full_im_hard 0.579 0.565    0.014
        gbm_full_all 0.573 0.279    0.294
    gbm_full_im+prod 0.562 0.283    0.279
      gbm_full_imskc 0.561 0.273    0.288
   gbm_stack_im_hard 0.510 0.239    0.271
    gbm_full_im_hard 0.505 0.240    0.264
         who_v1_hard 0.502 0.228    0.274
    mlp_full_im_hard 0.501 0.257    0.244
            rule_seg 0.492 0.489    0.003
                rule 0.481 0.477    0.004
           rule_hard 0.478 0.477    0.002
            smooth50 0.404 0.353    0.051
             nearest 0.383 0.050    0.333
                hold 0.280 0.190    0.091
```


### Per alive frame (calibration)

```
                        acc    auc  logloss  brier    ece
method                                                   
tcn_full_all          0.913  0.968    0.229  0.065  0.032
gbm_full_impff        0.900  0.963    0.244  0.073  0.029
mlp_full_all          0.901  0.963    0.244  0.072  0.032
gbm_stack_im          0.900  0.962    0.246  0.073  0.029
who_v1                0.898  0.961    0.248  0.074  0.030
gbm_full_all          0.898  0.962    0.249  0.074  0.036
gbm_full_im           0.899  0.961    0.249  0.074  0.030
gbm_who_im            0.898  0.961    0.250  0.075  0.030
mlp_full_im           0.897  0.961    0.250  0.075  0.029
gbm_full_im+override  0.898  0.960    0.253  0.075  0.030
gbm_full_imskc        0.894  0.958    0.263  0.078  0.047
gbm_full_im+prod0.5   0.894  0.959    0.269  0.079  0.043
tcn_full_im           0.902  0.956    0.304  0.082  0.066
tcn_full_im+override  0.901  0.949    0.307  0.083  0.063
tcn_full_im+prod0.5   0.900  0.956    0.311  0.084  0.071
gbm_full_im+prod      0.883  0.955    0.315  0.091  0.069
tcn_full_im+prod      0.898  0.954    0.327  0.088  0.079
tcn_full_im_hard      0.902  0.900    0.403  0.095  0.078
rule                  0.841  0.884    0.408  0.125  0.041
gbm_stack_im_hard     0.900  0.898    0.411  0.097  0.080
gbm_full_im_hard      0.899  0.896    0.415  0.098  0.081
who_v1_hard           0.898  0.896    0.416  0.098  0.082
mlp_full_im_hard      0.897  0.895    0.420  0.099  0.083
nearest               0.728  0.809    0.560  0.182  0.067
rule_hard             0.841  0.838    0.639  0.153  0.139
smooth50              0.803  0.816    0.718  0.177  0.155
hold                  0.789  0.787    0.838  0.210  0.210
```

Accuracy where the ball is NOT tracked: `tcn_full_all` 0.863, `tcn_full_im_hard` 0.844, `tcn_full_im` 0.844, `tcn_full_im+override` 0.844, `mlp_full_all` 0.843, `tcn_full_im+prod0.5` 0.840, `gbm_full_impff` 0.840, `gbm_full_all` 0.839, `gbm_stack_im` 0.838, `gbm_stack_im_hard` 0.838
