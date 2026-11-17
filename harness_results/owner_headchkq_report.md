# Heading check `headchkq` (other games only)

Pass-direction columns (feats.HEADING_COLS, ported from who/who_lab.py) through the #38 protocol. Paired Δ spellF1 = a − b per game × noise seed; cells: mean (se) wins · per source. Rule (#38): layer dead_v1, mean > 2 se in every condition and wins on more than half the runs.


## GBM arm — #38's conditions

Sanity: `bar` who_v1 0.555


### Layer `dead_v1`

| comparison | bar |
|---|---|
| gbm_full_im vs who_v1 | -0.0355 (0.0046) 0/3 · idsse -0.028 metrica -0.034 pff -0.044 |
| gbm_stack_im vs who_v1 | -0.0480 (0.0049) 0/3 · idsse -0.050 metrica -0.039 pff -0.055 |
| gbm_full_heading_im vs who_v1 | -0.0308 (0.0036) 0/3 · idsse -0.035 metrica -0.024 pff -0.034 |
| gbm_stack_heading_im vs who_v1 | -0.0464 (0.0104) 0/3 · idsse -0.041 metrica -0.032 pff -0.067 |
| blend vs who_v1 | -0.1512 (0.0284) 0/3 · idsse -0.107 metrica -0.204 pff -0.143 |
| nearest vs who_v1 | -0.2396 (0.0420) 0/3 · idsse -0.201 metrica -0.324 pff -0.194 |
| heading effect: gbm_full_heading_im - gbm_full_im | +0.0047 (0.0058) 2/3 · idsse -0.007 metrica +0.011 pff +0.010 |
| heading effect: gbm_stack_heading_im - gbm_stack_im | +0.0016 (0.0064) 2/3 · idsse +0.009 metrica +0.007 pff -0.011 |
| blend - nearest (who_lab's comparison) | +0.0883 (0.0198) 3/3 · idsse +0.094 metrica +0.119 pff +0.052 |

### Layer `oracle`

| comparison | bar |
|---|---|
| gbm_full_im vs who_v1 | -0.0438 (0.0070) 0/3 · idsse -0.031 metrica -0.055 pff -0.045 |
| gbm_stack_im vs who_v1 | -0.0604 (0.0043) 0/3 · idsse -0.063 metrica -0.067 pff -0.052 |
| gbm_full_heading_im vs who_v1 | -0.0420 (0.0076) 0/3 · idsse -0.028 metrica -0.044 pff -0.054 |
| gbm_stack_heading_im vs who_v1 | -0.0576 (0.0078) 0/3 · idsse -0.044 metrica -0.058 pff -0.071 |
| blend vs who_v1 | -0.1916 (0.0471) 0/3 · idsse -0.098 metrica -0.242 pff -0.235 |
| nearest vs who_v1 | -0.3126 (0.0432) 0/3 · idsse -0.247 metrica -0.394 pff -0.296 |
| heading effect: gbm_full_heading_im - gbm_full_im | +0.0018 (0.0058) 2/3 · idsse +0.003 metrica +0.011 pff -0.009 |
| heading effect: gbm_stack_heading_im - gbm_stack_im | +0.0028 (0.0112) 2/3 · idsse +0.019 metrica +0.008 pff -0.019 |
| blend - nearest (who_lab's comparison) | +0.1210 (0.0298) 3/3 · idsse +0.150 metrica +0.152 pff +0.061 |

## Verdict (layer dead_v1, every condition)

- fail  gbm_full_heading_im vs who_v1  → replaces who_v1 (#38 rule)
- fail  gbm_stack_heading_im vs who_v1  → replaces who_v1 (#38 rule)
- fail  blend vs who_v1  → replaces who_v1 (#38 rule)
- fail  heading effect: gbm_full_heading_im - gbm_full_im  → the columns add something to the trees
- fail  heading effect: gbm_stack_heading_im - gbm_stack_im  → the columns add something to the trees

No heading GBM passes against who_v1: the #38 decision stands for the trees. The columns can only reach the submission through the TCN: run with --tcn.
