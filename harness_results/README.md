# harness_results — every score we decided on

Aggregates on **other games only** (IDSSE, Metrica, PFF, SkillCorner, synthetic), scored with the official metric.
Nothing in here comes from Barça–Athletic. Each folder is one stage of the pipeline; the experiment numbers
point to `docs/EXPERIMENTS.md`.

| folder | what | EXPERIMENTS |
|---|---|---|
| `0_baseline/` | v0 (nearest team + speed-threshold stoppages, HMM) on the clean bundles and on IDSSE with ATD noise | #4–#5 |
| `1_tracking/` | the tracklet cleaner: wrong-team repair (`teamfix_*`), cleaner on/off under each possession model | #21, #23, #39 |
| `2_ball/` | missing-ball imputation: interpolation vs transformers, gap buckets, downstream spellF1; `train/` = training curves | #10–#18 |
| `3_dead_ball/` | dead_v1, the mqf.alive suite (GBM / MLP / TCN / ASFormer-lite, rules, Smart Tagging prior), sensitivity, Italo's inputs (i8); `train/` = training curves (+ v8) | #42, #46 |
| `4_possession/` | who_v1 / who_v2, the owner suite (rules, features, sources, NNs), TCN seed check, pass-direction check; `train/`; `who_lab/` = the who_lab.py runs | #34–#41 |
| `5_final_check/` | FINAL vs FINAL_V1 on the 11 validation games × 6 noise columns, and the runs exported for the decoder suite | #43, #45, #47 |
| `6_decoder/` | HMM cost sweep, the fusion suite (8 decoder families, nested leave-one-game-out), the stoppage decoder (sampled + grid) | #36, #47, #48 |
| `fusion_preds/` | the decoder suite's inputs (`*.npz`, gitignored: regenerate with `python tasks.py fusion-export`) | #47 |

New runs write to the top of this folder; file them into their stage before committing.
