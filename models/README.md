# models — trained on other games only

The submitted recipe (`FINAL` in `mqf/make_submission.py`) reads these. Every one was trained on IDSSE + Metrica
(+ PFF where noted) training games of `splits.json`, never on Barça–Athletic.

| file | stage | what | trained on |
|---|---|---|---|
| `ball_B_off_v5.pt` (+ `.json`) | 1 · missing ball | axial agents × time transformer, residual over interpolation (`mqf/ball`, frozen setting `impute.FROZEN`) | IDSSE 5 + Metrica 2 + PFF 56 |
| `alive_asf_full_impff_s0/1/2.pt` | 2 · dead ball | ASFormer-lite, 3 seeds (`mqf/alive`) | IDSSE + Metrica + PFF |
| `deadball_v8.joblib` | 2 · dead ball | Italo's gradient boosting with ball inputs (`mqf/deadball`) | IDSSE + Metrica + PFF |
| `owner_tcn_full_impff_s0/1/2.pt` | 3 · possession | temporal CNN + H↔A boundary head, 3 seeds (`mqf/owner`) | IDSSE + Metrica + PFF |
| `dead_v1.joblib`, `who_v1.joblib` | reference | the 16 Nov recipe `FINAL_V1` (final_check compares against it; tests use them) | IDSSE 5 + Metrica 2 |

`old/` keeps every other model we trained, by family (`ball/`, `dead_ball/`, `possession/`). Nothing in the
submission reads them; `docs/EXPERIMENTS.md` says why each was not adopted.
