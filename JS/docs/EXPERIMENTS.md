# Experiment log

Decisions are made on the harness (other games × noise seeds), never on the public 5 minutes.
Noise: spellF1 ±0.04 on one match; harness noise seeds alone ≈ ±0.06 → average matches × seeds.
Record aggregates only (never frame-level outputs of the real match).

| # | date | failure mode / hypothesis | change | harness spellF1 (source, runs) | synthetic file spellF1 / macroF1 | public check ok? | decision |
|---|---|---|---|---|---|---|---|
| 0 | 2026-11-02 | reference | official `baseline_smooth50` | — | — | — | floor 0.2433 (organisers, full match) |
| 1 | 2026-11-02 | flicker → interval model | v0 HMM (`decode.Params()` defaults) + Smart Tagging votes | 0.667 ± 0.062 (synthetic, 3 matches × 3 seeds) | 0.602 / 0.719 | pending | keep as scaffold; IDSSE next |
| 2 | 2026-11-02 | harness blind to stillness: white 0.25 m jitter differentiated into ~11–16 m/s player speed (stoppages read 11 m/s), so no stillness-based D cue could ever fire on the harness | noise: AR(1) jitter (`jitter_corr_s`=0.5) + tracker speed = centred 0.4 s difference per physical track; synthetic: 1 s smoothing of player paths (in play ~3.3 m/s, stoppages ~1.6 m/s); stale `synthetic_data/` now regenerated on generator change | 0.728 ± 0.038, macroF1 0.720 ± 0.013, alive_dead_f1 0.887, spells 92/84 (synthetic, 3 matches × 3 seeds). Generator changed → **not comparable to #1** (re-run of #1 code: 0.667 ± 0.062, spells 111/84) | 0.704 / 0.784 (with tags) | — | keep; it is a harness fix, not a model change. Stoppages still under-predicted (D 12 % vs 30 % on SYN-101): `still_speed`=1.3 m/s sits below stoppage speed → experiment 1 |
| 3 | 2026-11-02 | IDSSE path at full-match scale | dry run: synthetic match pushed through floodlight-shaped objects → `idsse.from_floodlight` → noise → pipeline → official metric, in the harness venv (pandas 2.3.3) | runs, 4.6 s per match × seed after load; real IDSSE download blocked from the cloud sandbox (figshare 403) → run locally | — | — | run `--source idsse` locally (see KICKOFF answers) |
