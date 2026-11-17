# Plan — IDSSE, Metrica and SkillCorner as "BAR-ATH-shaped" tactical games

*10 Nov 2026 · Més que un Frame · scope agreed: tactical definition only, no Smart Tagging*

## Context
The harness has one external source (IDSSE, in memory, via floodlight download) and decisions need
more matches whose inputs look like the competition's tactical ATD. Goal: every external game
becomes a bundle in the **exact tactical data definition** (DATA.md), with a GT CSV in the
official shape, so that `atd.load`, `make_submission.run`, `mqf.profile`, kloppy and the official
scorer run on it unchanged. Then the noise model is fitted so the bundles' aggregate statistics
match the real tactical ATD — without the real ATD ever reaching me.

**On "read the data folder":** I won't open `data/` (hard rule 1, T&Cs). I don't need it: the fps
and format are published (DATA.md, games.py — read). Everything beyond the definition (gap
lengths, jitter, tracklet life…) comes from the aggregate block `python -m mqf.profile` prints.

## Target: the tactical definition (DATA.md)
| item | definition |
|---|---|
| timeline | one record per video frame, 25 fps; ATD `frame_id` 1…n, GT/submission rows 0…n−1 |
| layout | pre-match · half 1 · halftime · half 2 (opens with a long pre-kickoff stoppage) · post-match; only the half blocks are graded |
| ATD raw | `fid:x,y,s;…(46 slots):bx,by,bs`, normalised 0–1 by pitch L×W, `NaN` = missing, s in m/s |
| ATD meta | 2 `<Team>` (first = home), 46 `<Player>` "Track k" with `teamId`, FrameRate 25 |
| noise (published) | ball seen 59.3 % of graded frames, bursty, worst in duels/crowds/aerials; ~20.5 objects in play, sometimes > 22; slots swap identity; no IDs |
| GT CSV | `frame,label`, H/A/D on graded frames, X elsewhere; dead flag + owning team from tracking; loose ball stays with last team |

## What the three datasets are (read today)
| | IDSSE | Metrica sample | SkillCorner open data |
|---|---|---|---|
| games | 7 (2 Bundesliga + 5 2. Bundesliga), CC-BY 4.0 | 3 | 20 A-League 2024/25, MIT |
| files | raw DFL XML locally: matchinformation, events, positions (349–419 MB each) | G1/G2 CSV; **G3 FIFA EPTS** (same family as the ATD: meta XML + `fid:x,y;…:bx,by`, no speed) + JSON events | per match `match.json`, `tracking_extrapolated.jsonl`, `phases_of_play.csv`, `dynamic_events.csv` |
| fps / pitch | 25 · 105×68 (matchinformation) | 25 · 105×68, 0–1 coords, origin top-left | **10** · 105×68 (match.json), centred metres |
| players | all, with IDs, subs NaN off pitch | all, with IDs | detected + extrapolated (`is_detected`) |
| ball | full | NaN on 39 % of G1 frames — ~74 % inside event-derived stoppages vs ~14 % in play | `is_detected`, z |
| labels | **per frame**: BallStatus + BallPossession | none — from manual, synced events | per-frame `possession.group`; phases exist **only while ball in play** |
| blocker | — | — | **all 20 `*_tracking_extrapolated.jsonl` are Git LFS pointers (133 B, real ≈ 90 MB)** → `git lfs pull` |

## Steps

### 0. Hygiene (before any code)
- `external_data/` is **not** gitignored in this folder (IDSSE alone ≈ 2.6 GB) → add it, plus the
  derived `external/` dir, to `.gitignore`.
- `mqf/config.py`: `EXTERNAL_RAW_DIR` (env `MQF_EXTERNAL_RAW`, default `<repo>/external_data`) for
  inputs, `EXTERNAL_DIR` stays for derived caches/bundles.
- Fix "7 Bundesliga" → "2 Bundesliga + 5 2. Bundesliga" (README, CONTEXT, `harness/idsse.py`).
- Confirm the working copy: this folder has `JS/mqf` + `JS/docs` but no `tests/`, `Makefile`,
  `bootstrap.py`, `requirements*` — I need those for `make test`.

### 1. Read → clean match (one loader per source, same contract as the harness)
Contract (already used by `harness/noise.corrupt`): `players (n,P,2)` m from the top-left corner,
`team_of_player`, `ball (n,2)`, `labels`, `halves`, `pitch`, `fps=25`, team names.
- **IDSSE** — `floodlight.io.dfl.read_position_data_xml` on the local files (no download; the
  cloud got 403 from figshare), then the existing `idsse.from_floodlight`. Labels: D = BallStatus 0,
  owner = BallPossession; alive with no owner → previous owner (today it is X and splits runs).
- **Metrica** — G3 via the EPTS meta (halves from `first_half_start/end…`), G1/G2 via the CSVs.
  Labels from events: dead = BALL OUT / shot out or goal / FAULT RECEIVED / offside → next
  SET PIECE start, dead start snapped to the tracking (ball lost/off-pitch within ±1 s). Owner =
  team of controlling events (PASS, CARRY, SET PIECE, SHOT, RECOVERY); CHALLENGE/aerials don't
  switch; switch at the opponent's RECOVERY or first controlled action.
- **SkillCorner** (after LFS pull) — alive = union of `phases_of_play` intervals; owner =
  `possession.group` forward-filled inside alive, fallback the phase's `team_in_possession_id`.
  10 → 25 fps by time: linear inside gaps ≤ 0.5 s, labels nearest-previous. Players: detected +
  extrapolated form the clean layer (the tactical camera sees the whole pitch); undetected ball →
  NaN (enters the noise model as `base_missing`).
- **Common timeline shaping** (`timeline.py`): ungraded pre-match / halftime / post-match padding
  with empty tracking; a synthetic pre-kickoff stoppage prepended to half 2 (length drawn from
  60–150 s per game, seeded; labelled D) so the stoppage detector is exercised. Flagged synthetic.
- **Label QA per game** (printed, other games so fully inspectable): no X inside graded, spells
  per half, mean duration, D share, share of spells < 3 s. Compared across sources and football
  norms — not against the public 5 minutes.

### 2. Compare with the tactical data
- Real side: you run `python -m mqf.profile --source tactical` and paste the block. I keep it in
  `external/targets/tactical_profile.json` (gitignored, never committed). The published DATA.md
  numbers are the hard targets in code, cited.
- External side: the same `profile_tracking` on (a) the clean game and (b) the noised bundle.
  `python -m mqf.external.compare --source idsse` prints `metric | tactical | clean | noised | in band`.
- Metrics: ball seen share, gaps/min, gap-length quantiles and buckets, where the ball goes missing
  (players moving / mid / still), ball speed p50/p90/p99, jumps > 40 m/s, off-pitch share;
  objects/frame mean/p5/p95, home/away, share > 22 and < 18; tracklets/min and length quantiles;
  speed field vs position speed; jitter (2nd difference); stillness quantiles.

### 3. Adjust so it fits the tactical data
- Per source `NoiseConfig` fit (their clean inputs differ: Metrica already drops the ball at
  stoppages, SkillCorner is 10-fps-smooth), deterministic coordinate search on graded frames,
  weighted distance to the targets. `ball_missing_target` stays the published 0.41.
- New knobs only where the compare shows a gap no existing knob closes, default off so earlier
  harness numbers stay comparable: ball loss at stillness/off-pitch, ball false detections
  (jumps), correlated multi-player dropout in crowds.
- Output `external/<source>/noise_fit.json` = centre + band. Decisions must hold at low / centre /
  high. CLAUDE.md line "don't calibrate on the real match" → "noise centred on profile
  aggregates; decisions must survive the band".

### 4. Write the bundle (tactical definition) and register it
`external/bundles/<SRC>-<ID>/` — names avoid the patterns reserved for the real data
(`*_atd.*`, `*_homography.csv`, `*_smart_tagging.xml`), like `viewer/demo.py`, so I can inspect them:
- `<GAME>_tracking_meta.xml` — kloppy-compatible, modelled on Metrica G3's meta (FrameRate, pitch,
  2 teams, 46 slots, channels x,y,s)
- `<GAME>_tracking_raw.txt` — ATD raw lines (vectorise `synthetic._write_raw`)
- `<GAME>_possession_gt.csv` — official shape
- `<GAME>_game.json` — games.py-like: frames, fps, pitch, half blocks, names, label provenance,
  noise config + seed, converter hash
`mqf.external.registry.game_spec(name) -> GameSpec`; `make_submission --external <GAME>`;
`mqf.profile --game <GAME>`.

### 5. Repeat for all datasets
IDSSE (best labels, adapter exists) → Metrica (G3 first: native EPTS) → SkillCorner (most surgery,
automatic labels; **droppable** — if its label QA looks off, keep it for profiling only).

## Where things run
IDSSE conversion runs on your PC (one positions file is 419 MB, over the 400 MB bridge limit; I
can't run commands there) — you paste the printed QA/compare tables, all other-game numbers.
Metrica and SkillCorner can be staged and converted here. Code + tests are written on fixtures here.

## Files
- new `mqf/external/`: `idsse.py` (moved from `harness/idsse.py`, which stays as a re-export),
  `metrica.py`, `skillcorner.py`, `timeline.py`, `bundle.py`, `registry.py`, `compare.py`, `fit.py`,
  `evaluate.py`
- edit: `config.py`, `harness/noise.py` (optional knobs), `harness/run.py` (`--source
  metrica|skillcorner`, `--noise-preset low|centre|high`), `make_submission.py` (`--external`),
  `profile.py` (`--game`), `synthetic.py` (vectorised writer, kloppy-compatible meta),
  `requirements-harness.txt` (kloppy for the check), `.gitignore`, `TECHNICAL.md` §4, `EXPERIMENTS.md`,
  `CLAUDE.md` (calibration line)

## Verification
- `tests/test_external.py`: bundle → `atd.load` round-trip (≤ 1e-5 normalised); frame_id 1…n
  contiguous; GT passes the official validator shape; X only outside halves; Metrica event → label
  rules and SkillCorner phases → alive and 10 → 25 resample on toy tables; IDSSE conversion unchanged.
- `make test` green.
- `kloppy.metrica.load_tracking_epts` loads one bundle per source.
- Per game: `make_submission --external` → official validator OK → `mqf.external.evaluate`
  (official metric + match rate by true length) → `harness_results/external_<tag>.json`.
- In-memory harness and bundle path agree within seed noise for the same config/seed.
- Log in `EXPERIMENTS.md`; TECHNICAL.md §4 rows (source, labeller, amount) for all three.

## Timeline
Tue 10: step 0 + IDSSE · Wed 11: Metrica · Thu 12: SkillCorner + noise fit + band runs ·
Fri 13 on: decoder work on ~30 games.
