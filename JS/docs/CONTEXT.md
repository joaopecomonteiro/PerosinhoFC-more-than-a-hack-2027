# Més que un Frame — Full Project Context

*Robust possession spells from uncorrected tracking · More than a Hack 2027 (FC Barcelona × Metrica Sports × Mobile World Capital Barcelona)*

Revised 2 Nov 2026 after a full read of the official repo (README, CHALLENGE, DATA, EVALUATION,
OVERVIEW, RULES, SUBMISSION, FAQ, `data/`, `evaluation/` incl. `possession_metrics.py`, `starter/`
incl. the three notebooks). This is **our own repo**; the official one is cloned read-only at a
pinned commit into `vendor/mtah2027` (`python bootstrap.py`) and is **authoritative**: if this
document disagrees with it, it wins. `python bootstrap.py --check-updates` lists organiser commits
newer than our pin — the rules on annotation were clarified twice in late October, so check weekly.

---

## 1. The competition

| | |
|---|---|
| Organisers | Metrica Sports, FC Barcelona (Barça Innovation Hub), Mobile World Capital Barcelona — Talent Arena, Barcelona |
| Edition | 2nd (2027). 2026 was an open GenAI/LLM jury hackathon; **2027 is one task, one number, no jury** |
| Team size | **3–5.** Teams with < 3 are not eligible for the Innovation Challenge or the Final |
| Registration | until **10 Nov 2026** (form + T&Cs → data download link) |
| Contact | hackathon.talentarena@mobileworldcapital.com · data/metric/rules questions → GitHub issue on the official repo |

1. **Initial Challenge (now)** — data released 20 Oct, **deadline 17 Nov 2026**, results before 26 Nov. **Top 20 advance.**
2. **Innovation Challenge** — 26 Nov 2026 → 14 Jan 2027: 10 matches with tracking + event data, open design challenges.
3. **Final pitch** — 2 Mar 2027, Talent Arena: 5 teams, professional jury.

---

## 2. Phase 1 task

One CSV `frame,label`, **199,319 rows** (frames 0…199318, in order), one per tactical-video frame:
`H` home (**Barcelona**) has the ball, alive · `A` away (**Athletic**) · `D` dead ball · `X` abstain.
Match: FC Barcelona vs Athletic Club, LaLiga 2024-25 MD2.

Graded: **11,185–90,651** and **110,808–190,197** (158,857 frames; `games.half_video_blocks`).
The 2nd half is graded ~2 min before its kickoff → that stretch is all `D` (see §5 on how we may handle it).

### What the labels mean — read this before designing anything
- The GT is **derived mechanically from Metrica's corrected tracking**: `D` where the corrected
  tracking flags the ball dead, otherwise the team the corrected tracking names as owner. No human
  watched video, no event data, nothing smoothed afterwards.
  → **Our target is a tracking-based ownership signal, not an analyst's notion of possession.**
  The external data we validate on should carry the same kind of label (per-frame alive flag +
  owning team from clean tracking), not possessions reconstructed from event logs.
- Loose/contested ball → stays with the last team. Change = opponent gains *controlled*
  possession; touches, deflections, aerial duels without control don't change it.
- A possession = maximal H or A run. `H D H` = two possessions.
- The public 5 minutes are the reference for borderline cases: 17 possessions, mean 14.2 s,
  21 segments, H 3,454 / A 2,577 / D 1,469 frames. Use them to understand semantics and check
  the pipeline — **never** to fit or choose anything.

---

## 3. Scoring — what `possession_metrics.py` actually does

- **spellF1 (ranking)**: possessions (maximal H/A runs, built *inside each half*, after
  forward-filling `X`) are matched **one-to-one, greedily by IoU, same team only**; F1 at IoU
  0.3/0.4/0.5/0.6/0.7, averaged. Constant guess = 0. Noise band ≈ 0.04 on the full match.
- **macroF1 (1st tie-break)**: mean per-class F1 over H/A/D on graded frames; `X` = miss.
  Further tie-breaks: transition F1 at ±25 frames, then mean |latency|.

Consequences we design around:
1. **Every possession counts once, whatever its length.** A 2 s regain is worth as much as a 40 s
   build-up. Short spells are where the points are, and where tracking is worst.
2. **Splitting costs.** A spurious `D` or opponent blip inside a true possession leaves at most one
   piece matchable (and only if it still covers enough IoU) and adds false positives.
3. **Merging costs too.** Missing a short stoppage in `H D H`, or a short opponent spell in
   `H A H`, merges spells: the merged prediction rarely reaches IoU ≥ 0.5 against either truth.
   → Hysteresis is not free; the switch/duration costs are a real trade-off that must be set on
   held-out games with the official metric, broken down by true possession length.
4. **Boundaries:** 1 frame late is free; for a 14 s possession 1 s of lag still matches at 0.7,
   for a 3 s possession it does not. Precision matters most on short spells.
5. **`X` never helps.** spellF1 forward-fills it (same as repeating the last label), macroF1
   charges it. Our decoder labels every graded frame; a test enforces it.

Diagnostics the official scorer gives for free (`--no-detail` off): `poss`/`dur` vs truth,
`transF1` at ±12 and ±25 frames (gap = lag detector), `lat_f`, `frag` (jitter detector),
`alive_dead_f1`, `team_acc_both_alive`, and `evaluation/reports/<name>_transitions.csv` (every
transition matched/missed/spurious). Use them freely on synthetic and external games; on the public
window only to confirm correctness.

| official baselines (full match) | macroF1 | spellF1 |
|---|---|---|
| raw nearest player | 0.6532 | 0.0486 |
| smooth ±25 | 0.6752 | 0.1996 |
| **smooth ±50 = floor** | 0.6724 | **0.2433** |
| smooth ±125 | 0.6356 | 0.2242 |

On the public window ±125 looks best (0.42) and is worse on the full match: 17 possessions can't rank anything.
**No leaderboard, no feedback before the deadline.**

---

## 4. Data (DATA.md, `data/README.md`)

```
data/tactical/  BAR-ATH_tactical.mp4 | _atd.xml | _atd.txt | _homography.csv | _smart_tagging.xml
data/broadcast/ BAR-ATH_broadcast.mp4 | _atd.xml | _atd.txt | _homography.csv
```
(or anywhere, via `MTAH_DATA_DIR`; `games.asset_paths()` builds every path.)

| | tactical (reference) | broadcast (optional) |
|---|---|---|
| frames | 199,319 @ 25 fps, 1920×1080 | 164,488, own timeline, **not synced** |
| homography usable | 100% | 65.8% |
| Smart Tagging | 638 instances | — |
| track slots | 46 | 51 |
| ball visible | 56.9% (59.3% of graded) | 50.5% |

- **ATD** (FIFA/EPTS): `<frame_id>:<x,y,s>;…:<bx,by,bs>`, normalised 0–1, NaN = missing,
  `video_frame = frame_id − 1`, pitch 105.3 × 68.0 m (metadata says 105 × 68). 46 tracklet slots,
  team per slot in the meta XML (first `<Team>` = home), **no player identities**, mean 20.5
  objects in play (sometimes > 22: duplicates). **Ball missing on 41% of graded frames**,
  concentrated in duels, crowded boxes and aerial balls.
- **Smart Tagging** (SportsCode XML, automatic): seconds × 25 = frame. Team ∈ {Barcelona,
  Atletic Club, **N/A** (36, every throw-in)}. Codes UPPER, values Title. **Intervals are windows,
  not instants**: set pieces open ~3.5 s before the restart (both `Kick Off (Start)` tags contain
  the real kickoff). Phases ≠ possessions. Use for *roughly when / which team*; take the instant
  from tracking.
- **Homography**: 3×3 per frame, normalised pitch → pixels; check `valid`; `reprojection_error`
  lower = better. Mainly for overlaying our labels on the video when debugging.

---

## 5. Rules (RULES.md, FAQ, OVERVIEW + T&Cs)

Allowed: any method/model/LLM; external data **about other games** (incl. hand labels on them);
own detection/tracking from the video (algorithmic); unsupervised/self-supervised learning and
algorithmic pseudo-labels on this game; watching the video to understand and debug.

Not allowed:
1. **Hand annotation of this game** — labels, corrections, *or rules and constants tied to
   specific frames of it*, in any amount, for any purpose.
   → We do **not** hard-code event frames inside the halves (e.g. the 2nd-half kickoff frame,
   which the starter notebook takes "from the ground truth"). The published half blocks are used
   only as the graded window; stoppages, including the long pre-kickoff one, are detected by the
   algorithm.
2. Private GT, or **any human-produced data about this game** (event feeds, corrected tracking…).
3. Public 5 min used for training or tuning.
4. A file the repo can't regenerate **byte-identically**.
5. ⚠️ **T&Cs: no uploading/disclosing the videos, datasets, confidential material "or any
   extracts thereof" to any third-party hosted or cloud LLM.** Claude must never see the real
   files or excerpts. We develop on the synthetic match and share only aggregate numbers.
6. Don't redistribute data, or publish derived data that would reconstruct it. Our `.gitignore`
   blocks `data/`, `cache/`, `results/`, `reports/`, `*.mp4` and the raw asset patterns.

---

## 6. Deliverables (Google Form; link emailed to registrants; Google account needed)

1. `<team>.csv` — exact file validated with `evaluation/validate_submission.py`.
2. **Public repo URL** — our repo. `python bootstrap.py` + one command regenerates the file;
   deps pinned (`requirements.txt`), official repo pinned by commit; no data.
3. **Technical document** `<team>.md` or `.pdf` — we keep `TECHNICAL.md` in the repo and upload a
   copy named after the team.
One response per team, editable; the last one before the deadline counts. Winners may be re-run.

---

## 7. Repository layout

```
mqf/                 our pipeline
  config.py          paths; match constants imported from the official games.py
  atd.py             raw ATD -> arrays (cached in cache/)
  smart_tagging.py   SportsCode XML -> soft H/A evidence + restart windows
  features.py        per-frame features
  decode.py          emission model + 3-state HMM/Viterbi per half
  submission.py      writer + official validator / scorer
  synthetic.py       synthetic match: clean_match() + real file formats, with truth
  make_submission.py CLI
  harness/           validation on other games
    noise.py         clean tracking -> ATD-like (41% bursty ball loss, 46 anonymous slots, dropouts, duplicates)
    idsse.py         IDSSE via floodlight -> clean match with per-frame H/A/D labels
    evaluate.py      our pipeline + OFFICIAL metric + match rate by true possession length
    run.py           CLI, writes harness_results/<source>_<tag>.json
  viewer/            python -m mqf.viewer: video / 2D pitch + GT and model lanes on one playhead
    game.py          ViewerGame contract + adapters (BAR-ATH, synthetic; PFF/IDSSE = one more adapter)
    sync.py          broadcast <-> tactical frame maps from the two ATD files (cached in cache/viewer/)
    server.py        127.0.0.1 server (Range requests for the videos)
    demo.py          synthetic tactical + broadcast videos with known sync, for tests and demos
    static/          the page (vanilla JS, no build step)
tests/               synthetic e2e (official scorer, validator, determinism, contract) + harness
vendor/mtah2027/     official repo, pinned, gitignored (bootstrap.py)
TECHNICAL.md · CLAUDE.md · bootstrap.py · Makefile · requirements*.txt · .claude/settings.json
docs/                CONTEXT.md (this) · EXPERIMENTS.md · KICKOFF_PROMPT.md
```
Gitignored and never shown to an AI: `data/`, `cache/`, `results/`, `reports/`. Safe to inspect:
`synthetic_data/`, `harness_results/`, IDSSE.

Commands (`make help` on Linux/macOS/WSL; on Windows run the python lines from the Makefile):
`python bootstrap.py` · `python -m pytest -q` · `python -m mqf.harness.run --source idsse` ·
`make baseline` · `python -m mqf.make_submission --team x` · `make check TEAM=x` ·
`make reproduce TEAM=x`.

---

## 8. Plan (revised)

**Core idea (unchanged):** per-frame evidence → duration-aware state model (H/A/D) per half →
whole possessions with sharp boundaries.

**What changed after reading the code and docs**
- **Own repo**, official repo pinned in `vendor/` and imported (games.py, validator, scorer,
  metric) — never copied or reimplemented. Pin bumped only after reading organiser changes.
- External validation moves **up** and switches to datasets whose labels look like ours:
  **per-frame alive flag + owning team from clean tracking.** Primary: **IDSSE** (Bassek et al.
  2025; 7 matches 2022/23 — 2 Bundesliga + 5 2. Bundesliga — official DFL tracking + events, ~2.4 GB) —
  `floodlight.io.datasets.IDSSEDataset().get(match_id)` returns positions plus `possession`
  (home/away) and `ballstatus` (alive/dead) codes, i.e. exactly our H/A/D shape. Licence: check the
  paper's terms before use and cite it. Second: SkillCorner open data (possession group per frame —
  verify in the current release). Event-derived possessions (Metrica sample data) only as a
  secondary check.
- Short possessions and short stoppages are first-class targets (every spell counts once).
- No frame-specific constants: stoppage/kickoff detection must be algorithmic.
- The harness now exists (`mqf/harness`): noise model built only from published statistics,
  IDSSE adapter (floodlight API checked against its source), official-metric scoring with match
  rate by possession length. First synthetic run: noise seeds alone move spellF1 by ~±0.06 →
  always average over matches × seeds before believing a gap.

### Priority list
| P | item | failure mode | expected effect |
|---|---|---|---|
| 0 | Set up the repo, reproduce official baselines locally, v0 HMM on real data, first validated submission | — | safety net > 0.2433 |
| 1 | **Validation harness on IDSSE** (code ready: run it, check the adapter on real files, add SkillCorner if time) | no feedback before deadline | makes every later decision measurable |
| 1 | **Stoppage detector** (D): ball off pitch; collective speed collapse; ball static at restart geometry (touchline, corner arc, centre spot, goal area); set-piece formations; Smart Tagging SET PIECES/GOALS windows as priors | merged/split spells around restarts; D class | spellF1 ↑↑, macroF1 ↑ |
| 1 | **Duration-aware decoding**: HSMM or explicit duration priors per state; costs fitted on the harness | flicker vs merged short spells | spellF1 ↑↑ |
| 2 | **Control model for switches**: opponent must hold/control (dwell near ball, ball speed drop, ≥2 same-team touches) before a switch; contested/loose → keep last team | deflections, duels | spellF1 ↑ |
| 2 | **Ball-gap inference**: short gaps interpolated; long gaps → ball location from player configuration (multi-agent set/temporal transformer trained on harness games) | 41% missing ball | spellF1 ↑ on short spells |
| 3 | Learned emissions (set transformer over tracklets + temporal conv) feeding the same decoder; self-training on this game with algorithmic pseudo-labels | everything | only if harness gain > 0.04 |
| 3 | Own ball detector on the tactical video (local only) | ball gaps | high cost; stretch |
| — | Broadcast feed | — | skip unless time remains |

### Validation discipline
- Caveats of the harness: DFL's possession rule is not Metrica's (close in kind, not identical),
  and Smart Tagging has no IDSSE equivalent — its contribution can only be checked on synthetic
  matches, so keep its weight modest.
- `make test` must pass on every change (synthetic, official scorer, determinism, contract).
- Decisions = harness games with the official metric, bootstrap intervals, gap > 0.04 to count.
- Public window = `make check` once per candidate: labels align, file valid, nothing absurd.
- Log every run in `docs/EXPERIMENTS.md`.

---

## 9. Timeline (today Mon 2 Nov → deadline Tue 17 Nov)

| when | goal |
|---|---|
| 2–3 Nov | Team of 3–5 confirmed, **everyone registered (10 Nov)**; create the public repo from the starter; data downloaded; `make baseline` reproduces the 4 baselines; `make submit` + `make check` on v0 |
| 4 Nov | **First form submission** (v0 + repo URL + draft TECHNICAL.md) as a safety net |
| 4–7 Nov | Harness running on all 7 IDSSE matches; stoppage detector |
| 7–11 Nov | Duration-aware decoder + control model, tuned on harness only |
| 11–14 Nov | Ball-gap model / learned emissions if the harness shows > 0.04 |
| 14–16 Nov | Freeze; `bootstrap.py --check-updates`; pinned deps; clean-clone `make reproduce`; TECHNICAL.md final |
| **17 Nov** | Final form edit (validated CSV, repo URL, `<team>.md`) |

Phase 2 seed ideas (top 20 only): possession-aware counterfactual simulation (diffusion /
transformer multi-agent rollouts — thesis work), pressing/transition tools for analysts.

---

## Sources
- Official repo: https://github.com/metrica-sports/more-than-a-hack-2027
- Metrica blog, 2027 edition: https://www.metrica-sports.com/blog/more-than-a-hack-2027-build-something-real-with-fc-barcelona-match-data
- Registration: https://landing.mobileworldcapital.com/more-than-a-hack-2027
- IDSSE dataset (Bassek et al., Scientific Data 2025): https://doi.org/10.1038/s41597-025-04505-y
- floodlight IDSSE loader: https://floodlight.readthedocs.io/en/latest/modules/io/datasets.html
