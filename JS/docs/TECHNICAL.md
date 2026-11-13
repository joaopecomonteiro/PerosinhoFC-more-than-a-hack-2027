# Més que un Frame — Technical document

*Team: Perosinho FC · Members: … · More than a Hack 2027, Phase 1 · Repository: <public repo URL>*

> Upload a copy named `Perosinho_FC.md` (or export to `Perosinho_FC.pdf`). A few pages.

## 1. Approach — what we built and why
Possession is scored as intervals (spellF1), so we treat it as segmentation: per-frame evidence
from the tracking and Smart Tagging → a three-state (H / A / D) hidden Markov model decoded with
Viterbi, separately in each half → whole possessions.
<!-- Fill in: final emission model, decoder (HMM/HSMM, duration priors), alternatives rejected and
what the validation harness showed. -->

## 2. Assets used (and dropped)
| asset | used? | how / why dropped |
|---|---|---|
| Tactical ATD | yes | ball–team distances, stoppage cues, team shape |
| Smart Tagging | yes | soft team evidence from phases; set-piece windows as restart priors |
| Homography | debugging only | tracking drawn over the video in `mqf.viewer`; not a model input |
| Tactical video | debugging only | watched in `mqf.viewer` next to the labels; not a model input |
| Broadcast video / ATD / homography | debugging only | `mqf.viewer` links the broadcast to the tactical timeline from the two ATD files (offset search, no hand input); not a model input |

## 3. Handling missing and noisy tracking
**Tracklets (`mqf/tracks.py`, team level — no player identities are needed for team possession).**
The ATD gives 46 anonymous slots; on our match a team shows more than 11 tracks on 8 % / 5 % of
graded frames, two same-team tracks sit within 1 m on 12 %, and ~1.4 tracks per minute jump faster
than a sprinter. The cleaner (thresholds from football knowledge, none fitted to the match):
1. cuts a track where it moves more than 10.5 m/s × Δt + 1 m;
2. treats two same-team tracks < 1 m apart with velocities within 1 m/s for ≥ 0.4 s as one player
   and shows only the newer while they stay together (< 2 m) — both come back if they separate;
3. caps each team at 11: drops a track > 1 m off the pitch, then one frozen (< 0.05 m/s for 2 s,
   never inside a penalty area) while its team moves, then one next to a team-mate, then the newest;
4. recomputes speeds from positions, never across a cut.
It keeps 97.8 % of the tracked points and removes the excess (players per graded frame 20.5 → 20.2).
**Missing ball (`mqf/ball/`; not in the submission yet, kept for the possession models).** Filled only while the ball is in play. Three
tactics compared on the validation games of other teams: interpolation (linear / Hermite), a
bidirectional transformer trained on IDSSE + Metrica with masked ball spans (A: per-frame set
encoder + temporal transformer, after Kim et al. KDD 2023; B: axial agents × time, after
TranSPORTmer, ACCV 2024), and a hybrid that interpolates gaps ≤ τ and uses the model beyond.
v1 regressed the absolute position and ignored the ball around the gap (validation error flat in
the gap length). v3 reads the gap context: the linear interpolation of the seen ball, whether both
edges exist, and the time since and until the nearest seen ball, read up to 60 s beyond the window
as inference reads the whole half. It predicts a correction to that interpolation, and its head
starts at zero, so training starts as interpolation. At inference the correction is upsampled from
6.25 Hz and added to the 25 fps interpolation. Augmentation: mirror x / y, swap team labels.
v4 adds the ball velocity at both gap edges to the context and, on the axial model, an auxiliary
head that classifies which player is nearest to the true ball at each step (cross-entropy, weight
0.5, on the same frames as the position loss).
Frozen setting (`impute.FROZEN`, EXPERIMENTS #17): linear interpolation for gaps ≤ 1 s, Hermite
≤ 2 s, the v5 axial model (d 128, 1.3 M parameters, trained on IDSSE 5 + Metrica 2 + PFF 56) anchored
to the gap edges beyond. On the 3 validation games with hidden alive gaps it averages 0.2 / 0.9 / 2.6
/ 2.6 / 3.1 / 2.9 / 2.6 m for gaps of 0.5 / 1 / 2 / 4 / 8 / 16 / 32 s (clean players; 3.0–4.1 m from
4 s up with ATD-like player dropouts), against Hermite's 4.3 / 8.9 / 17.9 / 29.0 m from 4 s up.
The model output is clipped to the pitch and anchored to the seen ball at each gap edge (edge
residuals blended across the gap, faded after 2 s). The decoder flags imputed frames
(`ball_filled`): their "who is closer" evidence gets its own weight `w_filled`, and their alive/dead
evidence is weighted as if the ball were missing. An imputed ball says nothing about stoppages.
Result on the validation games (25 % of alive frames hidden, oracle alive, EXPERIMENTS #12–17): a
perfect fill would add +0.026 (clean) / +0.020 (dropouts) spellF1 with the v0 decoder; the frozen v5
fill adds +0.019–0.022 / +0.013–0.014. In the submission it is opt-in (`make_submission --ball-fill`):
CPU inference in float64 with filled positions rounded to 1 cm, so last-bit differences between
machines cannot change a label; the run options are stored in the report and `--reproduce` reuses
them. With the ball also hidden on every dead frame and filled everywhere (BAR-ATH's pattern,
`eval-ball --hide-dead`, EXPERIMENTS #18) the fill still adds +0.026 (clean) / +0.019 (dropouts),
~90 % of a perfect fill, with `w_filled` 0.4 (the default since).

**Stoppages (`mqf/dead.py`, in validation).** A per-frame P(dead) learned on other games replaces
v0's single 1.3 m/s player-speed threshold inside the HMM. Inputs are the players only, computed the
same way for every source and for BAR-ATH: speeds from positions over 1 s (not the tracker's speed
field), team speeds and spreads, the gap between team centres, players in the boxes / near the
touchlines / off the pitch, players tracked, each smoothed over 1–30 s with the trend and the rank
within the half (every match judged against its own pace). Ball visibility is left out on purpose:
whether a missing ball means a stoppage depends on the tracking system (PFF's broadcast loses it in
stoppages, DFL keeps it), which we cannot check for our match; a ball seen off the pitch stays a hand
rule (P(dead) 0.97). Gradient-boosted trees (scikit-learn, pinned; the fitted model is committed),
trained on frames every 0.2 s of the training games plus a copy with ATD-like player dropouts. In
the decoder its weight is `w_dead`, the same with or without a ball, and `dead_who="max"` stops the
"who is closer" evidence from favouring or penalising D (v0 gave D no who term, so an uncertain owner
quietly pushed towards D). Settings are chosen on the validation games with the official metric
(`mqf.external.evaluate --grid`).

## 4. Training / validation data
| dataset | source / licence | labelled by | amount | used for |
|---|---|---|---|---|
| Synthetic matches | generated by `mqf/synthetic.py` | algorithm | n × 30 min | tests, harness |
| IDSSE (Bassek et al. 2025, Sci. Data 12:195) | figshare via floodlight | DFL official tracking (per-frame ball status + possession) | 7 matches 2022/23 (2 Bundesliga + 5 2. Bundesliga), CC-BY 4.0 | model selection with simulated ATD noise; ball-imputation and dead-ball training (5 train / 2 val games, `splits.json`) |
| Metrica Sports sample data (github.com/metrica-sports/sample-data) | free use, acknowledge the source | tracking: Metrica; labels derived by us from Metrica's manual synced events (dead from ball out / goal / before set pieces to the restart; owner = last controlling event) | 3 anonymised matches, 25 fps | validation (clean, no noise); ball-imputation and dead-ball training (2 train / 1 val) |
| SkillCorner open data (github.com/SkillCorner/opendata) | MIT | SkillCorner automatic broadcast tracking; labels derived by us (alive = phases of play; owner = per-frame possession group, carried forward) | 20 A-League 2024/25 matches, 10 fps | validation (clean, no noise) |
| PFF FC World Cup 2022 tracking (64 matches; download by the team from PFF FC, raw files not redistributed) | PFF FC's terms for the World Cup 2022 release | tracking: Sportlogiq automatic broadcast tracking (off-screen players estimated from context; we keep VISIBLE + ESTIMATED players and only VISIBLE balls); labels derived by us from PFF's manual game events (dead = OUT to the next controlling event, END, VID; owner = team of the player in control of the last on-the-ball / kick-off event, carried forward) | 64 matches (48 group + 16 knock-out), 29.97 fps, 105 × 68 m assumed (no metadata in the download) | ball-imputation and dead-ball training (56 train / 8 val, stratified by stage, `splits.json`); the ball model reads every 5th frame (5.99 Hz) |

No labels, corrections or frame-specific constants from the provided game were used. The public
5-minute ground truth was used only to check pipeline correctness.

## 5. Possession rules implemented
- Loose / contested balls:
- Challenges, deflections, aerial duels:
- Restarts and dead ball (incl. the stoppage before the 2nd-half kickoff, detected, not hard-coded):
- Controlled-possession threshold:

## 6. Results
| version | harness spellF1 (IDSSE, mean ± sd) | synthetic spellF1 | notes |
|---|---|---|---|

## 7. Limitations and next steps

## 8. Reproduction
```bash
python bootstrap.py                     # official repo at the pinned commit
pip install -r requirements.txt         # pinned
# unpack the match bundle into ./data  (or set MTAH_DATA_DIR)
python -m mqf.make_submission --team Perosinho_FC   # -> results/Perosinho_FC.csv, validated
python -m mqf.make_submission --team Perosinho_FC --reproduce   # regenerate + byte-compare
```
Python 3.13, deterministic.
