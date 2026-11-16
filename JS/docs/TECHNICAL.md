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
v4 also checks the speed limit over 0.2 s and 1 s windows (a track sliding onto another player in
steps under the one-frame limit is hidden there and cut) and repairs the TEAM of a track — the ATD
gives a team per slot from kit classification, a player picked up again gets a new slot, and the
goalkeeper, whose kit is neither team's, is where we saw it go wrong:
5. at a kickoff every player is in his own half (Law 8). Evidence of a kickoff: the ball seen lying on
   the centre spot, or — because this tracker usually loses a ball that lies still — a kickoff tagged
   by Smart Tagging (automatic, ±3 s). Within it, the players must hold a kickoff shape for ≥ 1 s (a
   player at the spot, each team's median in its own half, ≤ 4 players in the centre circle, ≥ 80 %
   of the others in their team's half, median player < 1.5 m/s). A track that stays more than 2 m
   inside the other half, outside the circle, over the wait gets the other team (≤ 2 per kickoff), but
   only a team that is short of 11 at that moment and only if the player counts agree over the whole
   life of that track, as for every other team change — otherwise it is an official or a lean.
   The shape alone is not enough: a slow build-up against a mid-block looks the same, and without
   the ball or a tag the check flipped real players on other games;
6. a new track that starts where exactly one track ended ≤ 2 s earlier (within 10.5 m/s × gap, at most
   5 m, + 1 m) or stands frozen (a held copy) takes that track's team; with nobody like that, a track
   that vanished up to 30 s earlier within 2 m of the new one — its saved last place, e.g. a keeper
   standing unseen — counts too. Conditions: the old track lived ≥ 2 s, nobody else is within 3 m
   (never in a duel), and the player counts agree: over the new track's life its label must give its
   team ≥ 1.5 more players than usual against the other (a wrong-team track adds one there and takes
   one away). A held copy is hidden only while it stays frozen and its pickup is there; if the
   pickup ends first, the saved place shows again and can be claimed by the next pickup. Repairs
   carry forward from pickup to pickup.
7a. v4g, Sigmar's keeper rule: which goal a team defends is counted once per half — over the first
   60 s of the half the team with more players left of the halfway line defends x = 0 for the whole
   half (Law 8) — instead of the smoothed medians. A new track in a penalty area with no MOVING track
   within 3 m and nearer the goal than every moving player is the defending side's keeper, whatever
   its colour and without the player-count check: when the keeper is picked up in the other team's
   colour while that team's own keeper is lost, the counts read even (EXPERIMENTS #31–#32). Tracks that
   do not move (frozen ≥ 0.4 s, or the tracker repeating a lost player's place) are no players here and
   are hidden while they stay still. One guard: it must stay in that box on ≥ 50 % of its life.
7. goalkeepers (one per team, in his own penalty area; which goal a team defends comes from the two
   teams' medians smoothed over 60 s, as at the kickoff): a track that appears alone (nobody within
   3 m) in a penalty area and deeper than every other player is that goal's keeper and gets the
   defending team, if the player counts agree that this team is the one short of a player (without
   that check the rule turned Barça attackers alone in Athletic's box into Athletic keepers: away over
   11 twice as often, EXPERIMENTS #26); and while a team is short of 11 with nobody of its own in its penalty area (play
   on), one more lane shows its keeper at his saved last place — where he was last seen as the
   deepest of all players, for at most 10 s; never seen yet in this half: the 6-yard line. One keeper,
   however many players are missing. The added point has no speed, so speed features ignore it.
8. keeper takeover (v4d, Sigmar's rule), only for a pickup that rules 6–7 leave undecided: a new
   track within 3 m of the OTHER team's goalkeeper is that keeper. The keeper is that team's track last
   seen as the deepest of all players in its own penalty area within the last 10 s (the same definition
   as the keeper fill). If he is still tracked (v4e), appearing next to him is enough: the new track
   takes his team and his old track is hidden while it is held (it repeats its place exactly: this
   tracker holds a lost player ~3 s) or still within 3 m of the new one — it shows again if they
   separate. (v4d asked the new track to sit on his held copy or within 1 m of him on every shared
   frame; in the viewer that left tracks of the other colour standing next to a tracked keeper.) If
   he is gone, his saved last place counts. In every case the player counts must agree, as for every
   team change: a real attacker next to a live keeper leaves the counts even (11 v 11) and is left
   alone; a second track on the keeper gives his team one too few and the other one too many.
   The counts can be fooled by other players the tracker misses at the same time, so when they are
   ambiguous (v4f) the new track's behaviour decides: in the keeper's penalty area and deeper than
   every other player (within 0.5 m) on ≥ 80 % of its life, over ≥ 2 s, it is the keeper — an
   outfield player that deep for that long would be offside (Law 11); an attacker who touches the
   keeper and leaves is not. A live keeper deeper than the new track counts against it; only his
   held copy is left out. Why: rules 6–7 need the new track alone, and the keeper's own held copy
   right next to it made it never alone, so in the viewer Fixed looked like Original exactly there.
On other games with ATD noise (EXPERIMENTS #21) the repair changes nothing measurable when no track has
the wrong team (spellF1 Δ +0.0001), and with simulated wrong-team pickups (an assumption: 5 % of
pickups, half of the keeper's) it recovers +0.008 ± 0.003 of the −0.060 they cost. Without the count
check it broke as many tracks as it fixed.
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

**Dead-ball experiments (`mqf/alive/`, in validation, opt-in, EXPERIMENTS #42).** With who_v1 fixed, the true
alive mask scores ~0.11 spellF1 above dead_v1 (#38), the largest gap left; dead_v1 is already good per frame
(AUC 0.95–0.98), so its losses are segment-level: missed short stoppages (merges), spurious D blips (splits)
and late boundaries. The same pattern appears in the literature on in-game status from player tracking (Lang
et al. 2022: frame F1 0.93 but stoppage F1 0.80, about a fifth of the boundaries more than 2 s off). The
package is built like `mqf/owner`: one feature table from positions (dead_v1's 36 columns; collective speed /
acceleration, restart-burst and stillness clocks; set-piece shapes from the Laws — box and goal-area crowding,
a player at a corner mark, beyond the touchline or goal line, the kickoff shape, crowding, bench side; the ball
only where TRACKED — out of the pitch, distance to the lines / corner / goal area / centre spot, still, a
restart kick as a speed jump from < 2 to > 7 m/s, as ELASTIC finds set-piece kicks). The ball's absence is
never a feature, and every training view makes it label-independent (spans of the better-seen class removed
until alive and dead frames are seen equally often), because whether a missing ball means a stoppage is a
property of the tracking system. Law-based rules (Vidal-Codina et al. 2022-style triggers; every half starts
dead and leaves that state at the first restart it sees — no frame constant) give a hand-made alive layer, rule
columns and restart / stop candidates. Models: GBM (dead_v1's settings), MLP, an ASRF-style TCN with a boundary
head and an ASFormer-lite (windowed self-attention), seeds and source sets as in #40. Decoder variants on the
production HMM: a half-start D prior, boundary-aware dead costs, a minimum stoppage length, snapping restarts /
stops to rule candidates, a rule veto, and a Smart Tagging prior (the first 2.5 s of every SET PIECES window;
its weight is set on tag-like windows simulated from other games' labels, never on BAR-ATH). Everything is
scored on the 11 validation games with the official metric under three ball-loss assumptions (ball removed
on exactly the dead frames / on most stoppages with jittered edges / at random), because how much BAR-ATH's
missing ball overlaps its stoppages is unknown; a change is kept only if it passes in all of them
(`python -m mqf.alive.suite`). `make_submission --alive-model … --alive-variant …` uses it; its `hmm` variant is
the production decoder bit for bit. Result (#42): the ASFormer-lite ensemble +0.030 … +0.049 spellF1 over
dead_v1 in every condition and ball mode; it is the dead-ball layer of `FINAL` from 16 Nov (#43).

**A second dead-ball line (`mqf/deadball/`, Italo, EXPERIMENTS #44).** Developed on its own branch and ported
into `JS/`: a gradient-boosted P(dead) on dead_v1's player inputs plus ball inputs (gaps, restart spots, off
the pitch, still, speed), slow-down / speed-up edges, a "went out" memory closed only by a restart, a keeper
holding the ball in his box (in play, Law 12) and a players-only P(dead) stacked out-of-fold by game; trained
on the same 63 training games with the ball's loss in stoppages randomised per copy (whole, partial, drifting
gaps, kept visible, carried like play — assumptions about trackers, never calibrated on BAR-ATH). Its decoder
is a two-state semi-Markov Viterbi at 5 Hz whose state lengths are the real stoppage / play lengths of the
training games. It is compared with the mqf.alive models on the validation games with the official metric
(`mqf.alive.final_check`), both as a P(dead) for our HMM and with its own decoder.

**Who has the ball (`mqf/who.py`, in validation).** The ground truth is ownership: the last team in
control keeps the ball while it flies, bounces or is contested. v0's "team of the nearest player
now" was right on ~78–79 % of alive frames of the validation games. A per-frame
P(home has the ball | alive), learned on other games, replaces it inside the HMM. It uses 59
features from positions only, the same for every source and for BAR-ATH: the nearest and
second-nearest player of each team to the ball, players within 2 / 5 m, the nearest player's speed,
ball speed, acceleration and distance to the touchline. It also has clocks: seconds since / until
each team was last / next "at the ball" (nearest of all, within 2 m), capped at 15 s, and since /
until the ball was seen. It has each team's share of the last / next 1 and 3 s at the ball, and
the shape in the direction of play (each half's attacking direction read from the two teams' mean
positions; ball position and velocity towards goal, players goal-side of the ball, team centres,
depth, width, speed). The windows are centred: labels are made offline and the whole match is known,
as for the submission. Gradient-boosted trees (scikit-learn, pinned; `models/who_v1.joblib` is
committed). They are trained on alive frames every 0.2 s of the training games (IDSSE 5 + Metrica 2,
`splits.json`), labels H vs A, 441k rows. Each game is used twice: as delivered, and "hard" (ATD-like
player dropouts, no ball on dead frames, 40 % of the alive ball hidden in 0.2–30 s spans), so the
model also answers when the ball is missing. Home and away are arbitrary, so every game is also
trained with the teams swapped. The prediction is the mean of the two team orders, which makes
P(home) + P(away) = 1 exactly. In the decoder the model's log-odds replace v0's "who is closer"
logit on every frame where it gives a value. Its weight is `w_who_model` (default 1), times
`w_filled` where the ball it read was imputed, and it is clipped at `who_clip` (0.02, at most 3.9
log-odds per frame). The model is opt-in: `make_submission --who-model models/who_v1.joblib`.
Without it the output is bit-for-bit unchanged.
Per alive frame on the 3 validation games, the model scores 0.92 (IDSSE) / 0.96 (Metrica) against
v0's 0.78 / 0.79, and 0.82 / 0.86 where the ball is hidden (v0 has no answer there; last team holds
0.66). On top of the ball fill and the dead model, spellF1 rises from 0.35 to 0.56 under the full
ATD noise model (2 seeds × 3 games, every run +0.15 or more). Short spells (< 3 s) are matched 0.16
instead of 0.07 (EXPERIMENTS #34). On all 11 validation games (IDSSE 2, Metrica 1, PFF 8; player
dropouts, no ball on dead frames, ball fill, no added noise) spellF1 rises from 0.31 to 0.52 and
every game improves. That includes the 8 PFF games, a source the model never saw (+0.19). A second
model trained on 63 games (`who_v2`: IDSSE 5 + Metrica 2 + PFF 56) is no better (0.50–0.51), so
`who_v1` stays. With the model on, the decoder predicts fewer spells than there are (54–78 % of the
true count), and spells under 3 s are still matched only 1 time in 8 (EXPERIMENTS #35). The decoder's
costs were swept with the model on (63 settings, on the 11 validation games and on 8 further PFF
games): a lower switch cost finds more of the short spells and adds as many unmatched ones, every
setting from 9 to 18 is within ±0.007 spellF1, and the costs stay as they were (EXPERIMENTS #36).
Pass-direction columns (`mqf/owner/feats.heading_columns`, experimental, not in the submission;
EXPERIMENTS #41): the cosine between the ball's velocity and the ball → nearest player of each team (a
pass in flight points at its receiver), that player's |v_player − v_ball|, a "close and moving with the
ball" log-odds over every player, and the last team in clear control (nearest within 1.2 m, 0.5 m ahead
of the other team's nearest) carried and faded over 4 s. Ported from a team-mate's experiment on other
games; thresholds from that experiment, none from BAR-ATH. They did not help: no change for the trees and,
in the TCN ensemble, −0.002 spellF1 under the ATD noise model (#41); the submission does not use them.

## 4. Training / validation data
| dataset | source / licence | labelled by | amount | used for |
|---|---|---|---|---|
| Synthetic matches | generated by `mqf/synthetic.py` | algorithm | n × 30 min | tests, harness |
| IDSSE (Bassek et al. 2025, Sci. Data 12:195) | figshare via floodlight | DFL official tracking (per-frame ball status + possession) | 7 matches 2022/23 (2 Bundesliga + 5 2. Bundesliga), CC-BY 4.0 | model selection with simulated ATD noise; ball-imputation, dead-ball and who-has-the-ball training (5 train / 2 val games, `splits.json`) |
| Metrica Sports sample data (github.com/metrica-sports/sample-data) | free use, acknowledge the source | tracking: Metrica; labels derived by us from Metrica's manual synced events (dead from ball out / goal / before set pieces to the restart; owner = last controlling event) | 3 anonymised matches, 25 fps | validation (clean, no noise); ball-imputation, dead-ball and who-has-the-ball training (2 train / 1 val) |
| SkillCorner open data (github.com/SkillCorner/opendata) | MIT | SkillCorner automatic broadcast tracking; labels derived by us (alive = phases of play; owner = per-frame possession group, carried forward) | 20 A-League 2024/25 matches, 10 fps | validation (clean, no noise); owner experiments (#38): one training-source ablation arm only (`gbm_full_imskc`, `gbm_full_all`, NN `_all`) |
| PFF FC World Cup 2022 tracking (64 matches; download by the team from PFF FC, raw files not redistributed) | PFF FC's terms for the World Cup 2022 release | tracking: Sportlogiq automatic broadcast tracking (off-screen players estimated from context; we keep VISIBLE + ESTIMATED players and only VISIBLE balls); labels derived by us from PFF's manual game events (dead = OUT to the next controlling event, END, VID; owner = team of the player in control of the last on-the-ball / kick-off event, carried forward) | 64 matches (48 group + 16 knock-out), 29.97 fps, 105 × 68 m assumed (no metadata in the download) | ball-imputation and dead-ball training (56 train / 8 val, stratified by stage, `splits.json`); the ball model reads every 5th frame (5.99 Hz); who-has-the-ball validation (8 val games; a `who_v2` also trained on the 56 train games was not adopted, EXPERIMENTS #35); owner experiments (#38): source ablation (`gbm_full_impff`, `gbm_full_all`) and NN on all sources |
| PathCRF (Kim et al., KDD 2026; github.com/hyunsungkim-ds/pathcrf, MPL-2.0) — an outside REFERENCE model, not in the submission | its own clone and venv, never copied into this repo (`mqf/owner/pathcrf_bridge.py` drives it) | its labels: DFL's manual events of the IDSSE games, synchronised to the tracking with ELASTIC (shipped in its repo) | trained by us on the 5 IDSSE train games of `splits.json`, predicted on the 2 IDSSE val games (EXPERIMENTS #38) | comparison only |

No labels, corrections or frame-specific constants from the provided game were used. The public
5-minute ground truth was used only to check pipeline correctness.

## 5. Possession rules implemented
In the submission path the who-has-the-ball evidence is learned (`mqf/who.py`, who_v1) and the decoder's
switch cost carries the "a touch is not a possession" rule. `mqf/owner/rules.py` states the rules explicitly,
from positions only (experimental, EXPERIMENTS #38; thresholds from football / tracking knowledge, none from BAR-ATH):
- Loose / contested balls: between contacts (ball flying, rolling, missing) the last owner keeps it.
- Challenges, deflections, aerial duels: a contact (nearest player within 1.5 m) by the team without the
  ball that is neither a control nor the start of a pass is a touch and never changes possession.
- Controlled-possession threshold: no opponent within 3 m of the ball and |v_ball − v_player| ≤ 2.5 m/s
  for ≥ 0.4 s inside one contact; or a pass — 2 consecutive contacts by different players (track slots)
  of one team, ≤ 3 s apart, no opponent contact between. The switch is dated at the first frame of the
  winning contact. On an imputed ball a contact can confirm the owner or extend a pass, never control alone.
- Restarts and dead ball (incl. the stoppage before the 2nd-half kickoff, detected, not hard-coded): dead
  frames come from the alive layer (learned P(dead), `mqf/dead.py`); with `rule_seg` the owner restarts at
  every alive run and the team of the first contact (the restart taker) owns it from the restart.

## 6. Results
| version | harness spellF1 (IDSSE, mean ± sd) | synthetic spellF1 | notes |
|---|---|---|---|

## 7. Limitations and next steps

## 8. Reproduction
```bash
python bootstrap.py                     # official repo at the pinned commit
pip install -r requirements.txt         # pinned: numpy, pandas, scikit-learn (the pickled models need 1.9.1)
pip install torch==2.15.0               # the ball fill; the CPU build is enough (we ran 2.15.0)
# unpack the match bundle into ./data  (or set MTAH_DATA_DIR)
python tasks.py submit --team Perosinho_FC      # = make_submission --final -> results/Perosinho_FC.csv, validated
python tasks.py reproduce --team Perosinho_FC   # regenerate into cache/reproduce/ + byte-compare
```
The submitted recipe is `FINAL` in `mqf/make_submission.py`, one place, and every flag it sets is in the run
report (`results/<team>.json`, aggregates only): tracklet cleaner (`mqf/tracks.py`) → frozen ball fill
(`models/ball_B_off_v5.pt`, CPU, float64, filled positions rounded to 1 cm so last-bit differences between
machines cannot change a label) → learned P(dead): the mean of the mqf.alive ASFormer-lite ensemble
(`models/alive_asf_full_impff_s0/1/2.pt`, #42) and the mqf.deadball v8 model (`models/deadball_v8.joblib`,
#44; the mean wins over each alone under every ball-loss assumption, #45) → learned P(home | alive): the owner TCN
ensemble (`models/owner_tcn_full_impff_s0/1/2.pt`, mean, its H ↔ A boundary lowering the switch cost with
α 0.5, #40) → Smart Tagging votes → 3-state HMM, Viterbi per half, decoder `Params()` with `dead_who="max"`
(#34–#39). The torch models run on the CPU in float64 with one thread and their outputs are rounded to 1e-4,
for the same reason as the ball fill. The model files are committed in `models/`. Until 16 Nov the recipe
was `FINAL_V1` (dead_v1 + who_v1); `python tasks.py final-harness` scores both on the validation games and
`python tasks.py final-check` builds, validates, reproduces and compares the new file before it is submitted
(EXPERIMENTS #43).
From a fresh clone with only the submitted file: copy it to `results/Perosinho_FC.csv` and run
`python tasks.py reproduce --team Perosinho_FC` — with no run report next to it, it rebuilds with `FINAL`.
Python 3.13, deterministic.
