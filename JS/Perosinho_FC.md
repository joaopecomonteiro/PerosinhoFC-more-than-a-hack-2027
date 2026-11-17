# Perosinho FC — *Més que un Frame*: robust possession spells from uncorrected tracking

More than a Hack 2027, Phase 1 · FC Barcelona vs Athletic Club, LaLiga 2024-25 MD2 · 199,319 tactical-video frames
labelled H / A / D.
Team: João Soares (Sigmar), Carlos, Italo · Code: `<public repo URL>` · submitted file: `Perosinho_FC.csv`
(`python tasks.py submit`, byte-identical with `python tasks.py reproduce`).

## 0. In one paragraph

The ranking metric scores **possessions as intervals**: every spell counts once, whatever its length, and a split
or a merge costs as much as a wrong team. So we treat the task as segmentation. Per-frame evidence from the
tracking feeds a decoder that outputs whole spells. The pipeline has four stages. Each stage was developed and
chosen on **other games only**: 11 validation matches from three providers, under a simulated version of this
match's tracking noise, always scored with the official metric code. No number from this match was used to tune
or choose.

```
 ATD (46 anonymous slots, ball missing on 41 % of graded frames)
  │  1. fix the data       tracklet cleaner (team level) ─ missing-ball imputation (transformer + interpolation)
  │  2. dead ball          P(dead) = mean( ASFormer-lite ×3 seeds , Italo's v8 gradient boosting )
  │  3. team possession    P(home | alive) = owner TCN ×3 seeds + its H↔A boundary ─ Smart Tagging team votes
  ▼  4. decoder            exact semi-Markov Viterbi per half, length rules only:
                           runs ≥ 1.4 s · dead balls ≥ 3 s · a possession between two dead balls ≥ 1 s
 Perosinho_FC.csv
```

Held out on the validation games, spellF1 went from **0.31** (our v0 HMM) to **0.65**. The official baseline's
floor on this match is 0.24.

## 1. How every decision was made

* **Validation data.** We converted four external sources into the competition's tactical format (one "bundle"
  per game, native fps, as delivered): IDSSE (DFL), Metrica sample data, PFF FC World Cup 2022 and SkillCorner
  open data. A fixed split by game lives in `splits.json`. The validation games are IDSSE ×2, Metrica ×1 and
  PFF ×8. Training data never comes from a validation game, and never from this match.
* **The noise of this match, simulated** (`mqf/harness/noise.py`). The noise model reproduces 46 tracklet slots,
  duplicates, dropouts, re-acquisitions, positional jitter, and tracks picked up under the wrong team (keepers
  especially). How the ball goes missing in this match is unknown, so every candidate is run under three
  ball-loss assumptions: lost on exactly the dead frames, lost on most stoppages with jittered edges, or lost at
  random. That gives **six columns** (`bar exact / fuzzy / random`, `atd exact / fuzzy`, `atd_flip fuzzy`) × 11
  games × noise seeds.
* **Decision rule** (EXPERIMENTS #38). A change is adopted only if, against the current recipe, it gains more than
  2 standard errors and wins on most games **in every column**. Results are averaged per source, then over
  sources. Hyper-parameters are chosen by nested leave-one-game-out, so every reported number is held out.
* **On this match** we only built the file, validated its format, and compared two of our own files in aggregate
  (share of frames changed, possession counts; `mqf.compare`). The public 5 minutes were used only to check that
  the pipeline runs correctly. We watched the video to debug, but no frame, label or constant from this match
  enters the code: the second-half kickoff, for example, is detected, not hard-coded.

Every run is logged in `docs/EXPERIMENTS.md` (48 entries, aggregates only).

## 2. Fixing the data

### 2.1 Tracklets (`mqf/tracks.py`, team level)

Possession needs teams, not identities, so the cleaner works at team level. In this match, at least one team shows
more than 11 tracks on 5–8 % of frames, two same-team tracks sit within 1 m on 12 %, and about 1.4 tracks per
minute jump faster than a sprinter. The cleaner applies these steps; every threshold comes from football or
tracking knowledge:

1. Cut a track that moves faster than 10.5 m/s × Δt + 1 m. The check runs over 1 frame, 0.2 s and 1 s, which
   catches tracks that slide onto another player in small steps.
2. Hide a same-team duplicate while the pair stays together (< 2 m). Both tracks come back if they separate.
3. Cap each team at 11, dropping in this order: off the pitch, frozen while the team moves (never inside a penalty
   area), next to a team-mate, newest.
4. **Repair the team of a track.** The ATD gives a team per slot from kit colour, and a re-acquired player gets a
   new slot. A new track inherits the team of a track that ended where it starts. Every team change must pass a
   player-count check: the new label has to bring its team back towards 11 against the other.
5. Law-based keeper rules (Sigmar). Each half, the side a team defends is read from its players' positions at the
   start of the half (Law 8). A track that appears alone, deepest, inside a penalty area is that area's keeper.
   A new track next to the opposing keeper takes over from him. Kickoff shape checks run only when the ball sits
   on the centre spot or a Smart Tagging kickoff confirms it.

**What did not work.** v1 hid duplicates for a track's whole life and deleted real players: 20.5 → 18.8 players
per frame. The first "frozen track" rule removed goalkeepers. An unguarded box rule turned Barça attackers alone in
Athletic's box into Athletic keepers. Team repair without the count check broke as many tracks as it fixed.

**Result** (#21, #39). With simulated wrong-team pickups the cleaner adds **+0.023 spellF1 (19/22 runs)**. When
every team is right, the change is +0.003, so it costs nothing. 97.8 % of tracked points are kept.

### 2.2 The missing ball (`mqf/ball/`)

The ball is filled only while it is in play. We compared interpolation with two bidirectional transformers trained
with masked ball spans on 63 games:

* A: per-frame set encoder + temporal attention, after Kim et al., KDD 2023.
* B: axial agents × time, after TranSPORTmer, ACCV 2024.

The key change came in v3. v1 regressed absolute positions, and its error did not depend on the gap length, a sign
it ignored the ball around the gap. v3 instead predicts a **correction to the linear interpolation**, with a head
initialised at zero. It also receives the gap context: time since and until the ball was last and next seen, and
whether each edge exists. v4 adds the ball's velocity at both edges and an auxiliary head that predicts the player
nearest the ball. v5 trains the same model on IDSSE + Metrica + PFF.

| validation error (m), hidden gap of | 0.5 s | 2 s | 4 s | 8 s | 32 s |
|---|---|---|---|---|---|
| Hermite interpolation | 0.7 | 2.6 | 4.3 | 8.9 | 29.0 |
| **frozen hybrid** | 0.2 | 2.6 | 2.6 | 3.1 | 2.6 |

v5's best validation ADE improved from 5.0 m (v1) to 2.8 m. The frozen hybrid is linear ≤ 1 s, Hermite ≤ 2 s, then
the v5 axial model anchored to the seen ball at both gap edges.

More capacity (6.7 M parameters) was worse than more data. In the decoder, filled frames are flagged:

* their "who has it" evidence is down-weighted (`w_filled` 0.4);
* their alive/dead evidence is treated as missing — **an imputed ball says nothing about stoppages**.

The fill recovers about 85–90 % of the spellF1 a perfect ball would give: **+0.019 to +0.026**. It runs on the
CPU in float64, with positions rounded to 1 cm, so it is reproducible.

## 3. Dead ball

* **v0.** A mean player speed under 1.3 m/s means a stoppage. This was fragile: the match's median player speed is
  1.33 m/s.
* **dead_v1** (`mqf/dead.py`). Gradient boosting on players-only features (collective speed and spread, box and
  touchline occupancy, 1–30 s context, rank within the half). **Ball visibility is deliberately not a feature.**
  Whether a missing ball means a stoppage is a property of the tracking system: PFF loses the ball in stoppages,
  DFL does not, and we cannot measure it for this match. A ball seen off the pitch stays a hand rule
  (P(dead) 0.97).
* **mqf.alive** (#42). We ran a model zoo on one feature table: dead_v1's columns plus Law-based set-piece shapes
  (corner mark, beyond the line, goal-area crowding, kickoff shape), restart-burst and stillness clocks, and the
  ball only where tracked. The ball's absence was made label-independent in every training view. Models: GBM, MLP,
  an ASRF-style TCN with a boundary head, and an **ASFormer-lite** (windowed self-attention). The 3-seed
  ASFormer-lite ensemble beat dead_v1 by **+0.030 to +0.049 in every column**, closing about 40 % of the gap to a
  perfect alive mask.
* **Italo's v8** (`mqf/deadball/`, #44). Gradient boosting with ball inputs, slow-down and speed-up edges, a "went
  out" memory that only a restart closes, a keeper holding the ball in his box (in play, Law 12), and a players-only
  P(dead) stacked out-of-fold. It comes with its own two-state semi-Markov decoder. It was trained with the ball's
  loss in stoppages randomised per copy.
* **The submission averages the two** (#45). The mean beats ASFormer alone by **+0.005 to +0.017 in every column**.
  v8 with its own decoder is better only if the ball vanishes exactly in stoppages, and loses when it does not, so
  we did not bet on it.

**Tried and dropped.** Hand Law rules as a veto or a snapping step; a Smart Tagging "set piece starts dead" prior;
Italo's inputs fed into our models (#46: small gain, failed the rule).

## 4. Team possession

The ground truth is **ownership**: the last team in control keeps the ball while it flies, bounces, is contested or
is lost by the tracker.

* **v0.** The team of the nearest player now: right on about 78 % of alive frames, wrong exactly when it matters.
* **who_v1** (Carlos, `mqf/who.py`, #34–#35). Gradient boosting on 59 position features. They include clocks:
  seconds since and until each team was last or next at the ball, and its share of the last and next 1–3 s. Each
  game is trained as delivered and in a "hard" version (dropouts, no ball when dead, 40 % of the alive ball
  hidden), and with the teams swapped. The prediction is the mean of both orders, so P(H) + P(A) = 1 exactly.
  **+0.21 spellF1 on 11/11 games**, including PFF, a source it never saw. who_v2, trained on 9× the games, was not
  better.
* **Owner TCN** (`mqf/owner/`, #38, #40). A temporal CNN on 121 position columns with a second head predicting H↔A
  boundaries. It uses 3 seeds and IDSSE + Metrica + PFF. The boundary probability lowers the switch cost locally
  (× (1 − 0.5·P(boundary))). Over who_v1 it adds **+0.048 to +0.058 in every column**. Ensembling and PFF data
  both mattered for the TCN; the trees gained nothing from either.
* **Smart Tagging** (this match only). Each phase's team adds a soft vote (weight 0.4) to the possession log-odds.
  Phases are not possessions and the tagging is automatic, so it is evidence, not truth.

**Tried and dropped** (#38, #41):

| idea | result |
|---|---|
| explicit possession rules from positions (control = alone at the ball ≥ 0.4 s, pass = two consecutive contacts by one team) | **−0.11** |
| training on SkillCorner (10 fps, different possession groups) | −0.06 |
| pass-direction features (ball heading towards a receiver) | ±0 |
| stacking | ±0 |

PathCRF (Kim et al., KDD 2026) was run as an outside reference on IDSSE only.

## 5. The decoder: three length rules

Until 16 Nov the decoder was a three-state HMM per half with tuned switch costs: H↔A 18 × (1 − 0.5·P(boundary)),
and 10 into and out of D. Sweeping 63 cost settings was flat (#36). A fusion suite then compared eight decoder
families on the same evidence (#47). The families were:

* the HMM
* semi-Markov with soft or empirical duration priors
* a hierarchical alive→team model
* ASRF-style boundary cutting
* forward-backward marginals re-decoded semi-Markov

The best family (soft durations) gained +0.012 but failed two columns, so it was not adopted.

**The submitted decoder** (`mqf/fusion.py`, `stoppage`) drops the costs altogether. It reads exactly the same
per-frame evidence as the HMM and finds, per half, the labelling with the highest total evidence such that:

* every H / A run lasts **≥ 1.4 s**: a touch, deflection or aerial duel shorter than that cannot become a
  possession;
* every dead ball lasts **≥ 3 s**: no flicker into D during play;
* a possession with a dead ball on both sides lasts **≥ 1 s**: a quick restart that goes straight out again is
  still a possession.

This is solved exactly by a 7-state semi-Markov Viterbi: dead, and for each team "between two dead balls", "after a
dead ball", "after the other team or at the half's start". States 1–4 cannot start a half, and "between" cannot
end one. Tests check it against brute-force enumeration and check that the numba and numpy kernels agree.

The three lengths were chosen on the validation games: 12 sampled + 37 grid settings, nested leave-one-game-out.
The optimum is a **flat plateau** (min run 1.4–1.8 s, dead 3–4.5 s, between 0.5–1 s, all 0.652), and we took its
round centre.

| decoder on the same evidence (held out, mean of 6 columns) | spellF1 | spells − true spells per game |
|---|---|---|
| per-frame argmax, no decoder | 0.594 | +42 |
| HMM with tuned costs (FINAL_HMM, 16 Nov) | 0.654 | −98 |
| length rules 0.5 s / 2 s / 2 s (first guess from football intuition) | 0.635 | −16 |
| **length rules 1.4 s / 3 s / 1 s (submitted)** | **0.650–0.652** | **−43** |

The submitted decoder ties the HMM: no column differs by more than 2 standard errors either way. It replaces four
tuned costs and a boundary weight with three durations a coach can read, and it merges half as many possessions.
The untuned 0.5 / 2 / 2 s version lost 0.019 in every column, which is why the lengths were set on data, not by
eye.

## 6. Possession rules implemented

* **Loose and contested balls.** The owner is learned from ownership labels (last team in control), so a ball in
  flight, bouncing, contested or missing stays with the last owner. The decoder never lets the other team own it
  for less than 1.4 s.
* **Challenges, deflections, aerial duels.** A contact by the team without the ball that does not lead to control
  is a run shorter than 1.4 s and is absorbed into the current spell. A change needs sustained evidence, which the
  owner model learns from controlled possessions of other games.
* **Restarts and stoppages.** Dead frames come from P(dead) (a ball seen off the pitch counts as dead) and last at
  least 3 s. At a restart the team comes from the possession evidence, and a short possession between two
  stoppages (≥ 1 s) is allowed. Each half is decoded on its own. The long stoppage before the second-half kickoff
  is found by the models; no frame number is used.
* **Abstain.** `X` only outside the graded halves.

## 7. Assets

| asset | used? | how |
|---|---|---|
| Tactical ATD | yes | every model input; cleaned, ball filled |
| Smart Tagging | yes | team votes (weight 0.4); kickoff windows gate the cleaner's kickoff check |
| Homography, tactical and broadcast video | debugging only | `mqf.viewer` draws the tracking on the video; no model input |
| Broadcast ATD | debugging only | aligns the broadcast to the tactical timeline in the viewer |

## 8. Training data (none from the provided game)

| dataset | labelled by | amount | used for |
|---|---|---|---|
| IDSSE (Bassek et al. 2025, CC-BY 4.0) | DFL official per-frame ball status + possession | 7 matches (5 train / 2 val) | every model; validation |
| Metrica Sports sample data | labels derived by us from Metrica's manual synced events (dead = out / goal / before set pieces; owner = last controlling event) | 3 matches (2 / 1) | every model; validation |
| PFF FC World Cup 2022 | labels derived by us from PFF's manual events (dead = out to next on-ball event; owner = last player in control) | 64 matches (56 / 8) | ball model, dead ball, owner TCN; validation |
| SkillCorner open data (MIT) | derived from its possession groups and phases | 20 matches | an ablation only (it hurt) |
| Synthetic matches (`mqf/synthetic.py`) | algorithm | — | tests |

## 9. Limitations and what we would do next

* Our validation labels are other providers' definitions (DFL possession, Metrica and PFF events), not Metrica's
  corrected-tracking ownership. The noise model is ours, and the ball-loss pattern of this match is unknown. We
  hedged with three assumptions but could not measure it.
* Short spells (< 3 s) are still rarely matched, and the file still merges: about 43 fewer spells than the truth
  per validation game. This is a limit of the evidence's precision on short spells, not of the decoder's
  stickiness (#36).
* A keeper read as the other team while his own keeper is lost leaves the player counts even, so only the box rule
  can catch it.
* Next steps: train dead ball and owner jointly with a segment-level loss (boundary F1, not frames); learn the
  duration rules inside a semi-Markov CRF instead of a grid; add sources labelled from corrected tracking.

## 10. Reproduction

```bash
python bootstrap.py                                         # official repo, pinned commit, into vendor/
pip install -r requirements.txt                             # pinned; Python 3.12-3.13
python tasks.py install-torch --cuda cpu                    # torch 2.15.0 (CPU build) + requirements-model.txt
# unpack the match bundle into data/   (or set MTAH_DATA_DIR)
python tasks.py submit --team Perosinho_FC       # = python -m mqf.make_submission --team Perosinho_FC --final
python tasks.py reproduce --team Perosinho_FC    # rebuild into cache/reproduce/ and byte-compare
```

The recipe is `FINAL` in `mqf/make_submission.py`. `FINAL_HMM` is the same recipe with the HMM decoder
(`--final-hmm`). Every option lands in the run report. The torch models run on the CPU in float64 on one thread,
with outputs rounded to 1e-4, so the file is byte-identical across machines. The models are committed in
`models/`; older ones are in `models/old/`. `python -m pytest -q` runs the tests, and `python run_fusion.py`
re-runs the decoder comparison on the validation exports.

*Tools.* Claude (Anthropic) was used as a coding assistant under the competition's data terms. It never received
the match data, only code, other games and aggregate scores.
