# Més que un Frame — Perosinho FC

**Robust possession spells from uncorrected tracking.** Perosinho FC's entry to
[More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027), Phase 1
(FC Barcelona × Metrica Sports × Mobile World Capital Barcelona). The repository's name plays on Barça's
*Més que un club*.

For each of the 199,319 frames of FC Barcelona vs Athletic Club (LaLiga 2024-25, MD2) we output `H` / `A` / `D`.
The pipeline has four stages:

1. Repair the automatic tracking: team-level tracklet cleaner, then a transformer that fills the missing ball.
2. Estimate P(dead ball) as the mean of an ASFormer-lite ensemble and a gradient-boosting model.
3. Estimate P(home has the ball) with a temporal CNN ensemble that has its own possession-change head, plus Smart
   Tagging votes.
4. Decode the possessions with an exact semi-Markov Viterbi per half, under three length rules only.

Every choice was made on 11 games from other providers under simulated tracking noise, scored with the official
metric. Held out, spellF1 went from 0.31 (our first HMM) to 0.65.

**→ The technical document: [Perosinho_FC.md](Perosinho_FC.md)** (approach, data, rules, limitations,
reproduction). Every experiment is logged in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), and the engineering notes
are in [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Reproduce the submission

You need Python 3.12–3.13, git, and the match bundle. The bundle is released to registered participants; it is not
in this repository and is never committed.

```powershell
python tasks.py venv ; .venv\Scripts\Activate.ps1   # Python 3.13 venv (uv if installed, else python -m venv)
python tasks.py setup                                # pinned deps + official repo (vendor/) + tests
python tasks.py install-torch --cuda cpu            # torch + requirements-model.txt (the CPU build is enough)
# unpack the bundle into data/  (data/tactical/..., data/broadcast/...)  or set MTAH_DATA_DIR
python tasks.py submit --team Perosinho_FC           # -> results/Perosinho_FC.csv, officially validated
python tasks.py reproduce --team Perosinho_FC        # regenerate into cache/reproduce/ + byte-compare
```

`make <target>` (Linux/macOS/WSL) is the same as `python tasks.py <target>`. `python tasks.py help` lists every
target; the ones marked `[REAL DATA]` read the match bundle. The recipe is `FINAL` in `mqf/make_submission.py`.
The run is deterministic: the same data and code give a byte-identical file.

The official code we import and score with is a **pinned** clone in `vendor/mtah2027` (`python bootstrap.py`;
`--check-updates` lists newer organiser commits). We never use the copies in `challenge/`, so a merge from upstream
cannot change the scorer under us. `challenge/` holds the organisers' README, rules, `games.py`, `evaluation/`
and `starter/` as they were when we forked.

## Layout

```
Perosinho_FC.md     the technical document (submitted with the CSV)
mqf/                the pipeline package
  make_submission.py  CLI; FINAL = the submitted recipe, FINAL_HMM = same models with the HMM decoder
  config.py           paths; match constants via the official games.py
  atd.py  smart_tagging.py  features.py  decode.py  submission.py  synthetic.py
  tracks.py           team-level tracklet cleaner          ball/      missing-ball imputation
  dead.py  alive/     dead-ball models (ASFormer-lite …)   deadball/  Italo's v8 dead-ball model
  who.py  owner/      who-has-the-ball models (owner TCN)  fusion.py  decoders (stoppage = submitted) + suite
  harness/  external/ ATD noise model; other games -> tactical-format bundles
  viewer/             local viewer: video / pitch + truth and model lanes
tests/              pytest (synthetic end to end, official scorer, determinism, decoders vs brute force)
experiments/        one-off lab scripts (who_lab.py, stoppage_grid.py)
models/             the submitted models (+ FINAL_V1's);  models/old/  every other model we trained
harness_results/    aggregates on other games (committed); its *.npz inputs are not
docs/               CONTEXT, EXPERIMENTS, TECHNICAL, EXTERNAL_DATA_PLAN, KICKOFF_PROMPT
challenge/          the organisers' files as forked
tasks.py  Makefile  bootstrap.py  run_fusion.py  requirements*.txt  splits.json
data/  external_data/  cache/  results/  reports/  vendor/  external/  synthetic_data/   (gitignored)
```

## Develop without the competition data

```powershell
python -m pytest -q                                  # every test; synthetic matches, official scorer included
python tasks.py bundles                              # IDSSE / Metrica / PFF / SkillCorner -> tactical-format bundles
python tasks.py final-harness                        # FINAL vs FINAL_V1 on the 11 validation games (other games only)
python tasks.py fusion-export ; python tasks.py fusion   # the decoder suite on those runs (EXPERIMENTS #47-#48)
python -m mqf.viewer --val list                      # validation games in the viewer, with their full truth
python -m mqf.viewer.demo && python -m mqf.viewer --synthetic   # no match data needed
```

On BAR-ATH the viewer's truth lane is the public 5 minutes: use it to check correctness, never to tune. Keep
screenshots of the real match out of AI chats.

External data: IDSSE (Bassek, Weber, Rein & Memmert 2025, *Scientific Data* 12:195, CC-BY 4.0), the Metrica
Sports sample data, PFF FC World Cup 2022 tracking (raw files not redistributed), SkillCorner open data (MIT). See
the technical document for who labelled what.

Code: MIT. The match data is licensed for the competition only and is never committed.
