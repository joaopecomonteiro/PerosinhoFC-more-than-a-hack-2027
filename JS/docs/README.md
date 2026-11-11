# Més que un Frame

**Robust possession spells from uncorrected tracking.** Our entry to
[More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027), Phase 1
(FC Barcelona × Metrica Sports × Mobile World Capital Barcelona).

For every frame of FC Barcelona vs Athletic Club (LaLiga 2024-25, MD2) we output `H` / `A` / `D`:
per-frame evidence from automatic tracking and Smart Tagging, decoded into whole possessions by a
three-state hidden Markov model per half. Approach: [TECHNICAL.md](TECHNICAL.md).

## Where the code lives

This folder (`JS/`) sits inside our fork of the official repo. `mqf/config.py` detects that layout:
the match bundle is read from the fork's `data/`, other games from the fork's `external_data/`,
and the official code we import and score with is a **pinned** clone in `JS/vendor/mtah2027`
(`python bootstrap.py`) — never the fork's own copies, so a merge from upstream can't change the
scorer under us (`python bootstrap.py --status` reports drift). Everything we generate
(`cache/`, `results/`, `reports/`, `vendor/`, `external/`, `synthetic_data/`) stays inside `JS/`
and is gitignored by `JS/.gitignore`.

Dependencies stay in `requirements*.txt` (the single source of pins); uv, when installed, is only
used as a faster venv/pip — no `uv init`/`pyproject.toml`, which at the fork root would collide
with the official files. `tasks.py` holds every command; `make <target>` (Linux/macOS/WSL) and `python tasks.py <target>`
(Windows) are the same thing. `python tasks.py help` lists them; `[REAL DATA]` ones read the match
bundle — run them yourself and share only the aggregates they print.

## Reproduce the submission

Needs Python 3.12–3.13, git, and the match bundle (released to registered participants; not in
this repo, never committed).

```powershell
cd JS
python tasks.py venv ; .venv\Scripts\Activate.ps1   # Python 3.13 venv (uv if installed, else python -m venv)
python tasks.py setup                                # pinned deps + official repo (vendor/) + tests
# unpack the bundle into the fork's data/  (data/tactical/..., data/broadcast/...)  or set MTAH_DATA_DIR
python tasks.py submit --team <team_name>            # -> results/<team_name>.csv, officially validated
python tasks.py reproduce --team <team_name>         # regenerate + byte-compare
```

The run is deterministic: same data + same code gives a byte-identical file.

## Develop without the competition data

```powershell
python tasks.py test                                 # synthetic match + harness, official scorer included
python tasks.py harness                              # validation harness on synthetic games
# IDSSE needs its own venv: floodlight 1.2.0 requires pandas < 3, the submission pins pandas 3.0.5
python tasks.py venv-harness ; .venv-harness\Scripts\Activate.ps1
python tasks.py idsse-cache                          # parse external_data/idsse once, cached in external/idsse/
python tasks.py harness-idsse                        # 7 IDSSE games (2 Bundesliga + 5 2. Bundesliga) x 3 seeds
python tasks.py fetch-skillcorner                    # SkillCorner tracking: LFS pointers -> real files
python tasks.py bundles                              # IDSSE / Metrica / SkillCorner -> tactical-format bundles (no noise, native fps)
python tasks.py eval-bundles --tag v0                # pipeline on the bundles, official metric, per source
# ball imputation (GPU machine; pip install -r requirements-model.txt, torch for your CUDA first)
python tasks.py train-ball-all --tag v1               # A / B x player mask off / on -> models/ball_*.pt
python tasks.py eval-ball --tag v1                   # buckets, team-nearest agreement, downstream spellF1
```

A bundle (`external/bundles/<GAME>/`) is another game in the tactical data definition: EPTS-style
meta + raw tracking, `frame,label` ground truth, `game.json` (frames, fps, pitch, halves, label
provenance). Nothing is added or resampled: positions, missing values, identities, frame rate and
pitch are the source's; only the normalisation and, where a source has none, the speed field
(centred difference of its own positions) are filled in. See `mqf/external/bundle.py`.

## Watch a match with its labels

```bash
python -m mqf.viewer                                 # BAR-ATH: tactical / broadcast video or 2D pitch,
                                                     # public GT + every results/*.csv as lanes
python -m mqf.viewer --pred results/a.csv --pred results/b.csv
python -m mqf.viewer.demo && python -m mqf.viewer --synthetic   # no match data needed (ffmpeg for the videos)
```

A local page (server on 127.0.0.1) with one playhead: the video or the pitch, then the ground truth
and each model as colour lanes (whole match + a zoomable lens), where they disagree, and what the
tracking saw at that frame. The broadcast feed is linked to the tactical timeline algorithmically
from the two tracking files (`mqf/viewer/sync.py`, cached in `cache/`). On BAR-ATH the ground truth
is the public 5 minutes: use it to check correctness, never to tune. Keep screenshots of the real
match out of AI chats.

## Layout

```
mqf/
  config.py          paths; match constants via the official games.py
  atd.py             FIFA/EPTS raw tracking -> arrays (cached)
  smart_tagging.py   SportsCode XML -> soft possession evidence
  features.py        per-frame features
  decode.py          emission model + HMM/Viterbi per half
  submission.py      writer + official validator/scorer
  synthetic.py       synthetic match (real file formats, known truth)
  make_submission.py CLI
  harness/           validation on other games: ATD noise model, IDSSE adapter, official-metric scoring
  external/          other games: SkillCorner LFS fetcher (converters to the tactical format next)
  viewer/            local match viewer: video / pitch + GT and model lanes, broadcast sync
tests/               synthetic e2e (official scorer + validator, determinism, ATD round-trip), harness, fetcher
tasks.py  Makefile   every command (python tasks.py help)
bootstrap.py         official repo -> vendor/mtah2027 at the pinned commit; --check-updates, --status
docs/                CONTEXT, EXPERIMENTS, KICKOFF_PROMPT, TECHNICAL, CLAUDE
```

External data: IDSSE — Bassek, Weber, Rein & Memmert (2025), *An integrated dataset of
spatiotemporal and event data in elite soccer*, Scientific Data 12:195. Used for validation only.

Code: MIT. The match data is licensed for the competition only and is never committed.
