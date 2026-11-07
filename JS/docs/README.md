# Més que un Frame

**Robust possession spells from uncorrected tracking.** Our entry to
[More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027), Phase 1
(FC Barcelona × Metrica Sports × Mobile World Capital Barcelona).

For every frame of FC Barcelona vs Athletic Club (LaLiga 2024-25, MD2) we output `H` / `A` / `D`:
per-frame evidence from automatic tracking and Smart Tagging, decoded into whole possessions by a
three-state hidden Markov model per half. Approach: [TECHNICAL.md](TECHNICAL.md).

## Reproduce the submission

Needs Python 3.12–3.13, git, and the match bundle (released to registered participants; not in
this repo, never committed).

```bash
python bootstrap.py                                  # official repo -> vendor/ (pinned commit)
pip install -r requirements.txt                      # pinned
# unpack the bundle into ./data  (data/tactical/..., data/broadcast/...)  or set MTAH_DATA_DIR
python -m mqf.make_submission --team <team_name>     # -> results/<team_name>.csv, officially validated
```

The run is deterministic: same data + same code gives a byte-identical file.

## Develop without the competition data

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                  # synthetic match + harness, official scorer included
python -m mqf.harness.run --source synthetic         # validation harness on synthetic games
# IDSSE needs its own venv: floodlight 1.2.0 requires pandas < 3, the submission pins pandas 3.0.5
python -m venv .venv-harness && . .venv-harness/bin/activate   # Windows: .venv-harness\Scripts\Activate.ps1
pip install -r requirements-harness.txt
python -m mqf.harness.idsse J03WMX                   # download + parse once, cached in external/idsse/
python -m mqf.harness.run --source idsse --matches all --seeds 0 1 2   # 7 Bundesliga games (IDSSE)
```

`make help` lists shortcuts (Linux/macOS/WSL).

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
  viewer/            local match viewer: video / pitch + GT and model lanes, broadcast sync
tests/  docs/ (CONTEXT, EXPERIMENTS, KICKOFF_PROMPT)  TECHNICAL.md  bootstrap.py
```

External data: IDSSE — Bassek, Weber, Rein & Memmert (2025), *An integrated dataset of
spatiotemporal and event data in elite soccer*, Scientific Data 12:195. Used for validation only.

Code: MIT. The match data is licensed for the competition only and is never committed.
