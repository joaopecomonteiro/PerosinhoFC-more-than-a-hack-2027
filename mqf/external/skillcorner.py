"""SkillCorner open data (github.com/SkillCorner/opendata, MIT) -> clean match for a bundle.

20 A-League 2024/25 matches of BROADCAST tracking at 10 fps: players off screen are extrapolated
(`is_detected` false), frames without a live picture (replays, cut-aways, halftime) have no data,
pitch size per match in match.json, coordinates in metres from the centre spot. Kept as is:
10 fps, every position and missing value, one slot per player in the match sheet. Coordinates are
only shifted to the corner origin of the tactical definition (no flip, no rescale).

Labels (SkillCorner's automatic data, not human):
  alive   inside a phase of play -- SkillCorner defines phases only while the ball is in play
  owner   the per-frame `possession.group` (home / away team), carried forward while nobody is
          in possession (loose ball stays with the last team); at the start of a phase with no
          possessor yet, the phase's team in possession
  halves  match_periods start..end frames
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EXTERNAL_RAW_DIR

FPS = 10
LICENCE = "SkillCorner open data, MIT licence"
GROUP = {"home team": 0, "away team": 1}


def raw_dir() -> Path:
    return EXTERNAL_RAW_DIR / "skill_corner" / "data" / "matches"


def match_ids(root: Path | None = None) -> list[str]:
    return sorted(p.name for p in Path(root or raw_dir()).iterdir()
                  if (p / f"{p.name}_tracking_extrapolated.jsonl").is_file())


def labels_from_phases(n: int, halves, phases: pd.DataFrame, group: np.ndarray, home_id: int) -> np.ndarray:
    """group: (n,) 0 home / 1 away / -1 nobody, per frame. phases: frame_start, frame_end, team_in_possession_id."""
    labels = np.full(n, "X", dtype="<U1")
    for _, a, b in halves:
        labels[a:b + 1] = "D"
    phase_team = np.full(n, -1, dtype=np.int8)
    alive = np.zeros(n, dtype=bool)
    for r in phases.itertuples(index=False):
        s, e = max(int(r.frame_start), 0), min(int(r.frame_end), n - 1)
        if e < s:
            continue
        alive[s:e + 1] = True
        phase_team[s:e + 1] = 0 if int(r.team_in_possession_id) == home_id else 1
    in_half = labels != "X"
    alive &= in_half
    run = np.cumsum(np.r_[True, alive[1:] != alive[:-1]])               # id of each alive/dead run
    owner = pd.Series(np.where(group >= 0, group, np.nan)).where(alive)
    owner = owner.groupby(run).ffill()                                   # carried inside one alive run
    owner = owner.fillna(pd.Series(phase_team.astype(float)).where(alive & (phase_team >= 0)))
    owner = owner.groupby(run).ffill().to_numpy()
    lab = np.where(owner == 0, "H", np.where(owner == 1, "A", "D"))
    labels[alive] = lab[alive]
    return labels


def load_match(match_id: str, root: Path | None = None) -> dict:
    d = Path(root or raw_dir()) / str(match_id)
    meta = json.loads((d / f"{match_id}_match.json").read_text(encoding="utf-8"))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = int(meta["home_team"]["id"])
    sheet = sorted(meta["players"], key=lambda p: (int(p["team_id"]) != home_id, p.get("number") or 0))
    slot = {int(p["id"]): k for k, p in enumerate(sheet)}

    recs = [json.loads(line) for line in open(d / f"{match_id}_tracking_extrapolated.jsonl", encoding="utf-8")
            if line.strip()]
    frames = np.array([r["frame"] for r in recs], dtype=np.int64)
    if frames[0] != 0 or np.any(np.diff(frames) != 1):
        raise ValueError(f"{match_id}: frames are not 0..n-1")
    n = len(recs)
    players = np.full((n, len(sheet), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    group = np.full(n, -1, dtype=np.int8)
    for i, r in enumerate(recs):
        for p in r.get("player_data") or []:
            k = slot.get(int(p["player_id"]))
            if k is not None and p.get("x") is not None:
                players[i, k] = (p["x"] + L / 2, p["y"] + W / 2)
        b = r.get("ball_data") or {}
        if b.get("x") is not None:
            ball[i] = (b["x"] + L / 2, b["y"] + W / 2)
        group[i] = GROUP.get((r.get("possession") or {}).get("group"), -1)
    halves = tuple((int(p["period"]), int(p["start_frame"]), min(int(p["end_frame"]), n - 1))
                   for p in meta["match_periods"])
    phases = pd.read_csv(d / f"{match_id}_phases_of_play.csv")
    labels = labels_from_phases(n, halves, phases, group, home_id)
    names = [f"{p.get('short_name') or p['id']}" for p in sheet]
    return {"name": f"SKC-{match_id}", "source": "skillcorner", "fps": FPS, "pitch": (L, W),
            "players": players, "ball": ball,
            "team_of_player": np.array([0 if int(p["team_id"]) == home_id else 1 for p in sheet], dtype=np.int8),
            "slot_names": names, "labels": labels, "halves": halves,
            "home": meta["home_team"]["name"], "away": meta["away_team"]["name"],
            "labels_from": "SkillCorner automatic data: alive = phases of play; owner = per-frame possession "
                           "group carried forward (loose ball stays), phase team at phase start",
            "licence": LICENCE}
