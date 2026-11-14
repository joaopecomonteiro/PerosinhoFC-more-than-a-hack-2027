"""PFF FC World Cup 2022 tracking -> a bundle in the tactical data definition (as is).

    python -m mqf.external.build --source pff --jobs 4          # every <id>.jsonl.bz2 found
    python -m mqf.external.build --source pff --matches 3831

Raw files: `<gameId>.jsonl.bz2`, one JSON record per video frame (PFF FC Tracking Data Specification
v2.2), in external_data/pff/ or JS/external/pff/ (first found; MQF_PFF_DIR overrides). 64 games,
29.97 fps, Sportlogiq BROADCAST tracking synchronised with PFF's manual event data.

What is kept, as delivered:
  frames      every frameNum (duplicated frames -- one per extra possession event -- kept once);
              bundle frame i = frameNum i, so the pre-kickoff video before the first record is empty
  fps         29.97 (from videoTimeMs / frameNum), never resampled
  players     the RAW `homePlayers` / `awayPlayers` (not the Kalman-smoothed lists), VISIBLE and
              ESTIMATED: broadcast tracking estimates the off-screen players from context, which keeps
              ~11 per team on every frame -- close to the tactical camera's ~20.5 tracked. One slot per
              jersey number per team (jerseys are estimated, mostly LOW confidence; a second player
              with the same jersey on one frame is dropped)
  ball        the RAW `balls` entry where its visibility is VISIBLE; ESTIMATED balls (filled in by
              Sportlogiq from context, mostly in the air) and frames with no ball are missing. The
              event-smoothed `ballsSmoothed` (ball glued to the carrier) is never used
  pitch       105 x 68 m (no metadata files in the download; every World Cup 2022 pitch is 105 x 68
              and the positions stay within it); origin moved from the centre to the corner
  halves      periods 1-4 (extra time included), first to last frame of each period

Labels (derived by us from PFF's manual game events, carried on the tracking frames):
  owner  the team of the player in control (`home_team`) of the last on-the-ball / kick-off game
         event, carried forward -- the same rule as the Metrica labels (last controlling event)
  dead   from an OUT game event (PFF: ball out of play -- includes fouls and offsides: in the sample
         every event gap > 8 s follows an OUT) to the start of the next controlling event (the
         restart); from END to the end of the period; VID (video missing); before the period's
         kick-off. Backup for a stoppage without OUT (e.g. a goal): a gap > GAP_DEAD_S between
         controlling events is dead from 1 s after the last control to the next one (counted in QA).
"""
from __future__ import annotations

import bz2
import json
import os
from pathlib import Path

import numpy as np

from ..config import EXTERNAL_DIR, EXTERNAL_RAW_DIR

PITCH = (105.0, 68.0)
CONTROL = ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF")
GAP_DEAD_S = 10.0
DEAD_LAG_S = 1.0
LICENCE = ("PFF FC World Cup 2022 data (PFF event data + Sportlogiq broadcast tracking), "
           "obtained by the team from PFF FC under PFF's terms; raw files are not redistributed")


def raw_dir() -> Path:
    if os.environ.get("MQF_PFF_DIR"):
        return Path(os.environ["MQF_PFF_DIR"])
    for d in (EXTERNAL_RAW_DIR / "pff", EXTERNAL_DIR / "pff"):
        if d.exists() and any(d.glob("*.jsonl.bz2")):
            return d
    return EXTERNAL_RAW_DIR / "pff"


def match_ids(root: Path | None = None) -> list[str]:
    return sorted((p.name.split(".")[0] for p in Path(root or raw_dir()).glob("*.jsonl.bz2")), key=int)


def _flag(v) -> int:
    return 1 if v in (True, 1, "1", "true", "True") else 0


def read(path: Path) -> dict:
    """Stream one game: per frame (deduplicated) period, players by team / jersey, visible ball;
    game events at their start frame."""
    frames, periods, vt = [], [], []
    home, away, ball = [], [], []
    events = {}
    seen = set()
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            fn = int(r["frameNum"])
            ge = r.get("game_event")
            gid = (r.get("game_event_id"), ge.get("start_frame"), ge.get("game_event_type")) if ge else None
            if ge and ge.get("start_frame") is not None and gid not in events:   # an event spans many frames
                events[gid] = {"type": ge.get("game_event_type"), "start": int(ge["start_frame"]),
                               "end": int(ge.get("end_frame") if ge.get("end_frame") is not None else ge["start_frame"]),
                               "home_team": _flag(ge.get("home_team")), "home_ball": _flag(ge.get("home_ball")),
                               "team": ge.get("team_name"), "period": int(r["period"])}
            if fn in seen:
                continue
            seen.add(fn)
            frames.append(fn)
            periods.append(int(r["period"]) if r.get("period") is not None else 0)
            vt.append(float(r["videoTimeMs"]) if r.get("videoTimeMs") is not None else np.nan)
            home.append([(str(p.get("jerseyNum")), p.get("x"), p.get("y")) for p in (r.get("homePlayers") or [])])
            away.append([(str(p.get("jerseyNum")), p.get("x"), p.get("y")) for p in (r.get("awayPlayers") or [])])
            b = next((b for b in (r.get("balls") or []) if b.get("x") is not None), None)
            ball.append((b["x"], b["y"]) if b is not None and b.get("visibility") == "VISIBLE" else (np.nan, np.nan))
    return {"frames": np.array(frames, dtype=np.int64), "periods": np.array(periods, dtype=np.int64),
            "video_ms": np.array(vt, dtype=np.float64), "home": home, "away": away,
            "ball": np.array(ball, dtype=np.float64), "events": sorted(events.values(), key=lambda e: e["start"])}


def fps_of(frames: np.ndarray, video_ms: np.ndarray) -> float:
    ok = ~np.isnan(video_ms)
    f, t = frames[ok], video_ms[ok]
    rate = (f[-1] - f[0]) / ((t[-1] - t[0]) / 1000.0)
    return round(float(rate), 3)


def slots(lists) -> list[str]:
    jerseys = {j for frame in lists for j, x, y in frame if x is not None}
    return sorted(jerseys, key=lambda j: (not j.isdigit(), int(j) if j.isdigit() else 0, j))


def labels_from_events(n: int, halves, events, fps: float) -> tuple[np.ndarray, dict]:
    """H / A / D per frame inside the halves, X outside (rules in the module docstring)."""
    lab = np.full(n, "X", dtype="<U1")
    stats = {"out": 0, "backup_gaps": 0, "vid": 0}
    for period, a, b in halves:
        ev = [e for e in events if e["period"] == period and a <= e["start"] <= b]
        ctrl = [e for e in ev if e["type"] in CONTROL]
        owner = np.full(b - a + 1, -1, dtype=np.int64)            # -1: before the first control (dead)
        for e in ctrl:
            owner[e["start"] - a:] = e["home_team"]
        seg = np.where(owner == 1, "H", np.where(owner == 0, "A", "D"))
        starts = np.array([e["start"] for e in ctrl], dtype=np.int64)
        for e in ev:
            if e["type"] == "OUT":
                nxt = starts[starts > e["start"]]
                stop = int(nxt[0]) if len(nxt) else b + 1
                seg[e["start"] - a:stop - a] = "D"
                stats["out"] += 1
            elif e["type"] == "END":
                seg[e["start"] - a:] = "D"
            elif e["type"] == "VID":
                seg[e["start"] - a:e["end"] - a + 1] = "D"
                stats["vid"] += 1
        outs = sorted(e["start"] for e in ev if e["type"] == "OUT")
        for p, q in zip(ctrl, ctrl[1:]):                          # stoppage without an OUT (e.g. a goal)
            if (q["start"] - p["end"]) / fps > GAP_DEAD_S and not any(p["end"] <= o < q["start"] for o in outs):
                s = p["end"] + int(round(DEAD_LAG_S * fps))
                seg[max(s, p["start"]) - a:q["start"] - a] = "D"
                stats["backup_gaps"] += 1
        lab[a:b + 1] = seg
    return lab, stats


def load_match(game_id, root: Path | None = None) -> dict:
    path = Path(root or raw_dir()) / f"{game_id}.jsonl.bz2"
    r = read(path)
    fps = fps_of(r["frames"], r["video_ms"])
    n = int(r["frames"].max()) + 1
    hs, as_ = slots(r["home"]), slots(r["away"])
    P = len(hs) + len(as_)
    col = {("h", j): i for i, j in enumerate(hs)} | {("a", j): len(hs) + i for i, j in enumerate(as_)}
    L, W = PITCH
    players = np.full((n, P, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    for k, fn in enumerate(r["frames"]):
        for side, lst in (("h", r["home"][k]), ("a", r["away"][k])):
            used = set()
            for j, x, y in lst:
                if x is None or y is None or j in used:
                    continue
                used.add(j)
                players[fn, col[(side, j)]] = (x + L / 2, y + W / 2)
    ball[r["frames"]] = r["ball"] + np.array([L / 2, W / 2])
    halves = []
    for p in sorted(set(r["periods"].tolist()) & {1, 2, 3, 4}):
        f = r["frames"][r["periods"] == p]
        halves.append((int(p), int(f.min()), int(f.max())))
    labels, stats = labels_from_events(n, halves, r["events"], fps)
    team = np.r_[np.zeros(len(hs)), np.ones(len(as_))].astype(np.int8)
    names = {e["home_team"]: e["team"] for e in r["events"] if e.get("team")}
    return {"name": f"PFF-{game_id}", "source": "pff", "fps": fps, "pitch": PITCH, "players": players,
            "team_of_player": team, "ball": ball, "labels": labels, "halves": tuple(halves),
            "home": names.get(1, "home"), "away": names.get(0, "away"),
            "slot_names": [f"home #{j}" for j in hs] + [f"away #{j}" for j in as_],
            "labels_from": ("derived by us from PFF manual game events: dead = OUT to the next controlling "
                            "event, END to period end, VID; owner = team of the player in control of the last "
                            f"on-the-ball / kick-off event (backup gaps > {GAP_DEAD_S:g} s: {stats['backup_gaps']})"),
            "licence": LICENCE, "label_stats": stats}


def build_one(game_id) -> tuple[str, float, dict, float]:
    """Convert and write one game (a process-pool worker): (name, fps, qa, seconds)."""
    import time
    from . import bundle
    t0 = time.time()
    m = load_match(game_id)
    bundle.write_bundle(m)
    return m["name"], m["fps"], bundle.qa(m), time.time() - t0
