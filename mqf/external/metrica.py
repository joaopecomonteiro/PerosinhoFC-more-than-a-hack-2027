"""Metrica Sports sample data (github.com/metrica-sports/sample-data) -> clean match for a bundle.

Three anonymised games, 25 fps, coordinates already normalised 0-1 on a 105 x 68 m pitch
((0,0) top left), synchronised manual events. Game 1 and 2 are CSV (one file per team); game 3 is
FIFA/EPTS (metadata XML + raw text, the same family as the competition ATD) with JSON events.
Kept as is: frame rate, positions, the ball's NaNs, every player in his own slot.

There is no per-frame possession in this data, so the labels come from the events (Metrica's own
manual tagging, synced to the tracking). Rules, matching the competition's GT semantics
(DATA.md / CONTEXT: loose ball stays with the last team; a change needs controlled possession):
  dead    from the frame the ball leaves play -- BALL OUT end, SHOT ...OUT / ...GOAL end -- until
          the next on-ball event starts (the restart: SET PIECE, or a PASS / RECOVERY that takes
          it); and before every SET PIECE that is not a half's opening kick-off, from the end of
          the last on-ball event before it (fouls, offsides, injuries)
  owner   the team of the last controlling event (PASS, CARRY, SHOT, SET PIECE, RECOVERY);
          BALL LOST and CHALLENGE never switch -- the opponent's RECOVERY or first pass does
  halves  from each period's first event to the end of its last event (inside the tracking)
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EXTERNAL_RAW_DIR

FPS = 25
PITCH = (105.0, 68.0)                     # README of the sample data, and FieldSize in game 3's metadata
GAMES = (1, 2, 3)
LICENCE = "Metrica Sports sample data -- free use, acknowledge the source"
CONTROL = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY"}
ON_BALL = CONTROL | {"BALL LOST", "CHALLENGE", "BALL OUT", "FAULT RECEIVED"}


def raw_dir() -> Path:
    return EXTERNAL_RAW_DIR / "metrica" / "data"


# ------------------------------------------------------------------------------------ tracking

def _read_team_csv(path: Path):
    df = pd.read_csv(path, skiprows=2)
    cols = list(df.columns)
    players = [c for c in cols if c.startswith("Player")]
    xy = np.stack([np.stack([df[c].to_numpy(float), df[cols[cols.index(c) + 1]].to_numpy(float)], axis=1)
                   for c in players], axis=1)                                  # (n, P, 2) normalised
    bi = cols.index("Ball")
    ball = np.stack([df[cols[bi]].to_numpy(float), df[cols[bi + 1]].to_numpy(float)], axis=1)
    return df["Frame"].to_numpy(int), df["Period"].to_numpy(int), xy, ball, players


def load_csv_tracking(game: int, root: Path | None = None) -> dict:
    d = Path(root or raw_dir()) / f"Sample_Game_{game}"
    fh, ph, xh, bh, nh = _read_team_csv(d / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
    fa, _, xa, _, na = _read_team_csv(d / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv")
    if not np.array_equal(fh, fa):
        raise ValueError(f"game {game}: home and away files are not on the same frames")
    return {"frame": fh, "period": ph, "players": np.concatenate([xh, xa], axis=1), "ball": bh,
            "team_of_player": np.r_[np.zeros(len(nh), np.int8), np.ones(len(na), np.int8)],
            "slot_names": [f"Home {n}" for n in nh] + [f"Away {n}" for n in na],
            "home": "Home", "away": "Away"}


def _strip_ns(text: str) -> ET.Element:
    text = re.sub(r'\sxmlns(:\w+)?="[^"]+"', "", text)
    text = re.sub(r'\s\w+:schemaLocation="[^"]+"', "", text)
    return ET.fromstring(text)


def load_epts_tracking(game: int = 3, root: Path | None = None) -> dict:
    d = Path(root or raw_dir()) / f"Sample_Game_{game}"
    meta = _strip_ns((d / f"Sample_Game_{game}_metadata.xml").read_text(encoding="utf-8"))
    teams = [(t.get("id"), t.findtext("Name")) for t in meta.iter("Team")]
    team_idx = {tid: i for i, (tid, _) in enumerate(teams)}                   # first Team = home
    plist = [(p.get("id"), p.get("teamId"), p.findtext("Name")) for p in meta.iter("Player")]
    plist.sort(key=lambda p: team_idx[p[1]])                                   # home slots first
    slot = {pid: k for k, (pid, _, _) in enumerate(plist)}
    channel_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    specs = []
    for dfs in meta.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in dfs.iter("PlayerChannelRef")]
        order = [slot[channel_player[r]] for r in refs if r.endswith("_x")]
        specs.append((int(dfs.get("startFrame")), int(dfs.get("endFrame")), np.array(order)))
    period_of = {}
    for pp in meta.iter("ProviderParameter"):
        period_of[pp.findtext("Name")] = pp.findtext("Value")

    lines = (d / f"Sample_Game_{game}_tracking.txt").read_text().splitlines()
    n = len(lines)
    frame = np.empty(n, dtype=np.int64)
    players = np.full((n, len(plist), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for i, line in enumerate(lines):
        fid, pl, bl = line.split(":")
        f = int(fid)
        frame[i] = f
        order = next(o for s, e, o in specs if s <= f <= e)
        vals = np.array([v.split(",") for v in pl.split(";")], dtype=float)  # 'NaN' parses
        players[i, order] = vals[:, :2]
        ball[i] = np.array(bl.split(",")[:2], dtype=float)
    period = np.where(frame >= int(period_of.get("second_half_start") or 10 ** 12), 2, 1)
    return {"frame": frame, "period": period, "players": players, "ball": ball,
            "team_of_player": np.array([team_idx[t] for _, t, _ in plist], dtype=np.int8),
            "slot_names": [nm or pid for pid, _, nm in plist], "home": teams[0][1], "away": teams[1][1]}


# ------------------------------------------------------------------------------------ events

def load_events(game: int, root: Path | None = None) -> pd.DataFrame:
    """Normalised events: team (0 home / 1 away / -1), type, subtype (upper), start, end (frame ids), period."""
    d = Path(root or raw_dir()) / f"Sample_Game_{game}"
    if game in (1, 2):
        e = pd.read_csv(d / f"Sample_Game_{game}_RawEventsData.csv")
        out = pd.DataFrame({"team": e["Team"].map({"Home": 0, "Away": 1}).fillna(-1).astype(int),
                            "type": e["Type"].str.upper().str.strip(),
                            "subtype": e["Subtype"].fillna("").str.upper().str.strip(),
                            "start": e["Start Frame"].astype(int), "end": e["End Frame"].astype(int),
                            "period": e["Period"].astype(int)})
    else:
        raw = json.loads((d / f"Sample_Game_{game}_events.json").read_text(encoding="utf-8"))
        meta = _strip_ns((d / f"Sample_Game_{game}_metadata.xml").read_text(encoding="utf-8"))
        team_idx = {t.get("id"): i for i, t in enumerate(meta.iter("Team"))}
        rows = []
        for ev in raw["data"] if isinstance(raw, dict) else raw:
            sub = ev.get("subtypes")
            subs = [sub] if isinstance(sub, dict) else (sub or [])
            rows.append({"team": team_idx.get((ev.get("team") or {}).get("id"), -1),
                         "type": ev["type"]["name"].upper().strip(),
                         "subtype": "-".join(s["name"].upper() for s in subs if s and s.get("name")),
                         "start": int(ev["start"]["frame"]), "end": int(ev["end"]["frame"] or 0),
                         "period": int(ev["period"])})
        out = pd.DataFrame(rows)
    out["end"] = np.maximum(out["end"], out["start"])
    return out.sort_values(["start", "end"], kind="stable").reset_index(drop=True)


def labels_from_events(ev: pd.DataFrame, frame: np.ndarray) -> tuple[np.ndarray, tuple]:
    """(labels per row of `frame`, halves in row indices). See the module docstring for the rules."""
    n = len(frame)
    first_id, last_id = int(frame[0]), int(frame[-1])
    if last_id - first_id + 1 != n:
        raise ValueError("tracking frames must be contiguous")
    labels = np.full(n, "X", dtype="<U1")
    halves = []
    for half, e in ev.groupby("period", sort=True):
        e = e[e["type"].isin(ON_BALL)].reset_index(drop=True)
        if e.empty:
            continue
        a = max(int(e["start"].min()), first_id)
        b = min(int(e["end"].max()), last_id)
        ids = np.arange(a, b + 1)
        dead = np.zeros(len(ids), dtype=bool)

        def mark(s, t):                                    # frame ids [s, t)
            s, t = max(s, a), min(t, b + 1)
            if t > s:
                dead[s - a:t - a] = True

        starts = e["start"].to_numpy()
        for i, r in e.iterrows():
            out_of_play = (r["type"] == "BALL OUT") or (r["type"] == "SHOT" and
                                                         ("OUT" in r["subtype"] or "GOAL" in r["subtype"]))
            if out_of_play:
                nxt = np.flatnonzero(starts > r["end"])
                mark(int(r["end"]), int(starts[nxt[0]]) if len(nxt) else b + 1)
            if r["type"] == "SET PIECE" and i > 0:
                prev = e.iloc[:i]
                prev = prev[prev["start"] < r["start"]]
                if len(prev):
                    mark(int(prev["end"].max()), int(r["start"]))

        ctrl = e[e["type"].isin(CONTROL) & (e["team"] >= 0)]
        cs, ct = ctrl["start"].to_numpy(), ctrl["team"].to_numpy()
        k = np.searchsorted(cs, ids, side="right") - 1
        owner = np.where(k >= 0, ct[np.clip(k, 0, None)], ct[0] if len(ct) else 0)
        lab = np.where(dead, "D", np.where(owner == 0, "H", "A"))
        rows = ids - first_id
        labels[rows] = lab
        halves.append((int(half), int(rows[0]), int(rows[-1])))
    return labels, tuple(halves)


# ------------------------------------------------------------------------------------ match

def load_match(game: int, root: Path | None = None) -> dict:
    trk = load_csv_tracking(game, root) if game in (1, 2) else load_epts_tracking(game, root)
    if np.any(np.diff(trk["frame"]) != 1):
        raise ValueError(f"game {game}: tracking frames are not contiguous")
    labels, halves = labels_from_events(load_events(game, root), trk["frame"])
    L, W = PITCH
    scale = np.array([L, W])
    return {"name": f"METRICA-{game}", "source": "metrica", "fps": FPS, "pitch": PITCH,
            "players": trk["players"] * scale, "ball": trk["ball"] * scale,
            "team_of_player": trk["team_of_player"], "slot_names": trk["slot_names"],
            "labels": labels, "halves": halves, "home": trk["home"], "away": trk["away"],
            "first_frame_id": int(trk["frame"][0]),
            "labels_from": "Metrica manual events (synced): dead from ball out / goal / before set pieces "
                           "to the restart; owner = last controlling event (loose ball stays)",
            "licence": LICENCE}
