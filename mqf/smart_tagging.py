"""Smart Tagging (SportsCode XML) -> per-frame soft evidence. Evidence, never labels.

<start>/<end> are seconds from the first tactical-video frame: video_frame = round(seconds * fps).
Codes are UPPER CASE, tag values Title Case, and `Team` can be "N/A" (every throw-in). Match
case-insensitively. Club names in the XML ("Atletic Club") differ from the ATD metadata, so they
are mapped to H/A by fuzzy match.
"""
from __future__ import annotations

import difflib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Phase codes whose Team tag names the team IN possession (+1) or OUT of possession (-1).
POSSESSING = {"build up", "progression", "creating chances", "attacking transition", "long ball",
              "ball in final third", "ball in the box", "players in the box", "goals"}
DEFENDING = {"defending in defensive third", "defending in middle third",
             "defending in attacking third", "defensive transition"}
SET_PIECES = "set pieces"


@dataclass
class TagEvidence:
    vote: np.ndarray        # (n_frames, 2) summed evidence for H / A in possession
    restart: np.ndarray     # (n_frames,) 1 inside a tagged set-piece window
    restart_team: np.ndarray  # (n_frames,) 0/1 team restarting, -1 unknown


def load_instances(path: Path, fps: int) -> list[dict]:
    out = []
    for inst in ET.parse(path).iter("instance"):
        out.append({
            "code": (inst.findtext("code") or "").strip().lower(),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
            "tags": {(lb.findtext("group") or "").strip().lower(): (lb.findtext("text") or "").strip()
                     for lb in inst.findall("label")},
        })
    return out


def team_names_from_meta(meta_path: Path) -> tuple[str, str]:
    root = ET.parse(meta_path).getroot()
    names = [t.findtext("Name") or "" for t in root.iter("Team")]
    return names[0], names[1]


def side_of(name: str, home: str, away: str) -> int:
    """0 home, 1 away, -1 unknown/N/A."""
    if not name or name.upper() == "N/A":
        return -1
    r = [difflib.SequenceMatcher(None, name.lower(), n.lower()).ratio() for n in (home, away)]
    return int(np.argmax(r)) if max(r) > 0.4 else -1


def evidence(path: Path, meta_path: Path, n_frames: int, fps: int) -> TagEvidence:
    home, away = team_names_from_meta(meta_path)
    vote = np.zeros((n_frames, 2), dtype=np.float32)
    restart = np.zeros(n_frames, dtype=np.int8)
    restart_team = np.full(n_frames, -1, dtype=np.int8)
    for ev in load_instances(path, fps):
        s, e = max(0, ev["start"]), min(n_frames, ev["end"])
        if e <= s:
            continue
        side = side_of(ev["tags"].get("team", ""), home, away)
        if ev["code"] == SET_PIECES:
            restart[s:e] = 1
            if side >= 0:
                restart_team[s:e] = side
            continue
        if side < 0:
            continue
        if ev["code"] in POSSESSING:
            vote[s:e, side] += 1.0
        elif ev["code"] in DEFENDING:
            vote[s:e, 1 - side] += 1.0
    return TagEvidence(vote, restart, restart_team)


def kickoff_windows(path: Path, fps: int, n_frames: int) -> list[tuple[int, int]]:
    """[(first, last)] frames of the tagged kickoffs (Type "Kick Off (Start)" / "Kick Off (After Goal)").
    Windows open ~3.5 s before the restart and contain it; the instant comes from tracking."""
    out = []
    for ev in load_instances(path, fps):
        if ev["tags"].get("type", "").strip().lower().startswith("kick off"):
            a, b = max(0, ev["start"]), min(n_frames - 1, ev["end"])
            if b >= a:
                out.append((a, b))
    return sorted(out)

