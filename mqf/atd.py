"""Read Metrica's FIFA/EPTS Automatic Tracking Data into dense numpy arrays, cached as .npz.

Raw line format (see the official starter/baseline.py):
    "<frame_id>:<x,y,s>;<x,y,s>;...;<x,y,s>:<ball_x,ball_y,ball_s>"
Coordinates are normalised 0-1; missing observations are NaN. frame_id counts from 1, so
video_frame = frame_id - 1. One record per video frame.

Track slots are tracklets, NOT players: 46 slots for 22 players, with gaps and identity switches.
Only team membership per slot is meaningful.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, GameSpec


@dataclass
class Tracking:
    players: np.ndarray      # (n_frames, n_slots, 2) metres, NaN if missing
    player_speed: np.ndarray  # (n_frames, n_slots) m/s, NaN if missing
    ball: np.ndarray         # (n_frames, 2) metres, NaN if missing
    team_of_slot: np.ndarray  # (n_slots,) 0 = home, 1 = away

    @property
    def n_frames(self) -> int:
        return self.players.shape[0]

    @property
    def ball_seen(self) -> np.ndarray:
        return ~np.isnan(self.ball[:, 0])


def team_of_each_slot(meta_path: Path) -> np.ndarray:
    """0 for home, 1 for away. First <Team> in the metadata is home (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def _parse_group(group: str) -> tuple[float, float, float]:
    if not group or group[0] == "N":
        return (np.nan, np.nan, np.nan)
    v = group.split(",")
    s = float(v[2]) if len(v) > 2 and v[2] and v[2][0] != "N" else np.nan
    return (float(v[0]), float(v[1]), s)


def parse_raw(raw_path: Path, n_frames: int, n_slots: int, pitch: tuple[float, float]) -> Tracking:
    length, width = pitch
    xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    sp = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < n_frames:
                continue
            for k, g in enumerate(parts[1].split(";")[:n_slots]):
                x, y, s = _parse_group(g)
                xy[f, k, 0], xy[f, k, 1], sp[f, k] = x * length, y * width, s
            bx, by, _ = _parse_group(parts[2])
            ball[f] = (bx * length, by * width)
    return Tracking(xy, sp, ball, np.zeros(n_slots, dtype=np.int8))


def load(spec: GameSpec, use_cache: bool = True, cache_dir: Path | None = None) -> Tracking:
    """Parse once (a minute or two on the real match), then load from cache in seconds.

    cache_dir: where the .npz goes (default cache/). Other games pass their own folder so their
    arrays never sit next to the ones derived from the real match.
    """
    team = team_of_each_slot(spec.atd_meta)
    stat = Path(spec.atd_raw).stat()
    key = hashlib.md5(f"{spec.atd_raw}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:10]
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache = cache_dir / f"{spec.name}_atd_{key}.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        return Tracking(z["players"], z["player_speed"], z["ball"], z["team_of_slot"])
    trk = parse_raw(spec.atd_raw, spec.n_frames, len(team), spec.pitch)
    trk.team_of_slot = team
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, players=trk.players, player_speed=trk.player_speed,
                            ball=trk.ball, team_of_slot=trk.team_of_slot)
    return trk
