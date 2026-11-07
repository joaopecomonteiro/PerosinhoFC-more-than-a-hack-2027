"""IDSSE (Bassek et al., Scientific Data 2025) -> clean match in our harness format.

7 Bundesliga 2022/23 matches, official DFL tracking at 25 fps with PER-FRAME ball status
(0 Dead / 1 Alive) and ball possession (1 Home / 2 Away) -- the same kind of label as the
competition ground truth (derived from tracking, not from events). Other games only, so allowed;
cite the paper in TECHNICAL.md and respect its licence.

    pip install -r requirements-harness.txt
    python -m mqf.harness.idsse J03WMX            # downloads (~350 MB/match) + prints a summary

floodlight API verified against floodlight 1.2.0 source:
  IDSSEDataset(match_id).get(match_id, events=False)
    -> (None, xy, possession, ballstatus, teamsheets, pitch)
  xy[segment]["Home"|"Away"|"Ball"].xy : (T, 2N) / (T, 2), metres, pitch centred (xlim = (-L/2, L/2))
  possession[segment].code / ballstatus[segment].code : (T,) floats
  segments: "firstHalf", "secondHalf"
"""
from __future__ import annotations

import sys

import numpy as np

from ..config import EXTERNAL_DIR, official

MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
SEGMENTS = ("firstHalf", "secondHalf")


def from_floodlight(xy, possession, ballstatus, pitch, fps: int = 25, gap_s: float = 60.0) -> dict:
    """Pure conversion (testable without floodlight). Halves are concatenated with an ungraded gap."""
    x0, x1 = pitch.xlim
    y0, y1 = pitch.ylim
    length, width = float(x1 - x0), float(y1 - y0)
    n_home = n_away = 0
    for seg in SEGMENTS:
        if seg in xy:
            n_home = max(n_home, xy[seg]["Home"].xy.shape[1] // 2)
            n_away = max(n_away, xy[seg]["Away"].xy.shape[1] // 2)

    gap = int(gap_s * fps)
    P = n_home + n_away
    players, ball, labels, halves = [], [], [], []
    t = gap                                              # leading ungraded stretch, like the real video
    players.append(np.full((gap, P, 2), np.nan)); ball.append(np.full((gap, 2), np.nan))
    labels.append(np.full(gap, "X", dtype="<U1"))
    for half, seg in enumerate([s for s in SEGMENTS if s in xy], start=1):
        home = xy[seg]["Home"].xy.reshape(len(xy[seg]["Home"].xy), -1, 2)
        away = xy[seg]["Away"].xy.reshape(len(xy[seg]["Away"].xy), -1, 2)
        T = len(home)
        p = np.full((T, P, 2), np.nan)
        p[:, :home.shape[1]] = home
        p[:, n_home:n_home + away.shape[1]] = away
        p[..., 0] -= x0; p[..., 1] -= y0
        b = np.asarray(xy[seg]["Ball"].xy, dtype=float).copy()
        b[:, 0] -= x0; b[:, 1] -= y0
        status = np.asarray(ballstatus[seg].code, dtype=float)
        owner = np.asarray(possession[seg].code, dtype=float)
        lab = np.where(status == 0, "D", np.where(owner == 1, "H", np.where(owner == 2, "A", "X")))
        players.append(p); ball.append(b); labels.append(lab.astype("<U1"))
        halves.append((half, t, t + T - 1))
        t += T
        players.append(np.full((gap, P, 2), np.nan)); ball.append(np.full((gap, 2), np.nan))
        labels.append(np.full(gap, "X", dtype="<U1"))
        t += gap
    return {"players": np.concatenate(players), "ball": np.concatenate(ball),
            "labels": np.concatenate(labels),
            "team_of_player": np.r_[np.zeros(n_home, np.int8), np.ones(n_away, np.int8)],
            "halves": tuple(halves), "pitch": (length, width), "fps": fps}


def _cache_path(match_id: str):
    return EXTERNAL_DIR / "idsse" / f"{match_id}_v1.npz"


def save_cache(m: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, players=m["players"].astype(np.float32), ball=m["ball"].astype(np.float32),
                        labels=m["labels"], team_of_player=m["team_of_player"],
                        halves=np.array(m["halves"]), pitch=np.array(m["pitch"]), fps=m["fps"])


def load_cache(path, name: str) -> dict:
    z = np.load(path)
    return {"players": z["players"].astype(float), "ball": z["ball"].astype(float),
            "labels": z["labels"], "team_of_player": z["team_of_player"],
            "halves": tuple(tuple(int(v) for v in h) for h in z["halves"]),
            "pitch": tuple(float(v) for v in z["pitch"]), "fps": int(z["fps"]), "name": name}


def load_match(match_id: str, use_cache: bool = True) -> dict:
    """Converted match, cached in external/idsse/ (gitignored) -- the DFL XML parse takes minutes.

    The first run also goes through the float32 cache, so a fresh and a cached load give
    bit-identical inputs and the harness stays deterministic.
    """
    cache = _cache_path(match_id)
    if not (use_cache and cache.exists()):
        try:
            from floodlight.io.datasets import IDSSEDataset
        except ImportError as e:
            raise SystemExit("floodlight missing:  pip install -r requirements-harness.txt "
                             "(in its own venv: floodlight 1.2.0 needs pandas < 3)") from e
        ds = IDSSEDataset(match_id=match_id)             # downloads into floodlight's .data/ once
        _, xy, possession, ballstatus, _, pitch = ds.get(match_id, events=False)
        save_cache(from_floodlight(xy, possession, ballstatus, pitch), cache)
    return load_cache(cache, f"IDSSE-{match_id}")


if __name__ == "__main__":
    for mid in sys.argv[1:] or MATCHES[:1]:
        m = load_match(mid)
        graded = np.zeros(len(m["labels"]), dtype=bool)
        for _, first, last in m["halves"]:
            graded[first:last + 1] = True
        lab = m["labels"][graded]          # X here = alive frame with no DFL owner (splits a run)
        _, pm = official()
        codes = pm.label_codes(m["labels"])
        stats = pm.possession_stats(codes, pm.valid_runs(codes))
        print(f"{m['name']}: {graded.sum()} graded frames, halves {m['halves']}, pitch {m['pitch']}, "
              f"H {np.mean(lab == 'H'):.2f} A {np.mean(lab == 'A'):.2f} D {np.mean(lab == 'D'):.2f} "
              f"X {np.mean(lab == 'X'):.3f}, ball seen {np.mean(~np.isnan(m['ball'][graded, 0])):.3f}, "
              f"possessions {stats['n_spells']} (mean {stats['mean_duration_s']:.1f} s), "
              f"players {m['players'].shape[1]}")
