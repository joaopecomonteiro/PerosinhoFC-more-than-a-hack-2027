"""IDSSE (Bassek et al., Scientific Data 2025) -> clean match in our harness format.

7 German matches 2022/23 (2 Bundesliga + 5 2. Bundesliga), official DFL tracking at 25 fps with PER-FRAME ball status
(0 Dead / 1 Alive) and ball possession (1 Home / 2 Away) -- the same kind of label as the
competition ground truth (derived from tracking, not from events). Other games only, so allowed;
cite the paper in TECHNICAL.md and respect its licence.

    pip install -r requirements-harness.txt
    python -m mqf.harness.idsse J03WMX            # parse + cache one match, print a summary
    python -m mqf.harness.idsse                   # all 7

Reads the DFL XML from external_data/idsse/ (MQF_EXTERNAL_RAW) when it is there, named as on
figshare (DFL_04_03_positions_raw_observed_<COM>_DFL-MAT-<id>.xml + matchinformation); otherwise
floodlight downloads it (~350 MB per match).

floodlight API verified against floodlight 1.2.0 source:
  IDSSEDataset(match_id).get(match_id, events=False)
    -> (None, xy, possession, ballstatus, teamsheets, pitch)
  dfl.read_position_data_xml(filepath_positions, filepath_mat_info)   (what .get calls)
    -> (xy, possession, ballstatus, teamsheets, pitch)
  xy[segment]["Home"|"Away"|"Ball"].xy : (T, 2N) / (T, 2), metres, pitch centred (xlim = (-L/2, L/2))
  possession[segment].code / ballstatus[segment].code : (T,) floats
  segments: "firstHalf", "secondHalf"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..config import EXTERNAL_DIR, EXTERNAL_RAW_DIR, official

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


def raw_files(match_id: str, raw_dir: Path | None = None) -> tuple[Path, Path] | None:
    """(positions XML, matchinformation XML) of a local IDSSE copy, named as on figshare, or None."""
    d = Path(raw_dir) if raw_dir else EXTERNAL_RAW_DIR / "idsse"
    pos = sorted(d.glob(f"DFL_04_03_positions_raw_observed_*_DFL-MAT-{match_id}.xml"))
    info = sorted(d.glob(f"DFL_02_01_matchinformation_*_DFL-MAT-{match_id}.xml"))
    return (pos[0], info[0]) if pos and info else None


def team_names(match_id: str, raw_dir: Path | None = None) -> tuple[str, str]:
    """(home, guest) from the local matchinformation XML; ("Home", "Away") without it."""
    import re
    d = Path(raw_dir) if raw_dir else EXTERNAL_RAW_DIR / "idsse"
    info = sorted(d.glob(f"DFL_02_01_matchinformation_*_DFL-MAT-{match_id}.xml"))
    if info:
        text = info[0].read_text(encoding="utf-8", errors="replace")
        h, g = re.search(r'HomeTeamName="([^"]*)"', text), re.search(r'GuestTeamName="([^"]*)"', text)
        if h and g:
            return h.group(1), g.group(1)
    return "Home", "Away"


def read_floodlight(match_id: str, raw_dir: Path | None = None):
    """(xy, possession, ballstatus, pitch): from external_data/idsse/ when the files are there,
    otherwise floodlight's own download (figshare, ~350 MB per match)."""
    try:
        from floodlight.io import dfl
    except ImportError as e:
        raise SystemExit("floodlight missing:  pip install -r requirements-harness.txt "
                         "(in its own venv: floodlight 1.2.0 needs pandas < 3)") from e
    files = raw_files(match_id, raw_dir)
    if files:
        print(f"{match_id}: parsing {files[0].name} (a few minutes)")
        xy, possession, ballstatus, _, pitch = dfl.read_position_data_xml(str(files[0]), str(files[1]))
        return xy, possession, ballstatus, pitch
    from floodlight.io.datasets import IDSSEDataset
    print(f"{match_id}: no local copy in {raw_dir or EXTERNAL_RAW_DIR / 'idsse'} -- downloading via floodlight")
    ds = IDSSEDataset(match_id=match_id)
    _, xy, possession, ballstatus, _, pitch = ds.get(match_id, events=False)
    return xy, possession, ballstatus, pitch


def load_match(match_id: str, use_cache: bool = True) -> dict:
    """Converted match, cached in external/idsse/ (gitignored) -- the DFL XML parse takes minutes.

    The first run also goes through the float32 cache, so a fresh and a cached load give
    bit-identical inputs and the harness stays deterministic.
    """
    cache = _cache_path(match_id)
    if not (use_cache and cache.exists()):
        xy, possession, ballstatus, pitch = read_floodlight(match_id)
        save_cache(from_floodlight(xy, possession, ballstatus, pitch), cache)
    return load_cache(cache, f"IDSSE-{match_id}")


if __name__ == "__main__":
    for mid in sys.argv[1:] or MATCHES:
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
