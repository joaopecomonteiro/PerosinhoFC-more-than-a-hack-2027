"""What the viewer shows, as one object (`ViewerGame`), and the adapters that build it.

The contract is deliberately small so that another source of games (PFF, IDSSE, ...) is one more
adapter function and nothing in the server or the page changes:

    ViewerGame
      timeline   n_frames at fps -- the *match* timeline every lane is indexed by (for BAR-ATH the
                 tactical video, i.e. a submission row)
      halves     [(half, first, last)] inclusive, the graded blocks
      tracking   players (n, slots, 2) metres + team_of_slot, ball (n, 2) metres; NaN = not tracked
                 ("Original", as delivered) and optional extra sets on the same timeline -- "Fixed" =
                 our cleaned tracklets + filled ball (fixed.py), switchable on the page
      lanes      label tracks on the match timeline: 0 H, 1 A, 2 D, 3 no label (X / outside GT)
      videos     each with its own frame count, an optional homography (normalised pitch -> pixels)
                 and, when its timeline is not the match timeline, the two frame maps
                 (video -> match, match -> video) plus a per-frame state (live, replay, ...)
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .. import atd
from ..config import CACHE_DIR, DATA_DIR, RESULTS_DIR, SYNTHETIC_DIR, GameSpec

LABELS = ("H", "A", "D", "X")
NO_LABEL = 3
VIEWER_CACHE = CACHE_DIR / "viewer"
Log = Callable[[str], None]


@dataclass
class Lane:
    name: str
    kind: str                  # "truth" | "model"
    codes: np.ndarray          # (n_frames,) uint8: 0 H, 1 A, 2 D, 3 no label
    source: str = ""           # shown under the lane name


@dataclass
class Video:
    key: str                   # "tactical" | "broadcast" | ...
    label: str
    path: Path | None
    fps: float
    n_frames: int
    homography: np.ndarray | None = None   # (n_frames, 9) float32, normalised pitch -> pixels, NaN = none
    to_match: np.ndarray | None = None     # (n_frames,) int32 match frame per video frame, -1 = none; None = identity
    from_match: np.ndarray | None = None   # (n_match,) int32 video frame per match frame, -1 = none; None = identity
    state: np.ndarray | None = None        # (n_frames,) uint8 sync.STATE_*; None = every frame live
    note: str = ""                         # why it is missing or unlinked, shown on the stage
    summary: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.path is not None and Path(self.path).is_file()


@dataclass
class TrackingSet:
    """Another version of the tracking on the same timeline (its own slots)."""
    key: str                   # blob suffix: players_<key>, ball_<key>, ballsrc_<key>
    label: str
    players: np.ndarray        # (n, slots, 2) metres
    team_of_slot: np.ndarray   # (slots,)
    ball: np.ndarray           # (n, 2) metres
    ball_source: np.ndarray | None = None   # (n,) uint8: 0 tracked, 1 interpolated, 2 model, 3 empty
    note: str = ""
    flags: np.ndarray | None = None         # (n, slots) uint8 tracks.FLAG_*: what the cleaner did to each point


@dataclass
class ViewerGame:
    key: str
    title: str
    home: str
    away: str
    fps: float
    n_frames: int
    pitch: tuple[float, float]
    halves: tuple
    players: np.ndarray        # (n, slots, 2) metres
    team_of_slot: np.ndarray   # (slots,) 0 home, 1 away
    ball: np.ndarray           # (n, 2) metres
    lanes: list[Lane]
    videos: list[Video]
    notes: list[str] = field(default_factory=list)
    extra_tracking: list[TrackingSet] = field(default_factory=list)


# ---------------------------------------------------------------------------------- readers

def read_labels(path: Path, n_frames: int) -> np.ndarray:
    """A frame,label CSV (submission or ground truth) -> uint8 codes. Strict on shape, like the
    official validator, so a lane never silently shifts by a row."""
    df = pd.read_csv(path, dtype={"label": "string"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"frame", "label"} <= set(df.columns):
        raise ValueError(f"{Path(path).name}: expected columns frame,label")
    if len(df) != n_frames:
        raise ValueError(f"{Path(path).name}: {len(df)} rows, the match has {n_frames}")
    if not np.array_equal(df["frame"].to_numpy(), np.arange(n_frames)):
        raise ValueError(f"{Path(path).name}: 'frame' is not 0..{n_frames - 1} in order")
    lab = df["label"].fillna("X").str.strip().str.upper().to_numpy(dtype=str)
    codes = np.full(n_frames, 255, dtype=np.uint8)
    for k, name in enumerate(LABELS):
        codes[lab == name] = k
    if (codes == 255).any():
        bad = sorted(set(lab[codes == 255]))[:5]
        raise ValueError(f"{Path(path).name}: labels outside H/A/D/X: {bad}")
    return codes


def read_homography(path: Path, n_frames: int) -> np.ndarray:
    """Official homography CSV (frame, h00..h22, reprojection_error, valid) -> (n, 9) float32,
    NaN where there is no valid matrix. Cached next to the tracking cache."""
    path = Path(path)
    st = path.stat()
    key = hashlib.md5(f"{path}|{st.st_size}|{st.st_mtime_ns}|{n_frames}".encode()).hexdigest()[:10]
    cache = VIEWER_CACHE / f"hom_{path.stem}_{key}.npy"
    if cache.exists():
        return np.load(cache)
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = [f"h{r}{c}" for r in range(3) for c in range(3)]
    out = np.full((n_frames, 9), np.nan, dtype=np.float32)
    frame = df["frame"].to_numpy(dtype=np.int64)
    ok = (frame >= 0) & (frame < n_frames)
    if "valid" in df.columns:
        ok &= df["valid"].fillna(0).to_numpy() > 0
    out[frame[ok]] = df.loc[ok, cols].to_numpy(dtype=np.float32)
    VIEWER_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def team_names(meta_path: Path) -> tuple[str, str]:
    """Team names from the FIFA/EPTS metadata (first <Team> is home)."""
    try:
        root = ET.parse(meta_path).getroot()
        names = []
        for t in root.iter("Team"):
            n = t.find("Name")
            names.append((n.text or "").strip() if n is not None else t.get("name", ""))
        if len(names) >= 2 and all(names[:2]):
            return names[0], names[1]
    except (ET.ParseError, OSError):
        pass
    return "Home", "Away"


def lanes_from_csvs(paths, n_frames: int, kind: str, log: Log) -> list[Lane]:
    out = []
    for p in paths:
        p = Path(p)
        try:
            out.append(Lane(p.stem, kind, read_labels(p, n_frames), str(p.name)))
        except (ValueError, OSError) as e:
            log(f"  skipped {p.name}: {e}")
    return out


def pitch_orientation(hom: np.ndarray | None) -> dict:
    """Which way the camera shows the pitch, so the 2D view is drawn the way the video looks."""
    if hom is None:
        return {"flip_x": False, "flip_y": False}
    rows = np.flatnonzero(np.isfinite(hom).all(axis=1))
    if not len(rows):
        return {"flip_x": False, "flip_y": False}
    H = hom[rows[np.linspace(0, len(rows) - 1, min(len(rows), 200)).astype(int)]].reshape(-1, 3, 3)
    pts = np.array([[0.5, 0, 1], [0.5, 1, 1], [0, 0.5, 1], [1, 0.5, 1]], dtype=np.float64)
    pr = np.einsum("nij,kj->nki", H.astype(np.float64), pts)
    uv = pr[..., :2] / pr[..., 2:3]
    flip_y = float(np.median(uv[:, 0, 1] - uv[:, 1, 1])) > 0     # y=0 lower in the image than y=1
    flip_x = float(np.median(uv[:, 2, 0] - uv[:, 3, 0])) > 0     # x=0 right of x=1
    return {"flip_x": bool(flip_x), "flip_y": bool(flip_y)}


def _video_with_sync(key, label, video_path, spec_b: GameSpec, hom_path, trk_t, n_match, fps,
                     resync, log) -> Video:
    """A video on its own timeline, linked to the match by sync.align (cached)."""
    from . import sync
    if not (Path(spec_b.atd_raw).is_file() and Path(spec_b.atd_meta).is_file()):
        return Video(key, label, video_path, fps, spec_b.n_frames,
                     note=f"No {label.lower()} tracking found, so it cannot be linked to the match.")
    trk_b = atd.load(spec_b)
    hom_b = read_homography(hom_path, spec_b.n_frames) if hom_path and Path(hom_path).is_file() else None
    if hom_b is None:
        return Video(key, label, video_path, fps, spec_b.n_frames,
                     note=f"No {label.lower()} homography found, so it cannot be linked to the match.")
    res = sync.align_cached(trk_t, trk_b, hom_b, spec_b, n_match, fps, force=resync, log=log)
    return Video(key, label, video_path, fps, spec_b.n_frames, homography=hom_b,
                 to_match=res.to_match, from_match=res.from_match, state=res.state,
                 summary=res.summary())


def find_video(expected: Path, log: Log) -> tuple[Path, str]:
    """The video games.py names, or -- when a download renamed it -- the one file next to it that
    starts with the same stem (BAR-ATH_tactical-002.mp4). Returns (path, note for the stage)."""
    expected = Path(expected)
    if expected.is_file():
        return expected, ""
    found = sorted(expected.parent.glob(f"{expected.stem}*{expected.suffix}")) if expected.parent.is_dir() else []
    if len(found) == 1:
        log(f"  using {found[0].name} (games.py expects {expected.name}; rename it to silence this)")
        return found[0], ""
    if len(found) > 1:
        return expected, (f"Several candidates for {expected.name} in {expected.parent}: "
                          + ", ".join(f.name for f in found) + ". Rename the right one to " + expected.name + ".")
    return expected, f"Video not found at {expected}"


# ---------------------------------------------------------------------------------- adapters

def _fixed(trk, spec, fixed, log) -> list[TrackingSet]:
    """The "Fixed" set, or nothing (no --no-fixed, never fatal)."""
    if not fixed:
        return []
    from .fixed import fixed_tracking
    try:
        windows = None
        if getattr(spec, "smart_tagging", None) and Path(spec.smart_tagging).is_file():
            from ..smart_tagging import kickoff_windows
            windows = kickoff_windows(spec.smart_tagging, spec.fps, spec.n_frames)
        return [fixed_tracking(trk, spec.fps, spec.pitch, spec.halves, log=log, kickoff_windows=windows, **fixed)]
    except Exception as e:                       # the viewer must open even if this fails
        log(f"fixed tracking skipped: {type(e).__name__}: {e}")
        return []


def competition(preds=None, gt: str | Path = "public", broadcast: bool = True, resync: bool = False,
                log: Log = print, fixed: dict | None = None) -> ViewerGame:
    """FC Barcelona v Athletic Club: tactical (the match timeline) + broadcast (auto-synced)."""
    from ..config import competition_game, homography_path, official
    games, _ = official()
    spec = competition_game()
    missing = [p for p in (spec.atd_meta, spec.atd_raw) if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"missing {missing[0]}\nUnpack the match bundle into data/ or set MTAH_DATA_DIR.")
    log("tracking (tactical)")
    trk = atd.load(spec)
    home, away = team_names(spec.atd_meta)
    notes = []

    lanes: list[Lane] = []
    if str(gt) != "none":
        gt_path = spec.gt_public if str(gt) == "public" else Path(gt)
        if gt_path and Path(gt_path).is_file():
            src = "public 5 min" if str(gt) == "public" else Path(gt_path).name
            lanes.append(Lane("Ground truth", "truth", read_labels(gt_path, spec.n_frames), src))
        else:
            notes.append(f"No ground truth at {gt_path}. Run python bootstrap.py.")
    pred_paths = list(preds) if preds else sorted(RESULTS_DIR.glob("*.csv"))
    lanes += lanes_from_csvs(pred_paths, spec.n_frames, "model", log)
    if not any(l.kind == "model" for l in lanes):
        notes.append("No model output in results/. Run python -m mqf.make_submission, or drop a CSV here.")

    tac_paths = games.asset_paths(games.GAME, str(DATA_DIR), source="tactical")
    hom_t_path = homography_path(source="tactical")
    hom_t = read_homography(hom_t_path, spec.n_frames) if Path(hom_t_path).is_file() else None
    tac_video, tac_note = find_video(tac_paths["video"], log)
    videos = [Video("tactical", "Tactical", tac_video, spec.fps, spec.n_frames, homography=hom_t, note=tac_note)]

    if broadcast:
        spec_b = competition_game(source="broadcast")
        b_paths = games.asset_paths(games.GAME, str(DATA_DIR), source="broadcast")
        b_video, b_note = find_video(b_paths["video"], log)
        log("broadcast sync")
        bv = _video_with_sync("broadcast", "Broadcast", b_video, spec_b,
                              homography_path(source="broadcast"), trk, spec.n_frames, spec.fps, resync, log)
        if not bv.available and not bv.note:
            bv.note = b_note
        videos.append(bv)

    return ViewerGame(key=games.GAME, title=f"{home} v {away}", home=home, away=away, fps=spec.fps,
                      n_frames=spec.n_frames, pitch=spec.pitch, halves=spec.halves,
                      players=trk.players, team_of_slot=trk.team_of_slot, ball=trk.ball,
                      lanes=lanes, videos=videos, notes=notes, extra_tracking=_fixed(trk, spec, fixed, log))


def synthetic_game(preds=None, gt=None, resync: bool = False, log: Log = print,
                   fixed: dict | None = None) -> ViewerGame:
    """The synthetic match (full ground truth). Videos appear once `python -m mqf.viewer.demo`
    has rendered them; until then the pitch view carries everything."""
    from .. import decode, make_submission, synthetic
    from . import demo
    log("synthetic match")
    spec = synthetic.generate(SYNTHETIC_DIR)
    trk = atd.load(spec)
    home, away = team_names(spec.atd_meta)

    lanes = []
    gt_path = Path(gt) if gt and str(gt) != "public" else spec.gt_public
    if str(gt) != "none":
        lanes.append(Lane("Ground truth", "truth", read_labels(gt_path, spec.n_frames), "synthetic, full match"))
    res_dir = SYNTHETIC_DIR / "results"
    pred_paths = list(preds) if preds else sorted(res_dir.glob("*.csv"))
    if not pred_paths:
        log("  no synthetic results yet: running the pipeline once")
        make_submission.run(spec, decode.Params(), res_dir / "synth.csv")
        pred_paths = [res_dir / "synth.csv"]
    lanes += lanes_from_csvs(pred_paths, spec.n_frames, "model", log)

    files = demo.paths(SYNTHETIC_DIR)
    notes = []
    hom_t = read_homography(files["tactical_hom"], spec.n_frames) if files["tactical_hom"].is_file() else None
    videos = [Video("tactical", "Tactical", files["tactical_video"], spec.fps, spec.n_frames, homography=hom_t,
                    note="" if files["tactical_video"].is_file() else
                    "No synthetic video yet. Run python -m mqf.viewer.demo to render one.")]
    spec_b = demo.broadcast_spec(SYNTHETIC_DIR, spec)
    if spec_b is not None:
        log("broadcast sync")
        videos.append(_video_with_sync("broadcast", "Broadcast", files["broadcast_video"], spec_b,
                                       files["broadcast_hom"], trk, spec.n_frames, spec.fps, resync, log))
    else:
        videos.append(Video("broadcast", "Broadcast", None, spec.fps, 0,
                            note="No synthetic broadcast yet. Run python -m mqf.viewer.demo."))
    return ViewerGame(key="SYN", title=f"{home} v {away}", home=home, away=away, fps=spec.fps,
                      n_frames=spec.n_frames, pitch=spec.pitch, halves=spec.halves,
                      players=trk.players, team_of_slot=trk.team_of_slot, ball=trk.ball,
                      lanes=lanes, videos=videos, notes=notes, extra_tracking=_fixed(trk, spec, fixed, log))


# ---------------------------------------------------------------------------------- to the page

def _cm(a: np.ndarray) -> bytes:
    v = np.round(np.asarray(a, dtype=np.float64) * 100.0)
    v = np.where(np.isfinite(v), np.clip(v, -32767, 32767), -32768)
    return v.astype("<i2").tobytes()


def bundle(game: ViewerGame) -> tuple[dict, dict[str, bytes]]:
    """JSON metadata + named binary blobs, exactly what the page fetches."""
    blobs: dict[str, bytes] = {"players": _cm(game.players), "ball": _cm(game.ball)}
    lanes = []
    for i, lane in enumerate(game.lanes):
        blobs[f"lane{i}"] = lane.codes.astype(np.uint8).tobytes()
        lanes.append({"name": lane.name, "kind": lane.kind, "source": lane.source, "blob": f"lane{i}"})
    videos = []
    for v in game.videos:
        entry = {"key": v.key, "label": v.label, "available": v.available, "fps": v.fps,
                 "n_frames": int(v.n_frames), "note": v.note, "summary": v.summary,
                 "homography": None, "to_match": None, "from_match": None, "state": None}
        for name, arr, dtype in (("homography", v.homography, "<f4"), ("to_match", v.to_match, "<i4"),
                                 ("from_match", v.from_match, "<i4"), ("state", v.state, "u1")):
            if arr is not None:
                blobs[f"{name}_{v.key}"] = np.ascontiguousarray(arr).astype(dtype).tobytes()
                entry[name] = f"{name}_{v.key}"
        videos.append(entry)
    tracking = [{"key": "original", "label": "Original", "players": "players", "ball": "ball", "ball_source": None,
                 "flags": None,
                 "n_slots": int(game.players.shape[1]), "team_of_slot": game.team_of_slot.astype(int).tolist(),
                 "note": "As delivered"}]
    for t in game.extra_tracking:
        entry = {"key": t.key, "label": t.label, "players": f"players_{t.key}", "ball": f"ball_{t.key}",
                 "ball_source": None, "flags": None, "n_slots": int(t.players.shape[1]),
                 "team_of_slot": np.asarray(t.team_of_slot).astype(int).tolist(), "note": t.note}
        blobs[entry["players"]], blobs[entry["ball"]] = _cm(t.players), _cm(t.ball)
        if t.flags is not None and np.shape(t.flags) == t.players.shape[:2]:
            entry["flags"] = f"flags_{t.key}"
            blobs[entry["flags"]] = np.ascontiguousarray(t.flags, dtype=np.uint8).tobytes()
        if t.ball_source is not None:
            entry["ball_source"] = f"ballsrc_{t.key}"
            blobs[entry["ball_source"]] = np.asarray(t.ball_source, dtype=np.uint8).tobytes()
        tracking.append(entry)
    hom_t = next((v.homography for v in game.videos if v.key == "tactical"), None)
    meta = {
        "key": game.key, "title": game.title, "home": game.home, "away": game.away,
        "fps": game.fps, "n_frames": int(game.n_frames), "pitch": list(game.pitch),
        "halves": [list(map(int, h)) for h in game.halves],
        "n_slots": int(game.players.shape[1]), "team_of_slot": game.team_of_slot.astype(int).tolist(),
        "orientation": pitch_orientation(hom_t), "lanes": lanes, "videos": videos, "notes": game.notes,
        "tracking": tracking,
    }
    return meta, blobs
