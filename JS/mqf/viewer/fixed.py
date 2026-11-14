"""The viewer's "Fixed" tracking: what our pipeline makes of the delivered tracking.

  players  mqf.tracks.clean -- team-level tracklet cleaning (cuts at jumps, duplicates hidden while
           together, at most 11 per team, team repairs, keeper rules); its output lanes become the
           slots. Each point carries a flag (tracks.FLAG_*) so the page can mark what was repaired:
           team changed, keeper takeover, keeper drawn at his saved place.
  ball     the missing ball filled with the frozen setting (mqf.ball.impute.FROZEN): linear <= 1 s,
           Hermite <= 2 s, the ball model anchored to the gap edges beyond. Every gap inside the
           halves is filled -- there is no dead-ball detector yet, so a filled ball during a
           stoppage is shown too (the page marks every filled frame as interpolated / model).
           Without torch or the model file only the gaps <= 2 s are filled.

Computed on this machine and cached in cache/viewer (gitignored) by a hash of the input tracking,
the model file, the settings AND the source of the cleaner / ball code, so an edit to mqf/tracks.py
always shows up (before, a logic change that left CleanConfig alone reused the old cache).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .. import tracks
from ..atd import Tracking
from ..ball import impute, interp
from ..config import MODELS_DIR
from .game import VIEWER_CACHE, Log, TrackingSet

VERSION = 3


def _key(trk: Tracking, fps, pitch, halves, model_path: Path | None, alive, kickoff_windows=None,
         clean_cfg: tracks.CleanConfig | None = None) -> str:
    h = hashlib.md5()
    h.update(json.dumps([list(map(int, w)) for w in kickoff_windows or []]).encode())
    for a in (trk.players, trk.ball, trk.team_of_slot):
        h.update(np.ascontiguousarray(a).tobytes())
    if alive is not None:
        h.update(np.ascontiguousarray(alive, dtype=bool).tobytes())
    for mod in (tracks, impute, interp):                  # the code itself: a logic change recomputes
        h.update(Path(mod.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    st = model_path.stat() if model_path is not None and model_path.is_file() else None
    h.update(json.dumps([VERSION, float(fps), list(pitch), [list(map(int, x)) for x in halves], impute.FROZEN,
                         repr(clean_cfg or tracks.CleanConfig()), str(model_path), st.st_size if st else None,
                         st.st_mtime_ns if st else None]).encode())
    return h.hexdigest()[:12]


def _model(model_path: Path | None, device: str, log: Log):
    """(model, cfg, device, note) -- None when torch or the file is missing."""
    if model_path is None or not model_path.is_file():
        return None, None, "cpu", f"no ball model at {model_path}: only gaps <= 2 s are filled"
    try:
        import torch
    except ImportError:
        return None, None, "cpu", "torch is not installed here: only gaps <= 2 s are filled"
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model, cfg = impute.load_model(model_path, dev)
    return model, cfg, dev, ""


def fixed_tracking(trk: Tracking, fps, pitch, halves, model_path: Path | None = None, device: str = "auto",
                   alive: np.ndarray | None = None, force: bool = False, log: Log = print,
                   kickoff_windows=None, clean_cfg: tracks.CleanConfig | None = None) -> TrackingSet:
    """alive: optional frame mask -- fill the ball only there (default: every gap inside the halves).
    kickoff_windows: tagged kickoffs for the cleaner's kickoff check (smart_tagging.kickoff_windows).
    clean_cfg: cleaner settings (default: the current CleanConfig) -- e.g. a rule switched off to compare."""
    F = impute.FROZEN
    model_path = Path(model_path) if model_path else MODELS_DIR / F["model"]
    clean_cfg = clean_cfg or tracks.CleanConfig()
    cache = VIEWER_CACHE / f"fixed_{_key(trk, fps, pitch, halves, model_path, alive, kickoff_windows, clean_cfg)}.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        log("fixed tracking (cached)")
        log("fixed tracking: " + str(z["note"]).split("; ")[0])
        return TrackingSet("fixed", "Fixed", z["players"], z["team_of_slot"], z["ball"], z["ball_source"],
                           str(z["note"]), flags=z["flags"])
    log("fixed tracking: cleaning tracklets")
    if clean_cfg != tracks.CleanConfig():
        d = tracks.CleanConfig()
        log("fixed tracking: cleaner settings " + ", ".join(f"{k}={getattr(clean_cfg, k)}" for k in d.__dataclass_fields__
                                                          if getattr(clean_cfg, k) != getattr(d, k)))
    ctrk, rep, flags = tracks.clean_with_flags(trk, fps, pitch, clean_cfg, kickoff_windows=kickoff_windows,
                                                halves=halves)
    model, cfg, dev, note = _model(model_path, device, log)
    n = len(trk.ball)
    log(f"fixed tracking: filling the ball ({'model ' + model_path.name + ' on ' + dev if model else 'interpolation only'})")
    kw = dict(method="hybrid", tau_s=F["tau_s"], tau_linear_s=F["tau_linear_s"], interp_method=F["interp_method"],
              anchor_s=F["anchor_s"], frames=alive)
    if model is not None:
        bf = impute.fill(ctrk.players, ctrk.team_of_slot, trk.ball, fps, pitch, halves, model=model, cfg=cfg,
                         device=dev, **kw)
    else:
        bf = impute.fill(None, None, trk.ball, fps, pitch, halves, model_ball=np.full((n, 2), np.nan), **kw)
    in_half = np.zeros(n, dtype=bool)
    for _, a, b in halves:
        in_half[a:b + 1] = True
    miss = in_half & np.isnan(trk.ball[:, 0])
    share = {k: float(np.mean(bf.source[miss] == c)) if miss.any() else 0.0
             for k, c in (("interpolated", impute.INTERP), ("model", impute.MODEL), ("empty", impute.EMPTY))}
    ppf = rep.get("players_per_frame_mean", {})
    changes = diff_summary(rep)
    summary = (f"players per frame {ppf.get('before')} -> {ppf.get('after')}; missing ball in the halves: "
               f"{share['interpolated']:.0%} interpolated, {share['model']:.0%} model, {share['empty']:.0%} still empty")
    log("fixed tracking: " + changes)
    log("fixed tracking: " + summary)
    note = "; ".join(x for x in (changes, summary, note) if x)
    ts = TrackingSet("fixed", "Fixed", ctrk.players.astype(np.float32), ctrk.team_of_slot.astype(np.int8),
                     bf.ball.astype(np.float32), bf.source.astype(np.uint8), note, flags=flags.astype(np.uint8))
    VIEWER_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, players=ts.players, team_of_slot=ts.team_of_slot, ball=ts.ball,
                        ball_source=ts.ball_source, note=np.array(note), flags=ts.flags)
    return ts


def diff_summary(rep: dict) -> str:
    """What Fixed changed against Original, in counts only (safe to paste)."""
    ti = rep.get("team_inherit") or {}
    kf = rep.get("keeper_fill_points") or {}
    hidden = (rep.get("frozen_copies_hidden_points", 0) + rep.get("duplicate_points_hidden", 0)
              + rep.get("cap_points_hidden", 0) + rep.get("fast_points_hidden", 0))
    return (f"vs Original: {rep.get('tracklets_team_changed', 0)} tracks changed team "
            f"({rep.get('tracklets_keeper_takeover', 0)} keeper takeovers: held {ti.get('keeper_takeover_held', 0)}, "
            f"duplicate {ti.get('keeper_takeover_duplicate', 0)}, next to him {ti.get('keeper_takeover_near', 0)}, "
            f"saved place {ti.get('keeper_takeover_saved', 0)}; "
            f"refused: apart {ti.get('keeper_takeover_apart', 0)}, counts {ti.get('keeper_takeover_counts_disagree', 0)}), "
            f"{hidden:,} points hidden, keeper at his saved place on {kf.get('home', 0):,} / {kf.get('away', 0):,} "
            f"frames (home / away), {100 * rep.get('points_kept_share', 1):.1f}% of points kept")
