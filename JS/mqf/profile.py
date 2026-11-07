"""Aggregate-only profile of the match recordings: tactical (the reference) and broadcast.

    python -m mqf.profile                       # both recordings
    python -m mqf.profile --source tactical     # one of them

Prints ONE block of aggregates -- shares, counts, quantiles -- and writes the same numbers to
reports/profile/profile.json (gitignored). It never prints frame numbers, rows, positions or
images, so the printed block is what the team may paste to an AI assistant (CLAUDE.md, rule 1).

What it answers (docs: "Next steps", section 1):
  ball gaps (how many, how long), where the ball goes missing (players moving vs still -- the
  ceiling for any imputation model), tracking noise (jitter, speed field vs positions, ball
  jumps), tracklets (slots, lengths, objects per frame), stillness (median player speed), off-pitch
  shares, Smart Tagging coverage, homography validity.

Use the numbers to set RANGES for noise-model robustness sweeps and to design features -- not to
point-calibrate anything to this match, and never to write rules tied to specific frames.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import atd, smart_tagging
from .config import REPORTS_DIR, GameSpec, competition_game, homography_path

QS = (5, 25, 50, 75, 95)
GAP_BUCKETS_S = (0.0, 0.5, 2.0, 10.0, np.inf)      # interpolate / short / medium / long
STILL_MS = (0.5, 1.0, 1.5)                          # median player speed thresholds, m/s
ACTIVE_MS, SLOW_MS = 2.0, 1.0                       # "players moving" / "players still"
JUMP_MS = 40.0                                      # ball speed no football kick reaches over 1 frame
OFF_PITCH_M = 0.5


def runs(mask: np.ndarray) -> np.ndarray:
    """Lengths of the True runs in a 1-D boolean array (vectorised)."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def run_length_at(mask: np.ndarray) -> np.ndarray:
    """For every frame inside a True run, the total length of that run (0 outside)."""
    lengths = runs(mask)
    out = np.zeros(len(mask), dtype=np.int64)
    out[np.asarray(mask, dtype=bool)] = np.repeat(lengths, lengths)
    return out


def q(x, qs=QS, digits=2) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"p{p}": None for p in qs}
    return {f"p{p}": round(float(v), digits) for p, v in zip(qs, np.percentile(x, qs))}


def share(mask, within=None, digits=4) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if within is not None:
        mask = mask[np.asarray(within, dtype=bool)]
    return round(float(mask.mean()), digits) if mask.size else None


def _nan(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*a, **k)


def graded_mask(spec: GameSpec) -> np.ndarray:
    """Tactical: the published half blocks. Broadcast: its own timeline has no graded blocks, so
    every frame with any tracked object counts."""
    m = np.zeros(spec.n_frames, dtype=bool)
    for _, first, last in spec.halves:
        m[first:last + 1] = True
    return m


def profile_tracking(trk: atd.Tracking, spec: GameSpec, graded: np.ndarray) -> dict:
    fps, (L, W) = spec.fps, spec.pitch
    seen = trk.ball_seen
    present = ~np.isnan(trk.players[..., 0])                      # (n, slots)
    if not graded.any():                                          # broadcast: frames with tracking
        graded = present.any(axis=1) | seen
    n_obj = present.sum(axis=1)

    # ---- ball gaps (inside graded stretches only, so halftime never joins two gaps)
    missing = ~seen & graded
    gaps = runs(missing) / fps
    gap_len_at = run_length_at(missing) / fps
    bucket_share = {}
    for lo, hi in zip(GAP_BUCKETS_S[:-1], GAP_BUCKETS_S[1:]):
        bucket_share[f"{lo:g}-{hi:g}s"] = share((gap_len_at >= lo) & (gap_len_at < hi), within=missing)

    # ---- ball kinematics on consecutive seen frames
    both = seen[1:] & seen[:-1]
    step = np.linalg.norm(np.diff(trk.ball, axis=0), axis=1)[both]
    ball_speed = step * fps
    tri = seen[2:] & seen[1:-1] & seen[:-2]
    d2_ball = np.linalg.norm(trk.ball[2:] - 2 * trk.ball[1:-1] + trk.ball[:-2], axis=1)[tri]
    bx, by = trk.ball[:, 0], trk.ball[:, 1]
    ball_off = seen & ((bx < -OFF_PITCH_M) | (bx > L + OFF_PITCH_M) | (by < -OFF_PITCH_M) | (by > W + OFF_PITCH_M))

    # ---- players: speed field vs positions, jitter, stillness
    sp = trk.player_speed
    pos_speed = np.full(sp.shape, np.nan, dtype=np.float32)
    pos_speed[1:] = np.linalg.norm(np.diff(trk.players, axis=0), axis=2) * fps
    has_field = ~np.isnan(sp) & present
    speed_used = np.where(np.isnan(sp), pos_speed, sp)
    med_speed = _nan(np.nanmedian, speed_used, axis=1)
    d2_pl = np.linalg.norm(trk.players[2:] - 2 * trk.players[1:-1] + trk.players[:-2], axis=2)
    px, py = trk.players[..., 0], trk.players[..., 1]
    pl_off = present & ((px < -OFF_PITCH_M) | (px > L + OFF_PITCH_M) | (py < -OFF_PITCH_M) | (py > W + OFF_PITCH_M))

    # ---- tracklets: runs of presence per slot
    tracklets = np.concatenate([runs(present[:, k]) for k in range(present.shape[1])]) / fps
    minutes = graded.sum() / fps / 60

    # ---- where the ball goes missing: players moving vs still (imputation ceiling)
    active, slow = med_speed >= ACTIVE_MS, med_speed < SLOW_MS
    live_gap = missing & active

    home = trk.team_of_slot == 0
    return {
        "frames": {"records": int(spec.n_frames), "graded_or_tracked": int(graded.sum())},
        "ball": {
            "seen_share_all": share(seen), "seen_share_graded": share(seen, within=graded),
            "gaps": {"n": int(len(gaps)), "per_min": round(len(gaps) / minutes, 2) if minutes else None,
                     "length_s": q(gaps, (25, 50, 75, 90, 99)),
                     "missing_frames_by_gap_length": bucket_share},
            "speed_ms_consecutive_seen": q(ball_speed, (50, 90, 99)),
            "jump_share_over_40ms": share(ball_speed > JUMP_MS),
            "second_difference_m": {"median": round(float(np.median(d2_ball)), 4) if len(d2_ball) else None,
                                    "p90": q(d2_ball, (90,), 4)["p90"]},
            "off_pitch_share_of_seen": share(ball_off, within=seen),
        },
        "players": {
            "slots": int(present.shape[1]), "slots_ever_used": int(present.any(axis=0).sum()),
            "objects_per_frame_graded": {"mean": round(float(n_obj[graded].mean()), 2),
                                         **q(n_obj[graded], (5, 50, 95), 1)},
            "objects_home_away_mean": [round(float(present[graded][:, home].sum(1).mean()), 2),
                                       round(float(present[graded][:, ~home].sum(1).mean()), 2)],
            "share_frames_over_22": share(n_obj > 22, within=graded),
            "share_frames_under_18": share(n_obj < 18, within=graded),
            "tracklets": {"n": int(len(tracklets)), "per_min": round(len(tracklets) / minutes, 1) if minutes else None,
                          "length_s": q(tracklets, (25, 50, 75, 95))},
            "speed_field_present_share": share(has_field, within=present),
            "speed_field_ms": q(sp[has_field & graded[:, None]], (50, 90, 99)),
            "speed_field_vs_positions_median_abs_ms": round(float(_nan(np.nanmedian, np.abs(sp - pos_speed)[has_field])), 3)
            if has_field.any() else None,
            "second_difference_m_median": round(float(_nan(np.nanmedian, d2_pl)), 4),
            "off_pitch_share": share(pl_off[graded], within=present[graded]),
        },
        "stillness": {
            "median_player_speed_ms_graded": q(med_speed[graded]),
            **{f"share_graded_median_below_{t:g}ms": share(med_speed < t, within=graded) for t in STILL_MS},
        },
        "missing_ball_context": {
            "share_of_graded": {"missing_players_moving": share(missing & active, within=graded),
                                "missing_players_mid": share(missing & ~active & ~slow, within=graded),
                                "missing_players_still": share(missing & slow, within=graded)},
            "live_gap_frames_by_gap_length": {
                f"{lo:g}-{hi:g}s": share((gap_len_at >= lo) & (gap_len_at < hi), within=live_gap)
                for lo, hi in zip(GAP_BUCKETS_S[:-1], GAP_BUCKETS_S[1:])},
        },
    }


def profile_tags(spec: GameSpec, graded: np.ndarray) -> dict | None:
    if not spec.smart_tagging or not Path(spec.smart_tagging).exists():
        return None
    inst = smart_tagging.load_instances(spec.smart_tagging, spec.fps)
    home, away = smart_tagging.team_names_from_meta(spec.atd_meta)
    side = np.array([smart_tagging.side_of(i["tags"].get("team", ""), home, away) for i in inst])
    code = np.array([i["code"] for i in inst])
    dur = np.array([(i["end"] - i["start"]) / spec.fps for i in inst])
    cover = {k: np.zeros(spec.n_frames, dtype=bool) for k in ("any", "possessing", "defending", "set_pieces")}
    for i in inst:
        s, e = max(0, i["start"]), min(spec.n_frames, i["end"])
        cover["any"][s:e] = True
        if i["code"] in smart_tagging.POSSESSING:
            cover["possessing"][s:e] = True
        elif i["code"] in smart_tagging.DEFENDING:
            cover["defending"][s:e] = True
        elif i["code"] == smart_tagging.SET_PIECES:
            cover["set_pieces"][s:e] = True
    sp = code == smart_tagging.SET_PIECES
    return {
        "instances": int(len(inst)),
        "by_code": {c: int(n) for c, n in zip(*np.unique(code, return_counts=True))},
        "team_share": {"home": share(side == 0), "away": share(side == 1), "unknown_or_na": share(side < 0)},
        "set_piece_team_unknown_share": share(side[sp] < 0) if sp.any() else None,
        "duration_s": {"set_pieces": q(dur[sp], (25, 50, 75)), "phases": q(dur[~sp], (25, 50, 75))},
        "coverage_of_graded": {k: share(v, within=graded) for k, v in cover.items()},
        "overlapping_phase_share": share(_overlap_count(inst, spec.n_frames) > 1, within=graded),
    }


def _overlap_count(inst, n):
    diff = np.zeros(n + 1, dtype=np.int32)
    for i in inst:
        if i["code"] == smart_tagging.SET_PIECES:
            continue
        s, e = max(0, i["start"]), min(n, i["end"])
        if e > s:
            diff[s] += 1
            diff[e] -= 1
    return np.cumsum(diff)[:n]


def profile_homography(path: Path, fps: int, graded: np.ndarray) -> dict | None:
    if not Path(path).exists():
        return None
    h = pd.read_csv(path, usecols=["frame", "reprojection_error", "valid"])
    valid = h["valid"].to_numpy().astype(bool)
    err = h["reprojection_error"].to_numpy(dtype=float)
    g = graded[: len(valid)] if graded.any() else np.ones(len(valid), dtype=bool)
    invalid = runs(~valid) / fps
    return {
        "rows": int(len(h)), "valid_share": share(valid), "valid_share_graded": share(valid, within=g),
        "reprojection_error_valid": q(err[valid & g], (50, 90, 99)),
        "invalid_stretches": {"n": int(len(invalid)), "length_s": q(invalid, (25, 50, 75, 95)),
                              "longest_s": round(float(invalid.max()), 1) if len(invalid) else None,
                              "share_5_to_60s": share((invalid >= 5) & (invalid <= 60)) if len(invalid) else None},
    }


def profile(spec: GameSpec, homography: Path | None = None, use_cache: bool = True) -> dict:
    trk = atd.load(spec, use_cache=use_cache)
    graded = graded_mask(spec)
    out = {"recording": spec.name, "tracking": profile_tracking(trk, spec, graded)}
    tags = profile_tags(spec, graded)
    if tags is not None:
        out["smart_tagging"] = tags
    if homography is not None:
        hom = profile_homography(homography, spec.fps, graded)
        if hom is not None:
            out["homography"] = hom
    return out


def _print(d: dict, indent: int = 0) -> None:
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict) and any(isinstance(x, dict) for x in v.values()):
            print(f"{pad}{k}:")
            _print(v, indent + 1)
        elif isinstance(v, dict):
            print(f"{pad}{k}: " + "  ".join(f"{a}={b}" for a, b in v.items()))
        else:
            print(f"{pad}{k}: {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", nargs="*", default=["tactical", "broadcast"], choices=["tactical", "broadcast"])
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    report = {}
    for source in args.source:
        spec = competition_game(source=source)
        if not spec.atd_raw.exists():
            print(f"[{source}] missing {spec.atd_raw.name} -- skipped")
            continue
        report[source] = profile(spec, homography_path(source=source), use_cache=not args.no_cache)
    print("=== mqf.profile -- aggregates only, safe to share ===")
    _print(report)
    out = REPORTS_DIR / "profile" / "profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
