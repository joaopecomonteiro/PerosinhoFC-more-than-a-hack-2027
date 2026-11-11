"""Aggregate-only profile of the match recordings: tactical (the reference) and broadcast.

    python -m mqf.profile                       # both recordings
    python -m mqf.profile --source tactical     # one of them
    python -m mqf.profile --source tactical --clean   # + the same numbers after mqf.tracks.clean

Prints ONE block of aggregates -- shares, counts, quantiles -- and writes the same numbers to
reports/profile/profile.json (gitignored). It never prints frame numbers, rows, positions or
images, so the printed block is what the team may paste to an AI assistant (CLAUDE.md, rule 1).

What it answers (docs: "Next steps", section 1):
  ball gaps (how many, how long), where the ball goes missing (players moving vs still -- the
  ceiling for any imputation model), tracking noise (jitter, speed field vs positions, ball
  jumps), tracklets (slots, lengths, objects per frame), stillness (median player speed), off-pitch
  shares, Smart Tagging coverage, homography validity; identity: tracks per team (> 11?), jumps
  faster than a sprinter, new tracks with a lost same-team track nearby (re-linkable), duplicates.

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
# identity diagnostics -- football knowledge, NOT fitted to this match (CLAUDE.md, rule 2)
VMAX_MS = 10.5                                      # elite sprint peak is ~36-38 km/h
JUMP_TOL_M = 1.0                                    # position error a tracker may add per step
GATE_M, GATE_MS = 2.0, 8.0                          # a lost player is within 2 m + 8 m/s x gap
DUP_M = 1.0                                         # two same-team tracks closer than this: one player


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


def _edges(on: np.ndarray):
    e = np.diff(np.r_[0, on.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1          # first / last frame of each run


def profile_identity(trk: atd.Tracking, spec: GameSpec, graded: np.ndarray) -> dict:
    """How badly the tracklets break player identity -- aggregates only.

    Answers: how often a team has more than 11 tracks at once; how often a track jumps faster than
    a human can run (frame to frame, and across short gaps); and, for every new track, whether a
    track of the same team ended nearby shortly before (the case where re-linking is easy), several
    did (ambiguous), or an active same-team track is right next to it (a duplicate, not a new player).
    """
    fps, n = spec.fps, trk.n_frames
    xy = trk.players
    present = ~np.isnan(xy[..., 0])
    if not graded.any():
        graded = present.any(axis=1)
    home = trk.team_of_slot == 0
    minutes = graded.sum() / fps / 60
    out = {"thresholds": {"vmax_ms": VMAX_MS, "jump_tol_m": JUMP_TOL_M, "gate": f"{GATE_M:g} m + {GATE_MS:g} m/s x gap",
                          "duplicate_m": DUP_M}}

    per_team = {"home": present[:, home].sum(axis=1), "away": present[:, ~home].sum(axis=1)}
    out["tracked_per_team"] = {k: q(v[graded], (5, 50, 95), 1) for k, v in per_team.items()}
    out["share_frames_over_11"] = {k: share(v > 11, within=graded) for k, v in per_team.items()}

    # jumps: displacement between consecutive observations of a slot vs what a sprinter can cover
    j1 = jg = 0
    sizes = []
    for k in range(xy.shape[1]):
        idx = np.flatnonzero(present[:, k])
        if len(idx) < 2:
            continue
        d = np.linalg.norm(np.diff(xy[idx, k], axis=0), axis=1)
        gap = np.diff(idx)
        jump = (d > VMAX_MS * gap / fps + JUMP_TOL_M) & (gap <= 2 * fps) & graded[idx[1:]]
        j1 += int(np.sum(jump & (gap == 1)))
        jg += int(np.sum(jump & (gap > 1)))
        sizes.append(d[jump])
    sizes = np.concatenate(sizes) if sizes else np.zeros(0)
    out["jumps_per_min"] = {"frame_to_frame": round(j1 / minutes, 2) if minutes else None,
                            "across_gap_le_2s": round(jg / minutes, 2) if minutes else None}
    out["jump_size_m"] = q(sizes, (50, 90))

    # births (a track starts) vs deaths (a track ends) of the same team
    first_graded = {a for _, a, _ in spec.halves} if spec.halves else set()
    births, deaths = [], []
    for k in range(xy.shape[1]):
        s, e = _edges(present[:, k])
        births += [(t, k) for t in s if graded[t] and t not in first_graded and t > 0]
        deaths += [(t, k) for t in e if graded[t] and t < n - 1]
    deaths_by_team = {}
    for team in (0, 1):
        d = sorted((t, k) for t, k in deaths if trk.team_of_slot[k] == team)
        deaths_by_team[team] = (np.array([t for t, _ in d], dtype=np.int64),
                                np.array([xy[t, k] for t, k in d]).reshape(-1, 2))
    win = 10 * fps
    has = {1: 0, 5: 0, 10: 0}
    ambiguous = dup = 0
    nearest5 = []
    for t, k in births:
        team = trk.team_of_slot[k]
        p = xy[t, k]
        others = np.flatnonzero(present[t] & (trk.team_of_slot == team))
        others = others[others != k]
        if len(others) and np.min(np.linalg.norm(xy[t, others] - p, axis=1)) < DUP_M:
            dup += 1
        td, pd_ = deaths_by_team[team]
        lo, hi = np.searchsorted(td, t - win), np.searchsorted(td, t)
        if hi <= lo:
            continue
        dt = (t - td[lo:hi]) / fps
        dist = np.linalg.norm(pd_[lo:hi] - p, axis=1)
        ok = dist <= GATE_M + GATE_MS * dt
        for s_ in has:
            has[s_] += int(np.any(ok & (dt <= s_)))
        ambiguous += int(np.sum(ok & (dt <= 2)) >= 2)
        if np.any(dt <= 5):
            nearest5.append(dist[dt <= 5].min())
    nb = max(len(births), 1)
    out["births_per_min"] = round(len(births) / minutes, 1) if minutes else None
    out["births_with_lost_track_in_gate"] = {f"within_{s_}s": round(v / nb, 3) for s_, v in has.items()}
    out["births_ambiguous_2plus_candidates_2s"] = round(ambiguous / nb, 3)
    out["births_next_to_active_same_team_track"] = round(dup / nb, 3)
    out["births_nearest_lost_track_m_5s"] = q(nearest5, (25, 50, 75, 90))

    # duplicates: frames where two tracks of one team sit on top of each other
    dupf = np.zeros(n, dtype=bool)
    for mask in (home, ~home):
        P = xy[:, mask]
        for a in range(0, n, 5000):
            c = P[a:a + 5000]
            D = np.linalg.norm(c[:, :, None, :] - c[:, None, :, :], axis=-1)
            iu = np.triu_indices(c.shape[1], 1)
            with np.errstate(invalid="ignore"):
                dupf[a:a + 5000] |= np.any(D[:, iu[0], iu[1]] < DUP_M, axis=1)
    out["share_frames_same_team_pair_within_1m"] = share(dupf, within=graded)

    # one person under both team labels? home and away tracks on top of each other
    cross = np.zeros(n, dtype=bool)
    H, A = xy[:, home], xy[:, ~home]
    for a in range(0, n, 5000):
        D = np.linalg.norm(H[a:a + 5000, :, None, :] - A[a:a + 5000, None, :, :], axis=-1)
        with np.errstate(invalid="ignore"):
            cross[a:a + 5000] = np.any(D < 0.3, axis=(1, 2))
    out["share_frames_home_away_pair_within_0.3m"] = share(cross, within=graded)
    return out


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


def profile(spec: GameSpec, homography: Path | None = None, use_cache: bool = True,
            clean: bool = False) -> dict:
    trk = atd.load(spec, use_cache=use_cache)
    graded = graded_mask(spec)
    out = {"recording": spec.name, "tracking": profile_tracking(trk, spec, graded),
           "identity": profile_identity(trk, spec, graded)}
    if clean:
        from . import tracks
        ctrk, report = tracks.clean(trk, spec.fps, spec.pitch)
        out["cleaning"] = report
        out["identity_after_cleaning"] = profile_identity(ctrk, spec, graded)
        out["tracking_after_cleaning"] = {k: v for k, v in profile_tracking(ctrk, spec, graded).items()
                                          if k in ("players", "stillness")}
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
    ap.add_argument("--clean", action="store_true", help="also profile after team-level cleaning (mqf.tracks)")
    args = ap.parse_args()
    report = {}
    for source in args.source:
        spec = competition_game(source=source)
        if not spec.atd_raw.exists():
            print(f"[{source}] missing {spec.atd_raw.name} -- skipped")
            continue
        report[source] = profile(spec, homography_path(source=source), use_cache=not args.no_cache, clean=args.clean)
    print("=== mqf.profile -- aggregates only, safe to share ===")
    _print(report)
    out = REPORTS_DIR / "profile" / "profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
