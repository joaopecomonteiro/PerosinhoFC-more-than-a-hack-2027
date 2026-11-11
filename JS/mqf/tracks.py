"""Clean the ATD tracklets at TEAM level: no player identities, just a plausible set of at most
11 points per team per frame, with speeds that never cross a broken track.

Why (mqf.profile on BAR-ATH tactical, aggregates): a team shows > 11 tracks on 8 % (home) / 5 %
(away) of graded frames; 12 % of frames have two same-team tracks within 1 m; 20 % of new tracks
start within 1 m of an active same-team track; ~1.4 jumps per minute faster than a sprinter (2-4 m
in one frame: hops between players). Our features are collective, so identity does not matter --
duplicates, ghosts and fake speeds do.

v1 hid the older track of a duplicate pair for the newer one's whole life and capped by "older of
the closest pair". On BAR-ATH that removed 8 % of all points and raised frames with < 18 players
from 9 % to 23 %: real players were deleted. In half the duplicate episodes the newer track died
first, so neither "older" nor "newer" is reliably the true one. v2 is conservative:

Steps (thresholds from football knowledge in CleanConfig -- never fitted to the match):
  1 cut     a track that moves more than vmax * dt + tol between two frames becomes two tracklets
  2 dedup   two tracklets of one team closer than dup_m, with velocities within dup_dv, for at
            least dup_min_s, are one player: the OLDER is hidden while they stay together (until
            they are more than release_m apart, or one ends). If the older ends meanwhile, the newer
            continues it (stitched into its lane). Nobody is hidden once the pair separates.
  3 cap     a team still above 11 in a frame: hide the least player-like track -- outside the pitch
            first, then a frozen track while its team moves (never inside a penalty area: the
            goalkeeper), then one right next to a team-mate,
            then the youngest (the team already had 11 when it appeared)
  4 speed   recomputed per tracklet from positions (centred, +/- 0.2 s), never across a cut
Output: a Tracking on the same frames with team-ordered lanes (a lane carries successive tracklets
of one team) and a report of aggregate counts -- features and decoder are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atd import Tracking


@dataclass(frozen=True)
class CleanConfig:
    vmax_ms: float = 10.5          # elite sprint peak ~36-38 km/h
    jump_tol_m: float = 1.0        # position error a tracker may add between two frames
    dup_m: float = 1.0             # two team-mates are almost never this close while moving alike
    dup_dv_ms: float = 1.0
    dup_min_s: float = 0.4         # sustained, so two players crossing are not merged
    release_m: float = 2.0         # a duplicate pair this far apart is two tracks again
    max_per_team: int = 11
    off_pitch_m: float = 1.0       # beyond the lines by more than this: bench, warm-up, officials
    frozen_ms: float = 0.05        # a track this still for frozen_s while its team moves is a ghost; a
    frozen_s: float = 2.0          # standing player still sways (> 0.1 m/s), and goalkeepers in their
    box_depth_m: float = 16.5      # penalty area are never called frozen
    box_half_width_m: float = 20.16
    team_moving_ms: float = 1.0
    near_mate_m: float = 1.5
    vel_half_window_s: float = 0.2
    lane_gap_s: float = 2.0        # unrelated tracklets share a lane only this far apart, so nothing
                                   # downstream (diffs, profile) can read a lane change as a jump


def _tracklets(present: np.ndarray, xy: np.ndarray, fps: int, cfg: CleanConfig):
    """[(slot, start, end)] inclusive: runs of presence, cut where a step exceeds what a sprinter covers."""
    out, cuts = [], 0
    step_max = cfg.vmax_ms / fps + cfg.jump_tol_m
    for k in range(present.shape[1]):
        e = np.diff(np.r_[0, present[:, k].astype(np.int8), 0])
        for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1):
            jump = np.zeros(0, dtype=np.int64)
            if t > s:
                d = np.linalg.norm(np.diff(xy[s:t + 1, k], axis=0), axis=1)
                jump = np.flatnonzero(d > step_max) + s + 1          # first frame after each jump
            cuts += len(jump)
            b = np.r_[s, jump, t + 1]
            out += [(k, int(a), int(c) - 1) for a, c in zip(b[:-1], b[1:])]
    return out, cuts


def _velocity(seg: np.ndarray, w: int, fps: int) -> np.ndarray:
    """(m, 2) m/s inside one tracklet: centred difference, one-sided at the ends, NaN if one frame."""
    m = len(seg)
    if m < 2:
        return np.full((m, 2), np.nan)
    i = np.arange(m)
    a, b = np.maximum(i - w, 0), np.minimum(i + w, m - 1)
    return (seg[b] - seg[a]) / ((b - a)[:, None] / fps)


def _runs(mask: np.ndarray):
    """(starts, ends exclusive) of True runs."""
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def _episodes(tid, xy, vel, team_of_slot, cfg: CleanConfig, fps: int):
    """Duplicate contacts: [(tracklet_a, tracklet_b, first_frame, last_frame)] inclusive."""
    min_run = max(1, int(round(cfg.dup_min_s * fps)))
    out = []
    for team in (0, 1):
        slots = np.flatnonzero(team_of_slot == team)
        for ii, i in enumerate(slots):
            for j in slots[ii + 1:]:
                idx = np.flatnonzero((tid[:, i] >= 0) & (tid[:, j] >= 0))
                if len(idx) < min_run:
                    continue
                key = tid[idx, i].astype(np.int64) * 1_000_000 + tid[idx, j]
                brk = np.flatnonzero((np.diff(idx) != 1) | (np.diff(key) != 0)) + 1
                for seg in np.split(np.arange(len(idx)), brk):            # same two tracklets, contiguous
                    if len(seg) < min_run:
                        continue
                    f = idx[seg]
                    with np.errstate(invalid="ignore"):
                        d = np.linalg.norm(xy[f, i] - xy[f, j], axis=1)
                        dv = np.linalg.norm(vel[f, i] - vel[f, j], axis=1)
                    close = (d < cfg.dup_m) & (dv < cfg.dup_dv_ms)
                    apart = ~(d < cfg.release_m)                          # NaN counts as apart
                    s_, e_ = _runs(close)
                    pos = 0
                    for a, b in zip(s_, e_):
                        if b - a < min_run or a < pos:
                            continue
                        rel = np.flatnonzero(apart[a:])
                        stop = a + (rel[0] if len(rel) else len(f) - a)  # exclusive
                        out.append((int(tid[f[a], i]), int(tid[f[a], j]), int(f[a]), int(f[stop - 1])))
                        pos = stop
    return out


def _slow_run(speed: np.ndarray, thr: float) -> np.ndarray:
    """(n, S) frames for which each slot has been slower than thr (NaN breaks the run)."""
    slow = np.nan_to_num(speed, nan=np.inf) < thr
    out = np.zeros(slow.shape, dtype=np.int32)
    run = np.zeros(slow.shape[1], dtype=np.int32)
    for t in range(len(slow)):
        run = np.where(slow[t], run + 1, 0)
        out[t] = run
    return out


def clean(trk: Tracking, fps: int, pitch=None, cfg: CleanConfig = CleanConfig()) -> tuple[Tracking, dict]:
    """pitch (L, W) in metres enables the off-pitch test of the cap."""
    xy = trk.players.astype(np.float64)
    n, S, _ = xy.shape
    team = trk.team_of_slot
    present = ~np.isnan(xy[..., 0])

    # 1 cut at jumps
    tr, cuts = _tracklets(present, xy, fps, cfg)
    T = len(tr)
    slot = np.array([k for k, _, _ in tr], dtype=np.int64)
    start = np.array([s for _, s, _ in tr], dtype=np.int64)
    end = np.array([t for _, _, t in tr], dtype=np.int64)
    tid = np.full((n, S), -1, dtype=np.int32)
    vel = np.full((n, S, 2), np.nan)
    w = max(1, int(round(cfg.vel_half_window_s * fps)))
    for q, (k, s, t) in enumerate(tr):
        tid[s:t + 1, k] = q
        vel[s:t + 1, k] = _velocity(xy[s:t + 1, k], w, fps)
    speed = np.linalg.norm(vel, axis=-1)

    # 2 dedup: hide the older while the pair stays together
    hidden = [[] for _ in range(T)]
    pred = np.full(T, -1, dtype=np.int64)
    handovers = newer_first = 0
    eps = _episodes(tid, xy, vel, team, cfg, fps)
    for a, b, t0, t1 in eps:
        old, new = (a, b) if (start[a], a) < (start[b], b) else (b, a)
        hidden[old].append((t0, t1))
        if t1 >= end[old] and end[new] > end[old]:
            handovers += 1                                   # the old track died inside the contact
            if pred[new] < 0:
                pred[new] = old
        elif t1 >= end[new]:
            newer_first += 1
    visible = present.copy()
    for q in range(T):
        for s, t in hidden[q]:
            visible[s:t + 1, slot[q]] = False
    dup_hidden = int(present.sum() - visible.sum())

    # 3 cap at max_per_team: hide the least player-like track
    over_before = {name: float(np.mean((present & (team == tm)).sum(axis=1) > cfg.max_per_team))
                   for name, tm in (("home", 0), ("away", 1))}
    frozen_run = _slow_run(speed, cfg.frozen_ms)
    frozen_frames = int(round(cfg.frozen_s * fps))
    reasons = {name: {"off_pitch": 0, "frozen": 0, "next_to_mate": 0, "youngest": 0} for name in ("home", "away")}
    cap_frames = {"home": 0, "away": 0}
    other_le_10 = {"home": 0, "away": 0}           # the other team short of a player: a mislabelled opponent?
    nearest_ball = {"home": [0, 0], "away": [0, 0]}  # [drops with the ball seen, of which the team's nearest to it]
    for name, tm in (("home", 0), ("away", 1)):
        mask = team == tm
        cnt = (visible & mask).sum(axis=1)
        for t in np.flatnonzero(cnt > cfg.max_per_team):
            cap_frames[name] += 1
            other_le_10[name] += int((visible[t] & ~mask).sum() <= cfg.max_per_team - 1)
            live = list(np.flatnonzero(visible[t] & mask))
            ball = trk.ball[t]
            team_speed = np.nanmedian(speed[t, live]) if np.isfinite(speed[t, live]).any() else 0.0
            while len(live) > cfg.max_per_team:
                P = xy[t, live]
                if pitch is not None:
                    L, W = pitch
                    out_m = np.maximum.reduce([-P[:, 0], P[:, 0] - L, -P[:, 1], P[:, 1] - W, np.zeros(len(P))])
                else:
                    out_m = np.zeros(len(P))
                frozen = (frozen_run[t, live] >= frozen_frames) & (team_speed > cfg.team_moving_ms)
                if pitch is not None:
                    in_box = (np.minimum(P[:, 0], L - P[:, 0]) < cfg.box_depth_m) & \
                             (np.abs(P[:, 1] - W / 2) < cfg.box_half_width_m)
                    frozen &= ~in_box
                D = np.linalg.norm(P[:, None] - P[None], axis=-1)
                np.fill_diagonal(D, np.inf)
                near = D.min(axis=1) < cfg.near_mate_m
                age = t - start[tid[t, live]]
                if (out_m > cfg.off_pitch_m).any():
                    k, why = int(np.argmax(out_m)), "off_pitch"
                elif frozen.any():
                    k, why = int(np.flatnonzero(frozen)[np.argmin(age[frozen])]), "frozen"
                elif near.any():
                    k, why = int(np.flatnonzero(near)[np.argmin(age[near])]), "next_to_mate"
                else:
                    k, why = int(np.argmin(age)), "youngest"
                if np.isfinite(ball[0]):
                    nearest_ball[name][0] += 1
                    nearest_ball[name][1] += int(k == int(np.argmin(np.linalg.norm(P - ball, axis=1))))
                visible[t, live[k]] = False
                reasons[name][why] += 1
                live.pop(k)

    # 4 lanes + speeds: visible pieces of each tracklet (hidden intervals split it)
    pieces = []
    for q in range(T):
        bounds, s = [], start[q]
        for hs, he in sorted(hidden[q]):
            if hs > s:
                bounds.append((s, hs - 1))
            s = max(s, he + 1)
        if s <= end[q]:
            bounds.append((s, end[q]))
        pieces += [(int(a), int(b), q) for a, b in bounds if visible[a:b + 1, slot[q]].any()]
    pieces.sort(key=lambda p: (team[slot[p[2]]], p[0], p[2]))

    gap = int(round(cfg.lane_gap_s * fps)) + 1
    lanes = {0: [], 1: []}                                   # per team: [last_end, last_tracklet]
    lane_of_piece = []
    for a, b, q in pieces:
        tm = int(team[slot[q]])
        L_ = lanes[tm]
        choice = None
        if pred[q] >= 0:                                     # stitch onto the older one's lane if free
            for li, (le, lq) in enumerate(L_):
                if lq == pred[q] and le < a:
                    choice = li
                    break
        if choice is None:
            for li, (le, _) in enumerate(L_):
                if le < a - gap:
                    choice = li
                    break
        if choice is None:
            L_.append([b, q])
            choice = len(L_) - 1
        else:
            L_[choice] = [b, q]
        lane_of_piece.append((tm, choice))
    n_home, n_away = len(lanes[0]), len(lanes[1])
    out_xy = np.full((n, n_home + n_away, 2), np.nan, dtype=np.float32)
    out_sp = np.full((n, n_home + n_away), np.nan, dtype=np.float32)
    for (a, b, q), (tm, li) in zip(pieces, lane_of_piece):
        col = li if tm == 0 else n_home + li
        seg = xy[a:b + 1, slot[q]]
        vis = visible[a:b + 1, slot[q]]
        sp = np.linalg.norm(_velocity(seg, w, fps), axis=1)
        out_xy[a:b + 1, col] = np.where(vis[:, None], seg, np.nan)
        out_sp[a:b + 1, col] = np.where(vis, sp, np.nan)
    team_out = np.r_[np.zeros(n_home, np.int8), np.ones(n_away, np.int8)]
    after = ~np.isnan(out_xy[..., 0])
    any_frame = present.any(axis=1)
    report = {
        "tracklets_in": int(T - cuts), "cuts_at_jumps": int(cuts),
        "duplicate_episodes": len(eps), "duplicate_points_hidden": dup_hidden,
        "episodes_handover_old_died": int(handovers), "episodes_newer_died_first": int(newer_first),
        "cap_frames": cap_frames, "cap_points_hidden": int(sum(sum(r.values()) for r in reasons.values())), "cap_reasons": reasons,
        "cap_frames_other_team_le_10": {k: round(v / max(cap_frames[k], 1), 3) for k, v in other_le_10.items()},
        "cap_drop_was_team_nearest_to_ball": {k: round(v[1] / v[0], 3) if v[0] else None for k, v in nearest_ball.items()},
        "share_frames_over_11": {"before": over_before,
                                 "after": {name: float(np.mean((after & (team_out == tm)).sum(axis=1) > cfg.max_per_team))
                                           for name, tm in (("home", 0), ("away", 1))}},
        "players_per_frame_mean": {"before": round(float(present[any_frame].sum(axis=1).mean()), 2) if any_frame.any() else None,
                                   "after": round(float(after[any_frame].sum(axis=1).mean()), 2) if any_frame.any() else None},
        "points_kept_share": round(float(after.sum() / max(present.sum(), 1)), 4),
        "lanes": {"home": n_home, "away": n_away},
    }
    return Tracking(players=out_xy, player_speed=out_sp, ball=trk.ball, team_of_slot=team_out), report
