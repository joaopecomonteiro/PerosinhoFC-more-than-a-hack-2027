"""Clean the ATD tracklets at TEAM level: no player identities, just a plausible set of at most
11 points per team per frame, each on the right team, with speeds that never cross a broken track.

Why (mqf.profile on BAR-ATH tactical, aggregates): a team shows > 11 tracks on 8 % (home) / 5 %
(away) of graded frames; 12 % of frames have two same-team tracks within 1 m; 20 % of new tracks
start within 1 m of an active same-team track; ~1.4 jumps per minute faster than a sprinter (2-4 m
in one frame: hops between players). Our features are collective, so identity does not matter --
duplicates, ghosts, fake speeds and tracks on the wrong team do.

v1 hid the older track of a duplicate pair for the newer one's whole life and capped by "older of
the closest pair". On BAR-ATH that removed 8 % of all points and raised frames with < 18 players
from 9 % to 23 %: real players were deleted. In half the duplicate episodes the newer track died
first, so neither "older" nor "newer" is reliably the true one. v2 is conservative.

v4 (13 Nov) adds what the viewer showed: tracks sliding across the pitch over several frames (each
step under the one-frame jump limit), and new tracks -- the goalkeeper's above all -- starting on the
wrong team exactly where a track of the other team had just been. The ATD gives a team per SLOT,
from the tracker's kit classification; a keeper's kit is neither team's, and a player picked up again
gets a new slot. So here a team belongs to a TRACKLET, and it can be repaired.

Steps (thresholds from football knowledge and the Laws of the Game in CleanConfig -- never fitted to
the match):
  1 cut     a track that moves more than vmax * dt + tol between two frames becomes two tracklets;
            where it covers more than vmax over a window of fast_windows_s (+ tol in total, not per
            frame) its points are hidden and it is cut there too -- a slide at 20-30 m/s spread over
            several frames never makes a single big hop
  2 team    a tracklet's team is its slot's, repaired by three rules:
            kickoff  at a kickoff (the ball seen on the centre spot, or a tagged kickoff when it is lost;
                     both teams standing in their own halves for a second, a kicker at the spot, the
                     circle nearly empty) every player is in his own
                     half (Law 8): a track more than kickoff_margin_m into the other half and outside
                     the centre circle for the whole wait has the wrong team -> flipped, but only to a team
                     that is short of 11 at that moment, and if the player counts agree over the
                     whole life of that track (otherwise it is an official or a lean)
            inherit  a new tracklet that starts where exactly one other tracklet just ended (within
                     inherit_gap_s and running reach) or stands frozen (a held copy) is that player
                     picked up again; with nobody like that, a track that vanished up to inherit_memory_s
                     earlier within inherit_same_place_m of the new one (its saved last place) is too:
                     it takes that tracklet's team -- only if that one lived at least
                     inherit_min_life_s, nobody else is within inherit_alone_m of the new track (a
                     player reappearing next to a moving opponent is a duel, never touched), and the
                     player counts agree: over the new track's life its label gives its team at least
                     inherit_count_margin more players than usual against the other (a wrong-team
                     track adds one there and takes one away, ~ +2). Without that last check a wrong
                     predecessor passes its error on as often as a right one fixes a pickup. The
                     frozen copy is hidden while it stays frozen and its pickup is there -- it shows
                     again (the saved place) if the pickup ends first. Repairs carry forward.
            keeper   a track that appears alone in a penalty area, deepest of all players, is the
                     keeper of the team defending that goal. v4g (Sigmar): tracks that do not move
                     (frozen, or the tracker repeating a lost player's place) are no players -- they do
                     not stop it being alone or deepest, and they are hidden while they stay still; the
                     defending team is the one counted on that side at the START of the half (more of its
                     players in that half over the first side_count_s, Law 8), constant for the half; no
                     player-count check (a keeper on the wrong colour while the other keeper is lost
                     leaves the counts even); it must stay in that box on keeper_box_stay_share of its
                     life (a keeper does, an attacker picked up there and running out does not).
                     (v4c-v4f: sides from the teams' medians, still tracks counted, count check on.)
            takeover (v4d/v4e, Sigmar) only for a pickup the three rules above leave undecided (not
                     alone, crowded, ambiguous, counts): a new track within keeper_near_m of the OTHER
                     team's goalkeeper is that keeper. The keeper = that team's track last seen deepest of
                     all in its own penalty area (as for keeper fill) within keeper_hold_s. If he is still
                     tracked (v4e): appearing next to him is enough -- the new track takes his team and his
                     old track is hidden while it is held (repeats its place exactly; this tracker holds a
                     lost player ~3 s) or still within keeper_near_m of the new one; it shows again if they
                     separate. (v4d asked more: the new track on his held copy or within dup_m of him on
                     every shared frame -- keeper_live_test="held_or_duplicate".) If he is gone, his saved
                     last place counts. Always only if the player counts agree (same test as pickups): a
                     real attacker next to a live keeper leaves the counts even and is left alone. Why:
                     every rule above needs the new track alone, and the keeper right next to it made it
                     never alone. v4f: when the counts are ambiguous, the new track's behaviour decides --
                     in the keeper's penalty area and deeper than every other player on keeper_behaviour_share
                     of its life (>= keeper_behaviour_min_s) is the keeper (an outfield player that deep for
                     that long would be offside, Law 11); an attacker who touches the keeper and leaves is not
  3 dedup   two tracklets of one team closer than dup_m, with velocities within dup_dv, for at
            least dup_min_s, are one player: the OLDER is hidden while they stay together (until
            they are more than release_m apart, or one ends). If the older ends meanwhile, the newer
            continues it (stitched into its lane). Nobody is hidden once the pair separates.
  4 cap     a team still above 11 in a frame: hide the least player-like track -- outside the pitch
            first, then a frozen track while its team moves (never inside a penalty area: the
            goalkeeper), then one right next to a team-mate,
            then the youngest (the team already had 11 when it appeared)
  5 speed   recomputed per tracklet from positions (centred, +/- 0.2 s), never across a cut
  6 keeper  a team short of 11 with nobody of its own in its penalty area (play on): one more lane
            shows its keeper at his saved last place (no speed) -- one keeper, however many are missing;
            saved only where he was the deepest of all players, held at most keeper_hold_s
Output: a Tracking on the same frames with team-ordered lanes (a lane carries successive tracklets
of one team) and a report of aggregate counts -- features and decoder are unchanged.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .atd import Tracking


@dataclass(frozen=True)
class CleanConfig:
    vmax_ms: float = 10.5          # elite sprint peak ~36-38 km/h
    jump_tol_m: float = 1.0        # position error a tracker may add between two frames
    fast_windows_s: tuple = (0.2, 1.0)  # the speed limit also holds over these spans (() = off)
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
    # team repair (v4)
    inherit_team: bool = True
    inherit_gap_s: float = 2.0     # a pickup this soon after a track ended can be the same player
    inherit_reach_m: float = 5.0   # ... and no farther than this (+ jump_tol_m) from where it ended
    inherit_min_life_s: float = 2.0  # the team of a track that short is no better than the new one's
    inherit_alone_m: float = 3.0   # anyone else this close at the pickup: ambiguous (duel, crowd)
    ghost_min_s: float = 0.4       # frozen this long before a pickup next to it: a held copy
    ghost_reshow: bool = True      # a held copy shows again (the saved place) once its pickup ends
    cap_held_first: bool = False   # over 11: hide a held copy (frozen >= frozen_s, even in a box) first
    inherit_memory_s: float = 30.0  # a track that ended longer ago is still remembered at its last place ...
    inherit_same_place_m: float = 2.0  # ... for a pickup this close to it (a keeper who stood unseen)
    inherit_count_margin: float = 1.5  # over the new track's life its label must give its team at least
                                   # this many more players than usual vs the other (a wrong-team track
                                   # adds one there and takes one from the other: ~ +2)
    # goalkeepers (v4c): one per team, in his own penalty area
    keeper_box_pickup: bool = True  # a track that appears alone in a penalty area, deepest of all, is the keeper
    keeper_alone_m: float = 3.0    # ... of the team defending that goal (sides as at the kickoff)
    keeper_box_count_check: bool = False  # v4g (Sigmar): off -- the keeper's team is the side's (counted at the half
                                   # start); v4c-v4f: on (only if that team is the one short of a player). A keeper on
                                   # the wrong colour while the other keeper is lost leaves the counts even
    keeper_box_stay_share: float = 0.5  # v4g: ... and it stays in that penalty area on this share of its life (a keeper
                                   # does; an attacker picked up there and running out does not)
    keeper_box_ignore_still: bool = True  # v4g: a track that does not move (frozen >= ghost_min_s, or the tracker
                                   # repeats its place exactly) is no player for the box rule -- it does not stop the
                                   # new track being alone or deepest, and it is hidden while it stays still
    keeper_fill: bool = True       # a team short of 11 with nobody of its own in its penalty area: add its
    keeper_spot_m: float = 5.5     # keeper at his saved last place (never seen yet in this half: the 6-yard line)
    keeper_min_tracked: int = 16   # ... only while play is on (most players tracked)
    keeper_hold_s: float = 10.0    # ... for at most this long after he was last seen there (0 = no limit)
    # keeper takeover (v4d, Sigmar): a new track next to the OTHER team's keeper is that keeper
    keeper_takeover: bool = True   # only for pickups the rules above leave undecided (not alone, crowded, ...)
    keeper_near_m: float = 3.0     # "next to": the keeper's held copy, a duplicate on him, or his saved place
    keeper_takeover_count_check: bool = True  # ... and his team must be the one short (same test as pickups)
    keeper_live_test: str = "near"  # keeper still tracked: "near" (v4e, Sigmar) = appearing within keeper_near_m
                                   # of him is enough -- his old track is hidden while it is held or still within
                                   # keeper_near_m of the new one; "held_or_duplicate" (v4d) = the new track must sit
                                   # on his held copy or within dup_m of him on every shared frame
    keeper_takeover_first: bool = False  # try the takeover before the pickup rule (which may pass on a wrong team)
    keeper_behaviour: bool = True  # (v4f) counts refuse, but the new track behaves like that keeper: in his penalty
    keeper_behaviour_share: float = 0.8  # area and deeper than every other player on this share of its life
    keeper_behaviour_min_s: float = 2.0  # ... over at least this long (an attacker deeper than everyone, keeper
    keeper_behaviour_tol_m: float = 0.5  # included, is offside -- Law 11; level within this counts as deepest)
    side_window_s: float = 60.0    # which goal each team defends: the two teams' medians, smoothed over this ...
    side_from_half_start: bool = True  # ... v4g (Sigmar): counted once per half instead (when the halves are given):
    side_count_s: float = 60.0     # over its first side_count_s, the team with more players left of the halfway
                                   # line defends x = 0 for the whole half (Law 8: everyone in his own half to start)
    kickoff_check: bool = True
    kickoff_spot_m: float = 1.5    # the ball, if seen, this close to the centre spot
    kickoff_still_s: float = 1.0   # the kickoff shape held this long (ball not needed: often lost when still)
    kickoff_still_ms: float = 1.5  # ... with the median player slower than this (waiting for the whistle)
    kickoff_kicker_m: float = 3.0  # ... and someone standing at the centre spot (the kicker)
    kickoff_tag_pad_s: float = 3.0  # a tagged kickoff window, widened by this much on both sides
    kickoff_wait_s: float = 2.0    # wrong sides are read over the last part of the wait, up to this long
    kickoff_min_per_team: int = 7
    kickoff_side_share: float = 0.8  # tracked players outside the circle in their team's half
    kickoff_margin_m: float = 2.0  # this far into the other half for the whole wait: a wrong team, not a lean
    kickoff_max_flips: int = 2     # more tracks than this on the wrong side: not a kickoff shape
    kickoff_max_in_circle: int = 4  # the kicker and a few team-mates; open play around the spot has more
    centre_circle_m: float = 9.15


def parse_clean_cfg(items) -> CleanConfig:
    """["inherit_team=0", "fast_windows_s=0.2,1"] -> CleanConfig (bools 0/1, tuples comma-separated, empty = ()).
    The --clean-cfg / --fixed-cfg parser of the viewer, the harness and mqf.external.evaluate. It lives
    here, next to CleanConfig, so the viewer needs nothing from mqf.external."""
    from dataclasses import replace
    base = CleanConfig()
    out = {}
    for kv in items or []:
        k, _, v = kv.partition("=")
        if not hasattr(base, k):
            raise SystemExit(f"--clean-cfg: unknown CleanConfig field {k}")
        cur = getattr(base, k)
        if isinstance(cur, bool):
            out[k] = v.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, tuple):
            out[k] = tuple(float(x) for x in v.split(",") if x.strip())
        elif isinstance(cur, str):
            out[k] = v.strip()
        else:
            out[k] = type(cur)(float(v))
    return replace(base, **out)


def _runs(mask: np.ndarray):
    """(starts, ends exclusive) of True runs."""
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def _fast(seg: np.ndarray, fps: float, cfg: CleanConfig) -> np.ndarray:
    """(m,) True where one tracklet covers more than vmax over any of cfg.fast_windows_s (centred,
    shortened at the ends), allowing jump_tol_m in total."""
    m = len(seg)
    bad = np.zeros(m, dtype=bool)
    if m < 3:
        return bad
    i = np.arange(m)
    for w_s in cfg.fast_windows_s:
        h = int(round(w_s * fps / 2))
        if h < 1:
            continue
        a, b = np.maximum(i - h, 0), np.minimum(i + h, m - 1)
        d = np.linalg.norm(seg[b] - seg[a], axis=1)
        bad |= d > cfg.vmax_ms * (b - a) / fps + cfg.jump_tol_m
    return bad


def _tracklets(present: np.ndarray, xy: np.ndarray, fps: float, cfg: CleanConfig):
    """([(slot, start, end)] inclusive, jump cuts, fast-window cuts, [(slot, first, last)] fast points).

    Runs of presence are cut where one step exceeds what a sprinter covers; inside each piece the
    points that break the speed limit over a window are dropped and the piece is cut around them."""
    out, fast_pts, cuts, fast_cuts = [], [], 0, 0
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
            for a, c in zip(b[:-1], b[1:]):                          # piece [a, c)
                bad = _fast(xy[a:c, k], fps, cfg)
                if not bad.any():
                    out.append((k, int(a), int(c) - 1))
                    continue
                bs, be = _runs(bad)
                fast_cuts += len(bs)
                fast_pts += [(k, int(a + x), int(a + y) - 1) for x, y in zip(bs, be)]
                gs, ge = _runs(~bad)
                out += [(k, int(a + x), int(a + y) - 1) for x, y in zip(gs, ge)]
    return out, cuts, fast_cuts, fast_pts


def _velocity(seg: np.ndarray, w: int, fps: float) -> np.ndarray:
    """(m, 2) m/s inside one tracklet: centred difference, one-sided at the ends, NaN if one frame."""
    m = len(seg)
    if m < 2:
        return np.full((m, 2), np.nan)
    i = np.arange(m)
    a, b = np.maximum(i - w, 0), np.minimum(i + w, m - 1)
    return (seg[b] - seg[a]) / ((b - a)[:, None] / fps)


def _slow_run(speed: np.ndarray, thr: float) -> np.ndarray:
    """(n, S) frames for which each slot has been slower than thr (NaN breaks the run)."""
    slow = np.nan_to_num(speed, nan=np.inf) < thr
    out = np.zeros(slow.shape, dtype=np.int32)
    run = np.zeros(slow.shape[1], dtype=np.int32)
    for t in range(len(slow)):
        run = np.where(slow[t], run + 1, 0)
        out[t] = run
    return out


def _kickoff_shape(xy, team_ft, speed, ball, pitch, cfg: CleanConfig):
    """(n,) True on frames where the players stand in a kickoff shape -- no ball needed (this tracker
    often loses a ball that lies still): a player within kickoff_kicker_m of the centre spot (the
    kicker), at least kickoff_min_per_team tracked per team, the two teams'
    medians in different halves, at most kickoff_max_in_circle players inside the centre circle,
    at least kickoff_side_share of the others in their team's half, the median player slower than
    kickoff_still_ms, and the ball, if seen, on the centre spot. Also (n, S) side and depth."""
    L, W = pitch
    centre = np.array([L / 2, W / 2])
    x, y = xy[..., 0], xy[..., 1]
    tol = cfg.off_pitch_m
    with np.errstate(invalid="ignore"):
        on = (team_ft >= 0) & (x >= -tol) & (x <= L + tol) & (y >= -tol) & (y <= W + tol)
        home, away = on & (team_ft == 0), on & (team_ft == 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mh = np.nanmedian(np.where(home, x, np.nan), axis=1) - L / 2
            ma = np.nanmedian(np.where(away, x, np.nan), axis=1) - L / 2
            v = np.nanmedian(np.where(on, speed, np.nan), axis=1)
        side = np.where(team_ft == 0, np.sign(mh)[:, None], np.sign(ma)[:, None])   # -1: team's half is x < L/2
        depth = -(x - L / 2) * side                                                  # > 0: in the other half
        dist = np.linalg.norm(xy - centre, axis=-1)
        outside = on & (dist > cfg.centre_circle_m)
        n_out = outside.sum(axis=1)
        own = (outside & (depth <= 0)).sum(axis=1)
        bd = np.linalg.norm(np.asarray(ball, dtype=np.float64) - centre, axis=1)
        kicker = (on & (dist <= cfg.kickoff_kicker_m)).any(axis=1)                    # someone at the ball
        ok = (kicker & (home.sum(axis=1) >= cfg.kickoff_min_per_team) & (away.sum(axis=1) >= cfg.kickoff_min_per_team)
              & (mh * ma < 0) & ((on & ~outside).sum(axis=1) <= cfg.kickoff_max_in_circle)
              & (n_out > 0) & (own >= cfg.kickoff_side_share * n_out) & (v < cfg.kickoff_still_ms)
              & (np.isnan(bd) | (bd <= cfg.kickoff_spot_m)))
    return ok, side, depth, outside


def _kickoff_pins(xy, tid, tteam, speed, ball, fps: float, pitch, cfg: CleanConfig, windows=None):
    """({tracklet: corrected team}, kickoff frames). A kickoff: the kickoff shape (_kickoff_shape) held
    for at least kickoff_still_s while there is evidence of a kickoff -- the ball seen lying on the
    centre spot, or a tagged kickoff window (Smart Tagging, +/- kickoff_tag_pad_s) when the ball is
    lost. The shape alone is not enough: a slow build-up against a mid-block looks the same. Law 8:
    everyone is in his own half, so a track that stays more than kickoff_margin_m inside the other
    half, outside the circle, over the last kickoff_wait_s of the wait carries the wrong team. More
    than kickoff_max_flips such tracks: not a kickoff, skipped."""
    L, W = pitch
    team_ft = np.where(tid >= 0, tteam[np.maximum(tid, 0)], -1)
    ok, side, depth, outside = _kickoff_shape(xy, team_ft, speed, ball, pitch, cfg)
    n = len(ok)
    with np.errstate(invalid="ignore"):
        trigger = np.linalg.norm(np.asarray(ball, dtype=np.float64) - [L / 2, W / 2], axis=1) <= cfg.kickoff_spot_m
    pad = int(round(cfg.kickoff_tag_pad_s * fps))
    for a, b in windows or ():
        trigger[max(0, int(a) - pad):min(n, int(b) + pad + 1)] = True
    ok &= trigger
    still = max(1, int(round(cfg.kickoff_still_s * fps)))
    wait_max = max(still, int(round(cfg.kickoff_wait_s * fps)))
    pins, frames, details = {}, [], []
    for a, b in zip(*_runs(ok)):
        if b - a < still:
            continue
        w0 = max(int(a), int(b) - wait_max)                # the last part of the wait, before the kick
        f = np.arange(w0, int(b))
        t = (w0 + int(b) - 1) // 2
        P = xy[t]
        tol = cfg.off_pitch_m
        with np.errstate(invalid="ignore"):
            on_t = (tid[t] >= 0) & (P[:, 0] >= -tol) & (P[:, 0] <= L + tol) & (P[:, 1] >= -tol) & (P[:, 1] <= W + tol)
        n_team = [int((on_t & (team_ft[t] == tm)).sum()) for tm in (0, 1)]
        wrong = []
        for j in np.flatnonzero(outside[t] & (depth[t] > cfg.kickoff_margin_m)):
            if n_team[1 - int(team_ft[t, j])] >= cfg.max_per_team:
                continue                                   # that team has its 11: an official, not a player
            q = int(tid[t, j])
            same = tid[f, j] == q                          # the same tracklet over the wait
            if same.sum() < len(f) / 2:
                continue
            if np.all(outside[f[same], j]) and np.all(depth[f[same], j] > cfg.kickoff_margin_m):
                wrong.append(q)
        if len(wrong) > cfg.kickoff_max_flips:
            continue
        frames.append(t)
        for q in wrong:
            pins[q] = 1 - int(tteam[q])
            j = int(np.flatnonzero(tid[t] == q)[0])
            xq = xy[t, j]
            in_box = (min(xq[0], L - xq[0]) < cfg.box_depth_m) and (abs(xq[1] - W / 2) < cfg.box_half_width_m)
            details.append({"_q": int(q), "from": "home" if tteam[q] == 0 else "away", "to": "away" if tteam[q] == 0 else "home",
                            "depth_m": round(float(depth[t, j]), 1), "in_box": bool(in_box),
                            "track_s": round(float((tid[:, j] == q).sum()) / fps, 1)})
    return pins, frames, details


def _home_defends_left(xy, team_ft, fps: float, cfg: CleanConfig) -> np.ndarray:
    """(n,) True where the home team defends the goal at x = 0. The team nearer a goal (its median x)
    defends it; smoothed over side_window_s so attacks never flip it -- only the change of ends does."""
    import pandas as pd
    x = xy[..., 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mh = np.nanmedian(np.where(team_ft == 0, x, np.nan), axis=1)
        ma = np.nanmedian(np.where(team_ft == 1, x, np.nan), axis=1)
    w = max(1, int(round(cfg.side_window_s * fps)))
    d = pd.Series(mh - ma).rolling(2 * w + 1, center=True, min_periods=1).median().ffill().bfill()
    return (d.to_numpy() < 0) if d.notna().any() else np.ones(len(x), dtype=bool)


def _sides_from_half_start(xy, team_ft, halves, pitch, fps: float, cfg: CleanConfig, fallback):
    """(home_left, [per half]) -- which goal each team defends, counted once per half (Sigmar, v4g): over the
    first side_count_s of the half, player-frames of each team left / right of the halfway line; the team
    with more on the left defends x = 0 for the whole half (Law 8). Frames outside the halves keep the
    fallback (the teams' medians). A half with no evidence keeps the fallback too."""
    L = pitch[0]
    out = np.asarray(fallback, dtype=bool).copy()
    w = max(1, int(round(cfg.side_count_s * fps)))
    per = []
    for _, a, b in halves:
        a, b = int(a), int(b)
        x, t = xy[a:min(b + 1, a + w), :, 0], team_ft[a:min(b + 1, a + w)]
        with np.errstate(invalid="ignore"):
            left, right = x < L / 2, x >= L / 2
        h = int(((t == 0) & left).sum()) - int(((t == 0) & right).sum())
        aw = int(((t == 1) & left).sum()) - int(((t == 1) & right).sum())
        if h != aw:
            out[a:b + 1] = h > aw
        per.append(bool(out[a]) if a < len(out) else None)
    return out, per


def _inherit(tid, start, end, slot, xy, present, speed, frozen_run, tteam, pins, fps: float, cfg: CleanConfig,
             home_left=None, pitch=None, why: dict | None = None, sides_counted: bool = False):
    """Repair tteam in place, pickup by pickup in time order (see step 2 of the module docstring).
    Returns (counts, [(tracklet, first, last)] frozen copies to hide, {tracklets the keeper takeover changed})."""
    T = len(start)
    gap = max(1, int(round(cfg.inherit_gap_s * fps)))
    memory = max(gap, int(round(cfg.inherit_memory_s * fps)))
    min_life = int(round(cfg.inherit_min_life_s * fps))
    ghost_min = max(1, int(round(cfg.ghost_min_s * fps)))
    hold = int(round(cfg.keeper_hold_s * fps)) if cfg.keeper_hold_s > 0 else memory
    by_end = np.argsort(end, kind="stable")
    end_sorted = end[by_end]
    used = np.zeros(T, dtype=bool)                        # an ended track passes its team on once
    claimed_until = np.full(T, -1, dtype=np.int64)        # a held copy is free again once its pickup ends
    ghosts, taken = [], set()
    st = {"pickups_same_team": 0, "pickups_team_changed": 0, "from_saved_place": 0, "ambiguous": 0,
          "crowded": 0, "short_predecessor": 0, "counts_disagree": 0, "kickoff_pinned": 0,
          "keeper_box_pickups_changed": 0, "keeper_box_pickups_same": 0, "keeper_box_counts_disagree": 0,
          "kickoff_counts_disagree": 0, "keeper_takeover_held": 0, "keeper_takeover_duplicate": 0,
          "keeper_takeover_near": 0, "keeper_takeover_saved": 0, "keeper_takeover_apart": 0,
          "keeper_takeover_counts_disagree": 0, "keeper_takeover_by_behaviour": 0, "keeper_box_still_hidden": 0}
    # players per team and frame under the current labels: a team cannot field more than the other
    # for long, so a change must also even the counts out (relative to the usual difference)
    team_ft = np.where(tid >= 0, tteam[np.maximum(tid, 0)], -1)
    cnt = np.stack([((team_ft == 0) & present).sum(axis=1), ((team_ft == 1) & present).sum(axis=1)], axis=1)
    full = cnt.sum(axis=1) >= 16
    usual = float(np.median(cnt[full, 0] - cnt[full, 1])) if full.any() else 0.0   # home minus away (red cards)
    held_pt = np.zeros(present.shape, dtype=bool)         # the tracker repeats a lost player's place exactly
    if home_left is not None and (cfg.keeper_takeover or cfg.keeper_box_ignore_still):
        with np.errstate(invalid="ignore"):
            held_pt[1:] = np.nan_to_num(np.linalg.norm(np.diff(xy, axis=0), axis=-1), nan=np.inf) < cfg.frozen_ms / fps

    def note(q, msg):                                     # decision trail per tracklet (trackdiag --explain)
        if why is not None:
            why.setdefault(int(q), []).append(msg)

    def excess_of(q, lab, new, f, hidden=None):
        """Mean extra players q's label gives its team over the other, beyond the usual difference."""
        d = cnt[f, lab] - cnt[f, new]
        if hidden is not None:
            d = d + ((f >= hidden[0]) & (f <= hidden[1]))  # a copy counted for `new` that is no player
        return float(np.mean(d)) - (usual if lab == 0 else -usual)

    for q in list(pins):                                  # kickoff flips pass the same count test
        lab, new = int(tteam[q]), int(pins[q])
        f = np.arange(int(start[q]), int(end[q]) + 1)
        if excess_of(q, lab, new, f) < cfg.inherit_count_margin:
            del pins[q]                                   # an official or a lean, not a wrong-team player
            st["kickoff_counts_disagree"] += 1
            note(q, "kickoff: counts refused")
            continue
        tteam[q] = new
        cnt[f, lab] -= 1
        cnt[f, new] += 1

    def still_at(t):
        """(S,) tracks at frame t that do not move: frozen for ghost_min_s, or the same place repeated exactly
        (a held copy) on the last 3 frames."""
        out = present[t] & (frozen_run[t - 1] >= ghost_min) if t >= 1 else np.zeros(present.shape[1], bool)
        if t >= 3:
            out |= present[t] & held_pt[t - 1] & held_pt[t - 2] & held_pt[t - 3]
        return out

    def hide_still(q, s, js, team_of):
        """Hide the non-moving tracks js next to the new keeper q while they stay still (and q lives)."""
        n_hid = 0
        for j in js:
            c = int(tid[s, j])
            if c < 0 or c == q or s <= claimed_until[c]:
                continue
            last = min(int(end[c]), int(end[q]))
            seg = (np.nan_to_num(speed[s:last + 1, j], nan=np.inf) < cfg.frozen_ms) | held_pt[s:last + 1, j]
            stop = len(seg) if seg.all() else int(np.argmin(seg))
            if stop <= 0:
                continue
            ghosts.append((c, s, s + stop - 1))
            cnt[s:s + stop, int(tteam[c])] -= 1
            claimed_until[c] = int(end[q])
            n_hid += 1
        return n_hid

    def keeper_box(q, s, k, P):
        """Alone in a penalty area, deepest of all: the keeper of the team defending it. True = decided.
        v4g: tracks that do not move do not count (and are hidden), the team is the side's (half start)."""
        L, W = pitch
        if not (abs(P[1] - W / 2) < cfg.box_half_width_m and (P[0] < cfg.box_depth_m or P[0] > L - cfg.box_depth_m)):
            note(q, "box rule: starts outside a penalty area")
            return False
        left = P[0] < L / 2
        keeper_team = 0 if left == bool(home_left[s]) else 1
        others = present[s].copy()
        others[k] = False
        still = still_at(s) & others if cfg.keeper_box_ignore_still else np.zeros_like(others)
        reach = cfg.inherit_reach_m + cfg.jump_tol_m       # the held copies the pickup rule would have hidden
        near_still = [int(j) for j in np.flatnonzero(still) if np.linalg.norm(xy[s, j] - P) <= reach]
        others &= ~still                                  # only moving tracks are players here
        Q = xy[s][others]
        dq = P[0] if left else L - P[0]
        do = Q[:, 0] if left else L - Q[:, 0]
        on = (do > -cfg.off_pitch_m) & (Q[:, 1] > -cfg.off_pitch_m) & (Q[:, 1] < W + cfg.off_pitch_m)
        alone = not (np.linalg.norm(Q - P, axis=1) < cfg.keeper_alone_m).any()
        if not alone or (do[on] < dq).any():
            note(q, "box rule: " + ("someone moving within keeper_alone_m" if not alone else "someone moving is deeper"))
            return False
        f = np.arange(s, int(end[q]) + 1)
        seen = present[f, k]
        with np.errstate(invalid="ignore"):
            dqf = xy[f, k, 0] if left else L - xy[f, k, 0]
            inbox = (dqf < cfg.box_depth_m) & (np.abs(xy[f, k, 1] - W / 2) < cfg.box_half_width_m)
        stay = float(inbox[seen].mean()) if seen.any() else 0.0
        if stay < cfg.keeper_box_stay_share:
            note(q, f"box rule: leaves the box (in it on {stay:.0%} of its life, needs {cfg.keeper_box_stay_share:.0%})")
            return False
        if tteam[q] == keeper_team:
            st["keeper_box_pickups_same"] += 1
            hid = hide_still(q, s, near_still, keeper_team)
            st["keeper_box_still_hidden"] += hid
            note(q, "box rule: already the defending team (decided)" + (f"; {hid} still track(s) next to it hidden" if hid else ""))
            return True
        lab = int(tteam[q])
        if not cfg.keeper_box_count_check or excess_of(q, lab, keeper_team, f) >= cfg.inherit_count_margin:
            cnt[f, lab] -= 1                              # the keeper's team is the one short
            cnt[f, keeper_team] += 1
            tteam[q] = keeper_team
            st["keeper_box_pickups_changed"] += 1
            hid = hide_still(q, s, near_still, keeper_team)
            st["keeper_box_still_hidden"] += hid
            note(q, "box rule: CHANGED to the defending team (sides " +
                 ("from the half start" if sides_counted else "from the teams' medians") + ")" +
                 (f"; {hid} still track(s) next to it hidden" if hid else ""))
            return True
        st["keeper_box_counts_disagree"] += 1             # an attacker alone in the box: normal rules
        note(q, "box rule: counts refused")
        return False

    def pickup(q, s, k, P):
        """A pickup takes the team of the one track that just ended / stands held there. True = decided."""
        cands, saved = [], []                             # (tracklet, frames of established life, slot if a frozen copy)
        lo, hi = np.searchsorted(end_sorted, [s - memory, s])
        for c in by_end[lo:hi]:                           # ended in [s - memory, s - 1]
            if used[c]:
                continue
            g = s - int(end[c])
            d = np.linalg.norm(xy[end[c], slot[c]] - P)
            if g <= gap and d <= min(cfg.vmax_ms * g / fps, cfg.inherit_reach_m) + cfg.jump_tol_m:
                cands.append((int(c), int(end[c] - start[c] + 1), -1))
            elif d <= cfg.inherit_same_place_m:           # longer gone, but back right where he was last seen
                saved.append((int(c), int(end[c] - start[c] + 1), -1))
        for j in np.flatnonzero(present[s]):              # alive, but standing frozen: a held copy
            c = int(tid[s, j])
            if j == k or c < 0 or s <= claimed_until[c] or start[c] >= s:
                continue
            r = min(int(frozen_run[s - 1, j]), s - int(start[c]))
            if r < ghost_min:
                continue
            reach = min(cfg.vmax_ms * r / fps, cfg.inherit_reach_m) + cfg.jump_tol_m
            if np.linalg.norm(xy[s, j] - P) <= reach:
                cands.append((c, s - r - int(start[c]), int(j)))
        if not cands and len(saved) == 1:                 # nobody recent nearby: the saved last place
            cands = saved
            st["from_saved_place"] += 1
        if not cands:
            note(q, "pickup: no predecessor")
            return False
        if len(cands) > 1:
            st["ambiguous"] += 1
            note(q, "pickup: several predecessors")
            return False
        c, life, gj = cands[0]
        if life < min_life:
            st["short_predecessor"] += 1
            note(q, "pickup: predecessor too short")
            return False
        others = present[s].copy()
        others[k] = False
        if gj >= 0:
            others[gj] = False
        if others.any() and (np.linalg.norm(xy[s][others] - P, axis=1) < cfg.inherit_alone_m).any():
            st["crowded"] += 1
            note(q, "pickup: crowded")
            return False
        ghost = None
        if gj >= 0:                                       # the held copy, while it stays frozen AND its
            last = min(int(end[c]), int(end[q])) if cfg.ghost_reshow else int(end[c])
            seg = np.nan_to_num(speed[s:last + 1, gj], nan=np.inf) < cfg.frozen_ms
            stop = len(seg) if seg.all() else int(np.argmin(seg))
            if stop > 0:
                ghost = (s, s + stop)                      # [first, end) -- not a player, not counted
        if tteam[q] != tteam[c]:
            lab, new = int(tteam[q]), int(tteam[c])
            f = np.arange(s, int(end[q]) + 1)
            if excess_of(q, lab, new, f, None if ghost is None else (ghost[0], ghost[1] - 1)) < cfg.inherit_count_margin:
                st["counts_disagree"] += 1
                note(q, "pickup: counts refused")
                return False
            tteam[q] = new
            cnt[f, lab] -= 1
            cnt[f, new] += 1
            st["pickups_team_changed"] += 1
            note(q, "pickup: CHANGED to the predecessor's team" + (" (held copy)" if gj >= 0 else ""))
        else:
            st["pickups_same_team"] += 1
            note(q, "pickup: predecessor has the same team (decided)" + (" (held copy)" if gj >= 0 else ""))
        if gj >= 0:
            claimed_until[c] = int(end[q]) if cfg.ghost_reshow else int(end[c])
        else:
            used[c] = True
        if ghost is not None:
            cnt[ghost[0]:ghost[1], int(tteam[c])] -= 1
            ghosts.append((c, ghost[0], ghost[1] - 1))
        return True

    def keeper_behaviour(q, k, left, f, hidden, kc):
        """Share of q's frames f inside the penalty area at x=0 (left) or x=L and deeper than every other
        track (within keeper_behaviour_tol_m); the keeper's old track is left out only while it is a held copy
        (a live keeper deeper than q means q is not the keeper). None if too short."""
        if len(f) < max(1, int(round(cfg.keeper_behaviour_min_s * fps))):
            return None
        L, W = pitch
        X = xy[f]
        pr = present[f].copy()
        pr[:, k] = False
        if cfg.keeper_box_ignore_still:                   # tracks that do not move are no players (v4g)
            pr &= ~(frozen_run[np.maximum(f - 1, 0)] >= ghost_min)
        if hidden is not None and kc >= 0:                # his held copy (exact repeats) is no player; a live
            pr[(f >= hidden[0]) & (f <= hidden[1]) & held_pt[f, kc], kc] = False   # keeper still counts
        off = cfg.off_pitch_m
        with np.errstate(invalid="ignore"):
            depth = X[..., 0] if left else L - X[..., 0]
            on = pr & (depth > -off) & (X[..., 1] > -off) & (X[..., 1] < W + off)
            dq = depth[:, k]
            inbox = present[f, k] & (dq < cfg.box_depth_m) & (np.abs(X[:, k, 1] - W / 2) < cfg.box_half_width_m)
            deepest = dq <= np.where(on, depth, np.inf).min(axis=1) + cfg.keeper_behaviour_tol_m
        seen = present[f, k]
        return float((inbox & deepest)[seen].mean()) if seen.any() else None

    def takeover(q, s, k, P):
        """v4d (Sigmar): a new track next to the OTHER team's goalkeeper -- his held copy, a duplicate on
        him, or his saved last place -- is that keeper: it takes the keeper's team and the held copy is
        hidden while the two overlap. True = q's team changed."""
        L, W = pitch
        B = int(tteam[q])
        A = 1 - B
        left = bool(home_left[s]) == (A == 0)             # the keeper's team defends the goal at x = 0
        if (P[0] if left else L - P[0]) > cfg.box_depth_m + cfg.keeper_near_m or \
                abs(P[1] - W / 2) > cfg.box_half_width_m + cfg.keeper_near_m:
            note(q, "takeover: not near the other team's penalty area")
            return False
        f0 = max(0, s - hold)
        tb, X, off = tid[f0:s], xy[f0:s], cfg.off_pitch_m
        with np.errstate(invalid="ignore"):
            depth = X[..., 0] if left else L - X[..., 0]
            on = present[f0:s] & (home_left[f0:s] == home_left[s])[:, None] & (depth > -off) & \
                (X[..., 1] > -off) & (X[..., 1] < W + off)
            inbox = on & (tb >= 0) & (tteam[np.maximum(tb, 0)] == A) & (depth < cfg.box_depth_m) & \
                (np.abs(X[..., 1] - W / 2) < cfg.box_half_width_m)
        dA = np.where(inbox, depth, np.inf)
        j = dA.argmin(axis=1)
        rows = np.arange(len(tb))
        kl = np.flatnonzero(np.isfinite(dA[rows, j]) & (dA[rows, j] <= np.where(on, depth, np.inf).min(axis=1)))
        seen, tried = set(), []
        if not len(kl):
            note(q, "takeover: other team's keeper not seen (deepest in his box) in the last keeper_hold_s")
        for i in kl[::-1]:                                # keeper-like tracks of A (as keeper_fill), newest first
            c = int(tb[i, j[i]])
            if c in seen or c == q:
                continue
            seen.add(c)
            kc = int(slot[c])
            hidden = None
            if end[c] >= s:                               # still tracked: a held copy or a duplicate on him
                if s <= claimed_until[c]:
                    tried.append("keeper's held copy already claimed")
                    continue
                ref = xy[s, kc] if present[s, kc] else xy[f0 + i, kc]
                if np.linalg.norm(ref - P) > cfg.keeper_near_m:
                    tried.append(f"keeper tracked, {_dist_bin(np.linalg.norm(ref - P))} away")
                    continue
                f = np.arange(s, min(int(end[c]), int(end[q])) + 1)
                valid = present[f, kc] & present[f, k]
                with np.errstate(invalid="ignore"):
                    dd = np.linalg.norm(xy[f, kc] - xy[f, k], axis=1)
                held, dup = held_pt[f, kc], dd < cfg.dup_m
                if cfg.keeper_live_test == "held_or_duplicate":
                    if not (held | dup | ~valid).all():   # two people (an attacker on the keeper): leave it
                        st["keeper_takeover_apart"] += 1
                        tried.append("keeper tracked and live, they move apart (max " +
                                     (_dist_bin(float(np.nanmax(dd[valid]))) if valid.any() else "?") +
                                     f", held on {held[valid].mean() if valid.any() else 0:.0%} of shared frames)")
                        continue
                    stop = len(f)
                else:                                     # "near": appearing next to him is enough; his old
                    keep = held | (dd <= cfg.keeper_near_m) | ~valid   # track is hidden while held or near
                    stop = len(f) if keep.all() else int(np.argmin(keep))
                hidden = (s, s + stop - 1)                # stop >= 1: the two are near at s
                v = valid[:stop]
                kind = ("keeper_takeover_held" if v.any() and held[:stop][v].mean() >= 0.5 else
                        "keeper_takeover_duplicate" if v.any() and dup[:stop][v].mean() >= 0.5 else
                        "keeper_takeover_near")
            else:                                         # gone: his saved last place
                if used[c]:
                    tried.append("keeper gone, his track already continued by another")
                    continue
                if np.linalg.norm(xy[f0 + i, kc] - P) > cfg.keeper_near_m:
                    tried.append(f"keeper gone, saved place {_dist_bin(np.linalg.norm(xy[f0 + i, kc] - P))} away")
                    continue
                kind = "keeper_takeover_saved"
            f = np.arange(s, int(end[q]) + 1)
            if cfg.keeper_takeover_count_check and excess_of(q, B, A, f, hidden) < cfg.inherit_count_margin:
                ex = excess_of(q, B, A, f, hidden)
                share = keeper_behaviour(q, k, left, f, hidden, kc if hidden is not None else -1) \
                    if cfg.keeper_behaviour else None
                if share is None or share < cfg.keeper_behaviour_share:
                    st["keeper_takeover_counts_disagree"] += 1
                    note(q, f"takeover ({kind[16:]}): counts refused (excess {ex:+.1f} players, needs "
                            f"{cfg.inherit_count_margin:+.1f}) and it does not behave like the keeper ("
                            + ("lives too short to tell" if share is None else
                               f"in his box and deepest of all on {share:.0%} of its life, needs "
                               f"{cfg.keeper_behaviour_share:.0%}") + ")")
                    return False
                st["keeper_takeover_by_behaviour"] += 1     # the counts are ambiguous, the behaviour is not
                note(q, f"takeover ({kind[16:]}): counts ambiguous (excess {ex:+.1f}) but it behaves like the keeper "
                        f"(in his box and deepest of all on {share:.0%} of its life)")
            tteam[q] = A
            cnt[f, B] -= 1
            cnt[f, A] += 1
            if hidden is not None:
                cnt[hidden[0]:hidden[1] + 1, A] -= 1
                ghosts.append((c, hidden[0], hidden[1]))
                if end[c] <= hidden[1]:                   # his old track ended while the new one carried him
                    used[c] = True
                else:
                    claimed_until[c] = int(end[q])      # he shows again if the new track ends first
            else:
                used[c] = True
            taken.add(int(q))
            st[kind] += 1
            note(q, f"takeover ({kind[16:]}): CHANGED to the keeper's team")
            return True
        if len(kl):
            note(q, "takeover: " + "; ".join(dict.fromkeys(tried)) if tried else "takeover: no candidate")
        return False

    box_on = home_left is not None and cfg.keeper_box_pickup
    take_on = home_left is not None and cfg.keeper_takeover
    for q in np.argsort(start, kind="stable"):
        if q in pins:
            st["kickoff_pinned"] += 1
            continue
        s, k = int(start[q]), int(slot[q])
        if s == 0:
            continue
        P = xy[s, k]
        note(q, "new track as " + ("home" if tteam[q] == 0 else "away"))
        if box_on and keeper_box(q, s, k, P):
            continue
        if take_on and cfg.keeper_takeover_first and takeover(q, s, k, P):
            continue
        if pickup(q, s, k, P):
            continue
        if take_on and not cfg.keeper_takeover_first:     # only what the rules above left undecided
            takeover(q, s, k, P)
    return st, ghosts, taken


def _dist_bin(d: float) -> str:
    """A distance as a coarse bin (trackdiag --explain prints no positions)."""
    for hi, lab in ((1.0, "< 1 m"), (2.0, "1-2 m"), (3.0, "2-3 m"), (5.0, "3-5 m"), (10.0, "5-10 m")):
        if d < hi:
            return lab
    return "> 10 m"


def _episodes(tid, xy, vel, tteam, cfg: CleanConfig, fps: float):
    """Duplicate contacts between tracklets of the same team (per tracklet, after the repair):
    [(tracklet_a, tracklet_b, first_frame, last_frame)] inclusive."""
    min_run = max(1, int(round(cfg.dup_min_s * fps)))
    out = []
    has = tid >= 0
    S = tid.shape[1]
    for i in range(S):
        for j in range(i + 1, S):
            idx = np.flatnonzero(has[:, i] & has[:, j])
            if len(idx) < min_run:
                continue
            idx = idx[tteam[tid[idx, i]] == tteam[tid[idx, j]]]
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


# what happened to each output point (clean_with_flags) -- the viewer marks them
FLAG_NONE, FLAG_TEAM, FLAG_TAKEOVER, FLAG_KEEPER_FILL = 0, 1, 2, 3


def clean(trk: Tracking, fps: float, pitch=None, cfg: CleanConfig = CleanConfig(),
          kickoff_windows=None, halves=None) -> tuple[Tracking, dict]:
    """pitch (L, W) in metres enables the off-pitch test of the cap and the kickoff check.
    kickoff_windows: [(first, last)] frames of tagged kickoffs (smart_tagging.kickoff_windows), used
    when the ball is not seen on the centre spot."""
    out, report, _ = clean_with_flags(trk, fps, pitch, cfg, kickoff_windows, halves=halves)
    return out, report


def clean_with_flags(trk: Tracking, fps: float, pitch=None, cfg: CleanConfig = CleanConfig(),
                     kickoff_windows=None, trace: dict | None = None, halves=None) -> tuple[Tracking, dict, np.ndarray]:
    """clean() plus (n, lanes) uint8 flags per output point: FLAG_NONE as delivered (team of its
    slot), FLAG_TEAM team repaired (kickoff, pickup, box rule), FLAG_TAKEOVER team given by the keeper
    takeover, FLAG_KEEPER_FILL the keeper drawn at his saved place (not tracked)."""
    xy = trk.players.astype(np.float64)
    n, S, _ = xy.shape
    team = np.asarray(trk.team_of_slot)
    present_in = ~np.isnan(xy[..., 0])

    # 1 cut at jumps and where the speed limit breaks over a window
    tr, cuts, fast_cuts, fast_pts = _tracklets(present_in, xy, fps, cfg)
    present = present_in.copy()
    for k, a, b in fast_pts:
        present[a:b + 1, k] = False
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
    frozen_run = _slow_run(speed, cfg.frozen_ms)

    # 2 team per tracklet: the slot's, repaired at kickoffs and carried across pickups
    tteam = team[slot].astype(np.int8)
    team_in = tteam.copy()
    hidden = [[] for _ in range(T)]
    pins, kick_frames, kick_details = {}, [], []
    if cfg.kickoff_check and pitch is not None and T:
        pins, kick_frames, kick_details = _kickoff_pins(xy, tid, tteam, speed, trk.ball, fps, pitch, cfg,
                                                        kickoff_windows)
    home_left, sides_by_half = None, []
    if pitch is not None and T and (cfg.keeper_box_pickup or cfg.keeper_fill or cfg.keeper_takeover):
        team_ft0 = np.where(tid >= 0, tteam[np.maximum(tid, 0)], -1)
        home_left = _home_defends_left(xy, team_ft0, fps, cfg)
        if cfg.side_from_half_start and halves:
            home_left, sides_by_half = _sides_from_half_start(xy, team_ft0, halves, pitch, fps, cfg, home_left)
    inherit, taken = None, set()
    if cfg.inherit_team and T:
        why = {} if trace is not None else None
        inherit, ghosts, taken = _inherit(tid, start, end, slot, xy, present, speed, frozen_run, tteam, pins, fps,
                                          cfg, home_left if (cfg.keeper_box_pickup or cfg.keeper_takeover) else None,
                                          pitch, why, sides_counted=bool(sides_by_half))
        if trace is not None:
            trace["why"] = why
        for c, a, b in ghosts:
            hidden[c].append((a, b))
    else:
        for q, tm in pins.items():
            tteam[q] = tm
    kick_details = [{k: v for k, v in d.items() if k != "_q"} for d in kick_details if d["_q"] in pins]
    team_ft = np.where(tid >= 0, tteam[np.maximum(tid, 0)] if T else 0, -1)   # (n, S) team per point, -1 none
    visible = present.copy()
    for q in range(T):
        for s, t in hidden[q]:
            visible[s:t + 1, slot[q]] = False
    ghost_hidden = int(present.sum() - visible.sum())

    # 3 dedup: hide the older while the pair stays together
    pred = np.full(T, -1, dtype=np.int64)
    handovers = newer_first = 0
    eps = _episodes(tid, xy, vel, tteam, cfg, fps)
    for a, b, t0, t1 in eps:
        old, new = (a, b) if (start[a], a) < (start[b], b) else (b, a)
        hidden[old].append((t0, t1))
        if t1 >= end[old] and end[new] > end[old]:
            handovers += 1                                   # the old track died inside the contact
            if pred[new] < 0:
                pred[new] = old
        elif t1 >= end[new]:
            newer_first += 1
    before_dedup = int(visible.sum())
    for q in range(T):
        for s, t in hidden[q]:
            visible[s:t + 1, slot[q]] = False
    dup_hidden = before_dedup - int(visible.sum())

    # 4 cap at max_per_team: hide the least player-like track
    over_before = {name: float(np.mean((present_in & (team == tm)).sum(axis=1) > cfg.max_per_team))
                   for name, tm in (("home", 0), ("away", 1))}
    frozen_frames = int(round(cfg.frozen_s * fps))
    reasons = {name: {"off_pitch": 0, "held": 0, "frozen": 0, "next_to_mate": 0, "youngest": 0} for name in ("home", "away")}
    cap_frames = {"home": 0, "away": 0}
    other_le_10 = {"home": 0, "away": 0}           # the other team short of a player: a mislabelled opponent?
    nearest_ball = {"home": [0, 0], "away": [0, 0]}  # [drops with the ball seen, of which the team's nearest to it]
    for name, tm in (("home", 0), ("away", 1)):
        cnt = (visible & (team_ft == tm)).sum(axis=1)
        for t in np.flatnonzero(cnt > cfg.max_per_team):
            cap_frames[name] += 1
            other_le_10[name] += int((visible[t] & (team_ft[t] == 1 - tm)).sum() <= cfg.max_per_team - 1)
            live = list(np.flatnonzero(visible[t] & (team_ft[t] == tm)))
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
                held = (frozen_run[t, live] >= frozen_frames) if cfg.cap_held_first else np.zeros(len(live), bool)
                if (out_m > cfg.off_pitch_m).any():
                    k, why = int(np.argmax(out_m)), "off_pitch"
                elif held.any():                             # a held copy: the saved place, not a 12th player
                    k, why = int(np.flatnonzero(held)[np.argmax(frozen_run[t, live][held])]), "held"
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

    # 5 lanes + speeds: visible pieces of each tracklet (hidden intervals split it)
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
    pieces.sort(key=lambda p: (int(tteam[p[2]]), p[0], p[2]))

    gap = int(round(cfg.lane_gap_s * fps)) + 1
    lanes = {0: [], 1: []}                                   # per team: [last_end, last_tracklet]
    lane_of_piece = []
    for a, b, q in pieces:
        tm = int(tteam[q])
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
    out_fl = np.zeros((n, n_home + n_away), dtype=np.uint8)
    out_tid = np.full((n, n_home + n_away), -1, dtype=np.int32) if trace is not None else None
    for (a, b, q), (tm, li) in zip(pieces, lane_of_piece):
        col = li if tm == 0 else n_home + li
        seg = xy[a:b + 1, slot[q]]
        vis = visible[a:b + 1, slot[q]]
        sp = np.linalg.norm(_velocity(seg, w, fps), axis=1)
        out_xy[a:b + 1, col] = np.where(vis[:, None], seg, np.nan)
        out_sp[a:b + 1, col] = np.where(vis, sp, np.nan)
        code = FLAG_TAKEOVER if q in taken else FLAG_TEAM if tteam[q] != team_in[q] else FLAG_NONE
        if code:
            out_fl[a:b + 1, col] = np.where(vis, code, FLAG_NONE)
        if out_tid is not None:
            out_tid[a:b + 1, col] = np.where(vis, q, -1)
    team_out = np.r_[np.zeros(n_home, np.int8), np.ones(n_away, np.int8)]
    after = ~np.isnan(out_xy[..., 0])
    any_frame = present_in.any(axis=1)
    changed = tteam != team_in
    life = end - start + 1
    report = {
        "tracklets_in": int((np.diff(present_in.astype(np.int8), axis=0, prepend=0) == 1).sum()),
        "cuts_at_jumps": int(cuts),
        "fast_window_cuts": int(fast_cuts), "fast_points_hidden": int(present_in.sum() - present.sum()),
        "kickoff_windows_given": len(kickoff_windows or ()), "kickoffs_found": len(kick_frames),
        "kickoff_team_flips": len(pins), "kickoff_flips_detail": kick_details,
        "team_inherit": inherit,
        "tracklets_team_changed": int(changed.sum()), "tracklets_keeper_takeover": len(taken),
        "home_defends_left_by_half": sides_by_half,
        "points_team_changed_share": round(float(life[changed].sum() / max(life.sum(), 1)), 4),
        "frozen_copies_hidden_points": ghost_hidden,
        "duplicate_episodes": len(eps), "duplicate_points_hidden": dup_hidden,
        "episodes_handover_old_died": int(handovers), "episodes_newer_died_first": int(newer_first),
        "cap_frames": cap_frames, "cap_points_hidden": int(sum(sum(r.values()) for r in reasons.values())), "cap_reasons": reasons,
        "cap_frames_other_team_le_10": {k: round(v / max(cap_frames[k], 1), 3) for k, v in other_le_10.items()},
        "cap_drop_was_team_nearest_to_ball": {k: round(v[1] / v[0], 3) if v[0] else None for k, v in nearest_ball.items()},
        "share_frames_over_11": {"before": over_before,
                                 "after": {name: float(np.mean((after & (team_out == tm)).sum(axis=1) > cfg.max_per_team))
                                           for name, tm in (("home", 0), ("away", 1))}},
        "players_per_frame_mean": {"before": round(float(present_in[any_frame].sum(axis=1).mean()), 2) if any_frame.any() else None,
                                   "after": round(float(after[any_frame].sum(axis=1).mean()), 2) if any_frame.any() else None},
        "points_kept_share": round(float(after.sum() / max(present_in.sum(), 1)), 4),
        "lanes": {"home": n_home, "away": n_away},
    }
    fill = {"home": 0, "away": 0}
    if cfg.keeper_fill and home_left is not None:
        out_xy, out_sp, team_out, fill = _keeper_fill(out_xy, out_sp, team_out, home_left, pitch, cfg, fps)
        kf = lambda col: np.where(np.isnan(out_xy[:, col, 0]), FLAG_NONE, FLAG_KEEPER_FILL).astype(np.uint8)[:, None]
        out_fl = np.concatenate([out_fl[:, :n_home], kf(n_home), out_fl[:, n_home:], kf(out_xy.shape[1] - 1)], axis=1)
        if out_tid is not None:
            m1 = np.full((n, 1), -2, dtype=np.int32)                  # -2: the keeper drawn at his saved place
            out_tid = np.concatenate([out_tid[:, :n_home], m1, out_tid[:, n_home:], m1], axis=1)
    report["keeper_fill_points"] = fill
    report["keeper_fill_share_of_play"] = {k: round(v / max(int((after.sum(axis=1) >= cfg.keeper_min_tracked).sum()), 1), 4)
                                           for k, v in fill.items()}
    if trace is not None:
        trace.update(start=start, end=end, slot=slot, team_in=team_in, team_out=tteam.copy(), home_left=home_left,
                     out_tid=out_tid)
        trace.setdefault("why", {})
    return Tracking(players=out_xy, player_speed=out_sp, ball=trk.ball, team_of_slot=team_out), report, out_fl


def _keeper_fill(out_xy, out_sp, team_out, home_left, pitch, cfg: CleanConfig, fps: float = 25.0):
    """One lane more per team: its keeper at his saved last place while the team is short of 11 and
    nobody of its own is in its penalty area (play on). Saved place = where the deepest of ALL players
    towards that goal, inside its penalty area and of that team, was last seen since the last change
    of ends, held for at most keeper_hold_s; never seen yet in this half: the 6-yard line."""
    import pandas as pd
    L, W = pitch
    n = out_xy.shape[0]
    x, y = out_xy[..., 0], out_xy[..., 1]
    vis = ~np.isnan(x)
    play = vis.sum(axis=1) >= cfg.keeper_min_tracked
    period = np.r_[0, np.cumsum(np.diff(home_left.astype(np.int8)) != 0)]
    hold = int(round(cfg.keeper_hold_s * fps)) if cfg.keeper_hold_s > 0 else n + 1
    lanes, counts = [], {}
    idx = np.arange(n)
    for tm, name in ((0, "home"), (1, "away")):
        cols = np.flatnonzero(team_out == tm)
        left = home_left if tm == 0 else ~home_left
        gx = np.where(left, 0.0, L)[:, None]
        with np.errstate(invalid="ignore"):
            depth_all = np.where(vis, np.abs(x - gx), np.inf)
            inbox = vis[:, cols] & (depth_all[:, cols] < cfg.box_depth_m) & (np.abs(y[:, cols] - W / 2) < cfg.box_half_width_m)
        has = inbox.any(axis=1)
        kp = np.full((n, 2), np.nan)
        if len(cols):
            j = np.where(inbox, depth_all[:, cols], np.inf).argmin(axis=1)
            keeperlike = has & (depth_all[idx, cols[j]] <= depth_all.min(axis=1))   # deepest of everybody
            kp[keeperlike] = out_xy[np.flatnonzero(keeperlike), cols[j[keeperlike]]]
        seen = np.where(~np.isnan(kp[:, 0]), idx, -1)
        last = pd.Series(np.where(seen >= 0, seen, np.nan)).groupby(period).ffill().to_numpy()
        kp = pd.DataFrame(kp).groupby(period).ffill().to_numpy()          # the saved place, per half (side)
        stale = ~np.isnan(last) & (idx - np.nan_to_num(last) > hold)
        spot = np.stack([np.where(left, cfg.keeper_spot_m, L - cfg.keeper_spot_m), np.full(n, W / 2)], axis=1)
        never = np.isnan(kp[:, 0])
        kp = np.where(never[:, None], spot, kp)
        need = play & ~has & (vis[:, cols].sum(axis=1) < cfg.max_per_team) & ~stale
        lanes.append(np.where(need[:, None], kp, np.nan).astype(np.float32))
        counts[name] = int(need.sum())
    n_home = int((team_out == 0).sum())
    nan = np.full((n, 1), np.nan, dtype=np.float32)
    out_xy = np.concatenate([out_xy[:, :n_home], lanes[0][:, None], out_xy[:, n_home:], lanes[1][:, None]], axis=1)
    out_sp = np.concatenate([out_sp[:, :n_home], nan, out_sp[:, n_home:], nan], axis=1)   # held: no speed
    team_out = np.r_[np.zeros(n_home + 1, np.int8), np.ones(len(team_out) - n_home + 1, np.int8)]
    return out_xy, out_sp, team_out, counts
