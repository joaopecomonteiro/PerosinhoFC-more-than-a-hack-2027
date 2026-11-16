"""Per-frame inputs for "is the ball dead?" -- the same code for every source and for our match.

players  mqf.dead.features (João): speeds, slow / fast shares, team speeds and spreads, distance
         between the team centres, players in the boxes / near the touchlines / off the pitch,
         smoothed over 1-30 s with trend and rank within the half
ball     tracked or not; the whole gap a frame sits in and the time since / until the ball is seen;
         whether the gap starts or ends at a RESTART SPOT (touchline, goal line incl. corners, goal
         area, centre spot -- Laws of the Game); ball off the pitch; distance to the nearest line;
         speed; still; share of frames with a ball over 1 / 3 / 10 s
edges    everyone slowing down or speeding up together: mean player speed over the next 0.5 / 1 / 2 s
         minus over the last 0.5 / 1 / 2 s (stoppages start and end with these)
What we learned on our match (aggregates, mqf.profile): the tracker loses the ball for long
stretches (150 gaps, median 10.7 s, 85 % of missing frames in gaps > 10 s) and these line up with
stoppages -- so the ball's ABSENCE is evidence too, but how strong is left to the model, trained
under randomised ball loss (model.py), never fitted to our match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import dead

GAP_CAP_S = 120.0


def restart_spot(x, y, pitch) -> np.ndarray:
    """On or beyond a touchline / goal line, in a goal area, on the centre spot. NaN -> False."""
    L, W = pitch
    with np.errstate(invalid="ignore"):
        return ((y < 2.0) | (y > W - 2.0) | (x < 1.5) | (x > L - 1.5)
                | (((x < 6.5) | (x > L - 6.5)) & (np.abs(y - W / 2) < 10.5))
                | (np.hypot(x - L / 2, y - W / 2) < 1.5))


def _roll(a, w: int, fn: str = "mean") -> np.ndarray:
    return getattr(pd.Series(a).rolling(max(1, w), center=True, min_periods=1), fn)().to_numpy()


def ball(b: np.ndarray, fps: float, pitch) -> pd.DataFrame:
    L, W = pitch
    n = len(b)
    bx, by = b[:, 0].astype(np.float64), b[:, 1].astype(np.float64)
    seen = ~np.isnan(bx)
    with np.errstate(invalid="ignore"):
        dline = np.minimum.reduce([bx, L - bx, by, W - by])          # < 0 outside the pitch
    w = max(1, int(round(0.2 * fps)))
    sp = np.full(n, np.nan)
    if n > 2 * w:
        sp[w:-w] = np.hypot(bx[2 * w:] - bx[:-2 * w], by[2 * w:] - by[:-2 * w]) * fps / (2 * w)
    spot = restart_spot(bx, by, pitch)

    gap = np.zeros(n); since = np.zeros(n); until = np.zeros(n)
    s_spot = np.zeros(n); e_spot = np.zeros(n)
    s_line = np.full(n, np.nan); e_line = np.full(n, np.nan)
    if (~seen).any():
        d = np.diff(np.r_[0, (~seen).astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            k = np.arange(e - s)
            gap[s:e] = (e - s) / fps
            since[s:e] = (k + 1) / fps
            until[s:e] = (e - s - k) / fps
            if s > 0:
                s_spot[s:e], s_line[s:e] = spot[s - 1], dline[s - 1]
            if e < n:
                e_spot[s:e], e_line[s:e] = spot[e], dline[e]
    off = (seen & (dline < -0.3)).astype(np.float64)
    still = (seen & (sp < 1.0)).astype(np.float64)
    f = {"b_seen": seen.astype(np.float64),
         "b_gap_s": np.minimum(gap, GAP_CAP_S), "b_since_s": np.minimum(since, GAP_CAP_S),
         "b_until_s": np.minimum(until, GAP_CAP_S),
         "b_gap_start_spot": s_spot, "b_gap_end_spot": e_spot,
         "b_gap_start_dline": s_line, "b_gap_end_dline": e_line,
         "b_dline": dline, "b_off": off, "b_speed": sp, "b_still": still,
         "b_at_spot": (seen & spot).astype(np.float64)}
    for sec in (1, 3, 10):
        f[f"b_seen_{sec}s"] = _roll(seen.astype(np.float64), int(round(sec * fps)))
    f["b_off_3s"] = _roll(off, int(round(3 * fps)))
    f["b_still_3s"] = _roll(still, int(round(3 * fps)))
    f["b_speed_1s"] = _roll(sp, int(round(fps)), "median")
    return pd.DataFrame(f)


def edges(speed_mean: np.ndarray, fps: float) -> pd.DataFrame:
    s = pd.Series(speed_mean)
    out = {}
    for sec in (0.5, 1.0, 2.0):
        w = max(1, int(round(sec * fps)))
        past = s.rolling(w, min_periods=1).mean()
        fut = s[::-1].rolling(w, min_periods=1).mean()[::-1]
        out[f"e_dspeed_{sec:g}s"] = (fut - past).to_numpy()
    return pd.DataFrame(out)


def _since(event: np.ndarray, fps: float, cap: float = 60.0) -> np.ndarray:
    """Seconds since the last frame where `event` was True (cap if none yet)."""
    n = len(event)
    last = np.maximum.accumulate(np.where(event, np.arange(n), -1))
    return np.where(last < 0, cap, np.minimum((np.arange(n) - last) / fps, cap))


def _run_s(flag: np.ndarray, fps: float, cap: float = 120.0) -> np.ndarray:
    """Seconds the current run of `flag` has lasted (0 where False)."""
    n = len(flag)
    start = np.maximum.accumulate(np.where(~flag, np.arange(n), -1))
    return np.where(flag, np.minimum((np.arange(n) - start) / fps, cap), 0.0)


def state(speed_mean: np.ndarray, share_slow: np.ndarray, seen: np.ndarray, fps: float) -> pd.DataFrame:
    """A stoppage is a STATE, not only a transition: how long since everyone braked, until everyone
    speeds up again, speed now vs the play before it, how long players have been slow without a ball."""
    s = pd.Series(speed_mean)
    w1 = max(1, int(round(fps)))
    past1 = s.rolling(w1, min_periods=1).mean()
    fut1 = s[::-1].rolling(w1, min_periods=1).mean()[::-1]
    d1 = (fut1 - past1).to_numpy()
    brake, go = d1 < -0.8, d1 > 0.8
    past20 = s.shift(int(round(3 * fps))).rolling(int(round(20 * fps)), min_periods=1).mean().to_numpy()
    now3 = s.rolling(max(1, int(round(3 * fps))), center=True, min_periods=1).mean().to_numpy()
    slow_nob = (~seen) & (np.nan_to_num(share_slow) > 0.3)
    return pd.DataFrame({
        "s_since_brake": _since(brake, fps),
        "s_until_go": _since(go[::-1], fps)[::-1],
        "s_speed_past20": past20,
        "s_speed_rel": now3 / np.maximum(past20, 0.3),
        "s_slow_nob_s": _run_s(slow_nob, fps),
        "s_slow_nob_share_10s": _roll(slow_nob.astype(np.float64), int(round(10 * fps))),
    })


def setpiece(players: np.ndarray, team: np.ndarray, b: np.ndarray, fps: float) -> pd.DataFrame:
    """A placed / held ball anywhere (free kicks, incl. offsides and fouls): how long it has lain still,
    one side near it and the other kept 9.15 m away (Law 13), a ring of players at wall distance, the
    taker next to it. NaN where there is no ball -- never "nobody near"."""
    n = len(b)
    seen = ~np.isnan(b[:, 0])
    w = max(1, int(round(0.2 * fps)))
    sp = np.full(n, np.nan)
    if n > 2 * w:
        sp[w:-w] = np.hypot(b[2 * w:, 0] - b[:-2 * w, 0], b[2 * w:, 1] - b[:-2 * w, 1]) * fps / (2 * w)
    still = seen & (sp < 1.0)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(players.astype(np.float64) - b[:, None, :].astype(np.float64), axis=-1)  # (n, P)
    valid = ~np.isnan(d)
    in9 = valid & (d < 9.15)
    home = team == 0
    n9h = in9[:, home].sum(axis=1).astype(np.float64)
    n9a = in9[:, ~home].sum(axis=1).astype(np.float64)
    ring = (valid & (d >= 8.5) & (d < 11.0)).sum(axis=1).astype(np.float64)
    near = np.where(valid.any(axis=1), np.nanmin(np.where(valid, d, np.inf), axis=1), np.nan)
    nan_if_no_ball = lambda a: np.where(seen, a, np.nan)
    out = {"fk_still_s": nan_if_no_ball(_run_s(still, fps)),
           "fk_n9_min": nan_if_no_ball(np.minimum(n9h, n9a)), "fk_n9_max": nan_if_no_ball(np.maximum(n9h, n9a)),
           "fk_ring": nan_if_no_ball(ring), "fk_near": nan_if_no_ball(near)}
    for k in ("fk_n9_min", "fk_ring"):
        out[f"{k}_2s"] = pd.Series(out[k]).rolling(max(1, int(round(2 * fps))), center=True, min_periods=1).mean().to_numpy()
    return pd.DataFrame(out)


def memory(b: np.ndarray, players: np.ndarray, share_slow: np.ndarray, fps: float, pitch) -> pd.DataFrame:
    """v4. A stoppage is remembered until play restarts, and a fetched ball is not a dribble.

    m_since_out_s   seconds since the ball was last seen off the pitch or vanished at a line (cap 60)
    m_kick_since    a restart-like kick happened since then (ball under 2 m/s, then over 8 m/s within 0.4 s)
    m_out_open_s    m_since_out_s while no kick has happened since -- 0 once play restarted
    c_carry_slow    ball within 1 m of a player, moving at 0.5-3 m/s, while at least half the players are slow
    c_carry_slow_3s / _10s  its share over 3 / 10 s
    """
    L, W = pitch
    n = len(b)
    bx, by = b[:, 0].astype(np.float64), b[:, 1].astype(np.float64)
    seen = ~np.isnan(bx)
    with np.errstate(invalid="ignore"):
        dline = np.minimum.reduce([bx, L - bx, by, W - by])
    w = max(1, int(round(0.2 * fps)))
    sp = np.full(n, np.nan)
    if n > 2 * w:
        sp[w:-w] = np.hypot(bx[2 * w:] - bx[:-2 * w], by[2 * w:] - by[:-2 * w]) * fps / (2 * w)
    off = seen & (dline < -0.3)
    vanish_at_line = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(np.diff(np.r_[1, seen.astype(np.int8)]) == -1)     # first missing frame of a gap
    for s in starts:
        if s > 0 and seen[s - 1] and dline[s - 1] < 2.0:
            vanish_at_line[s] = True
    out_event = off | vanish_at_line
    k4 = max(1, int(round(0.4 * fps)))
    slow_before = np.r_[np.full(k4, False), (sp[:-k4] < 2.0)] if n > k4 else np.zeros(n, dtype=bool)
    kick = seen & (sp > 8.0) & slow_before
    since_out = _since(out_event, fps)
    # a kick after the last out-event: compare "time since kick" with "time since out"
    since_kick = _since(kick, fps)
    kicked = since_kick < since_out
    open_s = np.where(kicked | (since_out >= 60.0), 0.0, since_out)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(players.astype(np.float64) - b[:, None, :].astype(np.float64), axis=-1)
    near1 = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1) < 1.0
    carry = seen & near1 & (sp >= 0.5) & (sp <= 3.0) & (np.nan_to_num(share_slow) >= 0.5)
    return pd.DataFrame({
        "m_since_out_s": since_out, "m_kick_since": kicked.astype(np.float64), "m_out_open_s": open_s,
        "c_carry_slow": carry.astype(np.float64),
        "c_carry_slow_3s": _roll(carry.astype(np.float64), int(round(3 * fps))),
        "c_carry_slow_10s": _roll(carry.astype(np.float64), int(round(10 * fps))),
    })


def _bridge(flag: np.ndarray, max_gap: int) -> np.ndarray:
    """flag with False dips of at most max_gap frames filled in (a run survives a brief dip)."""
    out = flag.copy()
    d = np.diff(np.r_[1, flag.astype(np.int8), 1])
    for s, e in zip(np.flatnonzero(d == -1), np.flatnonzero(d == 1)):     # False runs [s, e)
        if 0 < s and e < len(flag) and e - s <= max_gap:
            out[s:e] = True
    return out


def stopped_v6(b: np.ndarray, players: np.ndarray, speed_mean: np.ndarray, share_slow: np.ndarray, fps: float,
               pitch) -> pd.DataFrame:
    """v6 (team, from the CSV and the video): the whistle stops the players, not the ball.

    u_unplayed_s     seconds since everyone braked sharply while the ball kept its path untouched (no change of
                     direction over 35 degrees, no jump of speed) -- the ball rolling on after the whistle; 0 once
                     someone plays it again
    m2_open_s        "the ball went out / vanished at a line" memory that closes only on a REAL restart: the
                     ball starting from slow / still / held / reappearing TOGETHER with the players speeding up.
                     A kick without the team speeding up is the ball being returned and does not close it.
                     Timeout: open > 20 s while play has looked clearly alive for 5 s
    m2_returns       touches (kicks, passes back) since the ball went out while the memory is open
    s2_team_stopped_s seconds the team has been mostly slow (share under 1 m/s averaged over 1 s > 0.4, dips under
                     1 s ignored) -- no ball needed
    s2_slow_nob_s    the same while the ball is missing (smoothed s_slow_nob_s)
    """
    L, W = pitch
    n = len(b)
    bx, by = b[:, 0].astype(np.float64), b[:, 1].astype(np.float64)
    seen = ~np.isnan(bx)
    w = max(1, int(round(0.2 * fps)))
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    if n > 2 * w:
        vx[w:-w] = (bx[2 * w:] - bx[:-2 * w]) * fps / (2 * w)
        vy[w:-w] = (by[2 * w:] - by[:-2 * w]) * fps / (2 * w)
    sp = np.hypot(vx, vy)
    k4 = max(1, int(round(0.4 * fps)))
    vx0, vy0, sp0 = np.roll(vx, k4), np.roll(vy, k4), np.roll(sp, k4)
    vx0[:k4] = vy0[:k4] = sp0[:k4] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (vx * vx0 + vy * vy0) / (sp * sp0)
    touch = seen & (((sp > 1.5) & (sp0 > 1.5) & (cosang < np.cos(np.radians(35)))) | (sp - sp0 > 3.0))
    touch = np.nan_to_num(touch, nan=0).astype(bool)

    s = pd.Series(speed_mean)
    w1 = max(1, int(round(fps)))
    d1 = (s[::-1].rolling(w1, min_periods=1).mean()[::-1] - s.rolling(w1, min_periods=1).mean()).to_numpy()
    brake, go = d1 < -0.8, d1 > 0.6
    since_brake, since_touch = _since(brake, fps, 30.0), _since(touch, fps, 30.0)
    unplayed = np.where(seen & (since_touch > since_brake) & (since_brake < 30.0), since_brake, 0.0)

    with np.errstate(invalid="ignore"):
        dline = np.minimum.reduce([bx, L - bx, by, W - by])
    out_event = seen & (dline < -0.3)
    starts = np.flatnonzero(np.diff(np.r_[1, seen.astype(np.int8)]) == -1)
    for g in starts:
        if g > 0 and seen[g - 1] and dline[g - 1] < 2.0:
            out_event[g] = True
    appear = seen & ~np.roll(seen, k4)
    ball_start = seen & (((sp > 4.0) & (sp0 < 2.0)) | appear)
    go_near = pd.Series(go.astype(np.float64)).rolling(int(4 * fps) + 1, center=True, min_periods=1).max().to_numpy() > 0
    restart = ball_start & go_near
    slow1 = pd.Series(np.nan_to_num(share_slow)).rolling(w1, center=True, min_periods=1).mean().to_numpy()
    sp3 = s.rolling(int(3 * fps), center=True, min_periods=1).mean().to_numpy()
    alive_look = (np.nan_to_num(sp3) > 2.2) & (slow1 < 0.25)
    alive_run = _run_s(alive_look, fps)

    open_s = np.zeros(n); returns = np.zeros(n)
    t_open, nret = -1.0, 0
    for i in range(n):                                    # sequential: open / close / timeout
        if out_event[i]:
            t_open, nret = 0.0, 0
        elif t_open >= 0:
            t_open += 1.0 / fps
            if restart[i] or (t_open > 20.0 and alive_run[i] >= 5.0):
                t_open = -1.0
            elif touch[i]:
                nret += 1
        open_s[i] = min(t_open, 120.0) if t_open >= 0 else 0.0
        returns[i] = nret if t_open >= 0 else 0
    stopped = _bridge(slow1 > 0.4, int(fps))
    slow_nob = _bridge((~seen) & (slow1 > 0.3), int(fps))
    return pd.DataFrame({"u_unplayed_s": unplayed, "m2_open_s": open_s, "m2_returns": returns,
                         "s2_team_stopped_s": _run_s(stopped, fps), "s2_slow_nob_s": _run_s(slow_nob, fps),
                         "s2_slow_1s": slow1})


def stopped_v7(b: np.ndarray, speed_mean: np.ndarray, share_slow: np.ndarray, fps: float, pitch) -> pd.DataFrame:
    """v7 "CR7" (team, from the v6 CSV): stoppages end at the RESTART, not when speeds drift back up.

    restart          a clear kick (ball from under 2 m/s to over 6 m/s within 0.4 s) or the ball reappearing,
                     with the players starting to speed up within 2 s (mean speed next 1 s minus last 1 s > 0.4)
    m3_open_s        "went out" memory. Opens ONLY on a real out: the ball seen beyond a line, or last seen within
                     1 m of a line moving outward. Closes on a restart; after a 10 s timeout it closes as soon as
                     play has looked clearly alive for 3 s (players fast, few slow); unconditionally after 120 s
    m3_returns       touches while the memory is open AND the whole team is stopped (the ball being returned)
    s3_team_stopped_s seconds the team has been mostly slow (1 s share > 0.4, dips < 1 s bridged); switched OFF at a
                     restart and kept off until the players have actually sped up and stopped again
    s3_slow_nob_s    the same, only while the ball is missing
    r_since_restart_s / r_until_restart_s   seconds since the last / until the next restart (offline: both known)
    """
    L, W = pitch
    n = len(b)
    bx, by = b[:, 0].astype(np.float64), b[:, 1].astype(np.float64)
    seen = ~np.isnan(bx)
    w = max(1, int(round(0.2 * fps)))
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    if n > 2 * w:
        vx[w:-w] = (bx[2 * w:] - bx[:-2 * w]) * fps / (2 * w)
        vy[w:-w] = (by[2 * w:] - by[:-2 * w]) * fps / (2 * w)
    sp = np.hypot(vx, vy)
    k4 = max(1, int(round(0.4 * fps)))
    sp0 = np.roll(sp, k4); sp0[:k4] = np.nan
    vx0, vy0 = np.roll(vx, k4), np.roll(vy, k4)
    vx0[:k4] = vy0[:k4] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (vx * vx0 + vy * vy0) / (sp * sp0)
        touch = seen & (((sp > 1.5) & (sp0 > 1.5) & (cosang < np.cos(np.radians(35)))) | (sp - sp0 > 3.0))
        dline = np.minimum.reduce([bx, L - bx, by, W - by])
    touch = np.nan_to_num(touch, nan=0).astype(bool)

    # a REAL out: seen beyond a line, or vanishing within 1 m of a line while moving outward
    out_event = seen & (dline < -0.3)
    for g in np.flatnonzero(np.diff(np.r_[1, seen.astype(np.int8)]) == -1):
        f = g - 1
        f0 = f - 2 * w                                    # direction over the last 0.4 s BEFORE it vanished
        if f0 < 0 or not seen[f] or not seen[f0] or not (dline[f] < 1.0):
            continue
        ux, uy = (bx[f] - bx[f0]) * fps / (2 * w), (by[f] - by[f0]) * fps / (2 * w)
        x, y = bx[f], by[f]
        outward = ((x < 1.0 and ux < -1.0) or (x > L - 1.0 and ux > 1.0)
                   or (y < 1.0 and uy < -1.0) or (y > W - 1.0 and uy > 1.0))
        out_event[g] = bool(outward)

    s = pd.Series(speed_mean)
    w1 = max(1, int(round(fps)))
    d1 = (s[::-1].rolling(w1, min_periods=1).mean()[::-1] - s.rolling(w1, min_periods=1).mean()).to_numpy()
    fut_speed = pd.Series((d1 > 0.4).astype(np.float64))[::-1].rolling(int(2 * fps) + 1, min_periods=1).max()[::-1]
    kick = seen & (sp > 6.0) & (np.nan_to_num(sp0, nan=99.0) < 2.0)
    appear = seen & ~np.roll(seen, k4)
    restart = (kick | appear) & (fut_speed.to_numpy() > 0)

    slow1 = pd.Series(np.nan_to_num(share_slow)).rolling(w1, center=True, min_periods=1).mean().to_numpy()
    sp3 = s.rolling(int(3 * fps), center=True, min_periods=1).mean().to_numpy()
    alive_run = _run_s((np.nan_to_num(sp3) > 2.2) & (slow1 < 0.25), fps)
    stopped_raw = _bridge(slow1 > 0.4, int(fps))
    nob_raw = _bridge((~seen) & (slow1 > 0.3), int(fps))

    open_s = np.zeros(n); returns = np.zeros(n); team = np.zeros(n); nob = np.zeros(n)
    t_open, nret, t_team, t_nob = -1.0, 0, 0.0, 0.0
    blocked = False                                       # after a restart: off until the team has sped up
    dt = 1.0 / fps
    for i in range(n):
        if restart[i]:
            blocked, t_team, t_nob = True, 0.0, 0.0
        if blocked and slow1[i] < 0.4:
            blocked = False
        t_team = t_team + dt if (stopped_raw[i] and not blocked) else 0.0
        t_nob = t_nob + dt if (nob_raw[i] and not blocked) else 0.0
        team[i], nob[i] = min(t_team, 120.0), min(t_nob, 120.0)
        if out_event[i]:
            t_open, nret = 0.0, 0
        elif t_open >= 0:
            t_open += dt
            if restart[i] or (t_open > 10.0 and alive_run[i] >= 3.0) or t_open > 120.0:
                t_open = -1.0
            elif touch[i] and t_team > 0:
                nret += 1
        open_s[i] = t_open if t_open >= 0 else 0.0
        returns[i] = nret if t_open >= 0 else 0
    return pd.DataFrame({"m3_open_s": open_s, "m3_returns": returns, "s3_team_stopped_s": team,
                         "s3_slow_nob_s": nob, "r_since_restart_s": _since(restart, fps),
                         "r_until_restart_s": _since(restart[::-1], fps)[::-1]})


def keeper_v8(b: np.ndarray, players: np.ndarray, fps: float, pitch) -> pd.DataFrame:
    """v8 (team, our match: a save and catch called dead). A keeper holding the ball is IN PLAY (Law 12; every
    training source labels it so). Geometry only -- a keeper's kit is often given the wrong team:

    k_hold           ball tracked in a penalty area, within 1.2 m of the player deepest toward that goal (+-1 m),
                     under 3 m/s; carried into a ball gap that starts while he holds it (lost in his hands, up to 10 s)
    k_hold_s         seconds of the current hold (dips under 0.5 s bridged);  k_hold_3s  its share over 3 s
    k_since_shot_s   seconds since a shot: the ball over 12 m/s, heading to the nearer goal, in the final 35 m
    k_save_hold      a hold that starts within 5 s of a shot (a save and catch)
    """
    L, W = pitch
    n = len(b)
    bx, by = b[:, 0].astype(np.float64), b[:, 1].astype(np.float64)
    seen = ~np.isnan(bx)
    w = max(1, int(round(0.2 * fps)))
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    if n > 2 * w:
        vx[w:-w] = (bx[2 * w:] - bx[:-2 * w]) * fps / (2 * w)
        vy[w:-w] = (by[2 * w:] - by[:-2 * w]) * fps / (2 * w)
    sp = np.hypot(vx, vy)
    if n > 2 * w:                                         # at a gap's edge: speed from the frames BEFORE it
        back = np.full(n, np.nan)
        back[2 * w:] = np.hypot(bx[2 * w:] - bx[:-2 * w], by[2 * w:] - by[:-2 * w]) * fps / (2 * w)
        sp = np.where(np.isnan(sp), back, sp)
    with np.errstate(invalid="ignore"):
        in_box_l = seen & (bx < 16.5) & (np.abs(by - W / 2) < 20.16)
        in_box_r = seen & (bx > L - 16.5) & (np.abs(by - W / 2) < 20.16)
        P = players.astype(np.float64)
        d = np.linalg.norm(P - b[:, None, :].astype(np.float64), axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    k = np.argmin(d, axis=1)
    near = d[np.arange(n), k]
    xs = P[..., 0]
    depth_l = np.where(np.isnan(xs), np.inf, xs)
    depth_r = np.where(np.isnan(xs), np.inf, L - xs)
    deepest_l = depth_l[np.arange(n), k] <= depth_l.min(axis=1) + 1.0
    deepest_r = depth_r[np.arange(n), k] <= depth_r.min(axis=1) + 1.0
    with np.errstate(invalid="ignore"):
        hold = ((in_box_l & deepest_l) | (in_box_r & deepest_r)) & (near < 1.2) & (np.nan_to_num(sp, nan=99.0) < 3.0)
    for g in np.flatnonzero(np.diff(np.r_[1, seen.astype(np.int8)]) == -1):   # lost in his hands
        if g > 0 and hold[g - 1]:
            e = g
            while e < n and not seen[e] and e - g < int(10 * fps):
                e += 1
            hold[g:e] = True
    with np.errstate(invalid="ignore"):
        to_left = (bx < L / 2) & (vx < 0)
        to_right = (bx >= L / 2) & (vx > 0)
        final35 = (bx < 35.0) | (bx > L - 35.0)
        shot = seen & (sp > 12.0) & (to_left | to_right) & final35
    since_shot = _since(np.nan_to_num(shot, nan=0).astype(bool), fps, 60.0)
    hold_b = _bridge(hold, int(0.5 * fps))
    hold_s = _run_s(hold_b, fps)
    start = hold_b & ~np.r_[False, hold_b[:-1]]
    save_start = start & (since_shot < 5.0)
    save_hold = np.zeros(n, dtype=bool)
    for s0 in np.flatnonzero(save_start):
        e = s0
        while e < n and hold_b[e]:
            e += 1
        save_hold[s0:e] = True
    return pd.DataFrame({"k_hold": hold.astype(np.float64), "k_hold_s": hold_s,
                         "k_hold_3s": _roll(hold.astype(np.float64), int(round(3 * fps))),
                         "k_since_shot_s": since_shot, "k_save_hold": save_hold.astype(np.float64)})


def back_v8(speed_mean: np.ndarray, share_slow: np.ndarray, fps: float) -> pd.DataFrame:
    """v8 (from João's model, EXPERIMENTS #43): a stoppage should START only once the players HAVE stopped.
    Centred windows see a stoppage coming and pull its start early; these look only BACK in time."""
    s, sl = pd.Series(speed_mean), pd.Series(np.nan_to_num(share_slow))
    w1, w3 = max(1, int(round(fps))), max(1, int(round(3 * fps)))
    return pd.DataFrame({"b8_slow_back1s": sl.rolling(w1, min_periods=1).mean().to_numpy(),
                         "b8_slow_back3s": sl.rolling(w3, min_periods=1).mean().to_numpy(),
                         "b8_speed_back1s": s.rolling(w1, min_periods=1).mean().to_numpy()})


def build(players, team, b, fps, pitch, halves) -> pd.DataFrame:
    """(n, F) inputs, NaN outside the halves. Older models read only the columns they were trained on."""
    P = dead.features(players, team, fps, pitch, halves)
    seen = ~np.isnan(b[:, 0])
    X = pd.concat([P, ball(b, fps, pitch), edges(P["speed_mean"].to_numpy(), fps),
                   state(P["speed_mean"].to_numpy(), P["share_slow"].to_numpy(), seen, fps),
                   setpiece(players, np.asarray(team), b, fps),
                   memory(b, players, P["share_slow"].to_numpy(), fps, pitch),
                   stopped_v6(b, players, P["speed_mean"].to_numpy(), P["share_slow"].to_numpy(), fps, pitch),
                   stopped_v7(b, P["speed_mean"].to_numpy(), P["share_slow"].to_numpy(), fps, pitch),
                   keeper_v8(b, players, fps, pitch),
                   back_v8(P["speed_mean"].to_numpy(), P["share_slow"].to_numpy(), fps)],
                  axis=1)
    inside = np.zeros(len(X), dtype=bool)
    for _, a, z in halves:
        inside[a:z + 1] = True
    X.loc[~inside] = np.nan
    return X
