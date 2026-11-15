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


def build(players, team, b, fps, pitch, halves) -> pd.DataFrame:
    """(n, F) inputs, NaN outside the halves. Older models read only the columns they were trained on."""
    P = dead.features(players, team, fps, pitch, halves)
    seen = ~np.isnan(b[:, 0])
    X = pd.concat([P, ball(b, fps, pitch), edges(P["speed_mean"].to_numpy(), fps),
                   state(P["speed_mean"].to_numpy(), P["share_slow"].to_numpy(), seen, fps),
                   setpiece(players, np.asarray(team), b, fps),
                   memory(b, players, P["share_slow"].to_numpy(), fps, pitch)], axis=1)
    inside = np.zeros(len(X), dtype=bool)
    for _, a, z in halves:
        inside[a:z + 1] = True
    X.loc[~inside] = np.nan
    return X
