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


def build(players, team, b, fps, pitch, halves) -> pd.DataFrame:
    """(n, F) inputs, NaN outside the halves."""
    P = dead.features(players, team, fps, pitch, halves)
    X = pd.concat([P, ball(b, fps, pitch), edges(P["speed_mean"].to_numpy(), fps)], axis=1)
    inside = np.zeros(len(X), dtype=bool)
    for _, a, z in halves:
        inside[a:z + 1] = True
    X.loc[~inside] = np.nan
    return X
