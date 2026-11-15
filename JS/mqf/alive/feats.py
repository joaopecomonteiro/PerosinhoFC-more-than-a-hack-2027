"""The dead-ball feature table, from POSITIONS only (players; the ball only where it is TRACKED).

One table, column groups by prefix; every model picks its set by name (FEATURE_SETS):

  v1         mqf.dead's 36 columns, unchanged -- a GBM on them trained like dead_v1 is its replica
  players    v1 + p_*: collective speed p95, players above 4 m/s, signed / absolute collective
             acceleration, slowest / fastest team, 60 s context, 1-2 s trends, the restart burst
             (>= 3 players above 4 m/s) and stillness (median < 1 m/s) clocks, still-run lengths
  formation  players + f_*: set-piece shapes from the Laws -- crowding of the busiest box and goal area
             (corners, free kicks near the box, goal kicks), a player at a corner mark (Law 17), beyond
             the touchline (throw-in, Law 15) or the goal line, kickoff shape (Law 8: everyone in his own
             half, someone on the centre spot), crowding (max team-mates + opponents within 3 m), mean
             nearest-neighbour distance, spread, distance of the crowd from the pitch's long axis
             (bench side: hydration breaks, injuries, substitutions), 3 / 10 s context
  full       formation + b_*: the ball where TRACKED -- out of the pitch (Law 9), signed distance to the
             lines, corner mark, goal area, centre spot, speed, still, a restart kick (speed jump from
             < 2 to > 7 m/s, as ELASTIC finds set-piece kicks), clocks since / until out and kick
  full_rules full + r_*: the rule outputs of rules.py

The ball's ABSENCE is never a feature (no "seen" flags, no missing-run clocks): whether a missing ball
means a stoppage depends on the tracking system (DFL keeps it, PFF's broadcast and BAR-ATH's tactical
camera lose it in stoppages), and the harness's --hide-dead-ball removes it on exactly the dead frames,
so any missingness cue would read the answer off the condition. data.py also makes the missingness
label-independent in every training view. A filled ball (mqf.ball) is never passed here.

Centred windows: labels are made offline and the whole half is known (as for the submission).
Everything is computed per half; NaN outside the halves.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import dead

CAP_S = 60.0                 # clocks stop counting here
BURST_MS, BURST_N = 4.0, 3   # a restart: this many players above this speed at once
STILL_MED_MS = 1.0           # collectively still: median player speed below this
FAST_MS = 4.0
BOX_DEPTH, BOX_HALF_W = 16.5, 20.16
GA_DEPTH, GA_HALF_W = 5.5, 9.16
CIRCLE_M = 9.15
CROWD_M = 3.0
KICK_FROM_MS, KICK_TO_MS = 2.0, 7.0
OUT_MARGIN_M = 0.5

FEATURE_SETS = ("v1", "players", "formation", "full", "full_rules")
_PREFIX_OF = {"v1": (), "players": ("p_",), "formation": ("p_", "f_"), "full": ("p_", "f_", "b_"),
              "full_rules": ("p_", "f_", "b_", "r_")}


def _nan(fn, a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def v1_columns() -> list[str]:
    """dead_v1's columns in its order (from the committed model when present, else from the code)."""
    from ..config import MODELS_DIR
    p = MODELS_DIR / "dead_v1.joblib"
    if p.exists():
        try:
            return list(dead.load_model(p)[1])
        except Exception:                                    # sklearn mismatch etc.: fall back to the code
            pass
    n, P = 400, 4
    xy = np.random.default_rng(0).uniform(0, 50, (n, P, 2))
    return list(dead.features(xy, np.array([0, 0, 1, 1]), 25, (105.0, 68.0), ((1, 0, n - 1),)).columns)


def keep_column(c: str, feature_set: str, v1: list[str]) -> bool:
    if c in v1:
        return True
    return any(c.startswith(p) for p in _PREFIX_OF[feature_set])


def columns_of(feature_set: str, all_columns: list[str]) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r} ({', '.join(FEATURE_SETS)})")
    v1 = v1_columns()
    cols = [c for c in all_columns if keep_column(c, feature_set, v1)]
    missing = [c for c in v1 if c not in cols]
    if missing:
        raise ValueError(f"the table lacks dead_v1 columns {missing[:3]}...")
    return cols


# ------------------------------------------------------------------------------------ helpers

def _since(flag: np.ndarray, fps: float, cap: float = CAP_S) -> np.ndarray:
    """Seconds since `flag` was last True (0 on a True frame); cap if never / longer ago."""
    t = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, t, -1))
    return np.where(last >= 0, np.minimum((t - last) / fps, cap), cap)


def _until(flag: np.ndarray, fps: float, cap: float = CAP_S) -> np.ndarray:
    return _since(flag[::-1], fps, cap)[::-1]


def _run_for(flag: np.ndarray, fps: float, cap: float = CAP_S) -> np.ndarray:
    """Seconds `flag` has been continuously True up to this frame (0 where False)."""
    f = np.asarray(flag, dtype=bool)
    t = np.arange(len(f))
    last_false = np.maximum.accumulate(np.where(~f, t, -1))
    return np.where(f, np.minimum((t - last_false) / fps, cap), 0.0)


def _run_left(flag: np.ndarray, fps: float, cap: float = CAP_S) -> np.ndarray:
    return _run_for(np.asarray(flag)[::-1], fps, cap)[::-1]


def _roll(x, fps: float, sec: float, fn: str = "mean", center: bool = True) -> np.ndarray:
    w = max(1, int(round(sec * fps)))
    r = pd.Series(np.asarray(x, dtype=np.float64)).rolling(w, center=center, min_periods=1)
    return getattr(r, fn)().to_numpy()


def _past_future(x, fps: float, sec: float):
    w = max(1, int(round(sec * fps)))
    s = pd.Series(np.asarray(x, dtype=np.float64))
    past = s.rolling(w, min_periods=1).mean().to_numpy()
    fut = s[::-1].rolling(w, min_periods=1).mean()[::-1].to_numpy()
    return past, fut


def _centred_diff(x: np.ndarray, fps: float, half_s: float) -> np.ndarray:
    w = max(1, int(round(half_s * fps)))
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if len(x) > 2 * w:
        out[w:-w] = (x[2 * w:] - x[:-2 * w]) * fps / (2 * w)
    return out


def sides(players: np.ndarray, team: np.ndarray, halves, n: int) -> np.ndarray:
    """(n,) +1 where home defends x = 0 (attacks +x), -1 otherwise -- per half, from the teams' mean x."""
    from ..who import attack_sign
    x = players[..., 0]
    cx_h = _nan(np.nanmean, x[:, team == 0])
    cx_a = _nan(np.nanmean, x[:, team == 1])
    return attack_sign(cx_h, cx_a, halves, n)


def _crowd(xy: np.ndarray, budget: int = 4_000_000):
    """(max players within CROWD_M of one player, mean nearest-neighbour distance) per frame.
    Lanes are compacted first (present players to the front): the cleaner can deliver ~200 lanes with
    ~22 present, and the pairwise block is (frames x K x K) with K the most present at once."""
    n = len(xy)
    cmax = np.full(n, np.nan)
    nnm = np.full(n, np.nan)
    pres_all = ~np.isnan(xy[..., 0])
    K = int(pres_all.sum(axis=1).max()) if n else 0
    if K == 0:
        return cmax, nnm
    order = np.argsort(~pres_all, axis=1, kind="stable")[:, :K]                  # present lanes first
    chunk = max(256, budget // (K * K))
    eye = np.eye(K, dtype=bool)[None]
    for a in range(0, n, chunk):
        p = np.take_along_axis(xy[a:a + chunk], order[a:a + chunk, :, None], axis=1)   # (c, K, 2), NaN padded
        d = np.sqrt(((p[:, :, None, :] - p[:, None, :, :]) ** 2).sum(axis=-1))   # (c, K, K), NaN if absent
        within = (d < CROWD_M).sum(axis=2).astype(np.float64)                     # self included (NaN < x: False)
        within[np.isnan(p[..., 0])] = np.nan
        cmax[a:a + chunk] = _nan(np.nanmax, within)
        d[np.broadcast_to(eye, d.shape)] = np.nan
        nnm[a:a + chunk] = _nan(np.nanmean, _nan(np.nanmin, d, axis=2))
    return cmax, nnm


# ------------------------------------------------------------------------------------ quantities

def frame_quantities(players: np.ndarray, team: np.ndarray, ball: np.ndarray, fps: float, pitch, halves) -> dict:
    """Per-frame arrays shared by the features and the rules (no time context yet)."""
    L, W = pitch
    xy = np.asarray(players, dtype=np.float64)
    team = np.asarray(team)
    n = len(xy)
    sp = dead.player_speed(xy, fps)
    acc = _centred_diff(sp, fps, 0.5)
    pres = ~np.isnan(xy[..., 0])
    x, y = xy[..., 0], xy[..., 1]
    q = {"n_pres": pres.sum(axis=1).astype(np.float64)}
    q["speed_mean"] = _nan(np.nanmean, sp)
    q["speed_median"] = _nan(np.nanmedian, sp)
    q["speed_p95"] = _nan(lambda a, axis: np.nanpercentile(a, 95, axis=axis), sp)
    q["n_fast"] = (np.nan_to_num(sp, nan=0.0) > FAST_MS).sum(axis=1).astype(np.float64)
    q["acc_mean"] = _nan(np.nanmean, acc)
    q["acc_abs"] = _nan(np.nanmean, np.abs(acc))
    th = _nan(np.nanmean, sp[:, team == 0])
    ta = _nan(np.nanmean, sp[:, team == 1])
    q["team_speed_min"] = np.fmin(th, ta)
    q["team_speed_max"] = np.fmax(th, ta)
    # formation
    in_box_l = pres & (x < BOX_DEPTH) & (np.abs(y - W / 2) < BOX_HALF_W)
    in_box_r = pres & (x > L - BOX_DEPTH) & (np.abs(y - W / 2) < BOX_HALF_W)
    bl, br = in_box_l.sum(axis=1).astype(np.float64), in_box_r.sum(axis=1).astype(np.float64)
    q["box_max"], q["box_min"] = np.maximum(bl, br), np.minimum(bl, br)
    q["box_l"], q["box_r"] = bl, br
    ga_l = pres & (x < GA_DEPTH) & (np.abs(y - W / 2) < GA_HALF_W)
    ga_r = pres & (x > L - GA_DEPTH) & (np.abs(y - W / 2) < GA_HALF_W)
    q["ga_l"], q["ga_r"] = ga_l.sum(axis=1).astype(np.float64), ga_r.sum(axis=1).astype(np.float64)
    q["goalarea_max"] = np.maximum(q["ga_l"], q["ga_r"])
    corners = np.array([[0, 0], [0, W], [L, 0], [L, W]], dtype=np.float64)
    dc = np.linalg.norm(xy - corners[0], axis=-1)                                           # (n, P), one corner
    for cpt in corners[1:]:                                                                 # at a time (memory)
        dc = np.fmin(dc, np.linalg.norm(xy - cpt, axis=-1))
    q["d_corner"] = _nan(np.nanmin, dc)
    q["n_corner2"] = (np.nan_to_num(dc, nan=99.0) < 2.0).sum(axis=1).astype(np.float64)
    dt = np.minimum(y, W - y)                                    # signed: < 0 beyond a touchline
    dg = np.minimum(x, L - x)
    q["d_touch"] = _nan(np.nanmin, dt)
    q["n_beyond_touch"] = (np.nan_to_num(dt, nan=99.0) < -0.3).sum(axis=1).astype(np.float64)
    q["n_at_touch"] = (np.nan_to_num(dt, nan=99.0) < 1.0).sum(axis=1).astype(np.float64)
    q["d_goalline"] = _nan(np.nanmin, dg)
    q["n_beyond_goal"] = (np.nan_to_num(dg, nan=99.0) < -0.3).sum(axis=1).astype(np.float64)
    dcen = np.linalg.norm(xy - np.array([L / 2, W / 2]), axis=-1)
    q["d_centre"] = _nan(np.nanmin, dcen)
    q["n_circle"] = (np.nan_to_num(dcen, nan=99.0) < CIRCLE_M).sum(axis=1).astype(np.float64)
    sgn = sides(xy, team, halves, n)                             # +1: home defends x = 0
    own_left = np.where(team[None, :] == 0, sgn[:, None] > 0, sgn[:, None] < 0)   # (n, P) this player defends x = 0
    in_own = np.where(own_left, x <= L / 2 + 1.0, x >= L / 2 - 1.0)
    q["own_half"] = np.where(q["n_pres"] > 0, (pres & in_own).sum(axis=1) / np.maximum(q["n_pres"], 1), np.nan)
    q["crowd_max"], q["nn_mean"] = _crowd(xy)
    q["spread_all"] = _nan(np.nanstd, x) + _nan(np.nanstd, y)
    q["extent_x"] = _nan(np.nanmax, x) - _nan(np.nanmin, x)
    q["cy_off"] = np.abs(_nan(np.nanmean, y) - W / 2)
    # ball, only where tracked
    b = np.asarray(ball, dtype=np.float64)
    bx, by = b[:, 0], b[:, 1]
    seen = ~np.isnan(bx)
    out = (bx < -OUT_MARGIN_M) | (bx > L + OUT_MARGIN_M) | (by < -OUT_MARGIN_M) | (by > W + OUT_MARGIN_M)
    q["b_seen"] = seen                                          # rules only: never a feature column
    q["b_out"] = np.where(seen, out.astype(np.float64), np.nan)
    q["b_d_touch"] = np.minimum(by, W - by)
    q["b_d_goal"] = np.minimum(bx, L - bx)
    q["b_d_line"] = np.minimum(q["b_d_touch"], q["b_d_goal"])
    q["b_d_corner"] = np.min(np.linalg.norm(b[:, None, :] - corners[None], axis=-1), axis=1)
    q["b_in_ga"] = np.where(seen, ((bx < GA_DEPTH) | (bx > L - GA_DEPTH)) & (np.abs(by - W / 2) < GA_HALF_W), np.nan)
    q["b_d_centre"] = np.linalg.norm(b - np.array([L / 2, W / 2]), axis=1)
    v = _centred_diff(b, fps, 0.2)
    q["b_speed"] = np.linalg.norm(v, axis=1)
    w = max(1, int(round(0.5 * fps)))
    disp = np.full(n, np.nan)
    if n > 2 * w:
        disp[w:-w] = np.linalg.norm(b[2 * w:] - b[:-2 * w], axis=1)
    q["b_still"] = np.where(np.isnan(disp), np.nan, (disp < 0.5).astype(np.float64))
    k = max(1, int(round(0.4 * fps)))
    s = q["b_speed"]
    prev_slow = pd.Series(s).shift(1).rolling(k, min_periods=max(1, k // 2)).max().to_numpy() < KICK_FROM_MS
    nxt_fast = pd.Series(s)[::-1].shift(1).rolling(k, min_periods=max(1, k // 2)).max()[::-1].to_numpy() > KICK_TO_MS
    q["b_kick"] = prev_slow & nxt_fast
    with np.errstate(invalid="ignore"):
        dpl = np.linalg.norm(xy - b[:, None, :], axis=-1)
    q["b_d_player"] = _nan(np.nanmin, dpl)
    return q


# ------------------------------------------------------------------------------------ table

def _half_columns(q: dict, a: int, b: int, fps: float) -> dict:
    """The p_ / f_ / b_ columns of one half (inclusive frames a..b)."""
    sl = slice(a, b + 1)
    g = {k: (v[sl] if isinstance(v, np.ndarray) else v) for k, v in q.items()}
    c = {}
    sm = g["speed_mean"]
    c["p_speed_p95"] = g["speed_p95"]
    c["p_n_fast"] = g["n_fast"]
    c["p_acc_mean"] = g["acc_mean"]
    c["p_acc_abs"] = g["acc_abs"]
    c["p_team_speed_min"] = g["team_speed_min"]
    c["p_team_speed_max"] = g["team_speed_max"]
    c["p_speed_mean_60s"] = _roll(sm, fps, 60)
    c["p_speed_min_10s"] = _roll(sm, fps, 10, "min")
    c["p_speed_max_3s"] = _roll(sm, fps, 3, "max")
    for sec in (1, 2):
        past, fut = _past_future(sm, fps, sec)
        c[f"p_trend_{sec}s"] = fut - past
    c["p_acc_max_next2s"] = pd.Series(g["acc_mean"])[::-1].rolling(int(2 * fps), min_periods=1).max()[::-1].to_numpy()
    c["p_acc_min_prev2s"] = pd.Series(g["acc_mean"]).rolling(int(2 * fps), min_periods=1).min().to_numpy()
    c["p_speed_rank_3s"] = pd.Series(_roll(sm, fps, 3)).rank(pct=True).to_numpy()
    burst = g["n_fast"] >= BURST_N
    still = np.nan_to_num(g["speed_median"], nan=0.0) < STILL_MED_MS
    c["p_since_burst_s"], c["p_until_burst_s"] = _since(burst, fps), _until(burst, fps)
    c["p_since_still_s"], c["p_until_still_s"] = _since(still, fps), _until(still, fps)
    c["p_still_for_s"], c["p_still_left_s"] = _run_for(still, fps), _run_left(still, fps)
    c["p_n_pres_10s"] = _roll(g["n_pres"], fps, 10)
    # formation
    for k in ("box_max", "box_min", "goalarea_max", "d_corner", "n_corner2", "d_touch", "n_beyond_touch", "n_at_touch",
              "d_goalline", "n_beyond_goal", "d_centre", "n_circle", "own_half", "crowd_max", "nn_mean", "spread_all",
              "extent_x", "cy_off"):
        c[f"f_{k}"] = g[k]
        c[f"f_{k}_3s"] = _roll(g[k], fps, 3)
    for k in ("box_max", "crowd_max", "own_half", "n_at_touch", "cy_off"):
        c[f"f_{k}_10s"] = _roll(g[k], fps, 10)
    c["f_n_beyond_touch_max3s"] = _roll(g["n_beyond_touch"], fps, 3, "max")
    c["f_n_corner2_max3s"] = _roll(g["n_corner2"], fps, 3, "max")
    # ball (tracked only; NaN-ignoring windows)
    for k in ("b_out", "b_d_touch", "b_d_goal", "b_d_line", "b_d_corner", "b_in_ga", "b_d_centre", "b_speed",
              "b_still", "b_d_player"):
        c[k] = g[k]
    c["b_out_3s"] = _roll(g["b_out"], fps, 3, "max")
    c["b_speed_med_3s"] = _roll(g["b_speed"], fps, 3, "median")
    c["b_still_3s"] = _roll(g["b_still"], fps, 3)
    out = np.nan_to_num(g["b_out"], nan=0.0) > 0.5
    c["b_since_out_s"], c["b_until_out_s"] = _since(out, fps), _until(out, fps)
    kick = np.asarray(g["b_kick"], dtype=bool)
    c["b_since_kick_s"], c["b_until_kick_s"] = _since(kick, fps), _until(kick, fps)
    stl = np.nan_to_num(g["b_still"], nan=0.0) > 0.5
    c["b_still_for_s"] = _run_for(stl, fps)
    return c


def features(players: np.ndarray, team: np.ndarray, ball: np.ndarray, fps: float, pitch, halves,
             feature_set: str = "full_rules", rule_cfg=None, return_rules: bool = False):
    """(n, F) DataFrame, NaN outside the halves: dead_v1's columns, then p_ / f_ / b_ (/ r_) columns.
    `ball` must be the TRACKED ball (NaN where missing), never a filled one."""
    from . import rules as rules_mod
    n = len(players)
    team = np.asarray(team)
    v1 = dead.features(players, team, fps, pitch, halves)
    q = frame_quantities(players, team, ball, fps, pitch, halves)
    parts, names = None, None
    for _, a, b in halves:
        c = _half_columns(q, a, b, fps)
        if parts is None:
            names = list(c)
            parts = np.full((n, len(names)), np.nan, dtype=np.float64)
        parts[a:b + 1] = np.column_stack([np.asarray(c[k], dtype=np.float64) for k in names])
    X = pd.concat([v1, pd.DataFrame(parts, columns=names)], axis=1)
    rr = None
    if feature_set == "full_rules" or return_rules:
        rr = rules_mod.apply(q, fps, pitch, halves, n, rule_cfg)
        if feature_set == "full_rules":
            X = pd.concat([X, rules_mod.columns(rr, halves, n)], axis=1)
    inside = np.zeros(n, dtype=bool)
    for _, a, b in halves:
        inside[a:b + 1] = True
    X.loc[~inside] = np.nan
    X.loc[(q["n_pres"] == 0) & inside, [c for c in X.columns if not c.startswith(("b_", "r_"))]] = np.nan
    if feature_set not in ("full", "full_rules"):
        X = X[columns_of(feature_set, list(X.columns))]
    return (X, rr) if return_rules else X
