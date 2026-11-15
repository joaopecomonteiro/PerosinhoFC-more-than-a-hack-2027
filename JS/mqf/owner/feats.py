"""Feature tables for the owner models -- identical code for every source (and, later, BAR-ATH).

Feature sets (all from positions only, no identities, every window in seconds so 10 / 25 / 29.97 fps
games give the same columns):

  who         who.py's 59 columns, unchanged (who_v1 was trained on these)
  full        who + the cues Sigmar listed that who lacks:
                rel_v1_h/a    |v_ball - v_player| of each team's nearest player (the ball moves WITH
                              a player who controls it)
                w3 windows    +-3 s centred mean / min of each team's nearest distance, players within
                              2 m, ball speed (mean / max) and the share of frames the ball is seen
                shape         per team, measured from its own goal: last line (2nd deepest player),
                              highest player, length, compactness (mean distance to the team centre),
                              share of players in the opponent half; and their +-3 s means
  full_rules  full + the rule owner's columns (rules.py: soft P(home), clocks to / from each team's
              controls and touches) -- "rules as features"

Sequences (seq_inputs): the 3 nearest players of each team relative to the ball (dx, dy towards the
opponent goal, and their velocity relative to the ball's) appended to either the per-frame part of
`full` (no clocks, no windows -- the GRU builds its own context) or the whole table (the TCN).
"""
from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd

from .. import who
from . import rules

FEATURE_SETS = ("who", "full", "full_rules")
W3_S = 3.0
K_NEAR = 3
VERSION = 1
CONTEXT_MARKERS = ("since", "until", "past", "next", "_1s", "_w3", "rule_")


@functools.lru_cache(maxsize=1)
def who_columns() -> list[str]:
    """who.features' column names (computed on a tiny dummy game, so they track who.py)."""
    n = 40
    players = np.zeros((n, 4, 2))
    players[:, :, 0] = np.arange(4) * 10.0
    ball = np.full((n, 2), 5.0)
    return list(who.features(players, np.array([0, 0, 1, 1]), ball, 25, (105.0, 68.0), ((1, 0, n - 1),)).columns)


def _nan(fn, a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def _slot_velocity(xy: np.ndarray, j: np.ndarray, fps: float, half_s: float) -> np.ndarray:
    """(n, 2) velocity of slot j[t] at frame t, centred over +- half_s; NaN at the ends / if missing."""
    n = len(xy)
    w = max(1, int(round(half_s * fps)))
    out = np.full((n, 2), np.nan)
    if n > 2 * w:
        t = np.arange(w, n - w)
        out[t] = (xy[t + w, j[t]] - xy[t - w, j[t]]) * fps / (2 * w)
    return out


def _team_axis(x: np.ndarray, team_sign: np.ndarray, L: float) -> np.ndarray:
    """Distance from the team's OWN goal line along x. team_sign (n,): +1 if the team attacks +x."""
    return np.where(team_sign[:, None] > 0, x, L - x)


def extra_columns(players, team, ball, fps, pitch, halves) -> dict:
    """The `full` columns that who.features lacks (per frame, before the +-3 s windows)."""
    L, W = pitch
    xy = np.asarray(players, dtype=np.float64)
    b = np.asarray(ball, dtype=np.float64)
    team = np.asarray(team)
    n = len(b)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(xy - b[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    vb = who._centred_diff(b, fps, who.VEL_HALF_S)
    x = xy[..., 0]
    cx_h = _nan(np.nanmean, np.where(team[None, :] == 0, x, np.nan))
    cx_a = _nan(np.nanmean, np.where(team[None, :] == 1, x, np.nan))
    sgn = who.attack_sign(cx_h, cx_a, halves, n)                 # home attacks towards sgn * x
    f = {}
    for tm, s in ((0, "h"), (1, "a")):
        m = team == tm
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            for c in ("rel_v1", "line_def", "line_att", "length", "compact", "opp_half"):
                f[f"{c}_{s}"] = np.full(n, np.nan)
            continue
        dt = d[:, idx]
        j = idx[np.argmin(dt, axis=1)]
        near_ok = np.isfinite(dt.min(axis=1))
        vp = _slot_velocity(xy, j, fps, who.VEL_HALF_S)
        with np.errstate(invalid="ignore"):
            f[f"rel_v1_{s}"] = np.where(near_ok, np.linalg.norm(vb - vp, axis=1), np.nan)
        ts = sgn if tm == 0 else -sgn
        depth = _team_axis(x[:, idx], ts, L)                      # from own goal, NaN where absent
        srt = np.sort(np.where(np.isnan(depth), np.inf, depth), axis=1)
        cnt = (~np.isnan(depth)).sum(axis=1)
        second = np.where(cnt >= 2, srt[:, min(1, srt.shape[1] - 1)], np.nan)
        top = _nan(np.nanmax, depth)
        f[f"line_def_{s}"] = second
        f[f"line_att_{s}"] = top
        f[f"length_{s}"] = top - second
        cxy = np.stack([_nan(np.nanmean, xy[:, idx, 0]), _nan(np.nanmean, xy[:, idx, 1])], axis=1)
        with np.errstate(invalid="ignore"):
            f[f"compact_{s}"] = _nan(np.nanmean, np.linalg.norm(xy[:, idx] - cxy[:, None, :], axis=-1))
            f[f"opp_half_{s}"] = np.where(cnt > 0, (depth > L / 2).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    for c in ("rel_v1", "line_def", "line_att", "length", "compact", "opp_half"):
        f[f"{c}_diff"] = f[f"{c}_a"] - f[f"{c}_h"]
    # the defenders' last line against the attackers' top: who is pinned back
    f["pin_h"] = f["line_att_h"] - (L - f["line_def_a"])          # > 0: home's top player beyond away's line
    f["pin_a"] = f["line_att_a"] - (L - f["line_def_h"])
    f["pin_diff"] = f["pin_a"] - f["pin_h"]
    return f


W3_COLS = {"d1_h": ("mean", "min"), "d1_a": ("mean", "min"), "n2m_h": ("mean",), "n2m_a": ("mean",),
           "ball_speed": ("mean", "max"), "rel_v1_h": ("mean",), "rel_v1_a": ("mean",),
           "centre_gap": ("mean",), "length_diff": ("mean",), "compact_diff": ("mean",), "pin_diff": ("mean",),
           "width_diff": ("mean",), "line_def_diff": ("mean",)}


def full_features(players, team, ball, fps, pitch, halves) -> pd.DataFrame:
    """who.features + extra_columns + the +-3 s centred windows; NaN outside the halves."""
    X = who.features(players, team, ball, fps, pitch, halves)
    ex = pd.DataFrame(extra_columns(players, team, ball, fps, pitch, halves))
    X = pd.concat([X, ex], axis=1)
    has = (~np.isnan(np.asarray(ball, dtype=np.float64)[:, 0])).astype(np.float64)
    w = max(1, int(round(2 * W3_S * fps)) + 1)
    out = {}
    inside = np.zeros(len(X), dtype=bool)
    for _, a, b in halves:
        inside[a:b + 1] = True
    for c, fns in W3_COLS.items():
        for fn in fns:
            out[f"{c}_w3{fn}"] = np.full(len(X), np.nan)
    out["ball_seen_w3"] = np.full(len(X), np.nan)
    for _, a, b in halves:
        seg = X.iloc[a:b + 1]
        for c, fns in W3_COLS.items():
            r = seg[c].rolling(w, center=True, min_periods=1)
            for fn in fns:
                out[f"{c}_w3{fn}"][a:b + 1] = getattr(r, fn)().to_numpy()
        out["ball_seen_w3"][a:b + 1] = pd.Series(has[a:b + 1]).rolling(w, center=True, min_periods=1).mean().to_numpy()
    X = pd.concat([X, pd.DataFrame(out)], axis=1)
    X.loc[~inside] = np.nan
    return X


def rule_columns(players, team, ball, fps, halves, rule_cfg=None, ball_filled=None) -> pd.DataFrame:
    res = rules.owner(players, team, ball, fps, halves, rule_cfg, ball_filled=ball_filled)
    return pd.DataFrame(res["columns"])


def features(players, team, ball, fps, pitch, halves, feature_set: str = "full", rule_cfg=None,
             ball_filled=None) -> pd.DataFrame:
    if feature_set == "who":
        return who.features(players, team, ball, fps, pitch, halves)
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature set {feature_set!r} not in {FEATURE_SETS}")
    X = full_features(players, team, ball, fps, pitch, halves)
    if feature_set == "full_rules":
        X = pd.concat([X, rule_columns(players, team, ball, fps, halves, rule_cfg, ball_filled)], axis=1)
    return X


# ------------------------------------------------------------------------------------ GRU inputs

def seq_columns(X: pd.DataFrame) -> list[str]:
    """The per-frame columns of a full table: no clocks, rolling windows or rule columns."""
    return [c for c in X.columns if not any(k in c for k in CONTEXT_MARKERS)]


def gru_columns(stored: list[str]) -> list[str]:
    """The GRU's inputs among the cached sequence columns: per-frame columns + nearest players."""
    return [c for c in stored if c.startswith("near_") or not any(k in c for k in CONTEXT_MARKERS)]


def tcn_columns(stored: list[str], feature_set: str = "full") -> list[str]:
    """The TCN's inputs: every cached column (full_rules) or all but the rule columns (full)."""
    return [c for c in stored if feature_set == "full_rules" or not c.startswith("rule_")]


def nearest_players(players, team, ball, fps, halves, k: int = K_NEAR) -> np.ndarray:
    """(n, 2 * k * 4): for each team, the k players nearest the ball -- dx, dy (dx towards the
    opponent goal of THAT row's home team, i.e. home's attacking direction), and velocity relative
    to the ball. NaN where the ball or the player is missing."""
    xy = np.asarray(players, dtype=np.float64)
    b = np.asarray(ball, dtype=np.float64)
    team = np.asarray(team)
    n = len(b)
    rows = np.arange(n)
    x = xy[..., 0]
    cx_h = _nan(np.nanmean, np.where(team[None, :] == 0, x, np.nan))
    cx_a = _nan(np.nanmean, np.where(team[None, :] == 1, x, np.nan))
    sgn = who.attack_sign(cx_h, cx_a, halves, n)
    vb = who._centred_diff(b, fps, who.VEL_HALF_S)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(xy - b[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    out = np.full((n, 2, k, 4), np.nan)
    for tm in (0, 1):
        idx = np.flatnonzero(team == tm)
        if not len(idx):
            continue
        dt = d[:, idx]
        kk = min(k, len(idx))
        order = np.argsort(dt, axis=1, kind="stable")[:, :kk]
        for r in range(kk):
            j = idx[order[:, r]]
            ok = np.isfinite(dt[rows, order[:, r]])
            rel = xy[rows, j] - b
            vp = _slot_velocity(xy, j, fps, who.VEL_HALF_S) - vb
            out[:, tm, r, 0] = np.where(ok, sgn * rel[:, 0], np.nan)
            out[:, tm, r, 1] = np.where(ok, rel[:, 1], np.nan)
            out[:, tm, r, 2] = np.where(ok, sgn * vp[:, 0], np.nan)
            out[:, tm, r, 3] = np.where(ok, vp[:, 1], np.nan)
    return out.reshape(n, -1)


def seq_inputs(players, team, ball, fps, pitch, halves, X_full: pd.DataFrame | None = None, columns: str = "frame"):
    """(n, F) float32 sequence inputs and their column names; NaN where unknown.
    columns="frame": the per-frame columns of X (no clocks / windows / rules) -- the GRU's inputs;
    columns="all": every column of X (the TCN reads them, the GRU picks its subset by name).
    Either way the 3 nearest players of each team relative to the ball are appended."""
    X = X_full if X_full is not None else full_features(players, team, ball, fps, pitch, halves)
    cols = seq_columns(X) if columns == "frame" else list(X.columns)
    near = nearest_players(players, team, ball, fps, halves)
    names = cols + [f"near_{t}{r}_{q}" for t in "ha" for r in range(K_NEAR) for q in ("dx", "dy", "dvx", "dvy")]
    return np.concatenate([X[cols].to_numpy(dtype=np.float64), near], axis=1).astype(np.float32), names
