"""Evidence -> labels: hard vs probabilistic evidence, rule x model hybrids, alive layers, decoders.

Alive layers (where the ball is in play -- the team methods never decide it):
  oracle     the true alive mask of the game (D exactly where the GT says D): the who part alone,
             its ceiling
  smooth50   the official baseline's alive (starter/baseline.py, imported from vendor/, not copied):
             nearest-player codes where the ball is TRACKED, majority vote over +- 2 s, D where no
             team appears in the window. NB: under --hide-dead-ball the condition itself removes
             the ball on every true dead frame, so this layer is optimistic there
  dead_v1    the learned P(dead) (mqf.dead) inside the production 3-state HMM (decode.emissions):
             the realistic setting, the one decisions rest on

Decoders: with a hard layer (oracle, smooth50) a 2-state Viterbi (switch cost) inside every alive
run, D elsewhere; with dead_v1 the 3-state HMM of decode.py (same emissions, a faster Viterbi with
identical tie-breaking). "raw" = per-frame argmax, no decoder (flicker, for reference).
Boundary-aware decoding (the TCN's boundary head, ASRF-style): the H <-> A switch cost of frame t
becomes switch_cost * (1 - alpha * P(boundary at t)) -- cheap where a change is likely, unchanged
elsewhere; switch_costs() builds that array.

Hybrids (a model's p and the rule result r from rules.owner):
  product   logit p + lam * logit r_soft  (the rule as a second, independent expert)
  override  r's answer while the rule sees a confirmed control of the owner (in_ctrl), p elsewhere
  veto      decode p, then merge back every short spell of team T sandwiched between spells of the
            other team inside one alive run when the rule saw no control or pass by T inside it
            (and the ball was tracked on enough of it to have seen one) -- "touches do not count"

No training: blend() = who/who_lab.py's 'blend', the mean of the z_couple and z_chain log-odds
(feats.heading_columns) where either is known, as P(home).
"""
from __future__ import annotations

import functools
import importlib.util
import math
import warnings

import numpy as np

from .. import decode
from ..config import LABELS, OFFICIAL_DIR


def _logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def blend(z_couple, z_chain) -> np.ndarray:
    """P(home) = sigmoid(mean of the two log-odds where known); 0.5 where neither is (rounded to 1e-4)."""
    z = np.vstack([np.asarray(z_couple, dtype=np.float64), np.asarray(z_chain, dtype=np.float64)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # all-NaN columns -> NaN, handled below
        m = np.nanmean(z, axis=0)
    return np.round(_sig(np.nan_to_num(m, nan=0.0)), 4)


def hard(p: np.ndarray, clip: float = 0.02) -> np.ndarray:
    """A classifier that only names the team: argmax as clipped evidence (0.5 stays 0.5)."""
    p = np.asarray(p, dtype=np.float64)
    return np.where(np.isnan(p), np.nan, np.where(p > 0.5, 1 - clip, np.where(p < 0.5, clip, 0.5)))


def owner_to_p(owner: np.ndarray, inside: np.ndarray, clip: float = 0.02) -> np.ndarray:
    p = np.where(owner == 0, 1 - clip, np.where(owner == 1, clip, 0.5))
    return np.where(inside, p, np.nan)


def product(p_model, p_rule, lam: float = 1.0):
    pm = np.asarray(p_model, dtype=np.float64)
    pr = np.nan_to_num(np.asarray(p_rule, dtype=np.float64), nan=0.5)
    return np.where(np.isnan(pm), np.nan, np.round(_sig(_logit(pm) + lam * _logit(pr)), 4))


def override(p_model, rule_res: dict):
    pm = np.asarray(p_model, dtype=np.float64)
    return np.where(rule_res["in_ctrl"] & ~np.isnan(pm), rule_res["p_home"], pm)


# ------------------------------------------------------------------------------------ alive layers

@functools.lru_cache(maxsize=1)
def _official_baseline():
    path = OFFICIAL_DIR / "starter" / "baseline.py"
    from ..config import official
    official()                                           # puts vendor/ on sys.path for its `games` import
    spec = importlib.util.spec_from_file_location("mtah_official_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nearest_codes(players, team, ball) -> np.ndarray:
    """The official baseline's step 1 on arrays: team of the nearest tracked player where the ball
    is tracked, -1 elsewhere (the baseline reads the raw file; same rule)."""
    xy = np.asarray(players, dtype=np.float64)
    b = np.asarray(ball, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(xy - b[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    ok = np.isfinite(d[np.arange(len(d)), j])
    return np.where(ok, np.asarray(team)[j], -1).astype(np.int8)


def smooth50(players, team, ball, fps, halves, n: int, window_s: float = 2.0):
    """(labels, alive) of the official smooth50 baseline (window 50 frames at 25 fps = +- 2 s)."""
    base = _official_baseline()
    codes = nearest_codes(players, team, ball)
    lab = np.asarray(base.smooth(codes, max(1, int(round(window_s * fps)))))
    out = np.full(n, "X", dtype="<U1")
    for _, a, b in halves:
        out[a:b + 1] = lab[a:b + 1]
    return out, np.isin(out, ["H", "A"])


# ------------------------------------------------------------------------------------ decoders

def switch_costs(p_boundary, base: float, alpha: float) -> np.ndarray:
    """(n,) per-frame H <-> A switch cost from a boundary probability (NaN -> no change)."""
    pb = np.nan_to_num(np.asarray(p_boundary, dtype=np.float64), nan=0.0)
    return base * (1.0 - alpha * np.clip(pb, 0.0, 1.0))


def viterbi2(l0, l1, cost) -> np.ndarray:
    """2-state Viterbi, symmetric switch cost (a number, or one per frame: the cost of switching INTO
    that frame); ties as np.argmax (first state)."""
    l0 = l0.tolist()
    l1 = l1.tolist()
    n = len(l0)
    b0 = bytearray(n)
    b1 = bytearray(n)
    s0, s1 = l0[0], l1[0]
    costs = None if np.isscalar(cost) else np.asarray(cost, dtype=np.float64).tolist()
    for t in range(1, n):
        if costs is not None:
            cost = costs[t]
        c = s1 - cost
        if s0 >= c:
            n0 = s0
        else:
            n0 = c
            b0[t] = 1
        c = s0 - cost
        if c >= s1:
            n1 = c
        else:
            n1 = s1
            b1[t] = 1
        s0, s1 = n0 + l0[t], n1 + l1[t]
    path = np.empty(n, dtype=np.int8)
    k = 0 if s0 >= s1 else 1
    for t in range(n - 1, -1, -1):
        path[t] = k
        k = b1[t] if k else b0[t]
    return path


def viterbi3(ll: np.ndarray, p: decode.Params, sw_t=None) -> np.ndarray:
    """decode.viterbi in plain Python floats (same transitions, same first-max tie-breaking).
    sw_t: optional (n,) H <-> A switch cost per frame (cost of switching INTO frame t)."""
    sw, td, fd = p.switch_cost, p.to_dead_cost, p.from_dead_cost
    T = [[0.0, -sw, -td], [-sw, 0.0, -td], [-fd, -fd, 0.0]]
    L = ll.tolist()
    n = len(L)
    back = [None] * n
    s = list(L[0])
    sws = None if sw_t is None else np.asarray(sw_t, dtype=np.float64).tolist()
    for t in range(1, n):
        if sws is not None:
            T[0][1] = T[1][0] = -sws[t]
        bt = [0, 0, 0]
        ns = [0.0, 0.0, 0.0]
        for j in range(3):
            best, bi = s[0] + T[0][j], 0
            v = s[1] + T[1][j]
            if v > best:
                best, bi = v, 1
            v = s[2] + T[2][j]
            if v > best:
                best, bi = v, 2
            ns[j] = best + L[t][j]
            bt[j] = bi
        back[t] = bt
        s = ns
    path = np.empty(n, dtype=np.int8)
    k = int(np.argmax(s))
    for t in range(n - 1, 0, -1):
        path[t] = k
        k = back[t][k]
    path[0] = k
    return path


def runs(mask: np.ndarray):
    m = np.asarray(mask, dtype=bool)
    if not len(m):
        return []
    change = np.flatnonzero(m[1:] != m[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(m)]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if m[a]]


def decode_given_alive(p_home, alive, halves, n: int, switch_cost=18.0, w: float = 1.0,
                       clip: float = 0.02) -> np.ndarray:
    """D outside `alive` (inside the halves), H/A by a 2-state Viterbi inside every alive run.
    switch_cost: a number or (n,) per frame (switch_costs)."""
    lab = np.full(n, "X", dtype="<U1")
    pc = np.clip(np.nan_to_num(np.asarray(p_home, dtype=np.float64), nan=0.5), clip, 1 - clip)
    l0, l1 = w * np.log(pc), w * np.log1p(-pc)
    per_frame = not np.isscalar(switch_cost)
    for _, a, b in halves:
        lab[a:b + 1] = "D"
        for s, e in runs(alive[a:b + 1]):
            s, e = s + a, e + a
            path = viterbi2(l0[s:e], l1[s:e], switch_cost[s:e] if per_frame else switch_cost)
            lab[s:e] = np.where(path == 0, "H", "A")
    return lab


def decode_raw(p_home, alive, halves, n: int) -> np.ndarray:
    lab = np.full(n, "X", dtype="<U1")
    p = np.nan_to_num(np.asarray(p_home, dtype=np.float64), nan=0.5)
    for _, a, b in halves:
        sl = slice(a, b + 1)
        lab[sl] = np.where(alive[sl], np.where(p[sl] >= 0.5, "H", "A"), "D")
    return lab


def decode_hmm3(feat, p_home, halves, n: int, params: decode.Params, sw_t=None) -> np.ndarray:
    """The production decoder with the method's P(home) as p_home_model (dead_v1 in feat); sw_t:
    optional per-frame H <-> A switch cost."""
    ll = decode.emissions(feat.assign(p_home_model=np.asarray(p_home, dtype=np.float64)), params, None)
    lab = np.full(n, "X", dtype="<U1")
    for _, a, b in halves:
        lab[a:b + 1] = np.array(LABELS)[viterbi3(ll[a:b + 1], params, None if sw_t is None else sw_t[a:b + 1])]
    return lab


# ------------------------------------------------------------------------------------ spell veto

def veto(labels: np.ndarray, rule_res: dict, tracked: np.ndarray, fps: float, max_s: float = 10.0,
         min_ball_share: float = 0.5, max_rule_share: float = 0.05) -> np.ndarray:
    """Merge back short T spells between two spells of the other team U (no D between) when the rule
    owner is T on (almost) none of the spell -- the rule saw no control or pass by T, only touches --
    though the ball was tracked on at least min_ball_share of it (so the rule could have seen one).
    Repeats until nothing changes (a merge can create a new sandwich)."""
    out = np.asarray(labels).copy()
    own = rule_res["owner"]
    own_h = np.r_[0, np.cumsum(own == 0)]
    own_a = np.r_[0, np.cumsum(own == 1)]
    trk_cum = np.r_[0, np.cumsum(np.asarray(tracked, dtype=bool))]
    for _ in range(5):
        change = np.flatnonzero(out[1:] != out[:-1]) + 1
        s_all, e_all = np.r_[0, change], np.r_[change, len(out)]
        v_all = out[s_all]
        done = True
        for i in range(1, len(v_all) - 1):
            T, U = v_all[i], v_all[i - 1]
            if T not in ("H", "A") or U not in ("H", "A") or U == T or v_all[i + 1] != U:
                continue
            s, e = s_all[i], e_all[i]
            if (e - s) > max_s * fps:
                continue
            if (trk_cum[e] - trk_cum[s]) < min_ball_share * (e - s):
                continue
            share_t = ((own_h if T == "H" else own_a)[e] - (own_h if T == "H" else own_a)[s]) / (e - s)
            if share_t <= max_rule_share:
                out[s:e] = U
                done = False
        if done:
            break
    return out


# ------------------------------------------------------------------------------------ per-frame scores

def frame_metrics(y: np.ndarray, p: np.ndarray, clip: float = 0.02) -> dict:
    """On alive frames: accuracy (an undecided 0.5 counts half), AUC, log-loss and Brier at the
    decoder's clip, ECE over 10 bins."""
    ok = ~np.isnan(p)
    y, p = np.asarray(y)[ok].astype(np.float64), np.asarray(p)[ok]
    if not len(y):
        return {"n": 0}
    pc = np.clip(p, clip, 1 - clip)
    acc = float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1))))
    out = {"n": int(len(y)), "acc": round(acc, 4),
           "logloss": round(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), 4),
           "brier": round(float(np.mean((p - y) ** 2)), 4)}
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for k in range(10):
        m = bins == k
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    out["ece"] = round(float(ece), 4)
    if 0 < y.mean() < 1:
        from sklearn.metrics import roc_auc_score
        out["auc"] = round(float(roc_auc_score(y, p)), 4)
    else:
        out["auc"] = math.nan
    return out
