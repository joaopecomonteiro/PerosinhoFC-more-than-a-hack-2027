"""P(dead) -> labels: the production decoder and its dead-ball variants, rule x model hybrids, the Smart
Tagging prior, and stoppage-level diagnostics.

Decoder (always the production 3-state HMM of decode.py, who_v1's P(home) held fixed):
  hmm        decode.emissions with the method's P(dead) as p_dead_model (v0: none -> the speed rule),
             Viterbi per half with the Params costs
  start      every half starts dead (a kickoff opens it, Law 8): D gets a bonus of from_dead_cost at the
             half's first frame -- the 2nd half of BAR-ATH is graded ~2 min before its kickoff
  bnd<a>     boundary-aware dead costs (ASRF-style): to_dead / from_dead cost of frame t times
             (1 - a P(alive <-> dead change near t)) from a sequence model's boundary head
  min<s>     stoppages shorter than s seconds are removed after decoding (physically a restart takes
             >= ~2 s; a 1 s D blip splits one possession into two)
  snap<s>    each restart moves to the nearest rule restart candidate (ball kick / burst onset) within
             +-s seconds, each stop to the nearest stop candidate (ball seen out / stillness onset)
  veto       predicted stoppages < 6 s that the rules do not support (rule P(dead) < 0.5 on most of it)
             are removed
  tags       the Smart Tagging prior (below), weight --tag-w
Variants combine with '+', e.g. "start+bnd0.5+min2".

Hybrids (a model's P(dead) p, the rule result r):
  prod       logit p + logit r_soft      (the rules as a second, independent expert)
  override   ball seen out -> 0.97, a tracked restart kick -> 0.05, p elsewhere

Smart Tagging prior: SET PIECES windows open ~3.5 s before the restart they contain (DATA.md) -> the
first `lead_s` seconds of every window get +1 log-odds unit of dead (times the weight). BAR-ATH: from the
XML (tag_windows_from_xml). Other games have no tags: tag-like windows are simulated from the labels
(simulate_tag_windows: each restart tagged with probability `recall`, window start 2.5-4.5 s before it,
end 2-8 s after, plus false windows at `fp_per90` per 90 min) -- assumptions, swept, never fitted to
BAR-ATH. GOALS windows: an optional prior after the window's end (w_goal, default off: not testable on
other games).
"""
from __future__ import annotations

import math
import re

import numpy as np

from .. import decode
from ..config import LABELS

EPS = 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def runs_of(lab: np.ndarray, value) -> list[tuple[int, int]]:
    m = np.asarray(lab) == value
    if not len(m):
        return []
    ch = np.flatnonzero(m[1:] != m[:-1]) + 1
    s, e = np.r_[0, ch], np.r_[ch, len(m)]
    return [(int(a), int(b)) for a, b in zip(s, e) if m[a]]


# ------------------------------------------------------------------------------------ hybrids

def product(p_model, p_rule, lam: float = 1.0):
    pm = np.asarray(p_model, dtype=np.float64)
    pr = np.nan_to_num(np.asarray(p_rule, dtype=np.float64), nan=0.5)
    return np.where(np.isnan(pm), np.nan, np.round(_sig(_logit(pm) + lam * _logit(pr)), 4))


def override(p_model, rr: dict):
    pm = np.asarray(p_model, dtype=np.float64)
    out = np.where(rr["out"] & ~np.isnan(pm), 0.97, pm)
    return np.where(rr["kick"] & ~np.isnan(pm), 0.05, out)


def with_prior(p, bonus, w: float):
    """P(dead) with w x bonus log-odds added (NaN kept)."""
    p = np.asarray(p, dtype=np.float64)
    if w == 0 or bonus is None:
        return p
    return np.where(np.isnan(p), np.nan, np.round(_sig(_logit(p) + w * bonus), 4))


# ------------------------------------------------------------------------------------ Smart Tagging prior

def tag_bonus(n: int, windows, fps: float, lead_s: float = 2.5, goal_windows=(), goal_after_s=(2.0, 20.0),
              w_goal: float = 0.0) -> np.ndarray:
    """(n,) log-odds bonus for dead: +1 on the first lead_s seconds of each SET PIECES window; w_goal on
    [end + goal_after_s[0], end + goal_after_s[1]] of each GOALS window."""
    out = np.zeros(n)
    L = int(round(lead_s * fps))
    for s, e in windows:
        s, e = max(0, int(s)), min(n, int(e))
        out[s:min(e, s + L)] = 1.0
    if w_goal:
        for s, e in goal_windows:
            a, b = int(e + goal_after_s[0] * fps), int(e + goal_after_s[1] * fps)
            out[max(0, a):min(n, b)] += w_goal
    return out


def simulate_tag_windows(gt: np.ndarray, halves, fps: float, rng, recall: float = 0.6, fp_per90: float = 2.0,
                         lead=(2.5, 4.5), tail=(2.0, 8.0)) -> list[tuple[int, int]]:
    """Tag-like SET PIECES windows from a game's labels (other games only)."""
    out = []
    inside_n = 0
    for _, a, b in halves:
        lab = np.asarray(gt[a:b + 1])
        inside_n += b - a + 1
        rest = np.flatnonzero((lab[1:] != "D") & (lab[:-1] == "D")) + 1 + a        # first alive frame after D
        for r in rest:
            if rng.random() < recall:
                out.append((int(r - rng.uniform(*lead) * fps), int(r + rng.uniform(*tail) * fps)))
    n_fp = rng.poisson(fp_per90 * inside_n / (90 * 60 * fps))
    alive = np.flatnonzero(np.isin(gt, ["H", "A"]))
    for _ in range(int(n_fp)):
        if len(alive):
            c = int(alive[rng.integers(0, len(alive))])
            out.append((c, int(c + rng.uniform(lead[0] + tail[0], lead[1] + tail[1]) * fps)))
    return sorted(out)


def tag_windows_from_xml(path, fps: float, n: int):
    """(set-piece windows, goal windows) of a Smart Tagging file -- BAR-ATH, on the team's machine."""
    from .. import smart_tagging
    sp, gl = [], []
    for ev in smart_tagging.load_instances(path, fps):
        a, b = max(0, ev["start"]), min(n - 1, ev["end"])
        if b < a:
            continue
        if ev["code"] == smart_tagging.SET_PIECES:
            sp.append((a, b))
        elif ev["code"] == "goals":
            gl.append((a, b))
    return sorted(sp), sorted(gl)


# ------------------------------------------------------------------------------------ decoder

def viterbi3(ll: np.ndarray, sw, td, fd, start_dead: float = 0.0) -> np.ndarray:
    """3-state Viterbi (H, A, D), first-max tie-breaking and the same float operations as decode.viterbi
    (cand = score[i] + T[i, j], then + ll[t, j]). sw / td / fd: a number or (n,) per frame (the cost of
    switching INTO that frame); start_dead: a bonus for D at the first frame. Unrolled plain Python."""
    L = np.asarray(ll, dtype=np.float64).tolist()
    n = len(L)
    per_frame = not (np.isscalar(sw) and np.isscalar(td) and np.isscalar(fd))
    if per_frame:
        SW = (np.full(n, sw, dtype=np.float64) if np.isscalar(sw) else np.asarray(sw, dtype=np.float64)).tolist()
        TD = (np.full(n, td, dtype=np.float64) if np.isscalar(td) else np.asarray(td, dtype=np.float64)).tolist()
        FD = (np.full(n, fd, dtype=np.float64) if np.isscalar(fd) else np.asarray(fd, dtype=np.float64)).tolist()
    else:
        c_sw, c_td, c_fd = float(sw), float(td), float(fd)
    b0, b1, b2 = bytearray(n), bytearray(n), bytearray(n)
    s0, s1, s2 = L[0]
    s2 = s2 + start_dead
    for t in range(1, n):
        if per_frame:
            c_sw, c_td, c_fd = SW[t], TD[t], FD[t]
        l0, l1, l2 = L[t]
        # -> H: from H (0), A (-sw), D (-fd)
        best = s0 + 0.0
        bi = 0
        v = s1 + -c_sw
        if v > best:
            best, bi = v, 1
        v = s2 + -c_fd
        if v > best:
            best, bi = v, 2
        n0 = best + l0
        b0[t] = bi
        # -> A: from H (-sw), A (0), D (-fd)
        best = s0 + -c_sw
        bi = 0
        v = s1 + 0.0
        if v > best:
            best, bi = v, 1
        v = s2 + -c_fd
        if v > best:
            best, bi = v, 2
        n1 = best + l1
        b1[t] = bi
        # -> D: from H (-td), A (-td), D (0)
        best = s0 + -c_td
        bi = 0
        v = s1 + -c_td
        if v > best:
            best, bi = v, 1
        v = s2 + 0.0
        if v > best:
            best, bi = v, 2
        n2 = best + l2
        b2[t] = bi
        s0, s1, s2 = n0, n1, n2
    path = np.empty(n, dtype=np.int8)
    k = int(np.argmax([s0, s1, s2]))
    backs = (b0, b1, b2)
    for t in range(n - 1, 0, -1):
        path[t] = k
        k = backs[k][t]
    path[0] = k
    return path


def decode_hmm(feat, p_dead, halves, n: int, params: decode.Params, p_bnd=None, alpha: float = 0.0,
               start: bool = False, sw_t=None, tags=None) -> np.ndarray:
    """The production HMM with this P(dead) (None: v0's speed rule). feat = features.build(..., p_home=...);
    tags: Smart Tagging team votes (smart_tagging.evidence), as decode.decode -- BAR-ATH only."""
    f = feat if p_dead is None else feat.assign(p_dead_model=np.asarray(p_dead, dtype=np.float64))
    if p_dead is None and "p_dead_model" in f:
        f = f.drop(columns=["p_dead_model"])
    ll = decode.emissions(f, params, tags)
    td, fd = params.to_dead_cost, params.from_dead_cost
    if p_bnd is not None and alpha > 0:
        pb = np.clip(np.nan_to_num(np.asarray(p_bnd, dtype=np.float64), nan=0.0), 0.0, 1.0)
        td, fd = td * (1.0 - alpha * pb), fd * (1.0 - alpha * pb)
    lab = np.full(n, "X", dtype="<U1")
    for _, a, b in halves:
        sl = slice(a, b + 1)
        pick = (lambda v: v if np.isscalar(v) else v[sl])
        path = viterbi3(ll[sl], params.switch_cost if sw_t is None else sw_t[sl], pick(td), pick(fd),
                        params.from_dead_cost if start else 0.0)
        lab[sl] = np.array(LABELS)[path]
    return lab


def _fill(lab: np.ndarray, s: int, e: int, lo: int, hi: int):
    """Relabel lab[s:e] (a D run) from its alive neighbours inside [lo, hi): the same team on both sides
    -> that team; two teams -> split at the middle; one side only -> that side."""
    left = lab[s - 1] if s - 1 >= lo and lab[s - 1] in ("H", "A") else None
    right = lab[e] if e < hi and lab[e] in ("H", "A") else None
    if left and right:
        if left == right:
            lab[s:e] = left
        else:
            m = (s + e) // 2
            lab[s:m], lab[m:e] = left, right
    elif left or right:
        lab[s:e] = left or right


def min_dead(lab: np.ndarray, halves, fps: float, min_s: float) -> np.ndarray:
    out = np.asarray(lab).copy()
    m = int(round(min_s * fps))
    for _, a, b in halves:
        seg = out[a:b + 1]
        for s, e in runs_of(seg, "D"):
            if e - s < m:
                _fill(seg, s, e, 0, len(seg))
        out[a:b + 1] = seg
    return out


def veto(lab: np.ndarray, rr: dict, halves, fps: float, max_s: float = 6.0, support: float = 0.5) -> np.ndarray:
    out = np.asarray(lab).copy()
    pr = np.nan_to_num(rr["p_dead"], nan=0.5)
    for _, a, b in halves:
        seg = out[a:b + 1]
        prs = pr[a:b + 1]
        for s, e in runs_of(seg, "D"):
            if (e - s) <= max_s * fps and np.mean(prs[s:e] >= 0.5) < support:
                _fill(seg, s, e, 0, len(seg))
        out[a:b + 1] = seg
    return out


def snap(lab: np.ndarray, rr: dict, halves, fps: float, max_s: float = 1.0) -> np.ndarray:
    """Move every restart (D -> alive) to the nearest rr['restart_cand'] frame within +-max_s, every stop
    (alive -> D) to the nearest rr['stop_cand'] frame; a move never crosses another boundary."""
    out = np.asarray(lab).copy()
    w = int(round(max_s * fps))
    rc = np.flatnonzero(rr["restart_cand"])
    sc = np.flatnonzero(rr["stop_cand"])

    def nearest(c, t, lo, hi):
        if not len(c):
            return None
        i = np.searchsorted(c, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(c) and lo < c[j] < hi and abs(c[j] - t) <= w:
                if best is None or abs(c[j] - t) < abs(best - t):
                    best = int(c[j])
        return best

    for _, a, b in halves:
        seg = out[a:b + 1].copy()
        ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = [0, *ch.tolist(), len(seg)]
        for k in range(1, len(bounds) - 1):
            t = bounds[k]
            lo, hi = bounds[k - 1], bounds[k + 1]
            prev, nxt = seg[t - 1], seg[t]
            if prev == "D" and nxt in ("H", "A"):
                c = nearest(rc, a + t, a + lo, a + hi)
                if c is not None:
                    c -= a
                    if c < t:
                        out[a + c:a + t] = nxt
                    else:
                        out[a + t:a + c] = "D"
            elif prev in ("H", "A") and nxt == "D":
                c = nearest(sc, a + t, a + lo, a + hi)
                if c is not None:
                    c -= a
                    if c < t:
                        out[a + c:a + t] = "D"
                    else:
                        out[a + t:a + c] = prev
    return out


VARIANT_RE = re.compile(r"^(hmm|start|veto|tags|bnd[0-9.]+|min[0-9.]+|snap[0-9.]+)$")


def parse_variant(v: str) -> list[str]:
    parts = [p for p in v.split("+") if p]
    bad = [p for p in parts if not VARIANT_RE.match(p)]
    if bad or not parts:
        raise SystemExit(f"unknown decoder variant {v!r} (hmm, start, veto, tags, bnd<a>, min<s>, snap<s>, joined by +)")
    return parts


def decode_variant(variant: str, feat, p_dead, halves, n: int, fps: float, params: decode.Params, rr: dict,
                   p_bnd=None, tag_bonus_=None, tag_w: float = 0.0, sw_t=None, tags=None):
    """Labels of one decoder variant; None when it does not apply (bnd without a boundary, tags without
    windows)."""
    parts = parse_variant(variant)
    alpha, start = 0.0, False
    p = p_dead
    for q in parts:
        if q.startswith("bnd"):
            if p_bnd is None:
                return None
            alpha = float(q[3:])
        elif q == "start":
            start = True
        elif q == "tags":
            if tag_bonus_ is None or p is None:
                return None
            p = with_prior(p, tag_bonus_, tag_w)
    lab = decode_hmm(feat, p, halves, n, params, p_bnd, alpha, start, sw_t, tags)
    for q in parts:                                   # post-processing, in the order given
        if q.startswith("min"):
            lab = min_dead(lab, halves, fps, float(q[3:]))
        elif q.startswith("snap"):
            lab = snap(lab, rr, halves, fps, float(q[4:]))
        elif q == "veto":
            lab = veto(lab, rr, halves, fps)
    return lab


# ------------------------------------------------------------------------------------ diagnostics

DEAD_BUCKETS_S = (0, 5, 12, 30, 90, math.inf)


def stoppage_stats(gt: np.ndarray, pred: np.ndarray, halves, fps: float) -> dict:
    """Stoppage-level view of the alive layer (a diagnostic, not the ranking metric):
    recall of true stoppages by length (found = predicted D covers >= 50 % of it), spurious predicted
    stoppages (< 20 % of them on true D) per 90 min, median |start / end error| of the found ones (s),
    share of them within 1 s, D share predicted / true."""
    found_by = {k: [0, 0] for k in range(len(DEAD_BUCKETS_S) - 1)}
    start_err, end_err = [], []
    spurious = 0
    minutes = 0.0
    d_true = d_pred = tot = 0
    for _, a, b in halves:
        g = np.asarray(gt[a:b + 1])
        p = np.asarray(pred[a:b + 1])
        ok = np.isin(g, ["H", "A", "D"])
        minutes += ok.sum() / fps / 60
        d_true += int((g[ok] == "D").sum())
        d_pred += int((p[ok] == "D").sum())
        tot += int(ok.sum())
        pd_ = p == "D"
        gd = g == "D"
        pruns = runs_of(p, "D")
        for s, e in runs_of(g, "D"):
            L = (e - s) / fps
            k = int(np.searchsorted(DEAD_BUCKETS_S, L, side="right") - 1)
            k = min(max(k, 0), len(DEAD_BUCKETS_S) - 2)
            found = pd_[s:e].mean() >= 0.5
            found_by[k][1] += 1
            if found:
                found_by[k][0] += 1
                best, ov = None, 0
                for ps, pe in pruns:
                    o = min(e, pe) - max(s, ps)
                    if o > ov:
                        best, ov = (ps, pe), o
                if best is not None:
                    start_err.append((best[0] - s) / fps)
                    end_err.append((best[1] - e) / fps)
        for ps, pe in pruns:
            if gd[ps:pe].mean() < 0.2:
                spurious += 1
    out = {}
    for k, (f, t) in found_by.items():
        lo, hi = DEAD_BUCKETS_S[k], DEAD_BUCKETS_S[k + 1]
        out[f"rec_{lo:g}-{hi:g}s"] = f / t if t else None
        out[f"n_{lo:g}-{hi:g}s"] = t
    out["spurious_per90"] = spurious / max(minutes, 1e-9) * 90
    se, ee = np.abs(start_err), np.abs(end_err)
    out["start_med_s"] = float(np.median(se)) if len(se) else None
    out["end_med_s"] = float(np.median(ee)) if len(ee) else None
    out["start_bias_s"] = float(np.median(start_err)) if len(se) else None
    out["end_bias_s"] = float(np.median(end_err)) if len(ee) else None
    out["start_1s"] = float(np.mean(se <= 1.0)) if len(se) else None
    out["end_1s"] = float(np.mean(ee <= 1.0)) if len(ee) else None
    out["d_share_true"] = d_true / max(tot, 1)
    out["d_share_pred"] = d_pred / max(tot, 1)
    return out
