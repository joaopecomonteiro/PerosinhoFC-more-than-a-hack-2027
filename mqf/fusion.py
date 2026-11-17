"""mqf.fusion -- the last stage: P(dead) x P(home | alive) [x boundaries] -> one H/A/D per frame.

Compares structured decoders under the official metric with nested leave-one-game-out (LOGO)
selection, so every number in the report is held out. v2 (16 Nov): the evidence is the
PRODUCTION emission model (decode.emissions with FINAL's settings), so `hmm3_bnd` config 0 is
FINAL bit for bit and every other config is a perturbation of it; columns, averaging and the
adoption rule are final_check's (#38).

INPUT CONTRACT (one .npz per column x game x noise seed; other games only, never BAR-ATH):
    <preds>/<column>/<game>__s<seed>.npz        column = <condition>_<ball mode>, e.g. bar_exact
      y            int8  (T,)   ground truth codes 0=H 1=A 2=D 3=X
      fps          float ()     native frame rate of the bundle
      source       str   ()     idsse / metrica / pff (means are per source, then over sources)
      halves       int   (k,2)  decode windows, inclusive [first, last] (spec.halves)
      p_dead       float (T,)   P(dead) exactly as FINAL uses it (round(mean(asf, v8), 1e-4))
      p_dead_asf / p_dead_v8    the two dead-ball models separately (optional: fusion knobs)
      p_home       float (T,)   owner-TCN ensemble P(home | alive), as FINAL uses it
      p_home_seeds float (S,T)  the S TCN seeds separately (optional: logit mean, disagreement)
      p_home_v1    float (T,)   who_v1 P(home) (optional: a second possession expert)
      p_bnd        float (T,)   TCN H <-> A boundary head (FINAL: switch cost x (1 - 0.5 P))
      p_bnd_dead   float (T,)   asf alive <-> dead boundary head (optional: dead costs)
      ball_out, ball_filled     bool (T,)  production emission context (ball seen off the pitch ->
                                P(dead) 0.97; imputed ball -> who weight w_filled)
      tag_vote     float (T,)   Smart Tagging team votes H - A (BAR-ATH only; 0 / absent elsewhere)
      feat_*, params_json, who_alpha   what the production decoder needs (--baseline-hook)
      pred_final   int8  (T,)   the labels final_check scored for FINAL (the sanity anchor)
    mqf.fusion_export writes all of it from the final_check runs.

DECODERS (each decodes inside every window of `halves`, i.e. per half, as production):
    argmax     per-frame argmax, no temporal model (floor)
    hmm3       3-state Viterbi, production arithmetic and tie-breaking, no boundary
    hmm3_bnd   hmm3 + H<->A cost x (1 - alpha P(bnd)) [+ dead costs x (1 - alpha_d P(bnd_dead))]
               -- config 0 = FINAL
    hsmm_soft  explicit-duration (semi-Markov) Viterbi: soft minimum duration per state
    hsmm_emp   semi-Markov Viterbi, duration prior from the OTHER games' GT (leave-self-out)
    hier       alive/dead first, then the team inside each alive run
    asrf       cut alive runs at boundary-head peaks, label each chunk by likelihood vote
    fb_hsmm    forward-backward marginals of hmm3, then semi-Markov Viterbi on log-marginals
    minrun     length rule only: every run (H, A or D) >= min_s (0.5 s); no switch costs, no boundary
    stoppage   length rules only (the submission's decoder since 17 Nov, make_submission.STOPPAGE): every run
               >= min_s, every dead-ball run >= min_dead_s, a possession with a dead ball on BOTH sides
               >= min_between_s (instead of min_s); no switch costs, no boundary. Exact 7-state semi-Markov
               Viterbi (stoppage_path). Centre 0.5 / 2 / 2 s = the viewer lane; the submission's lengths are tuned
    prod       the production decoder from the same arrays (--baseline-hook mqf.decode:fusion_baseline)
    final      the exported FINAL labels themselves (baseline when pred_final is present)

USAGE (from the repo root):
    python -m mqf.fusion synth --out harness_results/fusion_synth           # fake data, smoke
    python -m mqf.fusion run --preds harness_results/fusion_synth --quick   # minutes
    python -m mqf.fusion run --preds <dir> --jobs 8 --tag f17               # full budget
    python run_fusion.py ...                                                # the whole sequence
Outputs harness_results/fusion_<tag>_{report.md,scores.csv,best.json}: aggregates only.

Deterministic: seeded config samples, strict-> tie-breaks, float64, no threads inside a decode.
numba is used when installed (pinned in requirements.txt; ~50x on the semi-Markov kernels); the
numpy fallback gives identical paths.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import json
import math
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

H, A, D, X = 0, 1, 2, 3
EPS = 1e-4              # decode.emissions' eps (P(dead), P(home) never closer than this to 0 / 1)
REF_FPS = 25.0          # with fps_scale, emissions are "nats per 1/25 s" (BAR-ATH's rate)
NEG = -1e300
BALL_OUT_P = 0.97       # decode.emissions: a TRACKED ball off the pitch

try:  # optional accelerator
    import numba

    _njit = numba.njit(cache=True, nogil=True)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def _njit(f):
        return f


# ============================================================================================
# kernels. Costs: C[i, j] is the cost of i -> j; Bm[i, j] = 0 fixed, 1 times mult[t, 0]
# (H <-> A boundary), 2 times mult[t, 1] (alive <-> dead boundary); the cost of switching INTO t.
# ============================================================================================

def _vit_loops(E, C, Bm, mult):
    """First-order Viterbi with alive.combine.viterbi3's float operations and tie-breaking:
    cand_i = score_i + (-cost(i -> j)), first max over i = 0..K-1 (stay included), + E[t, j]."""
    T, K = E.shape
    back = np.zeros((T, K), np.int8)
    s = np.empty(K)
    nxt = np.empty(K)
    for j in range(K):
        s[j] = E[0, j]
    for t in range(1, T):
        for j in range(K):
            best = 0.0
            bi = 0
            for i in range(K):
                if i == j:
                    c = 0.0
                else:
                    k = Bm[i, j]
                    c = C[i, j] if k == 0 else C[i, j] * mult[t, k - 1]
                v = s[i] + (-c)
                if i == 0 or v > best:
                    best = v
                    bi = i
            nxt[j] = best + E[t, j]
            back[t, j] = bi
        for j in range(K):
            s[j] = nxt[j]
    out = np.empty(T, np.int64)
    k = 0
    for j in range(1, K):
        if s[j] > s[k]:
            k = j
    for t in range(T - 1, 0, -1):
        out[t] = k
        k = back[t, k]
    out[0] = k
    return out


def _vit_numpy(E, C, Bm, mult):
    T, K = E.shape
    back = np.zeros((T, K), np.int8)
    s = E[0].copy()
    eye = np.eye(K, dtype=bool)
    Bi = Bm.astype(np.int64)
    for t in range(1, T):
        m = np.concatenate(([1.0], mult[t]))
        Ct = np.where(eye, 0.0, np.where(Bi == 0, C, C * m[Bi]))
        cand = s[:, None] + (-Ct)
        bi = cand.argmax(0)                       # first max = lowest i, as the loops
        back[t] = bi
        s = cand[bi, np.arange(K)] + E[t]
    out = np.empty(T, np.int64)
    k = int(s.argmax())
    for t in range(T - 1, 0, -1):
        out[t] = k
        k = back[t, k]
    out[0] = k
    return out


def _smk_loops(E, C, Bm, mult, pen, M):
    """Semi-Markov Viterbi with a soft minimum-duration penalty (loops; jitted when numba exists).

    E (T,K) per-frame log scores. pen (K,M) penalty for a segment of length d=1..M, NON-INCREASING
    in d (stack_pens enforces it); longer segments pay pen[:, M-1] (0 by construction), so the
    recursion is O(T K^2 + T K M) and exact (a segment of length L pays pen[min(L, M) - 1]).
    """
    T, K = E.shape
    S = np.zeros((T + 1, K))
    for t in range(T):
        for j in range(K):
            S[t + 1, j] = S[t, j] + E[t, j]
    delta = np.full((T, K), NEG)
    G = np.full((T, K), NEG)
    GI = np.full((T, K), -1, np.int64)
    bp = np.zeros((T, K), np.int64)
    for t in range(T):
        for j in range(K):
            if t == 0:
                G[0, j] = 0.0
            else:
                best = NEG
                bi = -1
                for i in range(K):
                    if i == j:
                        continue
                    k = Bm[i, j]
                    c = C[i, j] if k == 0 else C[i, j] * mult[t, k - 1]
                    v = delta[t - 1, i] - c
                    if v > best:
                        best = v
                        bi = i
                G[t, j] = best
                GI[t, j] = bi
        for j in range(K):
            best = delta[t - 1, j] + E[t, j] if t > 0 else NEG
            arg = -1
            dmax = M if M < t + 1 else t + 1
            for d in range(1, dmax + 1):
                s = t - d + 1
                v = G[s, j] + (S[t + 1, j] - S[s, j]) - pen[j, d - 1]
                if v > best:
                    best = v
                    arg = d
            delta[t, j] = best
            bp[t, j] = arg
    out = np.empty(T, np.int64)
    j = 0
    best = delta[T - 1, 0]
    for k in range(1, K):
        if delta[T - 1, k] > best:
            best = delta[T - 1, k]
            j = k
    t = T - 1
    while t >= 0:
        a = bp[t, j]
        if a == -1:
            out[t] = j
            t -= 1
        else:
            s = t - a + 1
            for u in range(s, t + 1):
                out[u] = j
            i = GI[s, j]
            t = s - 1
            j = i
    return out


def _cost_t(C, Bm, mult_t):
    K = len(C)
    m = np.concatenate(([1.0], mult_t))
    Bi = Bm.astype(np.int64)
    return np.where(np.eye(K, dtype=bool), 0.0, np.where(Bi == 0, C, C * m[Bi]))


def _smk_numpy(E, C, Bm, mult, pen, M):
    """Same recursion and tie-breaks as _smk_loops, vectorised over d and i (numba-free path)."""
    T, K = E.shape
    S = np.vstack([np.zeros((1, K)), np.cumsum(E, 0)])
    delta = np.full((T, K), NEG)
    G = np.full((T, K), NEG)
    GI = np.full((T, K), -1, np.int64)
    bp = np.zeros((T, K), np.int64)
    eye = np.eye(K, dtype=bool)
    ar = np.arange(K)
    for t in range(T):
        if t == 0:
            G[0] = 0.0
        else:
            cand = delta[t - 1][:, None] - _cost_t(C, Bm, mult[t])
            cand[eye] = -np.inf
            gi = cand.argmax(0)
            GI[t] = gi
            G[t] = cand[gi, ar]
        L = min(M, t + 1)
        s_idx = t - np.arange(L)
        vals = G[s_idx] - S[s_idx] - pen[:, :L].T + S[t + 1]
        k = vals.argmax(0)
        enter = vals[k, ar]
        stay = delta[t - 1] + E[t] if t > 0 else np.full(K, NEG)
        use_enter = enter > stay
        delta[t] = np.where(use_enter, enter, stay)
        bp[t] = np.where(use_enter, k + 1, -1)
    out = np.empty(T, np.int64)
    j = int(delta[T - 1].argmax())
    t = T - 1
    while t >= 0:
        a = bp[t, j]
        if a == -1:
            out[t] = j
            t -= 1
        else:
            s = t - a + 1
            out[s:t + 1] = j
            i = GI[s, j]
            t = s - 1
            j = i
    return out


def _fb_loops(E, C, Bm, mult):
    """Forward-backward log-marginals of the linear-chain model with the same potentials."""
    T, K = E.shape
    la = np.empty((T, K))
    lb = np.zeros((T, K))
    tmp = np.empty(K)
    for j in range(K):
        la[0, j] = E[0, j]
    for t in range(1, T):
        for j in range(K):
            m = NEG
            for i in range(K):
                if i == j:
                    c = 0.0
                else:
                    k = Bm[i, j]
                    c = C[i, j] if k == 0 else C[i, j] * mult[t, k - 1]
                tmp[i] = la[t - 1, i] - c
                if tmp[i] > m:
                    m = tmp[i]
            s = 0.0
            for i in range(K):
                s += math.exp(tmp[i] - m)
            la[t, j] = E[t, j] + m + math.log(s)
    for t in range(T - 2, -1, -1):
        for i in range(K):
            m = NEG
            for j in range(K):
                if i == j:
                    c = 0.0
                else:
                    k = Bm[i, j]
                    c = C[i, j] if k == 0 else C[i, j] * mult[t + 1, k - 1]
                tmp[j] = lb[t + 1, j] + E[t + 1, j] - c
                if tmp[j] > m:
                    m = tmp[j]
            s = 0.0
            for j in range(K):
                s += math.exp(tmp[j] - m)
            lb[t, i] = m + math.log(s)
    post = la + lb
    for t in range(T):
        m = NEG
        for j in range(K):
            if post[t, j] > m:
                m = post[t, j]
        s = 0.0
        for j in range(K):
            s += math.exp(post[t, j] - m)
        z = m + math.log(s)
        for j in range(K):
            post[t, j] -= z
    return post


def _fb_numpy(E, C, Bm, mult):
    T, K = E.shape
    la = np.empty((T, K))
    lb = np.zeros((T, K))
    la[0] = E[0]
    for t in range(1, T):
        la[t] = E[t] + np.logaddexp.reduce(la[t - 1][:, None] - _cost_t(C, Bm, mult[t]), axis=0)
    for t in range(T - 2, -1, -1):
        lb[t] = np.logaddexp.reduce((lb[t + 1] + E[t + 1])[None, :] - _cost_t(C, Bm, mult[t + 1]),
                                    axis=1)
    post = la + lb
    return post - np.logaddexp.reduce(post, axis=1, keepdims=True)


if HAVE_NUMBA:
    _vit_fast = _njit(_vit_loops)
    _smk_fast = _njit(_smk_loops)
    _fb_fast = _njit(_fb_loops)
else:  # pragma: no cover
    _vit_fast = _vit_numpy
    _smk_fast = _smk_numpy
    _fb_fast = _fb_numpy


def _prep(E, C, Bm, mult):
    """Normalise: Bm None / bool (-> class 1) / int classes; mult None / (T,) (-> class 1) / (T,2)."""
    E = np.ascontiguousarray(E, dtype=np.float64)
    T, K = E.shape
    C = np.ascontiguousarray(C, dtype=np.float64)
    if Bm is None:
        Bm = np.zeros((K, K), np.int8)
    Bm = np.ascontiguousarray(np.asarray(Bm).astype(np.int8))
    if mult is None:
        mult = np.ones((T, 2))
    mult = np.asarray(mult, dtype=np.float64)
    if mult.ndim == 1:
        mult = np.stack([mult, np.ones(T)], 1)
    return E, C, Bm, np.ascontiguousarray(mult)


def viterbi(E, C, Bm=None, mult=None):
    E, C, Bm, mult = _prep(E, C, Bm, mult)
    return _vit_fast(E, C, Bm, mult)


def semimarkov_viterbi(E, C, Bm=None, mult=None, pen=None):
    E, C, Bm, mult = _prep(E, C, Bm, mult)
    K = E.shape[1]
    pen = np.zeros((K, 1)) if pen is None else np.ascontiguousarray(pen, dtype=np.float64)
    return _smk_fast(E, C, Bm, mult, pen, pen.shape[1])


def forward_backward(E, C, Bm=None, mult=None):
    E, C, Bm, mult = _prep(E, C, Bm, mult)
    return _fb_fast(E, C, Bm, mult)


# ============================================================================================
# evidence -> emissions (decode.emissions, learned P(dead) + learned P(home), dead_who="max")
# ============================================================================================

def logit(p, eps=EPS):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _sig(x):                                   # decode._sig
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _clip_logit(p, clip):                      # decode.emissions on a learned P(home)
    pc = np.clip(np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.5), clip, 1 - clip)
    return np.log(pc) - np.log(1 - pc)


def windows(g):
    """Decode windows, half-open: the halves when given (production decodes per half), else the
    graded runs of y."""
    if "halves" in g:
        return [(int(a), int(b) + 1) for a, b in np.asarray(g["halves"]).reshape(-1, 2)]
    return graded_runs(g["y"])


def graded_runs(y):
    gr = (np.asarray(y) != X).astype(np.int8)
    e = np.diff(np.concatenate(([0], gr, [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def _dead_p(g, cfg):
    if cfg.get("dead_mix", "prob_mean") == "prob_mean":
        if "p_dead" in g:
            p = np.asarray(g["p_dead"], dtype=np.float64)
        else:
            p = np.round(0.5 * (g["p_dead_asf"] + g["p_dead_v8"]), 4)
    else:                                       # log-linear pool = logistic stacking, tied scale
        s = cfg.get("v8_share", 0.5)
        p = _sig((1 - s) * logit(g["p_dead_asf"]) + s * logit(g["p_dead_v8"]))
    wd, bd = cfg.get("wd", 1.0), cfg.get("bd", 0.0)
    if wd != 1.0 or bd != 0.0:                  # temperature + bias of the dead stream
        p = np.where(np.isnan(p), np.nan, _sig(wd * logit(p) + bd))
    p = np.where(np.isnan(p), 0.5, p)           # outside the halves only (export counts it)
    if "ball_out" in g:
        p = np.where(np.asarray(g["ball_out"], bool), BALL_OUT_P, p)
    return np.clip(p, EPS, 1 - EPS)


def _home_logit(g, cfg):
    clip = cfg.get("who_clip", 0.02)
    mix = cfg.get("home_mix", "prob_mean")
    seeds = np.asarray(g["p_home_seeds"], dtype=np.float64) if "p_home_seeds" in g else None
    if mix == "prob_mean" or seeds is None:
        lh = _clip_logit(g["p_home"] if "p_home" in g else np.round(seeds.mean(0), 4), clip)
    else:
        lh = np.mean([_clip_logit(s, clip) for s in seeds], axis=0)
    wh = cfg.get("wh", 1.0)
    w = wh
    if cfg.get("kappa", 0.0) > 0 and seeds is not None:   # seeds disagree -> trust the team less
        w = wh / (1.0 + cfg["kappa"] * np.std([_clip_logit(s, clip) for s in seeds], axis=0))
    v = cfg.get("v1_share", 0.0)
    if v > 0 and "p_home_v1" in g:
        lh = (1 - v) * lh + v * _clip_logit(g["p_home_v1"], clip)
    if not (np.isscalar(w) and w == 1.0):
        lh = w * lh
    if "tag_vote" in g:                          # Smart Tagging votes (BAR-ATH), scaled with wh; dtype kept
        lh = lh + (wh * cfg.get("w_tags", 0.4)) * np.asarray(g["tag_vote"])   # (float32, as decode.emissions)
    return lh


def evidence(g, cfg):
    """Per-frame weighted log-evidence, decode.emissions' terms: alive / dead (P(dead) x w_ad),
    home / away / dwho (P(home) x w_who; dwho = the better team term, dead_who="max")."""
    pd = _dead_p(g, cfg)
    p_h = np.clip(_sig(_home_logit(g, cfg)), EPS, 1 - EPS)
    T = len(pd)
    w_ad = np.ones(T)                            # FINAL: w_seen = w_dead = 1
    filled = np.asarray(g["ball_filled"], bool) if "ball_filled" in g else np.zeros(T, bool)
    w_who = 1.0 * np.where(filled, cfg.get("w_filled", 0.4), 1.0)
    l1, l0 = np.log(1 - pd), np.log(pd)
    lh, la = np.log(p_h), np.log(1 - p_h)
    ev = {"alive": l1 * w_ad, "dead": l0 * w_ad, "home": lh * w_who, "away": la * w_who,
          "dwho": np.maximum(lh, la) * w_who}
    if cfg.get("fps_scale", False):
        sc = REF_FPS / float(g["fps"])
        ev = {k: v * sc for k, v in ev.items()}
    mult = np.ones((T, 2))
    alpha = cfg.get("alpha", 0.0)
    if alpha > 0 and "p_bnd" in g:
        pb = np.nan_to_num(np.asarray(g["p_bnd"], dtype=np.float64), nan=0.0)
        mult[:, 0] = 1.0 - alpha * np.clip(pb, 0.0, 1.0)
    ad = cfg.get("alpha_d", 0.0)
    if ad > 0 and "p_bnd_dead" in g:
        pb = np.nan_to_num(np.asarray(g["p_bnd_dead"], dtype=np.float64), nan=0.0)
        mult[:, 1] = 1.0 - ad * np.clip(pb, 0.0, 1.0)
    ev["mult"] = mult
    return ev


def emissions3(ev):
    return np.stack([ev["alive"] + ev["home"], ev["alive"] + ev["away"], ev["dead"] + ev["dwho"]], 1)


def costs3(cfg):
    cha, cad, cda = cfg.get("c_ha", 18.0), cfg.get("c_ad", 10.0), cfg.get("c_da", 10.0)
    C = np.array([[0.0, cha, cad], [cha, 0.0, cad], [cda, cda, 0.0]])
    Bm = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]], np.int8)
    return C, Bm


def soft_pen(m_frames, lam, power):
    """lam * (1 - d/m)^power for d < m, 0 from m on. Shape (M,) with M = max(1, m)."""
    m = max(1, int(round(m_frames)))
    d = np.arange(1, m + 1, dtype=np.float64)
    return lam * np.clip(1.0 - d / m, 0.0, None) ** power


def stack_pens(rows):
    """(K, M) penalties, each row made non-increasing in d (the kernel's exactness condition) and
    padded with its last value (0 for soft_pen / emp_pen)."""
    M = max(len(r) for r in rows)
    out = np.zeros((len(rows), M))
    for k, r in enumerate(rows):
        r = np.minimum.accumulate(np.asarray(r, dtype=np.float64))
        out[k, :len(r)] = r
        out[k, len(r):] = r[-1] if len(r) else 0.0
    return out


def emp_pen(prior, key, fps, lam, cap_s):
    """Empirical duration prior -> penalty: lam * (log f(mode) - log f(d)) for d < mode."""
    centers, logf = np.asarray(prior[key]["centers_s"]), np.asarray(prior[key]["logf"])
    mode_s = min(float(centers[int(np.argmax(logf))]), cap_s)
    m = max(1, int(round(mode_s * fps)))
    d_s = np.arange(1, m + 1) / fps
    lf = np.interp(np.log10(d_s), np.log10(centers), logf)
    pen = lam * np.clip(logf.max() - lf, 0.0, None) * (d_s < mode_s)
    return np.minimum.accumulate(pen)


# ============================================================================================
# decoders: each returns codes for one window [a, b)
# ============================================================================================

def _sl(ev, a, b):
    return {k: v[a:b] for k, v in ev.items()}


def _dec_argmax(g, ev, cfg, a, b):
    return emissions3(_sl(ev, a, b)).argmax(1)


def _dec_hmm3(g, ev, cfg, a, b):
    C, Bm = costs3(cfg)
    return viterbi(emissions3(_sl(ev, a, b)), C, Bm, ev["mult"][a:b])


def _pens3(g, cfg):
    fps = float(g["fps"])
    p = cfg.get("power", 1.0)
    sp = soft_pen(cfg.get("m_spell", 1.0) * fps, cfg.get("lam_spell", 8.0), p)
    dp = soft_pen(cfg.get("m_dead", 2.0) * fps, cfg.get("lam_dead", 8.0), p)
    return stack_pens([sp, sp, dp])


def _dec_hsmm_soft(g, ev, cfg, a, b):
    C, Bm = costs3(cfg)
    return semimarkov_viterbi(emissions3(_sl(ev, a, b)), C, Bm, ev["mult"][a:b], _pens3(g, cfg))


def _dec_hsmm_emp(g, ev, cfg, a, b):
    prior = g.get("_dur_prior")
    if prior is None:
        raise ValueError("hsmm_emp needs a duration prior (the suite attaches it)")
    fps = float(g["fps"])
    sp = emp_pen(prior, "spell", fps, cfg.get("lam_spell", 1.0), cfg.get("cap_spell", 4.0))
    dp = emp_pen(prior, "dead", fps, cfg.get("lam_dead", 1.0), cfg.get("cap_dead", 6.0))
    C, Bm = costs3(cfg)
    return semimarkov_viterbi(emissions3(_sl(ev, a, b)), C, Bm, ev["mult"][a:b],
                              stack_pens([sp, sp, dp]))


def _dec_hier(g, ev, cfg, a, b):
    fps = float(g["fps"])
    p = cfg.get("power", 1.0)
    E1 = np.stack([ev["alive"][a:b], ev["dead"][a:b]], 1)
    C1 = np.array([[0.0, cfg.get("c_ad", 10.0)], [cfg.get("c_da", 10.0), 0.0]])
    Bm1 = np.array([[0, 2], [2, 0]], np.int8)
    pen1 = stack_pens([soft_pen(cfg.get("m_alive", 1.0) * fps, cfg.get("lam_alive", 8.0), p),
                       soft_pen(cfg.get("m_dead", 2.0) * fps, cfg.get("lam_dead", 8.0), p)])
    ad = semimarkov_viterbi(E1, C1, Bm1, ev["mult"][a:b], pen1)
    out = np.full(b - a, D, np.int64)
    alive = (ad == 0).astype(np.int8)
    e = np.diff(np.concatenate(([0], alive, [0])))
    C2 = np.array([[0.0, cfg.get("c_ha", 18.0)], [cfg.get("c_ha", 18.0), 0.0]])
    Bm2 = np.array([[0, 1], [1, 0]], np.int8)
    pen2 = stack_pens([soft_pen(cfg.get("m_spell", 1.0) * fps, cfg.get("lam_spell", 8.0), p)] * 2)
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        E2 = np.stack([ev["home"][a + s:a + t], ev["away"][a + s:a + t]], 1)
        out[s:t] = semimarkov_viterbi(E2, C2, Bm2, ev["mult"][a + s:a + t], pen2)
    return out


def _peaks(bnd, fps, smooth_s, theta, min_gap_s):
    w = max(1, int(round(smooth_s * fps)))
    b = np.convolve(bnd, np.ones(w) / w, mode="same") if w > 1 else bnd
    if len(b) < 3:
        return np.zeros(0, np.int64)
    loc = np.flatnonzero((b[1:-1] >= b[:-2]) & (b[1:-1] > b[2:]) & (b[1:-1] >= theta)) + 1
    order = loc[np.argsort(-b[loc], kind="stable")]
    gap = max(1, int(round(min_gap_s * fps)))
    taken = np.zeros(len(b), bool)
    keep = []
    for p in order:                      # non-maximum suppression, highest first
        lo, hi = max(0, p - gap), min(len(b), p + gap + 1)
        if not taken[lo:hi].any():
            keep.append(p)
            taken[p] = True
    return np.sort(np.asarray(keep, np.int64))


def _dec_asrf(g, ev, cfg, a, b):
    if "p_bnd" not in g:
        raise ValueError("asrf needs p_bnd")
    base = _dec_hmm3(g, ev, cfg, a, b)
    fps = float(g["fps"])
    pb = np.nan_to_num(np.asarray(g["p_bnd"][a:b], dtype=np.float64), nan=0.0)
    pk = _peaks(pb, fps, cfg.get("smooth_s", 0.2), cfg.get("theta", 0.5), cfg.get("min_gap_s", 0.5))
    margin = ev["home"][a:b] - ev["away"][a:b]
    out = base.copy()
    alive = (base != D).astype(np.int8)
    e = np.diff(np.concatenate(([0], alive, [0])))
    union = cfg.get("cuts", "peaks") == "union"
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        cuts = pk[(pk > s) & (pk < t)].tolist()
        if union:
            sw = np.flatnonzero(base[s + 1:t] != base[s:t - 1]) + s + 1
            cuts = sorted(set(cuts) | set(sw.tolist()))
        edges = [s] + cuts + [t]
        for u, v in zip(edges[:-1], edges[1:]):
            out[u:v] = H if margin[u:v].sum() >= 0 else A
    return out


def _dec_fb_hsmm(g, ev, cfg, a, b):
    C, Bm = costs3(cfg)
    post = forward_backward(emissions3(_sl(ev, a, b)), C, Bm, ev["mult"][a:b])
    c2 = cfg.get("c2", 2.0)
    C2 = np.array([[0.0, c2, c2], [c2, 0.0, c2], [c2, c2, 0.0]])
    return semimarkov_viterbi(post * cfg.get("wpost", 1.0), C2, pen=_pens3(g, cfg))


HARD = 1e12                     # cost of a forbidden length / transition: never chosen when a legal path exists
EDGE = 1e8                      # first / last frame of a window: forbids a state there (small next to the cumsums)


def frames(s: float, fps: float) -> int:
    """At least s seconds, in frames (0.5 s = 13 frames at 25 fps, 15 at 29.97)."""
    return max(1, math.ceil(s * fps - 1e-9))


def _dec_minrun(g, ev, cfg, a, b):
    E3 = emissions3(_sl(ev, a, b))
    m = frames(cfg.get("min_s", 0.5), float(g["fps"]))
    if b - a < m:
        return np.full(b - a, int(E3.sum(0).argmax()), np.int64)
    pen = np.zeros((3, m))
    pen[:, :m - 1] = HARD                                       # lengths 1 .. m-1 forbidden, m and up free
    return semimarkov_viterbi(E3, np.zeros((3, 3)), pen=pen)


# stoppage: 7 states -- 0 dead; per team (H, A): 1-2 between two dead balls, 3-4 after a dead ball and before the
# other team or the window's end, 5-6 after the other team or at the window's start
_ST_TEAM = np.array([D, H, A, H, A, H, A])
_ST_ALLOWED = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (2, 0), (3, 6), (4, 5), (5, 0), (6, 0), (5, 6), (6, 5))
_ST_NO_START = (1, 2, 3, 4)     # entered from a dead ball only
_ST_NO_END = (1, 2)             # must be followed by a dead ball


def stoppage_path(E3, m: int, m_dead: int, m_between: int) -> np.ndarray:
    """(T,) codes 0 H / 1 A / 2 D maximising the summed evidence E3 (T, 3) subject to: every D run >= max(m, m_dead);
    every H / A run >= m, except one with D on BOTH sides, which needs >= m_between instead (may be shorter or
    longer than m). Exact (semi-Markov Viterbi over 7 states, hard minimum durations)."""
    E3 = np.asarray(E3, dtype=np.float64)
    T = len(E3)
    E = np.ascontiguousarray(E3[:, _ST_TEAM])
    E[0, list(_ST_NO_START)] -= EDGE
    E[T - 1, list(_ST_NO_END)] -= EDGE
    C = np.full((7, 7), HARD)
    for i, j in _ST_ALLOWED:
        C[i, j] = 0.0
    mins = [max(m, m_dead), m_between, m_between, m, m, m, m]
    pen = np.zeros((7, max(mins)))
    for k, mk in enumerate(mins):
        pen[k, :mk - 1] = HARD
    return _ST_TEAM[semimarkov_viterbi(E, C, pen=pen)]


def _dec_stoppage(g, ev, cfg, a, b):
    E3 = emissions3(_sl(ev, a, b))
    fps = float(g["fps"])
    m, md, mb = (frames(cfg.get("min_s", 0.5), fps), frames(cfg.get("min_dead_s", 2.0), fps),
                 frames(cfg.get("min_between_s", 2.0), fps))
    if b - a < max(m, md):
        return np.full(b - a, int(E3.sum(0).argmax()), np.int64)
    return stoppage_path(E3, m, md, mb)


DECODERS = {
    "argmax": _dec_argmax, "hmm3": _dec_hmm3, "hmm3_bnd": _dec_hmm3, "hsmm_soft": _dec_hsmm_soft,
    "hsmm_emp": _dec_hsmm_emp, "hier": _dec_hier, "asrf": _dec_asrf, "fb_hsmm": _dec_fb_hsmm,
    "minrun": _dec_minrun, "stoppage": _dec_stoppage,
}
NEEDS_BND = {"hmm3_bnd", "asrf"}
BASELINES = ("prod", "final")


def decode(name, g, cfg):
    """Public entry point (make_submission calls it): codes for the whole array, X outside the
    windows. g needs `halves` (or y with X outside the graded blocks), fps and the streams."""
    ev = evidence(g, cfg)
    n = len(ev["alive"])
    out = np.full(n, X, np.int8)
    for a, b in windows(g):
        if b > a:
            out[a:b] = DECODERS[name](g, ev, cfg, a, b)
    return out


# ============================================================================================
# configuration spaces (config 0 = the centre = FINAL's values, then seeded samples: half global,
# half local around the centre -- the likely gain is a nudge of FINAL, not a new region)
# ============================================================================================

EMIT = {"wd": ("log", 0.5, 2.0), "bd": ("lin", -1.5, 1.5), "wh": ("log", 0.5, 2.0),
        "who_clip": ("log", 0.005, 0.1), "w_filled": ("lin", 0.1, 1.0)}
COST = {"c_ha": ("log", 4.0, 60.0), "c_ad": ("log", 3.0, 30.0), "c_da": ("log", 3.0, 30.0)}
SOFT = {"m_spell": ("lin", 0.2, 3.0), "lam_spell": ("log", 1.0, 40.0),
        "m_dead": ("lin", 0.4, 5.0), "lam_dead": ("log", 1.0, 40.0), "power": ("choice", [1.0, 2.0])}
ALPHA = {"alpha": ("lin", 0.0, 0.95)}

SPACES = {
    "argmax": {},
    "hmm3": {**EMIT, **COST},
    "hmm3_bnd": {**EMIT, **COST, **ALPHA},
    "hsmm_soft": {**EMIT, **COST, **SOFT, **ALPHA},
    "hsmm_emp": {**EMIT, **COST, **ALPHA, "lam_spell": ("log", 0.1, 5.0),
                 "lam_dead": ("log", 0.1, 5.0), "cap_spell": ("lin", 1.0, 5.0),
                 "cap_dead": ("lin", 2.0, 8.0)},
    "hier": {**EMIT, **COST, **SOFT, **ALPHA,
             "m_alive": ("lin", 0.4, 4.0), "lam_alive": ("log", 1.0, 40.0)},
    "asrf": {**EMIT, "c_ad": COST["c_ad"], "c_da": COST["c_da"], "c_ha": ("log", 0.5, 20.0),
             "theta": ("lin", 0.2, 0.8), "min_gap_s": ("lin", 0.2, 1.5),
             "smooth_s": ("lin", 0.04, 0.5), "cuts": ("choice", ["peaks", "union"])},
    "fb_hsmm": {**EMIT, **COST, **SOFT, **ALPHA,
                "c2": ("log", 0.5, 20.0), "wpost": ("log", 0.3, 3.0)},
    "minrun": {"min_s": ("lin", 0.2, 1.5)},
    "stoppage": {"min_s": ("lin", 0.2, 1.5), "min_dead_s": ("lin", 0.5, 5.0), "min_between_s": ("lin", 0.5, 5.0)},
}
LENGTH_ONLY = {"minrun", "stoppage"}     # evidence = FINAL's emission terms as they are (no fusion knobs)
# FINAL (make_submission.FINAL, decode.Params with dead_who="max"; owner TCN boundary alpha 0.5)
CENTRE = {"wd": 1.0, "bd": 0.0, "wh": 1.0, "who_clip": 0.02, "w_filled": 0.4, "w_tags": 0.4,
          "c_ha": 18.0, "c_ad": 10.0, "c_da": 10.0, "alpha": 0.5, "alpha_d": 0.0,
          "fps_scale": False, "dead_mix": "prob_mean", "v8_share": 0.5,
          "home_mix": "prob_mean", "kappa": 0.0, "v1_share": 0.0,
          "m_spell": 1.0, "lam_spell": 8.0, "m_dead": 2.0, "lam_dead": 8.0, "power": 1.0,
          "m_alive": 1.0, "lam_alive": 8.0, "theta": 0.5, "min_gap_s": 0.5, "smooth_s": 0.2,
          "cuts": "peaks", "c2": 2.0, "wpost": 1.0, "cap_spell": 4.0, "cap_dead": 6.0,
          "min_s": 0.5, "min_dead_s": 2.0, "min_between_s": 2.0}
USES_COSTS3 = {"hmm3", "hmm3_bnd", "hsmm_soft", "hsmm_emp", "hier", "fb_hsmm"}


def stream_space(meta, name):
    """Fusion knobs that exist only if the inputs carry the separate model outputs."""
    sp = {}
    if meta.get("has_dead_pair"):
        sp["dead_mix"] = ("choice", ["prob_mean", "loglin"])
        sp["v8_share"] = ("lin", 0.0, 1.0)
    if meta.get("has_seeds"):
        sp["home_mix"] = ("choice", ["prob_mean", "logit_mean"])
        sp["kappa"] = ("choice", [0.0, 0.25, 0.5, 1.0])
    if meta.get("has_v1"):
        sp["v1_share"] = ("lin", 0.0, 0.6)
    if meta.get("has_bnd_dead") and name in USES_COSTS3:
        sp["alpha_d"] = ("lin", 0.0, 0.9)
    if meta.get("mixed_fps"):
        sp["fps_scale"] = ("choice", [False, True])
    return sp


def _draw(rng, spec):
    if spec[0] == "log":
        return float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
    if spec[0] == "lin":
        return float(rng.uniform(spec[1], spec[2]))
    return spec[1][int(rng.integers(len(spec[1])))]


def _nudge(rng, spec, c):
    if spec[0] == "log":
        return float(np.clip(c * np.exp(rng.normal(0, 0.35)), spec[1], spec[2]))
    if spec[0] == "lin":
        return float(np.clip(c + rng.normal(0, 0.15 * (spec[2] - spec[1])), spec[1], spec[2]))
    return c if rng.random() < 0.7 else spec[1][int(rng.integers(len(spec[1])))]


def sample_configs(name, n, meta, seed=0, local_frac=0.5):
    space = dict(SPACES[name])
    if name != "argmax" and name not in LENGTH_ONLY:
        space.update(stream_space(meta, name))
    centre = {k: CENTRE[k] for k in space}
    if name == "hmm3":
        centre.pop("alpha", None)
    if name == "hsmm_emp":
        centre.update(lam_spell=1.0, lam_dead=1.0)
    if name == "asrf":
        centre.update(c_ha=2.0)
    cfgs = [centre]
    if not space:
        return cfgs
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    every = int(round(1 / local_frac)) if local_frac > 0 else 0
    for k in range(1, max(1, n)):
        local = every > 0 and k % every == 1 % every
        c = {}
        for key, spec in sorted(space.items()):
            base = centre.get(key, CENTRE.get(key))
            c[key] = _nudge(rng, spec, base) if local and base is not None else _draw(rng, spec)
        cfgs.append({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in c.items()})
    return cfgs


# ============================================================================================
# official metric
# ============================================================================================

_PM = None


def load_metric(path=None):
    """The organisers' possession_metrics.py -- never a reimplementation."""
    global _PM
    if _PM is not None:
        return _PM
    cands = [path] if path else []
    if os.environ.get("MQF_OFFICIAL"):
        cands.append(os.environ["MQF_OFFICIAL"])
    here = os.path.abspath(os.path.dirname(__file__))
    for up in range(6):
        base = os.path.normpath(os.path.join(here, *([".."] * up)))
        cands += [os.path.join(base, "vendor", "mtah2027", "evaluation"),
                  os.path.join(base, "evaluation")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "possession_metrics.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            _PM = importlib.import_module("possession_metrics")
            return _PM
    raise SystemExit("official possession_metrics.py not found; pass --official DIR")


def score(y, pred):
    """spellF1 / macroF1 / alive-dead F1 / count error with the official building blocks, as
    possession_metrics.evaluate combines them (tests/test_fusion.py checks they agree)."""
    pm = load_metric()
    y = np.asarray(y, np.int8)
    pred = np.asarray(pred, np.int8)
    runs = pm.valid_runs(y)
    cm3 = pm.scored_confusion(pm.confusion(y, pred))
    filled = pm.fill_abstentions(pred, runs)
    sp = pm.spell_scores(y, filled, runs)
    return {"spellF1": sp["mean_f1"], "macroF1": pm.macro_f1(cm3),
            "adF1": pm.alive_dead_f1(cm3), "cnt_err": sp["n_pred"] - sp["n_gt"]}


BUCKETS = ((0, 2), (2, 5), (5, 15), (15, 1e9))


def bucket_hits(y, pred, fps):
    """Match rate at IoU 0.5 by true possession length, via the official matching functions."""
    pm = load_metric()
    y = np.asarray(y, np.int8)
    runs = pm.valid_runs(y)
    filled = pm.fill_abstentions(np.asarray(pred, np.int8), runs)
    gs = pm.possession_spells(y, runs)
    ps = pm.possession_spells(filled, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5)
    lens = np.array([(b - a) / fps for a, b, _ in gs])
    out = []
    for lo, hi in BUCKETS:
        m = (lens >= lo) & (lens < hi)
        out.append((int(flags[m].sum()), int(m.sum())))
    return out


# ============================================================================================
# data
# ============================================================================================

FEAT_COLS = ("ball_seen", "ball_filled", "ball_out", "d_home", "d_away", "d_diff", "spread_diff",
             "mean_player_speed")


def save_game(root, cond, game, seed, y, fps, *, source=None, halves=None, p_dead=None,
              p_dead_asf=None, p_dead_v8=None, p_home=None, p_home_seeds=None, p_home_v1=None,
              p_bnd=None, p_bnd_dead=None, ball_out=None, ball_filled=None, tag_vote=None,
              feat=None, pred_final=None, params_json=None, who_alpha=None, extra=None):
    """One call per (column, val game, noise seed). Writes <root>/<cond>/<game>__s<seed>.npz in
    the input contract above, float64 (the arrays the decoder read, bit for bit). Other games only."""
    if str(game).upper().startswith("BAR"):
        raise ValueError("fusion inputs are for other games only")
    arrs = {"y": np.asarray(y, np.int8), "fps": np.float64(fps)}
    T = len(arrs["y"])
    if not set(np.unique(arrs["y"]).tolist()) <= {H, A, D, X}:
        raise ValueError(f"{game}: y must hold codes 0..3")
    for k, v in (("p_dead", p_dead), ("p_dead_asf", p_dead_asf), ("p_dead_v8", p_dead_v8),
                 ("p_home", p_home), ("p_home_seeds", p_home_seeds), ("p_home_v1", p_home_v1),
                 ("p_bnd", p_bnd), ("p_bnd_dead", p_bnd_dead)):
        if v is not None:
            v = np.asarray(v, np.float64)
            if v.shape[-1] != T:
                raise ValueError(f"{game}: {k} has length {v.shape[-1]}, y has {T}")
            if k.startswith("p_") and np.nanmax(np.abs(v - 0.5)) > 0.5:
                raise ValueError(f"{game}: {k} outside [0, 1]")
            arrs[k] = v
    if tag_vote is not None:
        arrs["tag_vote"] = np.asarray(tag_vote)
    for k, v in (("ball_out", ball_out), ("ball_filled", ball_filled)):
        if v is not None:
            arrs[k] = np.asarray(v, bool)
    for c, v in (feat or {}).items():
        arrs["feat_" + c] = np.asarray(v)
    if pred_final is not None:
        arrs["pred_final"] = np.asarray(pred_final, np.int8)
    if halves is not None:
        arrs["halves"] = np.asarray([(h[-2], h[-1]) for h in halves], np.int64)
    if source is not None:
        arrs["source"] = np.str_(source)
    if params_json is not None:
        arrs["params_json"] = np.str_(params_json)
    if who_alpha is not None:
        arrs["who_alpha"] = np.float64(who_alpha)
    for k, v in (extra or {}).items():
        arrs[k] = np.asarray(v)
    if "p_dead" not in arrs and not ("p_dead_asf" in arrs and "p_dead_v8" in arrs):
        raise ValueError(f"{game}: needs p_dead or p_dead_asf + p_dead_v8")
    if "p_home" not in arrs and "p_home_seeds" not in arrs:
        raise ValueError(f"{game}: needs p_home or p_home_seeds")
    os.makedirs(os.path.join(root, cond), exist_ok=True)
    np.savez_compressed(os.path.join(root, cond, f"{game}__s{seed}.npz"), **arrs)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        g = {k: z[k] for k in z.files}
    g["fps"] = float(g["fps"]) if "fps" in g else REF_FPS
    g["y"] = g["y"].astype(np.int8)
    for k in ("source", "params_json"):
        if k in g:
            g[k] = str(g[k])
    if "p_dead" not in g and not ("p_dead_asf" in g and "p_dead_v8" in g):
        raise ValueError(f"{path}: needs p_dead or p_dead_asf + p_dead_v8")
    if "p_home" not in g and "p_home_seeds" not in g:
        raise ValueError(f"{path}: needs p_home or p_home_seeds")
    return g


def discover(root):
    files = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*.npz"))):
        cond = os.path.basename(os.path.dirname(p))
        stem = os.path.splitext(os.path.basename(p))[0]
        game, _, seed = stem.rpartition("__s")
        if not game:
            game, seed = stem, "0"
        files.append({"path": p, "cond": cond, "game": game, "seed": seed})
    if not files:
        raise SystemExit(f"no <column>/<game>__s<seed>.npz under {root}")
    return files


def duration_prior(ys_fps):
    """Smoothed log-density of spell and dead-run lengths (s) from ground-truth arrays."""
    pm = load_metric()
    edges = np.arange(-1.0, 2.6, 0.1)                  # log10 seconds, 0.1 s .. 400 s
    centers = 10 ** (0.5 * (edges[:-1] + edges[1:]))
    out = {}
    for key, codes in (("spell", (H, A)), ("dead", (D,))):
        lens = []
        for y, fps in ys_fps:
            for a, b, k in pm.segments(y, pm.valid_runs(y)):
                if k in codes:
                    lens.append((b - a) / fps)
        h, _ = np.histogram(np.log10(np.clip(lens, 0.11, 399)), bins=edges)
        h = np.convolve(h + 1.0, np.array([1, 2, 3, 2, 1]) / 9.0, mode="same")
        out[key] = {"centers_s": centers.tolist(), "logf": np.log(h / h.sum()).tolist()}
    return out


def _call_hook(hook, g):
    mod, fn = hook.split(":")
    return np.asarray(getattr(importlib.import_module(mod), fn)(g), np.int8)


# ============================================================================================
# suite: score every config of every decoder on every file
# ============================================================================================

_W = {}


def _winit(official, plan, priors, hook):
    load_metric(official)
    _W["plan"], _W["priors"], _W["hook"] = plan, priors, hook


def _row(f, g, name, ci, pred, same=-1):
    return {"decoder": name, "cfg": ci, "cond": f["cond"], "game": f["game"], "seed": f["seed"],
            "source": g.get("source", "all"), "same_final": same, **score(g["y"], pred)}


def _work(f):
    g = load_npz(f["path"])
    g["_dur_prior"] = _W["priors"].get(f["game"])
    fin = g.get("pred_final")
    rows = []
    if fin is not None:
        rows.append(_row(f, g, "final", 0, fin))
    if _W["hook"]:
        pred = _call_hook(_W["hook"], g)
        rows.append(_row(f, g, "prod", 0, pred, -1 if fin is None else int(np.array_equal(pred, fin))))
    for name, cfgs in _W["plan"].items():
        if name in NEEDS_BND and "p_bnd" not in g:
            continue
        for ci, cfg in enumerate(cfgs):
            pred = decode(name, g, cfg)
            same = int(np.array_equal(pred, fin)) if (fin is not None and ci == 0) else -1
            rows.append(_row(f, g, name, ci, pred, same))
    return rows


def _bucket_work(item):
    f, chosen = item
    g = load_npz(f["path"])
    g["_dur_prior"] = _W["priors"].get(f["game"])
    out = {}
    for name, cfg in chosen.items():
        base = name.split("@")[0]
        if base in NEEDS_BND and "p_bnd" not in g:
            continue
        if base == "final":
            if "pred_final" not in g:
                continue
            pred = g["pred_final"]
        elif base == "prod":
            pred = _call_hook(_W["hook"], g)
        else:
            pred = decode(base, g, cfg)
        out[name] = bucket_hits(g["y"], pred, g["fps"])
    return out


def _pmap(fn, items, jobs, init_args):
    if jobs <= 1:
        _winit(*init_args)
        return [fn(it) for it in items]
    with ProcessPoolExecutor(max_workers=jobs, initializer=_winit, initargs=init_args) as ex:
        return list(ex.map(fn, items, chunksize=1))


def probe_meta(files):
    probe = load_npz(files[0]["path"])
    fps = {round(load_npz(f["path"])["fps"], 3) for f in
           {f["game"]: f for f in files}.values()}
    return {"has_dead_pair": "p_dead_asf" in probe and "p_dead_v8" in probe,
            "has_seeds": "p_home_seeds" in probe, "has_bnd": "p_bnd" in probe,
            "has_bnd_dead": "p_bnd_dead" in probe, "has_v1": "p_home_v1" in probe,
            "has_final": "pred_final" in probe, "has_feat": "feat_d_diff" in probe,
            "mixed_fps": len(fps) > 1, "fps": sorted(fps)}


def priors_for(files, games, duration_dir=None):
    one = {}
    for f in files:
        one.setdefault(f["game"], f["path"])
    ys = {gm: load_npz(p) for gm, p in one.items()}
    extra = []
    if duration_dir:
        for p in sorted(glob.glob(os.path.join(duration_dir, "*.npz"))):
            with np.load(p, allow_pickle=False) as z:     # GT only: y (+ fps)
                extra.append((z["y"].astype(np.int8),
                              float(z["fps"]) if "fps" in z.files else REF_FPS))
    return {gm: duration_prior([(v["y"], v["fps"]) for k, v in ys.items() if k != gm] + extra)
            for gm in games}


def run_suite(args):
    """Decode + score; returns {"rows", "plan", "meta", "files", "baseline", "hook"} and, unless
    args.no_report, writes the report of these rows."""
    t0 = time.time()
    load_metric(args.official)
    files = discover(args.preds)
    if args.conds:
        files = [f for f in files if f["cond"] in args.conds]
    if args.seeds:
        files = [f for f in files if f["seed"] in args.seeds]
    if any(f["game"].upper().startswith("BAR") for f in files):
        raise SystemExit("a game named BAR* is in --preds: the suite is for other games only")
    meta = probe_meta(files)
    budget = 12 if args.quick else args.budget
    names = args.decoders or [n for n in DECODERS if meta["has_bnd"] or n not in NEEDS_BND]
    plan = {n: sample_configs(n, budget, meta, args.sample_seed, args.local_frac) for n in names}
    hook = args.baseline_hook
    games = sorted({f["game"] for f in files})
    priors = priors_for(files, games, args.duration_dir) if "hsmm_emp" in plan else {}
    n_dec = sum(len(c) for c in plan.values())
    print(f"[fusion] {len(files)} files ({len(games)} games x {len({f['cond'] for f in files})} "
          f"columns), {len(plan)} decoders, {n_dec} configs/file, numba={HAVE_NUMBA}, "
          f"jobs={args.jobs}", flush=True)
    rows = [r for rs in _pmap(_work, files, args.jobs, (args.official, plan, priors, hook))
            for r in rs]
    print(f"[fusion] scored {len(rows)} decodes in {time.time() - t0:.0f}s", flush=True)
    baseline = pick_baseline(args.baseline, hook, meta)
    out = {"rows": rows, "plan": plan, "meta": meta, "files": files, "baseline": baseline,
           "hook": hook, "priors": priors, "secs": time.time() - t0}
    if not getattr(args, "no_report", False):
        out["res"] = report(args, out)
    return out


def pick_baseline(given, hook, meta):
    if given:
        return given
    if meta.get("has_final"):
        return "final"
    return "prod" if hook else "hmm3_bnd@0"


# ============================================================================================
# selection: per column, mean per source then over sources (final_check); nested LOGO by game
# ============================================================================================

class Table:
    """Scores of every candidate (decoder, cfg) on every (game, column, seed)."""

    def __init__(self, rows):
        self.games = sorted({r["game"] for r in rows})
        self.conds = sorted({r["cond"] for r in rows})
        self.seeds = sorted({r["seed"] for r in rows})
        src = {r["game"]: r["source"] for r in rows}
        self.sources = sorted(set(src.values()))
        self.src_of = np.array([self.sources.index(src[g]) for g in self.games])
        self.keys = sorted({(r["decoder"], r["cfg"]) for r in rows})
        ki = {k: i for i, k in enumerate(self.keys)}
        gi = {g: i for i, g in enumerate(self.games)}
        ci = {c: i for i, c in enumerate(self.conds)}
        si = {s: i for i, s in enumerate(self.seeds)}
        shape = (len(self.keys), len(self.games), len(self.conds), len(self.seeds))
        self.v = {m: np.full(shape, np.nan) for m in ("spellF1", "macroF1", "adF1", "cnt_err")}
        self.same = {}
        for r in rows:
            idx = (ki[(r["decoder"], r["cfg"])], gi[r["game"]], ci[r["cond"]], si[r["seed"]])
            for m in self.v:
                self.v[m][idx] = r[m]
            if r.get("same_final", -1) >= 0:
                self.same.setdefault((r["decoder"], r["cfg"]), []).append(r["same_final"])
        with np.errstate(invalid="ignore"), _quiet():
            self.m = {m: np.nanmean(a, 3) for m, a in self.v.items()}        # (K, G, C)

    def sb(self, arr, gmask):
        """(K, G, C) -> (K, C): mean over the masked games per source, then over sources."""
        per = []
        for s in range(len(self.sources)):
            sel = gmask & (self.src_of == s)
            if sel.any():
                with _quiet():
                    per.append(np.nanmean(arr[:, sel, :], 1))
        with _quiet():
            return np.nanmean(np.stack(per), 0)

    def objective(self, kidx, gmask):
        o = np.nanmean(self.sb(self.m["spellF1"][kidx], gmask), 1)
        t = np.nanmean(self.sb(self.m["macroF1"][kidx], gmask), 1)
        return o, t

    def logo(self, kidx):
        """Nested LOGO over candidates kidx: per held-out game, the candidate best on the others."""
        kidx = np.asarray(kidx)
        picks = []
        for h in range(len(self.games)):
            gmask = np.arange(len(self.games)) != h
            o, t = self.objective(kidx, gmask)
            o = np.where(np.isnan(o), -np.inf, o)
            t = np.where(np.isnan(t), -np.inf, t)
            best = int(np.lexsort((np.arange(len(kidx)), -t, -o))[0])
            picks.append(int(kidx[best]))
        o, t = self.objective(kidx, np.ones(len(self.games), bool))
        final = int(kidx[int(np.lexsort((np.arange(len(kidx)), -np.nan_to_num(t, nan=-9),
                                         -np.nan_to_num(o, nan=-9)))[0])])
        return picks, final

    def held(self, picks):
        """Per-run held-out scores {metric: (G, C, S)} of a LOGO pick list."""
        return {m: np.stack([a[p, g] for g, p in enumerate(picks)]) for m, a in self.v.items()}


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        return self._w.__exit__(*a)


def _col_means(tab, runs):
    """(G, C, S) -> (C,) mean over seeds, per source, then over sources."""
    with _quiet():
        gc = np.nanmean(runs, 2)[None]
    return tab.sb(gc, np.ones(len(tab.games), bool))[0]


def _paired(a, b):
    """(G, C, S) vs (G, C, S) -> per column {mean, se, wins, n} over (game, seed) runs."""
    out = []
    for c in range(a.shape[1]):
        d = (a[:, c, :] - b[:, c, :]).ravel()
        d = d[~np.isnan(d)]
        n = len(d)
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        out.append({"mean": float(d.mean()) if n else float("nan"), "se": se,
                    "wins": int((d > 0).sum()), "n": n})
    return out


def _passes(cols):
    return all(np.isfinite(c["se"]) and c["mean"] > 2 * c["se"] and 2 * c["wins"] > c["n"]
               for c in cols)


def analyse(rows, baseline):
    """Everything the report needs, from the rows alone (quick + deep rows can be merged)."""
    tab = Table(rows)
    res = {}
    by_dec = {}
    for i, (d, c) in enumerate(tab.keys):
        by_dec.setdefault(d, []).append(i)
    for d, ks in by_dec.items():
        picks, final = tab.logo(ks)
        agree = max(picks.count(p) for p in set(picks)) / len(picks)
        res[d] = {"picks": [tab.keys[p] for p in picks], "final": tab.keys[final],
                  "held": tab.held(picks), "fold_agreement": agree}
        if d not in BASELINES:
            k0 = tab.keys.index((d, 0)) if (d, 0) in tab.keys else None
            if k0 is not None:
                res[d + "@0"] = {"picks": [(d, 0)] * len(tab.games), "final": (d, 0),
                                 "held": tab.held([k0] * len(tab.games)), "fold_agreement": 1.0}
    cand = [i for i, (d, c) in enumerate(tab.keys) if d not in BASELINES]
    if cand:
        picks, final = tab.logo(cand)
        agree = max(picks.count(p) for p in set(picks)) / len(picks)
        res["PIPELINE"] = {"picks": [tab.keys[p] for p in picks], "final": tab.keys[final],
                           "held": tab.held(picks), "fold_agreement": agree}
    if baseline in res:
        b = res[baseline]["held"]["spellF1"]
        for n, r in res.items():
            if n == baseline:
                continue
            r["paired"] = _paired(r["held"]["spellF1"], b)
            r["paired_macro"] = _paired(r["held"]["macroF1"], res[baseline]["held"]["macroF1"])
            r["passes"] = _passes(r["paired"])
    for n, r in res.items():
        r["cols"] = {m: _col_means(tab, r["held"][m]) for m in r["held"]}
        r["mean"] = float(np.nanmean(r["cols"]["spellF1"]))
    return tab, res


def report(args, out):
    rows, plan, baseline = out["rows"], out["plan"], out["baseline"]
    tab, res = analyse(rows, baseline)
    chosen = {}
    for n, r in res.items():
        d, c = r["final"]
        chosen[n] = (plan[d][c] if d in plan else {"hook": out["hook"]} if d == "prod" else {})
    files = out["files"]
    buckets = {}
    if not getattr(args, "no_buckets", False):
        keep = {n: cfg for n, cfg in chosen.items() if n != "PIPELINE"}
        bk = _pmap(_bucket_work, [(f, keep) for f in files], args.jobs,
                   (args.official, plan, out["priors"], out["hook"]))
        for dct in bk:
            for n, lst in dct.items():
                acc = buckets.setdefault(n, [[0, 0] for _ in BUCKETS])
                for k, (hit, tot) in enumerate(lst):
                    acc[k][0] += hit
                    acc[k][1] += tot
    write_outputs(args, rows, tab, res, chosen, buckets, baseline, out["meta"], out["secs"])
    return {"tab": tab, "res": res, "chosen": chosen}


def _fmt_p(p):
    if not p or not np.isfinite(p["mean"]):
        return "—"
    se = "n/a" if not np.isfinite(p["se"]) else f"{p['se']:.3f}"
    return f"{p['mean']:+.3f} ({se}) {p['wins']}/{p['n']}"


def write_outputs(args, rows, tab, res, chosen, buckets, baseline, meta, secs):
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"fusion_{args.tag}")
    with open(stem + "_scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = {n: {"decoder": res[n]["final"][0], "config": chosen[n],
                "held_out_spellF1": res[n]["mean"]} for n in res}
    with open(stem + "_best.json", "w") as fh:
        json.dump(best, fh, indent=2, sort_keys=True, default=str)

    order = sorted(res, key=lambda n: -res[n]["mean"])
    C = tab.conds
    L = [f"# Fusion suite `{args.tag}`", "",
         f"- Inputs `{args.preds}`: {len(tab.games)} games ({', '.join(tab.sources)}), "
         f"columns {', '.join(C)}, seeds {', '.join(tab.seeds)}; streams: dead pair "
         f"{meta['has_dead_pair']}, TCN seeds {meta['has_seeds']}, boundary {meta['has_bnd']}, "
         f"dead boundary {meta['has_bnd_dead']}, who_v1 {meta['has_v1']}, FINAL labels "
         f"{meta['has_final']}; fps {meta['fps']}",
         "- Every number is HELD OUT: per game, the config (or for PIPELINE the decoder AND config) "
         "best on the other games (spellF1 per column, mean per source then over sources, then over "
         "columns; tie macroF1) -- final_check's averaging",
         f"- Baseline `{baseline}`; official metric `{load_metric().__file__}`; "
         f"{secs / 60:.1f} min, numba={HAVE_NUMBA}", ""]
    same = {k: v for k, v in tab.same.items()}
    if same:
        L += ["## Sanity: labels identical to the exported FINAL labels", ""]
        for (d, c), v in sorted(same.items()):
            L.append(f"- `{d}`{'@' + str(c) if d not in BASELINES else ''}: {sum(v)}/{len(v)} files")
        L.append("")
    L += ["## Held-out spellF1 per column (mean per source, then over sources)", "",
          "| decoder | " + " | ".join(C) + " | **mean** | macroF1 | adF1 | count err | folds agree |",
          "|---|" + "---|" * (len(C) + 5)]
    for n in order:
        r = res[n]
        cells = [f"{x:.3f}" for x in r["cols"]["spellF1"]]
        L.append(f"| {n} | " + " | ".join(cells) + f" | **{r['mean']:.3f}** | "
                 f"{np.nanmean(r['cols']['macroF1']):.3f} | {np.nanmean(r['cols']['adF1']):.3f} | "
                 f"{np.nanmean(r['cols']['cnt_err']):+.0f} | {r['fold_agreement']:.2f} |")
    L += ["", f"## Paired vs `{baseline}` (held out): Δ spellF1 mean (se) wins/n over game × seed "
          "runs; PASS = Δ > 2 se and wins > n/2 in EVERY column (#38)", "",
          "| decoder | " + " | ".join(C) + " | passes |", "|---|" + "---|" * (len(C) + 1)]
    for n in order:
        r = res[n]
        if "paired" in r:
            L.append(f"| {n} | " + " | ".join(_fmt_p(p) for p in r["paired"]) +
                     f" | {'PASS' if r['passes'] else 'no'} |")
    L += ["", "`PIPELINE` = the whole suite as a procedure: inside each fold the decoder AND its "
          "config are chosen on the other games. It is the honest estimate of 'ship the suite's "
          "winner'; a single decoder's row is optimistic by the choice among decoders. `@0` = config "
          "0 untuned (hmm3_bnd@0 = FINAL). `count err` = predicted − true spells (negative = "
          "merging). Low `folds agree` = flat optimum: prefer config 0 or round values.", "",
          "## Match rate at IoU 0.5 by true possession length (final configs, pooled)", "",
          "| decoder | " + " | ".join(f"{lo}-{hi if hi < 1e8 else 'inf'} s" for lo, hi in BUCKETS)
          + " |", "|---|" + "---|" * len(BUCKETS)]
    for n in order:
        if n in buckets:
            L.append(f"| {n} | " + " | ".join(f"{h / t:.3f} ({t})" if t else "-"
                                              for h, t in buckets[n]) + " |")
    L += ["", "## Final configs (chosen on all games -- what would ship)", ""]
    for n in order:
        d, c = res[n]["final"]
        L.append(f"- `{n}` → `{d}` cfg {c}: `{json.dumps(chosen[n], sort_keys=True, default=str)}`")
    with open(stem + "_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:len(order) + 30]))
    print(f"\n[fusion] wrote {stem}_report.md, _scores.csv, _best.json")


# ============================================================================================
# synthetic predictions (for smoke tests only -- never for decisions)
# ============================================================================================

def _ar1(rng, T, rho, sigma):
    e = rng.normal(0, sigma * math.sqrt(1 - rho ** 2), T)
    out = np.empty(T)
    acc = rng.normal(0, sigma)
    for t in range(T):
        acc = rho * acc + e[t]
        out[t] = acc
    return out


def synth_game(rng, fps=25.0, half_min=45.0, noise=1.0, source="syn"):
    """GT with realistic spell/stoppage lengths + model outputs with the known failure modes
    (late boundaries, flicker, false short stoppages, weak short spells, spurious boundary peaks),
    plus the production context (features, params) so the production hook and FINAL labels exist."""
    y_parts, T_half = [], int(half_min * 60 * fps)
    halves = []
    pos = 0
    for h in range(2):
        seq = [(D, int(rng.uniform(5, 30) * fps))]
        n = seq[0][1]
        team = int(rng.integers(2))
        while n < T_half:
            for _ in range(1 + int(rng.geometric(0.3))):
                d = int(np.clip(rng.lognormal(np.log(5.0), 0.9), 0.4, 60) * fps)
                seq.append((team, d))
                n += d
                team = 1 - team if rng.random() < 0.92 else team
            d = int(np.clip(rng.lognormal(np.log(12.0), 0.8), 1.0, 120) * fps)
            seq.append((D, d))
            n += d
            team = int(rng.integers(2))
        y = np.concatenate([np.full(d, k, np.int8) for k, d in seq])[:T_half]
        halves.append((h + 1, pos, pos + len(y) - 1))
        pos += len(y) + int(60 * fps)
        y_parts += [y, np.full(int(60 * fps), X, np.int8)]
    y = np.concatenate(y_parts[:-1])
    T = len(y)
    dead = (y == D).astype(float)
    team = np.where(y == H, 1.0, np.where(y == A, -1.0, 0.0))
    lag = int(0.6 * fps)
    zd = 3.0 * (2 * np.roll(dead, lag) - 1)
    segs = np.flatnonzero(np.diff(np.concatenate(([0], dead, [0]))) != 0).reshape(-1, 2)
    for s, e in segs:
        if (e - s) < 3 * fps:
            zd[s + lag:e + lag] -= 4.0 * rng.random()
    for _ in range(int(T / fps / 60 * 1.0)):
        s = int(rng.integers(T))
        zd[s:s + int(rng.uniform(1, 4) * fps)] += 4.0
    zd_common = zd + _ar1(rng, T, 0.995, 1.5 * noise)
    p_asf = np.round(1 / (1 + np.exp(-(zd_common + _ar1(rng, T, 0.99, 0.8 * noise)))), 4)
    p_v8 = 1 / (1 + np.exp(-(zd_common + _ar1(rng, T, 0.99, 0.8 * noise) + 0.5)))
    zh = 2.5 * team
    last = 0.0
    for t in range(1, T):
        if team[t] != team[t - 1]:
            if team[t - 1] != 0:
                last = team[t - 1]
            if team[t] != 0 and last != 0 and last != team[t]:
                L = int(rng.uniform(0, 1.5) * fps)
                zh[t:t + L] = np.where(team[t:t + L] != 0, 2.5 * last, zh[t:t + L])
    spells = np.flatnonzero(np.diff(np.concatenate(([0.0], team, [0.0]))) != 0)
    for s, e in zip(spells[:-1], spells[1:]):
        if team[s] != 0 and (e - s) < 1.5 * fps:
            zh[s:e] *= (e - s) / (1.5 * fps)
    zh_common = zh + _ar1(rng, T, 0.98, 1.4 * noise) + rng.normal(0, 0.8 * noise, T)
    seeds = np.round(np.stack([1 / (1 + np.exp(-(zh_common + _ar1(rng, T, 0.95, 0.5 * noise))))
                               for _ in range(3)]), 4)
    v1 = np.round(1 / (1 + np.exp(-(0.7 * zh_common + rng.normal(0, 1.0, T)))), 4)
    bnd = np.zeros(T)
    tt = np.arange(T)
    sw = np.flatnonzero((team[1:] != team[:-1]) & (team[1:] != 0) & (team[:-1] != 0)) + 1
    sig = 0.3 * fps
    for s in sw:
        if rng.random() < 0.75:
            c = s + rng.normal(0, 0.3 * fps)
            lo, hi = int(max(0, c - 4 * sig)), int(min(T, c + 4 * sig))
            bnd[lo:hi] = np.maximum(bnd[lo:hi], rng.uniform(0.4, 1.0) *
                                    np.exp(-0.5 * ((tt[lo:hi] - c) / sig) ** 2))
    for _ in range(int(T / fps / 60 * 2 * noise)):
        c = int(rng.integers(T))
        lo, hi = max(0, c - int(4 * sig)), min(T, c + int(4 * sig))
        bnd[lo:hi] = np.maximum(bnd[lo:hi], rng.uniform(0.2, 0.7) *
                                np.exp(-0.5 * ((tt[lo:hi] - c) / sig) ** 2))
    dsw = np.flatnonzero(np.diff(dead) != 0) + 1
    bnd_d = np.zeros(T)
    for s in dsw:
        lo, hi = max(0, s - int(4 * sig)), min(T, s + int(4 * sig))
        bnd_d[lo:hi] = np.maximum(bnd_d[lo:hi], 0.8 * np.exp(-0.5 * ((tt[lo:hi] - s - lag) / sig) ** 2))
    seen = rng.random(T) > 0.4
    filled = ~seen & (rng.random(T) < 0.8)
    ball_out = seen & (dead > 0) & (rng.random(T) < 0.05)
    d_home = rng.gamma(2.0, 2.0, T)
    d_away = rng.gamma(2.0, 2.0, T)
    feat = {"ball_seen": seen | filled, "ball_filled": filled, "ball_out": ball_out,
            "d_home": d_home, "d_away": d_away, "d_diff": d_away - d_home,
            "spread_diff": rng.normal(0, 3, T), "mean_player_speed": rng.gamma(3.0, 0.6, T)}
    p_dead = np.round(0.5 * (p_asf + p_v8), 4)
    params = {"tau_m": 1.5, "loose_m": 4.0, "still_speed": 1.3, "still_width": 0.35, "w_seen": 1.0,
              "w_unseen": 0.15, "w_filled": 0.4, "w_spread": 0.15, "w_tags": 0.4,
              "switch_cost": 18.0, "to_dead_cost": 10.0, "from_dead_cost": 10.0, "w_dead": 1.0,
              "dead_who": "max", "w_who_model": 1.0, "who_clip": 0.02}
    g = {"y": y, "fps": np.float64(fps), "source": source, "halves": np.array([h[1:] for h in halves]),
         "p_dead_asf": p_asf, "p_dead_v8": p_v8, "p_dead": p_dead, "p_home_seeds": seeds,
         "p_home": np.round(seeds.mean(0), 4), "p_home_v1": v1, "p_bnd": np.round(np.clip(bnd, 0, 1), 4),
         "p_bnd_dead": np.round(bnd_d, 4), "ball_out": ball_out, "ball_filled": filled,
         "params_json": json.dumps(params), "who_alpha": np.float64(0.5),
         **{"feat_" + k: v for k, v in feat.items()}}
    try:                                      # FINAL's labels, from the production decoder
        from .fusion_export import prod
        g["pred_final"] = prod(g)
    except Exception:  # pragma: no cover -- standalone use without the pipeline
        pass
    return g


def run_synth(args):
    srcs = ("idsse", "metrica", "pff")
    for cond, noise in (("clean_exact", 0.7), ("noisy_fuzzy", 1.2)):
        os.makedirs(os.path.join(args.out, cond), exist_ok=True)
        for k in range(args.games):
            for s in range(args.n_seeds):
                rng = np.random.default_rng([k, s, int(noise * 10)])
                fps = 25.0 if k % 3 else 29.97
                g = synth_game(rng, fps=fps, half_min=args.half_min, noise=noise,
                               source=srcs[k % 3])
                np.savez_compressed(os.path.join(args.out, cond, f"SYN{k:02d}__s{s}.npz"), **g)
    print(f"[fusion] synthetic predictions in {args.out}")


def run_apply(args):
    with open(args.best) as fh:
        best = json.load(fh)[args.decoder]
    g = load_npz(args.npz)
    np.save(args.out, decode(best["decoder"], g, best["config"]))
    print(f"[fusion] {best['decoder']} -> {args.out}")


def add_run_args(r):
    r.add_argument("--preds", required=True)
    r.add_argument("--decoders", nargs="*")
    r.add_argument("--conds", nargs="*")
    r.add_argument("--seeds", nargs="*")
    r.add_argument("--budget", type=int, default=40)
    r.add_argument("--quick", action="store_true")
    r.add_argument("--sample-seed", type=int, default=0)
    r.add_argument("--local-frac", type=float, default=0.5, help="share of configs sampled near FINAL")
    r.add_argument("--baseline", help="row the deltas are paired with (default: final, else prod)")
    r.add_argument("--baseline-hook", default="mqf.decode:fusion_baseline",
                   help="module:function(g) -> codes, the production decoder ('' = none)")
    r.add_argument("--duration-dir", help="extra GT-only npz (train games) for hsmm_emp priors")
    r.add_argument("--official")
    r.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--tag", default="dev")
    r.add_argument("--out", default="harness_results")
    r.add_argument("--no-buckets", action="store_true")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mqf.fusion")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run"))
    s = sub.add_parser("synth")
    s.add_argument("--out", default="harness_results/fusion_synth")
    s.add_argument("--games", type=int, default=6)
    s.add_argument("--n-seeds", type=int, default=2)
    s.add_argument("--half-min", type=float, default=45.0)
    p = sub.add_parser("apply")
    p.add_argument("--npz", required=True)
    p.add_argument("--best", required=True)
    p.add_argument("--decoder", required=True, help="a row name of _best.json")
    p.add_argument("--out", required=True)
    p.add_argument("--official")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        args.baseline_hook = args.baseline_hook or None
        run_suite(args)
    elif args.cmd == "synth":
        run_synth(args)
    else:
        load_metric(args.official)
        run_apply(args)


if __name__ == "__main__":
    main()
