"""Ball in play or dead: a two-state semi-Markov decoder over a per-frame P(dead).

A per-frame threshold flickers, and a plain HMM charges one price per switch, so a 2 s and a 2 min
stoppage cost the same. Here both states have a real LENGTH distribution -- how long stoppages and
stretches of play last on the training games of other teams (stored with the model) -- and the
best split of each half into stoppages and play is found exactly (semi-Markov Viterbi). 5 Hz steps:
boundaries within 0.2 s, and minutes-long stoppages stay cheap.

Emission: w x the log likelihood ratio of P(dead) against the model's training prior, summed over
the frames of a step (frames are strongly correlated, hence w < 1). The first and last run of a half
are cut by its edges and use the survival function (our 2nd half opens with a long stoppage).
"""
from __future__ import annotations

import numpy as np

HZ = 5.0
DMAX_S = 900.0
BANDWIDTH = 0.25


def length_tables(lengths_s, hz: float = HZ, dmax_s: float = DMAX_S, bw: float = BANDWIDTH):
    """(log pmf, log survival) over lengths of 0..dmax steps (index 0 unused): a kernel density in
    log-seconds, so short and long runs are both smoothed sensibly."""
    dmax = int(round(dmax_s * hz))
    d = np.arange(1, dmax + 1) / hz
    L = np.log(np.clip(np.asarray(lengths_s, dtype=np.float64), 1.0 / hz, None))
    x = np.log(d)
    dens = np.zeros(dmax)
    for chunk in np.array_split(L, max(1, len(L) // 2000)):
        dens += np.exp(-0.5 * ((x[None, :] - chunk[:, None]) / bw) ** 2).sum(axis=0)
    pmf = np.maximum(dens / d, 1e-300)
    pmf = np.maximum(pmf / pmf.sum(), 1e-12)
    pmf /= pmf.sum()
    surv = np.cumsum(pmf[::-1])[::-1]
    return np.r_[-np.inf, np.log(pmf)], np.r_[0.0, np.log(surv)]


def viterbi(ll: np.ndarray, logp, logs) -> np.ndarray:
    """ll (2, m): step log-likelihoods, 0 = in play, 1 = dead. Exact best split -> (m,) states."""
    m = ll.shape[1]
    dmax = len(logp[0]) - 1
    C = np.zeros((2, m + 1))
    C[:, 1:] = np.cumsum(ll, axis=1)
    F = np.full((2, m + 1), -np.inf)
    arg = np.zeros((2, m + 1), dtype=np.int64)
    for t in range(1, m + 1):
        dm = min(t - 1, dmax)
        ds = np.arange(1, dm + 1)
        for s in (0, 1):
            best, bd = logs[s][min(t, dmax)] + C[s, t], t            # first run, cut by the half's start
            if dm:
                cand = F[1 - s, t - ds] + logp[s][ds] + C[s, t] - C[s, t - ds]
                j = int(np.argmax(cand))
                if cand[j] > best:
                    best, bd = cand[j], int(ds[j])
            F[s, t], arg[s, t] = best, bd
    best, bs, bd = -np.inf, 0, m                                     # last run, cut by the half's end
    for s in (0, 1):
        v = logs[s][min(m, dmax)] + C[s, m]
        if v > best:
            best, bs, bd = v, s, m
        dm = min(m - 1, dmax)
        if dm:
            ds = np.arange(1, dm + 1)
            cand = F[1 - s, m - ds] + logs[s][ds] + C[s, m] - C[s, m - ds]
            j = int(np.argmax(cand))
            if cand[j] > best:
                best, bs, bd = cand[j], s, int(ds[j])
    out = np.empty(m, dtype=np.int8)
    t, s, d = m, bs, bd
    while t > 0:
        out[t - d:t] = s
        t -= d
        if t > 0:
            s = 1 - s
            d = int(arg[s, t])
    return out


def decode(p_dead: np.ndarray, halves, fps: float, tables, w: float = 0.2, prior: float = 0.45,
           hz: float = HZ) -> np.ndarray:
    """(n,) bool, True where the ball is dead (inside the halves; False outside)."""
    n = len(p_dead)
    out = np.zeros(n, dtype=bool)
    (lp0, ls0), (lp1, ls1) = tables
    k = max(1, int(round(fps / hz)))
    eps = 1e-4
    for _, a, b in halves:
        p = np.clip(np.nan_to_num(p_dead[a:b + 1], nan=prior), eps, 1 - eps)
        l1 = w * (np.log(p) - np.log(prior))
        l0 = w * (np.log(1 - p) - np.log(1 - prior))
        m = int(np.ceil(len(p) / k))
        pad = m * k - len(p)
        ll = np.vstack([np.r_[l0, np.zeros(pad)].reshape(m, k).sum(1),
                        np.r_[l1, np.zeros(pad)].reshape(m, k).sum(1)])
        out[a:b + 1] = np.repeat(viterbi(ll, (lp0, lp1), (ls0, ls1)), k)[:len(p)].astype(bool)
    return out
