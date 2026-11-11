"""Classical ball imputation inside one gap (consecutive frames without a ball), per half.

linear   straight line between the last frame seen before the gap and the first seen after it
hermite  cubic Hermite through the same two points with the ball's velocity at each edge (from the
         last / first `edge_s` seen); clipped to the pitch. Over long gaps it overshoots -- which is
         why the hybrid only uses interpolation for short ones.
A gap touching a half's edge has one side only: the ball is held at that side. A half with no ball
at all stays empty.
"""
from __future__ import annotations

import numpy as np


def gaps(seen: np.ndarray, halves) -> list[tuple[int, int]]:
    """[(first, last)] inclusive missing runs inside the halves."""
    out = []
    for _, a, b in halves:
        miss = ~seen[a:b + 1]
        e = np.diff(np.r_[0, miss.astype(np.int8), 0])
        out += [(a + int(s), a + int(t) - 1) for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]
    return out


def gap_length_s(seen: np.ndarray, halves, fps: int) -> np.ndarray:
    """(n,) length in seconds of the gap each missing frame belongs to (0 where the ball is seen)."""
    out = np.zeros(len(seen), dtype=np.float32)
    for a, b in gaps(seen, halves):
        out[a:b + 1] = (b - a + 1) / fps
    return out


def _edge_velocity(ball, seen, i, direction, fps, edge_s):
    """m/s at frame i from the seen frames up to edge_s away (direction -1 before, +1 after)."""
    k = max(1, int(round(edge_s * fps)))
    j = i + direction * k
    if 0 <= j < len(ball) and seen[j] and seen[i]:
        return (ball[i] - ball[j]) * fps / k * (-direction)
    return np.zeros(2)


def fill_gap(ball, seen, a, b, lo, hi, fps, method="linear", edge_s=0.2, pitch=None):
    """Positions for frames a..b (inclusive) of one gap inside the half [lo, hi]."""
    left, right = a - 1, b + 1
    has_l, has_r = left >= lo and seen[left], right <= hi and seen[right]
    n = b - a + 1
    if not (has_l or has_r):
        return np.full((n, 2), np.nan)
    if not has_r:
        return np.repeat(ball[left][None], n, axis=0)
    if not has_l:
        return np.repeat(ball[right][None], n, axis=0)
    s = (np.arange(a, b + 1) - left) / (right - left)          # 0..1 across the gap, exclusive
    p0, p1 = ball[left], ball[right]
    if method == "linear":
        out = p0 + (p1 - p0) * s[:, None]
    elif method == "hermite":
        dur = (right - left) / fps
        m0 = _edge_velocity(ball, seen, left, -1, fps, edge_s) * dur
        m1 = _edge_velocity(ball, seen, right, +1, fps, edge_s) * dur
        h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
        h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
        out = h00[:, None] * p0 + h10[:, None] * m0 + h01[:, None] * p1 + h11[:, None] * m1
    else:
        raise ValueError(method)
    if pitch is not None:
        out = np.clip(out, [0.0, 0.0], [pitch[0], pitch[1]])
    return out


def fill(ball, halves, fps, method="linear", max_gap_s=None, pitch=None, frames=None):
    """(filled ball, mask of frames filled). Only gaps <= max_gap_s (if given); only `frames` (bool
    mask, e.g. alive) are written."""
    ball = np.asarray(ball, dtype=np.float64)
    seen = ~np.isnan(ball[:, 0])
    out = ball.copy()
    done = np.zeros(len(ball), dtype=bool)
    for _, lo, hi in halves:
        for a, b in gaps(seen, ((0, lo, hi),)):
            if max_gap_s is not None and (b - a + 1) / fps > max_gap_s:
                continue
            seg = fill_gap(ball, seen, a, b, lo, hi, fps, method, pitch=pitch)
            ok = ~np.isnan(seg[:, 0])
            if frames is not None:
                ok &= frames[a:b + 1]
            idx = np.arange(a, b + 1)[ok]
            out[idx] = seg[ok]
            done[idx] = True
    return out, done
