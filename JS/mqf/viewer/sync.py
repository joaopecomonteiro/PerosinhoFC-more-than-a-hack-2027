"""Link the broadcast feed to the match (tactical) timeline using the two tracking files only.

The broadcast is live TV: broadcast frame b and the tactical frame t it was recorded alongside
differ by a constant offset, t = b + O, for as long as the recording runs without a cut. Replays
and close-ups do not move that offset -- they replace the live picture for a while -- so the
mapping is a step function with very few steps (typically one per half, plus any recording cut).
What changes is only whether the picture at b *shows* live play.

So we estimate O(b) and then label every broadcast frame:
  1. coarse: occupancy grids of both teams (Gaussian splats on an 18 x 12 cell pitch grid),
     compared only on the cells the broadcast camera sees (from its homography), plus the ball.
     A Viterbi pass over all offsets at 1 s resolution with a large cost for changing the offset
     finds the few constant stretches; replays are short and cannot pay for a change.
     The pitch-axis convention and team order of the broadcast file are not assumed: the 8
     flip / team-swap variants are tried on a subsample and the cheapest is kept.
  2. fine: inside each 30 s window, the frame offset within +/- 2.5 s that minimises the median
     matched distance between broadcast and tactical objects; a running median across windows,
     then exact change points where consecutive windows disagree.
  3. state of each broadcast frame: live (objects match the tactical tracking at the offset),
     off-pitch (no homography / no tracking: close-up, crowd, studio), replay (a pitch view that
     does not match the live tracking), or unlinked (falls outside the tactical recording).

Purely algorithmic, no hand input. It feeds the viewer only, never the submission.
"""
from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

STATE_UNLINKED, STATE_LIVE, STATE_OFF_PITCH, STATE_REPLAY = 0, 1, 2, 3
STANDARD_SIZES = ((640, 360), (960, 540), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160))


@dataclass(frozen=True)
class SyncParams:
    coarse_t_s: float = 1.0      # tactical step of the coarse search
    coarse_ratio: int = 2        # broadcast step = coarse_ratio x tactical step
    grid: tuple = (18, 12)       # occupancy cells (length, width); exact tiling so flips are permutations
    sigma_m: float = 3.0         # splat width
    ball_weight: float = 0.5
    ball_scale_m: float = 20.0
    jump_cost: float = 15.0      # coarse cost of changing the offset (a recording cut)
    min_players: int = 3         # a broadcast frame with fewer tracked players carries no evidence
    window_s: float = 30.0       # fine refinement window
    fine_step: int = 5           # broadcast frames between evaluated frames
    fine_range_s: float = 2.5    # +/- search around the coarse offset
    cap_m: float = 3.0           # cap on one object's matched distance
    live_cost_m: float = 1.5     # mean matched distance under which the picture is live play
    min_gain_m: float = 0.3      # a window's best offset must beat its typical offset by this much


@dataclass
class SyncResult:
    to_match: np.ndarray         # (n_broadcast,) int32 tactical frame per broadcast frame, -1 none
    from_match: np.ndarray       # (n_match,) int32 broadcast frame per tactical frame, -1 none
    state: np.ndarray            # (n_broadcast,) uint8 STATE_*
    offset: np.ndarray           # (n_broadcast,) int32 t - b (meaningful where to_match >= 0)
    image_size: tuple
    variant: str
    seconds: float = 0.0

    def summary(self) -> dict:
        """Aggregates only -- safe to share."""
        linked = self.to_match >= 0
        runs = int(1 + np.count_nonzero(np.diff(self.offset[linked]))) if linked.any() else 0
        live_b = self.state == STATE_LIVE
        covered = self.from_match >= 0
        live_t = np.zeros_like(covered)
        live_t[covered] = live_b[self.from_match[covered]]
        return {"offset_runs": runs, "linked_share": round(float(linked.mean()), 3),
                "live_share": round(float(live_b.mean()), 3),
                "match_covered_share": round(float(covered.mean()), 3),
                "match_live_share": round(float(live_t.mean()), 3),
                "image_size": list(self.image_size), "variant": self.variant,
                "seconds": round(self.seconds, 1)}


# ---------------------------------------------------------------------------------- helpers

def _project(H: np.ndarray, xy_norm: np.ndarray) -> np.ndarray:
    """H (..., 9) applied to normalised pitch points (..., k, 2) -> (..., k, 3) homogeneous."""
    Hm = H.reshape(*H.shape[:-1], 3, 3)
    pts = np.concatenate([xy_norm, np.ones(xy_norm.shape[:-1] + (1,), dtype=xy_norm.dtype)], axis=-1)
    return np.einsum("...ij,...kj->...ki", Hm, pts)


def image_extent(players_b, hom_b, pitch, n_sample: int = 3000) -> tuple[int, int]:
    """Pixel size of the broadcast picture, read from where its own tracked players project."""
    L, W = pitch
    ok = np.flatnonzero(np.isfinite(hom_b).all(axis=1) & np.isfinite(players_b[..., 0]).any(axis=1))
    if not len(ok):
        return STANDARD_SIZES[3]
    rows = ok[np.linspace(0, len(ok) - 1, min(n_sample, len(ok))).astype(int)]
    p = players_b[rows] / np.array([L, W], dtype=np.float32)
    pr = _project(hom_b[rows].astype(np.float64), p.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        u, v = pr[..., 0] / pr[..., 2], pr[..., 1] / pr[..., 2]
    good = np.isfinite(u) & np.isfinite(v)
    if good.sum() < 50:
        return STANDARD_SIZES[3]
    w_est, h_est = np.percentile(u[good], 99.5), np.percentile(v[good], 99.5)
    for w, h in STANDARD_SIZES:
        if w >= 0.97 * w_est and h >= 0.97 * h_est:
            return (w, h)
    return (int(np.ceil(w_est)), int(np.ceil(h_est)))


def _centres(pitch, grid):
    L, W = pitch
    nx, ny = grid
    cx = (np.arange(nx) + 0.5) * L / nx
    cy = (np.arange(ny) + 0.5) * W / ny
    return np.stack(np.meshgrid(cx, cy, indexing="ij"), axis=-1).reshape(-1, 2)   # k = ix*ny + iy


def _occupancy(players, team, centres, sigma, chunk: int = 512) -> np.ndarray:
    """(m, slots, 2) metres -> (m, 2K) float32: per team, sum of Gaussian splats at cell centres."""
    m, K = len(players), len(centres)
    out = np.zeros((m, 2 * K), dtype=np.float32)
    c = centres.astype(np.float32)
    for k in (0, 1):
        P = players[:, team == k].astype(np.float32)
        for a in range(0, m, chunk):
            d2 = ((P[a:a + chunk, :, None, :] - c[None, None]) ** 2).sum(-1)
            out[a:a + chunk, k * K:(k + 1) * K] = np.nansum(np.exp(-d2 / (2 * sigma ** 2)), axis=1)
    return out


def _visible_cells(hom, players_b, pitch, centres, size) -> np.ndarray:
    """(m, K) bool: cell centre projects inside the picture (homography sign taken from the
    frame's own tracked players, since a matrix may be stored up to a factor of -1)."""
    L, W = pitch
    cn = (centres / np.array([L, W])).astype(np.float64)
    pr = _project(hom.astype(np.float64), np.broadcast_to(cn, (len(hom),) + cn.shape))
    pl = _project(hom.astype(np.float64), (players_b / np.array([L, W], dtype=np.float32)).astype(np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sign = np.sign(np.nanmedian(pl[..., 2], axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            u, v = pr[..., 0] / pr[..., 2], pr[..., 1] / pr[..., 2]
    return ((pr[..., 2] * sign[:, None] > 0) & (u >= 0) & (u <= size[0]) & (v >= 0) & (v <= size[1]))


def _variants(grid):
    """8 column permutations of a (2K,) descriptor: x flip, y flip, team swap."""
    nx, ny = grid
    K = nx * ny
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    out = {}
    for fx in (False, True):
        for fy in (False, True):
            cell = ((nx - 1 - ix) if fx else ix) * ny + ((ny - 1 - iy) if fy else iy)
            cell = cell.reshape(-1)
            for swap in (False, True):
                a, b = (cell + K, cell) if swap else (cell, cell + K)
                name = ("x" if fx else "") + ("y" if fy else "") + ("t" if swap else "")
                out[name or "none"] = (np.concatenate([a, b]), cell)
    return out


# ---------------------------------------------------------------------------------- coarse

def _coarse_path(Db, Mb, Bb, inf, Dt, Bt, ratio, p: SyncParams):
    """Viterbi over offsets o = j - ratio*i (tactical samples j, broadcast samples i).
    Returns (path offsets per broadcast sample, total cost)."""
    nb, nt = len(Db), len(Dt)
    o_min, o_max = -ratio * (nb - 1), nt - 1
    n_off = o_max - o_min + 1
    Dt2T = (Dt ** 2).T.copy()
    DtT = Dt.T.copy()
    bad = 1.0 + p.ball_weight
    V = np.zeros(n_off, dtype=np.float64)
    jumps = np.zeros((nb, (n_off + 7) // 8), dtype=np.uint8)
    argmins = np.zeros(nb, dtype=np.int64)
    t_seen = np.isfinite(Bt[:, 0])
    chunk = 128
    for a in range(0, nb, chunk):
        rows = np.arange(a, min(a + chunk, nb))
        M2 = np.concatenate([Mb[rows], Mb[rows]], axis=1).astype(np.float32)
        X = (M2 * Db[rows]) @ DtT
        S2 = M2 @ Dt2T
        S1 = (M2 * Db[rows] ** 2).sum(1)
        C = 1.0 - 2.0 * X / (S1[:, None] + S2 + 1e-6)
        bb = Bb[rows]
        dist = np.sqrt(((bb[:, None, :] - Bt[None, :, :]) ** 2).sum(-1))
        both = np.isfinite(bb[:, 0])[:, None] & t_seen[None, :]
        C += p.ball_weight * np.where(both, np.minimum(dist / p.ball_scale_m, 1.0), 0.5)
        for r, i in enumerate(rows):
            if inf[i]:
                e = np.full(n_off, bad)
                k0 = ratio * i + o_min                    # k = j - k0 ... j = k + k0
                lo, hi = max(0, -k0), min(n_off, nt - k0)
                if hi > lo:
                    e[lo:hi] = C[r, lo + k0:hi + k0]
            else:
                e = 0.0
            if i:
                m = int(np.argmin(V))
                jc = V[m] + p.jump_cost
                jump = jc < V
                V = np.where(jump, jc, V)
                jumps[i] = np.packbits(jump)
                argmins[i] = m
            V = V + e
    path = np.zeros(nb, dtype=np.int64)
    k = int(np.argmin(V))
    total = float(V[k])
    for i in range(nb - 1, -1, -1):
        path[i] = k
        if i and (jumps[i, k >> 3] >> (7 - (k & 7))) & 1:
            k = int(argmins[i])
    return path + o_min, total


# ---------------------------------------------------------------------------------- fine

class _Side:
    """Tracking split by team once (indexing a full-length array per call would copy it)."""

    def __init__(self, players, ball, team_of_slot, team_order=(0, 1)):
        self.team = [np.ascontiguousarray(players[:, team_of_slot == k]) for k in team_order]
        self.ball = ball
        self.n = len(players)


def _frame_costs(T: _Side, B: _Side, fb, tt, p: SyncParams) -> np.ndarray:
    """Mean capped distance from each broadcast object to its nearest tactical object of the same
    team (ball counted twice). fb (F,) broadcast frames, tt (F, D) tactical frames -> (F, D)."""
    F, D = tt.shape
    num = np.zeros((F, D), dtype=np.float32)
    cnt = np.zeros((F, 1), dtype=np.float32)
    ttc = np.clip(tt, 0, T.n - 1)
    for k in (0, 1):
        Pb = B.team[k][fb]                                   # (F, sb, 2)
        keep = np.isfinite(Pb[..., 0]).any(axis=0)
        Pb = Pb[:, keep]
        if not Pb.shape[1]:
            continue
        Pt = T.team[k][ttc]                                  # (F, D, st, 2)
        d = np.sqrt(((Pb[:, None, :, None, :] - Pt[:, :, None, :, :]) ** 2).sum(-1))   # (F, D, sb, st)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dmin = np.nanmin(d, axis=-1)                     # (F, D, sb)
        dmin = np.where(np.isfinite(dmin), np.minimum(dmin, p.cap_m), p.cap_m)
        valid = np.isfinite(Pb[..., 0])                      # (F, sb)
        num += np.where(valid[:, None, :], dmin, 0).sum(-1)
        cnt += valid.sum(-1, keepdims=True)
    bb = B.ball[fb]
    bt = T.ball[ttc]
    db = np.sqrt(((bb[:, None, :] - bt) ** 2).sum(-1))
    both = np.isfinite(db)
    num += np.where(both, 2 * np.minimum(db, p.cap_m), 0)
    cnt = cnt + 2 * both
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / cnt
    out[(cnt == 0) | (tt < 0) | (tt >= T.n)] = np.nan
    return out


def _costs_chunked(T, B, fb, tt, p, chunk: int = 40):
    return np.concatenate([_frame_costs(T, B, fb[a:a + chunk], tt[a:a + chunk], p)
                           for a in range(0, len(fb), chunk)]) if len(fb) else np.zeros((0, tt.shape[1]))


def _running_median(v: np.ndarray, k: int = 5) -> np.ndarray:
    h = k // 2
    out = v.copy()
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - h):i + h + 1])
    return out


# ---------------------------------------------------------------------------------- main

def align(trk_t, trk_b, hom_b, pitch, fps, params: SyncParams = SyncParams(), log=print) -> SyncResult:
    p = params
    t0 = time.time()
    nt, nb = trk_t.n_frames, trk_b.n_frames
    hom_ok = np.isfinite(hom_b).all(axis=1)
    n_pl = np.isfinite(trk_b.players[..., 0]).sum(axis=1)
    informative = hom_ok & (n_pl >= p.min_players)
    size = image_extent(trk_b.players, hom_b, pitch)

    # ---- coarse
    st = max(1, int(round(p.coarse_t_s * fps)))
    sb = st * p.coarse_ratio
    tj = np.arange(0, nt, st)
    bi = np.arange(0, nb, sb)
    centres = _centres(pitch, p.grid)
    Dt = _occupancy(trk_t.players[tj], trk_t.team_of_slot, centres, p.sigma_m)
    Bt = trk_t.ball[tj].astype(np.float32)
    inf = informative[bi]
    Db0 = np.zeros((len(bi), Dt.shape[1]), dtype=np.float32)
    M0 = np.zeros((len(bi), len(centres)), dtype=bool)
    if inf.any():
        Db0[inf] = _occupancy(trk_b.players[bi[inf]], trk_b.team_of_slot, centres, p.sigma_m)
        M0[inf] = _visible_cells(hom_b[bi[inf]], trk_b.players[bi[inf]], pitch, centres, size)
    Bb0 = trk_b.ball[bi].astype(np.float32)
    L, W = pitch

    variants = _variants(p.grid)
    sub = np.arange(0, len(bi), 4)
    best = None
    for name, (cols, cells) in variants.items():
        fx, fy = "x" in name, "y" in name
        Bb = Bb0.copy()
        if fx:
            Bb[:, 0] = L - Bb[:, 0]
        if fy:
            Bb[:, 1] = W - Bb[:, 1]
        Db, Mb = np.empty_like(Db0), np.empty_like(M0)
        Db[:, cols] = Db0
        Mb[:, cells] = M0
        _, cost = _coarse_path(Db[sub], Mb[sub], Bb[sub], inf[sub], Dt, Bt, p.coarse_ratio * 4, p)
        if best is None or cost < best[1]:
            best = (name, cost, Db, Mb, Bb)
    name, _, Db, Mb, Bb = best
    path, _ = _coarse_path(Db, Mb, Bb, inf, Dt, Bt, p.coarse_ratio, p)
    coarse_off = path * st                                        # t = b + coarse_off, per sample
    log(f"  coarse alignment done ({time.time() - t0:.0f} s, variant {name})")

    # the fine stage compares positions directly: bring the broadcast into tactical axes
    fx, fy, swap = "x" in name, "y" in name, "t" in name
    pb = trk_b.players.copy()
    ball_b = trk_b.ball.copy()
    if fx:
        pb[..., 0] = L - pb[..., 0]; ball_b[:, 0] = L - ball_b[:, 0]
    if fy:
        pb[..., 1] = W - pb[..., 1]; ball_b[:, 1] = W - ball_b[:, 1]
    T = _Side(trk_t.players, trk_t.ball, trk_t.team_of_slot)
    B = _Side(pb, ball_b, trk_b.team_of_slot, (1, 0) if swap else (0, 1))

    # ---- fine, per window
    wlen = int(p.window_s * fps)
    R = int(p.fine_range_s * fps)
    starts = np.arange(0, nb, wlen)
    sample_of = np.minimum(np.arange(nb) // sb, len(bi) - 1)
    win_off = np.full(len(starts), np.nan)
    win_coarse = np.zeros(len(starts), dtype=np.int64)
    for w, a in enumerate(starts):
        b1 = min(a + wlen, nb)
        vals, counts = np.unique(coarse_off[sample_of[a:b1]], return_counts=True)
        O = int(vals[np.argmax(counts)])
        win_coarse[w] = O
        fb = np.arange(a, b1, p.fine_step)
        fb = fb[informative[fb]]
        if len(fb) < 10:
            continue
        deltas = np.arange(-R, R + 1, 5)
        c = _costs_chunked(T, B, fb, fb[:, None] + O + deltas[None, :], p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            score = np.nanmedian(c, axis=0)
        if not np.isfinite(score).any():
            continue
        d0 = deltas[int(np.nanargmin(score))]
        fine = np.arange(d0 - 4, d0 + 5)
        c2 = _costs_chunked(T, B, fb, fb[:, None] + O + fine[None, :], p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s2 = np.nanmedian(c2, axis=0)
            typical = np.nanmedian(score)
        if not np.isfinite(s2).any():
            continue
        k = int(np.nanargmin(s2))
        if s2[k] <= p.live_cost_m and typical - s2[k] >= p.min_gain_m:
            win_off[w] = O + fine[k]

    # smooth inside each coarse stretch; windows without evidence inherit from their neighbours
    final = win_coarse.astype(np.float64).copy()
    seg_id = np.r_[0, np.cumsum(np.diff(win_coarse) != 0)]
    for s in np.unique(seg_id):
        idx = np.flatnonzero(seg_id == s)
        conf = idx[np.isfinite(win_off[idx])]
        if not len(conf):
            continue
        final[conf] = _running_median(win_off[conf], 5)
        for i in idx[~np.isfinite(win_off[idx])]:
            final[i] = final[conf[np.argmin(np.abs(conf - i))]]
    final = final.astype(np.int64)

    offset = np.repeat(final, wlen)[:nb]
    # exact change points between windows that disagree
    for w in range(len(starts) - 1):
        oa, ob = final[w], final[w + 1]
        if oa == ob:
            continue
        a, b1 = starts[w], min(starts[w + 1] + wlen, nb)
        fb = np.arange(a, b1, p.fine_step)
        fb = fb[informative[fb]]
        if len(fb) < 2:
            continue
        cA = _costs_chunked(T, B, fb, (fb + oa)[:, None], p)[:, 0]
        cB = _costs_chunked(T, B, fb, (fb + ob)[:, None], p)[:, 0]
        diff = np.nan_to_num(cA - cB, nan=0.0)                # > 0 where the second offset fits better
        if not np.any(diff):
            continue
        # switch before sample s: cost = sum(cA[:s]) + sum(cB[s:]) = sum(cB) + sum(diff[:s])
        s = int(np.argmin(np.r_[0.0, np.cumsum(diff)]))
        switch = fb[s] if s < len(fb) else b1
        offset[a:switch] = oa
        offset[switch:b1] = ob

    # ---- states
    to_match = np.arange(nb, dtype=np.int64) + offset
    linked = (to_match >= 0) & (to_match < nt)
    state = np.full(nb, STATE_OFF_PITCH, dtype=np.uint8)
    fb = np.arange(0, nb, p.fine_step)
    ev = fb[informative[fb] & linked[fb]]
    if len(ev):
        c = _costs_chunked(T, B, ev, (ev + offset[ev])[:, None], p)[:, 0]
        live = np.where(np.isfinite(c), c <= p.live_cost_m, False).astype(float)
        live = _smooth_flags(live, 5) > 0.5                   # majority over ~1 s
        sample_state = np.where(live, STATE_LIVE, STATE_REPLAY).astype(np.uint8)
        pos = np.searchsorted(ev, np.arange(nb), side="right") - 1
        near = (pos >= 0) & (np.arange(nb) - ev[np.clip(pos, 0, None)] < p.fine_step)
        state[near & informative] = sample_state[pos[near & informative]]
    state[~linked] = STATE_UNLINKED
    to_match = np.where(linked, to_match, -1).astype(np.int32)

    from_match = np.full(nt, -1, dtype=np.int32)
    bs = np.flatnonzero(linked)[::-1]                          # reversed so the earliest b wins
    from_match[to_match[bs]] = bs
    # 1-2 frame seams where the offset steps by a frame or two: borrow a neighbour's picture.
    # Longer holes (a recording cut, halftime) stay empty -- there is no picture for them.
    orig = from_match.copy()
    for t in np.flatnonzero(orig < 0):
        for d in (1, -1, 2, -2):
            if 0 <= t + d < nt and orig[t + d] >= 0:
                from_match[t] = orig[t + d]
                break

    res = SyncResult(to_match, from_match, state, offset.astype(np.int32), size, name, time.time() - t0)
    return res


def _smooth_flags(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < k:
        return x
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _digest(*arrays) -> str:
    h = hashlib.md5()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:12]


def align_cached(trk_t, trk_b, hom_b, spec_b, n_match, fps, force=False, log=print,
                 params: SyncParams = SyncParams()) -> SyncResult:
    """align(), cached in cache/viewer/ keyed by the inputs and this file's code."""
    from .game import VIEWER_CACHE
    key = _digest(trk_t.players, trk_t.ball, trk_b.players, trk_b.ball, hom_b,
                  np.frombuffer(repr(asdict(params)).encode(), dtype=np.uint8))
    path = VIEWER_CACHE / f"sync_{spec_b.name}_{key}.npz"
    if path.exists() and not force:
        z = np.load(path)
        return SyncResult(z["to_match"], z["from_match"], z["state"], z["offset"],
                          tuple(int(v) for v in z["image_size"]), str(z["variant"]), float(z["seconds"]))
    log("  aligning broadcast to tactical (one-off, cached afterwards)")
    res = align(trk_t, trk_b, hom_b, spec_b.pitch, fps, params, log)
    VIEWER_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, to_match=res.to_match, from_match=res.from_match, state=res.state,
                        offset=res.offset, image_size=np.array(res.image_size), variant=res.variant,
                        seconds=res.seconds)
    return res
