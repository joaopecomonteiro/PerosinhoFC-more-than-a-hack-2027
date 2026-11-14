"""Fill the ball on missing frames: interpolation, model, or hybrid (interpolate gaps <= tau).

    python -m mqf.ball.impute            # BAR-ATH with the frozen setting (FROZEN), aggregates only

`fill(...)` returns a BallFill: the ball with gaps filled, the source of every frame (0 seen,
1 interpolated, 2 model, 3 left empty) and the length of the gap each frame sits in. Only frames in
`frames` (e.g. alive) are written -- the model is meant for the game in play only.

Model inference: sliding 48 s windows at 6.25 Hz with 50 % overlap, triangular weights, then linear
upsampling to the native frame rate; seen frames are never changed. Players are split into pieces
at their gaps (no identities), the same as a cleaned ATD gives. Deterministic on CPU.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data, interp

SEEN, INTERP, MODEL, EMPTY = 0, 1, 2, 3

# Frozen after EXPERIMENTS #15 / #17 (validation games only): linear <= 1 s, Hermite <= 2 s, the v5
# axial model (IDSSE + Metrica + PFF) anchored to the gap edges beyond. Decoder weight for its
# frames: decode.Params.w_filled.
FROZEN = {"model": "ball_B_off_v5.pt", "method": "hybrid", "tau_s": 2.0, "tau_linear_s": 1.0,
          "interp_method": "hermite", "anchor_s": 2.0}


@dataclass
class BallFill:
    ball: np.ndarray        # (n, 2) metres
    source: np.ndarray      # (n,) uint8
    gap_s: np.ndarray       # (n,) seconds of the gap a frame sits in, 0 where seen


def load_model(path, device="cpu", dtype="float32"):
    """dtype 'float64': the submission's reproducible CPU inference (see fill's round_m)."""
    import torch
    from . import models
    ck = torch.load(path, map_location=device, weights_only=False)
    m = models.build(ck["config"])
    m.load_state_dict(ck["state"])
    m.eval().to(device)
    if dtype == "float64":
        m.double()
    return m, ck["config"]


def predict_ball(model, cfg, players, team, ball, fps, pitch, halves, visible=None, device="cpu", batch=8):
    """(n, 2) metres: the model's ball on every frame of the halves (NaN elsewhere). Runs in the
    model's own dtype (a model loaded with dtype='float64' gets float64 inputs)."""
    import torch
    from . import models
    torch.use_deterministic_algorithms(True, warn_only=True)
    fdtype = next(model.parameters()).dtype
    step = max(1, int(round(fps / cfg["hz"])))
    hz = fps / step                         # 6.25 at 25 fps; PFF 29.97 fps -> 5.994 (as in training)
    T = int(round(cfg["window_s"] * cfg["hz"]))
    L, W = pitch
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    if visible is not None:
        seen &= visible
    out = np.full((n, 2), np.nan)
    tri = np.minimum(np.arange(1, T + 1), np.arange(T, 0, -1)).astype(np.float64)
    use_ctx = bool(cfg.get("ctx", False))
    ball_vis = np.where(seen[:, None], np.asarray(ball, dtype=np.float64), np.nan)
    for _, a, b in halves:
        frames = np.arange(a, b + 1, step)
        S = len(frames)
        # v3: the same context as training, read over the whole half on the step grid
        an_s, dp_s, dn_s = data.ball_context(ball_vis[frames], seen[frames], hz)
        vp_s, vn_s = data.edge_velocity(ball_vis[frames], seen[frames], hz)
        starts = list(range(0, max(1, S - T + 1), T // 2))
        if S > T and starts[-1] != S - T:
            starts.append(S - T)
        acc, wsum = np.zeros((S, 2)), np.zeros(S)
        jobs = []
        for s0 in starts:
            idx = frames[s0:s0 + T]
            pad = T - len(idx)
            xy = players[idx].astype(np.float64)
            keep = ~np.all(np.isnan(xy[..., 0]), axis=0)
            xy_k, tm = data.split_pieces(xy[:, keep], team[keep])
            life = (~np.isnan(xy_k[..., 0])).sum(axis=0)
            order = np.argsort(-life, kind="stable")[:cfg["a_max"]]
            order = order[life[order] > 0]
            xy_k, tm = xy_k[:, order], tm[order]
            bl, ob = ball[idx].astype(np.float64), seen[idx]
            w = slice(s0, s0 + T)
            cx = (an_s[w], dp_s[w], dn_s[w], vp_s[w], vn_s[w])
            if pad:
                xy_k = np.concatenate([xy_k, np.full((pad,) + xy_k.shape[1:], np.nan)])
                bl = np.concatenate([bl, np.full((pad, 2), np.nan)])
                ob = np.r_[ob, np.zeros(pad, dtype=bool)]
                cx = (np.concatenate([cx[0], np.full((pad, 2), np.nan)]), np.r_[cx[1], np.full(pad, np.inf)],
                      np.r_[cx[2], np.full(pad, np.inf)], np.concatenate([cx[3], np.zeros((pad, 2))]),
                      np.concatenate([cx[4], np.zeros((pad, 2))]))
            vel = data._velocity(xy_k, hz)
            smp = data._pack(xy_k, vel, tm, bl, ob, np.zeros(T, dtype=bool), (L, W), ctx=cx)
            jobs.append((s0, len(idx), smp))
        model.eval()
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            b_ = data.collate([c[2] for c in chunk])
            tb = {k: torch.from_numpy(np.asarray(b_[k])).to(device)
                  for k in ("agents", "present", "team", "ball_in", "ball_obs", "anchor", "ctx", "pitch")}
            tb["present"], tb["ball_obs"] = tb["present"].bool(), tb["ball_obs"].bool()
            for k in ("agents", "ball_in", "anchor", "ctx", "pitch"):
                tb[k] = tb[k].to(fdtype)
            with torch.no_grad():
                pred = models.forward(model, tb)
                pm = models.to_metres(pred.to(fdtype), tb["pitch"]).cpu().numpy().astype(np.float64)
            for (s0, m, _), p in zip(chunk, pm):
                acc[s0:s0 + m] += p[:m] * tri[:m, None]
                wsum[s0:s0 + m] += tri[:m]
        est = acc / np.maximum(wsum, 1e-9)[:, None]
        t_all = np.arange(a, b + 1)
        if use_ctx and not np.isnan(an_s[:, 0]).all():
            # correction on the step grid, upsampled, added to the native-rate interpolation: the
            # fine structure next to the gap edges comes from the 25 fps ball, not from 6.25 Hz
            corr = est - an_s
            nat, _ = interp.fill(ball_vis, ((0, a, b),), fps, "linear")
            for c in range(2):
                out[a:b + 1, c] = nat[a:b + 1, c] + np.interp(t_all, frames, corr[:, c])
        else:
            out[a:b + 1, 0] = np.interp(t_all, frames, est[:, 0])
            out[a:b + 1, 1] = np.interp(t_all, frames, est[:, 1])
    return clip_to_pitch(out, pitch)        # alive ball is on the pitch; v0 reads off-pitch as dead


def clip_to_pitch(xy: np.ndarray, pitch) -> np.ndarray:
    return np.clip(xy, [0.0, 0.0], [pitch[0], pitch[1]])


def anchor(model_ball, ball, halves, fps, tau_s: float = 2.0, pitch=None) -> np.ndarray:
    """The model ball made to meet the seen ball at every gap edge.

    The model's own output on the seen edge frames is a reconstruction, a few metres off, and v1
    showed it barely uses the ball context (val ADE <= 2 s gaps ~= > 8 s gaps). So per gap the edge
    residuals (seen - model) are blended linearly across the gap, in full up to tau_s from the nearer
    edge, then faded with exp(-(distance - tau_s) / tau_s): gaps <= 2 tau_s get the model's shape
    plus residual interpolation (exact for a constant model bias), the middle of long gaps stays the
    model. One-sided gaps use the one edge."""
    out = np.array(model_ball, dtype=np.float64, copy=True)
    ball = np.asarray(ball, dtype=np.float64)
    seen = ~np.isnan(ball[:, 0])
    k = tau_s * fps
    for _, lo, hi in halves:
        e = np.diff(np.r_[0, (~seen[lo:hi + 1]).astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(e == 1) + lo, np.flatnonzero(e == -1) - 1 + lo):
            t = np.arange(a, b + 1)
            L, R = a - 1, b + 1
            oL = ball[L] - model_ball[L] if L >= lo and not np.isnan(model_ball[L, 0]) else None
            oR = ball[R] - model_ball[R] if R <= hi and not np.isnan(model_ball[R, 0]) else None
            if oL is not None and oR is not None:
                u = ((t - L) / (R - L))[:, None]
                off, d = (1 - u) * oL + u * oR, np.minimum(t - L, R - t)
            elif oL is not None:
                off, d = np.broadcast_to(oL, (len(t), 2)), t - L
            elif oR is not None:
                off, d = np.broadcast_to(oR, (len(t), 2)), R - t
            else:
                continue
            out[a:b + 1] += off * np.exp(-np.maximum(d - k, 0.0) / k)[:, None]
    return clip_to_pitch(out, pitch) if pitch is not None else out


def fill(players, team, ball, fps, pitch, halves, method="linear", tau_s=1.0, model=None, cfg=None,
         frames=None, interp_method="linear", device="cpu", model_ball=None, anchor_s=None,
         tau_linear_s=None, round_m=None) -> BallFill:
    """method: 'linear' | 'hermite' | 'model' | 'hybrid' (interp_method for gaps <= tau_s, model beyond).
    tau_linear_s (hybrid): linear for gaps <= tau_linear_s first, then interp_method up to tau_s.
    model_ball: a precomputed predict_ball output (lets the evaluation reuse one inference).
    anchor_s: if set, the model ball is anchored to the gap edges first (see `anchor`).
    round_m: if set, filled positions are rounded to this grid (0.01 for the submission: float64
    inference that differs across machines in the last bits cannot change a label)."""
    ball = np.asarray(ball, dtype=np.float64)
    seen = ~np.isnan(ball[:, 0])
    gap_s = interp.gap_length_s(seen, halves, fps)
    out = ball.copy()
    source = np.where(seen, SEEN, EMPTY).astype(np.uint8)
    allow = np.ones(len(ball), dtype=bool) if frames is None else np.asarray(frames, dtype=bool)
    if method in ("linear", "hermite"):
        f, done = interp.fill(ball, halves, fps, method, pitch=pitch, frames=allow)
        out[done], source[done] = f[done], INTERP
    elif method in ("model", "hybrid"):
        done = np.zeros(len(ball), dtype=bool)
        if method == "hybrid":
            if tau_linear_s:
                f, done = interp.fill(ball, halves, fps, "linear", max_gap_s=tau_linear_s, pitch=pitch, frames=allow)
                out[done], source[done] = f[done], INTERP
            f, d2 = interp.fill(ball, halves, fps, interp_method, max_gap_s=tau_s, pitch=pitch, frames=allow)
            d2 &= ~done
            out[d2], source[d2] = f[d2], INTERP
            done |= d2
        mb = model_ball if model_ball is not None else predict_ball(model, cfg, players, team, ball, fps, pitch,
                                                                    halves, device=device)
        if anchor_s:
            mb = anchor(mb, ball, halves, fps, anchor_s, pitch)
        m = ~seen & ~done & allow & ~np.isnan(mb[:, 0])
        out[m], source[m] = mb[m], MODEL
    else:
        raise ValueError(method)
    if round_m:
        f = (source == INTERP) | (source == MODEL)
        out[f] = np.round(out[f] / round_m) * round_m
    return BallFill(out, source, gap_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    from ..config import MODELS_DIR
    F = FROZEN
    ap.add_argument("--model", type=Path, default=MODELS_DIR / F["model"])
    ap.add_argument("--method", default=F["method"], choices=["linear", "hermite", "model", "hybrid"])
    ap.add_argument("--tau", type=float, default=F["tau_s"])
    ap.add_argument("--tau-linear", type=float, default=F["tau_linear_s"])
    ap.add_argument("--interp", default=F["interp_method"], choices=["linear", "hermite"])
    ap.add_argument("--anchor", type=float, default=F["anchor_s"], help="anchor the model ball to the gap edges (fade, s)")
    ap.add_argument("--no-clean-tracks", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    from .. import atd, tracks
    from ..config import CACHE_DIR, competition_game
    spec = competition_game()
    trk = atd.load(spec)
    if not args.no_clean_tracks:
        trk, _ = tracks.clean(trk, spec.fps, spec.pitch, halves=spec.halves)
    model = cfg = None
    if args.method in ("model", "hybrid"):
        if not args.model:
            raise SystemExit("--model is required for model / hybrid")
        model, cfg = load_model(args.model, args.device)
    bf = fill(trk.players, trk.team_of_slot, trk.ball, spec.fps, spec.pitch, spec.halves, args.method,
              args.tau, model, cfg, interp_method=args.interp, device=args.device, anchor_s=args.anchor,
              tau_linear_s=args.tau_linear)
    graded = np.zeros(spec.n_frames, dtype=bool)
    for _, a, b in spec.halves:
        graded[a:b + 1] = True
    miss = graded & np.isnan(trk.ball[:, 0])
    print("=== mqf.ball.impute -- aggregates only, safe to share ===")
    print(f"method {args.method}  linear <= {args.tau_linear:g} s, {args.interp} <= {args.tau:g} s, "
          f"model {args.model.name if args.model else '-'} (anchor {args.anchor:g} s) beyond")
    print(f"missing graded frames: {miss.mean() / graded.mean():.3f} of graded")
    for name, code in (("interpolated", INTERP), ("model", MODEL), ("left empty", EMPTY)):
        print(f"  {name:<12} {np.mean(bf.source[miss] == code):.3f} of missing")
    for lo, hi in ((0, 0.5), (0.5, 2), (2, 10), (10, np.inf)):
        sel = miss & (bf.gap_s > lo) & (bf.gap_s <= hi)
        if sel.any():
            print(f"  gaps {lo:g}-{hi:g} s: {sel.sum() / miss.sum():.3f} of missing  "
                  f"interp {np.mean(bf.source[sel] == INTERP):.2f}  model {np.mean(bf.source[sel] == MODEL):.2f}")
    filled = miss & (bf.source != EMPTY)
    if filled.any():
        d = np.linalg.norm(trk.players[filled] - bf.ball[filled][:, None], axis=-1)
        with np.errstate(all="ignore"):
            near = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
        near = near[np.isfinite(near)]
        print(f"  filled ball to nearest player: median {np.median(near):.1f} m, p90 {np.percentile(near, 90):.1f} m")
    out = CACHE_DIR / "ball" / f"{spec.name}_{args.method}_{args.tau:g}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ball=bf.ball.astype(np.float32), source=bf.source, gap_s=bf.gap_s)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
