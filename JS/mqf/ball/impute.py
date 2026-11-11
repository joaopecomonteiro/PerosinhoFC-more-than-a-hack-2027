"""Fill the ball on missing frames: interpolation, model, or hybrid (interpolate gaps <= tau).

    python -m mqf.ball.impute --model models/ball_A_off.pt --method hybrid --tau 1     # BAR-ATH, aggregates only

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


@dataclass
class BallFill:
    ball: np.ndarray        # (n, 2) metres
    source: np.ndarray      # (n,) uint8
    gap_s: np.ndarray       # (n,) seconds of the gap a frame sits in, 0 where seen


def load_model(path, device="cpu"):
    import torch
    from . import models
    ck = torch.load(path, map_location=device, weights_only=False)
    m = models.build(ck["config"])
    m.load_state_dict(ck["state"])
    m.eval().to(device)
    return m, ck["config"]


def predict_ball(model, cfg, players, team, ball, fps, pitch, halves, visible=None, device="cpu", batch=8):
    """(n, 2) metres: the model's ball on every frame of the halves (NaN elsewhere)."""
    import torch
    from . import models
    torch.use_deterministic_algorithms(True, warn_only=True)
    step = max(1, int(round(fps / cfg["hz"])))
    T = int(round(cfg["window_s"] * cfg["hz"]))
    L, W = pitch
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    if visible is not None:
        seen &= visible
    out = np.full((n, 2), np.nan)
    tri = np.minimum(np.arange(1, T + 1), np.arange(T, 0, -1)).astype(np.float64)
    for _, a, b in halves:
        frames = np.arange(a, b + 1, step)
        S = len(frames)
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
            if pad:
                xy_k = np.concatenate([xy_k, np.full((pad,) + xy_k.shape[1:], np.nan)])
                bl = np.concatenate([bl, np.full((pad, 2), np.nan)])
                ob = np.r_[ob, np.zeros(pad, dtype=bool)]
            vel = data._velocity(xy_k, cfg["hz"])
            smp = data._pack(xy_k, vel, tm, bl, ob, np.zeros(T, dtype=bool), (L, W))
            jobs.append((s0, len(idx), smp))
        model.eval()
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            b_ = data.collate([c[2] for c in chunk])
            with torch.no_grad():
                pred = model(torch.from_numpy(b_["agents"]).to(device), torch.from_numpy(b_["present"]).bool().to(device),
                             torch.from_numpy(b_["team"]).to(device), torch.from_numpy(b_["ball_in"]).to(device),
                             torch.from_numpy(b_["ball_obs"]).bool().to(device))
                pm = models.to_metres(pred.float(), torch.from_numpy(b_["pitch"]).to(device)).cpu().numpy()
            for (s0, m, _), p in zip(chunk, pm):
                acc[s0:s0 + m] += p[:m] * tri[:m, None]
                wsum[s0:s0 + m] += tri[:m]
        est = acc / np.maximum(wsum, 1e-9)[:, None]
        t_all = np.arange(a, b + 1)
        out[a:b + 1, 0] = np.interp(t_all, frames, est[:, 0])
        out[a:b + 1, 1] = np.interp(t_all, frames, est[:, 1])
    return out


def fill(players, team, ball, fps, pitch, halves, method="linear", tau_s=1.0, model=None, cfg=None,
         frames=None, interp_method="linear", device="cpu", model_ball=None) -> BallFill:
    """method: 'linear' | 'hermite' | 'model' | 'hybrid' (interp_method for gaps <= tau_s, model beyond).
    model_ball: a precomputed predict_ball output (lets the evaluation reuse one inference)."""
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
            f, done = interp.fill(ball, halves, fps, interp_method, max_gap_s=tau_s, pitch=pitch, frames=allow)
            out[done], source[done] = f[done], INTERP
        mb = model_ball if model_ball is not None else predict_ball(model, cfg, players, team, ball, fps, pitch,
                                                                    halves, device=device)
        m = ~seen & ~done & allow & ~np.isnan(mb[:, 0])
        out[m], source[m] = mb[m], MODEL
    else:
        raise ValueError(method)
    return BallFill(out, source, gap_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path)
    ap.add_argument("--method", default="hybrid", choices=["linear", "hermite", "model", "hybrid"])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--interp", default="linear", choices=["linear", "hermite"])
    ap.add_argument("--no-clean-tracks", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    from .. import atd, tracks
    from ..config import CACHE_DIR, competition_game
    spec = competition_game()
    trk = atd.load(spec)
    if not args.no_clean_tracks:
        trk, _ = tracks.clean(trk, spec.fps, spec.pitch)
    model = cfg = None
    if args.method in ("model", "hybrid"):
        if not args.model:
            raise SystemExit("--model is required for model / hybrid")
        model, cfg = load_model(args.model, args.device)
    bf = fill(trk.players, trk.team_of_slot, trk.ball, spec.fps, spec.pitch, spec.halves, args.method,
              args.tau, model, cfg, interp_method=args.interp, device=args.device)
    graded = np.zeros(spec.n_frames, dtype=bool)
    for _, a, b in spec.halves:
        graded[a:b + 1] = True
    miss = graded & np.isnan(trk.ball[:, 0])
    print("=== mqf.ball.impute -- aggregates only, safe to share ===")
    print(f"method {args.method}  tau {args.tau:g} s  model {args.model.name if args.model else '-'}")
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
