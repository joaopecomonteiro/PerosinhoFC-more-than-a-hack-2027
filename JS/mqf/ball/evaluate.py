"""The ball-imputation protocol on the VALIDATION games (other games only) -- decides method and tau.

    python -m mqf.ball.evaluate --tag v1                                   # every models/ball_*.pt found
    python -m mqf.ball.evaluate --models models/ball_A_off.pt --taus 0.5 1 2 4 --seeds 0 1 2 --tag v1

For each validation game (IDSSE J03WMX, J03WQQ, METRICA-3), player condition (clean as recorded /
'on': dropouts and broken tracks like our ATD) and mask seed:
  buckets   gaps of exactly 0.5, 1, 2, 4, 8, 16, 32 s cut into the ball on ALIVE frames (both edges
            seen, >= 10 s apart). Per method: error on the hidden frames (mean / median / p90, m)
            and team-nearest agreement -- the team of the player nearest to the filled ball equals
            the team nearest to the true ball (what possession reads).
  mixed     gaps of log-uniform 0.2-30 s until 25 % of alive frames are hidden. Methods filled on
            the true alive frames only (oracle alive), then the v0 pipeline and the OFFICIAL metric:
            spellF1 / macroF1 against no fill and against the true ball (ceiling); hybrids for
            every tau, and interpolation capped at tau (longer gaps left empty). Pick by downstream
            spellF1; ties by team-nearest agreement.
Writes harness_results/ball_<tag>.json. Aggregates on other games only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import decode, features
from ..atd import Tracking
from ..config import HARNESS_DIR, MODELS_DIR
from ..external.bundle import speed_from_positions
from ..harness.evaluate import score
from . import data, impute, interp

BUCKETS_S = (0.5, 1, 2, 4, 8, 16, 32)
PER_BUCKET = 40
SEP_S = 10.0
MIXED_SHARE = 0.25


def place_gaps(ok: np.ndarray, lengths, rng, sep: int, occupied=None) -> list[tuple[int, int]]:
    """Gaps (a, b) inclusive with frames a-1 and b+1 also `ok`; `sep` frames kept free around each."""
    occ = np.zeros(len(ok), dtype=bool) if occupied is None else occupied
    out = []
    for n in lengths:
        free = ok & ~occ
        e = np.diff(np.r_[0, free.astype(np.int8), 0])
        s, t = np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1
        room = (t - s + 1) - (n + 2)
        cand = room >= 0
        if not cand.any():
            continue
        s, room = s[cand], room[cand]
        r = int(rng.choice(len(s), p=(room + 1) / (room + 1).sum()))
        a = int(s[r] + 1 + rng.integers(0, room[r] + 1))
        b = a + n - 1
        out.append((a, b))
        occ[max(0, a - 1 - sep):b + 2 + sep] = True
    return out


def nearest_team(players, team, pts) -> np.ndarray:
    d = np.linalg.norm(players - pts[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    k = np.argmin(d, axis=1)
    out = team[k].astype(np.int64)
    out[~np.isfinite(d.min(axis=1)) | np.isnan(pts[:, 0])] = -1
    return out


def masked_players(g: data.Game, rng) -> np.ndarray:
    """The 'on' condition at native fps: dropout bursts as in training (data._player_mask)."""
    return data._player_mask(g.players.astype(np.float64), rng, g.fps)


def gap_metrics(filled, true, players, team, frames) -> dict:
    sel = frames & ~np.isnan(filled[:, 0])
    if not sel.any():
        return {"n": 0, "err_mean": None, "err_median": None, "err_p90": None, "team_agree": None}
    e = np.linalg.norm(filled[sel] - true[sel], axis=1)
    tf, tt = nearest_team(players[sel], team, filled[sel]), nearest_team(players[sel], team, true[sel])
    ok = (tf >= 0) & (tt >= 0)
    return {"n": int(sel.sum()), "err_mean": float(e.mean()), "err_median": float(np.median(e)),
            "err_p90": float(np.percentile(e, 90)), "team_agree": float(np.mean(tf[ok] == tt[ok])) if ok.any() else None,
            "coverage": float(sel.sum() / frames.sum())}


def downstream(g: data.Game, players, ball) -> dict:
    trk = Tracking(players=players.astype(np.float32), player_speed=speed_from_positions(players, g.fps),
                   ball=ball.astype(np.float32), team_of_slot=g.team)
    feat = features.build(trk, g.pitch, g.fps)
    pred = decode.decode(feat, g.halves, len(g.labels), decode.Params(), tags=None)
    r = score(g.labels, pred, g.fps)
    return {"spellF1": r["spellF1"], "macroF1": r["macroF1"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", type=Path, help="default: every models/ball_*.pt")
    ap.add_argument("--taus", nargs="*", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--players", nargs="*", default=["clean", "on"], choices=["clean", "on"])
    ap.add_argument("--per-bucket", type=int, default=PER_BUCKET)
    ap.add_argument("--no-downstream", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args(argv)
    paths = args.models if args.models is not None else sorted(MODELS_DIR.glob("ball_*.pt"))
    device = "cpu"
    if paths:
        import torch
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    loaded = {p.stem: impute.load_model(p, device) for p in paths}
    games = data.load_split("val")
    rows = []
    for g in games:
        seen_true = ~np.isnan(g.ball[:, 0])
        in_half = np.zeros(len(g.ball), dtype=bool)
        for _, a, b in g.halves:
            in_half[a:b + 1] = True
        ok = g.alive & seen_true & in_half
        for cond in args.players:
            for seed in args.seeds:
                rng = np.random.default_rng([seed, 7])
                P = g.players.astype(np.float64) if cond == "clean" else masked_players(g, np.random.default_rng([seed, 11]))
                # ---------------- buckets
                occ = np.zeros(len(ok), dtype=bool)
                gaps = {}
                for bkt in sorted(BUCKETS_S, reverse=True):
                    n = max(1, int(round(bkt * g.fps)))
                    gaps[bkt] = place_gaps(ok, [n] * args.per_bucket, rng, int(SEP_S * g.fps), occ)
                hidden = np.zeros(len(ok), dtype=bool)
                for v in gaps.values():
                    for a, b in v:
                        hidden[a:b + 1] = True
                ball_m = g.ball.astype(np.float64).copy()
                ball_m[hidden] = np.nan
                fills = {m: impute.fill(P, g.team, ball_m, g.fps, g.pitch, g.halves, m, frames=g.alive).ball
                         for m in ("linear", "hermite")}
                mballs = {}
                for name, (model, cfg) in loaded.items():
                    mballs[name] = impute.predict_ball(model, cfg, P, g.team, ball_m, g.fps, g.pitch, g.halves, device=device)
                    fills[name] = impute.fill(P, g.team, ball_m, g.fps, g.pitch, g.halves, "model", frames=g.alive,
                                              model_ball=mballs[name]).ball
                for bkt, v in gaps.items():
                    fr = np.zeros(len(ok), dtype=bool)
                    for a, b in v:
                        fr[a:b + 1] = True
                    for m, f in fills.items():
                        rows.append({"game": g.name, "players": cond, "seed": seed, "set": f"{bkt:g}s", "method": m,
                                     **gap_metrics(f, g.ball, P, g.team, fr)})
                # ---------------- mixed + downstream
                n_alive = int(ok.sum())
                lengths, tot = [], 0
                while tot < MIXED_SHARE * n_alive:
                    n = max(1, int(round(np.exp(rng.uniform(np.log(0.2), np.log(30.0))) * g.fps)))
                    lengths.append(n)
                    tot += n
                mg = place_gaps(ok, lengths, rng, int(1.0 * g.fps))
                fr = np.zeros(len(ok), dtype=bool)
                for a, b in mg:
                    fr[a:b + 1] = True
                ball_x = g.ball.astype(np.float64).copy()
                ball_x[fr] = np.nan
                cands = {"none": ball_x, "true": g.ball.astype(np.float64)}
                for m in ("linear", "hermite"):
                    cands[m] = impute.fill(P, g.team, ball_x, g.fps, g.pitch, g.halves, m, frames=g.alive).ball
                    for tau in args.taus:                    # interpolate short gaps only, leave the rest empty
                        cands[f"{m}_le{tau:g}s"] = interp.fill(ball_x, g.halves, g.fps, m, max_gap_s=tau,
                                                               pitch=g.pitch, frames=g.alive)[0]
                for name, (model, cfg) in loaded.items():
                    mb = impute.predict_ball(model, cfg, P, g.team, ball_x, g.fps, g.pitch, g.halves, device=device)
                    cands[name] = impute.fill(P, g.team, ball_x, g.fps, g.pitch, g.halves, "model", frames=g.alive,
                                              model_ball=mb).ball
                    for tau in args.taus:
                        cands[f"hybrid_{name}_tau{tau:g}"] = impute.fill(
                            P, g.team, ball_x, g.fps, g.pitch, g.halves, "hybrid", tau_s=tau, frames=g.alive,
                            model_ball=mb).ball
                for m, f in cands.items():
                    row = {"game": g.name, "players": cond, "seed": seed, "set": "mixed", "method": m,
                           **gap_metrics(f, g.ball, P, g.team, fr)}
                    if not args.no_downstream:
                        row.update(downstream(g, P, f))
                    rows.append(row)
                print(f"{g.name:<14} players {cond:<5} seed {seed}: {len(mg)} mixed gaps, "
                      f"{sum(len(v) for v in gaps.values())} bucket gaps", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n=== error on hidden alive frames: mean m (team-nearest agreement) ===")
    for cond, d in df[df["set"] != "mixed"].groupby("players"):
        t = d.groupby(["method", "set"]).apply(lambda x: np.average(x["err_mean"], weights=x["n"]) if x["n"].sum() else np.nan,
                                               include_groups=False).unstack()
        a = d.groupby(["method", "set"]).apply(lambda x: np.average(x["team_agree"].fillna(0), weights=x["n"]) if x["n"].sum() else np.nan,
                                               include_groups=False).unstack()
        cols = [f"{b:g}s" for b in BUCKETS_S if f"{b:g}s" in t.columns]
        print(f"\nplayers: {cond}")
        print(pd.DataFrame({c: [f"{t.loc[m, c]:.1f} ({a.loc[m, c]:.2f})" for m in t.index] for c in cols}, index=t.index).to_string())
    mixed = df[df["set"] == "mixed"]
    if not mixed.empty:
        print("\n=== mixed gaps (25 % of alive hidden), oracle alive, v0 pipeline ===")
        agg = {"coverage": "mean", "err_mean": "mean", "team_agree": "mean"}   # coverage: share of hidden frames filled
        if "spellF1" in mixed:
            agg.update({"spellF1": ["mean", "std"], "macroF1": "mean"})
        for cond, d in mixed.groupby("players"):
            s = d.groupby("method").agg(agg)
            s.columns = ["_".join(c) if isinstance(c, tuple) else c for c in s.columns]
            sort = "spellF1_mean" if "spellF1_mean" in s else "team_agree_mean"
            print(f"\nplayers: {cond}")
            print(s.sort_values(sort, ascending=False).round(3).to_string())
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    out = HARNESS_DIR / f"ball_{args.tag}.json"
    out.write_text(json.dumps({"models": [str(p.name) for p in paths], "taus": args.taus, "seeds": args.seeds,
                               "rows": rows}, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
