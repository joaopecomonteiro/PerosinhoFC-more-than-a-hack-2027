"""The ball-imputation protocol on the VALIDATION games (other games only) -- decides method and tau.

    python -m mqf.ball.evaluate --tag v1                                   # every models/ball_*.pt found
    python -m mqf.ball.evaluate --models models/ball_B_off.pt models/ball_B_on.pt --tag v2

For each validation game (IDSSE J03WMX, J03WQQ, METRICA-3; --sources ... pff adds 8 PFF games), player
condition (clean as recorded /
'on': dropouts and broken tracks like our ATD) and mask seed:
  buckets   gaps of exactly 0.5, 1, 2, 4, 8, 16, 32 s cut into the ball on ALIVE frames (both edges
            seen, >= 10 s apart). Per method: error on the hidden frames (mean / median / p90, m)
            and team-nearest agreement -- the team of the player nearest to the filled ball equals
            the team nearest to the true ball (what possession reads).
            'hold' = the ball and its owner carried from the last seen frame (what v0's HMM does).
            Models raw, '+a' (anchored to the gap edges, impute.anchor) and 'route_' (linear <= 1 s,
            Hermite <= 2 s, anchored model beyond).
  mixed     gaps of log-uniform 0.2-30 s until 25 % of alive frames are hidden. Methods filled on
            the true alive frames only (oracle alive), then the v0 decoder and the OFFICIAL metric.
            Filled frames are flagged (features ball_filled): their "who" evidence gets weight
            w_filled, their alive/dead evidence stays as if missing. Rows: none, true (ceiling),
            oracle_fill (true ball as a fill), Hermite, Hermite capped at tau, models, hybrids
            (Hermite <= tau, anchored model beyond). d_vs_none is paired on the same gaps. Pick by
            downstream spellF1; differences under ~2 d_se are noise.
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
ROUTE_LINEAR_S, ROUTE_HERMITE_S = 1.0, 2.0      # route_<model>: linear <= 1 s, Hermite <= 2 s, model beyond (v3)


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


def gap_metrics(filled, true, players, team, frames, team_fill=None) -> dict:
    """team_fill: a per-frame team to score instead of the team nearest to `filled` (owner hold)."""
    sel = frames & ~np.isnan(filled[:, 0])
    if not sel.any():
        return {"n": 0, "err_mean": None, "err_median": None, "err_p90": None, "team_agree": None, "coverage": 0.0}
    e = np.linalg.norm(filled[sel] - true[sel], axis=1)
    tf = team_fill[sel] if team_fill is not None else nearest_team(players[sel], team, filled[sel])
    tt = nearest_team(players[sel], team, true[sel])
    ok = (tf >= 0) & (tt >= 0)
    return {"n": int(sel.sum()), "err_mean": float(e.mean()), "err_median": float(np.median(e)),
            "err_p90": float(np.percentile(e, 90)), "team_agree": float(np.mean(tf[ok] == tt[ok])) if ok.any() else None,
            "coverage": float(sel.sum() / frames.sum())}


def owner_hold(players, team, ball) -> tuple[np.ndarray, np.ndarray]:
    """(ball held at the last seen position, team nearest to it THEN, held) -- what v0's HMM carries."""
    held = pd.DataFrame(ball).ffill().to_numpy()
    nt = nearest_team(players, team, ball).astype(np.float64)
    nt[nt < 0] = np.nan
    return held, pd.Series(nt).ffill().fillna(-1).to_numpy().astype(np.int64)


def downstream(g: data.Game, players, ball, filled=None, w_filled=None) -> dict:
    trk = Tracking(players=players.astype(np.float32), player_speed=speed_from_positions(players, g.fps),
                   ball=ball.astype(np.float32), team_of_slot=g.team)
    feat = features.build(trk, g.pitch, g.fps, ball_filled=filled)
    p = decode.Params() if w_filled is None else decode.Params(w_filled=w_filled)
    pred = decode.decode(feat, g.halves, len(g.labels), p, tags=None)
    r = score(g.labels, pred, g.fps)
    return {"spellF1": r["spellF1"], "macroF1": r["macroF1"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", type=Path, help="default: every models/ball_*.pt")
    ap.add_argument("--taus", nargs="*", type=float, default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--anchor", type=float, default=2.0, help="fade (s) of the edge anchoring ('+a' rows)")
    ap.add_argument("--w-filled", nargs="*", type=float, default=[0.15, 0.4, 1.0],
                    help="decoder 'who' weight on imputed frames (seen = 1, missing = 0.15)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--players", nargs="*", default=["clean", "on"], choices=["clean", "on"])
    ap.add_argument("--per-bucket", type=int, default=PER_BUCKET)
    ap.add_argument("--no-downstream", action="store_true")
    ap.add_argument("--hide-dead", action="store_true",
                    help="mixed set: also hide the ball on every dead frame (BAR-ATH's pattern) and fill / let the "
                         "model see every frame -- no oracle alive")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sources", nargs="*", default=["idsse", "metrica"],
                    help="validation games of these sources (add pff for the 8 World Cup games)")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args(argv)
    paths = args.models if args.models is not None else sorted(MODELS_DIR.glob("ball_*.pt"))
    device = "cpu"
    if paths:
        import torch
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    loaded = {p.stem: impute.load_model(p, device) for p in paths}
    games = data.load_split("val", args.sources)
    A = args.anchor
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

                def model_fills(ball_x, extra_taus=(), frames=g.alive):
                    """{name: raw, name+a: anchored, hybrid_name+a_tauX: ...} -- one inference per model.
                    The model sees the ball on alive frames only, as in training."""
                    out = {}
                    for name, (model, cfg) in loaded.items():
                        mb = impute.predict_ball(model, cfg, P, g.team, ball_x, g.fps, g.pitch, g.halves,
                                                 visible=frames, device=device)
                        kw = dict(frames=frames, model_ball=mb)
                        out[name] = impute.fill(P, g.team, ball_x, g.fps, g.pitch, g.halves, "model", **kw)
                        out[f"{name}+a"] = impute.fill(P, g.team, ball_x, g.fps, g.pitch, g.halves, "model", anchor_s=A, **kw)
                        out[f"route_{name}"] = impute.fill(              # best per regime in v3
                            P, g.team, ball_x, g.fps, g.pitch, g.halves, "hybrid", tau_s=ROUTE_HERMITE_S,
                            tau_linear_s=ROUTE_LINEAR_S, interp_method="hermite", anchor_s=A, **kw)
                        for tau in extra_taus:
                            out[f"hybrid_{name}+a_tau{tau:g}"] = impute.fill(
                                P, g.team, ball_x, g.fps, g.pitch, g.halves, "hybrid", tau_s=tau, interp_method="hermite",
                                anchor_s=A, **kw)
                    return out

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
                fills.update({k: v.ball for k, v in model_fills(ball_m).items()})
                held, held_team = owner_hold(P, g.team, ball_m)
                for bkt, v in gaps.items():
                    fr = np.zeros(len(ok), dtype=bool)
                    for a, b in v:
                        fr[a:b + 1] = True
                    base = {"game": g.name, "players": cond, "seed": seed, "set": f"{bkt:g}s"}
                    rows.append({**base, "method": "hold", **gap_metrics(held, g.ball, P, g.team, fr, held_team)})
                    for m, f in fills.items():
                        rows.append({**base, "method": m, **gap_metrics(f, g.ball, P, g.team, fr)})
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
                hid = fr.copy()
                if args.hide_dead:                              # stoppages without a ball, as on BAR-ATH
                    dead_seen = ~g.alive & in_half & ~np.isnan(ball_x[:, 0])
                    ball_x[dead_seen] = np.nan
                    hid |= dead_seen
                fill_at = None if args.hide_dead else g.alive
                base = {"game": g.name, "players": cond, "seed": seed, "set": "mixed"}
                # as tracked: no fill, and the true ball on the hidden frames (the ceiling)
                for m, f in (("none", ball_x), ("true", g.ball.astype(np.float64))):
                    row = {**base, "method": m, "w_filled": np.nan, **gap_metrics(f, g.ball, P, g.team, fr)}
                    if not args.no_downstream:
                        row.update(downstream(g, P, f))
                    rows.append(row)
                # imputed: decoded with the filled frames flagged, for every w_filled
                cands = {"oracle_fill": (g.ball.astype(np.float64), hid)}
                bf = impute.fill(P, g.team, ball_x, g.fps, g.pitch, g.halves, "hermite", frames=fill_at)
                cands["hermite"] = (bf.ball, (bf.source == impute.INTERP))
                for tau in args.taus:                         # interpolate short gaps only, leave the rest empty
                    f_, done = interp.fill(ball_x, g.halves, g.fps, "hermite", max_gap_s=tau, pitch=g.pitch, frames=fill_at)
                    cands[f"hermite_le{tau:g}s"] = (f_, done)
                for k, v in model_fills(ball_x, args.taus, frames=fill_at).items():
                    cands[k] = (v.ball, np.isin(v.source, (impute.INTERP, impute.MODEL)))
                for m, (f, filled) in cands.items():
                    metr = gap_metrics(f, g.ball, P, g.team, fr)
                    for w in ([None] if args.no_downstream else args.w_filled):
                        row = {**base, "method": m, "w_filled": np.nan if w is None else w, **metr}
                        if w is not None:
                            row.update(downstream(g, P, f, filled, w))
                        rows.append(row)
                print(f"{g.name:<14} players {cond:<5} seed {seed}: {len(mg)} mixed gaps, "
                      f"{sum(len(v) for v in gaps.values())} bucket gaps", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 400)
    print("\n=== error on hidden alive frames: mean m (team-nearest agreement; hold = last owner carried) ===")
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
        cond_txt = "ball also hidden on every dead frame, filled everywhere" if args.hide_dead else "oracle alive"
        print(f"\n=== mixed gaps (25 % of alive hidden), {cond_txt}, v0 decoder; w = 'who' weight on filled frames ===")
        agg = {"coverage": "mean", "err_mean": "mean", "team_agree": "mean"}   # coverage: share of hidden frames filled
        if "spellF1" in mixed:
            agg.update({"spellF1": ["mean", "std"], "macroF1": "mean"})
        for cond, d in mixed.groupby("players"):
            d = d.assign(w_filled=d["w_filled"].fillna(-1.0))              # -1: not a fill (none / true)
            s = d.groupby(["method", "w_filled"]).agg(agg)
            s.columns = ["_".join(c) if isinstance(c, tuple) else c for c in s.columns]
            sort = "spellF1_mean" if "spellF1_mean" in s else "team_agree"
            s = s.sort_values(sort, ascending=False).round(3)
            if "spellF1_mean" in s:                                           # paired with 'none', same gaps
                key = ["game", "seed"]
                none = d[d["method"] == "none"][key + ["spellF1"]].rename(columns={"spellF1": "sp_none"})
                dd = d.merge(none, on=key)
                g2 = (dd["spellF1"] - dd["sp_none"]).groupby([dd["method"], dd["w_filled"]])
                s["d_vs_none"] = g2.mean().round(3)
                s["d_se"] = (g2.std() / np.sqrt(g2.count())).round(3)
            top = s.head(25)
            ref = s[s.index.get_level_values(0).isin(["none", "true"]) & ~s.index.isin(top.index)]
            print(f"\nplayers: {cond}   (top 25 + references; every row in the JSON)")
            print(pd.concat([top, ref]).to_string())
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    out = HARNESS_DIR / f"ball_{args.tag}.json"
    out.write_text(json.dumps({"models": [str(p.name) for p in paths], "taus": args.taus, "anchor_s": A,
                               "w_filled": args.w_filled, "seeds": args.seeds, "rows": rows}, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
