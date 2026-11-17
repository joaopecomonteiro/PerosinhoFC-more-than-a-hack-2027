"""What did the dead-ball model see around one moment? FOR YOUR OWN TERMINAL ONLY.

    python -m mqf.deadball.inspect --half 1 --from 1:06 --to 1:13       # exact range, a row every 0.2 s
    python -m mqf.deadball.inspect --half 1 --at 5:17                   # +/- 15 s around a moment
    python -m mqf.deadball.inspect --csv --tag v4                       # stoppages of the public window (true and called)
                                                                        #   + 5 s around each, every frame -> reports/deadball/
    python -m mqf.deadball.inspect --csv --scope match --tag v4         # every stoppage the model calls in the match
    python -m mqf.deadball.inspect --csv --context 10 --all-inputs --tag v4   # more context, all model inputs
    python -m mqf.deadball.inspect --csv --all-frames --tag v4          # the old way: every frame of the public window
CSV extra columns: seg (stoppage number), seg_kind (truth+model = found, truth only = missed, model only =
called where the truth says play -- or outside the public window, no truth there), seg_start, t_rel_s
(seconds since that stoppage started; negative = the context before it).

The clock is the viewer's ("1st half 05:17" = 5 min 17 s after the half's first graded frame).
This prints real values of the competition match (positions, speeds, probabilities): never paste
its output into an AI chat (T&Cs; CLAUDE.md rule 1). Read it, compare a moment the model gets right
with one it gets wrong, and describe the DIFFERENCE in words ("the ball stays tracked and keeps
rolling for 6 s", "players keep running at 3 m/s") -- that is what we can turn into a general rule
and test on other games.

Columns
  clock      half and mm:ss.s            truth    public ground truth: D = dead, . = in play (first 5 min only)
  p_dead     the model's P(dead)          p_pl     João's players-only P(dead) (models/dead_v1)
  dead       the decoder's call (1 = dead)
  ball       tracked or '-'
  ball_xy    position (m, 0-105 x 0-68)  b_speed  ball speed (m/s)
  b_acc      ball speed now minus 0.4 s ago (m/s; a kick or throw jumps)
  still_s    how long the ball has been tracked and under 1 m/s (s)
  to_line    ball's distance to the nearest line (m; < 0 = outside)
  gap_s      length of the ball gap this frame sits in (0 = ball tracked)
  pl_speed   mean player speed (m/s)     slow     share of players under 1 m/s
  near_m     nearest player to the ball (m)  in9m   players within 9.15 m of the ball
  dspeed1s   mean player speed next 1 s minus last 1 s (< 0 = everyone slowing down)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .. import atd
from ..config import MODELS_DIR, competition_game
from . import decoder, features, model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--half", type=int, default=1, choices=[1, 2])
    ap.add_argument("--at", help="mm:ss on the viewer's half clock (with --window)")
    ap.add_argument("--from", dest="t_from", help="mm:ss: start of an exact range")
    ap.add_argument("--to", dest="t_to", help="mm:ss: end of an exact range")
    ap.add_argument("--window", type=float, default=15.0, help="seconds before and after")
    ap.add_argument("--step", type=float, default=0.2, help="seconds between rows")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--w", type=float, default=0.2)
    ap.add_argument("--prior", type=float, default=None, help="decoder prior (default: the model's; frozen v8 = 0.5)")
    ap.add_argument("--csv", action="store_true",
                    help="write the table to reports/deadball/ instead of printing it (default range: the public "
                         "window, 1st half 0:00-5:00; every frame unless --step is given)")
    ap.add_argument("--all-inputs", action="store_true", help="with --csv: add every model input as a column")
    ap.add_argument("--context", type=float, default=5.0,
                    help="with --csv: seconds kept before and after each stoppage")
    ap.add_argument("--scope", choices=["public", "match"], default="public",
                    help="with --csv: stoppages in the public window (truth + model) or in the whole match "
                         "(model only outside the public window)")
    ap.add_argument("--all-frames", action="store_true",
                    help="with --csv: every frame of the range instead of only the stoppages")
    a = ap.parse_args(argv)
    if a.csv and not (a.t_from or a.at):
        a.half, a.t_from, a.t_to = 1, "0:00", "5:00"
    if a.csv and "--step" not in (argv if argv is not None else sys.argv[1:]):
        a.step = 1.0 / 25
    if sys.stdout.isatty() is False:
        print("note: output is not a terminal -- do not paste it into an AI chat", file=sys.stderr)

    spec = competition_game()
    fps = spec.fps
    half = dict((h, (s, e)) for h, s, e in spec.halves)[a.half]
    def frame(clock: str) -> int:
        mm, ss = clock.split(":")
        return half[0] + int(round((int(mm) * 60 + float(ss)) * fps))
    if a.t_from and a.t_to:
        lo, hi = frame(a.t_from), frame(a.t_to)
        centre = None
    elif a.at:
        centre = frame(a.at)
        lo, hi = centre - int(a.window * fps), centre + int(a.window * fps)
    else:
        raise SystemExit("give --from and --to, or --at")
    lo, hi = max(half[0], lo), min(half[1], hi)
    truth = np.full(spec.n_frames, " ", dtype="<U1")
    if spec.gt_public and spec.gt_public.is_file():
        import pandas as pd
        g = pd.read_csv(spec.gt_public, dtype={"label": str})["label"].to_numpy()
        truth = np.where(g == "D", "D", np.where(g == "X", " ", "."))
    m = model.load(MODELS_DIR / f"deadball_{a.tag}.joblib")
    trk = atd.load(spec)
    p = model.predict(m, trk, spec)
    from .. import dead as players_model
    pp = players_model.predict(players_model.load_model(MODELS_DIR / "dead_v1.joblib"), trk.players, trk.team_of_slot,
                               fps, spec.pitch, spec.halves)
    tables = decoder.length_tables(m["lengths"]["alive"]), decoder.length_tables(m["lengths"]["dead"])
    dead = decoder.decode(p, spec.halves, fps, tables, w=a.w,
                          prior=a.prior if a.prior is not None else m["meta"]["prior"])
    X = features.build(trk.players, trk.team_of_slot, trk.ball, fps, spec.pitch, spec.halves)
    model.add_players_columns(X, m, spec)                 # v6+: the players-only inputs too

    still_run = features._run_s(X["b_still"].to_numpy() > 0, fps)
    bsp = X["b_speed"].to_numpy()
    k4 = max(1, int(round(0.4 * fps)))
    step = max(1, int(round(a.step * fps)))

    def clock(f: int) -> str:
        for h, s0, e0 in spec.halves:
            if s0 <= f <= e0:
                t = (f - s0) / fps
                return f"{h}H {int(t // 60):02d}:{t % 60:05.2f}"
        return "-"

    seg_of = {}                                   # frame -> (seg, kind, start_frame)
    if a.csv and not a.all_frames:                # only stoppages (true and called) + context around each
        if a.scope == "match":
            ranges = [(s0, e0) for _, s0, e0 in spec.halves]
        else:
            ranges = [(lo, hi)]
        runs = lambda mask: [(int(s), int(e)) for s, e in zip(np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]) == 1),
                                                              np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]) == -1))]
        frames, segs = [], []
        ctx = int(round(a.context * fps))
        for r0, r1 in ranges:
            inside = np.zeros(spec.n_frames, dtype=bool)
            inside[r0:r1 + 1] = True
            t_mask, d_mask = inside & (truth == "D"), inside & dead
            for s, e in runs(t_mask | d_mask):    # merged stoppages: true, called, or both
                has_t, has_d = t_mask[s:e].any(), d_mask[s:e].any()
                kind = "truth+model" if has_t and has_d else ("truth only" if has_t else "model only")
                segs.append((s, e, kind))
        for k, (s, e, kind) in enumerate(segs, start=1):
            for f in range(max(0, s - ctx), min(spec.n_frames, e + ctx), step):
                if f not in seg_of:
                    seg_of[f] = (k, kind, s)
        frames = sorted(seg_of)
    else:
        frames = list(range(lo, hi + 1, step))
    rows = []
    for f in frames:
        bx, by = trk.ball[f]
        seen = not np.isnan(bx)
        d = np.linalg.norm(trk.players[f] - trk.ball[f], axis=-1) if seen else np.array([np.nan])
        near = float(np.nanmin(d)) if seen and np.isfinite(d).any() else np.nan
        row = {"frame": f, "clock": clock(f), "truth": truth[f].strip(),
               "p_dead": p[f], "p_pl": pp[f], "dead": int(dead[f]), "ball": int(seen),
               "ball_x": bx if seen else np.nan, "ball_y": by if seen else np.nan, "b_speed": bsp[f],
               "b_acc": bsp[f] - bsp[f - k4] if f >= k4 else np.nan, "still_s": still_run[f],
               "to_line": X["b_dline"][f], "gap_s": X["b_gap_s"][f], "pl_speed": X["speed_mean"][f],
               "slow": X["share_slow"][f], "near_m": near,
               "in9m": int(np.nansum(d < 9.15)) if seen else np.nan,      # no ball: unknown, not "nobody near"
               "dspeed1s": X["e_dspeed_1s"][f]}
        if f in seg_of:
            k, kind, s0 = seg_of[f]
            row = {"seg": k, "seg_kind": kind, "seg_start": clock(s0), "t_rel_s": round((f - s0) / fps, 2), **row}
        if a.csv and a.all_inputs:
            row.update({f"x_{c}": X[c][f] for c in m["columns"]})
        rows.append(row)
    if a.csv:
        import pandas as pd
        from ..config import REPORTS_DIR
        if seg_of:
            pr = f"_prior{a.prior:g}" if a.prior is not None else ""
            name = f"inspect_{a.tag}{pr}_stoppages_{a.scope}_ctx{a.context:g}s.csv"
        else:
            name = (f"inspect_{a.tag}_{a.half}H_{a.t_from or a.at}_{a.t_to or ''}".replace(":", "m").rstrip("_")
                    + ".csv")
        out = REPORTS_DIR / "deadball" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
        print(f"wrote {out}  ({len(rows)} rows) -- LOCAL ONLY: open it yourself, never paste it into an AI chat")
        return 0
    print("LOCAL ONLY -- real match values, never paste into an AI chat\n")
    print(f"{'clock':>12} {'truth':>5} {'p_dead':>6} {'p_pl':>5} {'dead':>4} {'ball':>4} {'ball_xy':>11} {'b_speed':>7} {'b_acc':>6} "
          f"{'still_s':>7} {'to_line':>7} {'gap_s':>6} {'pl_speed':>8} {'slow':>5} {'near_m':>6} {'in9m':>4} {'dspeed1s':>8}")
    for r in rows:
        mark = " <" if centre is not None and abs(r["frame"] - centre) < step / 2 else ""
        seen = r["ball"] == 1
        xy = f"{r['ball_x']:5.1f},{r['ball_y']:4.1f}" if seen else "-"
        in9 = str(int(r["in9m"])) if seen else "-"
        print(f"{r['clock'][:11]:>12} {r['truth'] or ' ':>5} {r['p_dead']:6.2f} {r['p_pl']:5.2f} {r['dead']:4d} "
              f"{'yes' if seen else '-':>4} {xy:>11} {r['b_speed']:7.1f} {r['b_acc']:6.1f} {r['still_s']:7.1f} "
              f"{r['to_line']:7.1f} {r['gap_s']:6.1f} {r['pl_speed']:8.2f} {r['slow']:5.2f} {r['near_m']:6.1f} "
              f"{in9:>4} {r['dspeed1s']:8.2f}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
