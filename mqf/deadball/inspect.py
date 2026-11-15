"""What did the dead-ball model see around one moment? FOR YOUR OWN TERMINAL ONLY.

    python -m mqf.deadball.inspect --half 1 --from 1:06 --to 1:13       # exact range, a row every 0.2 s
    python -m mqf.deadball.inspect --half 1 --at 5:17                   # +/- 15 s around a moment

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
    ap.add_argument("--half", type=int, required=True, choices=[1, 2])
    ap.add_argument("--at", help="mm:ss on the viewer's half clock (with --window)")
    ap.add_argument("--from", dest="t_from", help="mm:ss: start of an exact range")
    ap.add_argument("--to", dest="t_to", help="mm:ss: end of an exact range")
    ap.add_argument("--window", type=float, default=15.0, help="seconds before and after")
    ap.add_argument("--step", type=float, default=0.2, help="seconds between rows")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--w", type=float, default=0.2)
    a = ap.parse_args(argv)
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
    dead = decoder.decode(p, spec.halves, fps, tables, w=a.w, prior=m["meta"]["prior"])
    X = features.build(trk.players, trk.team_of_slot, trk.ball, fps, spec.pitch, spec.halves)

    still = (X["b_still"].to_numpy() > 0).astype(np.int32)
    still_run = np.zeros(len(still))
    run = 0
    for i in range(lo - int(60 * fps) if lo > int(60 * fps) else 0, hi + 1):
        run = run + 1 if still[i] else 0
        still_run[i] = run / fps
    bsp = X["b_speed"].to_numpy()
    k4 = max(1, int(round(0.4 * fps)))
    step = max(1, int(round(a.step * fps)))
    print("LOCAL ONLY -- real match values, never paste into an AI chat\n")
    print(f"{'clock':>12} {'truth':>5} {'p_dead':>6} {'p_pl':>5} {'dead':>4} {'ball':>4} {'ball_xy':>11} {'b_speed':>7} {'b_acc':>6} "
          f"{'still_s':>7} {'to_line':>7} {'gap_s':>6} {'pl_speed':>8} {'slow':>5} {'near_m':>6} {'in9m':>4} {'dspeed1s':>8}")
    for f in range(lo, hi + 1, step):
        t = (f - half[0]) / fps
        bx, by = trk.ball[f]
        seen = not np.isnan(bx)
        d = np.linalg.norm(trk.players[f] - trk.ball[f], axis=-1) if seen else np.array([np.nan])
        near = np.nanmin(d) if seen and np.isfinite(d).any() else np.nan
        in9 = str(int(np.nansum(d < 9.15))) if seen else "-"     # no ball: unknown, not "nobody near"
        mark = " <" if centre is not None and abs(f - centre) < step / 2 else ""
        acc = bsp[f] - bsp[f - k4] if f >= k4 else np.nan
        print(f"{a.half}H {int(t // 60):02d}:{t % 60:04.1f} {truth[f]:>5} {p[f]:6.2f} {pp[f]:5.2f} {int(dead[f]):4d} {'yes' if seen else '-':>4} "
              f"{(f'{bx:5.1f},{by:4.1f}' if seen else '-'):>11} {bsp[f]:7.1f} {acc:6.1f} {still_run[f]:7.1f} {X['b_dline'][f]:7.1f} "
              f"{X['b_gap_s'][f]:6.1f} {X['speed_mean'][f]:8.2f} {X['share_slow'][f]:5.2f} {near:6.1f} {in9:>4} "
              f"{X['e_dspeed_1s'][f]:8.2f}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
