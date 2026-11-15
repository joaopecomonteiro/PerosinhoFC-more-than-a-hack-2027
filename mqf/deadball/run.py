"""Dead ball on the competition match -> results/deadball/<tag>.csv  (D where dead, X elsewhere).

    python -m mqf.deadball.run --tag v1                 # models/deadball_v1.joblib, decoder weight 0.2
    python -m mqf.viewer --deadball                     # gray where the ball is dead

Not a submission (no H / A): a dead-ball track for the viewer and for checks. Prints aggregates
only -- number of stoppages, their lengths, dead share -- which are safe to share.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .. import atd
from ..config import MODELS_DIR, RESULTS_DIR, competition_game
from . import decoder, model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--model", default=None, help="default: models/deadball_<tag>.joblib")
    ap.add_argument("--w", type=float, default=0.2, help="decoder emission weight")
    ap.add_argument("--pl", type=float, default=0.0,
                    help="weight of João's players-only P(dead) (models/dead_v1) combined in, 0 = off (evaluate.combine)")
    ap.add_argument("--prior", type=float, default=None, help="decoder prior (default: the model's training prior)")
    ap.add_argument("--out-tag", default=None, help="file name tag (default: --tag, + _pl<b> / _prior<x> when set)")
    args = ap.parse_args(argv)
    t0 = time.time()
    spec = competition_game()
    if not spec.atd_raw.exists():
        raise SystemExit(f"missing {spec.atd_raw}\nUnpack the match bundle into data/ or set MTAH_DATA_DIR.")
    m = model.load(args.model or MODELS_DIR / f"deadball_{args.tag}.joblib")
    trk = atd.load(spec)
    p = model.predict(m, trk, spec)
    if args.pl:
        from .. import dead as players_model
        from .evaluate import combine
        pp = players_model.predict(players_model.load_model(MODELS_DIR / "dead_v1.joblib"), trk.players, trk.team_of_slot, spec.fps,
                          spec.pitch, spec.halves)
        p = combine(p, pp, args.pl, spec.fps)
    prior = args.prior if args.prior is not None else m["meta"]["prior"]
    tables = decoder.length_tables(m["lengths"]["alive"]), decoder.length_tables(m["lengths"]["dead"])
    dead = decoder.decode(p, spec.halves, spec.fps, tables, w=args.w, prior=prior)

    name = args.out_tag or (args.tag + (f"_pl{args.pl:g}" if args.pl else "") +
                            (f"_prior{args.prior:g}" if args.prior is not None else ""))
    out = RESULTS_DIR / "deadball" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{'D' if d else 'X'}\n" for i, d in enumerate(dead))
    graded = np.zeros(spec.n_frames, dtype=bool)
    for _, a, b in spec.halves:
        graded[a:b + 1] = True
    d = np.diff(np.r_[0, (dead & graded).astype(np.int8), 0])
    lens = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) / spec.fps
    report = {"file": f"deadball/{out.name}", "model": m["meta"]["tag"], "w": args.w, "pl": args.pl,
              "prior": round(float(prior), 4),
              "seconds": round(time.time() - t0, 1),
              "dead_share_graded": round(float(dead[graded].mean()), 3), "stoppages": int(len(lens)),
              "stoppage_s": {q: round(float(np.percentile(lens, q)), 1) for q in (10, 50, 90)} if len(lens) else None,
              "stoppages_by_length": {"0-5s": int((lens < 5).sum()), "5-15s": int(((lens >= 5) & (lens < 15)).sum()),
                                      "15-40s": int(((lens >= 15) & (lens < 40)).sum()), "40s+": int((lens >= 40).sum())}}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
