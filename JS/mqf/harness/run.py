"""Validation harness: other games + simulated ATD noise -> our pipeline -> official metric.

    python -m mqf.harness.run --source synthetic --seeds 0 1 2 --tag v0
    python -m mqf.harness.run --source idsse --matches all --seeds 0 1 --tag v0
    python -m mqf.harness.run --source idsse --noise jitter_corr_s=1.0 ball_burst_mean_s=3 --tag v0_robust
    python -m mqf.harness.run --seeds 0 1 2 --clean-tracks --clean-cfg keeper_takeover=0 \
        --noise team_flip_p=0.05 gk_team_flip_p=0.5 hold_lost_s=3 --tag takeover_off

This is where model choices are made (never on the public 5 minutes). Results are aggregates on
games that are not the competition match, written to harness_results/<tag>.json -- safe to share.
Differences below ~0.04 spellF1 are noise on one match; average over matches and noise seeds.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

import numpy as np

from .. import decode, synthetic
from ..config import HARNESS_DIR
from .evaluate import predict, score
from .noise import NoiseConfig


def matches(source: str, ids):
    if source == "synthetic":
        for seed in (ids or ["101", "102", "103"]):
            m = synthetic.clean_match(seed=int(seed))
            m["name"] = f"SYN-{seed}"
            yield m
    elif source == "idsse":
        from . import idsse
        for mid in (idsse.MATCHES if not ids or ids == ["all"] else ids):
            yield idsse.load_match(mid)
    else:
        raise SystemExit(f"unknown source {source}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "idsse"])
    ap.add_argument("--matches", nargs="*", help="match ids (idsse) or seeds (synthetic); default all")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0], help="noise seeds per match")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--clean-tracks", action="store_true", help="team-level tracklet cleaning (mqf.tracks) before features")
    ap.add_argument("--noise", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="override NoiseConfig assumptions, e.g. jitter_corr_s=1.0 (robustness checks)")
    ap.add_argument("--clean-cfg", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="CleanConfig overrides for --clean-tracks, e.g. keeper_takeover=0")
    args = ap.parse_args()
    from ..external.evaluate import parse_clean_cfg
    clean_cfg = parse_clean_cfg(args.clean_cfg)

    default = NoiseConfig()
    over = dict(kv.split("=", 1) for kv in args.noise)
    base = replace(default, **{k: type(getattr(default, k))(float(v)) for k, v in over.items()})
    params, rows = decode.Params(), []
    for m in matches(args.source, args.matches):
        for s in args.seeds:
            cfg = replace(base, seed=s)
            r = score(m["labels"], predict(m, params, cfg, clean_tracks=args.clean_tracks, clean_cfg=clean_cfg),
                      m["fps"])
            rows.append({"match": m["name"], "noise_seed": s, **r})
            buckets = "  ".join(f"{k}:{v['match_rate']:.2f}(n={v['n']})" for k, v in r["by_length"].items()
                                if v["match_rate"] is not None)
            print(f"{m['name']:<14} seed {s}  spellF1 {r['spellF1']:.3f}  macroF1 {r['macroF1']:.3f}  "
                  f"spells {r['n_pred_spells']}/{r['n_gt_spells']}  {buckets}")

    sf = np.array([r["spellF1"] for r in rows]); mf = np.array([r["macroF1"] for r in rows])
    print(f"\nMEAN  spellF1 {sf.mean():.3f} ± {sf.std():.3f}   macroF1 {mf.mean():.3f} ± {mf.std():.3f}   "
          f"({len(rows)} runs)")
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    out = HARNESS_DIR / f"{args.source}_{args.tag}.json"
    out.write_text(json.dumps({"params": asdict(params), "noise": asdict(base), "clean_tracks": args.clean_tracks,
                               "clean_cfg": args.clean_cfg,
                               "mean_spellF1": float(sf.mean()), "mean_macroF1": float(mf.mean()),
                               "runs": rows}, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
