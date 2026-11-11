"""Regenerate our submission: tracking (+ Smart Tagging) -> features -> HMM decode -> CSV.

    python -m mqf.make_submission --team <team_name>      # real match (data/ or MTAH_DATA_DIR)
    python -m mqf.make_submission --team <team_name> --reproduce   # regenerate + byte-compare (any OS)
    python -m mqf.make_submission --synthetic             # synthetic match, no data needed

Writes results/<team>.csv (+ .json run report) and runs the OFFICIAL validator.
Deterministic: same data + same code -> byte-identical file.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from . import atd, decode, features, smart_tagging, submission
from .config import CACHE_DIR, RESULTS_DIR, SYNTHETIC_DIR, competition_game


def run(spec, params: decode.Params, out: Path, use_tags: bool = True, clean_tracks: bool = False) -> dict:
    t0 = time.time()
    trk = atd.load(spec)
    clean_report = None
    if clean_tracks:
        from . import tracks
        trk, clean_report = tracks.clean(trk, spec.fps, spec.pitch)
    feat = features.build(trk, spec.pitch, spec.fps)
    tags = None
    if use_tags and spec.smart_tagging and Path(spec.smart_tagging).exists():
        tags = smart_tagging.evidence(spec.smart_tagging, spec.atd_meta, spec.n_frames, spec.fps)
    labels = decode.decode(feat, spec.halves, spec.n_frames, params, tags)
    submission.write(out, labels)
    # Aggregates only -- this report is what the team may share (e.g. with an AI assistant).
    report = {"file": Path(out).name, "seconds": round(time.time() - t0, 1),
              "ball_seen_pct": round(100 * float(feat["ball_seen"].mean()), 1),
              "used_tags": tags is not None, "clean_tracks": clean_report, **decode.spell_stats(labels, spec.fps),
              "params": asdict(params)}
    Path(out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="Perosinho_FC", help="file is named after the team")
    ap.add_argument("--synthetic", action="store_true", help="generate and use the synthetic match")
    ap.add_argument("--no-tags", action="store_true", help="ablation: ignore Smart Tagging")
    ap.add_argument("--clean-tracks", action="store_true", help="team-level tracklet cleaning (mqf.tracks) before features")
    ap.add_argument("--out", type=Path, help="override output path")
    ap.add_argument("--reproduce", action="store_true",
                    help="regenerate into cache/reproduce/ and check it is byte-identical to results/<team>.csv")
    args = ap.parse_args()

    if args.synthetic:
        from . import synthetic
        spec = synthetic.generate(SYNTHETIC_DIR)
        out = args.out or SYNTHETIC_DIR / "results" / f"{args.team}.csv"
    else:
        spec = competition_game()
        missing = [p for p in (spec.atd_meta, spec.atd_raw) if not p.exists()]
        if missing:
            raise SystemExit(f"missing {missing[0]}\nUnpack the match bundle into data/ or set MTAH_DATA_DIR.")
        out = args.out or RESULTS_DIR / f"{args.team}.csv"
        if args.reproduce:
            reference, out = out, CACHE_DIR / "reproduce" / f"{args.team}.csv"
            if not reference.exists():
                raise SystemExit(f"nothing to compare against: {reference} does not exist")

    report = run(spec, decode.Params(), out, use_tags=not args.no_tags, clean_tracks=args.clean_tracks)
    print(json.dumps({k: v for k, v in report.items() if k != "params"}, indent=2))
    if args.synthetic:
        return
    if args.reproduce:
        same = out.read_bytes() == reference.read_bytes()
        print(f"byte-identical to {reference.name}: {'OK' if same else 'FAIL'}")
        raise SystemExit(0 if same else 1)
    raise SystemExit(submission.validate(out))


if __name__ == "__main__":
    main()
