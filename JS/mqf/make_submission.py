"""Regenerate our submission: tracking (+ Smart Tagging) -> features -> HMM decode -> CSV.

    python -m mqf.make_submission --team <team_name>      # real match (data/ or MTAH_DATA_DIR)
    python -m mqf.make_submission --team <team_name> --reproduce   # regenerate + byte-compare (any OS)
    python -m mqf.make_submission --synthetic             # synthetic match, no data needed
    python -m mqf.make_submission --team <team_name> --ball-fill   # + missing ball filled (needs torch, CPU ok)
    python -m mqf.make_submission --team <team_name> --dead-model models/dead_v1.joblib   # + learned P(dead)

Writes results/<team>.csv (+ .json run report) and runs the OFFICIAL validator.
Deterministic: same data + same code -> byte-identical file. The run options are saved in the .json
report and --reproduce reuses them. --ball-fill runs the frozen ball model (mqf.ball.impute.FROZEN)
on the CPU in float64 and rounds filled positions to 1 cm, so last-bit differences between machines
cannot change a label.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import atd, decode, features, smart_tagging, submission
from .config import CACHE_DIR, MODELS_DIR, RESULTS_DIR, SYNTHETIC_DIR, competition_game


def fill_ball(trk: atd.Tracking, spec, model_path: Path | None = None):
    """The frozen ball fill, reproducible: (tracking with the filled ball, filled-frame mask, aggregates)."""
    import numpy as np
    from .ball import impute
    F = impute.FROZEN
    path = Path(model_path) if model_path else MODELS_DIR / F["model"]
    if not path.is_file():
        raise SystemExit(f"--ball-fill: no ball model at {path}")
    model, cfg = impute.load_model(path, "cpu", dtype="float64")
    bf = impute.fill(trk.players, trk.team_of_slot, trk.ball, spec.fps, spec.pitch, spec.halves, F["method"],
                     F["tau_s"], model, cfg, interp_method=F["interp_method"], device="cpu",
                     anchor_s=F["anchor_s"], tau_linear_s=F["tau_linear_s"], round_m=0.01)
    filled = (bf.source == impute.INTERP) | (bf.source == impute.MODEL)
    graded = np.zeros(len(bf.ball), dtype=bool)
    for _, a, b in spec.halves:
        graded[a:b + 1] = True
    miss = graded & np.isnan(trk.ball[:, 0])
    rep = {"model": path.name, "missing_graded_pct": round(100 * float(miss.mean() / max(graded.mean(), 1e-9)), 1),
           "interpolated_pct_of_missing": round(100 * float(np.mean(bf.source[miss] == impute.INTERP)), 1) if miss.any() else 0.0,
           "model_pct_of_missing": round(100 * float(np.mean(bf.source[miss] == impute.MODEL)), 1) if miss.any() else 0.0}
    out = atd.Tracking(trk.players, trk.player_speed, bf.ball.astype(np.float32), trk.team_of_slot)
    return out, filled, rep


def run(spec, params: decode.Params, out: Path, use_tags: bool = True, clean_tracks: bool = False,
        ball_fill: bool = False, ball_model: Path | None = None, dead_model: Path | None = None) -> dict:
    t0 = time.time()
    trk = atd.load(spec)
    clean_report = None
    if clean_tracks:
        from . import tracks
        windows = None
        if use_tags and spec.smart_tagging and Path(spec.smart_tagging).exists():
            windows = smart_tagging.kickoff_windows(spec.smart_tagging, spec.fps, spec.n_frames)
        trk, clean_report = tracks.clean(trk, spec.fps, spec.pitch, kickoff_windows=windows, halves=spec.halves)
    filled, fill_report = None, None
    if ball_fill:
        trk, filled, fill_report = fill_ball(trk, spec, ball_model)
    p_dead = None
    if dead_model:
        from . import dead
        p_dead = dead.predict(dead.load_model(dead_model), trk.players, trk.team_of_slot, spec.fps, spec.pitch,
                              spec.halves)
    feat = features.build(trk, spec.pitch, spec.fps, ball_filled=filled, p_dead=p_dead)
    tags = None
    if use_tags and spec.smart_tagging and Path(spec.smart_tagging).exists():
        tags = smart_tagging.evidence(spec.smart_tagging, spec.atd_meta, spec.n_frames, spec.fps)
    labels = decode.decode(feat, spec.halves, spec.n_frames, params, tags)
    submission.write(out, labels)
    # Aggregates only -- this report is what the team may share (e.g. with an AI assistant).
    report = {"file": Path(out).name, "seconds": round(time.time() - t0, 1),
              "ball_seen_pct": round(100 * float((feat["ball_seen"] & ~feat["ball_filled"]).mean()), 1),
              "used_tags": tags is not None, "clean_tracks": clean_report, "ball_fill": fill_report,
              **decode.spell_stats(labels, spec.fps),
              "dead_share_model": None if p_dead is None else round(float(np.mean(p_dead[~np.isnan(p_dead)] >= 0.5)), 3),
              "options": {"use_tags": use_tags, "clean_tracks": clean_tracks, "ball_fill": ball_fill,
                          "ball_model": str(ball_model) if ball_model else None,
                          "dead_model": str(dead_model) if dead_model else None},
              "params": asdict(params)}
    Path(out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="Perosinho_FC", help="file is named after the team")
    ap.add_argument("--synthetic", action="store_true", help="generate and use the synthetic match")
    ap.add_argument("--no-tags", action="store_true", help="ablation: ignore Smart Tagging")
    ap.add_argument("--clean-tracks", action="store_true", help="team-level tracklet cleaning (mqf.tracks) before features")
    ap.add_argument("--ball-fill", action="store_true", help="fill the missing ball (mqf.ball.impute.FROZEN; torch, CPU)")
    ap.add_argument("--ball-model", type=Path, default=None, help="override the frozen ball model file")
    ap.add_argument("--dead-model", type=Path, default=None, help="learned P(dead) (mqf.dead), e.g. models/dead_v1.joblib")
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

    opts = {"use_tags": not args.no_tags, "clean_tracks": args.clean_tracks, "ball_fill": args.ball_fill,
            "ball_model": args.ball_model, "dead_model": args.dead_model}
    if args.reproduce and not args.synthetic and reference.with_suffix(".json").exists():   # options of the reference
        saved = json.loads(reference.with_suffix(".json").read_text()).get("options")
        if saved:
            opts = {**saved, "ball_model": Path(saved["ball_model"]) if saved.get("ball_model") else None,
                    "dead_model": Path(saved["dead_model"]) if saved.get("dead_model") else None}
            print(f"reproducing with the reference's options: {saved}")
    report = run(spec, decode.Params(), out, **opts)
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
