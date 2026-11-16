"""Regenerate our submission: tracking (+ Smart Tagging) -> features -> HMM decode -> CSV.

    python -m mqf.make_submission --team <team_name> --final  # THE submission: the frozen recipe FINAL below
    python -m mqf.make_submission --team <team_name> --final --reproduce   # regenerate + byte-compare (any OS)
    python -m mqf.make_submission --team <team_name>      # v0 (no models), real match (data/ or MTAH_DATA_DIR)
    python -m mqf.make_submission --synthetic             # synthetic match, no data needed
    python -m mqf.make_submission --team <team_name> --ball-fill   # + missing ball filled (needs torch, CPU ok)
    python -m mqf.make_submission --team <team_name> --dead-model models/dead_v1.joblib   # + learned P(dead)
    python -m mqf.make_submission --team <team_name> --who-model models/who_v1.joblib     # + learned P(home | alive)
    python -m mqf.make_submission --team <team_name> ... --params dead_who=max       # decoder overrides (decode.Params)
    python -m mqf.make_submission --team <team_name> ... --alive-model models/alive_asf_full_impff_s0.pt \
        models/alive_asf_full_impff_s1.pt models/alive_asf_full_impff_s2.pt [--alive-variant start] [--alive-tag-w 1]
                                                     # the mqf.alive layer instead of --dead-model (opt-in; several = mean)

Writes results/<team>.csv (+ .json run report) and runs the OFFICIAL validator.
Deterministic: same data + same code -> byte-identical file. The run options and decoder parameters
are saved in the .json report and --reproduce reuses them; with no report next to the file (a fresh
clone with the submitted CSV copied to results/<team>.csv) it uses the flags given, so
`--final --reproduce` checks the submitted file from the data bundle and the repository alone. --ball-fill runs the frozen ball model (mqf.ball.impute.FROZEN)
on the CPU in float64 and rounds filled positions to 1 cm, so last-bit differences between machines
cannot change a label.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

from . import atd, decode, features, smart_tagging, submission
from .config import CACHE_DIR, MODELS_DIR, RESULTS_DIR, ROOT, SYNTHETIC_DIR, competition_game

# The submitted configuration (docs/EXPERIMENTS.md #35-#39): learned P(home | alive) and P(dead), the frozen
# ball fill, the tracklet cleaner and Smart Tagging; the decoder as validated (dead_who="max", #34/#35).
# Model paths are relative to the package root (JS/). Change it only with a new EXPERIMENTS entry.
FINAL = {"options": {"use_tags": True, "clean_tracks": True, "ball_fill": True, "ball_model": None,
                     "dead_model": "models/dead_v1.joblib", "who_model": "models/who_v1.joblib"},
         "params": {"dead_who": "max"}}
_PATH_OPTS = ("ball_model", "dead_model", "who_model", "alive_model")
_ALIVE_DEFAULTS = {"alive_model": None, "alive_variant": "hmm", "alive_tag_w": 0.0, "alive_goal_w": 0.0}


def parse_params(items) -> dict:
    """["dead_who=max", "w_tags=0.4"] -> decode.Params overrides, typed like the dataclass fields."""
    types = {f.name: type(f.default) for f in fields(decode.Params)}
    out = {}
    for it in items or []:
        k, eq, v = it.partition("=")
        if not eq or k not in types:
            raise SystemExit(f"--params: expected key=value with key in decode.Params ({', '.join(types)}), got {it!r}")
        out[k] = types[k](v)
    if out.get("dead_who", "zero") not in ("zero", "max"):
        raise SystemExit(f"--params: dead_who must be zero or max, got {out['dead_who']!r}")
    return out


def _alive_opt(alive_model):
    """The alive model(s) as saved in the run report: None, one posix path, or a list of them."""
    from .alive.predict import alive_paths
    ps = [p.as_posix() for p in alive_paths(alive_model)]
    return None if not ps else (ps[0] if len(ps) == 1 else ps)


def _model_file(p) -> Path | None:
    """A model path as given, or under the package root when relative and not found from here."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() or p.exists() else ROOT / p


def settings(args, saved: dict | None = None) -> tuple[dict, decode.Params]:
    """(options for run(), decoder Params) from the command line -- or from a saved run report (--reproduce)."""
    if saved:
        opts = {**saved["options"]}
        unknown = sorted(set(saved.get("params", {})) - {f.name for f in fields(decode.Params)})
        if unknown:
            raise SystemExit(f"the reference report has decoder parameters this code does not know: {unknown}")
        params = decode.Params(**saved.get("params", {}))
    elif args.final:
        given = [f for f in ("no_tags", "clean_tracks", "ball_fill", "ball_model", "dead_model", "who_model", "params",
                             "alive_model", "alive_tag_w", "alive_goal_w") if getattr(args, f, None)]
        if getattr(args, "alive_variant", "hmm") != "hmm":
            given.append("alive_variant")
        if given:
            raise SystemExit(f"--final is the frozen recipe; drop {', '.join('--' + g.replace('_', '-') for g in given)}")
        opts, params = {**FINAL["options"]}, replace(decode.Params(), **FINAL["params"])
    else:
        opts = {"use_tags": not args.no_tags, "clean_tracks": args.clean_tracks, "ball_fill": args.ball_fill,
                "ball_model": args.ball_model, "dead_model": args.dead_model, "who_model": args.who_model,
                **{k: getattr(args, k, v) for k, v in _ALIVE_DEFAULTS.items()}}
        if opts["alive_model"] and opts["dead_model"]:
            raise SystemExit("--alive-model replaces --dead-model: give one of them")
        from .alive import combine as acombine
        acombine.parse_variant(opts["alive_variant"])
        params = replace(decode.Params(), **parse_params(args.params))
    return {k: (Path(v) if k in _PATH_OPTS and v and not isinstance(v, (list, tuple)) else v)
            for k, v in opts.items()}, params


def _non_default(params: decode.Params) -> dict:
    base = asdict(decode.Params())
    return {k: v for k, v in asdict(params).items() if base[k] != v}


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
        ball_fill: bool = False, ball_model: Path | None = None, dead_model: Path | None = None,
        who_model: Path | None = None, alive_model=None, alive_variant: str = "hmm",
        alive_tag_w: float = 0.0, alive_goal_w: float = 0.0) -> dict:
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
    tracked_ball = trk.ball.copy()                     # the alive layer reads the TRACKED ball, never a filled one
    if ball_fill:
        trk, filled, fill_report = fill_ball(trk, spec, _model_file(ball_model))
    p_dead = None
    alive_on = bool(alive_model) or alive_variant != "hmm" or alive_tag_w > 0
    p_bnd = rr = None
    if alive_model:                   # one model or several (averaged, as experiment.py's --ensemble)
        from .alive import predict as apredict
        alive_files = [_model_file(p) for p in apredict.alive_paths(alive_model)]
        p_dead, p_bnd, rr = apredict.p_dead(alive_files, trk.players, trk.team_of_slot, tracked_ball,
                                            spec.fps, spec.pitch, spec.halves, device="cpu")
    elif alive_on:
        from .alive import predict as apredict
        rr = apredict.rules_only(trk.players, trk.team_of_slot, tracked_ball, spec.fps, spec.pitch, spec.halves)
    if dead_model:
        from . import dead
        p_dead = dead.predict(dead.load_model(_model_file(dead_model)), trk.players, trk.team_of_slot, spec.fps, spec.pitch,
                              spec.halves)
    p_home = None
    if who_model:                     # reads the ball as the decoder gets it (filled when --ball-fill)
        from . import who
        p_home = who.predict(who.load_model(_model_file(who_model)), trk.players, trk.team_of_slot, trk.ball, spec.fps,
                             spec.pitch, spec.halves)
    feat = features.build(trk, spec.pitch, spec.fps, ball_filled=filled, p_dead=p_dead, p_home=p_home)
    tags = None
    if use_tags and spec.smart_tagging and Path(spec.smart_tagging).exists():
        tags = smart_tagging.evidence(spec.smart_tagging, spec.atd_meta, spec.n_frames, spec.fps)
    alive_report = None
    if alive_on:                      # the mqf.alive decoder (its "hmm" variant = decode.decode, bit for bit)
        from .alive import combine as acombine
        variant, tb = alive_variant, None
        if alive_tag_w > 0 or alive_goal_w > 0:
            if not (use_tags and spec.smart_tagging and Path(spec.smart_tagging).exists()):
                raise SystemExit("--alive-tag-w / --alive-goal-w need the Smart Tagging file")
            sp, gl = acombine.tag_windows_from_xml(spec.smart_tagging, spec.fps, spec.n_frames)
            tb = acombine.tag_bonus(spec.n_frames, sp, spec.fps, goal_windows=gl, w_goal=alive_goal_w / max(alive_tag_w, 1e-9))
            variant = variant if "tags" in variant.split("+") else variant + "+tags"
            alive_report = {"set_piece_windows": len(sp), "goal_windows": len(gl)}
        if p_dead is None and "tags" in variant.split("+"):
            raise SystemExit("the Smart Tagging dead prior needs a learned P(dead) (--alive-model or --dead-model)")
        labels = acombine.decode_variant(variant, feat, p_dead, spec.halves, spec.n_frames, spec.fps, params, rr,
                                         p_bnd, tb, max(alive_tag_w, 1e-9) if tb is not None else 0.0, None, tags)
        if labels is None:
            raise SystemExit(f"--alive-variant {alive_variant}: needs a boundary (a tcn / asf --alive-model)")
        alive_report = {**(alive_report or {}), "variant": variant,
                        "rule_dead_share": round(float(rr["dead"][~np.isnan(rr["p_dead"])].mean()), 3) if rr else None}
    else:
        labels = decode.decode(feat, spec.halves, spec.n_frames, params, tags)
    submission.write(out, labels)
    # Aggregates only -- this report is what the team may share (e.g. with an AI assistant).
    report = {"file": Path(out).name, "seconds": round(time.time() - t0, 1),
              "ball_seen_pct": round(100 * float((feat["ball_seen"] & ~feat["ball_filled"]).mean()), 1),
              "used_tags": tags is not None, "clean_tracks": clean_report, "ball_fill": fill_report,
              **decode.spell_stats(labels, spec.fps),
              "dead_share_model": None if p_dead is None else round(float(np.mean(p_dead[~np.isnan(p_dead)] >= 0.5)), 3),
              "home_share_model": None if p_home is None else round(float(np.mean(p_home[~np.isnan(p_home)] >= 0.5)), 3),
              "alive": alive_report,
              "options": {"use_tags": use_tags, "clean_tracks": clean_tracks, "ball_fill": ball_fill,
                          "ball_model": Path(ball_model).as_posix() if ball_model else None,
                          "dead_model": Path(dead_model).as_posix() if dead_model else None,
                          "who_model": Path(who_model).as_posix() if who_model else None,
                          **({"alive_model": _alive_opt(alive_model),
                              "alive_variant": alive_variant, "alive_tag_w": alive_tag_w, "alive_goal_w": alive_goal_w}
                             if alive_on else {})},
              "params": asdict(params)}
    Path(out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="Perosinho_FC", help="file is named after the team")
    ap.add_argument("--final", action="store_true", help="the submitted recipe (FINAL in this file); no other option flags")
    ap.add_argument("--synthetic", action="store_true", help="generate and use the synthetic match")
    ap.add_argument("--no-tags", action="store_true", help="ablation: ignore Smart Tagging")
    ap.add_argument("--clean-tracks", action="store_true", help="team-level tracklet cleaning (mqf.tracks) before features")
    ap.add_argument("--ball-fill", action="store_true", help="fill the missing ball (mqf.ball.impute.FROZEN; torch, CPU)")
    ap.add_argument("--ball-model", type=Path, default=None, help="override the frozen ball model file")
    ap.add_argument("--dead-model", type=Path, default=None, help="learned P(dead) (mqf.dead), e.g. models/dead_v1.joblib")
    ap.add_argument("--who-model", type=Path, default=None, help="learned P(home | alive) (mqf.who), e.g. models/who_v1.joblib")
    ap.add_argument("--params", nargs="*", default=[], metavar="KEY=VALUE", help="decoder overrides, e.g. dead_who=max")
    ap.add_argument("--alive-model", nargs="+", type=Path, default=None,
                    help="learned P(dead) of mqf.alive (replaces --dead-model); several = their mean, e.g. "
                         "models/alive_asf_full_impff_s0.pt models/alive_asf_full_impff_s1.pt models/alive_asf_full_impff_s2.pt")
    ap.add_argument("--alive-variant", default="hmm", help="mqf.alive decoder variant, e.g. start+min2 (combine.py)")
    ap.add_argument("--alive-tag-w", type=float, default=0.0, help="Smart Tagging dead prior weight (SET PIECES windows)")
    ap.add_argument("--alive-goal-w", type=float, default=0.0, help="dead prior after GOALS windows (not validated)")
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

    saved = None
    if args.reproduce and not args.synthetic and reference.with_suffix(".json").exists():   # how the reference was made
        saved = json.loads(reference.with_suffix(".json").read_text())
        if not saved.get("options"):
            saved = None
    opts, params = settings(args, saved)
    if saved:
        print(f"reproducing with the reference's options {saved['options']} and decoder parameters "
              f"(non-default: {_non_default(params)})")
    report = run(spec, params, out, **opts)
    print(json.dumps({**{k: v for k, v in report.items() if k != "params"},
                      "params_non_default": _non_default(params)}, indent=2))
    if args.synthetic:
        return
    if args.reproduce:
        same = out.read_bytes() == reference.read_bytes()
        print(f"byte-identical to {reference.name}: {'OK' if same else 'FAIL'}")
        raise SystemExit(0 if same else 1)
    raise SystemExit(submission.validate(out))


if __name__ == "__main__":
    main()
