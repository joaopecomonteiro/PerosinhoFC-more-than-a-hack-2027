"""Regenerate our submission: tracking (+ Smart Tagging) -> features -> models -> length-rule decoder -> CSV.

    python -m mqf.make_submission --team <team_name> --final  # THE submission: the frozen recipe FINAL below
    python -m mqf.make_submission --team <team_name> --final --reproduce   # regenerate + byte-compare (any OS)
    python -m mqf.make_submission --team <team_name>_hmm --final-hmm    # FINAL_HMM: same models, the HMM decoder
    python -m mqf.make_submission --team <team_name>      # v0 (no models), real match (data/ or MTAH_DATA_DIR)
    python -m mqf.make_submission --synthetic             # synthetic match, no data needed
    python -m mqf.make_submission --team <team_name> --ball-fill   # + missing ball filled (needs torch, CPU ok)
    python -m mqf.make_submission --team <team_name> --dead-model models/dead_v1.joblib   # + learned P(dead)
    python -m mqf.make_submission --team <team_name> --who-model models/who_v1.joblib     # + learned P(home | alive)
    python -m mqf.make_submission --team <team_name> --who-model models/owner_tcn_full_impff_s0.pt \
        models/owner_tcn_full_impff_s1.pt models/owner_tcn_full_impff_s2.pt   # owner TCN ensemble + its boundary
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

# The submitted configuration = FINAL_HMM's models and evidence + the stoppage decoder (#48). Model paths are
# relative to the package root. Change it only with a new EXPERIMENTS entry, and test it before submitting
# (python tasks.py final-check for the models, python run_fusion.py / mqf.fusion for the decoder).
#   possession: the owner TCN ensemble (3 seeds, IDSSE + Metrica + PFF) with its own H <-> A boundary on the
#               switch cost, alpha 0.5 (#40: +0.048 / +0.050 / +0.054 vs who_v1)
#   dead ball:  the mean P(dead) of the ASFormer-lite ensemble of mqf.alive (3 seeds, IDSSE + Metrica + PFF, #42)
#               and Italo's mqf.deadball v8 (#44), in the production HMM ("hmm"), no Smart Tagging dead prior.
#               final_check (#45): +0.005 ... +0.017 over asf alone under the owner TCN, in every column -- v8
#               alone scores higher where the ball vanishes in stoppages but loses when it does not
#   as before:  frozen ball fill, tracklet cleaner, Smart Tagging team votes, dead_who="max" (#34/#35)
FINAL_HMM = {"options": {"use_tags": True, "clean_tracks": True, "ball_fill": True, "ball_model": None,
                     "dead_model": None,
                     "who_model": ["models/owner_tcn_full_impff_s0.pt", "models/owner_tcn_full_impff_s1.pt",
                                   "models/owner_tcn_full_impff_s2.pt"],
                     "who_bnd_alpha": 0.5,
                     "alive_model": ["models/alive_asf_full_impff_s0.pt", "models/alive_asf_full_impff_s1.pt",
                                     "models/alive_asf_full_impff_s2.pt"],
                     "alive_variant": "hmm", "alive_tag_w": 0.0, "alive_goal_w": 0.0,
                     "deadball_model": "models/deadball_v8.joblib", "deadball_decoder": "hmm", "deadball_prior": 0.5},
         "params": {"dead_who": "max"}}
#   decoder (#48, 17 Nov): mqf.fusion "stoppage" on exactly the per-frame evidence FINAL_HMM's HMM reads (same
#               models, tags, clips and weights) but no switch costs and no boundary -- only length rules: every
#               possession and dead ball >= min_s, every dead ball >= min_dead_s, a possession with a dead ball on
#               both sides >= min_between_s (instead of min_s); exact semi-Markov Viterbi per half. The lengths
#               come from the 11 validation games x 6 noise columns (official metric, never BAR-ATH): 12 sampled
#               + 37 grid settings, nested leave-one-game-out 0.650-0.652 vs FINAL_HMM 0.654 (tied: no column
#               beyond 2 se either way), on a flat plateau (min_s 1.4-1.8 s, min_dead_s 3-4.5 s, min_between_s
#               0.5-1 s, all 0.652); its round centre, with half FINAL_HMM's spell-count error (-43 vs -98 per
#               game). 0.5 / 2 / 2 s scored 0.635 (-0.019, lost in every column). FINAL_HMM stays the reference
#               of final_check and of the viewer's "HMM decoder" lane.
STOPPAGE = {"min_s": 1.4, "min_dead_s": 3.0, "min_between_s": 1.0}
FINAL = {"options": {**FINAL_HMM["options"], "fusion_decoder": "stoppage", "fusion_config": STOPPAGE},
         "params": dict(FINAL_HMM["params"])}
# The recipe submitted until 16 Nov (#35-#39), kept for the comparison in final-check.
FINAL_V1 = {"options": {"use_tags": True, "clean_tracks": True, "ball_fill": True, "ball_model": None,
                        "dead_model": "models/dead_v1.joblib", "who_model": "models/who_v1.joblib"},
            "params": {"dead_who": "max"}}
_PATH_OPTS = ("ball_model", "dead_model", "who_model", "alive_model", "deadball_model")
_ALIVE_DEFAULTS = {"alive_model": None, "alive_variant": "hmm", "alive_tag_w": 0.0, "alive_goal_w": 0.0,
                   "deadball_model": None, "deadball_decoder": "hsmm", "deadball_prior": 0.5}
_WHO_BND_ALPHA = 0.5


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
    elif args.final or getattr(args, "final_hmm", False):
        if args.final and getattr(args, "final_hmm", False):
            raise SystemExit("--final or --final-hmm, not both")
        recipe = FINAL if args.final else FINAL_HMM
        flag = "--final" if args.final else "--final-hmm"
        given = [f for f in ("no_tags", "clean_tracks", "ball_fill", "ball_model", "dead_model", "who_model", "params",
                             "alive_model", "alive_tag_w", "alive_goal_w", "deadball_model", "fusion_decoder",
                             "fusion_config") if getattr(args, f, None)]
        if getattr(args, "who_bnd_alpha", None) is not None:
            given.append("who_bnd_alpha")
        if getattr(args, "deadball_decoder", "hsmm") != "hsmm":
            given.append("deadball_decoder")
        if getattr(args, "deadball_prior", 0.5) != 0.5:
            given.append("deadball_prior")
        if getattr(args, "alive_variant", "hmm") != "hmm":
            given.append("alive_variant")
        if given:
            raise SystemExit(f"{flag} is a frozen recipe; drop {', '.join('--' + g.replace('_', '-') for g in given)}")
        opts, params = {**recipe["options"]}, replace(decode.Params(), **recipe["params"])
        if opts.get("fusion_config") is not None:
            opts["fusion_config"] = dict(opts["fusion_config"])
    else:
        who_m = args.who_model
        if isinstance(who_m, (list, tuple)):
            who_m = None if not who_m else (who_m[0] if len(who_m) == 1 else list(who_m))
        opts = {"use_tags": not args.no_tags, "clean_tracks": args.clean_tracks, "ball_fill": args.ball_fill,
                "ball_model": args.ball_model, "dead_model": args.dead_model, "who_model": who_m,
                "who_bnd_alpha": (_WHO_BND_ALPHA if getattr(args, "who_bnd_alpha", None) is None else args.who_bnd_alpha),
                **{k: getattr(args, k, v) for k, v in _ALIVE_DEFAULTS.items()}}
        if opts["alive_model"] and opts["dead_model"]:
            raise SystemExit("--alive-model replaces --dead-model: give one of them")
        if opts["deadball_model"] and opts["dead_model"]:
            raise SystemExit("--deadball-model replaces --dead-model: give one of them")
        if opts["deadball_model"] and opts["alive_model"] and opts["deadball_decoder"] != "hmm":
            raise SystemExit("--deadball-model with --alive-model averages their P(dead): needs --deadball-decoder hmm")
        from .alive import combine as acombine
        acombine.parse_variant(opts["alive_variant"])
        params = replace(decode.Params(), **parse_params(args.params))
        fdec, fcfg = load_fusion(getattr(args, "fusion_decoder", None), getattr(args, "fusion_config", None))
        if fdec:
            opts["fusion_decoder"], opts["fusion_config"] = fdec, fcfg
    return {k: (Path(v) if k in _PATH_OPTS and v and not isinstance(v, (list, tuple)) else v)
            for k, v in opts.items()}, params


def load_fusion(decoder, config_path):
    """(--fusion-decoder, config dict) from the flags; the config file is a run_fusion verdict / _best.json
    entry ({"decoder": .., "config": {..}}) or a bare config dict."""
    if not decoder and not config_path:
        return None, None
    if not (decoder and config_path):
        raise SystemExit("--fusion-decoder and --fusion-config go together")
    from . import fusion
    if decoder not in fusion.DECODERS:
        raise SystemExit(f"--fusion-decoder: one of {', '.join(fusion.DECODERS)}")
    cfg = json.loads(Path(config_path).read_text())
    cfg = cfg.get("config", cfg)
    if decoder == "hsmm_emp" and "_dur_prior" not in cfg:
        raise SystemExit("--fusion-decoder hsmm_emp: the config needs its frozen _dur_prior (run_fusion writes it)")
    return decoder, cfg


def _fusion_labels(spec, trk, filled, feat, params, decoder, cfg, p_dead, p_asf, p_db, p_bnd, p_home, pb_who,
                   who_files, who_bnd_alpha, tags):
    """Labels of a fusion decoder (mqf.fusion) on the arrays FINAL's decoder reads (opt-in)."""
    from . import fusion, fusion_export
    from .alive import predict as apredict
    if p_dead is None or p_home is None or pb_who is None:
        raise SystemExit("--fusion-decoder needs a learned P(dead) and owner --who-model(s) with a boundary")
    cfg = dict(cfg)
    prior = cfg.pop("_dur_prior", None)
    seeds = v1 = None
    if cfg.get("home_mix", "prob_mean") != "prob_mean" or cfg.get("kappa", 0.0) > 0:
        seeds = np.stack([apredict.owner_p_home([f], trk.players, trk.team_of_slot, trk.ball, filled, spec.fps,
                                                spec.pitch, spec.halves, device="cpu", float64=True)[0]
                          for f in who_files])
    if cfg.get("v1_share", 0.0) > 0:
        from . import who
        v1 = who.predict(who.load_model(MODELS_DIR / "who_v1.joblib"), trk.players, trk.team_of_slot, trk.ball,
                         spec.fps, spec.pitch, spec.halves)
    g = fusion_export.submission_g(spec, feat, p_dead, p_home, pb_who, who_bnd_alpha, params, tags,
                                   p_asf, p_db, p_bnd, seeds, v1)
    if prior is not None:
        g["_dur_prior"] = prior
    codes = fusion.decode(decoder, g, cfg)
    return np.array(["H", "A", "D", "X"])[codes]


def _dump_arrays(path, spec, feat, params, labels, p_dead, p_asf, p_db, p_bnd, p_home, pb_who, who_bnd_alpha, tags,
                 decoder=None, config=None):
    """Save the arrays this run's decoder read (mqf.fusion_export.submission_g) + its labels, for the viewer's
    'Models, no decoder' / 'FINAL (decoder)' lanes. Derived from the match: only inside cache/ (gitignored) or
    outside the repo. Prints aggregates only."""
    from . import fusion, fusion_export
    path = Path(path).resolve()
    if path.is_relative_to(ROOT.resolve()) and not path.is_relative_to(CACHE_DIR.resolve()):
        raise SystemExit(f"--dump-arrays: {path} is inside the repo but not in cache/ -- arrays derived from the "
                         "match must never be committed")
    if p_dead is None or p_home is None or pb_who is None:
        print("--dump-arrays skipped: needs a learned P(dead) and owner --who-model(s) with a boundary (FINAL has both)")
        return
    g = fusion_export.submission_g(spec, feat, p_dead, p_home, pb_who, who_bnd_alpha, params, tags,
                                   p_asf, p_db, p_bnd)
    g["pred_final"] = fusion_export.to_codes(labels)
    dec, cfg = decoder or "hmm3_bnd", config if decoder else fusion.CENTRE
    same = bool(np.array_equal(fusion.decode(dec, g, cfg), g["pred_final"]))
    if decoder:
        g["submission_decoder"], g["submission_config"] = np.str_(decoder), np.str_(json.dumps(cfg, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: (np.str_(v) if isinstance(v, str) else np.asarray(v)) for k, v in g.items()})
    print(f"arrays for the viewer: {path} (fusion {dec} == these labels: {'yes' if same else 'NO'})")


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
        who_model=None, alive_model=None, alive_variant: str = "hmm",
        alive_tag_w: float = 0.0, alive_goal_w: float = 0.0, who_bnd_alpha: float = _WHO_BND_ALPHA,
        deadball_model=None, deadball_decoder: str = "hsmm", deadball_prior: float = 0.5,
        fusion_decoder: str | None = None, fusion_config: dict | None = None, dump_arrays: Path | None = None) -> dict:
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
    p_dead = p_asf = p_db = None
    alive_on = bool(alive_model) or alive_variant != "hmm" or alive_tag_w > 0
    p_bnd = rr = None
    if alive_model:                   # one model or several (averaged, as experiment.py's --ensemble)
        from .alive import predict as apredict
        alive_files = [_model_file(p) for p in apredict.alive_paths(alive_model)]
        p_dead, p_bnd, rr = apredict.p_dead(alive_files, trk.players, trk.team_of_slot, tracked_ball,
                                            spec.fps, spec.pitch, spec.halves, device="cpu", float64=True)
        p_asf = p_dead
    elif alive_on:
        from .alive import predict as apredict
        rr = apredict.rules_only(trk.players, trk.team_of_slot, tracked_ball, spec.fps, spec.pitch, spec.halves)
    dead_mask = None
    if deadball_model:                # mqf.deadball (Italo's v8): its P(dead); "hsmm" = its own semi-Markov decoder
        from .deadball import decoder as ddec, model as dmod
        dm = dmod.load(_model_file(deadball_model))
        p_db = dmod.predict(dm, atd.Tracking(trk.players, trk.player_speed, tracked_ball.astype(np.float32),
                                             trk.team_of_slot), spec)
        if deadball_decoder == "hsmm":
            tables = ddec.length_tables(dm["lengths"]["alive"]), ddec.length_tables(dm["lengths"]["dead"])
            dead_mask = ddec.decode(p_db, spec.halves, spec.fps, tables, w=0.2, prior=deadball_prior)
        elif deadball_decoder != "hmm":
            raise SystemExit(f"--deadball-decoder must be hsmm or hmm, got {deadball_decoder!r}")
        p_dead = p_db if p_dead is None else np.round(np.mean(np.stack([p_dead, p_db]), axis=0), 4)
    if dead_model:
        from . import dead
        p_dead = dead.predict(dead.load_model(_model_file(dead_model)), trk.players, trk.team_of_slot, spec.fps, spec.pitch,
                              spec.halves)
    p_home = pb_who = sw_t = None
    who_files = []
    if who_model:                     # reads the ball as the decoder gets it (filled when --ball-fill)
        from .alive import predict as apredict
        who_files = [_model_file(p) for p in apredict.alive_paths(who_model)]
        if len(who_files) == 1 and apredict.is_who_joblib(who_files[0]):           # mqf.who (who_v1)
            from . import who
            p_home = who.predict(who.load_model(who_files[0]), trk.players, trk.team_of_slot, trk.ball, spec.fps,
                                 spec.pitch, spec.halves)
        else:                                                                       # mqf.owner model(s), e.g. the TCNs
            p_home, pb_who = apredict.owner_p_home(who_files, trk.players, trk.team_of_slot, trk.ball, filled, spec.fps,
                                                   spec.pitch, spec.halves, device="cpu", float64=True)
            if pb_who is not None and who_bnd_alpha > 0:
                from .owner import combine as ocombine
                sw_t = ocombine.switch_costs(pb_who, params.switch_cost, who_bnd_alpha)
    feat = features.build(trk, spec.pitch, spec.fps, ball_filled=filled, p_dead=p_dead, p_home=p_home)
    tags = None
    if use_tags and spec.smart_tagging and Path(spec.smart_tagging).exists():
        tags = smart_tagging.evidence(spec.smart_tagging, spec.atd_meta, spec.n_frames, spec.fps)
    alive_report = None
    if fusion_decoder:                # opt-in: a decoder of the fusion suite on FINAL's arrays (run_fusion.py)
        labels = _fusion_labels(spec, trk, filled, feat, params, fusion_decoder, fusion_config, p_dead, p_asf, p_db,
                                p_bnd, p_home, pb_who, who_files, who_bnd_alpha, tags)
        alive_report = {"fusion_decoder": fusion_decoder}
    elif dead_mask is not None:       # stoppages fixed by the deadball decoder, possession decoded inside them
        from .alive import combine as acombine
        labels = acombine.decode_masked(feat, dead_mask, spec.halves, spec.n_frames, params, sw_t, tags)
        alive_report = {"deadball_decoder": "hsmm", "deadball_prior": deadball_prior}
    elif alive_on or sw_t is not None:  # the mqf.alive decoder (its "hmm" variant = decode.decode, bit for bit)
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
                                         p_bnd, tb, max(alive_tag_w, 1e-9) if tb is not None else 0.0, sw_t, tags)
        if labels is None:
            raise SystemExit(f"--alive-variant {alive_variant}: needs a boundary (a tcn / asf --alive-model)")
        alive_report = {**(alive_report or {}), "variant": variant,
                        "rule_dead_share": round(float(rr["dead"][~np.isnan(rr["p_dead"])].mean()), 3) if rr else None}
    else:
        labels = decode.decode(feat, spec.halves, spec.n_frames, params, tags)
    submission.write(out, labels)
    if dump_arrays:                   # the viewer's fusion lanes (mqf.viewer --fusion-arrays); never committed
        _dump_arrays(dump_arrays, spec, feat, params, labels, p_dead, p_asf, p_db, p_bnd, p_home, pb_who,
                     who_bnd_alpha, tags, fusion_decoder, fusion_config)
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
                          "who_model": _alive_opt(who_model),
                          **({"who_bnd_alpha": who_bnd_alpha} if pb_who is not None else {}),
                          **({"deadball_model": Path(deadball_model).as_posix(), "deadball_decoder": deadball_decoder,
                              "deadball_prior": deadball_prior} if deadball_model else {}),
                          **({"alive_model": _alive_opt(alive_model),
                              "alive_variant": alive_variant, "alive_tag_w": alive_tag_w, "alive_goal_w": alive_goal_w}
                             if alive_on else {}),
                          **({"fusion_decoder": fusion_decoder, "fusion_config": fusion_config} if fusion_decoder else {})},
              "params": asdict(params)}
    Path(out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="Perosinho_FC", help="file is named after the team")
    ap.add_argument("--final", action="store_true", help="the submitted recipe (FINAL in this file); no other option flags")
    ap.add_argument("--final-hmm", action="store_true",
                    help="FINAL_HMM: FINAL's models with the production HMM decoder (the recipe of 16 Nov)")
    ap.add_argument("--synthetic", action="store_true", help="generate and use the synthetic match")
    ap.add_argument("--no-tags", action="store_true", help="ablation: ignore Smart Tagging")
    ap.add_argument("--clean-tracks", action="store_true", help="team-level tracklet cleaning (mqf.tracks) before features")
    ap.add_argument("--ball-fill", action="store_true", help="fill the missing ball (mqf.ball.impute.FROZEN; torch, CPU)")
    ap.add_argument("--ball-model", type=Path, default=None, help="override the frozen ball model file")
    ap.add_argument("--dead-model", type=Path, default=None, help="learned P(dead) (mqf.dead), e.g. models/dead_v1.joblib")
    ap.add_argument("--who-model", nargs="+", type=Path, default=None,
                    help="learned P(home | alive): a mqf.who model (models/who_v1.joblib) or mqf.owner model(s), several = "
                         "their mean, e.g. models/owner_tcn_full_impff_s0.pt models/owner_tcn_full_impff_s1.pt "
                         "models/owner_tcn_full_impff_s2.pt")
    ap.add_argument("--who-bnd-alpha", type=float, default=None,
                    help=f"H <-> A switch-cost alpha of an owner model's boundary (default {_WHO_BND_ALPHA}, as #40)")
    ap.add_argument("--params", nargs="*", default=[], metavar="KEY=VALUE", help="decoder overrides, e.g. dead_who=max")
    ap.add_argument("--alive-model", nargs="+", type=Path, default=None,
                    help="learned P(dead) of mqf.alive (replaces --dead-model); several = their mean, e.g. "
                         "models/alive_asf_full_impff_s0.pt models/alive_asf_full_impff_s1.pt models/alive_asf_full_impff_s2.pt")
    ap.add_argument("--alive-variant", default="hmm", help="mqf.alive decoder variant, e.g. start+min2 (combine.py)")
    ap.add_argument("--alive-tag-w", type=float, default=0.0, help="Smart Tagging dead prior weight (SET PIECES windows)")
    ap.add_argument("--alive-goal-w", type=float, default=0.0, help="dead prior after GOALS windows (not validated)")
    ap.add_argument("--deadball-model", type=Path, default=None,
                    help="mqf.deadball P(dead) (Italo's models/deadball_v8.joblib) instead of --dead-model; with "
                         "--alive-model and --deadball-decoder hmm: the mean of the two")
    ap.add_argument("--deadball-decoder", default="hsmm", choices=["hsmm", "hmm"],
                    help="hsmm: its own semi-Markov decoder fixes the stoppages; hmm: its P(dead) in our HMM")
    ap.add_argument("--deadball-prior", type=float, default=0.5, help="hsmm decoder prior (0.5 = the frozen v8)")
    ap.add_argument("--fusion-decoder", default=None, help="opt-in: a mqf.fusion decoder (run_fusion.py verdict)")
    ap.add_argument("--fusion-config", type=Path, default=None, help="its config JSON (run_fusion.py writes it)")
    ap.add_argument("--dump-arrays", type=Path, default=None,
                    help="also save the decoder's input arrays for the viewer (inside cache/, e.g. "
                         "cache/viewer/final_arrays.npz); does not change the CSV")
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
    report = run(spec, params, out, **opts, dump_arrays=args.dump_arrays)
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
