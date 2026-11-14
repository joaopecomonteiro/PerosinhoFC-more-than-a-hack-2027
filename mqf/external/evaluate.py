"""Our pipeline on the bundles of other games, scored with the OFFICIAL metric -- no noise added.

    python -m mqf.external.evaluate --tag v0
    python -m mqf.external.evaluate --source metrica --tag v0
    python -m mqf.external.evaluate --split val --tag v0     # only the validation games (splits.json)
    python -m mqf.external.evaluate --split val --source idsse metrica pff --dead-model models/dead_v1.joblib \
        --players-drop --hide-dead-ball --ball-fill --grid w_dead=0.5,1,2 dead_who=zero,max --tag dead_v1
    python -m mqf.external.evaluate --split val --source idsse metrica pff --atd-noise --seeds 0 1 \
        --noise team_flip_p=0.05 gk_team_flip_p=0.5 --clean-tracks --tag teamfix_on
    ... --clean-cfg inherit_team=0 kickoff_check=0 fast_windows_s= --tag teamfix_off    # the v3 cleaner

Exactly the real-match path: tracking file -> atd.load -> features -> decode (HMM per half), then
possession_metrics.py from vendor/. Smart Tagging is not used (other games have none). Writes the
predictions to external/results/<tag>/ and the aggregates to harness_results/bundles_<tag>.json.
Report sources separately: their label rules differ (DFL tracking, Metrica events, SkillCorner
phases) and SkillCorner runs at 10 fps.
"""
from __future__ import annotations

import argparse
import itertools
import json
import zlib
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import atd, decode, features, submission
from ..config import EXTERNAL_DIR, HARNESS_DIR
from ..harness.evaluate import score
from ..tracks import parse_clean_cfg  # noqa: F401  (lives in mqf.tracks; still importable from here)
from . import bundle
from .build import SOURCES


def load(d, players_drop: bool = False, hide_dead_ball: bool = False, clean_tracks: bool = False, seed: int = 0,
         noise_cfg=None, clean_cfg=None):
    """(spec, tracking, ground truth) of one bundle, with the optional BAR-ATH-like conditions:
    players_drop   ATD-like dropout bursts on the players (mqf.ball.data._player_mask)
    hide_dead_ball the ball removed on every dead frame inside the halves (BAR-ATH loses it in stoppages)
    noise_cfg      the full ATD noise model (mqf.harness.noise.corrupt: 46 anonymous slots, ball loss,
                   dropouts, duplicates, optional wrong-team pickups) -- bundles have one slot per player
    clean_cfg      a tracks.CleanConfig for --clean-tracks (default: the current one)"""
    spec = bundle.spec(d)
    trk = atd.load(spec, cache_dir=d / "cache")
    gt = pd.read_csv(spec.gt_public, dtype={"label": str})["label"].to_numpy()
    if noise_cfg is not None:
        from ..harness.noise import corrupt
        trk = corrupt(trk.players.astype(np.float64), trk.team_of_slot, trk.ball.astype(np.float64), gt != "X",
                      spec.fps, noise_cfg)
    if players_drop:
        from ..ball import data as bdata
        rng = np.random.default_rng([seed, 31, zlib.crc32(spec.name.encode())])   # stable across runs
        xy = bdata._player_mask(trk.players.astype(np.float64), rng, spec.fps).astype(np.float32)
        sp = trk.player_speed.copy()
        sp[np.isnan(xy[..., 0])] = np.nan
        trk = atd.Tracking(xy, sp, trk.ball, trk.team_of_slot)
    if hide_dead_ball:
        ball = trk.ball.copy()
        ball[gt == "D"] = np.nan
        trk = atd.Tracking(trk.players, trk.player_speed, ball, trk.team_of_slot)
    if clean_tracks:
        from .. import tracks
        trk, _ = tracks.clean(trk, spec.fps, spec.pitch, clean_cfg or tracks.CleanConfig(), halves=spec.halves)
    return spec, trk, gt


def evidence(spec, trk, ball_model=None, dead_model=None, device="cpu"):
    """features.build with the optional filled ball (frozen setting) and learned P(dead)."""
    filled = p_dead = None
    if ball_model is not None:
        from ..ball import impute
        F = impute.FROZEN
        m, cfg = ball_model
        bf = impute.fill(trk.players, trk.team_of_slot, trk.ball, spec.fps, spec.pitch, spec.halves, F["method"],
                         F["tau_s"], m, cfg, interp_method=F["interp_method"], device=device,
                         anchor_s=F["anchor_s"], tau_linear_s=F["tau_linear_s"])
        filled = (bf.source == impute.INTERP) | (bf.source == impute.MODEL)
        trk = atd.Tracking(trk.players, trk.player_speed, bf.ball.astype(np.float32), trk.team_of_slot)
    if dead_model is not None:
        from .. import dead
        p_dead = dead.predict(dead_model, trk.players, trk.team_of_slot, spec.fps, spec.pitch, spec.halves)
    return features.build(trk, spec.pitch, spec.fps, ball_filled=filled, p_dead=p_dead)


def predict(d, params: decode.Params, clean_tracks: bool = False) -> tuple:
    spec, trk, _ = load(d, clean_tracks=clean_tracks)
    feat = features.build(trk, spec.pitch, spec.fps)
    return spec, decode.decode(feat, spec.halves, spec.n_frames, params, tags=None)


def parse_grid(items) -> list[dict]:
    """["w_dead=0.5,1", "dead_who=zero,max"] -> every combination as Params overrides."""
    types = {f.name: type(f.default) for f in fields(decode.Params)}
    axes = []
    for it in items or []:
        k, _, vals = it.partition("=")
        if k not in types:
            raise SystemExit(f"--grid: unknown decoder parameter {k} (Params: {', '.join(types)})")
        axes.append([(k, types[k](v)) for v in vals.split(",")])
    return [dict(c) for c in itertools.product(*axes)] if axes else [{}]


def parse_noise(items):
    from ..harness.noise import NoiseConfig
    base = NoiseConfig()
    over = dict(kv.split("=", 1) for kv in items or [])
    bad = [k for k in over if not hasattr(base, k)]
    if bad:
        raise SystemExit(f"--noise: unknown NoiseConfig field(s) {bad}")
    return replace(base, **{k: type(getattr(base, k))(float(v)) for k, v in over.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", nargs="*", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--split", choices=["train", "val"], help="only the games of that part of splits.json")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--clean-tracks", action="store_true", help="team-level tracklet cleaning (mqf.tracks) before features")
    ap.add_argument("--dead-model", type=Path, help="learned P(dead) (mqf.dead), e.g. models/dead_v1.joblib")
    ap.add_argument("--ball-fill", action="store_true", help="fill the missing ball with the frozen setting")
    ap.add_argument("--players-drop", action="store_true", help="ATD-like player dropouts (condition)")
    ap.add_argument("--hide-dead-ball", action="store_true", help="no ball on dead frames (condition, as BAR-ATH)")
    ap.add_argument("--grid", nargs="*", help="decoder parameters to sweep, e.g. w_dead=0.5,1,2 dead_who=zero,max")
    ap.add_argument("--atd-noise", action="store_true", help="the full ATD noise model (mqf.harness.noise) on each bundle")
    ap.add_argument("--noise", nargs="*", default=[], metavar="FIELD=VALUE", help="NoiseConfig overrides (with --atd-noise)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0], help="noise seeds per game (with --atd-noise)")
    ap.add_argument("--clean-cfg", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="CleanConfig overrides for --clean-tracks, e.g. inherit_team=0 kickoff_check=0 fast_windows_s=")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    combos = parse_grid(args.grid)
    noise_base = parse_noise(args.noise) if args.atd_noise else None
    clean_cfg = parse_clean_cfg(args.clean_cfg)
    ball_model = dead_model = None
    device = "cpu"
    if args.ball_fill:
        import torch
        from ..ball import impute
        from ..config import MODELS_DIR
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        ball_model = impute.load_model(MODELS_DIR / impute.FROZEN["model"], device)
    if args.dead_model:
        from .. import dead
        dead_model = dead.load_model(args.dead_model)
    rows = []
    for source in args.source:
        names = args.matches
        if args.split:
            from . import splits
            names = splits.games(args.split, sources=[source])
        dirs = bundle.find(source, names)
        if not dirs:
            print(f"[{source}] no bundles -- python -m mqf.external.build --source {source}")
            continue
        for d, seed in itertools.product(dirs, args.seeds if noise_base is not None else [0]):
            ncfg = replace(noise_base, seed=seed) if noise_base is not None else None
            spec, trk, gt = load(d, args.players_drop, args.hide_dead_ball, args.clean_tracks, noise_cfg=ncfg,
                                 clean_cfg=clean_cfg)
            feat = evidence(spec, trk, ball_model, dead_model, device)
            for ci, over in enumerate(combos):
                params = replace(decode.Params(), **over)
                pred = decode.decode(feat, spec.halves, spec.n_frames, params, tags=None)
                if len(combos) == 1:
                    suffix = f"_s{seed}" if ncfg is not None else ""
                    submission.write(EXTERNAL_DIR / "results" / args.tag / f"{spec.name}{suffix}.csv", pred)
                r = score(gt, pred, spec.fps)
                rows.append({"source": source, "match": spec.name, "fps": spec.fps, "combo": ci,
                             "noise_seed": seed if ncfg is not None else None, "params": json.dumps(over), **r})
                if len(combos) == 1:
                    buckets = "  ".join(f"{k}:{v['match_rate']:.2f}(n={v['n']})" for k, v in r["by_length"].items()
                                        if v["match_rate"] is not None)
                    print(f"{spec.name:<14} spellF1 {r['spellF1']:.3f}  macroF1 {r['macroF1']:.3f}  "
                          f"spells {r['n_pred_spells']}/{r['n_gt_spells']}  dur {r['pred_dur_s']:.1f}/{r['gt_dur_s']:.1f} s  "
                          f"aliveF1 {r['alive_dead_f1']:.3f}  team {r['team_acc_both_alive']:.3f}  {buckets}", flush=True)
            if len(combos) > 1:
                print(f"{spec.name:<14} {len(combos)} decoder settings scored", flush=True)
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    print()
    if len(combos) == 1:
        for source, g in df.groupby("source", sort=False):
            print(f"MEAN {source:<12} spellF1 {g.spellF1.mean():.3f} ± {g.spellF1.std(ddof=0):.3f}   "
                  f"macroF1 {g.macroF1.mean():.3f} ± {g.macroF1.std(ddof=0):.3f}   "
                  f"aliveF1 {g.alive_dead_f1.mean():.3f}   ({len(g)} games)")
    else:
        per = df.groupby(["params", "source"])[["spellF1", "macroF1", "alive_dead_f1"]].mean().unstack("source")
        per.columns = [f"{m}_{s}" for m, s in per.columns]
        for m in ("spellF1", "macroF1", "alive_dead_f1"):         # every source counts the same
            per[f"{m}_mean"] = per[[c for c in per.columns if c.startswith(m + "_") and not c.endswith("_mean")]].mean(axis=1)
        pd.set_option("display.width", 240)
        pd.set_option("display.max_colwidth", 60)
        cols = ["spellF1_mean", "macroF1_mean", "alive_dead_f1_mean"] + sorted(c for c in per.columns if c.startswith("spellF1_") and c != "spellF1_mean")
        print("=== decoder grid (mean over games per source, then over sources) ===")
        print(per.sort_values("spellF1_mean", ascending=False)[cols].round(3).to_string())
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    out = HARNESS_DIR / f"bundles_{args.tag}.json"
    cond = {"clean_tracks": args.clean_tracks, "clean_cfg": args.clean_cfg,
            "atd_noise": asdict(noise_base) if noise_base is not None else None, "seeds": args.seeds if noise_base else None,
            "dead_model": str(args.dead_model) if args.dead_model else None,
            "ball_fill": args.ball_fill, "players_drop": args.players_drop, "hide_dead_ball": args.hide_dead_ball}
    out.write_text(json.dumps({"params": asdict(decode.Params()), "grid": combos, **cond, "runs": rows,
                               "mean_by_source": {s: {"spellF1": float(g.spellF1.mean()), "macroF1": float(g.macroF1.mean()),
                                                      "n": int(len(g))} for s, g in df[df.combo == 0].groupby("source")}},
                              indent=2, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
