"""Is the ball in play? Every alive-layer method on the same held-out games, official metric, WHO fixed.

    python -m mqf.alive.experiment --split val --source idsse metrica pff --condition bar \
        --ball-modes exact fuzzy random --clean-tracks \
        --models gbm_full=models/alive_gbm_full_impff.joblib tcn0=models/alive_tcn_full_impff_s0.pt \
        --ensemble tcn=tcn0,tcn1,tcn2 --hybrid-base gbm_full --variants hmm start+min2 bnd0.5 --tag alive_dev

Conditions (players / noise; as EXPERIMENTS #34-#40):
  bar    ATD-like player dropouts (--players-drop); the ball as the bundle delivers it, then the ball mode
  atd    the full ATD noise model (46 anonymous slots, bursty ball loss, dropouts, duplicates), noise
         seeds --seeds, then the ball mode; --noise overrides NoiseConfig (atd_flip: team_flip_p=0.05 ...)
Ball modes (how the ball goes missing in stoppages -- BAR-ATH loses it mostly there, by how much is
unknown, so a decision must hold in all three):
  exact   removed on exactly the dead frames (the harness's --hide-dead-ball, #34-#40). Optimistic for any
          cue that reads where the ball vanishes; the features never read that (feats.py)
  fuzzy   removed on 75 % of the stoppages with jittered edges (start -1..+3 s, end -2..+1 s), plus
          label-free spans up to 41 % missing (DATA.md's share) -- assumption
  random  label-free spans only, up to 41 % missing (the delivered loss pattern of each source on top:
          PFF's broadcast tracker already loses it mostly in stoppages)
--clean-tracks runs the tracklet cleaner (as make_submission FINAL) after the noise / ball mode and before
the ball fill. The ball is filled with the frozen setting (mqf.ball.impute.FROZEN) for WHO and the
decoder, unless --no-ball-fill; the dead-ball methods always read the TRACKED ball.

WHO (fixed): --who who_v1 (default, the FINAL recipe) or --who NAME=TCN1.pt,TCN2.pt,... (an owner-TCN
ensemble with its own boundary on the H <-> A switch cost, alpha --who-bnd-alpha: #40's ens_impff).

Methods (P(dead) per frame), each decoded with every --variants entry (combine.py):
  v0         the speed rule (no learned P(dead))        dead_v1   mqf.dead's model (the reference)
  rules      rules.py's soft P(dead)                    <name>    every --models name=path (AliveModel)
  <e>        --ensemble <e>=<m1>,<m2>,..: mean P(dead) and mean boundary (members: models or earlier ensembles)
  <b>+prod / <b>+override   hybrids of the rules with each --hybrid-base b
  oracle     the true alive mask (2-state decoder inside it): the ceiling
bnd<a> uses the method's own boundary, or --bnd-from's for methods without one.
tags uses tag-like windows simulated from the labels (recall --tag-recall, --tag-fp per 90') with weight --tag-w.

Writes harness_results/alive_<tag>.json (every row) and .md (tables). Aggregates only, other games only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import atd, decode, features
from ..config import EXTERNAL_DIR, HARNESS_DIR, MODELS_DIR
from ..harness.evaluate import score
from . import combine, data, feats, rules
from .models import AliveModel

BALL_MODES = ("exact", "fuzzy", "random")
EVIDENCE_CACHE = EXTERNAL_DIR / "cache" / "alive_eval"
MISSING_TARGET = 0.41                     # DATA.md: share of graded frames without a ball
FUZZY = {"keep_p": 0.25, "start": (-1.0, 3.0), "end": (-2.0, 1.0)}


def _file_tag(p) -> str:
    if p is None:
        return "none"
    p = Path(p)
    return f"{p.name}:{p.stat().st_size}" if p.exists() else p.name


def _inside(halves, n):
    m = np.zeros(n, dtype=bool)
    for _, a, b in halves:
        m[a:b + 1] = True
    return m


# ------------------------------------------------------------------------------------ ball modes

def _top_up(ball, halves, fps, rng, target=MISSING_TARGET):
    n = len(ball)
    ins = _inside(halves, n)
    out = ball
    tries = 0
    while np.isnan(out[ins, 0]).mean() < target and tries < 10_000:
        h = halves[int(rng.integers(0, len(halves)))]
        a, b = h[1], h[2]
        ln = data._span(rng, fps)
        s = a + int(rng.integers(0, max(1, b - a + 1 - ln)))
        out[s:s + ln] = np.nan
        tries += 1
    return out


def apply_ball_mode(ball: np.ndarray, gt: np.ndarray, halves, fps: float, mode: str, rng) -> np.ndarray:
    out = np.array(ball, dtype=np.float64, copy=True)
    if mode == "exact":
        out[np.asarray(gt) == "D"] = np.nan
        return out
    if mode == "fuzzy":
        for _, a, b in halves:
            for s, e in combine.runs_of(np.asarray(gt[a:b + 1]), "D"):
                if rng.random() < FUZZY["keep_p"]:
                    continue
                s2 = int(a + s + rng.uniform(*FUZZY["start"]) * fps)
                e2 = int(a + e + rng.uniform(*FUZZY["end"]) * fps)
                s2, e2 = max(a, s2), min(b + 1, e2)
                if e2 > s2:
                    out[s2:e2] = np.nan
        return _top_up(out, halves, fps, rng)
    if mode == "random":
        return _top_up(out, halves, fps, rng)
    raise ValueError(mode)


# ------------------------------------------------------------------------------------ loading

def load_condition(d, condition: str, seed: int, ball_mode: str, clean_tracks: bool, noise_cfg=None):
    """(spec, tracking with the TRACKED ball, gt) of one bundle under condition x ball mode."""
    from ..external import evaluate as ev
    from ..harness.noise import NoiseConfig
    ncfg = replace(noise_cfg or NoiseConfig(), seed=seed) if condition == "atd" else None
    spec, trk, gt = ev.load(d, players_drop=(condition == "bar"), hide_dead_ball=False, noise_cfg=ncfg,
                            clean_tracks=False)
    rng = np.random.default_rng([seed, 97, BALL_MODES.index(ball_mode), zlib.crc32(spec.name.encode())])
    ball = apply_ball_mode(trk.ball, gt, spec.halves, spec.fps, ball_mode, rng).astype(np.float32)
    trk = atd.Tracking(trk.players, trk.player_speed, ball, trk.team_of_slot)
    if clean_tracks:
        from .. import tracks
        trk, _ = tracks.clean(trk, spec.fps, spec.pitch, tracks.CleanConfig(), halves=spec.halves)
    return spec, trk, gt


def parse_who(spec: str):
    """'who_v1' | 'NAME=PATH' | 'NAME=TCN1.pt,TCN2.pt' -> (name, [paths])."""
    name, eq, rest = spec.partition("=")
    if not eq:
        if name != "who_v1":
            raise SystemExit(f"--who: {spec!r} needs =PATH (only who_v1 has a default)")
        return name, [MODELS_DIR / "who_v1.joblib"]
    return name, [Path(p) for p in rest.split(",") if p]


def who_evidence(spec, trk, ball_f, filled, who_paths, device="cpu"):
    """(P(home), P(H <-> A boundary) or None) of the fixed owner model (who.py joblib or owner models)."""
    n, fps, halves = spec.n_frames, spec.fps, spec.halves
    if len(who_paths) == 1 and Path(who_paths[0]).suffix == ".joblib":
        import joblib
        o = joblib.load(who_paths[0])
        if "kind" not in o:                                         # who.py model (who_v1)
            from .. import who
            return who.predict((o["model"], o["columns"], o.get("meta", {})), trk.players, trk.team_of_slot, ball_f,
                               fps, spec.pitch, halves), None
    from ..owner import feats as ofeats, rules as orules
    from ..owner.models import OwnerModel
    members = [OwnerModel.load(p) for p in who_paths]
    views = []
    need_seq = any(m.kind in ("gru", "tcn") for m in members)
    for k in (0, 1):
        team = trk.team_of_slot if k == 0 else 1 - trk.team_of_slot
        X = ofeats.features(trk.players, team, ball_f, fps, spec.pitch, halves, "full_rules_heading", orules.RuleConfig(),
                            ball_filled=filled)
        S, sc = ofeats.seq_inputs(trk.players, team, ball_f, fps, spec.pitch, halves, X_full=X, columns="all") \
            if need_seq else (None, None)
        views.append({"X": X, "S": S, "seq_columns": sc})
    ps, pbs = [], []
    for m in members:
        p, pb = m.p_home_and_boundary(views, halves, n, fps, device)
        ps.append(p)
        if pb is not None:
            pbs.append(pb)
    return np.round(np.mean(ps, axis=0), 4), (np.round(np.mean(pbs, axis=0), 4) if pbs else None)


def evidence(spec, trk, who_paths, ball_model, ball_path, device, key_extra, use_cache=True):
    """Filled ball, filled mask, P(home), WHO boundary -- cached per game x condition."""
    key = hashlib.sha1(json.dumps([spec.name, *key_extra, _file_tag(ball_path), *[_file_tag(p) for p in who_paths]],
                                  default=str).encode()).hexdigest()[:12]
    path = EVIDENCE_CACHE / f"{spec.name}_{key}.npz"
    if use_cache and path.exists():
        z = np.load(path)
        return z["ball"], z["filled"], z["p_home"], (z["pb_who"] if z["has_pb"] else None)
    ball, filled = trk.ball.astype(np.float64), np.zeros(len(trk.ball), dtype=bool)
    if ball_model is not None:
        from ..ball import impute
        F = impute.FROZEN
        m, cfg = ball_model
        bf = impute.fill(trk.players, trk.team_of_slot, trk.ball, spec.fps, spec.pitch, spec.halves, F["method"],
                         F["tau_s"], m, cfg, interp_method=F["interp_method"], device=device,
                         anchor_s=F["anchor_s"], tau_linear_s=F["tau_linear_s"])
        filled = ((bf.source == impute.INTERP) | (bf.source == impute.MODEL)) & ~np.isnan(bf.ball[:, 0])
        ball = bf.ball.astype(np.float64)
    p_home, pb = who_evidence(spec, trk, ball, filled, who_paths, device)
    if use_cache:
        EVIDENCE_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ball=ball, filled=filled, p_home=p_home, pb_who=pb if pb is not None else np.zeros(1),
                            has_pb=pb is not None)
    return ball, filled, p_home, pb


# ------------------------------------------------------------------------------------ one game

def dead_methods(X, rr, spec, models: dict, ensembles: dict, hybrid_bases, device="cpu"):
    """{method: P(dead) or None (v0)}, {method: P(boundary)}."""
    n, fps, halves = spec.n_frames, spec.fps, spec.halves
    P = {"v0": None, "rules": rr["p_dead"]}
    PB = {}
    for name, m in models.items():
        P[name], pb = m.predict(X, halves, n, fps, device)
        if pb is not None:
            PB[name] = pb
    for e, members in ensembles.items():
        P[e] = np.round(np.mean(np.stack([P[k] for k in members]), axis=0), 4)
        bs = [PB[k] for k in members if k in PB]
        if bs:
            PB[e] = np.round(np.mean(np.stack(bs), axis=0), 4)
    for b in hybrid_bases:
        if b not in P or P[b] is None:
            raise SystemExit(f"--hybrid-base {b}: no such learned method")
        P[f"{b}+prod"] = combine.product(P[b], rr["p_dead"])
        P[f"{b}+override"] = combine.override(P[b], rr)
        if b in PB:
            PB[f"{b}+prod"] = PB[f"{b}+override"] = PB[b]
    return P, PB


def run_game(spec, trk, gt, ball_f, filled, p_home, pb_who, P, PB, rr, variants, params, meta, who_alpha,
             bnd_from=None, tag_cfg=None, variant_methods=None):
    from ..owner import combine as ocombine
    n, fps, halves = spec.n_frames, spec.fps, spec.halves
    tracking = atd.Tracking(trk.players, trk.player_speed, np.asarray(ball_f, dtype=np.float32), trk.team_of_slot)
    feat = features.build(tracking, spec.pitch, fps, ball_filled=filled, p_home=p_home)
    sw_t = None if pb_who is None else ocombine.switch_costs(pb_who, params.switch_cost, who_alpha)
    tb = None
    if tag_cfg:
        rng = np.random.default_rng([tag_cfg["seed"], zlib.crc32(spec.name.encode())])
        win = combine.simulate_tag_windows(gt, halves, fps, rng, tag_cfg["recall"], tag_cfg["fp_per90"])
        tb = combine.tag_bonus(n, win, fps)
    rows = []
    ins = _inside(halves, n)
    alive_true = np.isin(gt, ["H", "A"]) & ins
    lab = ocombine.decode_given_alive(p_home, alive_true, halves, n,
                                      params.switch_cost if sw_t is None else sw_t, params.w_who_model, params.who_clip)
    rows.append({**meta, "method": "oracle", "variant": "-", **score(gt, lab, fps),
                 **combine.stoppage_stats(gt, lab, halves, fps)})
    for name, p in P.items():
        if variant_methods and name not in variant_methods and name not in ("v0", "dead_v1"):
            vlist = ["hmm"]
        else:
            vlist = variants
        pb = PB.get(name, PB.get(bnd_from) if bnd_from else None)
        for v in vlist:
            lab = combine.decode_variant(v, feat, p, halves, n, fps, params, rr, pb, tb,
                                         tag_cfg["w"] if tag_cfg else 0.0, sw_t)
            if lab is None:
                continue
            rows.append({**meta, "method": name, "variant": v, **score(gt, lab, fps),
                         **combine.stoppage_stats(gt, lab, halves, fps)})
    return rows


# ------------------------------------------------------------------------------------ report

def _mean_by_source(df, col):
    if col not in df:
        return float("nan")
    per = df.groupby("source")[col].mean()
    return float(per.mean()) if len(per) else float("nan")


REPORT_COLS = ("spellF1", "macroF1", "alive_dead_f1", "rec_0-5s", "rec_5-12s", "rec_12-30s", "spurious_per90",
               "start_med_s", "end_med_s", "end_bias_s", "d_share_pred", "d_share_true")


def summarise(rows: list[dict], ref=("dead_v1", "hmm")) -> str:
    df = pd.DataFrame(rows)
    out = []
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    df["spell_ratio"] = df["n_pred_spells"] / df["n_gt_spells"].clip(lower=1)
    for (cond, mode), dl in df.groupby(["condition", "ball_mode"], sort=False):
        tab = []
        base = dl[(dl.method == ref[0]) & (dl.variant == ref[1])].set_index(["game", "seed"])["spellF1"]
        for (method, var), g in dl.groupby(["method", "variant"]):
            r = {"method": method, "variant": var, "n": len(g)}
            for c in REPORT_COLS + ("spell_ratio",):
                r[c] = _mean_by_source(g, c)
            for s, gs in g.groupby("source"):
                r[f"spellF1_{s}"] = float(gs.spellF1.mean())
            cur = g.set_index(["game", "seed"])["spellF1"]
            both = cur.index.intersection(base.index)
            if len(both) and (method, var) != ref:
                dlt = (cur[both] - base[both]).to_numpy()
                r["d_vs_ref"] = float(dlt.mean())
                r["se"] = float(dlt.std(ddof=1) / np.sqrt(len(dlt))) if len(dlt) > 1 else float("nan")
                r["wins"] = f"{int((dlt > 0).sum())}/{len(dlt)}"
            tab.append(r)
        t = pd.DataFrame(tab).sort_values("spellF1", ascending=False)
        out.append(f"## {cond} / ball {mode} -- mean per source, then over sources; d_vs_ref paired vs "
                   f"`{ref[0]}` `{ref[1]}`\n\n```\n" + t.round(3).to_string(index=False) + "\n```\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------------ main

def parse_models(items) -> dict:
    out = {}
    for it in items or []:
        name, _, path = it.partition("=")
        if not path:
            raise SystemExit(f"--models wants name=path, got {it}")
        out[name] = Path(path)
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", nargs="*", default=["idsse", "metrica", "pff"])
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--matches", nargs="*", help="only these games (names)")
    ap.add_argument("--max-games", type=int, help="first N games per source (smoke runs)")
    ap.add_argument("--condition", default="bar", choices=["bar", "atd"])
    ap.add_argument("--ball-modes", nargs="*", default=["exact", "fuzzy", "random"], choices=BALL_MODES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0], help="noise seeds (atd); bar uses seed 0")
    ap.add_argument("--noise", nargs="*", default=[], metavar="FIELD=VALUE", help="NoiseConfig overrides (atd)")
    ap.add_argument("--clean-tracks", action="store_true", help="tracklet cleaner (as the submission)")
    ap.add_argument("--who", default="who_v1", help="who_v1 | NAME=MODEL1,MODEL2 (owner TCN ensemble)")
    ap.add_argument("--who-bnd-alpha", type=float, default=0.5, help="H <-> A switch-cost alpha of a WHO boundary")
    ap.add_argument("--dead-model", type=Path, default=MODELS_DIR / "dead_v1.joblib")
    ap.add_argument("--models", nargs="*", default=[], metavar="NAME=PATH")
    ap.add_argument("--ensemble", nargs="*", default=[], metavar="NAME=M1,M2")
    ap.add_argument("--hybrid-base", nargs="*", default=[])
    ap.add_argument("--variants", nargs="*", default=["hmm"], help="decoder variants (combine.py), e.g. hmm start+min2")
    ap.add_argument("--variant-methods", nargs="*", default=None,
                    help="methods that get every variant (default: all); the others get hmm only")
    ap.add_argument("--bnd-from", default=None, help="sequence model whose boundary the others use for bnd<a>")
    ap.add_argument("--tag-recall", type=float, default=None, help="simulate tag windows (tags variant)")
    ap.add_argument("--tag-fp", type=float, default=2.0, help="false tag windows per 90 min")
    ap.add_argument("--tag-w", type=float, default=1.0)
    ap.add_argument("--tag-seed", type=int, default=0)
    ap.add_argument("--params", nargs="*", default=["dead_who=max"], metavar="FIELD=VALUE")
    ap.add_argument("--rule-cfg", nargs="*", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--no-ball-fill", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--jobs", type=int, default=1, help="games in parallel (processes; each holds one game, ~3-4 GB)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--tag", default="alive_dev")
    return ap


_CTX: dict = {}


def _setup(a_dict: dict):
    """Everything one worker needs, built once per process (models, ball model, device)."""
    a = argparse.Namespace(**a_dict)
    from ..external.evaluate import parse_grid, parse_noise
    over = parse_grid([f"{kv.split('=')[0]}={kv.split('=', 1)[1]}" for kv in a.params])[0]
    params = replace(decode.Params(), **over)
    rule_cfg = rules.parse_rule_cfg(a.rule_cfg)
    models = {"dead_v1": AliveModel.load(a.dead_model)}
    models.update({k: AliveModel.load(p) for k, p in parse_models(a.models).items()})
    ensembles = {}
    for it in a.ensemble:
        e, eq, members = it.partition("=")
        if not eq or not members or e in models:
            raise SystemExit(f"--ensemble: expected NAME=M1,M2 with a new NAME, got {it!r}")
        ensembles[e] = members.split(",")
        bad = [m for m in ensembles[e] if m not in models and m not in list(ensembles)[:-1]]
        if bad:
            raise SystemExit(f"--ensemble {e}: {bad} not in --models")
    device = "cpu"
    try:
        import torch
        device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    except ImportError:
        pass
    ball_model = ball_path = None
    if not a.no_ball_fill:
        from ..ball import impute
        ball_path = MODELS_DIR / impute.FROZEN["model"]
        ball_model = impute.load_model(ball_path, device)
    tag_cfg = None if a.tag_recall is None else {"recall": a.tag_recall, "fp_per90": a.tag_fp, "w": a.tag_w,
                                                 "seed": a.tag_seed}
    _CTX.clear()
    _CTX.update(a=a, params=params, rule_cfg=rule_cfg, models=models, ensembles=ensembles, device=device,
                ball_model=ball_model, ball_path=ball_path, noise_cfg=parse_noise(a.noise) if a.noise else None,
                who=parse_who(a.who), tag_cfg=tag_cfg)


def _job(job):
    """One game x noise seed x ball mode -> (rows, seconds, log line)."""
    d, source, seed, mode = job
    c = _CTX
    a = c["a"]
    t0 = time.time()
    who_name, who_paths = c["who"]
    spec, trk, gt = load_condition(Path(d), a.condition, seed, mode, a.clean_tracks, c["noise_cfg"])
    extra = [a.condition, seed, mode, a.clean_tracks, a.noise, who_name]
    ball_f, filled, p_home, pb_who = evidence(spec, trk, who_paths, c["ball_model"], c["ball_path"], c["device"], extra,
                                              not a.no_cache)
    X, rr = feats.features(trk.players, trk.team_of_slot, trk.ball, spec.fps, spec.pitch, spec.halves,
                           "full_rules", c["rule_cfg"], return_rules=True)
    P, PB = dead_methods(X, rr, spec, c["models"], c["ensembles"], a.hybrid_base, c["device"])
    del X
    meta = {"game": spec.name, "source": source, "seed": seed, "condition": a.condition, "ball_mode": mode}
    r = run_game(spec, trk, gt, ball_f, filled, p_home, pb_who, P, PB, rr, a.variants, c["params"], meta,
                 a.who_bnd_alpha, a.bnd_from, c["tag_cfg"], a.variant_methods)
    dt = time.time() - t0
    best = sorted(((x["spellF1"], x["method"], x["variant"]) for x in r if x["method"] != "oracle"), reverse=True)[:3]
    orc = next(x["spellF1"] for x in r if x["method"] == "oracle")
    ref = next((x["spellF1"] for x in r if x["method"] == "dead_v1" and x["variant"] == "hmm"), float("nan"))
    line = (f"{spec.name:<14} s{seed} {mode:<6} {dt:5.0f} s  oracle {orc:.3f}  dead_v1 {ref:.3f}  "
            + "  ".join(f"{m}[{v}] {s_:.3f}" for s_, m, v in best) + f"  | rules {rules.summary(rr, spec.fps)}")
    return r, round(dt, 1), line, f"{spec.name}_s{seed}_{mode}"


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    log = lambda m: print(m, flush=True)
    from ..external import bundle, splits
    for v in a.variants:
        combine.parse_variant(v)
    if a.tag_recall is None and any("tags" in combine.parse_variant(v) for v in a.variants):
        raise SystemExit("a tags variant needs --tag-recall")
    a_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    _setup(a_dict)                                         # validates everything in the main process too
    seeds = a.seeds if a.condition == "atd" else [0]
    jobs = []
    for source in a.source:
        names = a.matches or splits.games(a.split, sources=[source])
        dirs = bundle.find(source, names)
        if a.max_games:
            dirs = dirs[:a.max_games]
        if not dirs:
            log(f"[{source}] no bundles for the {a.split} games")
            continue
        jobs += [(str(d), source, seed, mode) for d in dirs for seed in seeds for mode in a.ball_modes]
    rows, timing = [], {}
    t_all = time.time()
    if a.jobs > 1 and len(jobs) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(a.jobs, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_setup, initargs=(a_dict,)) as ex:            # spawn: CUDA-safe
            for r, dt, line, key in ex.map(_job, jobs):
                rows += r
                timing[key] = dt
                log(line)
    else:
        for job in jobs:
            r, dt, line, key = _job(job)
            rows += r
            timing[key] = dt
            log(line)
    if not rows:
        return 1
    c = _CTX
    text = summarise(rows)
    print("\n" + text)
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {"tag": a.tag, "condition": a.condition, "ball_modes": a.ball_modes, "seeds": seeds, "noise": a.noise,
           "clean_tracks": a.clean_tracks, "who": a.who, "who_bnd_alpha": a.who_bnd_alpha,
           "models": {k: str(v) for k, v in parse_models(a.models).items()}, "dead_model": str(a.dead_model),
           "ensemble": a.ensemble, "hybrid_base": a.hybrid_base, "variants": a.variants,
           "variant_methods": a.variant_methods, "bnd_from": a.bnd_from, "tags": c["tag_cfg"],
           "params": asdict(c["params"]), "rule_cfg": asdict(c["rule_cfg"]), "ball_fill": not a.no_ball_fill,
           "fuzzy": FUZZY, "split": a.split, "sources": a.source, "jobs": a.jobs,
           "minutes": round((time.time() - t_all) / 60, 1)}
    (HARNESS_DIR / f"alive_{a.tag}.json").write_text(json.dumps({"config": cfg, "runs": rows, "timing": timing},
                                                                indent=1, default=float))
    (HARNESS_DIR / f"alive_{a.tag}.md").write_text(f"# alive experiment `{a.tag}`\n\n```\n{json.dumps(cfg, indent=1, default=str)}\n```\n\n" + text)
    log(f"wrote {HARNESS_DIR / f'alive_{a.tag}.json'} and .md  ({cfg['minutes']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
