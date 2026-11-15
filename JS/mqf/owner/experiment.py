"""Who owns the ball, given an alive layer: every method on the same held-out games, official metric.

    python -m mqf.owner.experiment --split val --source idsse metrica pff \
        --models who_v1=models/who_v1.joblib gbm_full=models/owner_gbm_full_im.joblib \
        --layers oracle dead_v1 --condition bar --tag owner_dev

Conditions (as EXPERIMENTS #34-36, so numbers line up with who_v1's):
  bar   --players-drop --hide-dead-ball --ball-fill   (#35: alive ball as delivered, dropouts)
  atd   the full ATD noise model (46 anonymous slots, bursty ball loss, dropouts, duplicates)
        --hide-dead-ball --ball-fill, noise seeds --seeds (#34 b)
  clean         players as delivered, ball hidden on dead frames, filled   (PathCRF ran on clean players)
  clean_noball  players as delivered, NO ball anywhere, no fill            (ball-free, as PathCRF)
The ball is filled with the frozen setting (mqf.ball.impute.FROZEN) unless --no-ball-fill; methods
read the filled ball (rules: contacts on a filled ball confirm, never control), the smooth50 layer
and method read the ball as tracked.

Methods (P(home | alive) per frame; the alive layer and the decoder are applied on top):
  nearest      who is closer now (in the dead_v1 layer: exactly v0's emissions, #35's "base")
  hold         the last team within 2 m of the ball keeps it
  blend        no training: who_lab's mean of z_couple and z_chain (feats.heading_columns) as log-odds
  smooth50     the official baseline's team (hard)
  rule         rules.py soft P(home); rule_hard = its owner as clipped evidence
  rule_seg     rules.py restarted at every alive run of the layer (who restarts = first contact)
  <model>      every --models name=path (OwnerModel: who.py joblib, owner gbm / mlp / gru);
  <model>_hard its argmax as clipped evidence (hard vs probabilistic, same decoder)
  <b>+prod / +prod0.5 / +override / +veto   hybrids of the rule with each --hybrid-base model b
  <m>+bnd<a>   <m>'s P(home) decoded with the TCN boundary (--boundary-from): switch cost x (1 - a P(bnd)),
               for the TCN itself, --ref and the hybrid bases, a in --bnd-alpha; with several TCNs each gets
               its own (<ref>+<tcn>bnd<a>)
  <e>          --ensemble <e>=<tcn1>,<tcn2>,...: the mean P(home) and the mean boundary of those TCNs (e.g. one
               per training seed), also decoded with its boundary (<e>+bnd<a>, <ref>+<e>bnd<a>)
--clean-tracks runs the tracklet cleaner (mqf.tracks, as make_submission --clean-tracks) after the noise and
before the ball fill and P(dead); --noise overrides the ATD noise model (atd condition), e.g. team_flip_p=0.05.
  <name>       --external name=DIR: P(home) per bundle frame from DIR/<GAME>.npz (pathcrf_bridge align)

Writes harness_results/owner_<tag>.json (every row) and owner_<tag>.md (tables), prints the tables.
Aggregates only, other games only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import decode, features, who
from ..config import EXTERNAL_DIR, HARNESS_DIR, MODELS_DIR
from ..harness.evaluate import score
from . import combine, feats, rules
from .models import OwnerModel

CLIP = 0.02
LAYERS = ("oracle", "smooth50", "dead_v1")
EVIDENCE_CACHE = EXTERNAL_DIR / "cache" / "owner_eval"


def _file_tag(p) -> str:
    if p is None:
        return "none"
    p = Path(p)
    return f"{p.name}:{p.stat().st_size}" if p.exists() else p.name


def load_condition(d, condition: str, seed: int, ball_model, dead_model, ball_model_path, dead_model_path,
                   device: str = "cpu", use_cache: bool = True, log=print, clean_tracks: bool = False,
                   noise_cfg=None) -> dict:
    """Tracking of one bundle under the condition, the filled ball and P(dead) (cached per game x seed).
    noise_cfg: the NoiseConfig for the atd condition (default NoiseConfig(); its seed is replaced by `seed`)."""
    from ..external import evaluate as ev
    from ..harness.noise import NoiseConfig
    ncfg = replace(noise_cfg or NoiseConfig(), seed=seed) if condition == "atd" else None
    spec, trk, gt = ev.load(d, players_drop=(condition == "bar"), hide_dead_ball=True, noise_cfg=ncfg,
                            clean_tracks=clean_tracks)
    if condition == "clean_noball":                          # ball-free: no ball anywhere, nothing to fill from
        from .. import atd
        trk = atd.Tracking(trk.players, trk.player_speed, np.full_like(trk.ball, np.nan), trk.team_of_slot)
        ball_model = None
    extra = []                                               # (no extra -> the keys of the earlier runs)
    if clean_tracks:
        from .. import tracks
        extra.append({"clean": asdict(tracks.CleanConfig())})
    if ncfg is not None and replace(ncfg, seed=0) != NoiseConfig():
        extra.append({"noise": asdict(replace(ncfg, seed=0))})
    key = hashlib.sha1(json.dumps([spec.name, condition, seed, _file_tag(ball_model_path),
                                   _file_tag(dead_model_path), *extra], default=str).encode()).hexdigest()[:12]
    path = EVIDENCE_CACHE / f"{spec.name}_{condition}_s{seed}_{key}.npz"
    if use_cache and path.exists():
        z = np.load(path)
        ball, filled, p_dead = z["ball"], z["filled"], z["p_dead"]
    else:
        ball, filled = trk.ball.astype(np.float64), np.zeros(len(trk.ball), dtype=bool)
        if ball_model is not None:
            from ..ball import impute
            F = impute.FROZEN
            m, cfg = ball_model
            bf = impute.fill(trk.players, trk.team_of_slot, trk.ball, spec.fps, spec.pitch, spec.halves, F["method"],
                             F["tau_s"], m, cfg, interp_method=F["interp_method"], device=device,
                             anchor_s=F["anchor_s"], tau_linear_s=F["tau_linear_s"])
            filled = (bf.source == impute.INTERP) | (bf.source == impute.MODEL)
            ball = bf.ball.astype(np.float64)
        p_dead = np.full(len(ball), np.nan)
        if dead_model is not None:
            from .. import dead
            p_dead = dead.predict(dead_model, trk.players, trk.team_of_slot, spec.fps, spec.pitch, spec.halves)
        if use_cache:
            EVIDENCE_CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, ball=ball, filled=filled, p_dead=p_dead)
    return {"spec": spec, "trk": trk, "gt": gt, "ball": ball, "filled": filled & ~np.isnan(ball[:, 0]),
            "p_dead": p_dead if dead_model is not None else None}


def _inside(halves, n):
    m = np.zeros(n, dtype=bool)
    for _, a, b in halves:
        m[a:b + 1] = True
    return m


def game_methods(G: dict, models: dict, hard_of, hybrid_bases, rule_cfg: rules.RuleConfig, device="cpu",
                 external: dict | None = None, boundary_from=None, bnd_alpha=(0.5, 0.9), ref=None,
                 ensembles: dict | None = None):
    """{method: P(home)} on one game (layer-independent ones), {method: (P(boundary), alpha)} for the
    boundary-decoded ones, the rule result, the smooth50 alive mask and the feature views."""
    spec, trk = G["spec"], G["trk"]
    n, fps, halves = spec.n_frames, spec.fps, spec.halves
    inside = _inside(halves, n)
    views = []
    for k in (0, 1):
        team = trk.team_of_slot if k == 0 else 1 - trk.team_of_slot
        X = feats.features(trk.players, team, G["ball"], fps, spec.pitch, halves, "full_rules_heading", rule_cfg,
                           ball_filled=G["filled"])                 # every set's columns: models pick theirs by name
        S, sc = feats.seq_inputs(trk.players, team, G["ball"], fps, spec.pitch, halves, X_full=X, columns="all") \
            if any(m.kind in ("gru", "tcn") for m in models.values()) else (None, None)
        views.append({"X": X, "S": S, "seq_columns": sc})
    X0 = views[0]["X"]
    P = {"nearest": np.where(inside, who.v0_p_home(X0), np.nan),
         "hold": np.where(inside, who.hold_p_home(X0), np.nan),
         "blend": np.where(inside, combine.blend(X0["z_couple"], X0["z_chain"]), np.nan)}
    sm_lab, sm_alive = combine.smooth50(trk.players, trk.team_of_slot, trk.ball, fps, halves, n)
    P["smooth50"] = np.where(inside, np.where(sm_lab == "H", 1 - CLIP, np.where(sm_lab == "A", CLIP, 0.5)), np.nan)
    rr = rules.owner(trk.players, trk.team_of_slot, G["ball"], fps, halves, rule_cfg, ball_filled=G["filled"])
    P["rule"] = rr["p_home"]
    P["rule_hard"] = combine.owner_to_p(rr["owner"], inside, CLIP)
    PB = {}
    for name, m in models.items():
        if feats.has_rules(m.feature_set) and m.rule_cfg != rule_cfg:
            raise SystemExit(f"{name}: trained with other rule settings than --rule-cfg ({m.rule_cfg})")
        t0 = time.time()
        P[name], pb = m.p_home_and_boundary(views, halves, n, fps, device)
        if pb is not None:
            PB[name] = pb
        if name in hard_of:
            P[f"{name}_hard"] = combine.hard(P[name], CLIP)
        G.setdefault("timing", {})[name] = round(time.time() - t0, 1)
    for name, folder in (external or {}).items():
        f = Path(folder) / f"{spec.name}.npz"
        if f.exists():
            z = np.load(f)
            if len(z["p_home"]) != n:
                raise SystemExit(f"--external {name}: {f} has {len(z['p_home'])} frames, the bundle {n}")
            P[name] = np.where(inside, z["p_home"].astype(np.float64), np.nan)
    for b in hybrid_bases:
        if b not in P:
            raise SystemExit(f"--hybrid-base {b}: no such model (--models)")
        P[f"{b}+prod"] = combine.product(P[b], rr["p_home"], 1.0)
        P[f"{b}+prod0.5"] = combine.product(P[b], rr["p_home"], 0.5)
        P[f"{b}+override"] = combine.override(P[b], rr)
    C = {}
    if isinstance(boundary_from, str):
        boundary_from = [boundary_from]
    srcs = list(boundary_from) if boundary_from else ([next(iter(PB))] if PB else [])
    for src in srcs:
        if src not in PB:
            raise SystemExit(f"--boundary-from {src}: not a TCN with a boundary head")
    sources = [(src, PB[src]) for src in srcs]
    for e, members in (ensembles or {}).items():            # TCNs averaged (e.g. one per training seed)
        bad = [m for m in members if m not in PB]
        if bad:
            raise SystemExit(f"--ensemble {e}: {bad} are not TCNs with a boundary head (--models)")
        P[e] = np.mean(np.stack([P[m] for m in members]), axis=0)
        sources.append((e, np.mean(np.stack([PB[m] for m in members]), axis=0)))
    for src, pb in sources:
        for b in dict.fromkeys([src, ref, *hybrid_bases]):
            if b in P:
                for al in bnd_alpha:
                    tag = f"{b}+bnd{al:g}" if b == src else f"{b}+{src}bnd{al:g}"
                    P[tag] = P[b]
                    C[tag] = (pb, float(al))
    return P, C, rr, sm_alive, views


def run_game(G, P, rr, sm_alive, layers, decoders, params: decode.Params, hybrid_bases, rule_cfg, meta: dict,
             switch_cost: float, C: dict | None = None):
    spec, trk, gt = G["spec"], G["trk"], G["gt"]
    n, fps, halves = spec.n_frames, spec.fps, spec.halves
    inside = _inside(halves, n)
    alive_true = np.isin(gt, ["H", "A"]) & inside
    tracked = ~np.isnan(trk.ball[:, 0])
    rows, frows = [], []
    # per-frame quality on the true alive frames (layer-independent)
    y = (gt == "H")
    C = C or {}
    for name, p in P.items():
        if name in C:                                     # same p as its base: nothing new per frame
            continue
        for part, m in (("all", alive_true), ("seen", alive_true & tracked), ("missing", alive_true & ~tracked)):
            frows.append({**meta, "method": name, "ball": part, **combine.frame_metrics(y[m], p[m], CLIP)})
    feat = None
    for layer in layers:
        if layer == "dead_v1":
            if G["p_dead"] is None:
                raise SystemExit("layer dead_v1 needs --dead-model")
            feat = features.build(atd_tracking(trk, G["ball"]), spec.pitch, fps, ball_filled=G["filled"], p_dead=G["p_dead"])
            alive = (G["p_dead"] < 0.5) & inside
        else:
            alive = alive_true if layer == "oracle" else (sm_alive & inside)
        layer_P = dict(P)
        rs = rules.owner(trk.players, trk.team_of_slot, G["ball"], fps, halves, rule_cfg, ball_filled=G["filled"],
                         alive=alive)
        layer_P["rule_seg"] = rs["p_home"]
        labels_of = {}
        for name, p in layer_P.items():
            mod = C.get(name)
            for dec in decoders:
                if dec == "raw":
                    if mod is not None:                       # a boundary only changes the decoder
                        continue
                    lab = combine.decode_raw(p, alive, halves, n)
                elif layer == "dead_v1":
                    if name == "nearest":                     # v0's own who evidence (#35's base)
                        lab = _hmm3_v0(feat, halves, n, params)
                    else:
                        sw = None if mod is None else combine.switch_costs(mod[0], params.switch_cost, mod[1])
                        lab = combine.decode_hmm3(feat, p, halves, n, params, sw)
                else:
                    sc = switch_cost if mod is None else combine.switch_costs(mod[0], switch_cost, mod[1])
                    lab = combine.decode_given_alive(p, alive, halves, n, sc, params.w_who_model, params.who_clip)
                if dec == "hmm":
                    labels_of[name] = lab
                rows.append({**meta, "layer": layer, "method": name, "decoder": dec, **score(gt, lab, fps)})
        for b in hybrid_bases:
            lab = combine.veto(labels_of[b], rr, tracked, fps)
            rows.append({**meta, "layer": layer, "method": f"{b}+veto", "decoder": "hmm", **score(gt, lab, fps)})
    return rows, frows


def atd_tracking(trk, ball):
    from .. import atd
    return atd.Tracking(trk.players, trk.player_speed, np.asarray(ball, dtype=np.float32), trk.team_of_slot)


def _hmm3_v0(feat, halves, n, params):
    """v0's emissions (no learned who) with dead_v1 -- #35's base."""
    ll = decode.emissions(feat, params, None)
    lab = np.full(n, "X", dtype="<U1")
    from ..config import LABELS
    for _, a, b in halves:
        lab[a:b + 1] = np.array(LABELS)[combine.viterbi3(ll[a:b + 1], params)]
    return lab


# ------------------------------------------------------------------------------------ report

def _mean_by_source(df, col):
    per = df.groupby("source")[col].mean()
    return float(per.mean()) if len(per) else float("nan")


def summarise(rows: list[dict], frows: list[dict], ref: str | None) -> str:
    df = pd.DataFrame(rows)
    fdf = pd.DataFrame(frows)
    out = []
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    # A. per-frame
    if len(fdf):
        g = fdf.groupby(["method", "ball", "source"])[["acc", "auc", "logloss", "brier", "ece"]].mean()
        g = g.groupby(["method", "ball"]).mean().unstack("ball")
        g.columns = [f"{m}_{b}" for m, b in g.columns]
        keep = [c for c in ("acc_all", "acc_seen", "acc_missing", "auc_all", "logloss_all", "brier_all", "ece_all") if c in g]
        out.append("## Per alive frame (mean over games per source, then over sources)\n\n```\n"
                   + g[keep].sort_values("acc_all", ascending=False).round(3).to_string() + "\n```\n")
    # B + C. per layer
    df["spell_ratio"] = df["n_pred_spells"] / df["n_gt_spells"].clip(lower=1)
    df["m0_3"] = df["by_length"].map(lambda b: b.get("0-3s", {}).get("match_rate"))
    df["m3_10"] = df["by_length"].map(lambda b: b.get("3-10s", {}).get("match_rate"))
    for layer, dl in df.groupby("layer", sort=False):
        tab = []
        for (method, dec), g in dl.groupby(["method", "decoder"]):
            r = {"method": method, "decoder": dec, "spellF1": _mean_by_source(g, "spellF1"),
                 "macroF1": _mean_by_source(g, "macroF1"), "team_acc": _mean_by_source(g, "team_acc_both_alive"),
                 "spells_pred/true": _mean_by_source(g, "spell_ratio"), "match_0-3s": _mean_by_source(g, "m0_3"),
                 "match_3-10s": _mean_by_source(g, "m3_10"), "n": len(g)}
            for s, gs in g.groupby("source"):
                r[f"spellF1_{s}"] = float(gs.spellF1.mean())
            if ref is not None:
                base = dl[(dl.method == ref) & (dl.decoder == dec)].set_index(["game", "seed"])["spellF1"]
                cur = g.set_index(["game", "seed"])["spellF1"]
                both = cur.index.intersection(base.index)
                if len(both) and method != ref:
                    dlt = (cur[both] - base[both]).to_numpy()
                    r["d_vs_ref"] = float(dlt.mean())
                    r["se"] = float(dlt.std(ddof=1) / np.sqrt(len(dlt))) if len(dlt) > 1 else float("nan")
                    r["wins"] = f"{int((dlt > 0).sum())}/{len(dlt)}"
            tab.append(r)
        t = pd.DataFrame(tab).sort_values("spellF1", ascending=False)
        out.append(f"## Layer `{layer}` -- spellF1 (mean per source, then over sources); d_vs_ref paired vs `{ref}`\n\n```\n"
                   + t.round(3).to_string(index=False) + "\n```\n")
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
    ap.add_argument("--condition", default="bar", choices=["bar", "atd", "clean", "clean_noball"])
    ap.add_argument("--clean-tracks", action="store_true", help="tracklet cleaner after the noise (as the submission)")
    ap.add_argument("--noise", nargs="*", default=[], metavar="FIELD=VALUE", help="NoiseConfig overrides (atd)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0], help="noise seeds (atd); bar uses seed 0")
    ap.add_argument("--layers", nargs="*", default=["oracle", "dead_v1"], choices=LAYERS)
    ap.add_argument("--decoders", nargs="*", default=["hmm"], choices=["hmm", "raw"])
    ap.add_argument("--models", nargs="*", default=[f"who_v1={MODELS_DIR / 'who_v1.joblib'}"], metavar="NAME=PATH")
    ap.add_argument("--hard", nargs="*", help="models that also get a _hard variant (default: all)")
    ap.add_argument("--hybrid-base", nargs="*", default=[], help="models the rule hybrids are built on")
    ap.add_argument("--boundary-from", nargs="*", default=None,
                    help="TCN model(s) whose boundary head modulates the switch cost (default: the first TCN)")
    ap.add_argument("--ensemble", nargs="*", default=[], metavar="NAME=TCN1,TCN2",
                    help="mean P(home) and boundary of these TCNs as method NAME")
    ap.add_argument("--bnd-alpha", nargs="*", type=float, default=[0.5, 0.9])
    ap.add_argument("--external", nargs="*", default=[], metavar="NAME=DIR", help="P(home) files, e.g. pathcrf=external/pathcrf")
    ap.add_argument("--ref", default="who_v1", help="paired deltas against this method")
    ap.add_argument("--dead-model", type=Path, default=MODELS_DIR / "dead_v1.joblib")
    ap.add_argument("--no-ball-fill", action="store_true")
    ap.add_argument("--params", nargs="*", default=["dead_who=max"], metavar="FIELD=VALUE",
                    help="decode.Params overrides (dead_v1 layer; w_who_model / who_clip also for the 2-state decoder)")
    ap.add_argument("--switch-cost", type=float, help="2-state decoder switch cost (default: Params.switch_cost)")
    ap.add_argument("--rule-cfg", nargs="*", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--tag", default="owner_dev")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    log = lambda m: print(m, flush=True)
    from ..external import bundle, splits
    from ..external.evaluate import parse_grid
    over = parse_grid([f"{kv.split('=')[0]}={kv.split('=', 1)[1]}" for kv in a.params])[0]
    params = replace(decode.Params(), **over)
    switch_cost = a.switch_cost if a.switch_cost is not None else params.switch_cost
    rule_cfg = rules.parse_rule_cfg(a.rule_cfg)
    paths = parse_models(a.models)
    models = {k: OwnerModel.load(p) for k, p in paths.items()}
    hard_of = set(models) if a.hard is None else set(a.hard)
    device = "cpu"
    if any(m.kind != "gbm" for m in models.values()) or not a.no_ball_fill:
        try:
            import torch
            device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
        except ImportError:
            device = "cpu"
    ball_model = None
    ball_path = None
    if not a.no_ball_fill:
        from ..ball import impute
        ball_path = MODELS_DIR / impute.FROZEN["model"]
        ball_model = impute.load_model(ball_path, device)
    dead_model = None
    if "dead_v1" in a.layers:
        from .. import dead
        dead_model = dead.load_model(a.dead_model)
    seeds = a.seeds if a.condition == "atd" else [0]
    from ..external.evaluate import parse_noise
    noise_cfg = parse_noise(a.noise) if a.noise else None
    ensembles = {}
    for it in a.ensemble:
        e, eq, members = it.partition("=")
        if not eq or not members or e in models:
            raise SystemExit(f"--ensemble: expected NAME=TCN1,TCN2 with a new NAME, got {it!r}")
        ensembles[e] = members.split(",")
    rows, frows, timing = [], [], {}
    t_all = time.time()
    for source in a.source:
        names = a.matches or splits.games(a.split, sources=[source])
        dirs = bundle.find(source, names)
        if a.max_games:
            dirs = dirs[:a.max_games]
        if not dirs:
            log(f"[{source}] no bundles for the {a.split} games")
            continue
        for d in dirs:
            for seed in seeds:
                t0 = time.time()
                G = load_condition(d, a.condition, seed, ball_model, dead_model, ball_path,
                                   a.dead_model if dead_model is not None else None, device, not a.no_cache, log,
                                   clean_tracks=a.clean_tracks, noise_cfg=noise_cfg)
                P, C, rr, sm_alive, _ = game_methods(G, models, hard_of, a.hybrid_base, rule_cfg, device,
                                                     parse_models(a.external), a.boundary_from, a.bnd_alpha, a.ref,
                                                     ensembles)
                meta = {"game": G["spec"].name, "source": source, "seed": seed, "condition": a.condition}
                r, fr = run_game(G, P, rr, sm_alive, a.layers, a.decoders, params, a.hybrid_base, rule_cfg, meta,
                                 switch_cost, C)
                rows += r
                frows += fr
                timing[f"{G['spec'].name}_s{seed}"] = {"total_s": round(time.time() - t0, 1), **G.get("timing", {})}
                best = {}
                for x in r:
                    if x["decoder"] == "hmm":
                        best.setdefault(x["layer"], {})[x["method"]] = x["spellF1"]
                line = "  ".join(f"{L}: " + " ".join(f"{m} {v:.3f}" for m, v in sorted(b.items(), key=lambda kv: -kv[1])[:4])
                                 for L, b in best.items())
                log(f"{G['spec'].name:<14} s{seed}  {time.time() - t0:5.0f} s  {line}  | rule {rules.summary(rr, G['spec'].fps)}")
    if not rows:
        return 1
    ref = a.ref if any(r["method"] == a.ref for r in rows) else None
    text = summarise(rows, frows, ref)
    print("\n" + text)
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {"tag": a.tag, "condition": a.condition, "seeds": seeds, "layers": a.layers, "decoders": a.decoders,
           "models": {k: str(v) for k, v in paths.items()}, "hybrid_base": a.hybrid_base, "ref": ref,
           "external": a.external, "boundary_from": a.boundary_from, "bnd_alpha": a.bnd_alpha,
           "ensemble": a.ensemble, "clean_tracks": a.clean_tracks, "noise": a.noise,
           "params": asdict(params), "switch_cost_2state": switch_cost, "rule_cfg": asdict(rule_cfg),
           "ball_fill": not a.no_ball_fill, "dead_model": str(a.dead_model) if dead_model is not None else None,
           "split": a.split, "sources": a.source, "minutes": round((time.time() - t_all) / 60, 1)}
    (HARNESS_DIR / f"owner_{a.tag}.json").write_text(json.dumps({"config": cfg, "runs": rows, "frames": frows,
                                                                  "timing": timing}, indent=1, default=float))
    (HARNESS_DIR / f"owner_{a.tag}.md").write_text(f"# owner experiment `{a.tag}`\n\n```\n{json.dumps(cfg, indent=1)}\n```\n\n" + text)
    log(f"wrote {HARNESS_DIR / f'owner_{a.tag}.json'} and .md  ({cfg['minutes']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
