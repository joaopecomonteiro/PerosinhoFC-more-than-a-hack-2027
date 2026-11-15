"""Every dead-ball experiment of 16 Nov 2026, in order, with one command (other games only).

    python -m mqf.alive.suite                     # everything: sens, tables, train, eval, tags, report (~4-6 h, GPU)
    python -m mqf.alive.suite --resume            # keep trained models and finished eval runs (after a crash)
    python -m mqf.alive.suite --stage sens        # or tables / train / eval / tags / report
    python -m mqf.alive.suite --quick             # smoke run: tiny models, 1 val game per source (~15 min)
    python -m mqf.alive.suite --skip asf atd_flip # leave blocks out
    python -m mqf.alive.suite --jobs 3            # games in parallel for tables and eval (each worker ~3-4 GB RAM)

Stages (models in models/alive_*, results in harness_results/alive_<tag>_*):
  sens     sensitivity.py on the 11 val games (bar, ball exact, cleaned tracks): what each kind of
           alive-layer error costs on the TRUE mask -- read it first, it ranks the failure modes
  tables   the 5 Hz feature tables of the 63 train games (IDSSE 5 + Metrica 2 + PFF 56), cached
  train    gbm   v1_impff (dead_v1's 36 columns, our views: replica), players_impff, formation_impff,
                 full_impff, rules_impff (full + rule columns), full_im (IDSSE + Metrica only: source ablation)
           mlp   full_impff                                      (control: NN vs trees on the same table)
           tcn   full_impff_s0/s1/s2, full_im_s0                 (ASRF-style TCN + boundary head)
           asf   full_impff_s0/s1/s2                             (ASFormer-lite: + windowed self-attention)
  eval     experiment.py on the 11 val games, WHO fixed (who_v1), tracks cleaned, ball filled:
             bar       dropouts, ball modes exact / fuzzy / random
             atd       ATD noise seeds 0 1, ball modes fuzzy / exact
             atd_flip  + wrong-team pickups (team_flip_p 0.05, keepers 0.5), seed 0, ball fuzzy
           methods v0, dead_v1, rules, every model, ensembles tcn / asf (3 seeds) and mix (gbm_full + tcn),
           hybrids (+prod, +override) on gbm_full and tcn; decoder variants on the main candidates:
           hmm, start, min2, snap1, veto, bnd0.5, start+min2 (combine.py)
  tags     the Smart Tagging prior on tag-like windows simulated from the labels, recall 0.3 / 0.6 / 1.0
           (2 false windows / 90'), bar exact + fuzzy -- sets its weight before BAR-ATH ever sees it
  report   harness_results/alive_<tag>_report.md: the questions below, paired vs dead_v1 (hmm)

Questions (spellF1, paired per game x seed, mean (se) wins; the #38 rule: mean > 2 se and more than
half the runs in EVERY condition x ball mode):
  Q0 sanity     gbm_v1_impff vs dead_v1 (same 36 columns, our training views)
  Q1 features   players -> formation -> full -> full + rules (each vs the previous)
  Q2 sources    gbm_full_im vs gbm_full_impff
  Q3 models     mlp / tcn / asf / mix vs gbm_full; tcn ensemble vs one seed
  Q4 rules      rules alone; +prod / +override vs their base
  Q5 decoder    each variant vs hmm, on dead_v1 and the candidates
  Q6 tags       tags vs hmm by simulated recall
Other games only: nothing here reads the competition match.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import HARNESS_DIR, MODELS_DIR
from . import data, experiment, rules, sensitivity, train

IMPFF = ["idsse", "metrica", "pff"]
IM = ["idsse", "metrica"]
SEEDS = (0, 1, 2)
REF = ("dead_v1", "hmm")
VARIANTS = ["hmm", "start", "min2", "snap1", "veto", "bnd0.5", "start+min2"]
CONDITIONS = {"bar": ["--condition", "bar", "--ball-modes", "exact", "fuzzy", "random"],
              "atd": ["--condition", "atd", "--seeds", "0", "1", "--ball-modes", "fuzzy", "exact"],
              "atd_flip": ["--condition", "atd", "--seeds", "0", "--noise", "team_flip_p=0.05", "gk_team_flip_p=0.5",
                           "--ball-modes", "fuzzy"]}
TAG_RECALLS = (0.3, 0.6, 1.0)
BLOCKS = ("sens", "gbm", "mlp", "tcn", "asf", "bar", "atd", "atd_flip", "tags")


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def model_plan(quick: bool):
    """[(kind, tag, feature_set, sources, seed)] -- the models of the train stage."""
    q = "q_" if quick else ""
    plan = [("gbm", f"{q}v1_impff", "v1", IMPFF, 0), ("gbm", f"{q}players_impff", "players", IMPFF, 0),
            ("gbm", f"{q}formation_impff", "formation", IMPFF, 0), ("gbm", f"{q}full_impff", "full", IMPFF, 0),
            ("gbm", f"{q}rules_impff", "full_rules", IMPFF, 0), ("gbm", f"{q}full_im", "full", IM, 0),
            ("mlp", f"{q}full_impff", "full", IMPFF, 0)]
    seeds = (0, 1) if quick else SEEDS
    plan += [("tcn", f"{q}full_impff_s{s}", "full", IMPFF, s) for s in seeds]
    plan += [("tcn", f"{q}full_im_s0", "full", IM, 0)]
    plan += [("asf", f"{q}full_impff_s{s}", "full", IMPFF, s) for s in seeds]
    return plan


def method_name(kind: str, tag: str) -> str:
    return f"{kind}_{tag[2:] if tag.startswith('q_') else tag}"


def _build_one(args):
    d, seed = args
    from ..ball import data as bdata
    g = bdata.load_game(Path(d))
    data.game_tables(g, seed)
    return g.name


def stage_tables(a, log):
    from ..external import bundle, splits
    names = splits.games("train", sources=IMPFF)
    if a.quick:
        names = [n for s in IMPFF for n in splits.games("train", sources=[s])[:1]]
    dirs = [str(d) for d in bundle.find(names=names)]
    log(f"tables: {len(dirs)} train games, {a.jobs} worker(s)")
    t0 = time.time()
    if a.jobs > 1:
        import multiprocessing
        with ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            for k, name in enumerate(ex.map(_build_one, [(d, 0) for d in dirs])):
                log(f"  {name} ({k + 1}/{len(dirs)}, {time.time() - t0:.0f} s)")
    else:
        for k, d in enumerate(dirs):
            log(f"  {_build_one((d, 0))} ({k + 1}/{len(dirs)}, {time.time() - t0:.0f} s)")


def stage_train(a, log):
    plan = [p for p in model_plan(a.quick) if p[0] not in a.skip]
    todo = [p for p in plan if not (a.resume and train.model_path(p[0], p[1]).exists())]
    if not todo:
        log("train: every model exists (--resume)")
        return
    names = None
    if a.quick:
        from ..external import splits
        names = {n for s in IMPFF for n in splits.games("train", sources=[s])[:1]}
    log(f"train: {len(todo)} models; loading the tables of the train games ...")
    T = data.load_tables("train", IMPFF, 0, rules.RuleConfig(), log=log)
    if names is not None:
        T = [t for t in T if t[0] in names]
    for kind, tag, fs, srcs, seed in todo:
        sub = [t for t in T if t[1] in srcs]
        log(f"train {kind} {tag} ({fs}, {'+'.join(srcs)}, {len(sub)} games, seed {seed})")
        kw = {"iters": 30} if a.quick else {}
        ep = 1 if a.quick else None
        train.fit(kind, sub, tag, fs, rules.RuleConfig(), epochs=ep, seed=seed, device=a.device, sources=srcs,
                  log=log, table_seed=0, **kw)


def available_models(quick: bool, skip=()):
    out = {}
    for kind, tag, *_ in model_plan(quick):
        p = train.model_path(kind, tag)
        if p.exists() and kind not in skip:
            out[method_name(kind, tag)] = p
    return out


def eval_args(a, models: dict, extra_variants=None):
    m = models
    ens = []
    tcns = [k for k in m if k.startswith("tcn_full_impff_s")]
    asfs = [k for k in m if k.startswith("asf_full_impff_s")]
    if len(tcns) > 1:
        ens.append("tcn=" + ",".join(tcns))
    if len(asfs) > 1:
        ens.append("asf=" + ",".join(asfs))
    if "gbm_full_impff" in m and tcns:
        ens.append("mix=gbm_full_impff," + ("tcn" if len(tcns) > 1 else tcns[0]))      # half trees, half TCN
    seq = "tcn" if len(tcns) > 1 else (tcns[0] if tcns else None)
    bases = [b for b in ("gbm_full_impff", seq) if b and (b in m or b == "tcn")]
    vm = ["gbm_full_impff", "gbm_rules_impff", "tcn", "asf", "mix", *(tcns[:1] if len(tcns) == 1 else [])]
    out = ["--split", "val", "--source", *IMPFF, "--clean-tracks", "--device", a.device, "--jobs", str(a.jobs),
           "--models", *[f"{k}={v}" for k, v in m.items()], "--variants", *(extra_variants or VARIANTS),
           "--variant-methods", *vm]
    if ens:
        out += ["--ensemble", *ens]
    if bases:
        out += ["--hybrid-base", *bases]
    if seq:
        out += ["--bnd-from", seq]
    if a.quick:
        out += ["--max-games", "1"]
    if a.no_ball_fill:
        out += ["--no-ball-fill"]
    return out


def _done(tag: str) -> bool:
    return (HARNESS_DIR / f"alive_{tag}.json").exists()


def stage_sens(a, log):
    tag = f"{a.tag}_sens"
    if a.resume and (HARNESS_DIR / f"alive_sens_{tag}.json").exists():
        log("sens: done (--resume)")
        return
    args = ["--clean-tracks", "--device", a.device, "--tag", tag] + (["--max-games", "1"] if a.quick else []) \
        + (["--no-ball-fill"] if a.no_ball_fill else [])
    log("sens: " + " ".join(args))
    sensitivity.main(args)


def stage_eval(a, log):
    models = available_models(a.quick, a.skip)
    for cname, cargs in CONDITIONS.items():
        if cname in a.skip or (a.quick and cname != "bar"):
            continue
        tag = f"{a.tag}_{cname}"
        if a.resume and _done(tag):
            log(f"eval {cname}: done (--resume)")
            continue
        args = eval_args(a, models) + list(cargs) + ["--tag", tag]
        if a.quick:
            args = [x for x in args if x != "random"]
        log(f"eval {cname}: {len(models)} models")
        experiment.main(args)


def stage_tags(a, log):
    if "tags" in a.skip:
        return
    models = {k: v for k, v in available_models(a.quick, a.skip).items()
              if k in ("gbm_full_impff", "gbm_rules_impff") or k.startswith("tcn_full_impff_s")}
    for r in TAG_RECALLS if not a.quick else TAG_RECALLS[1:2]:
        tag = f"{a.tag}_tags_r{r:g}"
        if a.resume and _done(tag):
            log(f"tags r={r:g}: done (--resume)")
            continue
        args = eval_args(a, models, ["hmm", "tags", "start+min2", "start+min2+tags"]) + \
            ["--condition", "bar", "--ball-modes", "exact", "fuzzy", "--tag-recall", str(r), "--tag-fp", "2",
             "--tag-w", "1.0", "--tag", tag]
        log(f"tags r={r:g}")
        experiment.main(args)


# ------------------------------------------------------------------------------------ report

def paired(df: pd.DataFrame, a: tuple, b: tuple):
    x = df[(df.method == a[0]) & (df.variant == a[1])].set_index(["game", "seed"])["spellF1"]
    y = df[(df.method == b[0]) & (df.variant == b[1])].set_index(["game", "seed"])["spellF1"]
    idx = x.index.intersection(y.index)
    if not len(idx):
        return None
    d = (x[idx] - y[idx]).to_numpy()
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"mean": float(d.mean()), "se": se, "wins": int((d > 0).sum()), "n": len(d)}


def _fmt(r):
    if r is None:
        return "—"
    se = "n/a" if not np.isfinite(r["se"]) else f"{r['se']:.3f}"
    return f"{r['mean']:+.3f} ({se}) {r['wins']}/{r['n']}"


def _load_groups(a):
    groups = {}
    for cname in CONDITIONS:
        p = HARNESS_DIR / f"alive_{a.tag}_{cname}.json"
        if p.exists():
            df = pd.DataFrame(json.loads(p.read_text())["runs"])
            for mode, g in df.groupby("ball_mode"):
                groups[f"{cname}/{mode}"] = g
    return groups


def questions(groups) -> list[tuple[str, tuple, tuple]]:
    have = set()
    for g in groups.values():
        have |= set(zip(g.method, g.variant))
    q = []

    def add(label, a, b):
        if a in have and b in have:
            q.append((label, a, b))
    add("Q0 replica: gbm_v1_impff vs dead_v1", ("gbm_v1_impff", "hmm"), REF)
    chain = ["gbm_v1_impff", "gbm_players_impff", "gbm_formation_impff", "gbm_full_impff", "gbm_rules_impff"]
    for x, y in zip(chain[1:], chain[:-1]):
        add(f"Q1 features: {x} vs {y}", (x, "hmm"), (y, "hmm"))
    add("Q2 sources: gbm_full_im vs gbm_full_impff", ("gbm_full_im", "hmm"), ("gbm_full_impff", "hmm"))
    for m in ("mlp_full_impff", "tcn", "asf", "mix", "tcn_full_impff_s0"):
        add(f"Q3 models: {m} vs gbm_full_impff", (m, "hmm"), ("gbm_full_impff", "hmm"))
    add("Q3 models: asf vs tcn", ("asf", "hmm"), ("tcn", "hmm"))
    add("Q3 models: tcn (3 seeds) vs tcn_full_impff_s0", ("tcn", "hmm"), ("tcn_full_impff_s0", "hmm"))
    add("Q3 sources (seq): tcn_full_im_s0 vs tcn_full_impff_s0", ("tcn_full_im_s0", "hmm"), ("tcn_full_impff_s0", "hmm"))
    add("Q4 rules alone vs dead_v1", ("rules", "hmm"), REF)
    for b in ("gbm_full_impff", "tcn"):
        for h in ("+prod", "+override"):
            add(f"Q4 hybrid: {b}{h} vs {b}", (b + h, "hmm"), (b, "hmm"))
    for b in ("dead_v1", "gbm_full_impff", "gbm_rules_impff", "tcn", "asf", "mix"):
        for v in VARIANTS[1:]:
            add(f"Q5 decoder: {b} [{v}] vs [hmm]", (b, v), (b, "hmm"))
    return q


def stage_report(a, log):
    groups = _load_groups(a)
    if not groups:
        log("report: no eval results yet")
        return
    cols = list(groups)
    out = [f"# Dead-ball experiments `{a.tag}` (other games only)\n",
           "Paired Δ spellF1 = a − b per game × noise seed, WHO fixed (who_v1), tracks cleaned, ball filled for WHO; "
           "cells: mean (se) wins/n. Rule (#38): mean > 2 se and more than half the runs in EVERY column.\n"]
    sens = HARNESS_DIR / f"alive_sens_{a.tag}_sens.md"
    if sens.exists():
        out.append(sens.read_text(encoding="utf-8"))
    # level table
    out.append("\n## spellF1 by method (hmm decoder; mean per source, then over sources)\n")
    out.append("| method | " + " | ".join(cols) + " |")
    out.append("|---|" + "---|" * len(cols))
    methods = sorted({m for g in groups.values() for m in g.method.unique()})
    for m in methods:
        cells = []
        for c in cols:
            g = groups[c]
            g = g[(g.method == m) & (g.variant.isin(["hmm", "-"]))]
            cells.append(f"{g.groupby('source').spellF1.mean().mean():.3f}" if len(g) else "—")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    # questions
    out.append("\n## Questions\n")
    out.append("| question | " + " | ".join(cols) + " |")
    out.append("|---|" + "---|" * len(cols))
    for label, x, y in questions(groups):
        out.append(f"| {label} | " + " | ".join(_fmt(paired(groups[c], x, y)) for c in cols) + " |")
    # verdict: every (method, variant) vs dead_v1 hmm
    cand = sorted({(m, v) for g in groups.values() for m, v in zip(g.method, g.variant)} - {REF, ("oracle", "-")})
    verdict = []
    for mv in cand:
        rs = {c: paired(groups[c], mv, REF) for c in cols}
        if any(r is None for r in rs.values()):
            continue
        ok = all(np.isfinite(r["se"]) and r["mean"] > 2 * r["se"] and 2 * r["wins"] > r["n"] for r in rs.values())
        verdict.append((ok, min(r["mean"] for r in rs.values()), float(np.mean([r["mean"] for r in rs.values()])), mv, rs))
    verdict.sort(key=lambda v: (not v[0], -v[1]))
    out.append("\n## Verdict vs dead_v1 [hmm] -- every column must pass\n")
    out.append("| | method [variant] | worst column | mean of columns | " + " | ".join(cols) + " |")
    out.append("|---|---|---|---|" + "---|" * len(cols))
    for ok, worst, mean, mv, rs in verdict[:40]:
        out.append(f"| {'PASS' if ok else 'fail'} | {mv[0]} [{mv[1]}] | {worst:+.3f} | {mean:+.3f} | "
                   + " | ".join(_fmt(rs[c]) for c in cols) + " |")
    # tags
    tag_lines = []
    for r in TAG_RECALLS:
        p = HARNESS_DIR / f"alive_{a.tag}_tags_r{r:g}.json"
        if not p.exists():
            continue
        df = pd.DataFrame(json.loads(p.read_text())["runs"])
        for mode, g in df.groupby("ball_mode"):
            for m in sorted(set(g.method) - {"oracle"}):
                for base, v in (("hmm", "tags"), ("start+min2", "start+min2+tags")):
                    rr = paired(g, (m, v), (m, base))
                    if rr:
                        tag_lines.append(f"| {r:g} | {mode} | {m} | [{v}] vs [{base}] | {_fmt(rr)} |")
    if tag_lines:
        out.append("\n## Q6 Smart Tagging prior (simulated tag-like windows, 2 false / 90', weight 1)\n")
        out.append("| recall | ball | method | comparison | Δ spellF1 |\n|---|---|---|---|---|")
        out += tag_lines
    passing = [v for v in verdict if v[0]]
    out.append("\n" + (f"Best passing: **{passing[0][3][0]} [{passing[0][3][1]}]** (worst column {passing[0][1]:+.3f}). "
                       "Before adopting it: make_submission --alive-model ... on BAR-ATH (aggregates only: dead share, "
                       "possessions; mqf.compare vs the current file), --reproduce byte-identical, an EXPERIMENTS entry."
                       if passing else "Nothing passes in every column: keep dead_v1 (or adopt only a decoder variant "
                       "that passes on dead_v1 itself)."))
    text = "\n".join(out) + "\n"
    (HARNESS_DIR / f"alive_{a.tag}_report.md").write_text(text, encoding="utf-8")
    (HARNESS_DIR / f"alive_{a.tag}_report.json").write_text(json.dumps(
        {"passing": [list(v[3]) for v in passing],
         "verdict": [{"method": v[3][0], "variant": v[3][1], "pass": v[0], "worst": v[1], "mean": v[2],
                      "cols": v[4]} for v in verdict]}, indent=1, default=float))
    print("\n" + text)
    log(f"wrote {HARNESS_DIR / f'alive_{a.tag}_report.md'}")


STAGES = {"sens": stage_sens, "tables": stage_tables, "train": stage_train, "eval": stage_eval, "tags": stage_tags,
          "report": stage_report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--skip", nargs="*", default=[], choices=BLOCKS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-ball-fill", action="store_true", help="(tests / machines without torch)")
    ap.add_argument("--tag", default=None, help="default: alive16 (alive16q with --quick)")
    a = ap.parse_args(argv)
    a.tag = a.tag or ("alive16q" if a.quick else "alive16")
    log = Tee(HARNESS_DIR / f"alive_{a.tag}.log")
    log(f"suite {a.tag}: stages {a.stage}, skip {a.skip}, resume {a.resume}, " + " ".join(sys.argv[1:]))
    for st in a.stage:
        if st == "sens" and "sens" in a.skip:
            continue
        t0 = time.time()
        STAGES[st](a, log)
        log(f"stage {st} done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
