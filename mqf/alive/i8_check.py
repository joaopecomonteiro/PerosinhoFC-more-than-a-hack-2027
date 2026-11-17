"""Do Italo's v8 inputs help OUR models? Train on full + i_ (feats.full_i8), compare under the FINAL possession.

    python -m mqf.alive.i8_check --jobs 3              # tables, train, eval, report (~3 h on the GPU)
    python -m mqf.alive.i8_check --jobs 3 --resume     # after a crash: keeps tables, models, finished eval runs
    python -m mqf.alive.i8_check --stage eval report   # only re-score
    python -m mqf.alive.i8_check --quick               # smoke run

Why: in final16 (#45) mean(asf, v8) beat asf in every column, also with a label-free ball loss -- v8 knows
something asf does not. Its inputs (keeper holding, out memory closed by a restart, carried ball, ...)
go into our table as i_ columns; our views keep the ball's absence label-independent (data.py), so they
help only through what they say beyond absence.

Stages
  tables  the 5 Hz tables of the 63 train games again (the table now has the i_ group; cached)
  train   gbm full_i8_impff (quick signal next to gbm_full_impff); asf full_i8_impff_s0/s1/s2 (as asf);
          --with-tcn: tcn full_i8_impff_s0/s1/s2 too
  eval    experiment.py on the 11 val games under FINAL's possession (owner TCN + boundary), FINAL's
          conditions (bar exact / fuzzy / random, atd fuzzy / exact seeds 0 1, atd_flip fuzzy), tracks
          cleaned, ball filled: asf, asf_i8 (new), v8, fin = mean(asf, v8) (= FINAL today),
          fin_i8 = mean(asf_i8, v8), gbm_full vs gbm_i8 (and tcn_i8 with --with-tcn)
  report  harness_results/i8_check_<tag>.md: levels, paired deltas, the #38 verdict
Other games only.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ..config import HARNESS_DIR, ROOT
from . import data, experiment, rules, suite, train
from .final_check import CONDITIONS, _fmt, _paired, _passes

IMPFF = ["idsse", "metrica", "pff"]
SEEDS = (0, 1, 2)
LABEL = {"asf": "asf (FINAL's alive models)", "asfi": "asf on full + i_ (new)", "tcni": "tcn on full + i_ (new)",
         "v8": "v8 in our HMM", "fin": "mean(asf, v8) = FINAL today", "fin_i8": "mean(asf_i8, v8)",
         "fin_ti8": "mean(tcn_i8, v8)", "gbm_full": "gbm_full_impff", "gbm_i8": "gbm on full + i_ (new)"}


def plan(a):
    q = "q_" if a.quick else ""
    seeds = (0, 1) if a.quick else SEEDS
    out = [("gbm", f"{q}full_i8_impff", "full_i8", IMPFF, 0)]
    out += [("asf", f"{q}full_i8_impff_s{s}", "full_i8", IMPFF, s) for s in seeds]
    if a.with_tcn:
        out += [("tcn", f"{q}full_i8_impff_s{s}", "full_i8", IMPFF, s) for s in seeds]
    return out


def stage_train(a, log):
    todo = [p for p in plan(a) if not (a.resume and train.model_path(p[0], p[1]).exists())]
    if not todo:
        log("train: every model exists (--resume)")
        return
    names = None
    if a.quick:
        from ..external import splits
        names = {n for s in IMPFF for n in splits.games("train", sources=[s])[:1]}
    log(f"train: {len(todo)} models; loading the tables ...")
    T = data.load_tables("train", IMPFF, 0, rules.RuleConfig(), log=log)
    if names is not None:
        T = [t for t in T if t[0] in names]
    for kind, tag, fs, srcs, seed in todo:
        log(f"train {kind} {tag} ({fs}, seed {seed})")
        train.fit(kind, T, tag, fs, rules.RuleConfig(), iters=30 if a.quick else 300, epochs=1 if a.quick else None,
                  seed=seed, device=a.device, sources=srcs, log=log, table_seed=0)


def stage_eval(a, log):
    from ..make_submission import FINAL
    fin = FINAL["options"]
    asf_old = [ROOT / p for p in fin["alive_model"]]
    who = "tcn=" + ",".join(str(ROOT / p) for p in fin["who_model"])
    v8 = ROOT / (fin.get("deadball_model") or "models/deadball_v8.joblib")
    new = [(k, t) for k, t, *_ in plan(a)]
    models = {f"asf_s{i}": p for i, p in enumerate(asf_old)}
    models["gbm_full"] = train.model_path("gbm", "full_impff")
    asfi, tcni = [], []
    for kind, tag in new:
        p = train.model_path(kind, tag)
        if not p.exists():
            raise SystemExit(f"missing {p}: run the train stage")
        if kind == "gbm":
            models["gbm_i8"] = p
        else:
            name = f"{kind}i_s{tag.rsplit('_s', 1)[1]}"
            models[name] = p
            (asfi if kind == "asf" else tcni).append(name)
    ens = ["asf=" + ",".join(k for k in models if k.startswith("asf_s")), "asfi=" + ",".join(asfi),
           "fin=asf,v8", "fin_i8=asfi,v8"]
    if tcni:
        ens += ["tcni=" + ",".join(tcni), "fin_ti8=tcni,v8"]
    base = ["--split", "val", "--source", *IMPFF, "--clean-tracks", "--device", a.device, "--jobs", str(a.jobs),
            "--who", who, "--who-bnd-alpha", str(fin.get("who_bnd_alpha", 0.5)),
            "--models", *[f"{k}={v}" for k, v in models.items() if v.exists()],
            "--deadball", f"v8={v8}:0.5", "--ensemble", *ens, "--variants", "hmm",
            "--params", *[f"{k}={v}" for k, v in FINAL["params"].items()]]
    if not models["gbm_full"].exists():
        log("  (no models/alive_gbm_full_impff.joblib: the gbm comparison is skipped)")
    if a.quick:
        base += ["--max-games", "1"]
    if a.no_ball_fill:
        base += ["--no-ball-fill"]
    for cname, cargs in CONDITIONS.items():
        if cname in a.skip or (a.quick and cname != "bar"):
            continue
        tag = f"{a.tag}_{cname}"
        if a.resume and (HARNESS_DIR / f"alive_{tag}.json").exists():
            log(f"eval {cname}: done (--resume)")
            continue
        log(f"eval {cname}")
        if experiment.main(base + list(cargs) + ["--tag", tag]) != 0:
            raise SystemExit(f"experiment failed: {cname}")


def stage_report(a, log):
    rows = []
    for cname in CONDITIONS:
        p = HARNESS_DIR / f"alive_{a.tag}_{cname}.json"
        if p.exists():
            rows += [{**r, "who": "tcn", "cond_label": cname} for r in json.loads(p.read_text())["runs"]]
    if not rows:
        log("report: no eval results yet")
        return
    df = pd.DataFrame(rows)
    groups, cols = {}, []
    for (cl, mode), g in df.groupby(["cond_label", "ball_mode"], sort=False):
        groups[f"{cl}/{mode}"] = g
        cols.append(f"{cl}/{mode}")
    have = set(df.method)
    out = [f"# Italo's inputs in our models `{a.tag}` (possession: FINAL's owner TCN; other games only)\n",
           "spellF1, mean per source then over sources; Δ paired per game × noise seed: mean (se) wins/n; "
           "PASS = Δ > 2 se and more than half the runs won in EVERY column (#38).\n",
           "| dead ball | " + " | ".join(cols) + " | mean |", "|---|" + "---|" * (len(cols) + 1)]
    for m in ("gbm_full", "gbm_i8", "asf", "asfi", "tcni", "v8", "fin", "fin_i8", "fin_ti8"):
        if m not in have:
            continue
        vals = []
        for c in cols:
            g = groups[c][(groups[c].method == m) & (groups[c].variant == "hmm")]
            vals.append(g.groupby("source").spellF1.mean().mean() if len(g) else float("nan"))
        out.append(f"| {LABEL[m]} | " + " | ".join(f"{x:.3f}" for x in vals) + f" | {np.nanmean(vals):.3f} |")
    comps = [("gbm_i8 − gbm_full (do the inputs carry signal?)", "gbm_i8", "gbm_full"),
             ("asf_i8 − asf", "asfi", "asf"), ("tcn_i8 − asf", "tcni", "asf"),
             ("**mean(asf_i8, v8) − FINAL today**", "fin_i8", "fin"), ("asf_i8 alone − FINAL today", "asfi", "fin"),
             ("mean(tcn_i8, v8) − FINAL today", "fin_ti8", "fin")]
    out += ["", "| comparison (spellF1) | " + " | ".join(cols) + " | passes |", "|---|" + "---|" * (len(cols) + 1)]
    verdict = {}
    for label, x, y in comps:
        if x not in have or y not in have:
            continue
        rs = {c: _paired(groups[c], ("tcn", x, "hmm"), ("tcn", y, "hmm")) for c in cols}
        verdict[x, y] = _passes(rs)
        out.append(f"| {label} | " + " | ".join(_fmt(rs[c]) for c in cols) + f" | {'PASS' if verdict[x, y] else 'no'} |")
    out.append("")
    if verdict.get(("fin_i8", "fin")):
        out.append("**mean(asf_i8, v8) beats FINAL in every column** -> FINAL's alive_model becomes the three "
                   "alive_asf_full_i8_impff_s*.pt (deadball_model stays v8).")
    elif verdict.get(("asfi", "fin")):
        out.append("**asf_i8 alone beats FINAL in every column** -> FINAL: alive_model = the asf_i8 files, no deadball_model.")
    else:
        out.append("Nothing beats FINAL in every column: keep FINAL as it is.")
    text = "\n".join(out) + "\n"
    (HARNESS_DIR / f"i8_check_{a.tag}.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    log(f"wrote {HARNESS_DIR / f'i8_check_{a.tag}.md'}")


STAGES = {"tables": suite.stage_tables, "train": stage_train, "eval": stage_eval, "report": stage_report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--skip", nargs="*", default=[], choices=list(CONDITIONS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--with-tcn", action="store_true", help="also train / score tcn on full + i_ (+~45 min)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-ball-fill", action="store_true")
    ap.add_argument("--tag", default=None, help="default: i8 (i8q with --quick)")
    a = ap.parse_args(argv)
    a.tag = a.tag or ("i8q" if a.quick else "i8")
    log = suite.Tee(HARNESS_DIR / f"i8_check_{a.tag}.log")
    log(f"i8_check {a.tag}: stages {a.stage}, resume {a.resume}, with_tcn {a.with_tcn}")
    for st in a.stage:
        t0 = time.time()
        STAGES[st](a, log)
        log(f"stage {st} done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
