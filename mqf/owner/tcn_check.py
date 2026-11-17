"""Is the owner TCN worth adding to the submission? Training seeds x training sources, cleaned tracks.

EXPERIMENTS #38 found who_v1 + the TCN boundary (alpha 0.5) at +0.008 (bar) // +0.006 (atd): ONE TCN (init
seed 0), trained on IDSSE + Metrica, scored on tracks that were NOT cleaned. Since then the submission cleans
the tracks (#39), PFF is allowed for training (and the TCN was the one model that gained from more games,
#38), and one training seed is one draw. This driver re-asks the question under the submission's conditions:

  train   tcn_full_im_s{0,1,2}      IDSSE + Metrica           (training tables seed 0, init seeds 0 1 2)
          tcn_full_impff_s{0,1,2}   IDSSE + Metrica + PFF
  eval    one runner pass per condition over the 11 val games, --clean-tracks, layers dead_v1 + oracle:
            who_v1, the six TCNs, their per-source ensembles ens_im / ens_impff (mean P(home), mean boundary),
            each one's boundary on itself and on who_v1 (alpha 0.5)
          conditions  bar | atd (noise seeds 0 1 2) | atd_flip (team_flip_p=0.05 gk_team_flip_p=0.5, seeds 0 1)
  report  paired delta spellF1 vs who_v1 (layer dead_v1), the #38 rule: > 2 se in EVERY condition and more
          than half the runs -> harness_results/owner_<tag>_report.md (+ .json)

    python -m mqf.owner.tcn_check                     # everything (~2 h on a GPU)
    python -m mqf.owner.tcn_check --resume            # keep trained models and finished conditions
    python -m mqf.owner.tcn_check --stage report

Sanity line in the report: who_v1 under atd / atd_flip with the cleaner, seeds 0 1, must equal #39's
`whoclean_on_nf` (0.507) / `whoclean_on` (0.472) -- same games, noise, models and decoder.
Other games only: nothing here reads the competition match.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from ..config import HARNESS_DIR, MODELS_DIR
from . import data, experiment, rules, train
from .suite import Tee, paired

REF = "who_v1"
SETS = {"im": ["idsse", "metrica"], "impff": ["idsse", "metrica", "pff"]}
CONDITIONS = {"bar": ("bar", [], [0]),
              "atd": ("atd", [], [0, 1, 2]),
              "atd_flip": ("atd", ["team_flip_p=0.05", "gk_team_flip_p=0.5"], [0, 1])}
SANITY = {"atd": ("whoclean_on_nf", 0.507), "atd_flip": ("whoclean_on", 0.472)}   # #39, seeds 0 1
ALPHA = 0.5


def tcn_tag(s: str, seed: int) -> str:
    return f"full_{s}_s{seed}"


def tcn_name(s: str, seed: int) -> str:
    return f"tcn_{tcn_tag(s, seed)}"


def stage_train(a, log):
    rc = rules.RuleConfig()
    for s in a.sets:
        todo = [k for k in a.train_seeds if not (a.resume and train.model_path("tcn", tcn_tag(s, k)).exists())]
        if not todo:
            log(f"== train {s}: all models exist (--resume)")
            continue
        log(f"== train {s}: sources {SETS[s]}, init seeds {todo}, tables seed {a.table_seed}, {a.epochs} epochs")
        T = data.load_tables("train", SETS[s], a.table_seed, rc, log=log)
        missing = sorted(set(SETS[s]) - {src for _, src, _ in T})
        if missing:
            log(f"  skipped: no {'/'.join(missing)} training bundles (python -m mqf.external.build)")
            continue
        for k in todo:
            train.fit("tcn", T, tcn_tag(s, k), "full", rc, epochs=a.epochs, seed=k, device=a.device,
                      sources=SETS[s], log=log, table_seed=a.table_seed)
        del T


def _models(a, log):
    out, groups = {}, {}
    for s in a.sets:
        have = {tcn_name(s, k): train.model_path("tcn", tcn_tag(s, k)) for k in a.train_seeds}
        lost = [n for n, p in have.items() if not p.exists()]
        if lost:
            log(f"  missing {lost} -- train them first")
        have = {n: p for n, p in have.items() if p.exists()}
        out.update(have)
        if len(have) > 1:
            groups[f"ens_{s}"] = list(have)
    return out, groups


def stage_eval(a, log):
    tcns, groups = _models(a, log)
    if not tcns:
        log("  no TCN models: nothing to evaluate")
        return
    for cname in a.conditions:
        cond, noise, seeds = CONDITIONS[cname]
        tag = f"{a.tag}_{cname}"
        if a.resume and (HARNESS_DIR / f"owner_{tag}.json").exists():
            log(f"== eval {cname}: done (--resume)")
            continue
        args = ["--split", "val", "--source", *a.sources, "--condition", cond, "--layers", "dead_v1", "oracle",
                "--decoders", "hmm", "--models", f"{REF}={MODELS_DIR / 'who_v1.joblib'}",
                *[f"{k}={v}" for k, v in tcns.items()], "--hard", "--boundary-from", *tcns,
                "--bnd-alpha", str(ALPHA), "--ref", REF, "--device", a.device, "--tag", tag]
        if groups:
            args += ["--ensemble", *[f"{e}={','.join(m)}" for e, m in groups.items()]]
        if not a.no_clean:
            args += ["--clean-tracks"]
        if cond == "atd":
            args += ["--seeds", *[str(k) for k in seeds]]
        if noise:
            args += ["--noise", *noise]
        log(f"== eval {cname}\n  python -m mqf.owner.experiment " + " ".join(args))
        rc = experiment.main(args)
        if rc:
            log(f"  experiment {cname} returned {rc}")


# ------------------------------------------------------------------------------------ report

def candidates(a) -> list[tuple[str, str, list[str]]]:
    """(family, label, methods): a single-TCN family is summarised over its training seeds."""
    out = []
    for s in a.sets:
        singles = [tcn_name(s, k) for k in a.train_seeds]
        e = f"ens_{s}"
        out += [(s, f"who_v1 + TCN_{s} boundary (1 seed)", [f"{REF}+{m}bnd{ALPHA:g}" for m in singles]),
                (s, f"who_v1 + {e} boundary", [f"{REF}+{e}bnd{ALPHA:g}"]),
                (s, f"TCN_{s} (1 seed)", singles),
                (s, f"TCN_{s} + own boundary (1 seed)", [f"{m}+bnd{ALPHA:g}" for m in singles]),
                (s, f"{e}", [e]),
                (s, f"{e} + own boundary", [f"{e}+bnd{ALPHA:g}"])]
    return out


def _cell(df, methods):
    """One table cell: single method -> mean (se) wins; several (training seeds) -> mean of means [min, max]."""
    rs = [r for r in (paired(df, m, REF) for m in methods) if r is not None]
    if not rs:
        return "—", None
    if len(methods) == 1:
        r = rs[0]
        src = " ".join(f"{k[2:]} {v:+.3f}" for k, v in r.items() if k.startswith("d_"))
        return f"{r['mean']:+.4f} ({r['se']:.4f}) {r['wins']} · {src}", r
    means = [r["mean"] for r in rs]
    return f"{np.mean(means):+.4f} [{min(means):+.4f}, {max(means):+.4f}] over {len(rs)} seeds", \
        {"mean": float(np.mean(means)), "min": float(min(means)), "max": float(max(means)), "n_seeds": len(rs)}


def stage_report(a, log):
    runs = {}
    for cname in a.conditions:
        path = HARNESS_DIR / f"owner_{a.tag}_{cname}.json"
        if path.exists():
            df = pd.DataFrame(json.loads(path.read_text())["runs"])
            runs[cname] = df[df.decoder == "hmm"]
    if not runs:
        log("  no results yet")
        return
    out = [f"# TCN check `{a.tag}` (other games only)\n",
           f"Paired Δ spellF1 vs `{REF}` per game × noise seed, tracks cleaned (unless --no-clean), α = {ALPHA}. "
           "Single-method cells: mean (se) wins · per source. Seed families: mean of the per-seed means "
           "[worst, best seed]. Rule (#38): mean > 2 se in every condition, layer dead_v1, wins on more than half.\n"]
    sanity = []
    for cname, df in runs.items():
        d = df[(df.layer == "dead_v1") & (df.method == REF)]
        line = f"`{cname}` who_v1 {d.spellF1.mean():.3f} ({d.game.nunique()} games, seeds {sorted(d.seed.unique().tolist())})"
        if cname in SANITY and not a.no_clean:
            ref_tag, ref_v = SANITY[cname]
            d01 = d[d.seed.isin([0, 1])]
            line += f"; seeds 0 1: {d01.spellF1.mean():.3f} (expect {ref_v:.3f} = #39 `{ref_tag}`)"
        sanity.append(line)
    out.append("Sanity: " + "; ".join(sanity) + "\n")
    summary = {}
    for layer in ("dead_v1", "oracle"):
        out.append(f"\n## Layer `{layer}`\n")
        out.append("| candidate | " + " | ".join(runs) + " |")
        out.append("|---|" + "---|" * len(runs))
        for fam, label, methods in candidates(a):
            cells, res = [], {}
            for cname, df in runs.items():
                c, r = _cell(df[df.layer == layer], methods)
                cells.append(c)
                res[cname] = r
            out.append(f"| {label} | " + " | ".join(cells) + " |")
            summary.setdefault(layer, {})[label] = res
    verdict = []
    for label, res in summary.get("dead_v1", {}).items():
        if not all(r is not None and "se" in r for r in res.values()) or len(res) < len(a.conditions):
            continue
        ok = all(r["mean"] > 2 * r["se"] and int(r["wins"].split("/")[0]) * 2 > int(r["wins"].split("/")[1])
                 for r in res.values())
        verdict.append((ok, min(r["mean"] for r in res.values()), label))
    out.append("\n## Verdict (layer dead_v1, every condition)\n")
    passing = sorted([v for v in verdict if v[0]], key=lambda v: -v[1])
    for ok, worst, label in sorted(verdict, key=lambda v: (not v[0], -v[1])):
        out.append(f"- {'PASS' if ok else 'fail'}  {label}  (worst-condition mean {worst:+.4f})")
    out.append("\n" + (f"Best passing: **{passing[0][2]}**. Before adopting it: it must run in make_submission on the "
                       "CPU, deterministically (float64, rounded outputs), and beat who_v1 by more than the cost of "
                       "the extra moving part." if passing else
                       "Nothing passes in every condition: keep who_v1 (EXPERIMENTS #38 decision stands)."))
    text = "\n".join(out) + "\n"
    (HARNESS_DIR / f"owner_{a.tag}_report.md").write_text(text, encoding="utf-8")
    (HARNESS_DIR / f"owner_{a.tag}_report.json").write_text(json.dumps(
        {"summary": summary, "passing": [v[2] for v in passing], "sanity": sanity}, indent=1, default=float))
    print("\n" + text)
    log(f"wrote {HARNESS_DIR / f'owner_{a.tag}_report.md'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all", choices=["all", "train", "eval", "report"])
    ap.add_argument("--resume", action="store_true", help="keep trained models and finished conditions")
    ap.add_argument("--sets", nargs="*", default=list(SETS), choices=list(SETS))
    ap.add_argument("--train-seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--table-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--sources", nargs="*", default=["idsse", "metrica", "pff"], help="validation sources")
    ap.add_argument("--no-clean", action="store_true", help="ablation: tracks as delivered (as #38)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="tcnchk")
    a = ap.parse_args(argv)
    log = Tee(HARNESS_DIR / f"owner_{a.tag}.log")
    log(f"=== mqf.owner.tcn_check {' '.join(sys.argv[1:] if argv is None else argv)}  ({time.strftime('%Y-%m-%d %H:%M')})")
    t0 = time.time()
    if a.stage in ("all", "train"):
        stage_train(a, log)
    if a.stage in ("all", "eval"):
        stage_eval(a, log)
    if a.stage in ("all", "report"):
        stage_report(a, log)
    log(f"=== done in {(time.time() - t0) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
