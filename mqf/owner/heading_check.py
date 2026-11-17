"""Do the pass-direction columns (who/who_lab.py, 15 Nov) beat who_v1? The #38 question, asked of them.

The colleague's lab found a GBM with pass-direction clues +0.028 (no noise) / +0.020 (noise) spellF1 over the
same GBM without them -- but against `nearest`, on the TRUE alive mask, one noise seed, no ball fill, no PFF
val games, one switch cost for every method. Here the same columns (feats.HEADING_COLS, ported exactly) go
through the #38 protocol instead:

  columns  head_h/a/diff   ball heading towards each team's nearest player (cosine)
           dvn_h/a/diff    |v_player - v_ball| of that player (+-0.25 s; ~ rel_v1_*, already in `full`)
           z_couple        close AND moving like the ball, every player, as log-odds
           z_chain         last clear controller, carried, faded over 4 s;  age_ctrl  seconds since it
  train    gbm_full_heading_im   full + heading            IDSSE 5 + Metrica 2 (who_v1's data and settings)
           gbm_stack_heading_im  full + rules + heading
           gbm_full_im / gbm_stack_im (#38) are reused if they exist, else trained -- the heading - no heading pairs
  methods  + blend (no training: mean of z_couple and z_chain as log-odds, who_lab's 'blend') and nearest
  eval     mqf.owner.experiment on the 11 val games, layers dead_v1 + oracle, hmm decoder, ball filled, #38's
           conditions: bar (seed 0) and atd (noise seeds 0 1), tracks NOT cleaned -> who_v1 0.516 // 0.504
  report   paired delta spellF1 vs who_v1, and heading - no heading; the #38 rule: layer dead_v1, mean > 2 se
           in EVERY condition and wins on more than half the runs -> harness_results/owner_<tag>_report.md

  --tcn    the only route to the submission: #40's candidate is a TCN ensemble, not a GBM. Three TCNs on
           full_heading, IDSSE + Metrica + PFF, init seeds 0 1 2, tables seed 0, 20 epochs (as #40), and their
           ensemble ens_impff_hd vs #40's ens_impff (both + own boundary, alpha 0.5), #40's conditions:
           --clean-tracks, bar | atd seeds 0 1 2 | atd_flip. Needs the #40 models (owner_tcn_full_impff_s*.pt).

    python -m mqf.owner.heading_check --quick [--tcn]  # smoke run: tiny q_ models, 1 val game per source, bar
    python -m mqf.owner.heading_check                  # GBM arm (~30-40 min: IDSSE/Metrica tables rebuilt once)
    python -m mqf.owner.heading_check --tcn            # + TCN arm (~2 h more on the GPU: PFF tables rebuilt once)
    python -m mqf.owner.heading_check --resume         # keep trained models and finished conditions
    python -m mqf.owner.heading_check --stage report [--tcn]

The first run after this change rebuilds the cached training tables (data.code_hash covers feats.py):
the old feature sets come out bit-identical, so #38 / #40 models stay valid and comparable.
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
ALPHA = 0.5
GBM_SOURCES = ["idsse", "metrica"]
TCN_SOURCES = ["idsse", "metrica", "pff"]
GBM_JOBS = (("full_im", "full", False), ("stack_im", "full_rules", False),          # (tag, feature set, new?)
            ("full_heading_im", "full_heading", True), ("stack_heading_im", "full_rules_heading", True))
GBM_CONDITIONS = {"bar": ("bar", [], [0]), "atd": ("atd", [], [0, 1])}              # #38
TCN_CONDITIONS = {"bar": ("bar", [], [0]), "atd": ("atd", [], [0, 1, 2]),           # #40
                  "atd_flip": ("atd", ["team_flip_p=0.05", "gk_team_flip_p=0.5"], [0, 1])}
SANITY_GBM = {"bar": 0.516, "atd": 0.504}                  # who_v1, layer dead_v1 (#38)
SANITY_TCN = {"bar": (0.518, 0.048), "atd": (0.507, 0.050), "atd_flip": (0.472, 0.054)}   # who_v1, ens_impff+bnd - who_v1 (#40)


QUICK = {"iters": 30, "epochs": 1, "train_seeds": [0], "conditions": ["bar"], "tcn_conditions": ["bar"],
         "tcn_sources": ["idsse", "metrica"]}
_PREFIX = ""                 # "q_" under --quick: the smoke run's models never overwrite the real ones


def gbm_name(tag: str) -> str:
    return f"gbm_{tag}"


def _gbm_file(tag: str):
    return train.model_path("gbm", _PREFIX + tag)


def tcn_tag(seed: int, heading: bool) -> str:
    return f"{_PREFIX}full_{'heading_' if heading else ''}impff_s{seed}"


def tcn_name(seed: int, heading: bool) -> str:
    return f"tcn_{tcn_tag(seed, heading)}"


ENS_OLD, ENS_NEW = "ens_impff", "ens_impff_hd"


# ------------------------------------------------------------------------------------ train

def stage_train(a, log):
    rc = rules.RuleConfig()
    todo = [(tag, fs) for tag, fs, new in GBM_JOBS
            if not _gbm_file(tag).exists() or (new and not a.resume) or a.quick]
    if todo:
        log(f"== train GBM: {[gbm_name(t) for t, _ in todo]} on {GBM_SOURCES}")
        T = data.load_tables("train", GBM_SOURCES, a.seed, rc, log=log, keep_seq=False)
        if not T:
            raise SystemExit("no training bundles: python tasks.py bundles")
        for tag, fs in todo:
            train.fit("gbm", T, _PREFIX + tag, fs, rc, iters=a.iters, seed=a.seed, device=a.device,
                      sources=GBM_SOURCES, log=log)
        del T
    else:
        log("== train GBM: all models exist")
    if not a.tcn:
        return
    old = [train.model_path("tcn", tcn_tag(k, False)) for k in a.train_seeds]
    heads = [True, False] if a.quick else [True]            # --quick trains its own tiny reference TCNs too
    if not a.quick and not all(p.exists() for p in old):
        log(f"  note: #40's TCNs missing ({[p.name for p in old if not p.exists()]}) -- "
            "run python -m mqf.owner.tcn_check --sets impff --stage train first, or the comparison has no reference")
    todo = [k for k in a.train_seeds if not (a.resume and train.model_path("tcn", tcn_tag(k, True)).exists())]
    if not todo:
        log("== train TCN heading: all models exist (--resume)")
        return
    log(f"== train TCN heading: {a.tcn_sources}, init seeds {todo}, tables seed {a.table_seed}, {a.epochs} epochs")
    T = data.load_tables("train", a.tcn_sources, a.table_seed, rc, log=log)
    missing = sorted(set(a.tcn_sources) - {src for _, src, _ in T})
    if missing:
        raise SystemExit(f"no {'/'.join(missing)} training bundles (python -m mqf.external.build)")
    for k in todo:
        for h in heads:
            train.fit("tcn", T, tcn_tag(k, h), "full_heading" if h else "full", rc, epochs=a.epochs, seed=k,
                      device=a.device, sources=a.tcn_sources, log=log, table_seed=a.table_seed)
    del T


# ------------------------------------------------------------------------------------ eval

def gbm_args(a, cname: str) -> list[str]:
    cond, noise, seeds = GBM_CONDITIONS[cname]
    models = [f"{REF}={MODELS_DIR / 'who_v1.joblib'}"] + [f"{gbm_name(t)}={_gbm_file(t)}" for t, _, _ in GBM_JOBS]
    args = ["--split", "val", "--source", *a.sources, "--condition", cond, "--layers", "dead_v1", "oracle",
            "--decoders", "hmm", "--models", *models, "--hard", "--ref", REF, "--device", a.device,
            "--tag", f"{a.tag}_{cname}"]
    if cond == "atd":
        args += ["--seeds", *[str(k) for k in seeds]]
    if noise:
        args += ["--noise", *noise]
    if a.clean_tracks:
        args += ["--clean-tracks"]
    return args + a.extra


def tcn_args(a, cname: str) -> list[str]:
    cond, noise, seeds = TCN_CONDITIONS[cname]
    old = {tcn_name(k, False): train.model_path("tcn", tcn_tag(k, False)) for k in a.train_seeds}
    new = {tcn_name(k, True): train.model_path("tcn", tcn_tag(k, True)) for k in a.train_seeds}
    tcns = {**old, **new}
    args = ["--split", "val", "--source", *a.sources, "--condition", cond, "--layers", "dead_v1", "oracle",
            "--decoders", "hmm", "--models", f"{REF}={MODELS_DIR / 'who_v1.joblib'}",
            *[f"{k}={v}" for k, v in tcns.items()], "--hard", "--boundary-from", *tcns,
            "--ensemble", f"{ENS_OLD}={','.join(old)}", f"{ENS_NEW}={','.join(new)}",
            "--bnd-alpha", str(ALPHA), "--ref", REF, "--device", a.device, "--clean-tracks",
            "--tag", f"{a.tag}_tcn_{cname}"]
    if cond == "atd":
        args += ["--seeds", *[str(k) for k in seeds]]
    if noise:
        args += ["--noise", *noise]
    return args + a.extra


def _run(a, log, cname, args, tag):
    if a.resume and (HARNESS_DIR / f"owner_{tag}.json").exists():
        log(f"== eval {tag}: done (--resume)")
        return
    log(f"== eval {tag}\n  python -m mqf.owner.experiment " + " ".join(args))
    rc = experiment.main(args)
    if rc:
        log(f"  experiment {tag} returned {rc}")


def stage_eval(a, log):
    missing = [gbm_name(t) for t, _, _ in GBM_JOBS if not _gbm_file(t).exists()]
    if missing:
        raise SystemExit(f"missing models {missing}: run --stage train first")
    for cname in a.conditions:
        _run(a, log, cname, gbm_args(a, cname), f"{a.tag}_{cname}")
    if a.tcn:
        need = [train.model_path("tcn", tcn_tag(k, h)) for k in a.train_seeds for h in (False, True)]
        lost = [p.name for p in need if not p.exists()]
        if lost:
            raise SystemExit(f"--tcn: missing {lost}")
        for cname in a.tcn_conditions:
            _run(a, log, cname, tcn_args(a, cname), f"{a.tag}_tcn_{cname}")


# ------------------------------------------------------------------------------------ report

def gbm_pairs() -> list[tuple[str, str, str]]:
    """(label, a, b): a candidate vs who_v1 (the decision) and the heading - no heading pairs (the cause)."""
    out = [(f"{gbm_name(t)} vs who_v1", gbm_name(t), REF) for t, _, _ in GBM_JOBS]
    out += [("blend vs who_v1", "blend", REF), ("nearest vs who_v1", "nearest", REF),
            ("heading effect: gbm_full_heading_im - gbm_full_im", "gbm_full_heading_im", "gbm_full_im"),
            ("heading effect: gbm_stack_heading_im - gbm_stack_im", "gbm_stack_heading_im", "gbm_stack_im"),
            ("blend - nearest (who_lab's comparison)", "blend", "nearest")]
    return out


def tcn_pairs(seeds) -> list[tuple[str, list[str], list[str]]]:
    """(label, a methods, b methods): several methods = one per training seed, summarised as mean [min, max]."""
    bnd = f"+bnd{ALPHA:g}"
    return [(f"{ENS_NEW} + own boundary vs who_v1", [ENS_NEW + bnd], [REF]),
            (f"{ENS_OLD} + own boundary vs who_v1 (#40)", [ENS_OLD + bnd], [REF]),
            (f"heading effect: {ENS_NEW} - {ENS_OLD} (both + own boundary)", [ENS_NEW + bnd], [ENS_OLD + bnd]),
            (f"heading effect: {ENS_NEW} - {ENS_OLD} (no boundary)", [ENS_NEW], [ENS_OLD]),
            ("heading effect per seed: TCN_hd - TCN (+ own boundary)",
             [tcn_name(k, True) + bnd for k in seeds], [tcn_name(k, False) + bnd for k in seeds])]


def _cell(df, am, bm):
    rs = [r for r in (paired(df, x, y) for x, y in zip(am, bm)) if r is not None]
    if not rs:
        return "—", None
    if len(rs) == 1:
        r = rs[0]
        src = " ".join(f"{k[2:]} {v:+.3f}" for k, v in r.items() if k.startswith("d_"))
        return f"{r['mean']:+.4f} ({r['se']:.4f}) {r['wins']} · {src}", r
    means = [r["mean"] for r in rs]
    return (f"{np.mean(means):+.4f} [{min(means):+.4f}, {max(means):+.4f}] over {len(rs)} seeds",
            {"mean": float(np.mean(means)), "min": float(min(means)), "max": float(max(means)), "n_seeds": len(rs)})


def passes(res: dict) -> bool:
    """#38's rule over the conditions present: mean > 2 se and wins on more than half the runs, in every one."""
    if not res or any(r is None or "se" not in r for r in res.values()):
        return False
    return all(r["mean"] > 2 * r["se"] and 2 * int(r["wins"].split("/")[0]) > int(r["wins"].split("/")[1])
               for r in res.values())


def _load(tags: dict) -> dict:
    out = {}
    for cname, tag in tags.items():
        path = HARNESS_DIR / f"owner_{tag}.json"
        if path.exists():
            df = pd.DataFrame(json.loads(path.read_text())["runs"])
            out[cname] = df[df.decoder == "hmm"]
    return out


def _table(runs: dict, pairs, layer: str, out: list, summary: dict):
    out.append(f"\n### Layer `{layer}`\n")
    out.append("| comparison | " + " | ".join(runs) + " |")
    out.append("|---|" + "---|" * len(runs))
    for label, am, bm in pairs:
        cells, res = [], {}
        for cname, df in runs.items():
            c, r = _cell(df[df.layer == layer], am, bm)
            cells.append(c)
            res[cname] = r
        out.append(f"| {label} | " + " | ".join(cells) + " |")
        summary.setdefault(layer, {})[label] = res


def stage_report(a, log):
    out = [f"# Heading check `{a.tag}` (other games only)\n",
           "Pass-direction columns (feats.HEADING_COLS, ported from who/who_lab.py) through the #38 protocol. "
           "Paired Δ spellF1 = a − b per game × noise seed; cells: mean (se) wins · per source. Rule (#38): "
           "layer dead_v1, mean > 2 se in every condition and wins on more than half the runs.\n"]
    summary, verdict = {}, []
    runs = _load({c: f"{a.tag}_{c}" for c in a.conditions})
    if runs:
        out.append(f"\n## GBM arm — #38's conditions{' + --clean-tracks' if a.clean_tracks else ''}\n")
        san = []
        for cname, df in runs.items():
            d = df[df.layer == "dead_v1"].groupby("method").spellF1.mean()
            line = f"`{cname}` who_v1 {d.get(REF, float('nan')):.3f}"
            if cname in SANITY_GBM and not a.clean_tracks and not a.quick:
                line += f" (expect {SANITY_GBM[cname]:.3f}, #38)"
            if cname == "bar" and not a.quick:
                line += f", nearest {d.get('nearest', float('nan')):.3f} (expect 0.309)"
            san.append(line)
        out.append("Sanity: " + "; ".join(san) + "\n")
        sg = {}
        _table(runs, [(lab, [x], [y]) for lab, x, y in gbm_pairs()], "dead_v1", out, sg)
        _table(runs, [(lab, [x], [y]) for lab, x, y in gbm_pairs()], "oracle", out, sg)
        summary["gbm"] = sg
        for lab, x, y in gbm_pairs():
            if y == REF and x not in ("nearest", gbm_name("full_im"), gbm_name("stack_im")):
                verdict.append((passes(sg["dead_v1"][lab]), lab, "replaces who_v1 (#38 rule)"))
            elif lab.startswith("heading effect"):
                verdict.append((passes(sg["dead_v1"][lab]), lab, "the columns add something to the trees"))
    truns = _load({c: f"{a.tag}_tcn_{c}" for c in a.tcn_conditions}) if a.tcn else {}
    if truns:
        out.append("\n## TCN arm — #40's conditions (--clean-tracks)\n")
        san = []
        for cname, df in truns.items():
            d = df[df.layer == "dead_v1"]
            w = d[d.method == REF].spellF1.mean()
            line = f"`{cname}` who_v1 {w:.3f}"
            if cname in SANITY_TCN and not a.quick:
                r = paired(d, ENS_OLD + f"+bnd{ALPHA:g}", REF)
                line += (f" (expect {SANITY_TCN[cname][0]:.3f}), {ENS_OLD}+bnd − who_v1 "
                         f"{(r or {}).get('mean', float('nan')):+.3f} (expect {SANITY_TCN[cname][1]:+.3f}, #40)")
            san.append(line)
        out.append("Sanity: " + "; ".join(san) + "\n")
        st = {}
        _table(truns, tcn_pairs(a.train_seeds), "dead_v1", out, st)
        _table(truns, tcn_pairs(a.train_seeds), "oracle", out, st)
        summary["tcn"] = st
        lab = f"heading effect: {ENS_NEW} - {ENS_OLD} (both + own boundary)"
        verdict.append((passes(st["dead_v1"][lab]), lab, "heading goes into #40's TCN candidate"))
        lab = f"{ENS_NEW} + own boundary vs who_v1"
        verdict.append((passes(st["dead_v1"][lab]), lab, "replaces who_v1 (#38 rule)"))
    if not runs and not truns:
        log("  no results yet")
        return
    out.append("\n## Verdict (layer dead_v1, every condition)\n")
    for ok, lab, meaning in verdict:
        out.append(f"- {'PASS' if ok else 'fail'}  {lab}  → {meaning}")
    gbm_ok = [lab for ok, lab, m in verdict if ok and m.startswith("replaces") and lab.startswith("gbm")]
    out.append("\n" + ("A heading GBM passes against who_v1 — but #40's TCN ensemble is ahead of who_v1 by ~+0.05: "
                       "compare against that before touching the submission." if gbm_ok else
                       "No heading GBM passes against who_v1: the #38 decision stands for the trees.")
               + (" The TCN arm decides whether the columns go into the #40 candidate." if a.tcn else
                  " The columns can only reach the submission through the TCN: run with --tcn."))
    text = "\n".join(out) + "\n"
    (HARNESS_DIR / f"owner_{a.tag}_report.md").write_text(text, encoding="utf-8")
    (HARNESS_DIR / f"owner_{a.tag}_report.json").write_text(json.dumps(
        {"summary": summary, "verdict": [{"pass": ok, "comparison": lab, "meaning": m} for ok, lab, m in verdict]},
        indent=1, default=float))
    print("\n" + text)
    log(f"wrote {HARNESS_DIR / f'owner_{a.tag}_report.md'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all", choices=["all", "train", "eval", "report"])
    ap.add_argument("--resume", action="store_true", help="keep trained models and finished conditions")
    ap.add_argument("--tcn", action="store_true", help="also the TCN arm (3 heading TCNs vs #40's ens_impff)")
    ap.add_argument("--conditions", nargs="*", default=list(GBM_CONDITIONS), choices=list(GBM_CONDITIONS))
    ap.add_argument("--tcn-conditions", nargs="*", default=list(TCN_CONDITIONS), choices=list(TCN_CONDITIONS))
    ap.add_argument("--clean-tracks", action="store_true", help="GBM arm with the cleaner (default off, as #38)")
    ap.add_argument("--sources", nargs="*", default=["idsse", "metrica", "pff"], help="validation sources")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0, help="GBM seed and its training tables' seed (#38: 0)")
    ap.add_argument("--train-seeds", nargs="*", type=int, default=[0, 1, 2], help="TCN init seeds (#40: 0 1 2)")
    ap.add_argument("--table-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="smoke run: tiny q_ models, 1 val game per source, bar only")
    ap.add_argument("--tag", default=None, help="default: headchk (headchkq with --quick)")
    a = ap.parse_args(argv)
    global _PREFIX
    _PREFIX = "q_" if a.quick else ""
    a.tag = a.tag or ("headchkq" if a.quick else "headchk")
    a.tcn_sources = list(TCN_SOURCES)
    a.extra = []
    if a.quick:
        for k, v in QUICK.items():
            setattr(a, k, v)
        a.extra = ["--max-games", "1"]
    log = Tee(HARNESS_DIR / f"owner_{a.tag}.log")
    log(f"=== mqf.owner.heading_check {' '.join(sys.argv[1:] if argv is None else argv)}  ({time.strftime('%Y-%m-%d %H:%M')})")
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
