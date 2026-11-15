"""Every owner experiment of 15 Nov 2026, in order, with one command (other games only).

    python -m mqf.owner.suite                 # train everything, evaluate (bar + atd), report  (~2-3 h, GPU)
    python -m mqf.owner.suite --quick         # smoke run: tiny models, 1 val game per source (~10 min)
    python -m mqf.owner.suite --stage train   # or eval / report -- resume a stage
    python -m mqf.owner.suite --skip nn_all source_ablation      # leave blocks out

Blocks (models land in models/owner_*, results in harness_results/owner_<tag>_*):
  train
    base             IDSSE + Metrica train games (who_v1's data):
                       gbm_who_im    GBM on who.py's 59 columns  (our pipeline reproduces who_v1)
                       gbm_full_im   GBM on the full set (+ relative velocity, +-3 s windows, team shape)
                       gbm_stack_im  GBM on full + rule columns    ("rules as features")
                       mlp_full_im   MLP on the full set            (control only: trees win on tabular data)
                       tcn_full_im   TCN + boundary head on 5 Hz sequences of the full set (ASRF-style)
                       gru_im        BiGRU on 5 Hz per-frame inputs (only with --with-gru)
    source_ablation  gbm_full on + SkillCorner / + PFF / all four  (SkillCorner: 10 fps, phase labels)
    nn_all           mlp_full_all, tcn_full_all on all four sources (the ball model gained from PFF; trees did not)
  eval               mqf.owner.experiment on the 11 val games (IDSSE 2, Metrica 1, PFF 8), layers oracle /
                     smooth50 / dead_v1, decoders hmm + raw, hard variants, hybrids on gbm_full_im and
                     tcn_full_im, the TCN boundary on tcn / who_v1 / the hybrid bases:
                       bar  players dropped, ball hidden on dead frames, ball filled          (#35)
                       atd  the ATD noise model, seeds 0 1, ball hidden on dead frames, filled (#34 b)
    pathcrf          only when external/pathcrf/*.npz exist (python -m mqf.owner.pathcrf_bridge ...): the IDSSE
                     val games under `clean` (players as delivered) and `clean_noball` (no ball, as PathCRF)
  report             harness_results/owner_<tag>_report.md: the questions below, paired vs who_v1

Questions the report answers (spellF1, paired per game x seed, mean +- se, wins):
  Q1 rules        rule / rule_hard / rule_seg vs who_v1
  Q2 features     gbm_full_im vs gbm_who_im, gbm_stack_im vs gbm_full_im
  Q3 sources      gbm_full_{imskc, impff, all} vs gbm_full_im
  Q4 NN           tcn_full_im / mlp_full_im vs gbm_full_im (+ the _all versions)
  Q5 probability  <model> vs <model>_hard, and per-frame log-loss / Brier / ECE
  Q6 hybrids      <base>+prod / +prod0.5 / +override / +veto vs <base>
  Q7 boundaries   tcn+bnd vs tcn; who_v1 / gbm_full_im with the TCN boundary vs without
  Q8 PathCRF      pathcrf vs who_v1, gbm_full_im, tcn_full_im (clean players, IDSSE val games only)
A gap counts when |mean| > 2 se and it beats the ~0.04 single-match noise on most games.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, HARNESS_DIR, MODELS_DIR
from . import data, experiment, rules, train

BLOCKS = ("base", "source_ablation", "nn_all", "bar", "atd", "pathcrf")
PATHCRF_DIR = "pathcrf"                 # external/<this>/<GAME>.npz, written by pathcrf_bridge align
REF = "who_v1"


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def _p(kind, tag, quick=False):
    """Model file of a block; --quick writes its tiny models under a q_ prefix (never over the real ones)."""
    return train.model_path(kind, ("q_" if quick else "") + tag)


def stage_train(a, log):
    rc = rules.RuleConfig()
    q = a.quick
    iters, ep_mlp, ep_gru, ep_tcn = (60, 1, 1, 2) if q else (a.iters, a.epochs_mlp, a.epochs_gru, a.epochs_tcn)
    if "base" not in a.skip:
        log("== train: base (IDSSE + Metrica)")
        T = data.load_tables("train", ["idsse", "metrica"], a.seed, rc, log=log)
        jobs = [("gbm", "who_im", "who", {}), ("gbm", "full_im", "full", {}),
                ("gbm", "stack_im", "full_rules", {}), ("mlp", "full_im", "full", {"epochs": ep_mlp}),
                ("tcn", "full_im", "full", {"epochs": ep_tcn})]
        if a.with_gru:
            jobs.append(("gru", "im", "full", {"epochs": ep_gru}))
        for kind, tag, fs, kw in jobs:
            if a.resume and _p(kind, tag, q).exists():
                log(f"  {_p(kind, tag, q).name} exists (--resume)")
                continue
            train.fit(kind, T, ("q_" if q else "") + tag, fs, rc, iters=iters, epochs=kw.get("epochs"), seed=a.seed,
                      device=a.device, sources=["idsse", "metrica"], log=log)
        del T
    if q:
        return
    extra = {}
    if "source_ablation" not in a.skip or "nn_all" not in a.skip:
        for src in ("skillcorner", "pff"):
            log(f"== tables: {src}")
            extra[src] = data.load_tables("train", [src], a.seed, rc, log=log, keep_seq="nn_all" not in a.skip)
        base = data.load_tables("train", ["idsse", "metrica"], a.seed, rc, log=log, keep_seq="nn_all" not in a.skip)
    if "source_ablation" not in a.skip:
        log("== train: source ablation (gbm full)")
        for tag, T, need in (("full_imskc", base + extra["skillcorner"], ["skillcorner"]),
                             ("full_impff", base + extra["pff"], ["pff"]),
                             ("full_all", base + extra["skillcorner"] + extra["pff"], ["skillcorner", "pff"])):
            if a.resume and _p("gbm", tag).exists():
                continue
            missing = [s for s in need if not extra[s]]
            if missing:
                log(f"  gbm_{tag}: skipped, no {'/'.join(missing)} training bundles (python -m mqf.external.build)")
                continue
            train.fit("gbm", T, tag, "full", rc, iters=iters, seed=a.seed, device=a.device,
                      sources=sorted({s for _, s, _ in T}), log=log)
    if "nn_all" not in a.skip and (extra.get("skillcorner") or extra.get("pff")):
        log("== train: NN on all sources")
        T = base + extra.get("skillcorner", []) + extra.get("pff", [])
        for kind, tag, kw in (("mlp", "full_all", {"epochs": ep_mlp}), ("tcn", "full_all", {"epochs": ep_tcn})):
            if a.resume and _p(kind, tag).exists():
                continue
            train.fit(kind, T, tag, "full", rc, iters=iters, epochs=kw["epochs"], seed=a.seed, device=a.device,
                      sources=sorted({s for _, s, _ in T}), log=log)


def _models_available(quick=False):
    cand = {REF: MODELS_DIR / "who_v1.joblib"}
    for kind, tag in (("gbm", "who_im"), ("gbm", "full_im"), ("gbm", "stack_im"), ("mlp", "full_im"),
                      ("tcn", "full_im"), ("gru", "im"), ("gbm", "full_imskc"), ("gbm", "full_impff"),
                      ("gbm", "full_all"), ("mlp", "full_all"), ("tcn", "full_all")):
        cand[f"{kind}_{tag}"] = _p(kind, tag, quick)
    return {k: v for k, v in cand.items() if Path(v).exists()}


def stage_eval(a, log):
    models = _models_available(a.quick)
    if REF not in models:
        log(f"  note: {MODELS_DIR / 'who_v1.joblib'} missing -- deltas will be against {next(iter(models), None)}")
    if not a.with_gru:
        models.pop("gru_im", None)
    hard = [m for m in (REF, "gbm_full_im", "mlp_full_im", "tcn_full_im", "gru_im", "gbm_stack_im") if m in models]
    hyb = [m for m in ("gbm_full_im", "tcn_full_im") if m in models]
    common = ["--models", *[f"{k}={v}" for k, v in models.items()], "--hard", *hard,
              "--hybrid-base", *hyb, "--ref", REF, "--device", a.device]
    if "tcn_full_im" in models:
        common += ["--boundary-from", "tcn_full_im"]
    if a.quick:
        common += ["--max-games", "1", "--layers", "oracle", "dead_v1", "--decoders", "hmm"]
    else:
        common += ["--layers", "oracle", "smooth50", "dead_v1", "--decoders", "hmm", "raw"]
    runs = [(c, a.sources, []) for c in ("bar", "atd") if c not in a.skip and not (a.quick and c == "atd")]
    pdir = EXTERNAL_DIR / PATHCRF_DIR
    if "pathcrf" not in a.skip:
        if list(pdir.glob("*.npz")):
            runs += [(c, ["idsse"], ["--external", f"pathcrf={pdir}"]) for c in ("clean", "clean_noball")]
        else:
            log(f"  (no PathCRF predictions in {pdir}: block pathcrf skipped -- see mqf.owner.pathcrf_bridge)")
    for cond, sources, extra in runs:
        log(f"== eval: condition {cond}")
        args = ["--source", *sources] + common + extra + ["--condition", cond, "--tag", f"{a.tag}_{cond}"]
        if cond == "atd":
            args += ["--seeds", *[str(s) for s in a.noise_seeds]]
        if cond == "clean_noball":
            args += ["--no-ball-fill"]
        log("  python -m mqf.owner.experiment " + " ".join(args))
        rc = experiment.main(args)
        if rc:
            log(f"  experiment {cond} returned {rc}")


# ------------------------------------------------------------------------------------ report

QUESTIONS = [
    ("Q1 rules vs who_v1", [("rule", REF), ("rule_hard", REF), ("rule_seg", REF), ("hold", REF), ("smooth50", REF),
                            ("nearest", REF)]),
    ("Q2 features", [("gbm_who_im", REF), ("gbm_full_im", "gbm_who_im"), ("gbm_full_im", REF),
                     ("gbm_stack_im", "gbm_full_im")]),
    ("Q3 training sources (gbm full)", [("gbm_full_imskc", "gbm_full_im"), ("gbm_full_impff", "gbm_full_im"),
                                        ("gbm_full_all", "gbm_full_im")]),
    ("Q4 NN vs trees", [("tcn_full_im", "gbm_full_im"), ("tcn_full_im", REF), ("mlp_full_im", "gbm_full_im"),
                        ("gru_im", "gbm_full_im"), ("tcn_full_all", "tcn_full_im"), ("mlp_full_all", "mlp_full_im"),
                        ("tcn_full_all", "gbm_full_im")]),
    ("Q5 probabilities vs hard labels", [(m, f"{m}_hard") for m in (REF, "gbm_full_im", "gbm_stack_im", "mlp_full_im",
                                                                    "tcn_full_im", "gru_im")] + [("rule", "rule_hard")]),
    ("Q6 rule x model hybrids", [(f"{b}+{h}", b) for b in ("gbm_full_im", "tcn_full_im")
                                 for h in ("prod", "prod0.5", "override", "veto")]),
    ("Q7 TCN boundaries in the decoder", [("tcn_full_im+bnd0.5", "tcn_full_im"), ("tcn_full_im+bnd0.9", "tcn_full_im"),
                                          (f"{REF}+tcn_full_imbnd0.5", REF), (f"{REF}+tcn_full_imbnd0.9", REF),
                                          ("gbm_full_im+tcn_full_imbnd0.5", "gbm_full_im"),
                                          ("gbm_full_im+tcn_full_imbnd0.9", "gbm_full_im")]),
    ("Q8 PathCRF (clean players, IDSSE val)", [("pathcrf", REF), ("pathcrf", "gbm_full_im"), ("pathcrf", "tcn_full_im"),
                                              ("pathcrf", "rule")]),
]


def paired(df: pd.DataFrame, a: str, b: str):
    x = df[df.method == a].set_index(["game", "seed"])["spellF1"]
    y = df[df.method == b].set_index(["game", "seed"])["spellF1"]
    idx = x.index.intersection(y.index)
    if not len(idx):
        return None
    d = (x[idx] - y[idx]).to_numpy()
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    by_src = df[df.method == a].set_index(["game", "seed"]).loc[idx, "source"]
    per_src = pd.Series(d, index=by_src.to_numpy()).groupby(level=0).mean()
    return {"a": a, "b": b, "mean": float(d.mean()), "se": se, "wins": f"{int((d > 0).sum())}/{len(d)}",
            "a_spellF1": float(x[idx].mean()), "b_spellF1": float(y[idx].mean()),
            **{f"d_{s}": float(v) for s, v in per_src.items()},
            "verdict": ("a better" if d.mean() > 2 * se else "b better" if d.mean() < -2 * se else "no gap")
            if np.isfinite(se) else "n=1"}


def stage_report(a, log):
    out = [f"# Owner experiments `{a.tag}` (other games only)\n",
           "Paired Δ spellF1 = a − b per game × seed; verdict needs |mean| > 2 se. hmm decoder unless noted.\n",
           "Sanity first: in condition `bar`, layer `dead_v1`, `who_v1` and `nearest` must reproduce EXPERIMENTS #35 "
           "on the 11 val games (spellF1 0.516 and 0.309) -- same models, same ball fill, same decoder. If they do "
           "not, stop and check the models / bundles before reading anything else.\n"]
    summary = {}
    for cond in ("bar", "atd", "clean", "clean_noball"):
        path = HARNESS_DIR / f"owner_{a.tag}_{cond}.json"
        if not path.exists():
            continue
        runs = pd.DataFrame(json.loads(path.read_text())["runs"])
        frames = pd.DataFrame(json.loads(path.read_text())["frames"])
        out.append(f"\n## Condition `{cond}` ({runs.game.nunique()} games, seeds {sorted(runs.seed.unique().tolist())})\n")
        if cond == "bar":
            chk = runs[(runs.layer == "dead_v1") & (runs.decoder == "hmm") & runs.method.isin([REF, "nearest"])]
            if len(chk):
                got = chk.groupby("method").spellF1.mean()
                out.append("Sanity vs #35 (plain mean over games; expect who_v1 0.516, nearest 0.309 on the 11 val games): "
                           + ", ".join(f"`{m}` {v:.3f}" for m, v in got.items()) + f" ({chk.game.nunique()} games)\n")
        for layer, dl in runs[runs.decoder == "hmm"].groupby("layer", sort=False):
            out.append(f"\n### Layer `{layer}`\n")
            top = (dl.groupby(["method", "source"]).spellF1.mean().groupby("method").mean()
                   .sort_values(ascending=False))
            out.append("Top 8 by spellF1 (mean per source, then over sources): "
                       + ", ".join(f"`{m}` {v:.3f}" for m, v in top.head(8).items()) + "\n")
            rows = []
            for q, pairs in QUESTIONS:
                for x, y in pairs:
                    r = paired(dl, x, y)
                    if r:
                        rows.append({"question": q, **r})
            if rows:
                t = pd.DataFrame(rows)
                cols = ["question", "a", "b", "a_spellF1", "b_spellF1", "mean", "se", "wins", "verdict"] + \
                       sorted(c for c in t.columns if c.startswith("d_"))
                out.append("```\n" + t[cols].round(3).to_string(index=False) + "\n```\n")
                summary[f"{cond}/{layer}"] = rows
        if "raw" in runs.decoder.unique():
            out.append("\n### Decoder matters: hmm − raw (oracle layer)\n")
            dl = runs[runs.layer == "oracle"]
            rows = []
            for m in sorted(dl.method.unique()):
                h = dl[(dl.method == m) & (dl.decoder == "hmm")].set_index(["game", "seed"]).spellF1
                r = dl[(dl.method == m) & (dl.decoder == "raw")].set_index(["game", "seed"]).spellF1
                idx = h.index.intersection(r.index)
                if len(idx):
                    rows.append({"method": m, "hmm": h[idx].mean(), "raw": r[idx].mean(), "hmm-raw": (h[idx] - r[idx]).mean()})
            if rows:
                out.append("```\n" + pd.DataFrame(rows).sort_values("hmm", ascending=False).round(3).to_string(index=False) + "\n```\n")
        if len(frames):
            out.append("\n### Per alive frame (calibration)\n")
            g = frames[frames.ball == "all"].groupby(["method", "source"])[["acc", "auc", "logloss", "brier", "ece"]].mean()
            g = g.groupby("method").mean().sort_values("logloss")
            out.append("```\n" + g.round(3).to_string() + "\n```\n")
            miss = frames[frames.ball == "missing"]
            if len(miss) and miss["n"].sum() > 0 and "acc" in miss:
                gm = miss.groupby(["method", "source"])["acc"].mean().groupby("method").mean().sort_values(ascending=False)
                out.append("Accuracy where the ball is NOT tracked: " + ", ".join(f"`{m}` {v:.3f}" for m, v in gm.head(10).items()) + "\n")
    text = "\n".join(out)
    path = HARNESS_DIR / f"owner_{a.tag}_report.md"
    path.write_text(text, encoding="utf-8")
    (HARNESS_DIR / f"owner_{a.tag}_report.json").write_text(json.dumps(summary, indent=1, default=float))
    log(text)
    log(f"wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all", choices=["all", "train", "eval", "report"])
    ap.add_argument("--quick", action="store_true", help="smoke run: tiny models, 1 val game per source, bar only")
    ap.add_argument("--skip", nargs="*", default=[], choices=BLOCKS)
    ap.add_argument("--resume", action="store_true", help="do not retrain models that already exist")
    ap.add_argument("--sources", nargs="*", default=["idsse", "metrica", "pff"], help="validation sources")
    ap.add_argument("--noise-seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--epochs-mlp", type=int, default=4)
    ap.add_argument("--epochs-gru", type=int, default=8)
    ap.add_argument("--epochs-tcn", type=int, default=20)
    ap.add_argument("--with-gru", action="store_true", help="also train / evaluate the BiGRU (replaced by the TCN)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default=None, help="default: owner15 (owner15q with --quick)")
    a = ap.parse_args(argv)
    a.tag = a.tag or ("owner15q" if a.quick else "owner15")
    log = Tee(HARNESS_DIR / f"owner_{a.tag}.log")
    log(f"=== mqf.owner.suite {' '.join(sys.argv[1:] if argv is None else argv)}  ({time.strftime('%Y-%m-%d %H:%M')})")
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
