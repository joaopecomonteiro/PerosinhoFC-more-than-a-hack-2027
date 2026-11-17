"""One command for the final fusion / Viterbi stage (v2, 16 Nov). Run from the repo root:

    python run_fusion.py --synthetic                  # smoke on generated data (~3 min), no real inputs
    python run_fusion.py --preds harness_results/fusion_preds \
        --final-check harness_results/final_check_fusion16.json --jobs 8

--preds is written by final_check (other games only, never BAR-ATH):
    python -m mqf.alive.final_check --jobs 3 --whos tcn --tag fusion16 --fusion-export harness_results/fusion_preds

Steps (each one stops the run if it fails):
  1 environment   numba present and pinned, official metric found
  2 inputs        every file: contract, games / sources / columns / seeds, streams, NaN inside halves
  3 tests         pytest tests/test_fusion.py
  4 sanity        on EVERY file: the production hook and hmm3_bnd@0 give the exported FINAL labels, and
                  those labels score what final_check scored (--final-check JSON); FINAL's column means
                  are printed next to #45's table -- STOP on any mismatch
  5 quick pass    every decoder, 12 configs (half of them near FINAL), nested leave-one-game-out
  6 deep pass     the top --top decoders by held-out spellF1, --budget configs
  7 verdict       quick + deep rows together: PIPELINE (decoder AND config chosen inside each fold) vs
                  FINAL, #38's rule in every column; the shipped config is rounded and re-checked

Outputs (aggregates only, safe to share): harness_results/fusion_<tag>_{quick,deep}_{report.md,scores.csv,
best.json}, fusion_<tag>_verdict.md and, if a decoder is adopted, fusion_<tag>_adopt.json (the
--fusion-config of make_submission).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def step(msg):
    print(f"\n=== {msg} ===", flush=True)


def fail(msg):
    print(f"\n[run_fusion] STOP: {msg}", flush=True)
    sys.exit(1)


def suite_ns(a, preds, tag, budget, quick, decoders, no_buckets=False):
    return argparse.Namespace(preds=preds, decoders=decoders, conds=None, seeds=None, budget=budget, quick=quick,
                              sample_seed=0, local_frac=a.local_frac, baseline=None,
                              baseline_hook=a.baseline_hook or None, duration_dir=a.duration_dir,
                              official=a.official, jobs=a.jobs, tag=tag, out=a.out, no_buckets=no_buckets)


# ------------------------------------------------------------------------------------ sanity

def _sanity_one(item):
    path, hook, official = item
    from mqf import fusion as fu
    fu.load_metric(official)
    g = fu.load_npz(path)
    fin = g.get("pred_final")
    out = {"path": path, "has_final": fin is not None, "nan_dead": int(g.get("nan_dead_in", 0)),
           "nan_home": int(g.get("nan_home_in", 0))}
    if fin is None:
        return out
    out["spell"] = fu.score(g["y"], fin)["spellF1"]
    out["centre_same"] = bool(np.array_equal(fu.decode("hmm3_bnd", g, fu.sample_configs("hmm3_bnd", 1, {})[0]), fin))
    if hook:
        out["prod_same"] = bool(np.array_equal(fu._call_hook(hook, g), fin))
    return out


def sanity(a, files, fu):
    items = [(f["path"], a.baseline_hook or None, a.official) for f in files]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(_sanity_one, items, chunksize=1))
    else:
        res = [_sanity_one(it) for it in items]
    by_path = {r["path"]: r for r in res}
    n = len(res)
    if not all(r["has_final"] for r in res):
        fail("some files have no pred_final: re-export with mqf.alive.final_check --fusion-export")
    bad_nan = [r for r in res if r["nan_dead"] or r["nan_home"]]
    if bad_nan:
        print(f"WARNING: {len(bad_nan)} files have NaN P(dead)/P(home) inside the halves; production falls back to "
              f"v0 cues there and the fusion evidence uses 0.5 -- only the hook is exact on those frames")
    cs = sum(r["centre_same"] for r in res)
    print(f"hmm3_bnd@0 == FINAL labels: {cs}/{n} files")
    if a.baseline_hook:
        ps = sum(r["prod_same"] for r in res)
        print(f"production hook ({a.baseline_hook}) == FINAL labels: {ps}/{n} files")
        if ps < n:
            fail("the production hook does not reproduce the exported FINAL labels: fix the export first")
    if cs < n - len(bad_nan):
        fail("hmm3_bnd config 0 is not FINAL: the fusion evidence does not match decode.emissions")
    # FINAL's column means (mean per source, then over sources) -- compare with #45 / final_check's table
    rows = [{"decoder": "final", "cfg": 0, "cond": f["cond"], "game": f["game"], "seed": f["seed"],
             "source": fu.load_npz(f["path"]).get("source", "all"), "spellF1": by_path[f["path"]]["spell"],
             "macroF1": np.nan, "adF1": np.nan, "cnt_err": np.nan} for f in files]
    tab = fu.Table(rows)
    cols = fu._col_means(tab, tab.v["spellF1"][0])
    print("FINAL (exported labels), spellF1 per column: " +
          ", ".join(f"{c} {v:.3f}" for c, v in zip(tab.conds, cols)) + f"; mean {np.nanmean(cols):.3f}")
    if a.final_check:
        ref = {}
        for r in json.loads(Path(a.final_check).read_text())["rows"]:
            if r.get("who") == "tcn" and r.get("method") == a.final_method and r.get("variant") == "hmm":
                ref[(f"{r['cond_label']}_{r['ball_mode']}", r["game"], str(r["seed"]))] = r["spellF1"]
        diffs, missing = [], 0
        for f in files:
            k = (f["cond"], f["game"], f["seed"])
            if k not in ref:
                missing += 1
                continue
            diffs.append(abs(by_path[f["path"]]["spell"] - ref[k]))
        if not diffs:
            fail(f"no (tcn, {a.final_method}, hmm) rows of {a.final_check} match the files")
        exact = sum(d == 0 for d in diffs)
        print(f"vs {a.final_check}: {exact}/{len(diffs)} runs identical, max |Δ| {max(diffs):.4f}, "
              f"{missing} files without a row")
        if max(diffs) > a.sanity_tol:
            fail(f"FINAL labels do not score what final_check scored (max |Δ| {max(diffs):.4f} > --sanity-tol "
                 f"{a.sanity_tol}); pass the JSON of the export run itself, or a tolerance for an older run")
    return cols


# ------------------------------------------------------------------------------------ verdict helpers

def _round_cfg(cfg):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, float) and not isinstance(v, bool) and v != 0:
            out[k] = float(f"{v:.2g}")
        else:
            out[k] = v
    return out


def _score_cfg(item):
    path, name, cfgs, official, prior = item
    from mqf import fusion as fu
    fu.load_metric(official)
    g = fu.load_npz(path)
    if prior is not None:
        g["_dur_prior"] = prior
    return [fu.score(g["y"], fu.decode(name, g, c))["spellF1"] for c in cfgs]


def merge_rows(quick, deep, top):
    keep = [r for r in quick["rows"] if r["decoder"] not in top and r["decoder"] not in ("final", "prod")]
    return keep + deep["rows"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--preds", help="dir of <column>/<game>__s<seed>.npz (val games only)")
    ap.add_argument("--synthetic", action="store_true", help="smoke run on generated data")
    ap.add_argument("--final-check", help="final_check_<tag>.json of the export run (per-run sanity)")
    ap.add_argument("--final-method", default="asf_v8", help="FINAL's dead-ball method in that JSON")
    ap.add_argument("--sanity-tol", type=float, default=0.0, help="max |Δ spellF1| vs --final-check (0 = exact)")
    ap.add_argument("--baseline-hook", default="mqf.decode:fusion_baseline", help="production decoder ('' = none)")
    ap.add_argument("--duration-dir", help="GT-only npz of train games for hsmm_emp priors")
    ap.add_argument("--official", help="dir holding the official possession_metrics.py")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--top", type=int, default=3, help="decoders carried to the deep pass")
    ap.add_argument("--budget", type=int, default=120, help="configs per decoder, deep pass")
    ap.add_argument("--local-frac", type=float, default=0.5, help="share of configs sampled near FINAL")
    ap.add_argument("--tag", default=time.strftime("f%d%H%M"))
    ap.add_argument("--out", default="harness_results")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()

    # 1. environment -------------------------------------------------------------------------
    step("1/7 environment")
    from mqf import fusion as fu
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, numba={fu.HAVE_NUMBA}")
    if not fu.HAVE_NUMBA:
        print("WARNING: numba missing -> numpy fallback, same results, ~50x slower. pip install -r requirements.txt")
    req = (HERE / "requirements.txt").read_text() if (HERE / "requirements.txt").exists() else ""
    if "numba==" not in req:
        print("WARNING: numba is not pinned in requirements.txt (make reproduce needs the pin if a decoder ships)")
    pm = fu.load_metric(a.official)
    print(f"official metric: {pm.__file__}")

    # 2. inputs ------------------------------------------------------------------------------
    step("2/7 inputs")
    if a.synthetic:
        preds = os.path.join(a.out, "fusion_synth")
        if not os.path.isdir(preds):
            fu.main(["synth", "--out", preds, "--games", "6", "--n-seeds", "2", "--half-min", "20"])
        a.tag = "synth_" + a.tag
    else:
        preds = a.preds
        if not preds or not os.path.isdir(preds):
            fail("--preds missing. Write the inputs first (other games only):\n\n"
                 "    python -m mqf.alive.final_check --jobs 3 --whos tcn --tag fusion16 "
                 "--fusion-export harness_results/fusion_preds")
    files = fu.discover(preds)
    games = sorted({f["game"] for f in files})
    conds = sorted({f["cond"] for f in files})
    if any(g.upper().startswith("BAR") for g in games):
        fail("a game named BAR* is in --preds: the suite is for other games only")
    if len(games) < 3:
        fail("need >= 3 games for leave-one-game-out")
    meta = fu.probe_meta(files)
    srcs = {}
    for f in {f["game"]: f for f in files}.values():
        srcs.setdefault(fu.load_npz(f["path"]).get("source", "all"), []).append(f["game"])
    print(f"{len(files)} files, {len(games)} games ({', '.join(f'{s} {len(v)}' for s, v in sorted(srcs.items()))}), "
          f"columns {conds}")
    print("streams: " + ", ".join(k for k, v in meta.items() if k.startswith("has_") and v) + f"; fps {meta['fps']}")

    # 3. tests -------------------------------------------------------------------------------
    if not a.skip_tests:
        step("3/7 tests")
        env = dict(os.environ)
        env.setdefault("MQF_OFFICIAL", os.path.dirname(pm.__file__))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_fusion.py"], env=env, cwd=HERE)
        if r.returncode:
            fail("tests/test_fusion.py failed")

    # 4. sanity ------------------------------------------------------------------------------
    step("4/7 sanity (every file)")
    sanity(a, files, fu)

    # 5. quick pass --------------------------------------------------------------------------
    step("5/7 quick pass (all decoders, 12 configs)")
    quick = fu.run_suite(suite_ns(a, preds, a.tag + "_quick", 12, True, None))
    qres = quick["res"]["res"]
    ranked = [n for n in sorted(qres, key=lambda n: -qres[n]["mean"])
              if n in fu.DECODERS and n != "argmax"]
    top = ranked[:a.top]
    print("\nquick ranking (held out): " + ", ".join(f"{n} {qres[n]['mean']:.3f}" for n in ranked))
    print(f"FINAL {qres['final']['mean']:.3f}; deep pass: {top}")

    # 6. deep pass ---------------------------------------------------------------------------
    step(f"6/7 deep pass ({', '.join(top)}, {a.budget} configs)")
    deep = fu.run_suite(suite_ns(a, preds, a.tag + "_deep", a.budget, False, top))

    # 7. verdict -----------------------------------------------------------------------------
    step("7/7 verdict (quick + deep rows, nested)")
    rows = merge_rows(quick, deep, top)
    tab, res = fu.analyse(rows, "final")
    plan = {**quick["plan"], **deep["plan"]}
    C = tab.conds
    pipe = res["PIPELINE"]
    lines = [f"# Fusion verdict `{a.tag}`", "",
             f"- inputs `{preds}`: {len(games)} games × {len(C)} columns ({', '.join(C)}); baseline = FINAL's "
             "exported labels; every number held out (nested leave-one-game-out)",
             "- PIPELINE = decoder AND config chosen inside each fold on the other games: the honest estimate "
             "of shipping this suite's winner", "",
             "| row | " + " | ".join(C) + " | mean | vs FINAL per column: Δ (se) wins/n | passes |",
             "|---|" + "---|" * (len(C) + 3)]
    show = ["final", "PIPELINE", "hmm3_bnd@0", *[n for n in sorted(res, key=lambda n: -res[n]["mean"])
                                                if n in fu.DECODERS]]
    for n in dict.fromkeys(show):
        if n not in res:
            continue
        r = res[n]
        pr = "; ".join(fu._fmt_p(p) for p in r["paired"]) if "paired" in r else ""
        lines.append(f"| {n} | " + " | ".join(f"{x:.3f}" for x in r["cols"]["spellF1"]) +
                     f" | {r['mean']:.3f} | {pr} | {'PASS' if r.get('passes') else ''} |")
    win_d, win_c = pipe["final"]
    picks = [f"{d}#{c}" for d, c in pipe["picks"]]
    lines += ["", f"PIPELINE picks per fold: {', '.join(picks)} (agreement {pipe['fold_agreement']:.2f})", ""]
    adopt = pipe.get("passes") and res.get(win_d, {}).get("passes") and pipe["fold_agreement"] >= 0.5
    if adopt:
        cfg = plan[win_d][win_c]
        prior = None
        if win_d == "hsmm_emp":
            prior = fu.priors_for(files, ["__all__"], a.duration_dir)["__all__"]
        rcfg = _round_cfg(cfg)
        items = [(f["path"], win_d, [cfg, rcfg], a.official, prior) for f in files]
        with ProcessPoolExecutor(a.jobs) as ex:
            sc = np.array(list(ex.map(_score_cfg, items, chunksize=1)))
        loss = float(sc[:, 0].mean() - sc[:, 1].mean())
        ship = rcfg if loss <= 0.002 else cfg
        rounded = ship is rcfg
        if prior is not None:
            ship = {**ship, "_dur_prior": prior}
        out = Path(a.out) / f"fusion_{a.tag}_adopt.json"
        out.write_text(json.dumps({"decoder": win_d, "config": ship}, indent=2, sort_keys=True))
        lines += [f"**Adopt `{win_d}`** (PIPELINE passes in every column; the decoder's own row passes; "
                  f"{pipe['fold_agreement']:.0%} of folds agree). Rounded config loses {loss:+.4f} in-sample → "
                  f"shipping the {'rounded' if rounded else 'unrounded'} one: `{out}`", "",
                  "Build it next to FINAL and compare (aggregates only, never to choose):", "```",
                  "python -m mqf.make_submission --team Perosinho_FC_fusion --clean-tracks --ball-fill "
                  "--who-model models/owner_tcn_full_impff_s0.pt models/owner_tcn_full_impff_s1.pt "
                  "models/owner_tcn_full_impff_s2.pt --who-bnd-alpha 0.5 --alive-model models/alive_asf_full_impff_s0.pt "
                  "models/alive_asf_full_impff_s1.pt models/alive_asf_full_impff_s2.pt --deadball-model "
                  "models/deadball_v8.joblib --deadball-decoder hmm --params dead_who=max "
                  f"--fusion-decoder {win_d} --fusion-config {out}",
                  "python -m mqf.compare results/Perosinho_FC.csv results/Perosinho_FC_fusion.csv", "```",
                  "Then freeze it: FINAL['options'] += {'fusion_decoder': ..., 'fusion_config': <the JSON's config>}, "
                  "make reproduce, EXPERIMENTS #47, TECHNICAL.md."]
    else:
        why = ("PIPELINE does not pass in every column" if not pipe.get("passes") else
               f"the winner `{win_d}` does not pass on its own" if not res.get(win_d, {}).get("passes") else
               "the folds disagree on the winner")
        lines += [f"**Keep FINAL** ({why}). Log this table as EXPERIMENTS #47 and submit FINAL."]
    lines += ["", f"Reports: fusion_{a.tag}_quick_report.md, fusion_{a.tag}_deep_report.md"]
    vpath = Path(a.out) / f"fusion_{a.tag}_verdict.md"
    vpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[run_fusion] verdict: {vpath}")


if __name__ == "__main__":
    main()
