"""Old FINAL vs new FINAL on the harness: possession model x dead-ball model (incl. Italo's v8), other games only.

    python -m mqf.alive.final_check --jobs 3            # ~1 h on the GPU (WHO evidence of who_v1 is cached
                                                        # from the alive16 suite when it ran on this machine)
    python -m mqf.alive.final_check --quick             # 1 val game per source, bar only (smoke)
    python -m mqf.alive.final_check --jobs 3 --whos tcn --tag fusion16 --fusion-export harness_results/fusion_preds
                                                        # FINAL's runs only + the fusion suite's inputs
                                                        # (mqf.fusion_export; then python run_fusion.py)

Possession {who_v1, owner TCN (3 seeds + its boundary)} x dead ball {dead_v1, asf (3 seeds, mqf.alive),
and -- when models/deadball_v8.joblib is there -- Italo's v8 (mqf.deadball): its P(dead) in our HMM, v8 with
its own semi-Markov decoder (prior 0.5, his frozen setting; possession decoded inside its stoppages), and
mean(asf, v8)}, on the same 11 val games, conditions and noise seeds as #40 / #42:

    who_v1 x dead_v1 = FINAL_V1 (old)        owner TCN x asf = FINAL (new)

The model paths come from make_submission.FINAL / FINAL_V1 (one source of truth). Conditions: bar (ball
exact / fuzzy / random), atd (noise seeds 0 1, ball fuzzy / exact), atd_flip (seed 0, ball fuzzy); tracks
cleaned, ball filled, production HMM (dead_who=max), the owner TCN's boundary on the switch cost at
alpha FINAL["who_bnd_alpha"]. Rows are paired across the two runs by condition x ball mode x game x seed.
The new recipe passes when new - old > 2 se and wins more than half the runs in EVERY column (#38);
new - possession-only and new - alive-only show that the two gains add up (#40 asked to re-check the
boundary once the dead layer changes).
Writes harness_results/final_check_<tag>.json / .md. Aggregates only.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ..config import HARNESS_DIR
from . import experiment

CONDITIONS = {"bar": ["--condition", "bar", "--ball-modes", "exact", "fuzzy", "random"],
              "atd": ["--condition", "atd", "--seeds", "0", "1", "--ball-modes", "fuzzy", "exact"],
              "atd_flip": ["--condition", "atd", "--seeds", "0", "--noise", "team_flip_p=0.05", "gk_team_flip_p=0.5",
                           "--ball-modes", "fuzzy"]}
DEAD = [("dead_v1", "hmm"), ("asf", "hmm"), ("v8", "hmm"), ("v8_hsmm", "hsmm"), ("asf_v8", "hmm")]
DEAD_LABEL = {"dead_v1": "dead_v1", "asf": "asf (mqf.alive, 3 seeds)", "v8": "v8 P(dead) in our HMM",
              "v8_hsmm": "v8 + its semi-Markov decoder (prior 0.5)", "asf_v8": "mean(asf, v8) in our HMM"}
WHOS = ("who_v1", "tcn")
OLD = ("who_v1", "dead_v1", "hmm")


def new_key() -> tuple:
    """The (possession, dead ball, variant) row that make_submission.FINAL corresponds to."""
    from ..make_submission import FINAL
    o = FINAL["options"]
    if o.get("deadball_model") and o.get("alive_model"):
        return ("tcn", "asf_v8", "hmm")
    if o.get("deadball_model"):
        return ("tcn", "v8_hsmm", "hsmm") if o.get("deadball_decoder", "hsmm") == "hsmm" else ("tcn", "v8", "hmm")
    return ("tcn", "asf", "hmm")


NEW = new_key()


def _sel(df, key):
    w, m, v = key
    return df[(df.who == w) & (df.method == m) & (df.variant == v)]


def _paired(df, a, b, col="spellF1"):
    x = _sel(df, a).set_index(["game", "seed"])[col]
    y = _sel(df, b).set_index(["game", "seed"])[col]
    idx = x.index.intersection(y.index)
    if not len(idx):
        return None
    d = (x[idx] - y[idx]).to_numpy()
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"mean": float(d.mean()), "se": se, "wins": int((d > 0).sum()), "n": len(d)}


def _passes(rs: dict) -> bool:
    return all(r is not None and np.isfinite(r["se"]) and r["mean"] > 2 * r["se"] and 2 * r["wins"] > r["n"]
               for r in rs.values())


def _fmt(r):
    if r is None:
        return "—"
    se = "n/a" if not np.isfinite(r["se"]) else f"{r['se']:.3f}"
    return f"{r['mean']:+.3f} ({se}) {r['wins']}/{r['n']}"


def report(rows: list[dict], tag: str) -> str:
    df = pd.DataFrame(rows)
    groups, cols = {}, []
    for (cl, mode), g in df.groupby(["cond_label", "ball_mode"], sort=False):
        groups[f"{cl}/{mode}"] = g
        cols.append(f"{cl}/{mode}")
    have = set(zip(df.who, df.method, df.variant))
    out = [f"# Final check `{tag}`: possession model × dead-ball model (other games only)\n",
           "spellF1, mean per source then over sources; Δ paired per game × noise seed: mean (se) wins/n. "
           "A row passes when its Δ > 2 se and it wins more than half the runs in EVERY column (#38).\n",
           "| possession × dead ball | " + " | ".join(cols) + " | mean |", "|---|" + "---|" * (len(cols) + 1)]
    for w in WHOS:
        for m, v in DEAD:
            if (w, m, v) not in have:
                continue
            vals = []
            for c in cols:
                g = _sel(groups[c], (w, m, v))
                vals.append(g.groupby("source").spellF1.mean().mean() if len(g) else float("nan"))
            mark = " **(new FINAL)**" if (w, m, v) == NEW else (" (old FINAL)" if (w, m, v) == OLD else "")
            out.append(f"| {w} × {DEAD_LABEL[m]}{mark} | " + " | ".join(f"{x:.3f}" for x in vals)
                       + f" | {np.nanmean(vals):.3f} |")
    vals = []
    for c in cols:
        g = groups[c][(groups[c].who == "who_v1") & (groups[c].method == "oracle")]
        vals.append(g.groupby("source").spellF1.mean().mean() if len(g) else float("nan"))
    if np.isfinite(vals).any():                                       # who_v1 not run (--whos tcn): no ceiling row
        out.append("| who_v1 × true alive mask (ceiling) | " + " | ".join(f"{x:.3f}" for x in vals)
                   + f" | {np.nanmean(vals):.3f} |")
    comps = [("**new FINAL − old FINAL**", NEW, OLD), ("possession only (tcn × dead_v1) − old", ("tcn", "dead_v1", "hmm"), OLD),
             ("alive only (who_v1 × asf) − old", ("who_v1", "asf", "hmm"), OLD),
             ("new FINAL − tcn × dead_v1 (the gains add up?)", NEW, ("tcn", "dead_v1", "hmm"))]
    for w in WHOS:
        for m, v in DEAD[2:]:
            comps.append((f"{w}: {DEAD_LABEL[m]} − asf", (w, m, v), (w, "asf", "hmm")))
        comps.append((f"{w}: v8 + its decoder − dead_v1", (w, "v8_hsmm", "hsmm"), (w, "dead_v1", "hmm")))
    out += ["", "| comparison (spellF1) | " + " | ".join(cols) + " | passes |", "|---|" + "---|" * (len(cols) + 1)]
    verdict = {}
    for label, a, b in comps:
        if a not in have or b not in have:
            continue
        rs = {c: _paired(groups[c], a, b) for c in cols}
        ok = _passes(rs)
        verdict[label] = ok
        out.append(f"| {label} | " + " | ".join(_fmt(rs[c]) for c in cols) + f" | {'PASS' if ok else 'no'} |")
    rs = {c: _paired(groups[c], NEW, OLD, "macroF1") for c in cols}
    out.append("| new FINAL − old FINAL (macroF1) | " + " | ".join(_fmt(rs[c]) for c in cols) + " | |")
    out.append("")
    if "**new FINAL − old FINAL**" not in verdict:
        out.append("Old FINAL not in this run (--whos): no old-vs-new verdict.")
    else:
        out.append("**PASS**: the new FINAL beats the old one in every column." if verdict["**new FINAL − old FINAL**"]
                   else "**FAIL**: the new FINAL does not beat the old one in every column -- do not submit it; send "
                        "this table.")
    alts = [k for k, ok in verdict.items() if k.startswith("tcn:") and k.endswith("− asf") and ok]
    out.append(("Better than asf under the new possession model, in every column: " + "; ".join(alts) + " -- "
                "consider it for FINAL.") if alts else
               "Nothing beats asf under the new possession model in every column: keep asf in FINAL.")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(CONDITIONS))
    ap.add_argument("--no-ball-fill", action="store_true")
    ap.add_argument("--no-v8", action="store_true", help="leave out Italo's mqf.deadball v8 (models/deadball_v8.joblib)")
    ap.add_argument("--tag", default="final16")
    ap.add_argument("--whos", nargs="*", default=list(WHOS), choices=WHOS, help="possession models to run")
    ap.add_argument("--fusion-export", default=None, metavar="DIR",
                    help="write the fusion suite's inputs from the tcn runs (python run_fusion.py --preds DIR)")
    ap.add_argument("--fusion-who-v1", action="store_true", help="... with who_v1's P(home) as a second stream")
    a = ap.parse_args(argv)
    from ..make_submission import FINAL, FINAL_V1
    fin, old = FINAL["options"], FINAL_V1["options"]
    asf = fin["alive_model"]
    who_tcn = fin["who_model"]
    if old["who_model"] != "models/who_v1.joblib" or old["dead_model"] != "models/dead_v1.joblib":
        raise SystemExit("final_check expects FINAL_V1 = who_v1 + dead_v1")
    from ..config import ROOT
    missing = [p for p in [*asf, *who_tcn] if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit(f"missing model files: {missing} (train them, or copy them into models/)")
    v8 = ROOT / "models" / "deadball_v8.joblib"
    use_v8 = v8.is_file() and not a.no_v8
    base = ["--split", "val", "--source", "idsse", "metrica", "pff", "--clean-tracks", "--device", a.device,
            "--jobs", str(a.jobs), "--models", *[f"asf_s{k}={ROOT / p}" for k, p in enumerate(asf)],
            "--ensemble", "asf=" + ",".join(f"asf_s{k}" for k in range(len(asf))),
            *(["asf_v8=asf,v8", "--deadball", f"v8={v8}:0.5"] if use_v8 else []),
            "--variants", "hmm", "--params", *[f"{k}={v}" for k, v in FINAL["params"].items()],
            "--who-bnd-alpha", str(fin.get("who_bnd_alpha", 0.5))]
    if a.quick:
        base += ["--max-games", "1"]
    if a.no_ball_fill:
        base += ["--no-ball-fill"]
    whos = {"who_v1": "who_v1", "tcn": "tcn=" + ",".join(str(ROOT / p) for p in who_tcn)}
    rows = []
    t0 = time.time()
    for cname, cargs in CONDITIONS.items():
        if cname in a.skip or (a.quick and cname != "bar"):
            continue
        for wname, wspec in whos.items():
            if wname not in a.whos:
                continue
            tag = f"{a.tag}_{cname}_{wname}"
            args = base + list(cargs) + ["--who", wspec, "--tag", tag]
            if a.fusion_export and wname == "tcn":
                if NEW[1] == "asf_v8" and not use_v8:
                    raise SystemExit("--fusion-export: FINAL uses v8 but models/deadball_v8.joblib is missing")
                args += ["--fusion-export", a.fusion_export, "--fusion-cond", cname, "--fusion-final", NEW[1]]
                if a.fusion_who_v1:
                    args.append("--fusion-who-v1")
            print(f"[final_check] {cname} / {wname}", flush=True)
            if experiment.main(args) != 0:
                raise SystemExit(f"experiment failed: {cname} / {wname}")
            for r in json.loads((HARNESS_DIR / f"alive_{tag}.json").read_text())["runs"]:
                rows.append({**r, "who": wname, "cond_label": cname})
    text = report(rows, a.tag)
    print("\n" + text)
    (HARNESS_DIR / f"final_check_{a.tag}.md").write_text(text, encoding="utf-8")
    (HARNESS_DIR / f"final_check_{a.tag}.json").write_text(json.dumps(
        {"rows": rows, "minutes": round((time.time() - t0) / 60, 1)}, indent=1, default=float))
    print(f"wrote {HARNESS_DIR / f'final_check_{a.tag}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
