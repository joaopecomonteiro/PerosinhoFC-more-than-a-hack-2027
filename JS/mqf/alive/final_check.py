"""Old FINAL vs new FINAL on the harness: possession model x dead-ball model, 2 x 2, other games only.

    python -m mqf.alive.final_check --jobs 3            # ~1 h on the GPU (WHO evidence of who_v1 is cached
                                                        # from the alive16 suite when it ran on this machine)
    python -m mqf.alive.final_check --quick             # 1 val game per source, bar only (smoke)

The four combinations, on the same 11 val games, conditions and noise seeds as #40 / #42:

    possession \\ dead ball    dead_v1            asf (3 seeds, mqf.alive)
    who_v1                     FINAL_V1 (old)     alive only
    owner TCN (3 seeds + bnd)  possession only    FINAL (new)

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
COMBOS = {"old (FINAL_V1)": ("who_v1", "dead_v1"), "possession only": ("tcn", "dead_v1"),
          "alive only": ("who_v1", "asf"), "new (FINAL)": ("tcn", "asf")}


def _paired(df, a, b, col="spellF1"):
    x = df[(df.who == a[0]) & (df.method == a[1])].set_index(["game", "seed"])[col]
    y = df[(df.who == b[0]) & (df.method == b[1])].set_index(["game", "seed"])[col]
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


def report(rows: list[dict], tag: str) -> str:
    df = pd.DataFrame(rows)
    df = df[df.variant.isin(["hmm", "-"])]
    cols = []
    groups = {}
    for (cl, mode), g in df.groupby(["cond_label", "ball_mode"], sort=False):
        groups[f"{cl}/{mode}"] = g
        cols.append(f"{cl}/{mode}")
    out = [f"# Final check `{tag}`: old FINAL vs new FINAL (other games only)\n",
           "spellF1, mean per source then over sources; Δ paired per game × noise seed: mean (se) wins/n.\n",
           "| combination (possession × dead ball) | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for label, (w, m) in COMBOS.items():
        cells = []
        for c in cols:
            g = groups[c]
            g = g[(g.who == w) & (g.method == m)]
            cells.append(f"{g.groupby('source').spellF1.mean().mean():.3f}" if len(g) else "—")
        out.append(f"| {label}: {w} × {m} | " + " | ".join(cells) + " |")
    cells = []
    for c in cols:
        g = groups[c]
        g = g[(g.who == "who_v1") & (g.method == "oracle")]
        cells.append(f"{g.groupby('source').spellF1.mean().mean():.3f}" if len(g) else "—")
    out.append("| (true alive mask, who_v1) | " + " | ".join(cells) + " |")
    out += ["", "| comparison | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    comps = [("new − old (spellF1)", COMBOS["new (FINAL)"], COMBOS["old (FINAL_V1)"], "spellF1"),
             ("new − possession only", COMBOS["new (FINAL)"], COMBOS["possession only"], "spellF1"),
             ("new − alive only", COMBOS["new (FINAL)"], COMBOS["alive only"], "spellF1"),
             ("new − old (macroF1)", COMBOS["new (FINAL)"], COMBOS["old (FINAL_V1)"], "macroF1")]
    verdict = None
    for label, a, b, col in comps:
        rs = {c: _paired(groups[c], a, b, col) for c in cols}
        out.append(f"| {label} | " + " | ".join(_fmt(rs[c]) for c in cols) + " |")
        if label.startswith("new − old (spell"):
            ok = all(r is not None and np.isfinite(r["se"]) and r["mean"] > 2 * r["se"] and 2 * r["wins"] > r["n"]
                     for r in rs.values())
            verdict = ok
    out.append("")
    out.append("**PASS**: the new FINAL beats the old one in every column (#38 rule)." if verdict else
               "**FAIL**: the new FINAL does not beat the old one in every column -- do not submit it; send this table.")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(CONDITIONS))
    ap.add_argument("--no-ball-fill", action="store_true")
    ap.add_argument("--tag", default="final16")
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
    base = ["--split", "val", "--source", "idsse", "metrica", "pff", "--clean-tracks", "--device", a.device,
            "--jobs", str(a.jobs), "--models", *[f"asf_s{k}={ROOT / p}" for k, p in enumerate(asf)],
            "--ensemble", "asf=" + ",".join(f"asf_s{k}" for k in range(len(asf))),
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
            tag = f"{a.tag}_{cname}_{wname}"
            args = base + list(cargs) + ["--who", wspec, "--tag", tag]
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
