"""How much two submissions differ -- aggregates only, no frame numbers, no ground truth.

    python -m mqf.compare results/Perosinho_FC.csv results/clean.csv
    python -m mqf.compare base_v0 ballfill_v5          # bare names are looked up in results/

For a change whose value the harness cannot measure on the real match (e.g. track cleaning), this
says how much it moves the output: share of graded frames that change, between which labels, how
long the changed stretches are, and what happens to the possession count. It never says which file
is better -- that is the harness's job (the public 5 minutes are for correctness only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import decode


def resolve(arg: str, results_dir) -> Path:
    """The submission a CLI argument means: the path as given, else results/<name>[.csv]. A missing
    file is a clear message listing what results/ holds (file names only), never a traceback."""
    from pathlib import Path
    p = Path(arg)
    for cand in (p, results_dir / p, results_dir / p.name, results_dir / f"{p.name}.csv"):
        if cand.is_file():
            return cand
    have = sorted(f.name for f in results_dir.glob("*.csv")) if results_dir.is_dir() else []
    raise SystemExit(f"no submission at {arg}.\n"
                     f"CSVs in {results_dir}: {', '.join(have) if have else 'none'}\n"
                     f"Build one with: python tasks.py submit --team <name>   (-> results/<name>.csv)")


def _labels(path) -> np.ndarray:
    df = pd.read_csv(path, dtype={"label": str})
    return df["label"].fillna("X").str.strip().str.upper().to_numpy()


def compare(a: np.ndarray, b: np.ndarray, halves, fps: int) -> dict:
    if len(a) != len(b):
        raise SystemExit(f"different lengths: {len(a)} vs {len(b)}")
    graded = np.zeros(len(a), dtype=bool)
    for _, first, last in halves:
        graded[first:last + 1] = True
    ga, gb = a[graded], b[graded]
    diff = ga != gb
    edges = np.diff(np.r_[0, diff.astype(np.int8), 0])
    runs = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / fps
    pairs = pd.Series([f"{x}->{y}" for x, y in zip(ga[diff], gb[diff])]).value_counts(normalize=True)

    def stats(lab):
        poss = []
        for _, first, last in halves:
            poss.append(decode.spell_stats(lab[first:last + 1], fps))
        n = sum(p["possessions"] for p in poss)
        return {"possessions": n, "share": {k: round(float(np.mean(lab[graded] == k)), 4) for k in "HADX"}}

    return {"graded_frames_changed": round(float(diff.mean()), 4),
            "changed_stretches": {"n": int(len(runs)),
                                  "length_s": {f"p{q}": round(float(np.percentile(runs, q)), 2) for q in (50, 90, 99)}
                                  if len(runs) else None},
            "changes_by_label": {k: round(float(v), 3) for k, v in pairs.items()},
            "a": stats(a), "b": stats(b)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--synthetic", action="store_true", help="files are on the synthetic match's timeline")
    args = ap.parse_args(argv)
    if args.synthetic:
        from .config import SYNTHETIC_DIR
        from . import synthetic
        spec = synthetic.generate(SYNTHETIC_DIR)
        results_dir = SYNTHETIC_DIR / "results"
    else:
        from .config import RESULTS_DIR, competition_game
        spec = competition_game()
        results_dir = RESULTS_DIR
    a, b = resolve(args.a, results_dir), resolve(args.b, results_dir)
    print(f"a = {a.name}   b = {b.name}")
    r = compare(_labels(a), _labels(b), spec.halves, spec.fps)
    print("=== mqf.compare -- aggregates only, safe to share ===")
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
