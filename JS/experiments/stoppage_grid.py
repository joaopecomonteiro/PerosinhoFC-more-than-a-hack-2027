"""EXPERIMENTS #48: the stoppage decoder's three lengths on a grid (+ minrun), nested leave-one-game-out.

    python run_fusion.py ...                       # first: the val-game exports (python tasks.py fusion-export)
    python experiments/stoppage_grid.py --preds harness_results/fusion_preds --jobs 4

Other games only (mqf.fusion refuses BAR*). Writes harness_results/fusion_stoppage48grid_{report.md,scores.csv,
best.json}. The sampled run of #48 is `python -m mqf.fusion run --preds ... --decoders stoppage minrun argmax
--budget 12 --tag stoppage48`.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mqf import fusion as f  # noqa: E402

GRID = [{"min_s": 0.5, "min_dead_s": 2.0, "min_between_s": 2.0}]           # config 0 = the viewer's first guess
for ms, md, mb in itertools.product([0.5, 1.0, 1.4, 1.8], [2.0, 3.0, 4.5], [0.5, 1.0, 2.0]):
    c = {"min_s": ms, "min_dead_s": md, "min_between_s": mb}
    if c not in GRID:
        GRID.append(c)
MINRUN = [{"min_s": s} for s in (0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.2)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", default="harness_results/fusion_preds")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="harness_results")
    a = ap.parse_args(argv)
    orig = f.sample_configs
    f.sample_configs = lambda name, n, meta, seed=0, lf=0.5: (GRID if name == "stoppage" else MINRUN
                                                              if name == "minrun" else orig(name, n, meta, seed, lf))
    f.run_suite(types.SimpleNamespace(
        preds=a.preds, decoders=["stoppage", "minrun"], conds=None, seeds=None, budget=1, quick=False, sample_seed=0,
        local_frac=0.5, baseline=None, baseline_hook="mqf.decode:fusion_baseline", duration_dir=None, official=None,
        jobs=a.jobs, tag="stoppage48grid", out=a.out, no_buckets=True))


if __name__ == "__main__":
    main()
