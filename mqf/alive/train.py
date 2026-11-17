"""Train a dead-ball model on the training games of splits.json.

    python -m mqf.alive.train gbm --tag v1_impff --feature-set v1 --sources idsse metrica pff   # dead_v1 replica
    python -m mqf.alive.train gbm --tag full_impff --feature-set full --sources idsse metrica pff
    python -m mqf.alive.train gbm --tag rules_impff --feature-set full_rules --sources idsse metrica pff
    python -m mqf.alive.train mlp --tag full_impff --feature-set full --sources idsse metrica pff
    python -m mqf.alive.train tcn --tag full_impff_s0 --feature-set full --sources idsse metrica pff --seed 0
    python -m mqf.alive.train asf --tag full_impff_s0 --feature-set full --sources idsse metrica pff --seed 0

Writes models/alive_<kind>_<tag>.joblib (gbm) or .pt and harness_results/alive_train_<kind>_<tag>.json
(games, rows, settings, time). Per-game tables are cached (data.py); --table-seed keeps the same
tables for several init seeds.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from ..config import HARNESS_DIR, MODELS_DIR
from . import data, feats, models, rules

KINDS = models.KINDS


def model_path(kind: str, tag: str):
    return MODELS_DIR / f"alive_{kind}_{tag}.{'joblib' if kind == 'gbm' else 'pt'}"


def fit(kind: str, tables, tag: str, feature_set: str = "full", rule_cfg: rules.RuleConfig | None = None,
        iters: int = 300, epochs: int | None = None, subsample: int = 1, seed: int = 0, device: str = "auto",
        sources=(), log=print, table_seed: int | None = None, w_bnd: float = 1.0):
    """Train one model on loaded tables (data.load_tables) and save it. Returns (path, info)."""
    rule_cfg = rule_cfg or rules.RuleConfig()
    t0 = time.time()
    names = [n for n, _, _ in tables]
    cols = feats.columns_of(feature_set, tables[0][2]["columns"])
    meta = {"tag": tag, "kind": kind, "feature_set": feature_set, "sources": list(sources), "train_games": names,
            "seed": seed, "table_seed": seed if table_seed is None else table_seed, "code": data.code_hash()}
    rc = asdict(rule_cfg)
    if kind in models.SEQ_KINDS:
        blocks = data.sequences(tables, cols)
        meta["rows"] = int(sum((y >= 0).sum() for _, y in blocks))
        m = models.train_seq(kind, blocks, cols, feature_set, meta, epochs=epochs or 20, seed=seed, device=device,
                             w_bnd=w_bnd, rule_cfg=rc, log=log)
        del blocks
    else:
        X, y, _ = data.stack_rows(tables, cols, subsample)
        meta["rows"] = int(len(X))
        log(f"  {kind} on {len(names)} games, {len(X):,} rows ({y.mean():.1%} dead), {len(cols)} features")
        if kind == "gbm":
            m = models.train_gbm(X, y, cols, feature_set, meta, iters=iters, seed=seed, rule_cfg=rc, log=log)
        else:
            m = models.train_mlp(X, y, cols, feature_set, meta, epochs=epochs or 4, seed=seed, device=device,
                                 rule_cfg=rc, log=log)
        del X, y
    meta["train_s"] = round(time.time() - t0, 1)
    path = m.save(model_path(kind, tag))
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    info = {**meta, "model": path.name, "n_columns": len(cols), "rule_cfg": rc,
            "iters": iters if kind == "gbm" else None, "epochs": epochs}
    (HARNESS_DIR / f"alive_train_{kind}_{tag}.json").write_text(json.dumps(info, indent=2))
    log(f"  wrote {path} ({meta['train_s']:.0f} s)")
    return path, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feature-set", default="full", choices=feats.FEATURE_SETS)
    ap.add_argument("--sources", nargs="*", default=["idsse", "metrica", "pff"])
    ap.add_argument("--rule-cfg", nargs="*", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--subsample", type=int, default=1, help="keep every k-th training row (gbm / mlp)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table-seed", type=int, help="seed of the (cached) training tables; default: --seed")
    ap.add_argument("--w-bnd", type=float, default=1.0, help="boundary loss weight (tcn / asf)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    rc = rules.parse_rule_cfg(a.rule_cfg)
    log = lambda m: print(m, flush=True)
    ts = a.seed if a.table_seed is None else a.table_seed
    tables = data.load_tables("train", a.sources, ts, rc, use_cache=not a.no_cache, log=log)
    if not tables:
        raise SystemExit("no training bundles: python tasks.py bundles")
    fit(a.kind, tables, a.tag, a.feature_set, rc, a.iters, a.epochs, a.subsample, a.seed, a.device, a.sources, log,
        table_seed=ts, w_bnd=a.w_bnd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
