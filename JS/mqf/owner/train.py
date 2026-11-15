"""Train an owner model on the training games of splits.json.

    python -m mqf.owner.train gbm --tag full_im --feature-set full --sources idsse metrica
    python -m mqf.owner.train gbm --tag stack_im --feature-set full_rules --sources idsse metrica
    python -m mqf.owner.train gbm --tag full_all --feature-set full --sources idsse metrica skillcorner pff
    python -m mqf.owner.train mlp --tag full_im --feature-set full --sources idsse metrica
    python -m mqf.owner.train tcn --tag im --sources idsse metrica          # TCN + boundary head (full set)
    python -m mqf.owner.train gru --tag im --sources idsse metrica
    python -m mqf.owner.train tcn --tag full_im_s1 --sources idsse metrica --seed 1 --table-seed 0   # another init, same tables

Writes models/owner_<kind>_<tag>.joblib (gbm) or .pt (mlp, gru, tcn) and harness_results/owner_train_<kind>_<tag>.json
(games, rows, settings, time). Per-game tables are cached (data.py): the first run on a source
builds them, the next ones only read them.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from ..config import HARNESS_DIR, MODELS_DIR
from . import data, feats, models, rules

KINDS = ("gbm", "mlp", "gru", "tcn")


def model_path(kind: str, tag: str):
    return MODELS_DIR / f"owner_{kind}_{tag}.{'joblib' if kind == 'gbm' else 'pt'}"


def fit(kind: str, tables, tag: str, feature_set: str = "full", rule_cfg: rules.RuleConfig | None = None,
        iters: int = 300, epochs: int | None = None, subsample: int = 1, seed: int = 0, device: str = "auto",
        sources=(), log=print, table_seed: int | None = None):
    """Train one model on loaded tables (data.load_tables) and save it. Returns (path, info)."""
    rule_cfg = rule_cfg or rules.RuleConfig()
    t0 = time.time()
    names = [n for n, _, _ in tables]
    meta = {"tag": tag, "kind": kind, "feature_set": feature_set if kind != "gru" else "seq",
            "sources": list(sources), "train_games": names, "seed": seed,
            "table_seed": seed if table_seed is None else table_seed, "code": data.code_hash()}
    if kind in ("gru", "tcn"):
        stored = list(tables[0][2]["seq_columns"])
        if kind == "gru":
            cols = feats.gru_columns(stored)
            m = models.train_gru(tables, cols, "seq", asdict(rule_cfg), meta, epochs=epochs or 8, seed=seed,
                                 device=device, log=log)
        else:
            cols = feats.tcn_columns(stored, feature_set)
            m = models.train_tcn(tables, cols, feature_set, asdict(rule_cfg), meta, epochs=epochs or 20, seed=seed,
                                 device=device, log=log)
        meta["rows"] = int(sum((t["sy"] >= 0).sum() for _, _, t in tables))
    else:
        cols = data.columns_of(feature_set, tables[0][2]["columns"])
        X, y, _ = data.stack_rows(tables, cols, subsample)
        meta["rows"] = int(len(X))
        log(f"  {kind} on {len(names)} games, {len(X):,} rows ({y.mean():.1%} home), {len(cols)} features")
        if kind == "gbm":
            m = models.train_gbm(X, y, cols, feature_set, asdict(rule_cfg), meta, iters=iters, seed=seed, log=log)
        else:
            m = models.train_mlp(X, y, cols, feature_set, asdict(rule_cfg), meta, epochs=epochs or 4, seed=seed,
                                 device=device, log=log)
        del X, y
    meta["train_s"] = round(time.time() - t0, 1)
    path = m.save(model_path(kind, tag))
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    info = {**meta, "model": path.name, "n_columns": len(cols), "rule_cfg": asdict(rule_cfg),
            "iters": iters if kind == "gbm" else None, "epochs": epochs}
    (HARNESS_DIR / f"owner_train_{kind}_{tag}.json").write_text(json.dumps(info, indent=2))
    log(f"  wrote {path} ({meta['train_s']:.0f} s)")
    return path, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feature-set", default="full", choices=("who", "full", "full_rules"),
                    help="gbm / mlp / tcn columns (gru always reads the per-frame sequence inputs)")
    ap.add_argument("--sources", nargs="*", default=["idsse", "metrica"])
    ap.add_argument("--rule-cfg", nargs="*", default=[], metavar="FIELD=VALUE", help="RuleConfig overrides (rule columns)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--subsample", type=int, default=1, help="keep every k-th training row (gbm / mlp)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table-seed", type=int, help="seed of the (cached) training tables' masks; default: --seed")
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
        table_seed=ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
