"""Training rows and sequences for the owner models, one bundle game at a time, cached.

Per game, four views like who.py: the game as delivered and "hard" (ATD-like player dropouts, no
ball on dead frames, 40 % of each half's alive ball hidden in 0.2-30 s spans), each in both team
orders (home / away is arbitrary; the label flips with them). For every view:

  rows   the full_rules_heading table (feats.py) on alive frames every 1 / SAMPLE_HZ s, label 1 = home.
         The `who`, `full`, `full_rules` and `full_heading` sets are column subsets of it, so one cache
         serves every model (the heading columns were added on 15 Nov: older sets never read them).
  seq    every 1 / SAMPLE_HZ s over each whole half: the full_rules_heading table + the nearest players
         (feats.seq_inputs), label 1 home / 0 away / -1 dead. The GRU reads its per-frame subset, the
         TCN the full columns.

Cache: external/cache/owner/<game>_<key>.npz (other games only, gitignored with external/). The key
hashes the code of who.py, feats.py, rules.py and this file plus the seed and the rule settings, so a
code change rebuilds it.
"""
from __future__ import annotations

import hashlib
import json
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import EXTERNAL_DIR
from . import feats, rules

SAMPLE_HZ = 5.0
CACHE_DIR = EXTERNAL_DIR / "cache" / "owner"
VIEWS = ((False, False), (False, True), (True, False), (True, True))     # (hard, swap)


def code_hash() -> str:
    from .. import who
    h = hashlib.sha1()
    for mod in (who, feats, rules):
        h.update(Path(mod.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:10]


def _key(seed: int, rule_cfg: rules.RuleConfig) -> str:
    blob = json.dumps({"code": code_hash(), "seed": seed, "rule": asdict(rule_cfg), "hz": SAMPLE_HZ}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def step_of(fps: float) -> int:
    return max(1, int(round(fps / SAMPLE_HZ)))


def view_inputs(g, rng, hard: bool):
    """(players, ball) of one view; the 'hard' masks as in who._game_rows."""
    from .. import who
    from ..ball import data as bdata
    players, ball = g.players, g.ball
    if hard:
        players = bdata._player_mask(players.astype(np.float64), rng, g.fps).astype(np.float32)
        ball = who.hide_ball(ball, np.asarray(g.labels), g.halves, g.fps, rng)
    return players, ball


def game_tables(g, seed: int = 0, rule_cfg: rules.RuleConfig | None = None,
                use_cache: bool = True, cache_dir: Path | None = None) -> dict:
    """{"X": (R, F) float32, "y": (R,) int8, "view": (R,) int8, "frame": (R,) int32, "columns": [...],
        "S": (T, Fs) float32, "sy": (T,) int8, "sview": (T,) int8, "shalf": (T,) int8, "seq_columns": [...]}"""
    rule_cfg = rule_cfg or rules.RuleConfig()
    cache_dir = Path(cache_dir or CACHE_DIR)
    path = cache_dir / f"{g.name}_{_key(seed, rule_cfg)}.npz"
    if use_cache and path.exists():
        z = np.load(path, allow_pickle=False)
        out = {k: z[k] for k in z.files if k not in ("columns", "seq_columns")}
        out["columns"] = [str(c) for c in z["columns"]]
        out["seq_columns"] = [str(c) for c in z["seq_columns"]]
        return out
    lab = np.asarray(g.labels)
    k = step_of(g.fps)
    idx_all = np.concatenate([np.arange(a, b + 1, k) for _, a, b in g.halves])
    half_of = np.concatenate([np.full(len(range(a, b + 1, k)), h, dtype=np.int8) for h, (_, a, b) in enumerate(g.halves)])
    alive_idx = idx_all[np.isin(lab[idx_all], ["H", "A"])]
    Xs, ys, vs, fs, Ss, sys_, svs, shs = [], [], [], [], [], [], [], []
    cols = seq_cols = None
    for v, (hard, swap) in enumerate(VIEWS):
        rng = np.random.default_rng([seed, zlib.crc32(g.name.encode()), int(hard)])   # per game, any game set
        players, ball = view_inputs(g, rng, hard)
        team = (1 - g.team) if swap else g.team
        X = feats.features(players, team, ball, g.fps, g.pitch, g.halves, "full_rules_heading", rule_cfg)
        cols = list(X.columns)
        home = "A" if swap else "H"
        Xs.append(X.iloc[alive_idx].to_numpy(dtype=np.float32))
        ys.append((lab[alive_idx] == home).astype(np.int8))
        vs.append(np.full(len(alive_idx), v, dtype=np.int8))
        fs.append(alive_idx.astype(np.int32))
        S, seq_cols = feats.seq_inputs(players, team, ball, g.fps, g.pitch, g.halves, X_full=X, columns="all")
        Ss.append(S[idx_all])
        sl = lab[idx_all]
        sys_.append(np.where(np.isin(sl, ["H", "A"]), (sl == home).astype(np.int8), -1).astype(np.int8))
        svs.append(np.full(len(idx_all), v, dtype=np.int8))
        shs.append(half_of)
    out = {"X": np.concatenate(Xs), "y": np.concatenate(ys), "view": np.concatenate(vs), "frame": np.concatenate(fs),
           "columns": cols, "S": np.concatenate(Ss), "sy": np.concatenate(sys_), "sview": np.concatenate(svs),
           "shalf": np.concatenate(shs), "seq_columns": seq_cols}
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{k: (np.array(v) if isinstance(v, list) else v) for k, v in out.items()})
        tmp.replace(path)
    return out


def load_tables(part: str, sources, seed: int = 0, rule_cfg=None, use_cache: bool = True, log=print,
                games=None, keep_seq: bool = True) -> list[tuple[str, str, dict]]:
    """[(game name, source, tables)] for one split part. `games`: an iterable of Game objects to use
    instead of the bundles (tests). keep_seq=False drops the GRU sequences (rows-only models, less memory)."""
    from .. import who
    out = []
    t0 = time.time()
    it = games if games is not None else who.load_games(part, sources, log)
    for g in it:
        t = game_tables(g, seed, rule_cfg, use_cache)
        if not keep_seq:
            for k in ("S", "sy", "sview", "shalf"):
                t.pop(k, None)
        out.append((g.name, g.source, t))
        del g
    if log:
        log(f"  {len(out)} games' tables ready in {time.time() - t0:.0f} s")
    return out


def stack_rows(tables, columns: list[str], subsample: int = 1):
    """(X, y, game id per row) over games, restricted to `columns` (a subset of the cached table)."""
    Xs, ys, gs = [], [], []
    for gi, (_, _, t) in enumerate(tables):
        pos = [t["columns"].index(c) for c in columns]
        X = t["X"][::subsample][:, pos]
        Xs.append(X)
        ys.append(t["y"][::subsample])
        gs.append(np.full(len(X), gi, dtype=np.int32))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def columns_of(feature_set: str, all_columns: list[str]) -> list[str]:
    """The columns of a feature set, in the cached table's order."""
    if feature_set == "who":
        w = feats.who_columns()
        assert list(all_columns[:len(w)]) == w, "cached table does not start with who.py's columns"
        return w
    if feature_set not in feats.FEATURE_SETS:
        raise ValueError(feature_set)
    cols = [c for c in all_columns if feats.keep_column(c, feature_set)]
    if feats.has_heading(feature_set) and not set(feats.HEADING_COLS) <= set(cols):
        raise ValueError(f"{feature_set}: the table has no heading columns (built by older code?)")
    return cols
