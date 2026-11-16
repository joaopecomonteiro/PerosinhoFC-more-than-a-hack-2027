"""Training rows (5 Hz) of the dead-ball models, one bundle game at a time, cached.

Two views per game (like dead_v1's clean + dropout copy):
  0 delivered   players as delivered
  1 hard        ATD-like player dropouts (mqf.ball.data._player_mask) and 40 % of each half's ball removed
                in 0.2-30 s spans placed WITHOUT looking at the labels
In BOTH views the ball's missingness is first made label-independent (balance_ball): where a source
loses the ball more often on dead than on alive frames (PFF's broadcast tracker: seen on ~14 % of dead
vs ~73 % of alive frames), spans of the better-seen class are removed until both are seen equally
often. The models can then learn what a TRACKED ball says (out, still on a spot, kicked) but not what its
absence says -- that habit belongs to a tracking system (feats.py module doc).

Rows: every 1 / SAMPLE_HZ s inside the halves -- the full_rules table (feats.py), label 1 dead / 0 alive /
-1 unlabelled, the view and the half. The rows of one (view, half) in order ARE the 5 Hz sequence the
TCN / ASFormer read, so one cache serves every model.

Cache: external/cache/alive/<game>_<key>.npz (other games only, gitignored with external/). The key
hashes the code of dead.py, feats.py, rules.py, deadball/features.py and this file plus the seed and
the rule settings.
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
CACHE_DIR = EXTERNAL_DIR / "cache" / "alive"
VIEWS = (False, True)                     # hard?
HIDE_SHARE = 0.40
HIDE_SPAN_S = (0.2, 30.0)


def code_hash() -> str:
    from .. import dead
    from ..deadball import features as dfeat
    h = hashlib.sha1()
    for mod in (dead, feats, rules, dfeat):
        h.update(Path(mod.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:10]


def _key(seed: int, rule_cfg: rules.RuleConfig) -> str:
    blob = json.dumps({"code": code_hash(), "seed": seed, "rule": asdict(rule_cfg), "hz": SAMPLE_HZ}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def step_of(fps: float) -> int:
    return max(1, int(round(fps / SAMPLE_HZ)))


def _span(rng, fps) -> int:
    lo, hi = np.log(HIDE_SPAN_S[0]), np.log(HIDE_SPAN_S[1])
    return max(1, int(np.exp(rng.uniform(lo, hi)) * fps))


def balance_ball(ball: np.ndarray, labels: np.ndarray, halves, fps: float, rng, tol: float = 0.02) -> np.ndarray:
    """Remove spans of the better-seen class (alive or dead) until both are seen equally often, per half.
    Only frames of that class are removed, so the other class's rate does not move."""
    out = np.array(ball, dtype=np.float64, copy=True)
    lab = np.asarray(labels)
    for _, a, b in halves:
        sl = slice(a, b + 1)
        seen = ~np.isnan(out[sl, 0])
        cls = {"alive": np.isin(lab[sl], ["H", "A"]), "dead": lab[sl] == "D"}
        if cls["alive"].sum() == 0 or cls["dead"].sum() == 0:
            continue
        rate = {k: (seen & m).sum() / m.sum() for k, m in cls.items()}
        hi = max(rate, key=rate.get)
        lo_rate = min(rate.values())
        if rate[hi] - lo_rate <= tol:
            continue
        m = cls[hi]
        target_seen = int(lo_rate * m.sum())
        cand = np.flatnonzero(m & seen)
        guard = 0
        while (seen & m).sum() > target_seen and len(cand) and guard < 100_000:
            s = int(cand[rng.integers(0, len(cand))])
            e = min(b - a + 1, s + _span(rng, fps))
            hide = np.zeros(b - a + 1, dtype=bool)
            hide[s:e] = True
            hide &= m
            out[a + np.flatnonzero(hide)] = np.nan
            seen[hide] = False
            cand = np.flatnonzero(m & seen)
            guard += 1
    return out


def random_spans(ball: np.ndarray, halves, fps: float, rng, share: float = HIDE_SHARE) -> np.ndarray:
    """Remove the ball in 0.2-30 s spans at random places until `share` of each half is hidden (labels unused)."""
    out = np.array(ball, dtype=np.float64, copy=True)
    for _, a, b in halves:
        n = b - a + 1
        hidden = 0
        while hidden < share * n:
            ln = _span(rng, fps)
            s = a + int(rng.integers(0, max(1, n - ln)))
            out[s:s + ln] = np.nan
            hidden += ln
    return out


def view_inputs(g, rng, hard: bool):
    """(players, tracked ball) of one view."""
    from ..ball import data as bdata
    ball = balance_ball(g.ball, g.labels, g.halves, g.fps, rng)
    players = g.players
    if hard:
        players = bdata._player_mask(players.astype(np.float64), rng, g.fps).astype(np.float32)
        ball = random_spans(ball, g.halves, g.fps, rng)
    return players, ball


def game_tables(g, seed: int = 0, rule_cfg: rules.RuleConfig | None = None, use_cache: bool = True,
                cache_dir: Path | None = None) -> dict:
    """{"X": (R, F) float32, "y": (R,) int8 (1 dead, 0 alive, -1 unlabelled), "view", "half": (R,) int8,
        "frame": (R,) int32, "columns": [...]} -- rows of each (view, half) in frame order."""
    rule_cfg = rule_cfg or rules.RuleConfig()
    cache_dir = Path(cache_dir or CACHE_DIR)
    path = cache_dir / f"{g.name}_{_key(seed, rule_cfg)}.npz"
    if use_cache and path.exists():
        z = np.load(path, allow_pickle=False)
        out = {k: z[k] for k in z.files if k != "columns"}
        out["columns"] = [str(c) for c in z["columns"]]
        return out
    lab = np.asarray(g.labels)
    k = step_of(g.fps)
    Xs, ys, vs, hs, fs = [], [], [], [], []
    cols = None
    for v, hard in enumerate(VIEWS):
        rng = np.random.default_rng([seed, zlib.crc32(g.name.encode()), int(hard), 7])
        players, ball = view_inputs(g, rng, hard)
        X = feats.features(players, g.team, ball, g.fps, g.pitch, g.halves, "full_rules", rule_cfg)
        cols = list(X.columns)
        for h, (_, a, b) in enumerate(g.halves):
            idx = np.arange(a, b + 1, k)
            Xs.append(X.iloc[idx].to_numpy(dtype=np.float32))
            ly = lab[idx]
            ys.append(np.where(ly == "D", 1, np.where(np.isin(ly, ["H", "A"]), 0, -1)).astype(np.int8))
            vs.append(np.full(len(idx), v, dtype=np.int8))
            hs.append(np.full(len(idx), h, dtype=np.int8))
            fs.append(idx.astype(np.int32))
    out = {"X": np.concatenate(Xs), "y": np.concatenate(ys), "view": np.concatenate(vs), "half": np.concatenate(hs),
           "frame": np.concatenate(fs), "columns": cols}
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{kk: (np.array(vv) if isinstance(vv, list) else vv) for kk, vv in out.items()})
        tmp.replace(path)
    return out


def load_tables(part: str, sources, seed: int = 0, rule_cfg=None, use_cache: bool = True, log=print,
                games=None) -> list[tuple[str, str, dict]]:
    """[(game name, source, tables)] for one split part; `games`: Game objects instead of the bundles (tests)."""
    from .. import dead
    out = []
    t0 = time.time()
    it = games if games is not None else dead.load_games(part, sources, log or (lambda m: None))
    for g in it:
        out.append((g.name, g.source, game_tables(g, seed, rule_cfg, use_cache)))
        del g
    if log:
        log(f"  {len(out)} games' tables ready in {time.time() - t0:.0f} s")
    return out


def stack_rows(tables, columns: list[str], subsample: int = 1):
    """(X, y, game id) over the labelled rows of every game, restricted to `columns`."""
    Xs, ys, gs = [], [], []
    for gi, (_, _, t) in enumerate(tables):
        pos = [t["columns"].index(c) for c in columns]
        ok = np.flatnonzero(t["y"] >= 0)[::subsample]
        Xs.append(t["X"][ok][:, pos])
        ys.append(t["y"][ok])
        gs.append(np.full(len(ok), gi, dtype=np.int32))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def sequences(tables, columns: list[str]):
    """[(S (T, F), y (T,))] per game x view x half, in frame order."""
    out = []
    for _, _, t in tables:
        pos = [t["columns"].index(c) for c in columns]
        key = t["view"].astype(np.int64) * 16 + t["half"]
        change = np.flatnonzero(key[1:] != key[:-1]) + 1
        for a, b in zip(np.r_[0, change], np.r_[change, len(key)]):
            out.append((t["X"][a:b][:, pos], t["y"][a:b]))
    return out
