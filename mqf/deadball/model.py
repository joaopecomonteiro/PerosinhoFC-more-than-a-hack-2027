"""P(ball dead) per frame, learned on other games.

    python -m mqf.deadball.model train --tag v1     # -> models/deadball_v1.joblib

Training games (splits.json "train"): IDSSE 5 + Metrica 2, each twice -- the full ATD noise model
(46 anonymous slots, bursty ball loss, dropouts, duplicates) and player dropouts only -- and PFF 56
as they are (broadcast tracking that loses the ball in stoppages on its own) with player dropouts.
How a tracker treats the ball in a stoppage is a property of the tracker, so for IDSSE / Metrica it
is RANDOMISED per game copy: the ball is removed for a whole stoppage with probability p ~ U(0.3, 1),
else for a random stretch of it. Frames every 0.2 s inside the halves; label = dead or not; every
source weighs the same in total. The saved file also holds the lengths of the real stoppages and
stretches of play, for the decoder.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import replace

import numpy as np
import pandas as pd

from ..config import HARNESS_DIR, MODELS_DIR
from . import features

SAMPLE_HZ = 5.0


def hide_stoppage_ball(b: np.ndarray, gt: np.ndarray, p_whole: float, rng) -> np.ndarray:
    """Remove the ball in true stoppages: the whole stoppage with probability p_whole, else a random
    stretch of it (a tracker sometimes keeps a placed or held ball)."""
    out = b.copy()
    d = np.diff(np.r_[0, (gt == "D").astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if rng.random() < p_whole:
            out[s:e] = np.nan
        else:
            n = e - s
            a = s + int(rng.integers(0, max(1, n // 2)))
            out[a:min(e, a + int(rng.integers(1, max(2, n))))] = np.nan
    return out


def drift_ball_loss(b: np.ndarray, gt: np.ndarray, fps: float, rng, p_whole: float, lead_p: float,
                    live_per_min: float, lead_max_s: float = 15.0, lag_p: float = 0.3, lag_max_s: float = 2.0,
                    live_gap_s=(5.0, 40.0)) -> np.ndarray:
    """Ball loss that does NOT line up with the stoppages, as a broadcast-like or tactical tracker does.

    hide_stoppage_ball first; then a hidden stoppage's gap starts up to lead_max_s EARLIER (in the play
    leading into it) with probability lead_p and lasts up to lag_max_s past the restart with probability
    lag_p; and long gaps also occur in open play (live_per_min per minute of play, lengths in
    live_gap_s). So "the ball is missing" or "a gap just started" is never the answer on its own -- the
    players and the restart have to say it. Every number is an assumption, randomised in training.
    """
    out = hide_stoppage_ball(b, gt, p_whole, rng)
    n = len(gt)
    dead = gt == "D"
    d = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if not np.isnan(out[s:e, 0]).all():
            continue                                     # only stoppages whose ball is gone entirely
        if rng.random() < lead_p:
            out[max(0, s - int(rng.uniform(0, lead_max_s) * fps)):s] = np.nan
        if rng.random() < lag_p:
            out[e:min(n, e + int(rng.uniform(0, lag_max_s) * fps))] = np.nan
    alive_idx = np.flatnonzero((gt == "H") | (gt == "A"))
    k = rng.poisson(live_per_min * len(alive_idx) / fps / 60.0)
    for a in rng.choice(alive_idx, size=min(k, len(alive_idx)), replace=False) if len(alive_idx) else []:
        out[a:min(n, a + int(rng.uniform(*live_gap_s) * fps))] = np.nan
    return out


def with_ball(trk, b):
    from .. import atd
    return atd.Tracking(trk.players, trk.player_speed, b.astype(np.float32), trk.team_of_slot)


def training_games(seed: int = 0, drift: bool = False, kept_share: float = 0.0):
    """(source, copy, spec, tracking, gt) for every training game copy. drift: drift_ball_loss (v2)."""
    from ..external import bundle, splits
    from ..external.evaluate import load, parse_noise
    for source in ("idsse", "metrica"):
        for i, d in enumerate(bundle.find(source, splits.games("train", sources=[source]))):
            for copy in ("atd_noise", "dropouts"):
                rng = np.random.default_rng([seed, i, zlib.crc32(f"{source}{copy}".encode())])
                if copy == "atd_noise":
                    spec, trk, gt = load(d, noise_cfg=replace(parse_noise([]), seed=1000 + 100 * seed + i))
                else:
                    spec, trk, gt = load(d, players_drop=True, seed=1000 + i)
                if drift and rng.random() < kept_share:  # v4: the ball stays visible through the stoppages
                    b = trk.ball.astype(np.float64)
                elif drift:                              # v2: gaps that do not line up with stoppages
                    b = drift_ball_loss(trk.ball.astype(np.float64), gt, spec.fps, rng,
                                        p_whole=float(rng.uniform(0.3, 1.0)), lead_p=float(rng.uniform(0.2, 0.8)),
                                        live_per_min=float(rng.uniform(0.0, 1.0)))
                else:
                    b = hide_stoppage_ball(trk.ball.astype(np.float64), gt, float(rng.uniform(0.3, 1.0)), rng)
                yield source, copy, spec, with_ball(trk, b), gt
    for i, d in enumerate(bundle.find("pff", splits.games("train", sources=["pff"]))):
        spec, trk, gt = load(d, players_drop=True, seed=2000 + i)
        yield "pff", "natural", spec, trk, gt


def run_lengths(gt: np.ndarray, halves, fps: float) -> dict:
    """Lengths (s) of full stoppages and full stretches of play inside the halves (edge runs dropped)."""
    out = {"dead": [], "alive": []}
    for _, a, b in halves:
        dd = (gt[a:b + 1] == "D").astype(np.int8)
        bounds = np.r_[0, np.flatnonzero(np.diff(dd)) + 1, len(dd)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if 0 < s and e < len(dd):
                out["dead" if dd[s] else "alive"].append((e - s) / fps)
    return out


def train(tag: str, iters: int = 400, seed: int = 0, drift: bool = False, kept_share: float = 0.0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    t0 = time.time()
    Xs, ys, srcs, names = [], [], [], []
    lengths = {"dead": [], "alive": []}
    for source, copy, spec, trk, gt in training_games(seed, drift, kept_share):
        X = features.build(trk.players, trk.team_of_slot, trk.ball, spec.fps, spec.pitch, spec.halves)
        k = max(1, int(round(spec.fps / SAMPLE_HZ)))
        idx = np.concatenate([np.arange(a, b + 1, k) for _, a, b in spec.halves])
        idx = idx[np.isin(gt[idx], ["H", "A", "D"])]
        Xs.append(X.iloc[idx].astype(np.float32))
        ys.append((gt[idx] == "D").astype(np.int8))
        srcs.append(np.full(len(idx), source))
        if copy != "dropouts":
            for key, v in run_lengths(gt, spec.halves, spec.fps).items():
                lengths[key] += v
        names.append(f"{spec.name}:{copy}")
        print(f"  {spec.name:<14} {copy:<9} {len(idx):>7,} frames  ({time.time() - t0:.0f} s)", flush=True)
    X = pd.concat(Xs, ignore_index=True)
    y, src = np.concatenate(ys), np.concatenate(srcs)
    del Xs
    w = np.ones(len(y))
    for s in np.unique(src):
        w[src == s] = len(y) / (len(np.unique(src)) * (src == s).sum())
    print(f"train {len(names)} game copies, {len(X):,} frames ({y.mean():.1%} dead), {X.shape[1]} inputs", flush=True)
    model = HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
                                           l2_regularization=1.0, early_stopping=False, random_state=seed)
    model.fit(X.to_numpy(), y, sample_weight=w)
    meta = {"tag": tag, "train": names, "frames": int(len(X)), "inputs": int(X.shape[1]),
            "prior": float(np.average(y, weights=w)), "iters": iters, "seed": seed, "drift": drift, "kept_share": kept_share,
            "n_stoppages": len(lengths["dead"]), "n_play": len(lengths["alive"])}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"deadball_{tag}.joblib"
    joblib.dump({"model": model, "columns": list(X.columns), "meta": meta,
                 "lengths": {k: np.asarray(v, dtype=np.float64) for k, v in lengths.items()}}, path)
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    (HARNESS_DIR / f"deadball_train_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path}  ({time.time() - t0:.0f} s)", flush=True)
    return meta


def load(path):
    import joblib
    return joblib.load(path)


def predict(m: dict, trk, spec) -> np.ndarray:
    """(n,) P(dead) inside the halves, NaN outside; rounded to 1e-4 (reproducible downstream)."""
    X = features.build(trk.players, trk.team_of_slot, trk.ball, spec.fps, spec.pitch, spec.halves)
    p = np.full(len(X), np.nan)
    inside = np.zeros(len(X), dtype=bool)
    for _, a, b in spec.halves:
        inside[a:b + 1] = True
    if inside.any():
        p[inside] = m["model"].predict_proba(X.loc[inside, m["columns"]].to_numpy(np.float32))[:, 1]
    return np.round(p, 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tag", default="v1")
    t.add_argument("--iters", type=int, default=400)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--drift", action="store_true", help="ball loss that does not line up with stoppages")
    t.add_argument("--kept-share", type=float, default=0.0,
                   help="with --drift: share of IDSSE / Metrica copies whose ball stays visible in stoppages")
    a = ap.parse_args(argv)
    train(a.tag, a.iters, a.seed, a.drift, a.kept_share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
