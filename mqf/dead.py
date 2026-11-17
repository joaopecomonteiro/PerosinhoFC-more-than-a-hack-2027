"""Is the ball dead? A per-frame probability from the PLAYERS, learned on other games.

    python -m mqf.dead train --tag v1                       # IDSSE + Metrica + PFF training games -> models/dead_v1.joblib
    python -m mqf.dead train --tag v1 --sources idsse metrica
    python -m mqf.dead eval --model models/dead_v1.joblib   # per-frame metrics on the validation games

Why players only (v1): on BAR-ATH the ball is missing on 41 % of graded frames, mostly while play
is stopped -- but whether a missing ball MEANS a stoppage depends on the tracking system (PFF's
broadcast loses it in stoppages, DFL keeps it), so a model would learn a source habit we cannot
check on our match. The ball enters the decoder elsewhere (who has it; seen off the pitch = dead).

Features, identical for every source (BAR-ATH included), from positions only:
  speed       per player, |x(t+0.5 s) - x(t-0.5 s)| / 1 s -- not the source's speed field, so tracker
              jitter and smoothing differences do not matter (a 1 s baseline keeps jitter small)
  per frame   mean / median / 80th percentile speed; share of players < 1 m/s and > 3 m/s; each
              team's mean speed and spread; distance between the team centres; players in the
              penalty boxes, within 3 m of a touchline, off the pitch; players tracked
  context     the mean speed smoothed over 1 / 3 / 10 / 30 s, its 5 s minimum, the trend (next 5 s
              minus last 5 s), its rank within the half (each match judged against itself), and 3 s
              means of the formation features. Centred windows: labels are made offline, the whole
              match is known (as for the submission).
Model: sklearn HistGradientBoostingClassifier (deterministic), trained on frames every 0.2 s of the
training games of splits.json, plus a copy with ATD-like player dropouts (ball.data._player_mask).
Labels: D vs H/A inside the halves of each bundle (source rules differ -- see TECHNICAL.md).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HARNESS_DIR, MODELS_DIR

SPEED_WINDOW_S = 1.0
SAMPLE_HZ = 5.0
MOVE_SLOW, MOVE_FAST = 1.0, 3.0
VERSION = 1


def player_speed(xy: np.ndarray, fps: float, window_s: float = SPEED_WINDOW_S) -> np.ndarray:
    """(n, P) m/s from positions, centred over window_s; NaN where an end point or the frame is missing."""
    w = max(1, int(round(window_s * fps / 2)))
    n = len(xy)
    out = np.full(xy.shape[:2], np.nan, dtype=np.float64)
    if n > 2 * w:
        out[w:-w] = np.linalg.norm(xy[2 * w:] - xy[:-2 * w], axis=-1) * fps / (2 * w)
    out[np.isnan(xy[..., 0])] = np.nan
    return out


def _nan(fn, a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def frame_features(players: np.ndarray, team: np.ndarray, fps: float, pitch) -> pd.DataFrame:
    """Per-frame, before the time context."""
    L, W = pitch
    xy = players.astype(np.float64)
    sp = player_speed(xy, fps)
    present = ~np.isnan(xy[..., 0])
    n_pres = present.sum(axis=1).astype(np.float64)
    has_sp = ~np.isnan(sp)
    n_sp = np.maximum(has_sp.sum(axis=1), 1)
    x, y = xy[..., 0], xy[..., 1]
    f = {
        "speed_mean": _nan(np.nanmean, sp), "speed_median": _nan(np.nanmedian, sp),
        "speed_p80": _nan(lambda a, axis: np.nanpercentile(a, 80, axis=axis), sp),
        "share_slow": np.where(has_sp.any(axis=1), (has_sp & (sp < MOVE_SLOW)).sum(axis=1) / n_sp, np.nan),
        "share_fast": np.where(has_sp.any(axis=1), (has_sp & (sp > MOVE_FAST)).sum(axis=1) / n_sp, np.nan),
        "n_players": n_pres,
    }
    cx = {}
    for tm, name in ((0, "home"), (1, "away")):
        m = team == tm
        f[f"speed_{name}"] = _nan(np.nanmean, sp[:, m])
        f[f"spread_{name}"] = _nan(np.nanstd, x[:, m]) + _nan(np.nanstd, y[:, m])
        cx[name] = np.stack([_nan(np.nanmean, x[:, m]), _nan(np.nanmean, y[:, m])], axis=1)
    f["centre_gap"] = np.linalg.norm(cx["home"] - cx["away"], axis=1)
    in_box = present & (np.abs(y - W / 2) < 20.16) & ((x < 16.5) | (x > L - 16.5))
    f["n_in_boxes"] = in_box.sum(axis=1).astype(np.float64)
    near_line = present & ((y < 3.0) | (y > W - 3.0))
    f["n_near_touchline"] = near_line.sum(axis=1).astype(np.float64)
    off = present & ((x < -1) | (x > L + 1) | (y < -1) | (y > W + 1))
    f["n_off_pitch"] = off.sum(axis=1).astype(np.float64)
    df = pd.DataFrame(f)
    df.loc[n_pres == 0] = np.nan
    return df


def context_features(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Centred rolling context inside one half (call per half)."""
    out = {}
    s = df["speed_mean"]

    def roll(series, sec, fn="mean"):
        w = max(1, int(round(sec * fps)))
        r = series.rolling(w, center=True, min_periods=1)
        return getattr(r, fn)()
    for sec in (1, 3, 10, 30):
        out[f"speed_mean_{sec}s"] = roll(s, sec)
    out["speed_min_5s"] = roll(s, 5, "min")
    out["speed_std_3s"] = roll(s, 3, "std")
    w5 = max(1, int(round(5 * fps)))
    past = s.rolling(w5, min_periods=1).mean()
    future = s[::-1].rolling(w5, min_periods=1).mean()[::-1]
    out["speed_trend_5s"] = future - past
    out["speed_rank_10s"] = out["speed_mean_10s"].rank(pct=True)
    for c in ("share_slow", "share_fast", "speed_home", "speed_away", "spread_home", "spread_away", "centre_gap",
              "n_in_boxes", "n_near_touchline", "n_off_pitch", "n_players", "speed_p80"):
        out[f"{c}_3s"] = roll(df[c], 3)
    out["share_slow_10s"] = roll(df["share_slow"], 10)
    out["centre_gap_10s"] = roll(df["centre_gap"], 10)
    return pd.DataFrame(out, index=df.index)


def features(players: np.ndarray, team: np.ndarray, fps: float, pitch, halves) -> pd.DataFrame:
    """(n_frames, F) -- NaN outside the halves."""
    base = frame_features(players, team, fps, pitch)
    cols = None
    out = None
    for _, a, b in halves:
        seg = base.iloc[a:b + 1].reset_index(drop=True)
        ctx = context_features(seg, fps)
        full = pd.concat([seg, ctx], axis=1)
        if out is None:
            cols = list(full.columns)
            out = np.full((len(base), len(cols)), np.nan, dtype=np.float64)
        out[a:b + 1] = full[cols].to_numpy(dtype=np.float64)
    return pd.DataFrame(out, columns=cols)


# ------------------------------------------------------------------------------------ model

def load_model(path):
    import joblib
    obj = joblib.load(path)
    return obj["model"], obj["columns"], obj.get("meta", {})


def predict(model_bundle, players, team, fps, pitch, halves) -> np.ndarray:
    """(n,) P(dead), NaN outside the halves; rounded to 1e-4 (reproducible downstream)."""
    model, cols, _ = model_bundle
    X = features(players, team, fps, pitch, halves)
    p = np.full(len(X), np.nan)
    inside = np.zeros(len(X), dtype=bool)
    for _, a, b in halves:
        inside[a:b + 1] = True
    if inside.any():
        p[inside] = model.predict_proba(X.loc[inside, cols].to_numpy(dtype=np.float32))[:, 1]   # as in training
    return np.round(p, 4)


def v0_p_dead(X: pd.DataFrame, still_speed=1.3, width=0.35) -> np.ndarray:
    """v0's rule on the same per-frame speed, for comparison only."""
    ms = X["speed_mean"].to_numpy()
    return 1 / (1 + np.exp(-(still_speed - np.nan_to_num(ms, nan=still_speed)) / width))


def _game_rows(g, rng, drop: bool):
    """Features + labels of one bundle game, sampled every 1 / SAMPLE_HZ s inside the halves."""
    from .ball import data as bdata
    players = g.players
    if drop:
        players = bdata._player_mask(players.astype(np.float64), rng, g.fps).astype(np.float32)
    X = features(players, g.team, g.fps, g.pitch, g.halves)
    lab = np.asarray(g.labels)
    k = max(1, int(round(g.fps / SAMPLE_HZ)))
    idx = []
    for _, a, b in g.halves:
        idx.append(np.arange(a, b + 1, k))
    idx = np.concatenate(idx)
    idx = idx[np.isin(lab[idx], ["H", "A", "D"])]
    return X.iloc[idx], (lab[idx] == "D").astype(np.int8), idx


def _metrics(y, p, thr=0.5) -> dict:
    from sklearn.metrics import roc_auc_score
    pred = p >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
            "f1_dead": round(f1, 4), "acc": round(float((pred == (y == 1)).mean()), 4),
            "share_dead_true": round(float(y.mean()), 3), "share_dead_pred": round(float(pred.mean()), 3)}


def load_games(part, sources, log=print):
    """Bundle games of one split part, ONE AT A TIME (a generator: 63 full-rate games would not fit)."""
    from .ball import data as bdata
    from .external import bundle, splits
    names = splits.games(part, sources=list(sources))
    for k, d in enumerate(bundle.find(names=names)):
        g = bdata.load_game(d)
        log(f"  {g.name} ({k + 1}/{len(names)})")
        yield g


def train(args) -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    t0 = time.time()
    log = lambda m: print(m, flush=True)
    Xs, ys, names = [], [], []
    for i, g in enumerate(load_games("train", args.sources, log)):
        names.append(g.name)
        for drop in ([False, True] if args.dropout_copy else [False]):
            X, y, _ = _game_rows(g, np.random.default_rng([args.seed, i, int(drop)]), drop)
            Xs.append(X.astype(np.float32))
            ys.append(y)
        del g
    if not Xs:
        raise SystemExit("no training bundles: python tasks.py bundles")
    X = pd.concat(Xs, ignore_index=True)
    del Xs
    y = np.concatenate(ys)
    cols = list(X.columns)
    log(f"train {len(names)} games, {len(X):,} frames ({y.mean():.1%} dead), {len(cols)} features, "
        f"built in {time.time() - t0:.0f} s")
    model = HistGradientBoostingClassifier(max_iter=args.iters, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
                                           random_state=args.seed)
    model.fit(X.to_numpy(), y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"dead_{args.tag}.joblib"
    meta = {"version": VERSION, "tag": args.tag, "sources": list(args.sources), "train_games": names,
            "frames": int(len(X)), "dropout_copy": bool(args.dropout_copy), "iters": args.iters}
    joblib.dump({"model": model, "columns": cols, "meta": meta}, path)
    log(f"wrote {path}  ({time.time() - t0:.0f} s)")
    return evaluate(argparse.Namespace(model=path, sources=args.val_sources, seed=args.seed, tag=args.tag))


def evaluate(args) -> int:
    """Per-frame metrics on the validation games, clean and with player dropouts, vs v0's rule."""
    bundle_ = load_model(args.model)
    rows = []
    for i, g in enumerate(load_games("val", args.sources, lambda m: None)):
        for drop in (False, True):
            X, y, _ = _game_rows(g, np.random.default_rng([args.seed, 99, i]), drop)
            p = bundle_[0].predict_proba(X[bundle_[1]].to_numpy(dtype=np.float32))[:, 1]
            for name, pp in (("model", p), ("v0_rule", v0_p_dead(X))):
                rows.append({"game": g.name, "source": g.source, "players": "dropout" if drop else "clean",
                             "method": name, **_metrics(y, pp)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== dead-ball detector, per frame (every 0.2 s) on the validation games ===")
    print(df.groupby(["players", "source", "method"])[["auc", "f1_dead", "acc", "share_dead_true", "share_dead_pred"]]
          .mean().round(3).to_string())
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    out = HARNESS_DIR / f"dead_{getattr(args, 'tag', 'eval')}.json"
    out.write_text(json.dumps({"model": str(Path(args.model).name), "rows": rows}, indent=2, default=float))
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tag", default="v1")
    t.add_argument("--sources", nargs="*", default=["idsse", "metrica", "pff"])
    t.add_argument("--val-sources", nargs="*", default=["idsse", "metrica", "pff"])
    t.add_argument("--iters", type=int, default=300)
    t.add_argument("--no-dropout-copy", dest="dropout_copy", action="store_false")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--model", type=Path, default=MODELS_DIR / "dead_v1.joblib")
    e.add_argument("--sources", nargs="*", default=["idsse", "metrica", "pff"])
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--tag", default="eval")
    args = ap.parse_args(argv)
    return train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
