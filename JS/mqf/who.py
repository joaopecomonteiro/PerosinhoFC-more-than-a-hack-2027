"""Who has the ball? P(home in possession | ball alive), per frame, learned on other games.

    python -m mqf.who train --tag v1                       # training games of splits.json -> models/who_v1.joblib
    python -m mqf.who train --tag v1 --sources idsse metrica
    python -m mqf.who eval --model models/who_v1.joblib    # per-frame metrics on the validation games

Why: v0 answers "the team of the player nearest the ball, NOW". The ground truth is ownership: the
last team that controlled the ball keeps it while it flies, bounces or is contested, so the answer
at a frame depends on who was at the ball before and who gets it next, not only on who is nearest.
v0 leaves that to one switch cost in the decoder; this model learns it per frame, and the decoder
then only has to turn it into whole possessions.

Features, identical for every source (BAR-ATH included), from positions only -- no identities:
  now       distance of each team's nearest and 2nd nearest player to the ball, players within
            2 / 5 m, speed of the nearest; ball speed, acceleration, distance to the touchline
  clocks    seconds since / until each team was last / is next "at the ball" (nearest of all and
            within 2 m), capped at 15 s; seconds since / until the ball was seen at all
  windows   share of the last / next 1 and 3 s each team was at the ball. Centred context: labels
            are made offline, the whole match is known (as for the submission)
  shape     in the direction of play (which goal a team defends is read per half from the two
            teams' mean positions): ball position and velocity towards the opponent's goal, share
            of each team's players goal-side of the ball, each team's centre relative to the ball,
            depth, width and mean speed
Home and away are arbitrary, so every game is also trained with the teams swapped and the
prediction is the average of both views: P(home) + P(away) = 1 exactly.
Model: sklearn HistGradientBoostingClassifier (deterministic, NaN-aware), on alive frames every
0.2 s of the training games, in two copies: as delivered, and "hard" -- ATD-like player dropouts,
no ball on dead frames (BAR-ATH loses it in stoppages) and alive spans of 0.2-30 s without a ball,
so the model also says what it knows when the ball is missing.
Labels: H vs A inside the halves of each bundle (source rules differ -- see TECHNICAL.md).
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
from .dead import player_speed

SAMPLE_HZ = 5.0
R_TOUCH = 2.0                 # m: the nearest player this close is "at the ball" (clocks, windows)
CAP_S = 15.0                  # clocks stop counting here
VEL_HALF_S = 0.2              # ball velocity / acceleration: centred difference over +- this
HIDE_SPAN_S = (0.2, 30.0)     # hidden alive spans, log-uniform -- the range of the ball model's masks
HIDE_SHARE = 0.4              # of each half, before overlaps
VERSION = 1


def _nan(fn, a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def _centred_diff(x: np.ndarray, fps: float, half_s: float) -> np.ndarray:
    """(x(t + h) - x(t - h)) / 2h along axis 0, per second; NaN where an end is missing."""
    w = max(1, int(round(half_s * fps)))
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if len(x) > 2 * w:
        out[w:-w] = (x[2 * w:] - x[:-2 * w]) * fps / (2 * w)
    return out


def _since(flag: np.ndarray, fps: float) -> np.ndarray:
    """Seconds since `flag` was last True (0 on a True frame); CAP_S if never or longer ago."""
    t = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, t, -1))
    return np.where(last >= 0, np.minimum((t - last) / fps, CAP_S), CAP_S)


def _until(flag: np.ndarray, fps: float) -> np.ndarray:
    return _since(flag[::-1], fps)[::-1]


def _roll(x: np.ndarray, w: int, future: bool = False) -> np.ndarray:
    """Mean over the last (or next) w frames, this one included."""
    s = pd.Series(x)
    if future:
        return s[::-1].rolling(w, min_periods=1).mean()[::-1].to_numpy()
    return s.rolling(w, min_periods=1).mean().to_numpy()


def attack_sign(cx_home: np.ndarray, cx_away: np.ndarray, halves, n: int) -> np.ndarray:
    """(n,) +1 where home attacks towards +x, -1 towards -x, per half: the team whose mean position
    over the half is at lower x defends x = 0 (the two blocks move together, each on its own side)."""
    sgn = np.ones(n, dtype=np.float64)
    for _, a, b in halves:
        gap = _nan(np.nanmean, (cx_away - cx_home)[a:b + 1], axis=0)
        sgn[a:b + 1] = -1.0 if np.isfinite(gap) and gap < 0 else 1.0
    return sgn


def frame_features(players: np.ndarray, team: np.ndarray, ball: np.ndarray, fps: float, pitch, halves) -> dict:
    """Per-frame columns before the time context, plus the 'at the ball' flags of each team."""
    L, W = pitch
    xy = players.astype(np.float64)
    b = ball.astype(np.float64)
    n = len(b)
    x, y = xy[..., 0], xy[..., 1]
    present = ~np.isnan(x)
    has_ball = ~np.isnan(b[:, 0])
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(xy - b[:, None, :], axis=-1)                   # (n, P), NaN if either is missing
    sp = player_speed(xy, fps)
    vb = _centred_diff(b, fps, VEL_HALF_S)
    f = {"ball_speed": np.linalg.norm(vb, axis=1),
         "ball_acc": np.linalg.norm(_centred_diff(vb, fps, VEL_HALF_S), axis=1),
         "ball_edge": np.minimum(b[:, 1], W - b[:, 1])}
    cx, xs, pres = {}, {}, {}
    rows = np.arange(n)
    for tm, s in ((0, "h"), (1, "a")):
        m = team == tm
        dt = np.where(np.isnan(d[:, m]), np.inf, d[:, m])
        if dt.shape[1] < 2:
            dt = np.concatenate([dt, np.full((n, 2 - dt.shape[1]), np.inf)], axis=1)
        two = np.partition(dt, 1, axis=1)[:, :2]
        d1 = np.where(np.isinf(two[:, 0]), np.nan, two[:, 0])
        f[f"d1_{s}"] = d1
        f[f"d2_{s}"] = np.where(np.isinf(two[:, 1]), np.nan, two[:, 1])
        for r in (2, 5):
            f[f"n{r}m_{s}"] = np.where(has_ball, (dt < r).sum(axis=1), np.nan)
        spm = sp[:, m]
        if spm.shape[1]:
            f[f"sp1_{s}"] = np.where(np.isnan(d1), np.nan, spm[rows, np.argmin(dt[:, :spm.shape[1]], axis=1)])
        else:
            f[f"sp1_{s}"] = np.full(n, np.nan)
        f[f"speed_{s}"] = _nan(np.nanmean, spm)
        f[f"depth_{s}"] = _nan(np.nanstd, x[:, m])
        f[f"width_{s}"] = _nan(np.nanstd, y[:, m])
        f[f"n_{s}"] = present[:, m].sum(axis=1).astype(np.float64)
        cx[s], xs[s], pres[s] = _nan(np.nanmean, x[:, m]), x[:, m], present[:, m]

    sgn = attack_sign(cx["h"], cx["a"], halves, n)                        # home attacks towards sgn * x
    f["ball_x_att"] = np.where(sgn > 0, b[:, 0] / L, 1.0 - b[:, 0] / L)   # 0 = home's own goal line
    f["ball_vx_att"] = sgn * vb[:, 0]                                     # > 0: towards away's goal
    with np.errstate(invalid="ignore"):
        rel_h = sgn[:, None] * (xs["h"] - b[:, None, 0])                  # < 0: goal-side of the ball (home)
        rel_a = sgn[:, None] * (xs["a"] - b[:, None, 0])                  # > 0: goal-side of the ball (away)
        f["behind_h"] = np.where(has_ball, (pres["h"] & (rel_h < 0)).sum(axis=1) / np.maximum(pres["h"].sum(axis=1), 1), np.nan)
        f["behind_a"] = np.where(has_ball, (pres["a"] & (rel_a > 0)).sum(axis=1) / np.maximum(pres["a"].sum(axis=1), 1), np.nan)
    f["centre_rel_h"] = sgn * (cx["h"] - b[:, 0])
    f["centre_rel_a"] = sgn * (cx["a"] - b[:, 0])
    f["centre_gap"] = sgn * (cx["a"] - cx["h"])
    for k in ("d1", "d2", "n2m", "n5m", "depth", "width", "speed", "behind"):
        f[f"{k}_diff"] = f[f"{k}_a"] - f[f"{k}_h"]                        # d1_diff > 0: home is closer
    with np.errstate(invalid="ignore"):
        at_h = has_ball & (f["d1_h"] <= R_TOUCH) & ~(f["d1_a"] < f["d1_h"])
        at_a = has_ball & (f["d1_a"] <= R_TOUCH) & ~(f["d1_h"] <= f["d1_a"])
    return {"cols": f, "at_h": at_h, "at_a": at_a, "has_ball": has_ball}


def features(players: np.ndarray, team: np.ndarray, ball: np.ndarray, fps: float, pitch, halves) -> pd.DataFrame:
    """(n_frames, F) -- NaN outside the halves."""
    fr = frame_features(players, team, ball, fps, pitch, halves)
    base = pd.DataFrame(fr["cols"])
    n = len(base)
    ctx_cols = ["since_h", "since_a", "until_h", "until_a", "since_ball", "until_ball"]
    for sec in (1, 3):
        ctx_cols += [f"past{sec}_h", f"past{sec}_a", f"next{sec}_h", f"next{sec}_a"]
    ctx_cols += ["d1_diff_1s", "ball_speed_1s", "ball_acc_1s"]
    ctx = np.full((n, len(ctx_cols)), np.nan, dtype=np.float64)
    inside = np.zeros(n, dtype=bool)
    w1 = max(1, int(round(fps)))
    for _, a, b in halves:
        sl = slice(a, b + 1)
        inside[sl] = True
        ah, aa, hb = fr["at_h"][sl], fr["at_a"][sl], fr["has_ball"][sl]
        cols = [_since(ah, fps), _since(aa, fps), _until(ah, fps), _until(aa, fps), _since(hb, fps), _until(hb, fps)]
        for sec in (1, 3):
            w = max(1, int(round(sec * fps)))
            cols += [_roll(ah.astype(float), w), _roll(aa.astype(float), w),
                     _roll(ah.astype(float), w, future=True), _roll(aa.astype(float), w, future=True)]
        seg = base.iloc[a:b + 1]
        cols += [seg[c].rolling(w1, center=True, min_periods=1).agg(fn).to_numpy()
                 for c, fn in (("d1_diff", "mean"), ("ball_speed", "mean"), ("ball_acc", "max"))]
        ctx[sl] = np.stack(cols, axis=1)
    out = pd.concat([base, pd.DataFrame(ctx, columns=ctx_cols)], axis=1)
    out["since_diff"] = out["since_a"] - out["since_h"]                   # > 0: home was at the ball more recently
    out["until_diff"] = out["until_a"] - out["until_h"]                   # > 0: home is at the ball sooner
    for k in ("past1", "past3", "next1", "next3"):
        out[f"{k}_diff"] = out[f"{k}_h"] - out[f"{k}_a"]
    out.loc[~inside] = np.nan
    return out


# ------------------------------------------------------------------------------------ model

def load_model(path):
    import joblib
    obj = joblib.load(path)
    return obj["model"], obj["columns"], obj.get("meta", {})


def _proba(model, cols, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[cols].to_numpy(dtype=np.float32))[:, 1]          # float32, as in training


def predict(model_bundle, players, team, ball, fps, pitch, halves) -> np.ndarray:
    """(n,) P(home has the ball | alive), NaN outside the halves; the mean of the two team orders,
    rounded to 1e-4 (reproducible downstream)."""
    model, cols, _ = model_bundle
    team = np.asarray(team)
    inside = np.zeros(len(ball), dtype=bool)
    for _, a, b in halves:
        inside[a:b + 1] = True
    p = np.full(len(ball), np.nan)
    if inside.any():
        ph = _proba(model, cols, features(players, team, ball, fps, pitch, halves).loc[inside])
        pa = _proba(model, cols, features(players, 1 - team, ball, fps, pitch, halves).loc[inside])
        p[inside] = 0.5 * (ph + 1.0 - pa)
    return np.round(p, 4)


def v0_p_home(X: pd.DataFrame, tau_m: float = 1.5, loose_m: float = 4.0) -> np.ndarray:
    """v0's "who is closer now" on the same distances, for comparison only (0.5 without a ball)."""
    d_min = np.fmin(X["d1_h"].to_numpy(), X["d1_a"].to_numpy())
    control = 1 / (1 + np.exp(-(loose_m - np.nan_to_num(d_min, nan=99.0))))
    logit = control * np.nan_to_num(X["d1_diff"].to_numpy()) / tau_m
    return 1 / (1 + np.exp(-np.clip(logit, -50, 50)))


def hold_p_home(X: pd.DataFrame) -> np.ndarray:
    """The last team at the ball keeps it (what a sticky decoder carries), for comparison only."""
    sh, sa = X["since_h"].to_numpy(), X["since_a"].to_numpy()
    p = np.where(sh < sa, 1.0, np.where(sa < sh, 0.0, 0.5))
    return np.where(np.isnan(sh), 0.5, p)


def hide_ball(ball: np.ndarray, labels: np.ndarray, halves, fps: float, rng) -> np.ndarray:
    """The 'hard' ball: none on dead frames, and alive spans of 0.2-30 s removed."""
    out = np.array(ball, dtype=np.float64, copy=True)
    out[np.asarray(labels) == "D"] = np.nan
    lo, hi = np.log(HIDE_SPAN_S[0]), np.log(HIDE_SPAN_S[1])
    for _, a, b in halves:
        n = b - a + 1
        hidden = 0
        while hidden < HIDE_SHARE * n:
            ln = max(1, int(np.exp(rng.uniform(lo, hi)) * fps))
            s = a + int(rng.integers(0, max(1, n - ln)))
            out[s:s + ln] = np.nan
            hidden += ln
    return out


def _game_rows(g, rng, hard: bool, swap: bool = False):
    """Features + labels (1 = home) of one bundle game on alive frames every 1 / SAMPLE_HZ s."""
    from .ball import data as bdata
    players, ball = g.players, g.ball
    lab = np.asarray(g.labels)
    if hard:
        players = bdata._player_mask(players.astype(np.float64), rng, g.fps).astype(np.float32)
        ball = hide_ball(ball, lab, g.halves, g.fps, rng)
    team = (1 - g.team) if swap else g.team
    X = features(players, team, ball, g.fps, g.pitch, g.halves)
    k = max(1, int(round(g.fps / SAMPLE_HZ)))
    idx = np.concatenate([np.arange(a, b + 1, k) for _, a, b in g.halves])
    idx = idx[np.isin(lab[idx], ["H", "A"])]
    y = (lab[idx] == ("A" if swap else "H")).astype(np.int8)
    return X.iloc[idx], y, idx


def _metrics(y, p) -> dict:
    from sklearn.metrics import roc_auc_score
    if not len(y):
        return {"n": 0, "auc": None, "acc": None, "share_home_true": None, "share_home_pred": None}
    pred = p > 0.5
    acc = float(np.mean(np.where(p == 0.5, 0.5, pred == (y == 1))))       # an undecided frame counts half
    return {"n": int(len(y)), "auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
            "acc": round(acc, 4), "share_home_true": round(float(y.mean()), 3),
            "share_home_pred": round(float(pred.mean()), 3)}


def load_games(part, sources, log=print):
    """Bundle games of one split part, one at a time."""
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
        for hard in (False, True):
            for swap in (False, True):
                X, y, _ = _game_rows(g, np.random.default_rng([args.seed, i, int(hard)]), hard, swap)
                Xs.append(X.astype(np.float32))
                ys.append(y)
        del g
    if not Xs:
        raise SystemExit("no training bundles: python tasks.py bundles")
    X = pd.concat(Xs, ignore_index=True)
    del Xs
    y = np.concatenate(ys)
    cols = list(X.columns)
    log(f"train {len(names)} games, {len(X):,} rows ({y.mean():.1%} home), {len(cols)} features, "
        f"built in {time.time() - t0:.0f} s")
    model = HistGradientBoostingClassifier(max_iter=args.iters, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
                                           random_state=args.seed)
    model.fit(X.to_numpy(), y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"who_{args.tag}.joblib"
    meta = {"version": VERSION, "tag": args.tag, "sources": list(args.sources), "train_games": names,
            "rows": int(len(X)), "iters": args.iters}
    joblib.dump({"model": model, "columns": cols, "meta": meta}, path)
    log(f"wrote {path}  ({time.time() - t0:.0f} s)")
    return evaluate(argparse.Namespace(model=path, sources=args.val_sources, seed=args.seed, tag=args.tag))


def evaluate(args) -> int:
    """Per-frame metrics on the alive frames of the validation games: as delivered and 'hard'
    (seen / hidden ball apart), against v0's rule and the last-team-holds rule."""
    model, cols, _ = load_model(args.model)
    rows = []
    for i, g in enumerate(load_games("val", args.sources, lambda m: None)):
        for hard in (False, True):
            seed = [args.seed, 99, i]
            X, y, _ = _game_rows(g, np.random.default_rng(seed), hard)
            Xs, _, _ = _game_rows(g, np.random.default_rng(seed), hard, swap=True)
            p = 0.5 * (_proba(model, cols, X) + 1.0 - _proba(model, cols, Xs))
            seen = ~np.isnan(X["d1_h"].to_numpy()) | ~np.isnan(X["d1_a"].to_numpy())
            parts = [("all", np.ones(len(y), dtype=bool))]
            if hard:
                parts += [("seen", seen), ("hidden", ~seen)]
            for name, pp in (("model", p), ("v0_rule", v0_p_home(X)), ("hold", hold_p_home(X))):
                for part, m in parts:
                    rows.append({"game": g.name, "source": g.source, "condition": "hard" if hard else "clean",
                                 "ball": part, "method": name, **_metrics(y[m], pp[m])})
    if not rows:
        raise SystemExit("no validation bundles: python tasks.py bundles")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== who has the ball, per alive frame (every 0.2 s) on the validation games ===")
    print(df.groupby(["condition", "ball", "source", "method"])[["auc", "acc", "share_home_true", "share_home_pred"]]
          .mean().round(3).to_string())
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    out = HARNESS_DIR / f"who_{getattr(args, 'tag', 'eval')}.json"
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
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--model", type=Path, default=MODELS_DIR / "who_v1.joblib")
    e.add_argument("--sources", nargs="*", default=["idsse", "metrica", "pff"])
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--tag", default="eval")
    args = ap.parse_args(argv)
    return train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
