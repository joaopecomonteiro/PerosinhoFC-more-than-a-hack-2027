"""who_lab.py -- WHICH TEAM has the ball? Compare approaches, on raw and on fixed tracking.

Run from JS/ (after `python bootstrap.py` and with the bundles built):

    python experiments/who_lab.py --split val --source idsse metrica skillcorner --tag who1
    python experiments/who_lab.py --split val --atd-noise --tag who_noise     # BAR-ATH-like noise
    python experiments/who_lab.py --synthetic                                  # smoke test, no data needed

The question is isolated on purpose. The ground truth tells us which frames are alive (this
stands in for your ball-in-play model), so every number below is about H vs A only:

    1. each approach turns every frame into a log-odds z  (> 0 = home has the ball, NaN = no idea)
    2. the SAME smoother (2-state Viterbi, one switch cost) turns z into possessions
    3. dead frames and frames outside the halves are copied from the truth, then the OFFICIAL
       metric scores the result (spellF1, macroF1)  + plain frame accuracy on alive frames

Each approach is run on every tracking VERSION: "raw" (as delivered / as noised) and "fixed"
(mqf.tracks.clean, the team-level tracklet cleaner). The learned approach is trained only on
the `train` split games and scored on the `val` split (no frame of a scored game ever trains).

Approaches (add yours in APPROACHES -- a function (trk, fps) -> z[n_frames]):
    nearest      v0 idea: closer team's nearest player, softened       (d_away - d_home)
    soft         every player counts: soft-max of -d^2 per team
    couple       "moves with the ball": close AND similar velocity to the ball (carry / dribble)
    chain        last team that CONTROLLED the ball, carried through passes, faded after ~4 s
    learned      gradient boosting on the features of the four above (+ simple context)

Only aggregates are printed or saved (harness_results/who_<tag>.json), so they are safe to share.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

JS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JS))
sys.path.insert(0, str(JS.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from mqf.config import HARNESS_DIR  # noqa: E402
from mqf.harness.evaluate import score  # noqa: E402

SWITCH_COST = 18.0          # same order as v0's H<->A cost; one value for every approach
CONDITIONS = ("raw", "fixed")


# ----------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------
def velocity(xy: np.ndarray, fps: float, win_s: float = 0.5) -> np.ndarray:
    """Central-difference velocity (m/s) over ~win_s; NaN stays NaN; edges NaN."""
    w = max(1, int(round(win_s * fps / 2)))
    v = (np.roll(xy, -w, axis=0) - np.roll(xy, w, axis=0)) / (2 * w / fps)
    v[:w] = np.nan
    v[-w:] = np.nan
    return v


def tmin(d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nearest distance of a team to the ball; inf when nobody is tracked."""
    return np.where(np.isnan(d[:, mask]), np.inf, d[:, mask]).min(axis=1)


class Geo:
    """Everything the approaches share, computed once per (game, tracking version)."""

    def __init__(self, trk, fps):
        self.fps = fps
        self.P = trk.players.astype(np.float64)                      # (n, S, 2)
        self.B = trk.ball.astype(np.float64)                         # (n, 2)
        self.home = np.asarray(trk.team_of_slot) == 0
        self.away = ~self.home
        self.seen = ~np.isnan(self.B[:, 0])
        self.d = np.linalg.norm(self.P - self.B[:, None, :], axis=2)          # NaN if anything missing
        self.vb = velocity(self.B, fps)
        self.vp = velocity(self.P, fps)
        self.dv = np.linalg.norm(self.vp - self.vb[:, None, :], axis=2)
        self.dh, self.da = tmin(self.d, self.home), tmin(self.d, self.away)
        self.ball_speed = np.linalg.norm(self.vb, axis=1)
        self.n_home = np.sum(~np.isnan(self.P[:, self.home, 0]), axis=1)
        self.n_away = np.sum(~np.isnan(self.P[:, self.away, 0]), axis=1)


# ----------------------------------------------------------------------------------------------
# approaches: each returns z[n] (log-odds, >0 home, NaN no information)
# ----------------------------------------------------------------------------------------------
def z_nearest(g: Geo, tau: float = 1.5) -> np.ndarray:
    ok = g.seen & np.isfinite(g.dh) & np.isfinite(g.da)
    z = np.clip((g.da - g.dh) / tau, -4, 4)
    return np.where(ok, z, np.nan)


def z_soft(g: Geo, sigma: float = 1.5) -> np.ndarray:
    s = np.where(np.isnan(g.d), -np.inf, -(g.d ** 2) / (2 * sigma ** 2))
    lh = np.logaddexp.reduce(s[:, g.home], axis=1)
    la = np.logaddexp.reduce(s[:, g.away], axis=1)
    ok = g.seen & (np.isfinite(lh) | np.isfinite(la))                  # at least one team tracked
    z = np.clip(np.where(np.isfinite(lh), lh, -50) - np.where(np.isfinite(la), la, -50), -6, 6)
    return np.where(ok, z, np.nan)


def z_couple(g: Geo, r: float = 2.0, sv: float = 2.5) -> np.ndarray:
    """A player 'has' the ball if it is close AND moves like the ball (same direction and speed)."""
    w = np.exp(-(g.d ** 2) / (2 * r ** 2) - (g.dv ** 2) / (2 * sv ** 2))
    w = np.nan_to_num(w, nan=0.0)
    z = np.log(w[:, g.home].sum(axis=1) + 1e-3) - np.log(w[:, g.away].sum(axis=1) + 1e-3)
    return np.where(g.seen, np.clip(z, -6, 6), np.nan)


def chain_parts(g: Geo, ctrl_m: float = 1.2, margin_m: float = 0.5, fade_s: float = 4.0):
    """(z, age_s): team of the last clear controller, carried through passes and faded with time."""
    n = len(g.dh)
    dmin = np.minimum(g.dh, g.da)
    clear = g.seen & np.isfinite(dmin) & (dmin <= ctrl_m) & (np.abs(g.dh - g.da) >= margin_m)
    sign = np.where(g.dh < g.da, 1.0, -1.0)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(clear, idx, -1))
    age = np.where(last >= 0, (idx - last) / g.fps, np.inf)
    carried = np.where(last >= 0, sign[np.maximum(last, 0)], 0.0)
    z = np.where(clear, 3.0 * sign, 2.0 * carried * np.clip(1 - age / fade_s, 0, 1))
    return np.where(np.isfinite(age) & (age <= 60), z, np.nan), age


def z_chain(g: Geo) -> np.ndarray:
    return chain_parts(g)[0]


def z_blend(g: Geo) -> np.ndarray:
    """No training: average of 'moves with the ball' (couple) and 'last controller' (chain)."""
    both = np.vstack([z_couple(g), z_chain(g)])
    return np.nanmean(both, axis=0)                           # NaN only when both are NaN


def heading_parts(g: Geo):
    """Pass-direction clues: is the ball heading TOWARDS the nearest home / away player, and how
    similar is that player's velocity to the ball's? (a pass in flight points at its receiver)."""
    n = len(g.dh)
    out = {}
    for name, mask in (("h", g.home), ("a", g.away)):
        dd = np.where(np.isnan(g.d[:, mask]), np.inf, g.d[:, mask])
        k = dd.argmin(axis=1)
        ok = np.isfinite(dd[np.arange(n), k]) & g.seen
        rel = g.P[:, mask, :][np.arange(n), k, :] - g.B                       # ball -> nearest player
        num = (rel * g.vb).sum(axis=1)
        den = np.linalg.norm(rel, axis=1) * g.ball_speed + 1e-6
        out["head_" + name] = np.where(ok, num / den, np.nan)
        out["dvn_" + name] = np.where(ok, g.dv[:, mask][np.arange(n), k], np.nan)
    return out


BASIC_COLS = ["dh", "da", "z_near", "z_soft", "z_couple", "z_chain", "age_ctrl", "ball_speed", "seen", "n_home", "n_away"]
PLUS_COLS = BASIC_COLS + ["head_h", "head_a", "head_diff", "dvn_h", "dvn_a", "dvn_diff"]


def feature_frame(g: Geo) -> pd.DataFrame:
    zc, age = chain_parts(g)
    hp = heading_parts(g)
    extra = {**hp, "head_diff": hp["head_h"] - hp["head_a"], "dvn_diff": hp["dvn_a"] - hp["dvn_h"]}
    return pd.DataFrame({**extra,
        "dh": np.where(np.isfinite(g.dh), g.dh, np.nan), "da": np.where(np.isfinite(g.da), g.da, np.nan),
        "z_near": z_nearest(g), "z_soft": z_soft(g), "z_couple": z_couple(g), "z_chain": zc,
        "age_ctrl": np.where(np.isfinite(age), age, np.nan), "ball_speed": g.ball_speed,
        "seen": g.seen.astype(float), "n_home": g.n_home, "n_away": g.n_away,
    })


APPROACHES = {"nearest": z_nearest, "soft": z_soft, "couple": z_couple, "chain": z_chain, "blend": z_blend}


# ----------------------------------------------------------------------------------------------
# smoother, scoring
# ----------------------------------------------------------------------------------------------
def smooth(z: np.ndarray, gt: np.ndarray, fps: float) -> np.ndarray:
    """H/A per frame: 2-state Viterbi inside every run of alive frames (gt in H/A)."""
    alive = (gt == "H") | (gt == "A")
    e = np.nan_to_num(np.clip(z, -6, 6), nan=0.0) * (25.0 / fps) / 2.0
    out = np.full(len(gt), "D", dtype="<U1")
    edges = np.flatnonzero(np.diff(np.r_[0, alive.view(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        seg = e[a:b]
        m = len(seg)
        back = np.zeros((m, 2), dtype=np.int8)
        sh, sa = seg[0], -seg[0]
        for t in range(1, m):
            h_from_a, a_from_h = sa - SWITCH_COST, sh - SWITCH_COST
            bh, ba = h_from_a > sh, a_from_h > sa
            back[t, 0], back[t, 1] = bh, ba
            sh, sa = max(sh, h_from_a) + seg[t], max(sa, a_from_h) - seg[t]
        s = 0 if sh >= sa else 1
        path = np.empty(m, dtype=np.int8)
        for t in range(m - 1, -1, -1):
            path[t] = s
            if t > 0 and back[t, s]:
                s = 1 - s
        out[a:b] = np.where(path == 0, "H", "A")
    return out


def smooth_hsmm(z: np.ndarray, gt: np.ndarray, fps: float) -> np.ndarray:
    """Same job as smooth(), but with explicit possession durations (mqf.decode.viterbi_hsmm)."""
    try:
        from mqf import decode
        decode.viterbi_hsmm
    except (ImportError, AttributeError):
        sys.exit("--smoother hsmm needs the HSMM decoder: apply hsmm_decoder.patch first")
    prm = decode.Params(decoder="hsmm")
    alive = (gt == "H") | (gt == "A")
    e = np.nan_to_num(np.clip(z, -6, 6), nan=0.0) * (25.0 / fps) / 2.0
    out = np.full(len(gt), "D", dtype="<U1")
    edges = np.flatnonzero(np.diff(np.r_[0, alive.view(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ll = np.zeros((b - a, 3))
        ll[:, 0], ll[:, 1], ll[:, 2] = e[a:b], -e[a:b], -1e6          # D impossible inside an alive run
        out[a:b] = np.where(decode.viterbi_hsmm(ll, prm, fps) == 0, "H", "A")
    return out


def evaluate(z: np.ndarray, g: Geo, gt: np.ndarray, fps: float, smoother: str = "viterbi") -> dict:
    sm = smooth_hsmm(z, gt, fps) if smoother == "hsmm" else smooth(z, gt, fps)
    pred = np.where(gt == "X", "X", np.where(gt == "D", "D", sm))
    r = score(gt, pred, fps)
    alive = (gt == "H") | (gt == "A")
    hit = (sm == gt)
    return {"acc": float(hit[alive].mean()), "acc_seen": float(hit[alive & g.seen].mean()) if (alive & g.seen).any() else np.nan,
            "spellF1": r["spellF1"], "macroF1": r["macroF1"], "spells": f"{r['n_pred_spells']}/{r['n_gt_spells']}"}


# ----------------------------------------------------------------------------------------------
# games: real bundles or the team's synthetic matches
# ----------------------------------------------------------------------------------------------
def bundle_games(sources, split, matches, version, args):
    from mqf.external import bundle, splits
    from mqf.external import evaluate as ev
    noise = ev.parse_noise(args.noise) if args.atd_noise else None
    for source in sources:
        names = matches
        if split:
            names = splits.games(split, sources=[source])
        for d in bundle.find(source, names):
            ncfg = replace(noise, seed=0) if noise is not None else None
            spec, trk, gt = ev.load(d, args.players_drop, False, version == "fixed", noise_cfg=ncfg)
            yield source, spec.name, spec.fps, trk, gt


def synthetic_games(seeds, version):
    from mqf import synthetic, tracks
    from mqf.harness.noise import NoiseConfig, corrupt
    for s in seeds:
        m = synthetic.clean_match(seed=s)
        trk = corrupt(m["players"], m["team_of_player"], m["ball"], m["labels"] != "X", m["fps"], replace(NoiseConfig(), seed=0))
        if version == "fixed":
            trk, _ = tracks.clean(trk, m["fps"], m["pitch"], tracks.CleanConfig(), halves=m["halves"])
        yield "synthetic", f"SYN-{s}", m["fps"], trk, m["labels"]


# ----------------------------------------------------------------------------------------------
# learned approach: train on `train` games, apply to the scored games
# ----------------------------------------------------------------------------------------------
def fit_learned(train_games, every: int = 5, cap: int = 40000):
    """One pass over the train games, two gradient-boosting models: 'learned' (the four hand-made
    scores + context) and 'stack' (the same + pass-direction clues)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y = [], []
    for _, _, fps, trk, gt in train_games:
        F = feature_frame(Geo(trk, fps))
        keep = np.flatnonzero(((gt == "H") | (gt == "A")))[::every][:cap]
        X.append(F.iloc[keep]); y.append((gt[keep] == "H").astype(int))
    if not X:
        return None
    X, y = pd.concat(X), np.concatenate(y)
    mk = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0)
    return {"learned": (mk().fit(X[BASIC_COLS], y), BASIC_COLS), "stack": (mk().fit(X[PLUS_COLS], y), PLUS_COLS)}


def z_learned(model_cols, g: Geo) -> np.ndarray:
    model, cols = model_cols
    p = np.clip(model.predict_proba(feature_frame(g)[cols])[:, 1], 1e-4, 1 - 1e-4)
    return np.where(g.seen, np.log(p / (1 - p)), np.nan)


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", nargs="*", default=["idsse", "metrica", "skillcorner"])
    ap.add_argument("--matches", nargs="*", help="bundle names (e.g. games of your new datasets, already built as bundles)")
    ap.add_argument("--split", choices=["train", "val"], help="games scored; use val")
    ap.add_argument("--atd-noise", action="store_true")
    ap.add_argument("--noise", nargs="*", default=[])
    ap.add_argument("--players-drop", action="store_true")
    ap.add_argument("--no-learned", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these approaches (names), e.g. --only couple chain learned stack")
    ap.add_argument("--smoother", choices=["viterbi", "hsmm"], default="viterbi",
                    help="hsmm = explicit possession durations (needs the hsmm_decoder.patch applied)")
    ap.add_argument("--synthetic", action="store_true", help="smoke test on the team's synthetic matches")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args(argv)

    rows = []
    for version in CONDITIONS:
        print(f"== tracking: {version}", flush=True)
        if args.synthetic:
            scored = synthetic_games((103, 104), version)
            model = None if args.no_learned else fit_learned(synthetic_games((101, 102), version))
        else:
            scored = bundle_games(args.source, args.split, args.matches, version, args)
            train_src = [s for s in args.source if s != "skillcorner"]     # 10 fps: not used for training
            model = None if args.no_learned else fit_learned(bundle_games(train_src, "train", None, version, args))
        for source, name, fps, trk, gt in scored:
            g = Geo(trk, fps)
            funcs = dict(APPROACHES)
            if model is not None:
                for k in ("learned", "stack"):
                    funcs[k] = lambda gg, mc=model[k]: z_learned(mc, gg)
            if args.only:
                funcs = {k: v for k, v in funcs.items() if k in args.only}
            for aname, fn in funcs.items():
                z = fn(g)
                rows.append({"tracking": version, "approach": aname, "source": source, "game": name,
                             **evaluate(z, g, gt, fps, args.smoother)})
            print(f"   {name:<16} done", flush=True)
    if not rows:
        print("no games found -- build the bundles first (python tasks.py bundles) or try --synthetic")
        return 1
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    order = [a for a in list(APPROACHES) + ["learned", "stack"] if a in set(df.approach)]       # your own approaches print too
    for source, sd in df.groupby("source", sort=False):
        print(f"\n=== {source}: {sd.game.nunique()} game(s); mean over games ===")
        t = sd.groupby(["approach", "tracking"])[["acc", "spellF1", "macroF1"]].mean().unstack("tracking").reindex(order)
        print(t.round(3).to_string())
    print("\n=== all sources (mean of per-source means) ===")
    t = df.groupby(["approach", "tracking", "source"])[["acc", "spellF1"]].mean().groupby(["approach", "tracking"]).mean().unstack("tracking").reindex(order)
    print(t.round(3).to_string())
    HARNESS_DIR.mkdir(exist_ok=True, parents=True)
    out = HARNESS_DIR / f"who_{args.tag}.json"
    out.write_text(df.to_json(orient="records", indent=1))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
