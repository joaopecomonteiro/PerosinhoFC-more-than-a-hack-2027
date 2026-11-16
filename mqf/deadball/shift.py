"""How different do the dead-ball model's inputs look on our match vs the games it learned from?

    python -m mqf.deadball.shift --tag v2

Aggregates only (percentiles over all graded frames -- no frame, no time), safe to share. For every
input: 10th / 50th / 90th percentile on our match and on the PFF + IDSSE + Metrica validation games
(as the model sees them in evaluation), and a shift score = how far our match's median sits outside the
other games' 10-90 % range, in units of that range. Inputs with a large score are where our tracker
differs from everything the model was trained on -- the first suspects when it misreads a stoppage.
Also: the model's P(dead) on both, and the share of frames above 0.5.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import atd
from ..config import MODELS_DIR, REPORTS_DIR, competition_game
from . import evaluate, features, model


def _graded_rows(X: pd.DataFrame, halves, k: int) -> pd.DataFrame:
    idx = np.concatenate([np.arange(a, b + 1, k) for _, a, b in halves])
    return X.iloc[idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)
    m = model.load(MODELS_DIR / f"deadball_{a.tag}.joblib")

    spec = competition_game()
    trk = atd.load(spec)
    Xo = model.add_players_columns(features.build(trk.players, trk.team_of_slot, trk.ball, spec.fps, spec.pitch, spec.halves), m, spec)
    po = model.predict(m, trk, spec)
    k = max(1, int(round(spec.fps / 5)))
    ours = _graded_rows(Xo, spec.halves, k)
    p_ours = po[np.concatenate([np.arange(s, e + 1, k) for _, s, e in spec.halves])]

    others, p_oth = [], []
    for cond in ("pff", "drift"):
        for _, sp, tk, gt, _ in evaluate.games(cond, [0], 0.8):
            X = model.add_players_columns(features.build(tk.players, tk.team_of_slot, tk.ball, sp.fps, sp.pitch, sp.halves), m, sp)
            kk = max(1, int(round(sp.fps / 5)))
            others.append(_graded_rows(X, sp.halves, kk))
            p = model.predict(m, tk, sp)
            p_oth.append(p[np.concatenate([np.arange(s, e + 1, kk) for _, s, e in sp.halves])])
    oth = pd.concat(others, ignore_index=True)

    rows = []
    for c in m["columns"]:
        o, t = ours[c].to_numpy(float), oth[c].to_numpy(float)
        o, t = o[np.isfinite(o)], t[np.isfinite(t)]
        if not len(o) or not len(t):
            continue
        q_o = np.percentile(o, [10, 50, 90])
        q_t = np.percentile(t, [10, 50, 90])
        width = max(q_t[2] - q_t[0], 1e-9)
        outside = max(q_t[0] - q_o[1], q_o[1] - q_t[2], 0.0) / width
        spread_ratio = (q_o[2] - q_o[0]) / width
        rows.append({"input": c, "ours_p10": q_o[0], "ours_p50": q_o[1], "ours_p90": q_o[2],
                     "others_p10": q_t[0], "others_p50": q_t[1], "others_p90": q_t[2],
                     "median_shift": (q_o[1] - q_t[1]) / width, "outside": outside, "spread_ratio": spread_ratio})
    df = pd.DataFrame(rows)
    df["score"] = df["median_shift"].abs() + (df["spread_ratio"] - 1).abs() * 0.5
    df = df.sort_values("score", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("=== inputs: our match vs the validation games (graded frames, every 0.2 s; aggregates only) ===")
    print(df.head(a.top).round(2).to_string(index=False))
    po_v, pt_v = p_ours[np.isfinite(p_ours)], np.concatenate(p_oth)
    pt_v = pt_v[np.isfinite(pt_v)]
    print("\nP(dead): ours p10/p50/p90", np.round(np.percentile(po_v, [10, 50, 90]), 3).tolist(),
          "share > 0.5:", round(float((po_v > 0.5).mean()), 3),
          "| others", np.round(np.percentile(pt_v, [10, 50, 90]), 3).tolist(), "share > 0.5:",
          round(float((pt_v > 0.5).mean()), 3))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"deadball_shift_{a.tag}.csv"            # our match: gitignored, never committed
    df.round(4).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
