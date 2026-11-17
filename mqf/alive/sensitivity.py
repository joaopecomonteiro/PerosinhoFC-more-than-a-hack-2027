"""What does each kind of alive-layer error cost? The true alive mask, broken on purpose, WHO fixed.

    python -m mqf.alive.sensitivity --clean-tracks                         # 11 val games, bar, ball exact
    python -m mqf.alive.sensitivity --condition atd --seeds 0 --clean-tracks

Per val game: the true alive mask (oracle) and the same mask with ONE kind of error --
  restart_late / restart_early <s>   every restart (D -> alive) moved by s seconds
  stop_late / stop_early <s>         every stop (alive -> D) moved by s seconds
  lag <s>                            both moved late by s (a detector that reacts s seconds late)
  drop_short <L>                     stoppages shorter than L seconds missed (two spells merge)
  blips <len>x<rate>                 spurious stoppages of len seconds at rate per 90 min inside play
-- decoded with who_v1's P(home) by the 2-state Viterbi inside the mask (the oracle layer of #38), scored
with the official metric; plus dead_v1 in the production HMM for scale. The table says which failure
mode is worth the most: compare its delta with dead_v1's stoppage diagnostics (experiment.py).
Writes harness_results/alive_sens_<tag>.json / .md. Other games only.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .. import atd, decode, features
from ..config import HARNESS_DIR, MODELS_DIR
from ..harness.evaluate import score
from . import combine, experiment

PERTURB = [("restart_late", (0.2, 0.5, 1.0, 2.0)), ("restart_early", (0.5, 1.0)), ("stop_late", (0.5, 1.0, 2.0)),
           ("stop_early", (0.5, 1.0)), ("lag", (0.5, 1.0, 2.0)), ("drop_short", (3.0, 5.0, 8.0, 12.0)),
           ("blips", ((1.0, 10), (2.0, 10), (4.0, 10), (2.0, 30)))]


def perturb(alive: np.ndarray, halves, fps: float, kind: str, val, rng) -> np.ndarray:
    out = alive.copy()
    for _, a, b in halves:
        seg = alive[a:b + 1]
        dead_runs = combine.runs_of(np.where(seg, "L", "D"), "D")
        new = seg.copy()
        n = len(seg)
        if kind in ("restart_late", "restart_early", "stop_late", "stop_early", "lag"):
            d = int(round(val * fps))
            moved = []
            for s, e in dead_runs:                            # an edge at the half's start / end stays there
                s2, e2 = s, e
                if e < n and kind in ("restart_late", "lag"):
                    e2 = e + d
                if e < n and kind == "restart_early":
                    e2 = max(s + 1, e - d)
                if s > 0 and kind in ("stop_late", "lag"):
                    s2 = s + d
                if s > 0 and kind == "stop_early":
                    s2 = s - d
                new[s:e] = True                               # remove the original stoppage ...
                moved.append((max(0, s2), min(n, e2)))
            for s2, e2 in moved:
                if e2 > s2:
                    new[s2:e2] = False                        # ... and put the moved one
        elif kind == "drop_short":
            for s, e in dead_runs:
                if (e - s) < val * fps:
                    new[s:e] = True
        elif kind == "blips":
            ln, rate = val
            L = max(1, int(round(ln * fps)))
            alive_idx = np.flatnonzero(seg)
            k = rng.poisson(rate * n / (90 * 60 * fps))
            for _ in range(int(k)):
                if len(alive_idx):
                    c = int(alive_idx[rng.integers(0, len(alive_idx))])
                    new[c:c + L] = False
        out[a:b + 1] = new
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", nargs="*", default=["idsse", "metrica", "pff"])
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--max-games", type=int)
    ap.add_argument("--condition", default="bar", choices=["bar", "atd"])
    ap.add_argument("--ball-mode", default="exact", choices=experiment.BALL_MODES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--clean-tracks", action="store_true")
    ap.add_argument("--who", default="who_v1")
    ap.add_argument("--dead-model", default=str(MODELS_DIR / "dead_v1.joblib"))
    ap.add_argument("--no-ball-fill", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--tag", default="sens")
    a = ap.parse_args(argv)
    log = lambda m: print(m, flush=True)
    from ..external import bundle, splits
    from ..owner import combine as ocombine
    from .models import AliveModel
    params = replace(decode.Params(), dead_who="max")
    who_name, who_paths = experiment.parse_who(a.who)
    device = "cpu"
    try:
        import torch
        device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    except ImportError:
        pass
    ball_model = ball_path = None
    if not a.no_ball_fill:
        from ..ball import impute
        ball_path = MODELS_DIR / impute.FROZEN["model"]
        ball_model = impute.load_model(ball_path, device)
    dead_m = AliveModel.load(a.dead_model)
    seeds = a.seeds if a.condition == "atd" else [0]
    rows = []
    t_all = time.time()
    for source in a.source:
        names = a.matches or splits.games(a.split, sources=[source])
        dirs = bundle.find(source, names)[:a.max_games] if a.max_games else bundle.find(source, names)
        for d in dirs:
            for seed in seeds:
                t0 = time.time()
                spec, trk, gt = experiment.load_condition(d, a.condition, seed, a.ball_mode, a.clean_tracks)
                extra = [a.condition, seed, a.ball_mode, a.clean_tracks, [], who_name]
                ball_f, filled, p_home, pb_who = experiment.evidence(spec, trk, who_paths, ball_model, ball_path, device,
                                                                     extra, not a.no_cache)
                n, fps, halves = spec.n_frames, spec.fps, spec.halves
                ins = experiment._inside(halves, n)
                alive = np.isin(gt, ["H", "A"]) & ins
                sw = params.switch_cost if pb_who is None else ocombine.switch_costs(pb_who, params.switch_cost, 0.5)
                meta = {"game": spec.name, "source": source, "seed": seed}
                rng = np.random.default_rng([seed, 5])
                cases = [("oracle", None, alive)]
                for kind, vals in PERTURB:
                    for v in vals:
                        tag = f"{kind} {v[0]:g}s x{v[1]}/90" if kind == "blips" else f"{kind} {v:g}s"
                        cases.append((tag, (kind, v), perturb(alive, halves, fps, kind, v, rng)))
                for tag, _, m in cases:
                    lab = ocombine.decode_given_alive(p_home, m & ins, halves, n, sw, params.w_who_model, params.who_clip)
                    rows.append({**meta, "case": tag, **score(gt, lab, fps), **combine.stoppage_stats(gt, lab, halves, fps)})
                X = experiment.feats.features(trk.players, trk.team_of_slot, trk.ball, fps, spec.pitch, halves, "v1")
                pdd, _ = dead_m.predict(X, halves, n, fps, device)
                feat = features.build(atd.Tracking(trk.players, trk.player_speed, np.asarray(ball_f, np.float32),
                                                   trk.team_of_slot), spec.pitch, fps, ball_filled=filled, p_home=p_home)
                lab = combine.decode_hmm(feat, pdd, halves, n, params, sw_t=None if np.isscalar(sw) else sw)
                rows.append({**meta, "case": "dead_v1 (HMM)", **score(gt, lab, fps),
                             **combine.stoppage_stats(gt, lab, halves, fps)})
                log(f"{spec.name:<14} s{seed} {time.time() - t0:5.0f} s  oracle {rows[-len(cases) - 1]['spellF1']:.3f}  "
                    f"dead_v1 {rows[-1]['spellF1']:.3f}")
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    base = df[df.case == "oracle"].set_index(["game", "seed"])["spellF1"]
    tab = []
    for case, g in df.groupby("case", sort=False):
        cur = g.set_index(["game", "seed"])["spellF1"]
        dl = (cur - base.loc[cur.index]).to_numpy()
        per = g.groupby("source")[["spellF1", "macroF1", "alive_dead_f1"]].mean().mean()
        tab.append({"case": case, "spellF1": per.spellF1, "d_vs_oracle": dl.mean(),
                    "se": dl.std(ddof=1) / np.sqrt(len(dl)) if len(dl) > 1 else np.nan,
                    "macroF1": per.macroF1, "alive_dead_f1": per.alive_dead_f1, "n": len(g)})
    t = pd.DataFrame(tab)
    text = (f"## Alive-layer sensitivity ({a.condition}, ball {a.ball_mode}, clean_tracks {a.clean_tracks}, who {who_name})"
            f" -- one error kind at a time on the TRUE alive mask\n\n```\n{t.round(3).to_string(index=False)}\n```\n")
    print("\n" + text)
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    (HARNESS_DIR / f"alive_sens_{a.tag}.json").write_text(json.dumps({"rows": rows, "minutes": round((time.time() - t_all) / 60, 1)},
                                                                     indent=1, default=float))
    (HARNESS_DIR / f"alive_sens_{a.tag}.md").write_text(text)
    log(f"wrote {HARNESS_DIR / f'alive_sens_{a.tag}.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
