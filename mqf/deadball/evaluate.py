"""Score ONLY "is the ball dead?" on other games, with the official building blocks.

    python -m mqf.deadball.evaluate --tag v1
    python -m mqf.deadball.evaluate --conditions pff --tag quick
    python -m mqf.deadball.evaluate --stoppage-loss 0.3 --tag v1_loss30     # vary the ball-loss assumption

Each frame becomes DEAD or IN PLAY and is scored with possession_metrics.py from vendor/ (never
reimplemented):
  alive_dead_f1   the official frame metric; plus precision / recall of dead frames
  stopF1          whole stoppages: the D runs of segments(), matched one-to-one by IoU with
                  spell_overlaps / match_spells -- exactly how spellF1 matches possessions -- F1
                  averaged over SPELL_IOUS (0.3 ... 0.7)
  start / end     stoppage starts and restarts matched within +/- 1 s (match_transitions): F1 and
                  median error in seconds
  found by length share of true stoppages matched at IoU 0.5, by their length
Conditions:
  pff     PFF validation games as they are (broadcast tracking loses the ball in stoppages by itself)
          + player dropouts -- the honest test: its ball loss was not made by us
  noisy   IDSSE + Metrica validation games with the ATD noise model and the ball removed for a whole
          stoppage with probability --stoppage-loss (default 0.8), else for part of it. Optimistic:
          training used the same mechanism.
Detectors (dead mask per game):
  players_frame   João's players-only P(dead) (models/dead_v1.joblib), threshold 0.5 -- the reference
  players         the same through our decoder
  <tag>_frame     our model, threshold 0.5 (what the model alone sees)
  <tag>_wX        our model through the decoder with emission weight X (several --model side by side)
Writes harness_results/deadball_eval_<tag>.json (aggregates only).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import zlib
from dataclasses import replace

import numpy as np
import pandas as pd

from ..config import HARNESS_DIR, MODELS_DIR, official
from . import decoder, model

LENGTH_BUCKETS_S = (0, 5, 15, 40, np.inf)
TOL_S = 1.0
WEIGHTS = (0.1, 0.2, 0.4)



def games(condition: str, seeds, stoppage_loss: float):
    from ..external import bundle, splits
    from ..external.evaluate import load, parse_noise
    if condition == "pff":
        for d in bundle.find("pff", splits.games("val", sources=["pff"])):
            spec, trk, gt = load(d, players_drop=True)
            yield "pff", spec, trk, gt, trk.ball            # PFF: the delivered (visible) ball types the restarts
    elif condition == "noisy":
        for source in ("idsse", "metrica"):
            for d, seed in itertools.product(bundle.find(source, splits.games("val", sources=[source])), seeds):
                spec, trk, gt = load(d, noise_cfg=replace(parse_noise([]), seed=seed))
                clean = load(d)[1].ball                    # the bundle's own ball types the restarts
                rng = np.random.default_rng([seed, 77, zlib.crc32(spec.name.encode())])
                b = model.hide_stoppage_ball(trk.ball.astype(np.float64), gt, stoppage_loss, rng)
                yield source, spec, model.with_ball(trk, b), gt, clean
    elif condition == "kept":
        for source in ("idsse", "metrica"):
            for d, seed in itertools.product(bundle.find(source, splits.games("val", sources=[source])), seeds):
                spec, trk, gt = load(d, noise_cfg=replace(parse_noise([]), seed=seed))
                yield source, spec, trk, gt, load(d)[1].ball      # ATD noise only: the ball is NOT removed in stoppages
    elif condition == "carried":
        for source in ("idsse", "metrica"):
            for d, seed in itertools.product(bundle.find(source, splits.games("val", sources=[source])), seeds):
                spec, trk, gt = load(d, noise_cfg=replace(parse_noise([]), seed=seed))
                clean = load(d)[1].ball
                rng = np.random.default_rng([seed, 79, zlib.crc32(spec.name.encode())])
                b = model.carried_ball(trk.ball, trk.players, gt, spec.fps, rng, p_carry=0.7)
                yield source, spec, model.with_ball(trk, b), gt, clean
    elif condition == "drift":
        for source in ("idsse", "metrica"):
            for d, seed in itertools.product(bundle.find(source, splits.games("val", sources=[source])), seeds):
                spec, trk, gt = load(d, noise_cfg=replace(parse_noise([]), seed=seed))
                clean = load(d)[1].ball
                rng = np.random.default_rng([seed, 78, zlib.crc32(spec.name.encode())])
                b = model.drift_ball_loss(trk.ball.astype(np.float64), gt, spec.fps, rng, p_whole=stoppage_loss,
                                          lead_p=0.5, live_per_min=0.5)
                yield source, spec, model.with_ball(trk, b), gt, clean
    else:
        raise SystemExit(f"unknown condition {condition}")


def score(gt: np.ndarray, dead: np.ndarray, fps: float) -> dict:
    """Dead / in-play scores of a dead mask against frame labels (H / A / D / X)."""
    _, pm = official()
    graded = gt != "X"
    g = np.where(graded, np.where(gt == "D", "D", "H"), "X")       # H stands for IN PLAY here
    p = np.where(graded, np.where(dead, "D", "H"), "X")
    gc, pc = pm.label_codes(g), pm.label_codes(p)
    runs = pm.valid_runs(gc)
    cm3 = pm.scored_confusion(pm.confusion(gc, pc))
    D, H = pm.LABELS.index("D"), pm.LABELS.index("H")
    tp, fp, fn = cm3[D, D], cm3[:D, D].sum(), cm3[D, :D].sum()
    out = {"alive_dead_f1": pm.alive_dead_f1(cm3),
           "dead_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
           "dead_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
           "dead_share_true": float((g[graded] == "D").mean()), "dead_share_pred": float((p[graded] == "D").mean())}
    gs = [s for s in pm.segments(gc, runs) if s[2] == D]
    ps = [s for s in pm.segments(pc, runs) if s[2] == D]
    ov = pm.spell_overlaps(gs, ps)
    f1s = []
    for thr in pm.SPELL_IOUS:
        t, f, m = pm.match_spells(ov, len(gs), len(ps), thr)
        f1s.append(2 * t / (2 * t + f + m) if t + f + m else 0.0)
    out["stopF1"] = float(np.mean(f1s))
    out["stops_true"], out["stops_pred"] = len(gs), len(ps)
    out["stop_s_true"] = float(np.mean([(b - a) / fps for a, b, _ in gs])) if gs else 0.0
    out["stop_s_pred"] = float(np.mean([(b - a) / fps for a, b, _ in ps])) if ps else 0.0
    hit = pm.matched_gt_flags(ov, len(gs), 0.5)
    dur = np.array([(b - a) / fps for a, b, _ in gs])
    for lo, hi in zip(LENGTH_BUCKETS_S[:-1], LENGTH_BUCKETS_S[1:]):
        sel = (dur >= lo) & (dur < hi)
        out[f"found_{lo:g}-{hi:g}s"] = float(hit[sel].mean()) if sel.any() else None
        out[f"n_{lo:g}-{hi:g}s"] = int(sel.sum())
    gt_tr, pr_tr = pm.transitions(gc, runs), pm.transitions(pc, runs)
    tol = int(round(TOL_S * fps))
    for name, to in (("start", D), ("end", H)):
        r = pm.transition_scores([t for t in gt_tr if t[2] == to], [t for t in pr_tr if t[2] == to], tol)
        errs = [abs(pp[0] - gg[0]) for gg, pp in r["pairs"]]
        out[f"{name}_f1"] = r["f1"]
        out[f"{name}_err_s"] = float(np.median(errs)) / fps if errs else None
    return out


TYPES = ("kick-off", "corner", "goal kick", "throw-in", "free kick", "unknown")


def restart_type(x: float, y: float, pitch) -> str:
    """Where play restarts says what the stoppage was (Laws of the Game). 'free kick' = everything
    else on the pitch: fouls AND offsides (both restart where they happened)."""
    L, W = pitch
    if not np.isfinite(x) or not np.isfinite(y):
        return "unknown"
    if np.hypot(x - L / 2, y - W / 2) < 3.0:
        return "kick-off"
    if min(np.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W)) < 3.0:
        return "corner"
    if (x < 7.0 or x > L - 7.0) and abs(y - W / 2) < 11.0:
        return "goal kick"
    if y < 2.5 or y > W - 2.5:
        return "throw-in"
    return "free kick"


def stoppage_types(gt: np.ndarray, clean_ball: np.ndarray, pitch, fps: float) -> list:
    """(start, end, type) of every true stoppage, typed by the CLEAN ball where play restarts (the first
    seen position from the restart up to 2 s later)."""
    out = []
    dd = (gt == "D").astype(np.int8)
    d = np.diff(np.r_[0, dd, 0])
    n2 = int(round(2 * fps))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        seg = clean_ball[e:e + n2]
        ok = np.flatnonzero(~np.isnan(seg[:, 0])) if len(seg) else []
        x, y = (seg[ok[0], 0], seg[ok[0], 1]) if len(ok) else (np.nan, np.nan)
        out.append((s, e, restart_type(float(x), float(y), pitch)))
    return out


def by_type(stops: list, dead: np.ndarray, fps: float) -> list:
    """Per true stoppage: its type, found (IoU >= 0.5 with one predicted stoppage), how late the
    prediction starts (s; NaN if it never does) and the share of it predicted dead."""
    rows = []
    n = len(dead)
    dpred = np.diff(np.r_[0, dead.astype(np.int8), 0])
    ps, pe = np.flatnonzero(dpred == 1), np.flatnonzero(dpred == -1)
    for s, e, typ in stops:
        best_iou = 0.0
        for a, b in zip(ps, pe):
            if b <= s or a >= e:
                continue
            best_iou = max(best_iou, (min(b, e) - max(a, s)) / (max(b, e) - min(a, s)))
        win = dead[max(0, s - int(2 * fps)):e]
        first = np.flatnonzero(win)
        late = (first[0] - min(s, int(2 * fps))) / fps if len(first) else np.nan
        rows.append({"type": typ, "found": best_iou >= 0.5, "late_s": late,
                     "covered": float(dead[s:min(e, n)].mean()) if e > s else 0.0, "length_s": (e - s) / fps})
    return rows


def detect_all(spec, trk, models, ref):
    """{detector: dead mask} for one game; each P(dead) is computed once. models: [(tag, model, tables)]."""
    from .. import dead
    out = {}
    if ref is not None:
        _, m0, t0 = models[0]
        pr = dead.predict(ref, trk.players, trk.team_of_slot, spec.fps, spec.pitch, spec.halves)
        out["players_frame"] = np.nan_to_num(pr, nan=0.0) >= 0.5
        out["players"] = decoder.decode(pr, spec.halves, spec.fps, t0, w=0.2, prior=m0["meta"]["prior"])
    for i, (tag, m, tables) in enumerate(models):
        p = model.predict(m, trk, spec)
        out[f"{tag}_frame"] = np.nan_to_num(p, nan=0.0) >= 0.5
        for w in WEIGHTS:
            out[f"{tag}_w{w:g}"] = decoder.decode(p, spec.halves, spec.fps, tables, w=w, prior=m["meta"]["prior"])
        if i == len(models) - 1:                      # extras for the newest model only
            for pr_x in (0.5, 0.55):                  # a STRICTER decision step: fewer false stoppages
                out[f"{tag}_prior{pr_x:g}"] = decoder.decode(p, spec.halves, spec.fps, tables, w=0.2, prior=pr_x)
            if ref is not None:                       # + João's players-only P(dead), only where it stays high
                for b in (0.5, 1.0):
                    pc = combine(p, pr, b, spec.fps)
                    out[f"{tag}+pl{b:g}"] = decoder.decode(pc, spec.halves, spec.fps, tables, w=0.2,
                                                           prior=m["meta"]["prior"])
    return out


def combine(p_ball: np.ndarray, p_players: np.ndarray, b: float, fps: float, smooth_s: float = 3.0) -> np.ndarray:
    """P from logit(p_ball) + b * logit(p_players averaged over smooth_s): a players-only signal that stays
    high for seconds can outweigh a moving ball. b is chosen on other games, never on our match."""
    eps = 1e-4
    lg = lambda x: np.log(np.clip(x, eps, 1 - eps)) - np.log(np.clip(1 - x, eps, 1 - eps))
    pp = pd.Series(np.nan_to_num(p_players, nan=0.5)).rolling(max(1, int(round(smooth_s * fps))), center=True,
                                                              min_periods=1).mean().to_numpy()
    z = lg(np.nan_to_num(p_ball, nan=0.5)) + b * lg(pp)
    out = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return np.where(np.isnan(p_ball), np.nan, out)


COLS_A = ["alive_dead_f1", "stopF1", "dead_precision", "dead_recall", "start_f1", "end_f1", "start_err_s", "end_err_s",
          "keeper_fp"]
COLS_B = ["keeper_hold_s", "stops_true", "stops_pred", "stop_s_true", "stop_s_pred", "found_0-5s", "found_5-15s", "found_15-40s",
          "found_40-infs"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="*", default=[str(MODELS_DIR / "deadball_v1.joblib")],
                    help="one or more models, compared side by side (detector names get the model's tag)")
    ap.add_argument("--conditions", nargs="*", default=["pff", "noisy"], choices=["pff", "noisy", "drift", "kept", "carried"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--stoppage-loss", type=float, default=0.8)
    ap.add_argument("--no-reference", action="store_true", help="skip João's players-only model")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args(argv)
    models = []
    for path in args.model:
        m = model.load(path)
        models.append((m["meta"]["tag"], m,
                       (decoder.length_tables(m["lengths"]["alive"]), decoder.length_tables(m["lengths"]["dead"]))))
    by_type_names = {"players", *(f"{t}_w0.2" for t, _, _ in models), f"{models[-1][0]}+pl1"}
    ref = None
    if not args.no_reference:
        from .. import dead
        ref = dead.load_model(MODELS_DIR / "dead_v1.joblib")
    t0, rows, trows = time.time(), [], []
    for cond in args.conditions:
        for source, spec, trk, gt, clean in games(cond, args.seeds, args.stoppage_loss):
            stops = stoppage_types(gt, clean, spec.pitch, spec.fps)
            from .features import keeper_v8                  # keeper holding while play is alive (Law 12)
            alive = (gt == "H") | (gt == "A")
            kh = (keeper_v8(trk.ball, trk.players, spec.fps, spec.pitch)["k_hold"].to_numpy() > 0) & alive
            for name, mask in detect_all(spec, trk, models, ref).items():
                rows.append({"condition": cond, "source": source, "game": spec.name, "detector": name,
                             **score(gt, mask, spec.fps),
                             "keeper_hold_s": float(kh.sum() / spec.fps),
                             "keeper_fp": float(mask[kh].mean()) if kh.any() else np.nan})
                if name in by_type_names:
                    trows += [{"condition": cond, "detector": name, "game": spec.name, **r}
                              for r in by_type(stops, mask, spec.fps)]
            print(f"  {cond:<6} {spec.name:<14} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    agg = (df.groupby(["condition", "detector", "source"])[COLS_A + COLS_B].mean()
             .groupby(["condition", "detector"]).mean())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== is the ball dead? (mean over games per source, then over sources) ===")
    print(agg[COLS_A].round(3).to_string())
    print()
    print(agg[COLS_B].round(2).to_string())
    if trows:
        t = pd.DataFrame(trows)
        tab = t.groupby(["condition", "type", "detector"]).agg(
            n=("found", "size"), found=("found", "mean"), late_s_median=("late_s", "median"),
            covered=("covered", "mean"), length_s=("length_s", "median"))
        print("\n=== by stoppage type (typed by where play restarts; 'free kick' = fouls + offsides) ===")
        print(tab.round(2).to_string())
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    out = HARNESS_DIR / f"deadball_eval_{args.tag}.json"
    out.write_text(json.dumps({"args": vars(args), "rows": rows, "by_type": trows}, indent=2, default=float))
    print(f"\nwrote {out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
