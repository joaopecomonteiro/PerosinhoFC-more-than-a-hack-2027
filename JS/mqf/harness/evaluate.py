"""Run our pipeline on a (noised) clean match and score it with the OFFICIAL metric code.

The metric is never reimplemented: everything below calls possession_metrics.py from vendor/.
On top of the official numbers we report spell match rate by TRUE possession length, built from
the official building blocks (possession_spells / spell_overlaps / matched_gt_flags), because
every possession counts once and short ones are where the points hide.
"""
from __future__ import annotations

import numpy as np

from .. import decode, features
from ..config import official
from .noise import NoiseConfig, corrupt

LENGTH_BUCKETS_S = (0, 3, 10, 30, np.inf)


def predict(match: dict, params: decode.Params, cfg: NoiseConfig) -> np.ndarray:
    trk = corrupt(match["players"], match["team_of_player"], match["ball"],
                  match["labels"] != "X", match["fps"], cfg)
    feat = features.build(trk, match["pitch"], match["fps"])
    return decode.decode(feat, match["halves"], len(match["labels"]), params, tags=None)


def by_length(pm, gt, pred, fps):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    g = pm.possession_spells(gt, runs)
    p = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(g, p)
    hit = np.vstack([pm.matched_gt_flags(ov, len(g), t) for t in pm.SPELL_IOUS]).mean(axis=0)
    dur = np.array([(b - a) / fps for a, b, _ in g])
    out = {}
    for lo, hi in zip(LENGTH_BUCKETS_S[:-1], LENGTH_BUCKETS_S[1:]):
        m = (dur >= lo) & (dur < hi)
        key = f"{lo:g}-{hi:g}s"
        out[key] = {"n": int(m.sum()), "match_rate": float(hit[m].mean()) if m.any() else None}
    return out


def score(gt_labels: np.ndarray, pred_labels: np.ndarray, fps: int) -> dict:
    _, pm = official()
    gt, pred = pm.label_codes(gt_labels), pm.label_codes(pred_labels)
    r = pm.evaluate(gt, pred)
    return {
        "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
        "n_gt_spells": r["n_gt_spells"], "n_pred_spells": r["n_pred_spells"],
        "gt_dur_s": r["gt_mean_spell_duration_s"], "pred_dur_s": r["mean_spell_duration_s"],
        "alive_dead_f1": r["alive_dead_f1"], "team_acc_both_alive": r["team_acc_both_alive"],
        "transF1_12": r["transitions"][12]["f1"], "transF1_25": r["transitions"][25]["f1"],
        "by_length": by_length(pm, gt, pred, fps),
    }
