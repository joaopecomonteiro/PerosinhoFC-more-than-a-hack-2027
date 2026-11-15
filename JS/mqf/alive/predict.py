"""The alive layer for make_submission: P(dead), P(boundary) and the rule result of one match.

Runs on the team's machine (the competition match never reaches this repository or an assistant).
`ball` must be the TRACKED ball -- before any fill. The model's own rule settings are used for its rule
columns, so a full_rules model sees the columns it was trained on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import feats, rules
from .models import AliveModel


def p_dead(model_path, players, team, ball, fps, pitch, halves, device: str = "cpu"):
    """((n,) P(dead), (n,) P(boundary) or None, rule result) -- NaN outside the halves, rounded to 1e-4."""
    m = AliveModel.load(Path(model_path))
    cfg = rules.RuleConfig(**m.rule_cfg) if m.rule_cfg else rules.RuleConfig()
    X, rr = feats.features(np.asarray(players), np.asarray(team), np.asarray(ball, dtype=np.float64), fps, pitch,
                           halves, "full_rules", cfg, return_rules=True)
    p, pb = m.predict(X, halves, len(X), fps, device)
    return p, pb, rr


def rules_only(players, team, ball, fps, pitch, halves, cfg: rules.RuleConfig | None = None) -> dict:
    """The rule result alone (for the snap / veto decoder variants on top of another P(dead))."""
    q = feats.frame_quantities(np.asarray(players), np.asarray(team), np.asarray(ball, dtype=np.float64), fps, pitch,
                               halves)
    return rules.apply(q, fps, pitch, halves, len(players), cfg)
