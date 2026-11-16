"""The alive layer for make_submission: P(dead), P(boundary) and the rule result of one match.

Runs on the team's machine (the competition match never reaches this repository or an assistant).
`ball` must be the TRACKED ball -- before any fill. The model's own rule settings are used for its rule
columns, so a full_rules model sees the columns it was trained on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import feats, rules
from .models import AliveModel


def alive_paths(alive_model) -> list[Path]:
    """None / a path / 'a.pt,b.pt' / a list of either -> [Path, ...]."""
    if not alive_model:
        return []
    items = alive_model if isinstance(alive_model, (list, tuple)) else [alive_model]
    out = []
    for it in items:
        out += [Path(x) for x in str(it).split(",") if x]
    return out


def p_dead(model_paths, players, team, ball, fps, pitch, halves, device: str = "cpu"):
    """((n,) P(dead), (n,) P(boundary) or None, rule result) -- NaN outside the halves, rounded to 1e-4.
    Several models: the mean of their (rounded) outputs, rounded again -- exactly experiment.py's
    --ensemble, so the harness number is the submission's number."""
    paths = alive_paths(model_paths)
    if not paths:
        raise ValueError("no alive model")
    models = [AliveModel.load(p) for p in paths]
    tables = {}
    ps, pbs, rr = [], [], None
    for m in models:
        key = json.dumps(m.rule_cfg or {}, sort_keys=True)
        if key not in tables:
            cfg = rules.RuleConfig(**m.rule_cfg) if m.rule_cfg else rules.RuleConfig()
            tables[key] = feats.features(np.asarray(players), np.asarray(team), np.asarray(ball, dtype=np.float64), fps,
                                         pitch, halves, "full_rules", cfg, return_rules=True)
        X, rr_m = tables[key]
        rr = rr if rr is not None else rr_m
        p, pb = m.predict(X, halves, len(X), fps, device)
        ps.append(p)
        if pb is not None:
            pbs.append(pb)
    if len(models) == 1:
        return ps[0], (pbs[0] if pbs else None), rr
    return (np.round(np.mean(np.stack(ps), axis=0), 4), np.round(np.mean(np.stack(pbs), axis=0), 4) if pbs else None,
            rr)


def rules_only(players, team, ball, fps, pitch, halves, cfg: rules.RuleConfig | None = None) -> dict:
    """The rule result alone (for the snap / veto decoder variants on top of another P(dead))."""
    q = feats.frame_quantities(np.asarray(players), np.asarray(team), np.asarray(ball, dtype=np.float64), fps, pitch,
                               halves)
    return rules.apply(q, fps, pitch, halves, len(players), cfg)
