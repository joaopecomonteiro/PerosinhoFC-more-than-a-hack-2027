"""The learned layers for make_submission: P(dead) + P(alive <-> dead boundary) (mqf.alive models) and
P(home) + P(H <-> A boundary) (mqf.owner models), one match at a time.

Runs on the team's machine (the competition match never reaches this repository or an assistant).
For P(dead), `ball` must be the TRACKED ball -- before any fill; the model's own rule settings are used
for its rule columns. For P(home), the FILLED ball and its mask, as the owner experiments (#40).

float64=True (the submission path): torch models run on the CPU in float64 with one thread, so
last-bit differences between machines cannot move an output across its 1e-4 rounding (the same
reasoning as the ball fill's float64 + 1 cm rounding). Several models: the mean of their rounded
outputs, rounded again -- exactly the experiments' --ensemble.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import feats, rules
from .models import AliveModel


class _cpu64:
    """Context: one torch thread (deterministic reductions), restored afterwards."""

    def __enter__(self):
        import torch
        self.n = torch.get_num_threads()
        torch.set_num_threads(1)
        return self

    def __exit__(self, *exc):
        import torch
        torch.set_num_threads(self.n)
        return False


def _to_float64(model):
    """Swap a loaded torch model's net for a float64 CPU copy that casts its input (OwnerModel / AliveModel)."""
    if model.kind == "gbm":
        return model
    import torch

    class F64(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net.double()

        def forward(self, x):
            return self.net(x.double())

    model._net = F64(model._torch_net("cpu")).eval()
    return model


def alive_paths(alive_model) -> list[Path]:
    """None / a path / 'a.pt,b.pt' / a list of either -> [Path, ...]."""
    if not alive_model:
        return []
    items = alive_model if isinstance(alive_model, (list, tuple)) else [alive_model]
    out = []
    for it in items:
        out += [Path(x) for x in str(it).split(",") if x]
    return out


def p_dead(model_paths, players, team, ball, fps, pitch, halves, device: str = "cpu", float64: bool = False):
    """((n,) P(dead), (n,) P(boundary) or None, rule result) -- NaN outside the halves, rounded to 1e-4.
    Several models: the mean of their (rounded) outputs, rounded again -- exactly experiment.py's
    --ensemble, so the harness number is the submission's number."""
    paths = alive_paths(model_paths)
    if not paths:
        raise ValueError("no alive model")
    models = [AliveModel.load(p) for p in paths]
    if float64:
        models = [_to_float64(m) for m in models]
        device = "cpu"
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
        if float64 and m.kind != "gbm":
            with _cpu64():
                p, pb = m.predict(X, halves, len(X), fps, device)
        else:
            p, pb = m.predict(X, halves, len(X), fps, device)
        ps.append(p)
        if pb is not None:
            pbs.append(pb)
    if len(models) == 1:
        return ps[0], (pbs[0] if pbs else None), rr
    return (np.round(np.mean(np.stack(ps), axis=0), 4), np.round(np.mean(np.stack(pbs), axis=0), 4) if pbs else None,
            rr)


def owner_p_home(model_paths, players, team, ball_filled, filled, fps, pitch, halves, device: str = "cpu",
                 float64: bool = False):
    """((n,) P(home), (n,) P(H <-> A boundary) or None) of mqf.owner models (one or an ensemble: mean P(home),
    mean boundary), each averaged over both team orders as OwnerModel does -- what #40 scored."""
    from ..owner import feats as ofeats, rules as orules
    from ..owner.models import OwnerModel
    paths = alive_paths(model_paths)
    members = [OwnerModel.load(p) for p in paths]
    if float64:
        members = [_to_float64(m) for m in members]
        device = "cpu"
    n = len(players)
    filled = np.zeros(n, dtype=bool) if filled is None else np.asarray(filled, dtype=bool)
    need_seq = any(m.kind in ("gru", "tcn") for m in members)
    views = []
    for k in (0, 1):
        tm = np.asarray(team) if k == 0 else 1 - np.asarray(team)
        X = ofeats.features(players, tm, ball_filled, fps, pitch, halves, "full_rules_heading", orules.RuleConfig(),
                            ball_filled=filled)
        S, sc = ofeats.seq_inputs(players, tm, ball_filled, fps, pitch, halves, X_full=X, columns="all") \
            if need_seq else (None, None)
        views.append({"X": X, "S": S, "seq_columns": sc})
    ps, pbs = [], []
    for m in members:
        if float64 and m.kind != "gbm":
            with _cpu64():
                p, pb = m.p_home_and_boundary(views, halves, n, fps, device)
        else:
            p, pb = m.p_home_and_boundary(views, halves, n, fps, device)
        ps.append(p)
        if pb is not None:
            pbs.append(pb)
    return np.round(np.mean(ps, axis=0), 4), (np.round(np.mean(pbs, axis=0), 4) if pbs else None)


def is_who_joblib(path) -> bool:
    """True for a mqf.who model (who_v1: who.predict), False for a mqf.owner model."""
    path = Path(path)
    if path.suffix != ".joblib":
        return False
    import joblib
    return "kind" not in joblib.load(path)


def rules_only(players, team, ball, fps, pitch, halves, cfg: rules.RuleConfig | None = None) -> dict:
    """The rule result alone (for the snap / veto decoder variants on top of another P(dead))."""
    q = feats.frame_quantities(np.asarray(players), np.asarray(team), np.asarray(ball, dtype=np.float64), fps, pitch,
                               halves)
    return rules.apply(q, fps, pitch, halves, len(players), cfg)
