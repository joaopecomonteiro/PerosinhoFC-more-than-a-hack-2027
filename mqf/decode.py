"""Frame evidence -> possession spells.

v0 = hand-set emission model + 3-state HMM (H, A, D) decoded with Viterbi, per half.
The transition costs are what turn flickering frame guesses into whole possessions, which is what
spellF1 rewards. Swap `emissions()` for a learned classifier later; keep the decoder.

Parameters are priors chosen from football knowledge, NOT fitted to the public 5 minutes
(that is against the rules and, per EVALUATION.md, picks the wrong answer anyway).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LABELS
from .smart_tagging import TagEvidence


@dataclass
class Params:
    tau_m: float = 1.5            # metres: softness of "who is closer to the ball"
    loose_m: float = 4.0          # beyond this nobody controls the ball -> evidence fades
    still_speed: float = 1.3      # m/s mean player speed typical of a stoppage
    still_width: float = 0.35
    w_seen: float = 1.0           # evidence weight when the ball is tracked
    w_unseen: float = 0.15        # ... when it is not (let the HMM carry the state)
    w_filled: float = 0.4         # "who" weight on an imputed ball (alive/dead stays at w_unseen there); EXP #18
    w_spread: float = 0.15        # team-shape hint when the ball is missing
    w_tags: float = 0.4           # Smart Tagging log-odds weight per vote
    switch_cost: float = 18.0     # H <-> A
    to_dead_cost: float = 10.0    # H/A -> D
    from_dead_cost: float = 10.0  # D -> H/A
    w_dead: float = 1.0           # weight of the learned P(dead) (mqf.dead) when features carry it
    dead_who: str = "zero"        # D's "who" term: "zero" (v0: uncertainty about who quietly favours D)
                                  # or "max" (who-evidence never favours or penalises D)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def emissions(feat: pd.DataFrame, p: Params, tags: TagEvidence | None = None) -> np.ndarray:
    """(n, 3) log-likelihoods for H, A, D."""
    eps = 1e-4
    seen = feat["ball_seen"].to_numpy()
    filled = feat["ball_filled"].to_numpy() if "ball_filled" in feat else np.zeros(len(feat), dtype=bool)
    tracked = seen & ~filled
    d_min = np.fmin(feat["d_home"].to_numpy(), feat["d_away"].to_numpy())
    d_diff = np.nan_to_num(feat["d_diff"].to_numpy())

    # who: closer team, faded when the ball is loose
    control = np.where(seen, _sig((p.loose_m - np.nan_to_num(d_min, nan=99.0)) / 1.0), 0.0)
    logit_h = control * d_diff / p.tau_m
    logit_h += p.w_spread * np.nan_to_num(feat["spread_diff"].to_numpy()) * (~tracked)
    if tags is not None:
        logit_h += p.w_tags * (tags.vote[:, 0] - tags.vote[:, 1])
    p_h = np.clip(_sig(logit_h), eps, 1 - eps)

    # alive vs dead: players standing still (v0 rule, or the learned P(dead)), or ball clearly off the pitch
    ms = np.nan_to_num(feat["mean_player_speed"].to_numpy(), nan=p.still_speed)
    p_dead = _sig((p.still_speed - ms) / p.still_width)
    learned = "p_dead_model" in feat and bool(feat["p_dead_model"].notna().any())
    if learned:
        pdm = feat["p_dead_model"].to_numpy(dtype=np.float64)
        p_dead = np.where(np.isnan(pdm), p_dead, pdm)
    p_dead = np.where(feat["ball_out"].to_numpy(), 0.97, p_dead)
    p_dead = np.clip(p_dead, eps, 1 - eps)

    if not filled.any() and not learned and p.dead_who == "zero":      # v0, bit for bit
        ll = np.stack([np.log(1 - p_dead) + np.log(p_h),
                       np.log(1 - p_dead) + np.log(1 - p_h),
                       np.log(p_dead)], axis=1)
        return ll * np.where(seen, p.w_seen, p.w_unseen)[:, None]
    alive_ll = np.stack([np.log(1 - p_dead), np.log(1 - p_dead), np.log(p_dead)], axis=1)
    d_who = np.maximum(np.log(p_h), np.log(1 - p_h)) if p.dead_who == "max" else np.zeros_like(p_h)
    who_ll = np.stack([np.log(p_h), np.log(1 - p_h), d_who], axis=1)
    if learned:                       # the learned P(dead) comes from the players: same weight with or without a ball
        w_ad = np.where(feat["ball_out"].to_numpy(), p.w_seen, p.w_dead)[:, None]
    else:
        w_ad = np.where(tracked, p.w_seen, p.w_unseen)[:, None]                 # an imputed ball says nothing
    w_who = np.where(filled, p.w_filled, np.where(tracked, p.w_seen, p.w_unseen))[:, None]  # about dead
    return alive_ll * w_ad + who_ll * w_who


def viterbi(ll: np.ndarray, p: Params) -> np.ndarray:
    T = np.array([[0.0, -p.switch_cost, -p.to_dead_cost],
                  [-p.switch_cost, 0.0, -p.to_dead_cost],
                  [-p.from_dead_cost, -p.from_dead_cost, 0.0]])
    n = len(ll)
    back = np.zeros((n, 3), dtype=np.int8)
    score = ll[0] + np.array([0.0, 0.0, 0.0])
    for t in range(1, n):
        cand = score[:, None] + T            # from i (rows) to j (cols)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + ll[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(feat: pd.DataFrame, halves, n_frames: int, p: Params,
           tags: TagEvidence | None = None) -> np.ndarray:
    """Label array of length n_frames; 'X' outside the graded halves."""
    labels = np.full(n_frames, "X", dtype="<U1")
    ll = emissions(feat, p, tags)
    for _, first, last in halves:
        path = viterbi(ll[first:last + 1], p)
        labels[first:last + 1] = np.array(LABELS)[path]
    return labels


def spell_stats(labels: np.ndarray, fps: int) -> dict:
    """Number and mean duration of possessions (maximal H or A runs) -- sanity, not a score."""
    lab = labels[labels != "X"]
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(lab)]
    poss = [(s, e) for s, e in zip(starts, ends) if lab[s] in ("H", "A")]
    dur = np.array([e - s for s, e in poss]) / fps if poss else np.array([0.0])
    return {"possessions": len(poss), "mean_s": float(dur.mean()), "median_s": float(np.median(dur))}
