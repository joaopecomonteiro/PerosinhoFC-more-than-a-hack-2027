"""Turn clean tracking (player identities, full ball) into ATD-like tracking.

Every default below comes from a number the organisers PUBLISHED (DATA.md / CHALLENGE.md), never
from looking at the match itself:
  - ball missing on 41% of graded frames, concentrated where players crowd the ball;
  - 46 anonymous tracklet slots (23 per team) for 22 players, with gaps and identity switches;
  - mean 20.5 objects tracked during play (of 22), occasionally > 22 (duplicates).
Other parameters (burst lengths, break rates, jitter) are assumptions -- vary them with
`NoiseConfig` and check that conclusions survive (robustness, not calibration).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..atd import Tracking


@dataclass(frozen=True)
class NoiseConfig:
    ball_missing_target: float = 0.41   # DATA.md: share of graded frames without a ball
    ball_burst_mean_s: float = 1.2       # assumption: mean length of a ball-loss burst
    crowd_radius_m: float = 3.0
    crowd_weight: float = 3.0            # bursts start this many times more often in crowds
    slots_per_team: int = 23             # DATA.md: 46 slots
    track_break_per_min: float = 1.2     # assumption: identity switches per player-minute
    dropout_share: float = 0.07          # DATA.md: 20.5 of 22 tracked on average
    dropout_burst_mean_s: float = 0.6
    duplicate_per_min: float = 0.3       # assumption: occasional > 22 objects
    duplicate_mean_s: float = 1.0
    pos_jitter_m: float = 0.25
    ball_jitter_m: float = 0.4
    jitter_corr_s: float = 0.5           # assumption: tracker error is smooth, not white (AR(1) e-folding time)
    speed_window_s: float = 0.4          # assumption: tracker speed = centred difference over this span
    seed: int = 0


def _ar1(rng, shape, sigma: float, corr_frames: float) -> np.ndarray:
    """Stationary AR(1) noise along axis 0 (std `sigma`, e-folding time `corr_frames`).

    White per-frame jitter differentiates into ~10 m/s of fake speed at 25 fps, which buries the
    "players stand still" stoppage cue; a tracker's error drifts smoothly instead.
    """
    rho = float(np.exp(-1.0 / max(corr_frames, 1e-6)))
    flat = rng.normal(0.0, 1.0, (shape[0], int(np.prod(shape[1:], dtype=np.int64))))
    # ewm(adjust=False): y_t = rho*y_{t-1} + (1-rho)*x_t, so scaling x_t keeps var(y_t) = sigma^2
    x = flat * sigma * np.sqrt((1 + rho) / (1 - rho))
    x[0] = flat[0] * sigma                                   # start in the stationary distribution
    y = pd.DataFrame(x).ewm(alpha=1 - rho, adjust=False).mean().to_numpy()
    return y.reshape(shape)


def _speed(xy: np.ndarray, fps: int, window_s: float) -> np.ndarray:
    """|centred difference| over +/- window/2 -- what a tracker reports as speed. NaN at gaps."""
    w = max(1, int(round(window_s * fps / 2)))
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if len(xy) > 2 * w:
        out[w:-w] = np.linalg.norm(xy[2 * w:] - xy[:-2 * w], axis=-1) * fps / (2 * w)
    return out


def _bursts(u: np.ndarray, lengths: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Boolean mask: a burst of lengths[t] frames starts wherever u[t] < p[t] (works on 1D or 2D)."""
    n = u.shape[0]
    starts = u < p
    diff = np.zeros((n + 1,) + u.shape[1:], dtype=np.int32)
    idx = np.nonzero(starts)
    diff[idx] += 1
    ends = np.minimum(idx[0] + lengths[idx], n)
    np.add.at(diff, (ends,) + idx[1:], -1)
    return np.cumsum(diff, axis=0)[:n] > 0


def drop_ball(ball, players, graded, fps, cfg: NoiseConfig, rng) -> np.ndarray:
    """Bursty ball loss, likelier in crowds, calibrated to the target share on graded frames."""
    with np.errstate(invalid="ignore"):
        near = np.sum(np.linalg.norm(players - ball[:, None, :], axis=2) < cfg.crowd_radius_m, axis=1)
    weight = np.where(near >= 2, cfg.crowd_weight, 1.0)
    u = rng.random(len(ball))
    lengths = np.maximum(1, rng.exponential(cfg.ball_burst_mean_s * fps, len(ball)).astype(int))
    base_missing = np.isnan(ball[:, 0])
    lo, hi = 0.0, 1.0 / (cfg.ball_burst_mean_s * fps)
    for _ in range(40):                                   # bisection on the start rate (monotone)
        mid = (lo + hi) / 2
        miss = _bursts(u, lengths, mid * weight) | base_missing
        if miss[graded].mean() < cfg.ball_missing_target:
            lo = mid
        else:
            hi = mid
    miss = _bursts(u, lengths, hi * weight) | base_missing
    out = ball + _ar1(rng, ball.shape, cfg.ball_jitter_m, cfg.jitter_corr_s * fps)
    out[miss] = np.nan
    return out


def _segments_of(on: np.ndarray, break_p: float, rng):
    """[(start, stop)] runs of `on`, additionally cut at random break points."""
    edges = np.diff(np.r_[0, on.astype(np.int8), 0])
    out = []
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        cuts = np.flatnonzero(rng.random(e - s) < break_p)
        bounds = np.r_[s, s + cuts[cuts > 0], e]
        out += [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    return out


def corrupt(players: np.ndarray, team_of_player: np.ndarray, ball: np.ndarray,
            graded: np.ndarray, fps: int, cfg: NoiseConfig = NoiseConfig()) -> Tracking:
    """players (n, P, 2) metres with NaN off-pitch; team_of_player (P,) 0/1; ball (n, 2) metres."""
    rng = np.random.default_rng(cfg.seed)
    n, P, _ = players.shape
    n_slots = 2 * cfg.slots_per_team
    pl = players + _ar1(rng, players.shape, cfg.pos_jitter_m, cfg.jitter_corr_s * fps)
    # speed follows the physical track (a tracker smooths over its own short gaps), per player,
    # so an identity switch inside a slot does not create a speed spike
    pl_speed = _speed(pl, fps, cfg.speed_window_s)

    # momentary dropouts (bursty), calibrated roughly: share = start_p * mean_len
    mean_len = max(1.0, cfg.dropout_burst_mean_s * fps)
    u = rng.random((n, P))
    lengths = np.maximum(1, rng.exponential(mean_len, (n, P)).astype(int))
    pl[_bursts(u, lengths, np.full((n, P), cfg.dropout_share / mean_len))] = np.nan

    # identity: each player's on-pitch time is cut into tracklets; each tracklet takes a free slot
    on = ~np.isnan(pl[..., 0])
    break_p = cfg.track_break_per_min / (60.0 * fps)
    pieces = sorted((s, e, p) for p in range(P) for s, e in _segments_of(on[:, p], break_p, rng))
    xy = np.full((n, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n, n_slots), np.nan, dtype=np.float32)
    busy_until = np.zeros(n_slots, dtype=np.int64)
    team_of_slot = np.repeat(np.array([0, 1], dtype=np.int8), cfg.slots_per_team)
    for s, e, p in pieces:
        team = int(team_of_player[p])
        free = np.flatnonzero((team_of_slot == team) & (busy_until <= s))
        if not len(free):
            continue                                       # tracker has no slot: player unseen
        k = int(free[rng.integers(len(free))])
        xy[s:e, k] = pl[s:e, p]
        speed[s:e, k] = pl_speed[s:e, p]
        busy_until[k] = e

    # duplicates: a second track on an existing player for a short while
    n_dup = rng.poisson(cfg.duplicate_per_min * n / (60.0 * fps))
    for _ in range(n_dup):
        t, p = int(rng.integers(n)), int(rng.integers(P))
        if not on[t, p]:
            continue
        e = min(n, t + max(1, int(rng.exponential(cfg.duplicate_mean_s * fps))))
        team = int(team_of_player[p])
        free = np.flatnonzero((team_of_slot == team) & (busy_until <= t))
        if len(free):
            k = int(free[0])
            xy[t:e, k] = pl[t:e, p] + rng.normal(0, 0.8, 2)
            speed[t:e, k] = pl_speed[t:e, p]
            busy_until[k] = e

    speed[np.isnan(xy[..., 0])] = np.nan
    ball_obs = drop_ball(ball, players, graded, fps, cfg, rng)
    return Tracking(players=xy, player_speed=speed, ball=ball_obs.astype(np.float32),
                    team_of_slot=team_of_slot)
