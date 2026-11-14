"""Per-frame features from (noisy) tracking. Vectorised; no per-player identity is assumed."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .atd import Tracking


def _team_min_dist(players: np.ndarray, ball: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(players[:, mask, :] - ball[:, None, :], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(d, axis=1)          # NaN where ball or whole team is missing


def _team_spread(players: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = players[:, mask, :]
        return np.nanstd(p[..., 0], axis=1) + np.nanstd(p[..., 1], axis=1)


def _run_length(flag: np.ndarray) -> np.ndarray:
    """For each frame, how many consecutive frames (including this one) `flag` has been True."""
    out = np.zeros(len(flag), dtype=np.int32)
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        out[i] = run
    return out


def build(trk: Tracking, pitch: tuple[float, float], fps: int, ball_filled: np.ndarray | None = None,
          p_dead: np.ndarray | None = None) -> pd.DataFrame:
    """ball_filled: frames whose ball was imputed (mqf.ball). They count as seen for "who is closer",
    never for ball_out or the missing-ball run (the decoder weights them with Params.w_filled).
    p_dead: the learned per-frame P(dead) (mqf.dead) -- replaces v0's speed rule in the decoder."""
    length, width = pitch
    home = trk.team_of_slot == 0
    away = ~home
    ball = trk.ball
    filled = np.zeros(len(ball), dtype=bool) if ball_filled is None else np.asarray(ball_filled, dtype=bool)

    vel = np.vstack([[np.nan, np.nan], np.diff(ball, axis=0)]) * fps
    ball_speed = pd.Series(np.linalg.norm(vel, axis=1)).rolling(5, center=True, min_periods=1).median()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_speed = np.nanmean(trk.player_speed, axis=1)

    margin = 0.5
    bx, by = ball[:, 0], ball[:, 1]
    ball_out = (bx < -margin) | (bx > length + margin) | (by < -margin) | (by > width + margin)

    seen = trk.ball_seen
    tracked = seen & ~filled
    df = pd.DataFrame({
        "ball_seen": seen,
        "ball_filled": filled & seen,
        "ball_x": bx, "ball_y": by,
        "ball_speed": ball_speed.to_numpy(),
        "ball_out": ball_out & tracked,
        "d_home": _team_min_dist(trk.players, ball, home),
        "d_away": _team_min_dist(trk.players, ball, away),
        "spread_home": _team_spread(trk.players, home),
        "spread_away": _team_spread(trk.players, away),
        "mean_player_speed": mean_speed,
        "n_tracked": np.sum(~np.isnan(trk.players[..., 0]), axis=1),
        "ball_missing_run": _run_length(~tracked),
    })
    if p_dead is not None:
        df["p_dead_model"] = np.asarray(p_dead, dtype=np.float64)
    df["d_diff"] = df["d_away"] - df["d_home"]          # > 0 -> home closer
    df["spread_diff"] = df["spread_home"] - df["spread_away"]  # > 0 -> home more spread (attacking shape)
    return df
