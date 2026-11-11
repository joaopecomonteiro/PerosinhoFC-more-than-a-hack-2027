"""Other games -> training windows for the ball model, with the masks living HERE (files untouched).

A window is 48 s at 6.25 Hz (every 4th frame of a 25 fps game, as TranSPORTmer), inside one half.
Every sample is the same tensor layout for both architectures:

  agents   (T, A, 4)  x, y in [-1, 1] (pitch-normalised), vx, vy in units of 10 m/s; 0 where absent
  present  (T, A)     agent visible at that step
  team     (A,)       0 home, 1 away, -1 padding
  ball_in  (T, 2)     the ball the model may see (normalised), 0 where hidden
  ball_obs (T,)       visible to the model
  target   (T, 2)     true ball in metres (NaN where the data has none)
  w_masked (T,)       1 where the loss applies to a HIDDEN alive ball (what we impute)
  w_seen   (T,)       1 where the loss applies to a VISIBLE alive ball (reconstruction, weight 0.1)

Only alive frames count (Sigmar: the model is used only while the ball is in play). Ball masks:
1-3 spans per window, length log-uniform 0.2-30 s (the range of BAR-ATH's ball gaps in the
profile), and 15 % fully hidden windows. Player mask variant ("on"): dropout bursts and tracks cut
into pieces (each piece its own agent), ranges from DATA.md / profile (~20.5 of 22 tracked,
p5 14, tracklets of seconds to minutes). Augmentation: mirror x and / or y.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HZ = 6.25
WINDOW_S = 48.0
STRIDE_S = 12.0
A_MAX = 48                          # agents per window (players or player pieces), ball excluded
SPAN_S = (0.2, 30.0)
P_FULL_MASK = 0.15
DROP_SHARE = (0.05, 0.35)
DROP_BURST_S = (1.0, 5.0)
PIECE_MEAN_S = (5.0, 30.0)
V_SCALE = 10.0


@dataclass
class Game:
    name: str
    source: str
    fps: int
    pitch: tuple
    players: np.ndarray          # (n, P, 2) metres, NaN absent
    team: np.ndarray             # (P,)
    ball: np.ndarray             # (n, 2) metres, NaN absent
    alive: np.ndarray            # (n,) bool
    halves: tuple
    labels: np.ndarray = field(default=None, repr=False)

    @property
    def step(self) -> int:
        return max(1, int(round(self.fps / HZ)))


def load_game(bundle_dir: Path) -> Game:
    from .. import atd
    from ..external import bundle
    spec = bundle.spec(bundle_dir)
    trk = atd.load(spec, cache_dir=Path(bundle_dir) / "cache")
    labels = pd.read_csv(spec.gt_public, dtype={"label": str})["label"].to_numpy()
    source = json.loads(next(Path(bundle_dir).glob("*_game.json")).read_text(encoding="utf-8"))["source"]
    return Game(spec.name, source, spec.fps, tuple(spec.pitch), trk.players.astype(np.float32),
                trk.team_of_slot.astype(np.int8), trk.ball.astype(np.float32), np.isin(labels, ["H", "A"]),
                spec.halves, labels)


def game_from_match(m: dict, name: str = "SYN") -> Game:
    """A harness-style clean match (synthetic.clean_match, harness.idsse) as a Game."""
    return Game(name, "synthetic", m["fps"], tuple(m["pitch"]), np.asarray(m["players"], dtype=np.float32),
                np.asarray(m["team_of_player"], dtype=np.int8), np.asarray(m["ball"], dtype=np.float32),
                np.isin(m["labels"], ["H", "A"]), tuple(m["halves"]), np.asarray(m["labels"]))


def load_split(part: str, sources=("idsse", "metrica")) -> list[Game]:
    from ..external import bundle, splits
    names = splits.games(part, sources=list(sources))
    games = []
    for d in bundle.find(names=names):
        g = load_game(d)
        if g.fps * 1.0 / g.step != HZ:
            continue                                   # 6.25 Hz needs a 25 fps source (not SkillCorner)
        games.append(g)
    return games


# ------------------------------------------------------------------------------------ windows

def window_starts(g: Game, stride_s: float = STRIDE_S, window_s: float = WINDOW_S) -> list[tuple[int, int]]:
    """[(half_index, start_frame)] of windows fully inside a half, in native frames."""
    T = int(round(window_s * HZ))
    span = T * g.step
    stride = int(round(stride_s * HZ)) * g.step
    out = []
    for h, (_, a, b) in enumerate(g.halves):
        if b - a + 1 < span:
            continue
        out += [(h, s) for s in range(a, b + 2 - span, stride)]
    return out


def _velocity(xy: np.ndarray, hz: float) -> np.ndarray:
    """(T, A, 2) m/s, central where both neighbours exist, one-sided at piece ends, 0 if alone."""
    v = np.zeros_like(xy)
    nxt, prv = xy[2:], xy[:-2]
    c = (nxt - prv) * hz / 2
    v[1:-1] = c
    fwd = np.vstack([(xy[1:] - xy[:-1]) * hz, np.full((1,) + xy.shape[1:], np.nan)])
    bwd = np.vstack([np.full((1,) + xy.shape[1:], np.nan), (xy[1:] - xy[:-1]) * hz])
    v = np.where(np.isnan(v), fwd, v)
    v = np.where(np.isnan(v), bwd, v)
    return np.nan_to_num(v)


def split_pieces(xy: np.ndarray, team: np.ndarray, extra_cuts=None):
    """Every column cut at its gaps (and at extra_cuts[k] steps): each piece its own agent."""
    cols, teams = [], []
    T = xy.shape[0]
    for k in range(xy.shape[1]):
        on = ~np.isnan(xy[:, k, 0])
        if extra_cuts is not None and len(extra_cuts[k]):
            on_cut = on.copy()
            on_cut[np.asarray(extra_cuts[k], dtype=np.int64)] = False     # one step of daylight at a cut
            on = on_cut
        e = np.diff(np.r_[0, on.astype(np.int8), 0])
        for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            c = np.full((T, 2), np.nan, dtype=xy.dtype)
            c[s:t] = xy[s:t, k]
            cols.append(c)
            teams.append(team[k])
    if not cols:
        return np.full((T, 0, 2), np.nan, dtype=xy.dtype), np.zeros(0, dtype=np.int8)
    return np.stack(cols, axis=1), np.array(teams, dtype=np.int8)


def _player_mask(xy: np.ndarray, rng, hz: float) -> np.ndarray:
    """Dropout bursts, in place on a copy (the 'on' variant)."""
    T, P, _ = xy.shape
    out = xy.copy()
    share = rng.uniform(*DROP_SHARE)
    mean_len = rng.uniform(*DROP_BURST_S) * hz
    starts = rng.random((T, P)) < share / mean_len
    for t, k in zip(*np.nonzero(starts)):
        out[t:t + max(1, int(rng.exponential(mean_len))), k] = np.nan
    return out


def make_sample(g: Game, h: int, s: int, rng, player_mask: bool, train: bool = True,
                window_s: float = WINDOW_S) -> dict:
    T = int(round(window_s * HZ))
    idx = s + np.arange(T) * g.step
    L, W = g.pitch
    xy = g.players[idx].astype(np.float64)
    keep = ~np.all(np.isnan(xy[..., 0]), axis=0)
    xy, team = xy[:, keep], g.team[keep]
    if player_mask:
        xy = _player_mask(xy, rng, HZ)
        mean_piece = rng.uniform(*PIECE_MEAN_S) * HZ
        cuts = [np.flatnonzero(rng.random(T) < 1.0 / mean_piece) for _ in range(xy.shape[1])]
        xy, team = split_pieces(xy, team, cuts)
    # keep the A_MAX longest-lived agents
    life = (~np.isnan(xy[..., 0])).sum(axis=0)
    order = np.argsort(-life, kind="stable")[:A_MAX]
    order = order[life[order] > 0]
    xy, team = xy[:, order], team[order]
    vel = _velocity(xy, HZ)
    if train:
        fx, fy = rng.random() < 0.5, rng.random() < 0.5
    else:
        fx = fy = False
    ball = g.ball[idx].astype(np.float64)
    alive = g.alive[idx]
    gt_ok = ~np.isnan(ball[:, 0]) & alive
    obs = gt_ok.copy()                                  # dead frames never shown (used only alive)
    if rng.random() < P_FULL_MASK:
        obs[:] = False
    else:
        for _ in range(int(rng.integers(1, 4))):
            n = max(1, int(round(np.exp(rng.uniform(np.log(SPAN_S[0]), np.log(SPAN_S[1]))) * HZ)))
            a = int(rng.integers(0, max(1, T - n)))
            obs[a:a + n] = False
    return _pack(xy, vel, team, ball, obs, gt_ok, (L, W), fx, fy)


def _pack(xy, vel, team, ball, obs, gt_ok, pitch, fx=False, fy=False) -> dict:
    L, W = pitch
    T, A = xy.shape[:2]
    present = ~np.isnan(xy[..., 0])
    nx = np.stack([xy[..., 0] / L * 2 - 1, xy[..., 1] / W * 2 - 1], axis=-1)
    nv = vel / V_SCALE
    nb = np.stack([ball[:, 0] / L * 2 - 1, ball[:, 1] / W * 2 - 1], axis=-1)
    target = ball.copy()
    if fx:
        nx[..., 0] *= -1; nv[..., 0] *= -1; nb[:, 0] *= -1; target[:, 0] = L - target[:, 0]
    if fy:
        nx[..., 1] *= -1; nv[..., 1] *= -1; nb[:, 1] *= -1; target[:, 1] = W - target[:, 1]
    agents = np.zeros((T, A_MAX, 4), dtype=np.float32)
    agents[:, :A, :2] = np.nan_to_num(nx)
    agents[:, :A, 2:] = np.where(present[..., None], nv, 0.0)
    pres = np.zeros((T, A_MAX), dtype=bool)
    pres[:, :A] = present
    tm = np.full(A_MAX, -1, dtype=np.int64)
    tm[:A] = team
    return {"agents": agents, "present": pres, "team": tm,
            "ball_in": np.where(obs[:, None], np.nan_to_num(nb), 0.0).astype(np.float32),
            "ball_obs": obs, "target": target.astype(np.float32),
            "w_masked": (gt_ok & ~obs).astype(np.float32), "w_seen": (gt_ok & obs).astype(np.float32),
            "pitch": np.array([L, W], dtype=np.float32), "flip": np.array([fx, fy])}


def collate(samples: list[dict]) -> dict:
    """Stack, then drop agent columns no sample uses (agents are packed from column 0), so the axial
    model never pays for padding."""
    out = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
    used = np.flatnonzero(out["present"].any(axis=(0, 1)))
    k = int(used[-1]) + 1 if len(used) else 1
    out["agents"], out["present"], out["team"] = out["agents"][:, :, :k], out["present"][:, :, :k], out["team"][:, :k]
    return out
