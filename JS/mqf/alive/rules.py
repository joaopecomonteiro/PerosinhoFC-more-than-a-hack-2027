"""Law-based stoppage rules: a hand-made alive layer, and rule columns for the stacked models.

A two-state machine per half (Vidal-Codina et al. 2022 detect set pieces with the same kind of
trigger; ELASTIC finds the restart kick as a ball-speed jump). Every half STARTS DEAD: it opens with a
kickoff (Law 8), and the dead ball before it is part of the graded window on BAR-ATH's 2nd half
(DATA.md). Football knowledge, not a frame constant: the machine leaves that state at the first restart
it sees, wherever it is.

  enter dead (any of, held for `hold_s` where marked *):
    out       the TRACKED ball is beyond a line by > out_margin_m (Law 9)
    spot      the tracked ball is still (< 0.5 m over 1 s) for >= spot_still_s on a restart spot -- a
              touchline (throw-in), a corner mark (corner), a goal area (goal kick), the centre spot
              (kickoff) -- or anywhere for >= 2 x spot_still_s while the players are slow (free kick)
    throw*    a player beyond a touchline while the players are slow (Law 15)
    corner*   a player within corner_m of a corner mark and >= corner_box players in a box (Law 17)
    goalkick* a player in a goal area, <= gk_box_max players in that penalty area, players slow (Law 16:
              the opponents stand outside the penalty area)
    kickoff*  >= kickoff_own_half of the players in their own half and one on the centre spot (Law 8)
    still     median player speed < still_med_ms and nobody fast (p95 < still_p95_ms) for >= still_min_s;
              the stoppage is dated back to the start of that still run
  leave dead (a restart):
    kick      the tracked ball jumps from < 2 to > 7 m/s within 0.4 s
    burst     >= burst_n players above burst_ms (from fewer the frame before)
  ... except while the ball is still seen out or still on its spot (a restart needs the ball to move).

Output: the hard mask, a soft P(dead) for the decoder, the signals (rule columns), and restart / stop
candidate frames for boundary snapping (combine.snap).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

from .feats import _run_for, _since


@dataclass(frozen=True)
class RuleConfig:
    out_margin_m: float = 0.5
    spot_still_s: float = 1.0
    spot_line_m: float = 1.5
    spot_corner_m: float = 2.0
    spot_centre_m: float = 1.5
    slow_median_ms: float = 2.0      # "players slow" for the shape triggers
    hold_s: float = 0.6              # shape triggers must hold this long
    corner_m: float = 2.0
    corner_box: int = 6
    gk_box_max: int = 3
    kickoff_own_half: float = 0.95
    kickoff_centre_m: float = 2.0
    still_med_ms: float = 1.0
    still_p95_ms: float = 3.0
    still_min_s: float = 2.0
    long_break_s: float = 20.0
    burst_ms: float = 4.0            # (feats.BURST_MS: the n_fast threshold of frame_quantities)
    burst_n: int = 3
    p_dead_on: float = 0.85          # soft output of the machine
    p_dead_off: float = 0.15
    p_out: float = 0.97
    p_kick: float = 0.05


def parse_rule_cfg(items) -> RuleConfig:
    base = RuleConfig()
    types = {f.name: type(getattr(base, f.name)) for f in fields(RuleConfig)}
    over = {}
    for it in items or []:
        k, eq, v = it.partition("=")
        if not eq or k not in types:
            raise SystemExit(f"--rule-cfg: expected FIELD=VALUE with FIELD in RuleConfig, got {it!r}")
        over[k] = types[k](float(v)) if types[k] is not bool else v in ("1", "true", "True")
    return RuleConfig(**{**asdict(base), **over})


def _held(flag: np.ndarray, fps: float, sec: float) -> np.ndarray:
    return _run_for(flag, fps, cap=1e9) >= sec


def apply(q: dict, fps: float, pitch, halves, n: int, cfg: RuleConfig | None = None) -> dict:
    """q = feats.frame_quantities(...). Returns {"dead", "p_dead", signal arrays, "restart_cand", "stop_cand"}."""
    cfg = cfg or RuleConfig()
    med = np.nan_to_num(q["speed_median"], nan=0.0)
    p95 = np.nan_to_num(q["speed_p95"], nan=0.0)
    slow = med < cfg.slow_median_ms
    b_out = np.nan_to_num(q["b_out"], nan=0.0) > 0.5
    b_still = np.nan_to_num(q["b_still"], nan=0.0) > 0.5
    still_run_b = _run_for(b_still, fps, cap=1e9)
    on_spot = ((np.nan_to_num(q["b_d_line"], nan=99.0) < cfg.spot_line_m)
               | (np.nan_to_num(q["b_d_corner"], nan=99.0) < cfg.spot_corner_m)
               | (np.nan_to_num(q["b_in_ga"], nan=0.0) > 0.5)
               | (np.nan_to_num(q["b_d_centre"], nan=99.0) < cfg.spot_centre_m))
    spot = b_still & (((still_run_b >= cfg.spot_still_s) & on_spot)
                      | ((still_run_b >= 2 * cfg.spot_still_s) & slow))
    # shape conditions; their run lengths (frames) are compared with the time since the last restart
    # inside the machine, so a shape still standing just after a restart (the thrower behind the line)
    # cannot switch it straight back
    c_throw = (q["n_beyond_touch"] >= 1) & slow
    c_corner = (np.nan_to_num(q["d_corner"], nan=99.0) < cfg.corner_m) & (q["box_max"] >= cfg.corner_box)
    gk_l = (q["ga_l"] >= 1) & (q["box_l"] <= cfg.gk_box_max)
    gk_r = (q["ga_r"] >= 1) & (q["box_r"] <= cfg.gk_box_max)
    c_gk = (gk_l | gk_r) & (med < 1.5)
    c_ko = ((np.nan_to_num(q["own_half"], nan=0.0) >= cfg.kickoff_own_half)
            & (np.nan_to_num(q["d_centre"], nan=99.0) < cfg.kickoff_centre_m) & (med < 1.5))
    still_flag = (med < cfg.still_med_ms) & (p95 < cfg.still_p95_ms) & (q["n_pres"] > 0)
    runs_f = {k: np.round(_run_for(c, 1.0, cap=1e12)).astype(np.int64)
              for k, c in (("throw", c_throw), ("corner", c_corner), ("goalkick", c_gk), ("kickoff", c_ko),
                           ("still", still_flag))}
    throw, corner = _held(c_throw, fps, cfg.hold_s), _held(c_corner, fps, cfg.hold_s)
    goalkick, kickoff = _held(c_gk, fps, cfg.hold_s), _held(c_ko, fps, cfg.hold_s)
    still_run = _run_for(still_flag, fps, cap=1e9)
    still = still_run >= cfg.still_min_s
    long_break = still_run >= cfg.long_break_s
    kick = np.asarray(q["b_kick"], dtype=bool)
    nf = q["n_fast"]
    burst = (nf >= cfg.burst_n) & (np.r_[0.0, nf[:-1]] < cfg.burst_n)
    block = b_out | spot                                            # the ball must move before a restart
    off = (kick | burst) & ~block
    dead = np.zeros(n, dtype=bool)
    h = max(1, int(round(cfg.hold_s * fps)))
    s_min = max(1, int(round(cfg.still_min_s * fps)))
    for _, a, b in halves:
        sl = slice(a, b + 1)
        ball_on = (b_out[sl] | spot[sl]).tolist()
        off_l = off[sl].tolist()
        rt, rc, rg, rk, rs = (runs_f[k][sl].tolist() for k in ("throw", "corner", "goalkick", "kickoff", "still"))
        out = np.zeros(b - a + 1, dtype=bool)
        state = True                                                # every half starts with a kickoff
        since_off = 10 ** 9
        for t in range(b - a + 1):
            since_off += 1
            if state:
                if off_l[t]:
                    state = False
                    since_off = 0
            else:
                by_still = min(rs[t], since_off) >= s_min
                if (ball_on[t] or by_still or min(rt[t], since_off) >= h or min(rc[t], since_off) >= h
                        or min(rg[t], since_off) >= h or min(rk[t], since_off) >= h):
                    state = True
                    if by_still and not ball_on[t]:                 # date it back to the start of the still run
                        out[max(0, t - min(rs[t], since_off)):t] = True
            out[t] = state
        dead[sl] = out
    p = np.where(dead, cfg.p_dead_on, cfg.p_dead_off)
    p = np.where(b_out | long_break, cfg.p_out, p)
    p = np.where(kick, cfg.p_kick, p)
    inside = np.zeros(n, dtype=bool)
    for _, a, b in halves:
        inside[a:b + 1] = True
    still_onset = still_flag & ~np.r_[False, still_flag[:-1]]
    out_onset = b_out & ~np.r_[False, b_out[:-1]]
    return {"dead": dead & inside, "p_dead": np.where(inside, p, np.nan),
            "out": b_out, "spot": spot, "throw": throw, "corner": corner, "goalkick": goalkick, "kickoff": kickoff,
            "still": still, "long": long_break, "kick": kick, "burst": burst,
            "restart_cand": (kick | burst) & inside, "stop_cand": (out_onset | still_onset) & inside, "cfg": cfg,
            "fps": float(fps)}


def columns(rr: dict, halves, n: int) -> pd.DataFrame:
    """The r_ columns (floats; clocks per half)."""
    from .feats import CAP_S
    c = {f"r_{k}": rr[k].astype(np.float64) for k in ("dead", "out", "spot", "throw", "corner", "goalkick",
                                                       "kickoff", "still", "long", "kick", "burst")}
    c["r_p_dead"] = np.nan_to_num(rr["p_dead"], nan=0.5)
    dead = rr["dead"]
    since_on = np.full(n, CAP_S)
    since_off = np.full(n, CAP_S)
    fps = rr["fps"]
    for _, a, b in halves:
        d = dead[a:b + 1]
        on_edge = d & ~np.r_[False, d[:-1]]
        off_edge = ~d & np.r_[False, d[:-1]]
        since_on[a:b + 1] = _since(on_edge, fps)
        since_off[a:b + 1] = _since(off_edge, fps)
    c["r_since_dead_onset_s"] = since_on
    c["r_since_restart_s"] = since_off
    return pd.DataFrame(c)


def summary(rr: dict, fps: float) -> str:
    d = rr["dead"]
    runs = int((d & ~np.r_[False, d[:-1]]).sum())
    return f"dead {d.mean():.1%} in {runs} stoppages"
