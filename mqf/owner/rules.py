"""Rule-based owner: who controls the ball, from positions only -- no model, no training.

Sigmar's rules (15 Nov 2026), in tracking terms:

  contact   the nearest player is within r_touch of the ball (one run of frames per player; runs of
            the same player split by <= gap_tol_s count as one)
  control   a contact where nobody of the other team is within r_free of the ball AND the ball moves
            with the player (|v_ball - v_player| <= v_match) for >= dwell_s in total
            -> possession belongs to that player's team ("close, nobody near, same velocity")
  pass      pass_chain consecutive contacts by DIFFERENT players (track slots) of one team, each
            within chain_gap_s of the previous one, with no contact of the other team in between
            -> that team has the ball from the first of them ("ball from a player to a team-mate")
  touch     any other contact by the team without the ball: a deflection, a block, a tackle that
            does not keep the ball, an aerial duel -- it does NOT change possession
  loose     between contacts (ball flying, rolling, missing) the last owner keeps it

Inside one segment (a half, or one alive run when an alive mask is given) the first contact's team
owns the ball from the segment start (the team that restarts). A switch is dated at the start of
the contact that wins the ball (GT: the loose ball stays with the last team until the other one
controls it).

Outputs, per frame:
  owner     0 home, 1 away, -1 no contact yet / outside the segments
  p_home    a soft version: p_ctrl during a controlled contact of the owner, then decaying from
            p_hold to p_floor with the time since the owner last touched the ball (tau_s),
            p_contest while the other team touches it without taking it
  columns   rule features for the learned models (clocks to / from each team's controls, ...)
  events    one row per contact: start, end, team, slot, controlled, kind

Thresholds are football / tracking knowledge (DATA.md: ~0.3-0.5 m tracker noise, ball imputation
~2.6 m on long gaps), never BAR-ATH. On an imputed ball (ball_filled) contacts may confirm the owner
or extend a pass chain but never make a control on their own (filled_can_control=False).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

CAP_S = 15.0

KIND_TOUCH, KIND_CONFIRM, KIND_CONTROL, KIND_CHAIN, KIND_FIRST = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class RuleConfig:
    r_touch_m: float = 1.5          # nearest player this close: in contact with the ball
    r_free_m: float = 3.0           # control: nobody of the other team this close to the ball
    free_from: str = "opponents"    # "opponents" or "all" (nobody else at all, team-mates included)
    v_match_ms: float = 2.5         # control: ball and player velocities this close
    dwell_s: float = 0.4            # control: free and velocity-matched for this long within the contact
    gap_tol_s: float = 0.2          # one contact even if the same player drops out of range this long
    pass_chain: int = 2             # contacts by different players of one team that make a pass; 0 = off
    chain_gap_s: float = 3.0        # ... each within this long of the previous one (a pass flight)
    vel_half_s: float = 0.2         # velocities: centred difference over +- this
    filled_can_control: bool = False  # an imputed ball can confirm / chain but never control alone
    tau_s: float = 4.0              # soft output: confidence decay with time since the owner's last contact
    p_ctrl: float = 0.97
    p_hold: float = 0.9
    p_floor: float = 0.65           # the last owner keeps a loose ball
    p_contest: float = 0.6          # while the other team touches it without taking it


def parse_rule_cfg(items) -> RuleConfig:
    """["r_touch_m=1.2", "pass_chain=0"] -> RuleConfig with those fields changed."""
    base = RuleConfig()
    types = {f.name: type(f.default) for f in fields(RuleConfig)}
    over = {}
    for kv in items or []:
        k, _, v = kv.partition("=")
        if k not in types:
            raise SystemExit(f"rule config: unknown field {k} (RuleConfig: {', '.join(types)})")
        t = types[k]
        over[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
    return replace(base, **over)


def _centred_diff(x: np.ndarray, fps: float, half_s: float) -> np.ndarray:
    w = max(1, int(round(half_s * fps)))
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if len(x) > 2 * w:
        out[w:-w] = (x[2 * w:] - x[:-2 * w]) * fps / (2 * w)
    return out


def _rle(a: np.ndarray):
    """(starts, ends_exclusive, values) of the runs of a 1-d array."""
    if len(a) == 0:
        return np.zeros(0, int), np.zeros(0, int), a[:0]
    change = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(a)]
    return starts, ends, a[starts]


def _bridge(slot: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill runs of -1 of length <= max_gap whose neighbours are the same slot."""
    if max_gap <= 0 or len(slot) == 0:
        return slot
    s, e, v = _rle(slot)
    out = slot.copy()
    k = np.arange(1, len(v) - 1)
    ok = (v[k] == -1) & ((e[k] - s[k]) <= max_gap) & (v[k - 1] == v[k + 1]) & (v[k - 1] >= 0)
    for i in k[ok]:
        out[s[i]:e[i]] = v[i - 1]
    return out


def segments_of(halves, n: int, alive: np.ndarray | None) -> list[tuple[int, int]]:
    """[(start, end_exclusive)]: the halves, or the alive runs inside them."""
    segs = []
    for _, a, b in halves:
        if alive is None:
            segs.append((a, b + 1))
            continue
        s, e, v = _rle(np.asarray(alive[a:b + 1], dtype=bool))
        segs += [(a + int(x), a + int(y)) for x, y, ok in zip(s, e, v) if ok]
    return segs


def frame_cues(players, team, ball, fps, cfg: RuleConfig, ball_filled=None) -> dict:
    """Per-frame: nearest slot in contact (-1 none), and whether that frame shows control."""
    xy = np.asarray(players, dtype=np.float64)
    b = np.asarray(ball, dtype=np.float64)
    team = np.asarray(team)
    n, P = xy.shape[:2]
    has = ~np.isnan(b[:, 0])
    filled = np.zeros(n, dtype=bool) if ball_filled is None else np.asarray(ball_filled, dtype=bool) & has
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(xy - b[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    rows = np.arange(n)
    j = np.argmin(d, axis=1) if P else np.zeros(n, dtype=int)
    d1 = d[rows, j] if P else np.full(n, np.inf)
    tj = team[j] if P else np.zeros(n, dtype=team.dtype)
    if cfg.free_from == "all":
        d_other = np.partition(d, 1, axis=1)[:, 1] if P > 1 else np.full(n, np.inf)
    else:
        d_other = np.where(team[None, :] != tj[:, None], d, np.inf).min(axis=1) if P else np.full(n, np.inf)
    vb = _centred_diff(b, fps, cfg.vel_half_s)
    vp = np.full((n, 2), np.nan)                                   # velocity of the nearest slot, same window
    w = max(1, int(round(cfg.vel_half_s * fps)))
    if P and n > 2 * w:
        t = np.arange(w, n - w)
        vp[t] = (xy[t + w, j[t]] - xy[t - w, j[t]]) * fps / (2 * w)
    with np.errstate(invalid="ignore"):
        vrel = np.linalg.norm(vb - vp, axis=1)
    contact = has & (d1 <= cfg.r_touch_m)
    ctrl = contact & (d_other >= cfg.r_free_m) & (vrel <= cfg.v_match_ms)
    if not cfg.filled_can_control:
        ctrl &= ~filled
    return {"slot": np.where(contact, j, -1).astype(np.int64), "ctrl": ctrl, "filled": filled,
            "d1": d1, "d_other": d_other, "vrel": vrel}


def contact_events(cues: dict, team: np.ndarray, segs, fps: float, cfg: RuleConfig) -> pd.DataFrame:
    """One row per contact inside the segments (a contact never crosses a segment edge)."""
    team = np.asarray(team)
    gap = int(round(cfg.gap_tol_s * fps))
    dwell = max(1, int(round(cfg.dwell_s * fps)))
    ctrl_cum = np.r_[0, np.cumsum(cues["ctrl"])]
    fill_cum = np.r_[0, np.cumsum(cues["filled"])]
    rows = []
    for k, (a, b) in enumerate(segs):
        slot = _bridge(cues["slot"][a:b], gap)
        s, e, v = _rle(slot)
        m = v >= 0
        s, e, v = s[m] + a, e[m] + a, v[m]
        if not len(s):
            continue
        n_ctrl = ctrl_cum[e] - ctrl_cum[s]
        rows.append(pd.DataFrame({"seg": k, "start": s, "end": e, "slot": v, "team": team[v].astype(np.int64),
                                  "n_ctrl": n_ctrl, "controlled": n_ctrl >= dwell,
                                  "filled_share": (fill_cum[e] - fill_cum[s]) / (e - s)}))
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                             (("seg", int), ("start", int), ("end", int), ("slot", int), ("team", int),
                              ("n_ctrl", int), ("controlled", bool), ("filled_share", float), ("kind", int))})
    return pd.concat(rows, ignore_index=True)


def _assign(ev: pd.DataFrame, fps: float, cfg: RuleConfig) -> tuple[np.ndarray, list]:
    """Walk the contacts of each segment: kind per contact and the switches [(seg, frame, team)]."""
    kind = np.full(len(ev), KIND_TOUCH, dtype=np.int64)
    switches = []
    if not len(ev):
        return kind, switches
    seg, st, en = ev["seg"].to_numpy(), ev["start"].to_numpy(), ev["end"].to_numpy()
    tm, sl, ctrl = ev["team"].to_numpy(), ev["slot"].to_numpy(), ev["controlled"].to_numpy()
    chain_gap = cfg.chain_gap_s * fps
    i, n = 0, len(ev)
    while i < n:
        j = i
        while j < n and seg[j] == seg[i]:
            j += 1
        owner = int(tm[i])
        kind[i] = KIND_FIRST
        switches.append((int(seg[i]), None, owner))           # None: from the segment start
        for k in range(i + 1, j):
            if tm[k] == owner:
                kind[k] = KIND_CONFIRM
                continue
            if ctrl[k]:
                kind[k] = KIND_CONTROL
            elif cfg.pass_chain >= 2:
                slots, last_end, q = {int(sl[k])}, en[k], k + 1
                while q < j and tm[q] == tm[k] and st[q] - last_end <= chain_gap and len(slots) < cfg.pass_chain:
                    slots.add(int(sl[q]))
                    last_end = en[q]
                    q += 1
                if len(slots) >= cfg.pass_chain:
                    kind[k] = KIND_CHAIN
            if kind[k] != KIND_TOUCH:
                owner = int(tm[k])
                switches.append((int(seg[k]), int(st[k]), owner))
        i = j
    return kind, switches


def _since(flag: np.ndarray, fps: float, cap: float = CAP_S) -> np.ndarray:
    t = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, t, -1))
    return np.where(last >= 0, np.minimum((t - last) / fps, cap), cap)


def owner(players, team, ball, fps, halves, cfg: RuleConfig | None = None, ball_filled=None,
          alive: np.ndarray | None = None) -> dict:
    """The rule owner of every frame (see the module docstring). alive: optional mask -- the owner
    is then reset at every alive run (who restarts is read from the first contact)."""
    cfg = cfg or RuleConfig()
    team = np.asarray(team)
    n = len(ball)
    segs = segments_of(halves, n, alive)
    cues = frame_cues(players, team, ball, fps, cfg, ball_filled)
    ev = contact_events(cues, team, segs, fps, cfg)
    kind, switches = _assign(ev, fps, cfg)
    ev["kind"] = kind

    own = np.full(n, -1, dtype=np.int64)
    in_seg = np.zeros(n, dtype=bool)
    for a, b in segs:
        in_seg[a:b] = True
    by_seg: dict[int, list] = {}
    for sg, f, t in switches:
        by_seg.setdefault(sg, []).append((f, t))
    for sg, lst in by_seg.items():
        a, b = segs[sg]
        starts = [a if f is None else f for f, _ in lst] + [b]
        for (f, t), s0, s1 in zip(lst, starts[:-1], starts[1:]):
            own[s0:s1] = t

    # frames in contact with the ball, by kind and team
    contact_team = np.full(n, -1, dtype=np.int64)
    contact_kind = np.full(n, -1, dtype=np.int64)
    if len(ev):
        lens = (ev["end"] - ev["start"]).to_numpy()
        idx = np.repeat(ev["start"].to_numpy(), lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        contact_team[idx] = np.repeat(ev["team"].to_numpy(), lens)
        contact_kind[idx] = np.repeat(ev["kind"].to_numpy(), lens)
    owner_touch = (contact_team >= 0) & (contact_team == own)
    other_touch = (contact_team >= 0) & (contact_team != own) & (contact_kind == KIND_TOUCH)
    in_ctrl = owner_touch & cues["ctrl"]

    since_conf = np.full(n, CAP_S)
    cols = {c: np.full(n, np.nan) for c in ("rule_since_owner_touch", "rule_since_switch", "rule_since_ctrl_h",
                                            "rule_since_ctrl_a", "rule_until_ctrl_h", "rule_until_ctrl_a",
                                            "rule_since_touch_h", "rule_since_touch_a")}
    ctrl_h = (contact_team == 0) & cues["ctrl"]
    ctrl_a = (contact_team == 1) & cues["ctrl"]
    touch_h, touch_a = contact_team == 0, contact_team == 1
    switch_flag = np.zeros(n, dtype=bool)
    for sg, lst in by_seg.items():
        for f, _ in lst:
            switch_flag[segs[sg][0] if f is None else f] = True
    for a, b in segs:
        sl = slice(a, b)
        sc = _since(owner_touch[sl], fps)
        first = np.flatnonzero(owner_touch[sl])
        if len(first):                                      # before the first contact: time until it
            pre = np.arange(first[0])
            sc[:first[0]] = np.minimum((first[0] - pre) / fps, CAP_S)
        since_conf[sl] = sc
        cols["rule_since_owner_touch"][sl] = sc
        cols["rule_since_switch"][sl] = _since(switch_flag[sl], fps, cap=2 * CAP_S)
        cols["rule_since_ctrl_h"][sl] = _since(ctrl_h[sl], fps)
        cols["rule_since_ctrl_a"][sl] = _since(ctrl_a[sl], fps)
        cols["rule_until_ctrl_h"][sl] = _since(ctrl_h[sl][::-1], fps)[::-1]
        cols["rule_until_ctrl_a"][sl] = _since(ctrl_a[sl][::-1], fps)[::-1]
        cols["rule_since_touch_h"][sl] = _since(touch_h[sl], fps)
        cols["rule_since_touch_a"][sl] = _since(touch_a[sl], fps)

    conf = cfg.p_floor + (cfg.p_hold - cfg.p_floor) * np.exp(-since_conf / cfg.tau_s)
    conf = np.where(in_ctrl, cfg.p_ctrl, conf)
    conf = np.where(other_touch, np.minimum(conf, cfg.p_contest), conf)
    p_home = np.where(own == 0, conf, np.where(own == 1, 1.0 - conf, 0.5))
    p_home = np.where(in_seg, p_home, np.nan)
    for c in cols.values():
        c[~in_seg] = np.nan
    cols["rule_p_home"] = p_home
    cols["rule_owner"] = np.where(in_seg, np.where(own == 0, 1.0, np.where(own == 1, -1.0, 0.0)), np.nan)
    cols["rule_other_touch"] = np.where(in_seg, other_touch.astype(np.float64), np.nan)
    cols["rule_in_ctrl"] = np.where(in_seg, in_ctrl.astype(np.float64), np.nan)
    return {"owner": own, "p_home": p_home, "columns": cols, "events": ev, "segments": segs,
            "in_ctrl": in_ctrl, "other_touch": other_touch}


def summary(res: dict, fps: float) -> dict:
    """Counts only (safe to print on any game)."""
    ev = res["events"]
    k = ev["kind"].to_numpy() if len(ev) else np.zeros(0, int)
    sw = int(((k == KIND_CONTROL) | (k == KIND_CHAIN)).sum())
    return {"contacts": int(len(ev)), "touches_ignored": int((k == KIND_TOUCH).sum()),
            "switch_by_control": int((k == KIND_CONTROL).sum()), "switch_by_pass": int((k == KIND_CHAIN).sum()),
            "segments": len(res["segments"]), "switches": sw,
            "contact_share": float(np.mean(res["columns"]["rule_since_owner_touch"][np.isfinite(
                res["columns"]["rule_since_owner_touch"])] == 0)) if len(ev) else 0.0}
