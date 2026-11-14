"""What each tracklet-cleaning rule does on a recording -- aggregates only, safe to share.

    python -m mqf.trackdiag                       # the competition match (tactical), Smart Tagging kickoffs
    python -m mqf.trackdiag --bundle METRICA-3    # another game (external/bundles), as is
    python -m mqf.trackdiag --explain 1:02:56     # what the cleaner did around 1st half 02:56 (viewer clock)

Runs mqf.tracks.clean once per variant (v3 = no repairs, v4 ... v4f, v4g = the default, and v4g with
one rule switched off or on) and prints, per variant: kickoff flips, tracks whose team changed, keeper
takeovers (and why the others were refused), held copies hidden, how often each team is capped at 11
and why, and how often the other team is short then.
Also profiles the frozen points of the delivered tracking: a tracker that HOLDS a lost player
repeats the same position exactly, a real player standing still never does.
Never prints a frame number, a position or a label -- only counts, shares and durations.

--explain H:MM:SS looks at the moment the viewer shows as "1st half MM:SS" (H = half, clock from the
start of that half's graded block), +/- --span seconds: for each team's keeper (its track deepest of
all in its own penalty area, as the keeper fill defines him) how long a track of the OTHER colour is
within 1.5 m / 3 m of him -- delivered and after each cleaner variant -- and, for every such track
in the default Fixed, which rule looked at it and what stopped it. Categories and binned distances
only, safe to paste.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import atd, tracks

OLD = dict(side_from_half_start=False, keeper_box_ignore_still=False, keeper_box_count_check=True)   # before v4g
VARIANTS = {
    "v3 (no repairs)": dict(inherit_team=False, kickoff_check=False, fast_windows_s=(), keeper_box_pickup=False,
                            keeper_fill=False, keeper_takeover=False, **OLD),
    "v4": dict(ghost_reshow=False, inherit_memory_s=2.0, keeper_box_pickup=False, keeper_fill=False, keeper_takeover=False,
               _no_windows=True, **OLD),
    "v4b": dict(keeper_box_pickup=False, keeper_fill=False, keeper_takeover=False, **OLD),
    "v4c": dict(keeper_takeover=False, keeper_behaviour=False, **OLD),
    "v4d": dict(keeper_live_test="held_or_duplicate", keeper_behaviour=False, **OLD),
    "v4e": dict(keeper_behaviour=False, **OLD),
    "v4f": dict(OLD),
    "v4g (default)": {},
    "v4g + box count check": dict(keeper_box_count_check=True),
    "v4g - half-start sides": dict(side_from_half_start=False),
    "v4g - still tracks ignored": dict(keeper_box_ignore_still=False),
    "v4g - takeover count check": dict(keeper_takeover_count_check=False),
    "v4g - keeper box rule": dict(keeper_box_pickup=False),
    "v4g - keeper fill": dict(keeper_fill=False),
    "v4g - kickoff": dict(kickoff_check=False),
    "v4g - saved place": dict(inherit_memory_s=2.0),
    "v4g - held copy shown again": dict(ghost_reshow=False),
    "v4g + cap hides held copies first": dict(cap_held_first=True),
}
DEFAULT = "v4g (default)"


def held_profile(trk: atd.Tracking, fps: float, pitch, cfg: tracks.CleanConfig = tracks.CleanConfig()) -> dict:
    """Frozen runs (>= frozen_s) in the delivered tracking: how many points, how many repeat the
    previous position EXACTLY (< 1 mm: a held copy), and how many sit in a penalty area."""
    xy = trk.players.astype(np.float64)
    L, W = pitch
    step = np.full(xy.shape[:2], np.nan)
    step[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    still_f = int(round(cfg.frozen_s * fps))
    exact = np.nan_to_num(step, nan=np.inf) < 1e-3
    slow = np.nan_to_num(step, nan=np.inf) < cfg.frozen_ms / fps
    x, y = xy[..., 0], xy[..., 1]
    with np.errstate(invalid="ignore"):
        box = (np.minimum(x, L - x) < cfg.box_depth_m) & (np.abs(y - W / 2) < cfg.box_half_width_m)

    def runs(mask):
        pts = in_box = n = 0
        lens = []
        for k in range(mask.shape[1]):
            for a, b in zip(*tracks._runs(mask[:, k])):
                if b - a >= still_f:
                    n += 1
                    pts += int(b - a)
                    in_box += int(box[a:b, k].sum())
                    lens.append((b - a) / fps)
        return {"runs": n, "points": pts, "share_in_a_box": round(in_box / pts, 3) if pts else None,
                "median_run_s": round(float(np.median(lens)), 1) if lens else None,
                "p90_run_s": round(float(np.percentile(lens, 90)), 1) if lens else None}

    present = ~np.isnan(x)
    return {"points_tracked": int(present.sum()),
            "frozen_runs": runs(slow), "exact_repeat_runs": runs(exact)}


def summary(rep: dict) -> dict:
    ti = rep.get("team_inherit") or {}
    return {
        "kickoffs_found": rep["kickoffs_found"], "kickoff_flips": rep["kickoff_team_flips"],
        "kickoff_flips_detail": rep.get("kickoff_flips_detail", []),
        "tracks_team_changed": rep["tracklets_team_changed"],
        "points_team_changed_pct": round(100 * rep["points_team_changed_share"], 2),
        "saved_place_links": ti.get("from_saved_place"),
        "held_copy_points_hidden": rep["frozen_copies_hidden_points"],
        "cap_frames": rep["cap_frames"], "cap_reasons": rep["cap_reasons"],
        "capped_other_team_le_10": rep["cap_frames_other_team_le_10"],
        "players_per_frame_after": rep["players_per_frame_mean"]["after"],
        "keeper_box_pickups_changed": ti.get("keeper_box_pickups_changed"),
        "keeper_takeovers": {k[len("keeper_takeover_"):]: v for k, v in ti.items() if k.startswith("keeper_takeover_")},
        "keeper_box_still_hidden": ti.get("keeper_box_still_hidden"),
        "home_defends_left_by_half": rep.get("home_defends_left_by_half"),
        "keeper_fill_points": rep.get("keeper_fill_points"),
    }


# ------------------------------------------------------------------------------------- --explain
EXPLAIN_VARIANTS = {
    "v4c (no takeover)": dict(keeper_takeover=False, keeper_behaviour=False, **OLD),
    "v4f": dict(OLD),
    "v4g (default)": {},
    "v4g + box count check": dict(keeper_box_count_check=True),
    "v4g - half-start sides": dict(side_from_half_start=False),
    "v4g - still tracks ignored": dict(keeper_box_ignore_still=False),
}


def parse_at(at: str, halves, fps: float) -> int:
    """"1:02:56" (half 1, 2 min 56 s on the viewer's half clock) -> frame."""
    parts = at.replace(" ", ":").split(":")
    if len(parts) != 3:
        raise SystemExit(f"--explain wants H:MM:SS (half, minutes, seconds on the viewer clock), got {at!r}")
    h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
    first = {int(a): int(b) for a, b, _ in halves}
    if h not in first:
        raise SystemExit(f"--explain: no half {h} (halves: {sorted(first)})")
    return first[h] + int(round((60 * m + sec) * fps))


def keeper_like(xy, team, home_left, pitch, cfg: tracks.CleanConfig = tracks.CleanConfig(),
                chunk: int = 20_000) -> np.ndarray:
    """(n, 2) lane of each team's keeper per frame (-1: none): its track deepest of all players, inside its
    own penalty area -- the keeper fill's definition. In chunks of frames (the match is 199k x ~100 lanes)."""
    L, W = pitch
    n = len(xy)
    team = np.asarray(team)
    out = np.full((n, 2), -1, dtype=np.int64)
    for a in range(0, n, chunk):
        X = np.asarray(xy[a:a + chunk], dtype=np.float32)
        x, y = X[..., 0], X[..., 1]
        vis = ~np.isnan(x)
        rows = np.arange(len(X))
        with np.errstate(invalid="ignore"):
            for tm in (0, 1):
                left = (home_left if tm == 0 else ~home_left)[a:a + chunk]
                depth = np.where(left[:, None], x, L - x)
                depth_all = np.where(vis, depth, np.inf)
                inbox = vis & (team[None, :] == tm) & (depth < cfg.box_depth_m) & \
                    (np.abs(y - W / 2) < cfg.box_half_width_m)
                d = np.where(inbox, depth, np.inf)
                j = d.argmin(axis=1)
                ok = np.isfinite(d[rows, j]) & (d[rows, j] <= depth_all.min(axis=1))
                out[a + np.flatnonzero(ok), tm] = j[ok]
    return out


def nearest_opponent(xy, team, kl, chunk: int = 20_000):
    """(n, 2) distance from each team's keeper to the nearest track of the other team, and (n, 2) that lane."""
    n = len(xy)
    dist = np.full((n, 2), np.inf)
    lane = np.full((n, 2), -1, dtype=np.int64)
    team = np.asarray(team)
    for a in range(0, n, chunk):
        X = np.asarray(xy[a:a + chunk], dtype=np.float32)
        K = kl[a:a + chunk]
        rows = np.arange(len(X))
        for tm in (0, 1):
            ok = K[:, tm] >= 0
            kp = np.full((len(X), 2), np.nan, dtype=np.float32)
            kp[ok] = X[np.flatnonzero(ok), K[ok, tm]]
            with np.errstate(invalid="ignore"):
                d = np.linalg.norm(X - kp[:, None], axis=-1)
            d = np.where((team[None, :] == 1 - tm) & ~np.isnan(d), d, np.inf)
            j = d.argmin(axis=1)
            lane[a:a + len(X), tm] = j
            dist[a:a + len(X), tm] = d[rows, j]
    return dist, lane


def _behaviour(trk, pitch, tr, q: int, tm: int, cfg: tracks.CleanConfig = tracks.CleanConfig()):
    """(share of tracklet q's life inside team tm's penalty area, share inside it AND deepest of all players)."""
    L, W = pitch
    a, b, k = int(tr["start"][q]), int(tr["end"][q]), int(tr["slot"][q])
    X = trk.players[a:b + 1].astype(np.float64)
    left = tr["home_left"][a:b + 1] if tm == 0 else ~tr["home_left"][a:b + 1]
    with np.errstate(invalid="ignore"):
        depth = np.where(left[:, None], X[..., 0], L - X[..., 0])
        seen = ~np.isnan(X[:, k, 0])
        inbox = seen & (depth[:, k] < cfg.box_depth_m) & (np.abs(X[:, k, 1] - W / 2) < cfg.box_half_width_m)
        others = np.delete(depth, k, axis=1)
        others = np.where(np.isnan(others), np.inf, others)
        deepest = depth[:, k] <= others.min(axis=1) + cfg.keeper_behaviour_tol_m
    if not seen.any():
        return 0.0, 0.0
    return float(inbox[seen].mean()), float((inbox & deepest)[seen].mean())


def _secs(mask, fps):
    return f"{mask.sum() / fps:6.1f} s"


def explain(trk, spec, windows, at: str, span_s: float, names) -> dict:
    fps = spec.fps
    t = parse_at(at, spec.halves, fps)
    w0, w1 = max(0, t - int(span_s * fps)), min(spec.n_frames - 1, t + int(span_s * fps))
    win = np.zeros(spec.n_frames, dtype=bool)
    win[w0:w1 + 1] = True
    graded = np.zeros(spec.n_frames, dtype=bool)
    for _, a, b in spec.halves:
        graded[a:b + 1] = True
    base = tracks.CleanConfig()
    xy0 = trk.players.astype(np.float64)
    team0 = np.asarray(trk.team_of_slot)
    team_ft0 = np.where(~np.isnan(xy0[..., 0]), team0[None, :], -1)
    home_left = tracks._home_defends_left(xy0, team_ft0, fps, base)
    home_left, _ = tracks._sides_from_half_start(xy0, team_ft0, spec.halves, spec.pitch, fps, base, home_left)
    rows, default = [], None
    d0, _ = nearest_opponent(xy0, team0, keeper_like(xy0, team0, home_left, spec.pitch))
    rows.append(("Original (as delivered)", d0))
    for name, over in EXPLAIN_VARIANTS.items():
        tr = {} if name == "v4g (default)" else None
        out, rep, fl = tracks.clean_with_flags(trk, fps, spec.pitch, replace(base, **over), kickoff_windows=windows,
                                               trace=tr, halves=spec.halves)
        xy = out.players.astype(np.float64)
        kl = keeper_like(xy, out.team_of_slot, home_left, spec.pitch)
        d, lane = nearest_opponent(xy, out.team_of_slot, kl)
        rows.append((name, d))
        if tr is not None:
            default = (out, fl, tr, kl, d, lane)
        print(f"  done: {name}", flush=True)

    home, away = names
    res = {"at": at, "span_s": span_s, "window_graded": bool(graded[w0:w1 + 1].any())}
    print(f"\n=== mqf.trackdiag --explain {at} +/- {span_s:g} s -- categories only, safe to share ===")
    print(f"window inside the graded halves: {res['window_graded']}")
    out, fl, tr, kl, d, lane = default
    for tm, nm in ((0, home), (1, away)):
        print(f"{nm} keeper seen as keeper-like (deepest of all, in his box) on {np.mean(kl[win, tm] >= 0):.0%} "
              f"of the window (Fixed)")
    print(f"\nopposite-colour track next to a keeper -- time in the window [whole graded match]")
    print(f"{'':36s}{'next to ' + home[:22] + ' keeper':>34s}  {'next to ' + away[:22] + ' keeper':>34s}")
    print(f"{'':36s}" + f"{'< 1.5 m':>17s}{'< 3 m':>17s}  " * 2)
    res["table"] = {}
    for name, dd in rows:
        cells = []
        for tm in (0, 1):
            for r in (1.5, 3.0):
                m = dd[:, tm] < r
                cells.append(f"{(m & win).sum() / fps:5.1f} [{(m & graded).sum() / fps:6.0f}]")
        res["table"][name] = cells
        print(f"{name:36s}" + "".join(f"{c:>17s}" + ("  " if i == 1 else "") for i, c in enumerate(cells)))

    # every opposite-colour track within 3 m of a keeper in the window, in the default Fixed
    tid = tr["out_tid"]
    print(f"\nin the default Fixed, each track of the other colour within 3 m of a keeper in the window:")
    found = {}
    for f in np.flatnonzero(win):
        for tm in (0, 1):
            if d[f, tm] < 3.0:
                q = int(tid[f, lane[f, tm]])
                key = (tm, q)
                e = found.setdefault(key, {"frames": 0, "min": np.inf, "flags": set()})
                e["frames"] += 1
                e["min"] = min(e["min"], d[f, tm])
                e["flags"].add(int(fl[f, lane[f, tm]]))
    flag_name = {0: "as delivered", 1: "team repaired", 2: "keeper takeover", 3: "keeper fill"}
    res["intruders"] = []
    if not found:
        print("  none")
    for i, ((tm, q), e) in enumerate(sorted(found.items(), key=lambda kv: -kv[1]["frames"]), 1):
        keeper_of = home if tm == 0 else away
        shown = away if tm == 0 else home
        if q == -2:
            desc = "the keeper fill lane of the other team"
            trail = []
        elif q < 0:
            desc = "unknown lane"
            trail = []
        else:
            started_in = bool(win[int(tr["start"][q])])
            delivered = home if int(tr["team_in"][q]) == 0 else away
            beh = _behaviour(trk, spec.pitch, tr, q, tm)
            desc = (f"{'new track (starts in the window)' if started_in else 'older track (started before the window)'}, "
                    f"delivered as {delivered}, lives {(tr['end'][q] - tr['start'][q] + 1) / fps:.1f} s, in the "
                    f"{keeper_of} box {beh[0]:.0%} of its life and deepest of all players there {beh[1]:.0%}")
            trail = tr["why"].get(q, ["(no decision recorded: it started at the first frame or was a kickoff flip)"])
        item = {"keeper_of": keeper_of, "shown_as": shown, "seconds": e["frames"] / fps,
                "closest": tracks._dist_bin(e["min"]), "what": desc, "flags": sorted(flag_name[x] for x in e["flags"]),
                "decisions": trail}
        res["intruders"].append(item)
        print(f"  #{i}: a {shown} track next to the {keeper_of} keeper for {item['seconds']:.1f} s, closest "
              f"{item['closest']}; {desc}; Fixed flag: {', '.join(item['flags'])}")
        for m in trail:
            print(f"        - {m}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", help="another game in external/bundles instead of the competition match")
    ap.add_argument("--no-tags", action="store_true", help="do not pass Smart Tagging kickoff windows")
    ap.add_argument("--json", type=Path, help="also write the table as JSON here")
    ap.add_argument("--explain", metavar="H:MM:SS", help="what the cleaner did around this moment (viewer half clock)")
    ap.add_argument("--span", type=float, default=15.0, help="--explain: seconds either side (default 15)")
    args = ap.parse_args(argv)
    windows = None
    if args.bundle:
        from .external import bundle
        d = bundle.BUNDLES_DIR / args.bundle
        spec = bundle.spec(d)
        trk = atd.load(spec, cache_dir=d / "cache")
    else:
        from . import smart_tagging
        from .config import competition_game
        spec = competition_game()
        trk = atd.load(spec)
        if not args.no_tags and spec.smart_tagging and Path(spec.smart_tagging).exists():
            windows = smart_tagging.kickoff_windows(spec.smart_tagging, spec.fps, spec.n_frames)
    if args.explain:
        from .viewer.game import team_names
        res = explain(trk, spec, windows, args.explain, args.span, team_names(spec.atd_meta))
        if args.json:
            args.json.write_text(json.dumps(res, indent=2, default=str))
            print(f"wrote {args.json}")
        return 0
    out = {"recording": spec.name, "kickoff_windows_given": len(windows or ()),
           "delivered_tracking": held_profile(trk, spec.fps, spec.pitch), "variants": {}}
    base = tracks.CleanConfig()
    for name, over in VARIANTS.items():
        over = dict(over)
        no_win = over.pop("_no_windows", False)
        _, rep = tracks.clean(trk, spec.fps, spec.pitch, replace(base, **over),
                              kickoff_windows=None if no_win else windows, halves=spec.halves)
        out["variants"][name] = summary(rep)
        print(f"  done: {name}", flush=True)

    print(f"\n=== mqf.trackdiag -- {spec.name} -- aggregates only, safe to share ===")
    print(f"kickoff windows given: {out['kickoff_windows_given']}")
    print("delivered tracking, frozen >= 2 s:", json.dumps(out["delivered_tracking"]))
    cols = ("kickoffs", "flips", "teamchg", "pts%", "saved", "kbox", "take", "t:refuse", "fillH", "fillA", "heldhid",
            "cap H", "cap A", "youngH", "youngA", "heldH", "heldA", "oth<=10 H", "oth<=10 A", "ppf")
    print(f"\n{'variant':36s}" + "".join(f"{c:>10s}" for c in cols))
    for name, v in out["variants"].items():
        r = v["cap_reasons"]
        kf = v["keeper_fill_points"] or {}
        tk = v["keeper_takeovers"] or {}
        took = sum(tk.get(k, 0) for k in ("held", "duplicate", "near", "saved")) if tk else None
        refused = (tk.get("apart", 0) + tk.get("counts_disagree", 0)) if tk else None
        vals = (v["kickoffs_found"], v["kickoff_flips"], v["tracks_team_changed"], v["points_team_changed_pct"],
                v["saved_place_links"], v["keeper_box_pickups_changed"], took, refused, kf.get("home"), kf.get("away"),
                v["held_copy_points_hidden"], v["cap_frames"]["home"], v["cap_frames"]["away"],
                r["home"]["youngest"], r["away"]["youngest"], r["home"]["held"], r["away"]["held"],
                v["capped_other_team_le_10"]["home"], v["capped_other_team_le_10"]["away"], v["players_per_frame_after"])
        print(f"{name:36s}" + "".join(f"{'-' if x is None else x!s:>10s}" for x in vals))
    print("\nkickoff flips (default): " + json.dumps(out["variants"][DEFAULT]["kickoff_flips_detail"]))
    print("keeper takeovers (default): " + json.dumps(out["variants"][DEFAULT]["keeper_takeovers"]))
    print("box rule (default): still tracks hidden next to a new keeper "
          f"{out['variants'][DEFAULT]['keeper_box_still_hidden']}; home defends x = 0 by half (counted at the start): "
          + json.dumps(out["variants"][DEFAULT]["home_defends_left_by_half"]))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
