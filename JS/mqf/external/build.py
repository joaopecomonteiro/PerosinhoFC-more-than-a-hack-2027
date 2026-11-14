"""Write other games as bundles in the tactical data definition -- no noise, native frame rate.

    python -m mqf.external.build                              # every source, every game found
    python -m mqf.external.build --source metrica skillcorner
    python -m mqf.external.build --source idsse --matches J03WMX --force

Sources (raw files in external_data/, see each module's docstring):
  idsse        7 DFL games, 25 fps, per-frame BallStatus + BallPossession (needs the cache from
               `python -m mqf.harness.idsse`, which needs floodlight -- or run this in .venv-harness)
  metrica      3 sample games, 25 fps, labels from the synced manual events
  skillcorner  20 A-League games, 10 fps broadcast tracking, labels from phases of play + possession
  pff          64 World Cup 2022 games, 29.97 fps broadcast tracking, labels from PFF's game events
               (--jobs N converts N games in parallel; ~1 min and ~2 GB of RAM per game and job)

Prints one QA line per game (other games: safe to share). Bundles go to external/bundles/<GAME>/.
"""
from __future__ import annotations

import argparse
import json
import time

from . import bundle

SOURCES = ("idsse", "metrica", "skillcorner", "pff")


def games(source: str, ids=None):
    if source == "idsse":
        from ..harness import idsse
        for mid in ids or idsse.MATCHES:
            m = idsse.load_match(mid)
            home, away = idsse.team_names(mid)
            m.update(name=f"IDSSE-{mid}", source="idsse", home=home, away=away,
                     labels_from="DFL official tracking: per-frame BallStatus (dead) + BallPossession (team)",
                     licence="IDSSE, Bassek et al. 2025 (Sci. Data 12:195), CC-BY 4.0")
            yield m
    elif source == "metrica":
        from . import metrica
        for g in ids or metrica.GAMES:
            yield metrica.load_match(int(g))
    elif source == "skillcorner":
        from . import skillcorner
        for mid in ids or skillcorner.match_ids():
            yield skillcorner.load_match(mid)
    elif source == "pff":
        from . import pff
        for mid in ids or pff.match_ids():
            yield pff.load_match(mid)
    else:
        raise SystemExit(f"unknown source {source}")


def line(name: str, fps: int, q: dict) -> str:
    s = q["share"]
    return (f"{name:<14} {fps:>2} fps  {q['graded_minutes']:5.1f} min  H {s['H']:.2f} A {s['A']:.2f} D {s['D']:.2f}"
            f"  poss {q['possessions']:>3} ({q['possessions_per_90min']:.0f}/90')  mean {q['mean_s']:4.1f} s"
            f"  <3s {q['share_under_3s']:.0%}  ball seen alive {q['ball_seen']['alive']:.2f}"
            f" dead {q['ball_seen']['dead'] if q['ball_seen']['dead'] is not None else float('nan'):.2f}"
            f"  players {q['players_tracked_mean']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", nargs="*", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--matches", nargs="*", help="ids within the source (J03WMX, 1, 1874553 ...)")
    ap.add_argument("--force", action="store_true", help="rewrite bundles that already exist")
    ap.add_argument("--jobs", type=int, default=1, help="pff: games converted in parallel")
    args = ap.parse_args(argv)
    for source in args.source:
        if source == "pff":
            _build_pff(args)
            continue
        for m in games(source, args.matches):
            d = bundle.BUNDLES_DIR / m["name"]
            info = d / f"{m['name']}_game.json"
            if info.exists() and not args.force:
                q = json.loads(info.read_text(encoding="utf-8"))
                if q.get("version") == bundle.VERSION:
                    print(line(m["name"], q["fps"], q["qa"]) + "  (exists)")
                    continue
            t0 = time.time()
            bundle.write_bundle(m)
            print(line(m["name"], m["fps"], bundle.qa(m)) + f"  ({time.time() - t0:.0f} s)", flush=True)
    print(f"bundles in {bundle.BUNDLES_DIR}")
    return 0


def _exists(name: str) -> dict | None:
    info = bundle.BUNDLES_DIR / name / f"{name}_game.json"
    if info.exists():
        q = json.loads(info.read_text(encoding="utf-8"))
        if q.get("version") == bundle.VERSION:
            return q
    return None


def _build_pff(args) -> None:
    """PFF games one per process: the raw files are large and each conversion is independent."""
    from concurrent.futures import ProcessPoolExecutor
    from . import pff
    ids = args.matches or pff.match_ids()
    if not ids:
        print(f"pff: no <id>.jsonl.bz2 in {pff.raw_dir()}")
        return
    todo = []
    for mid in ids:
        q = None if args.force else _exists(f"PFF-{mid}")
        if q:
            print(line(f"PFF-{mid}", q["fps"], q["qa"]) + "  (exists)")
        else:
            todo.append(mid)
    if args.jobs <= 1:
        for name, fps, q, secs in map(pff.build_one, todo):
            print(line(name, fps, q) + f"  ({secs:.0f} s)", flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for name, fps, q, secs in pool.map(pff.build_one, todo):
            print(line(name, fps, q) + f"  ({secs:.0f} s)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
