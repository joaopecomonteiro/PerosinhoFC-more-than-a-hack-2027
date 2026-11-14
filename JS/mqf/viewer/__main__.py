"""python -m mqf.viewer -- open a match with its ground truth and our predictions underneath.

    python -m mqf.viewer                               # BAR-ATH, public GT + every results/*.csv
    python -m mqf.viewer --pred results/Perosinho_FC.csv
    python -m mqf.viewer --gt none                     # model lanes only
    python -m mqf.viewer --synthetic                   # synthetic match (try: python -m mqf.viewer.demo)
    python -m mqf.viewer --no-broadcast                # skip the broadcast feed and its sync
    python -m mqf.viewer --resync                      # recompute the broadcast sync
    python -m mqf.viewer --no-fixed                    # only the tracking as delivered
    python -m mqf.viewer --fixed-device cpu            # where the ball model runs (default: GPU if any)
    python -m mqf.viewer --fixed-cfg keeper_takeover_count_check=0   # Fixed with a cleaner rule changed

Tracking switch (top bar, key T): Original = as delivered; Fixed = our cleaned tracklets
(mqf.tracks) + the missing ball filled with the frozen setting (mqf.ball.impute.FROZEN). The
fixed set is computed once on this machine and cached in cache/viewer (recomputed whenever the
cleaner's code changes). In Fixed, a white ring marks a player whose team the cleaner repaired, an
amber ring a keeper takeover (a new track next to the other team's keeper given the keeper's team),
a dashed circle the keeper drawn at his saved place; the startup log prints the counts
("fixed tracking: vs Original: ...") -- aggregates, safe to paste.

The page runs in your browser from a server bound to 127.0.0.1. More model lanes can be dropped
onto the page as frame,label CSVs.
"""
from __future__ import annotations

import argparse
import time
import webbrowser
from pathlib import Path

from . import game as G
from . import server


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--pred", action="append", type=Path, help="a frame,label CSV (repeatable); default: results/*.csv")
    ap.add_argument("--gt", default="public", help="'public' (official 5-min GT), 'none', or a CSV path")
    ap.add_argument("--synthetic", action="store_true", help="the synthetic match instead of BAR-ATH")
    ap.add_argument("--no-broadcast", action="store_true", help="skip the broadcast feed and its sync")
    ap.add_argument("--resync", action="store_true", help="recompute the broadcast sync")
    ap.add_argument("--no-fixed", action="store_true", help="skip the cleaned / ball-filled tracking")
    ap.add_argument("--fixed-model", type=Path, default=None, help="ball model (default: models/<FROZEN model>)")
    ap.add_argument("--fixed-device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--refix", action="store_true", help="recompute the fixed tracking (ignore its cache)")
    ap.add_argument("--fixed-cfg", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="mqf.tracks.CleanConfig overrides for Fixed, e.g. keeper_takeover=0 (cached separately)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)

    t0 = time.time()

    def log(msg: str):
        print(f"[{time.time() - t0:5.1f} s] {msg}", flush=True)

    from ..external.evaluate import parse_clean_cfg
    fixed = None if args.no_fixed else {"model_path": args.fixed_model, "device": args.fixed_device,
                                        "force": args.refix, "clean_cfg": parse_clean_cfg(args.fixed_cfg)}
    if args.synthetic:
        g = G.synthetic_game(args.pred, None if args.gt == "public" else args.gt, args.resync, log, fixed=fixed)
    else:
        g = G.competition(args.pred, args.gt, broadcast=not args.no_broadcast, resync=args.resync, log=log,
                          fixed=fixed)
    meta, blobs = G.bundle(g)

    log(f"{g.title}: {g.n_frames:,} frames, lanes: " + (", ".join(l.name for l in g.lanes) or "none"))
    for v in g.videos:
        status = "found" if v.available else "missing"
        extra = ""
        if v.summary:
            s = v.summary
            extra = (f", linked {100 * s['linked_share']:.1f}% of its frames, live picture on "
                     f"{100 * s['match_live_share']:.1f}% of match frames, {s['offset_runs']} offset run(s)")
        log(f"{v.label.lower()} video {status}{extra}")
    for n in g.notes:
        log(n)

    srv = server.start(meta, blobs, {v.key: v.path for v in g.videos if v.available}, args.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"\nViewer: {url}   (Ctrl+C to stop; local only)")
    if not args.synthetic:
        print("Real match on screen: keep screenshots and clips of it out of AI chats.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
