"""python -m mqf.viewer -- open a match with its ground truth and our predictions underneath.

    python -m mqf.viewer                               # BAR-ATH, public GT + every results/*.csv
    python -m mqf.viewer --pred results/Perosinho_FC.csv
    python -m mqf.viewer --gt none                     # model lanes only
    python -m mqf.viewer --synthetic                   # synthetic match (try: python -m mqf.viewer.demo)
    python -m mqf.viewer --no-broadcast                # skip the broadcast feed and its sync
    python -m mqf.viewer --resync                      # recompute the broadcast sync

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
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)

    t0 = time.time()

    def log(msg: str):
        print(f"[{time.time() - t0:5.1f} s] {msg}", flush=True)

    if args.synthetic:
        g = G.synthetic_game(args.pred, None if args.gt == "public" else args.gt, args.resync, log)
    else:
        g = G.competition(args.pred, args.gt, broadcast=not args.no_broadcast, resync=args.resync, log=log)
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
