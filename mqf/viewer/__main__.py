"""python -m mqf.viewer -- open a match with its ground truth and our predictions underneath.

    python -m mqf.viewer                               # BAR-ATH: GT, best so far without the decoder / with it /
                                                       #   with runs >= 0.5 s only, v8 alone with the decoder
    python -m mqf.viewer --all-lanes                   # + the old TCN-only lane and every results/*.csv
    python -m mqf.viewer --pred results/Perosinho_FC.csv
    python -m mqf.viewer --gt none                     # model lanes only
    python -m mqf.viewer --synthetic                   # synthetic match (try: python -m mqf.viewer.demo)
    python -m mqf.viewer --no-broadcast                # skip the broadcast feed and its sync
    python -m mqf.viewer --resync                      # recompute the broadcast sync
    python -m mqf.viewer --no-fixed                    # only the tracking as delivered
    python -m mqf.viewer --fixed-device cpu            # where the ball model runs (default: GPU if any)
    python -m mqf.viewer --fixed-cfg keeper_takeover_count_check=0   # Fixed with a cleaner rule changed
    python -m mqf.viewer --best-dead dead_v1           # Best so far with the learned P(dead) (default: none)
    python -m mqf.viewer --no-best                     # skip the Best so far lane
    python -m mqf.viewer --val list                    # val games exported by final_check --fusion-export
    python -m mqf.viewer --val PFF-3822 --column atd_fuzzy --seed 0   # full GT + Models, no decoder + FINAL
    python -m mqf.viewer --fusion-best harness_results/6_decoder/fusion_f162229_deep_best.json   # + the fusion pick

Three bars (fusion_lanes.py): Ground truth, "Best so far · no decoder" (per-frame argmax of FINAL's evidence)
and "Best so far · decoder" (FINAL: the production HMM on the same arrays), from harness_results/fusion_preds
(--val, full ground truth) or, on BAR-ATH, from cache/viewer/final_arrays.npz written by
    python -m mqf.make_submission --final --dump-arrays cache/viewer/final_arrays.npz --out cache/viewer/final.csv
plus "Best so far · runs >= 0.5 s" (the same evidence, one rule only: every run >= 0.5 s), "Best so far · runs
>= 0.5 s, stoppages >= 2 s" (also: dead balls >= 2 s, a possession between two dead balls >= 2 s) and
"v8 dead ball · decoder" (P(dead) = v8 alone, the same decoder).
With those arrays the page shows only these lanes (more with --pred, --all-lanes or by dropping a CSV on it).

Best so far (lane under the ground truth, the one compared by default): our current best pipeline
run here on the Fixed tracking -- the owner TCN ensemble ens_impff + its boundary head (EXPERIMENTS
#40), and no dead-ball model yet (--best-dead none): every frame of the halves is H or A, and the
truth's dead frames are left out of the disagreements for this lane. The white line on the lane is
P(home) (top = home), the ticks on its top edge P(boundary). Cached in cache/viewer like Fixed; the
startup log prints its aggregates ("best lane: ..."), safe to paste. Needs torch and the three
models/owner_tcn_full_impff_s*.pt.

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


def lane_plan(args, fusion, best):
    """(old best-lane options or None, whether results/*.csv become lanes). With the fusion arrays: only the
    three bars, unless --all-lanes (CSVs named with --pred are always shown)."""
    if fusion is not None and not args.all_lanes:
        return None, False
    return best, True


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
    ap.add_argument("--no-best", action="store_true", help="skip the Best so far lane (TCN ensemble)")
    ap.add_argument("--best-dead", default="none", choices=["none", "dead_v1"],
                    help="dead-ball part of Best so far: none (H/A only, default) or dead_v1 (3-state HMM)")
    ap.add_argument("--rebest", action="store_true", help="recompute Best so far (ignore its cache)")
    ap.add_argument("--val", default=None, metavar="GAME|list",
                    help="a val game from --fusion-preds with its full ground truth (no video), or 'list'")
    ap.add_argument("--column", default="atd_fuzzy", help="--val: condition_ballmode, e.g. bar_exact, atd_flip_fuzzy")
    ap.add_argument("--seed", default="0", help="--val: noise seed of that column")
    ap.add_argument("--fusion-preds", type=Path, default=None, help="default harness_results/fusion_preds")
    ap.add_argument("--fusion-arrays", type=Path, default=None,
                    help="BAR-ATH: make_submission --dump-arrays output (default cache/viewer/final_arrays.npz)")
    ap.add_argument("--no-fusion", action="store_true", help="BAR-ATH: skip the no-decoder / decoder lanes")
    ap.add_argument("--all-lanes", action="store_true",
                    help="BAR-ATH: also the old TCN-only lane and every results/*.csv (default: the three bars only)")
    ap.add_argument("--fusion-best", type=Path, default=None,
                    help="also a fusion decoder lane: a fusion _best.json or _adopt.json (looking only)")
    ap.add_argument("--fusion-row", default="PIPELINE", help="row of --fusion-best (PIPELINE, hsmm_soft, ...)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)

    t0 = time.time()

    def log(msg: str):
        print(f"[{time.time() - t0:5.1f} s] {msg}", flush=True)

    fixed = None
    if not args.no_fixed:                        # BAR-ATH alone is enough: nothing from mqf.external
        from ..tracks import parse_clean_cfg
        fixed = {"model_path": args.fixed_model, "device": args.fixed_device, "force": args.refix,
                 "clean_cfg": parse_clean_cfg(args.fixed_cfg)}
    best = None if args.no_best else {"dead_mode": args.best_dead, "device": args.fixed_device, "force": args.rebest}
    from ..config import HARNESS_DIR
    from .fusion_lanes import load_best, val_game, val_list
    fbest = load_best(args.fusion_best, args.fusion_row) if args.fusion_best else None
    preds = args.fusion_preds or HARNESS_DIR / "fusion_preds"
    arrays = args.fusion_arrays or G.VIEWER_CACHE / "final_arrays.npz"
    fusion = None if args.no_fusion or not Path(arrays).is_file() else {"arrays": arrays, "best": fbest}
    if args.val == "list":
        print(f"val games in {preds}:\n{val_list(preds)}")
        return
    if args.val:
        g = val_game(preds, args.val, args.column, args.seed, fbest, log)
    elif args.synthetic:
        g = G.synthetic_game(args.pred, None if args.gt == "public" else args.gt, args.resync, log, fixed=fixed,
                             best=best)
    else:
        if fusion is None and not args.no_fusion:
            log(f"no {arrays}: for the three bars (GT, best so far without / with the decoder) run  "
                f"python -m mqf.make_submission --final --dump-arrays {Path(arrays).as_posix()} "
                f"--out cache/viewer/final.csv  -- showing the older lanes meanwhile")
        best, csv_lanes = lane_plan(args, fusion, best)
        g = G.competition(args.pred, args.gt, broadcast=not args.no_broadcast, resync=args.resync, log=log,
                          fixed=fixed, best=best, fusion=fusion, csv_lanes=csv_lanes)
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
    if not args.synthetic and not args.val:
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
