"""PathCRF (Kim et al., KDD 2026) as an outside reference for the owner experiments -- IDSSE only.

PathCRF infers who has the ball from PLAYER trajectories alone: one edge per 5 Hz step (a self-loop
= a player in control, sender -> receiver = ball in flight), set-attention backbone, CRF that
forbids impossible transitions. Its repo trains on the same 7 DFL games we call IDSSE, with labels
from DFL's manual events synchronised to the tracking (ELASTIC). Code: MPL-2.0, used here as an
external tool in its own clone -- nothing of it is copied into this repo.

Fair comparison: train it on OUR 5 IDSSE training games (splits.json), keep the final-epoch weights
(no checkpoint picked on a validation game), predict our 2 IDSSE validation games, map its edges to
the team per frame (owner = sender's team; a ball coming in from outside belongs to the receiver),
align its frames to the bundle's, and hand P(home) to mqf.owner.experiment --external.

    # 0. once: clone it OUTSIDE this repo, with its own venv (its requirements.txt misses most of these)
    git clone https://github.com/hyunsungkim-ds/pathcrf ..\\..\\pathcrf
    cd ..\\..\\pathcrf ; python -m venv .venv ; .venv\\Scripts\\python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
    .venv\\Scripts\\python -m pip install torch_geometric kloppy pyarrow rdp scipy tqdm matplotlib scikit-learn "pandas<3" numpy
    # 1. raw IDSSE XMLs (external_data/idsse) -> its data/sportec/{metadata,event,tracking} (hard links)
    python -m mqf.owner.pathcrf_bridge setup --pathcrf ..\\..\\pathcrf
    # 2. its preprocessing (touch detection, per-frame possession labels)          [its python]
    python -m mqf.owner.pathcrf_bridge preprocess --pathcrf ..\\..\\pathcrf --python ..\\..\\pathcrf\\.venv\\Scripts\\python.exe
    # 3. train on our 5 IDSSE train games (paper settings, 10 epochs: its val loss bottoms out by ~5-10)
    python -m mqf.owner.pathcrf_bridge train --pathcrf ..\\..\\pathcrf --python ... --epochs 10 --trial 901
    # 4. predict the val games -> <pathcrf>/mqf_out/<id>.npz                         [its python]
    python -m mqf.owner.pathcrf_bridge predict --pathcrf ..\\..\\pathcrf --python ... --trial 901
    # 5. align to our bundles -> external/pathcrf/IDSSE-<id>.npz (prints offsets + agreement checks)
    python -m mqf.owner.pathcrf_bridge align --pathcrf ..\\..\\pathcrf
    # 6. python -m mqf.owner.suite --stage eval  (or experiment --condition clean --external pathcrf=external/pathcrf)

Steps 2-4 run in PathCRF's environment (--python; default: this interpreter); this file is then
executed as a plain script there (`--inner`), so it imports nothing from mqf at module level.
Other games only: nothing here touches the competition match.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TRAIN_IDS = ("J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WR9")   # = splits.json idsse train (checked in `train`)
VAL_IDS = ("J03WMX", "J03WQQ")
CLIP = 0.02
OUT_DIR = "mqf_out"
# the paper's PathCRF (saved/140 in its repo: set-attention social + SAB temporal + dynamic masked CRF)
PAPER_ARGS = ["--sources", "sportec", "--agent_model", "set_tf", "--seq_model", "sab", "--crf_model",
              "dynamic_sparse_crf", "--macro_type", "poss_edge", "--micro_type", "poss_edge", "--node_in_dim", "8",
              "--target_fps", "5", "--window_stride", "1", "--flip_pitch", "--seq_dim", "128", "--crf_edge_dim", "16",
              "--emit_loss", "--src_loss", "--dst_loss", "--ce_epochs", "0", "--macro_weight", "1", "--rloss_weight",
              "0", "--crf_weight", "1", "--batch_size", "32", "--start_lr", "1e-3", "--min_lr", "1e-5",
              "--weight_decay", "1e-4", "--seed", "100", "--print_batch", "200"]
SPLIT_ANCHOR = "            if args.lomocv:\n"
SPLIT_PATCH = '''            if os.environ.get("MQF_SPLIT"):              # mqf.owner.pathcrf_bridge: explicit game lists
                import json as _json
                _sp = _json.loads(os.environ["MQF_SPLIT"])
                _by = {os.path.splitext(os.path.basename(_f))[0]: _f for _f in src_files}
                train_files = [_by[_g] for _g in _sp["train"]]
                val_files = [_by[_g] for _g in _sp["val"]]
                print(f"MQF split: train={_sp['train']} val={_sp['val']}")
            elif args.lomocv:
'''


# ------------------------------------------------------------------------------------ pure helpers (tested)

def node_team(node) -> int:
    """'home_7' -> 0, 'away_23' -> 1, outside nodes / None -> -1."""
    if not isinstance(node, str):
        return -1
    return 0 if node.startswith("home_") else 1 if node.startswith("away_") else -1


def edges_to_team(src, dst) -> np.ndarray:
    """Team owning the ball per frame from PathCRF edges: the sender's (a self-loop is a control, a
    pass in flight still belongs to the passer's team -- the loose ball stays with the last team);
    a ball coming in from an outside node (throw-in, goal kick) belongs to the receiver's team."""
    s = np.array([node_team(x) for x in src], dtype=np.int8)
    d = np.array([node_team(x) for x in dst], dtype=np.int8)
    return np.where(s >= 0, s, d).astype(np.int8)


def best_offset(ours: np.ndarray, theirs: np.ndarray, max_shift: int = 500) -> tuple[int, float]:
    """Shift o maximising the correlation of ours[k + o] with theirs[k] (ball speed series of one
    half; NaN ignored). Returns (o, correlation)."""
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    best = (0, -np.inf)
    for o in range(-max_shift, max_shift + 1):
        lo, hi = max(0, -o), min(len(b), len(a) - o)
        if hi - lo < 1000:
            continue
        x, y = a[lo + o:hi + o], b[lo:hi]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 1000:
            continue
        x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        r = float((x * y).sum() / den) if den > 0 else -np.inf
        if r > best[1]:
            best = (o, r)
    return best


def speed(xy: np.ndarray, fps: float = 25.0) -> np.ndarray:
    """Ball speed per frame (m/s), centred difference, NaN where an end is missing."""
    xy = np.asarray(xy, dtype=np.float64)
    out = np.full(len(xy), np.nan)
    if len(xy) > 2:
        out[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=1) * fps / 2
    return out


# ------------------------------------------------------------------------------------ outer steps (our env)

def _run_inner(a, step: str, extra: list[str]) -> int:
    py = a.python or sys.executable
    cmd = [py, str(Path(__file__).resolve()), "--inner", step, *extra]
    print("$", " ".join(cmd), f"(in {a.pathcrf})", flush=True)
    return subprocess.call(cmd, cwd=str(a.pathcrf), env={**os.environ, **getattr(a, "env", {})})


def cmd_setup(a) -> int:
    from ..config import EXTERNAL_RAW_DIR
    raw = Path(a.idsse_raw or EXTERNAL_RAW_DIR / "idsse")
    root = Path(a.pathcrf)
    if not (root / "train.py").is_file() or not (root / "inference.py").is_file():
        raise SystemExit(f"{root} is not a PathCRF clone (git clone https://github.com/hyunsungkim-ds/pathcrf)")
    n = 0
    for kind, sub in (("matchinformation", "metadata"), ("events", "event"), ("positions", "tracking")):
        dst = root / "data" / "sportec" / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(raw.glob(f"*{kind}*.xml")):
            t = dst / f.name
            if t.exists():
                continue
            try:
                os.link(f, t)                         # no copy of 350 MB files when the disk allows it
            except OSError:
                import shutil
                shutil.copy2(f, t)
            n += 1
    found = {sub: len(list((root / "data" / "sportec" / sub).glob("*.xml"))) for sub in ("metadata", "event", "tracking")}
    print(f"linked {n} files into {root / 'data' / 'sportec'}: {found} (expect 7 each)")
    synced = sorted(p.stem for p in (root / "data" / "sportec_synced").glob("*.parquet"))
    print(f"its synced events (labels): {synced}")
    return 0 if all(v == 7 for v in found.values()) else 1


def cmd_preprocess(a) -> int:
    return _run_inner(a, "preprocess", [])


def cmd_train(a) -> int:
    from ..external import splits
    tr = sorted(g.split("-", 1)[1] for g in splits.games("train", sources=["idsse"]))
    va = sorted(g.split("-", 1)[1] for g in splits.games("val", sources=["idsse"]))
    if tuple(tr) != TRAIN_IDS or tuple(va) != VAL_IDS:
        raise SystemExit(f"splits.json changed (train {tr}, val {va}) -- update TRAIN_IDS / VAL_IDS")
    # the validation file only drives its "best" checkpoints, which we do not use (final epoch only)
    a.env = {"MQF_SPLIT": json.dumps({"train": tr, "val": [va[0]]})}
    return _run_inner(a, "train", ["--trial", str(a.trial), "--epochs", str(a.epochs), "--workers", str(a.workers),
                                   *(["--extra", *a.extra] if a.extra else [])])


def cmd_predict(a) -> int:
    return _run_inner(a, "predict", ["--trial", str(a.trial), "--epochs", str(a.epochs),
                                     "--games", *(a.games or VAL_IDS)])


def cmd_align(a) -> int:
    """<pathcrf>/mqf_out/<id>.npz -> external/pathcrf/IDSSE-<id>.npz (P(home) per bundle frame)."""
    import pandas as pd
    from .. import atd
    from ..config import EXTERNAL_DIR
    from ..external import bundle
    out_dir = Path(a.out or EXTERNAL_DIR / "pathcrf")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for gid in a.games or VAL_IDS:
        f = Path(a.pathcrf) / OUT_DIR / f"{gid}.npz"
        if not f.exists():
            print(f"{gid}: no {f} -- run predict first")
            continue
        z = np.load(f, allow_pickle=False)
        d = bundle.find("idsse", [f"IDSSE-{gid}"])
        if not d:
            print(f"{gid}: no bundle IDSSE-{gid}")
            continue
        spec = bundle.spec(d[0])
        trk = atd.load(spec, cache_dir=d[0] / "cache")
        gt = pd.read_csv(spec.gt_public, dtype={"label": str})["label"].to_numpy()
        n = spec.n_frames
        team = np.full(n, -1, dtype=np.int8)
        covered = np.zeros(n, dtype=bool)
        their_alive = np.zeros(n, dtype=bool)
        their_gt = np.full(n, -1, dtype=np.int8)
        info = {"game": spec.name}
        for h, (half, first, last) in enumerate(spec.halves):
            m = z["period_id"] == half
            if not m.any():
                continue
            fid = z["frame_id"][m]
            order = np.argsort(fid)
            fid = fid[order]
            k = fid - fid[0]                                       # their frames, from the start of the period
            ours = speed(trk.ball[first:last + 1])
            theirs = np.full(k[-1] + 1, np.nan)
            theirs[k] = speed(np.c_[z["ball_x"][m][order], z["ball_y"][m][order]])
            o, r = best_offset(ours, theirs, a.max_shift)
            pos = first + k + o
            ok = (pos >= first) & (pos <= last)
            team[pos[ok]] = z["team"][m][order][ok]
            covered[pos[ok]] = True
            their_alive[pos[ok]] = z["alive"][m][order][ok].astype(bool)
            their_gt[pos[ok]] = z["gt_team"][m][order][ok]
            info[f"half{half}"] = {"offset": int(o), "corr": round(r, 4), "frames": int(ok.sum()),
                                   "their_gaps": int((np.diff(fid) != 1).sum())}
        ours_alive = np.isin(gt, ["H", "A"])
        both = ours_alive & their_alive
        agree_alive = float(np.mean(ours_alive[covered] == their_alive[covered])) if covered.any() else float("nan")
        gt_team = np.where(gt == "H", 0, np.where(gt == "A", 1, -1))
        m2 = both & (their_gt >= 0) & (gt_team >= 0)
        agree_team = float(np.mean(their_gt[m2] == gt_team[m2])) if m2.any() else float("nan")
        flipped = bool(np.isfinite(agree_team) and agree_team < 0.5)
        if flipped:                                                  # their home is our away
            team = np.where(team >= 0, 1 - team, team)
            agree_team = 1 - agree_team
        p_home = np.where(team == 0, 1 - CLIP, np.where(team == 1, CLIP, np.nan))
        cover = float(np.mean(np.isfinite(p_home[ours_alive]))) if ours_alive.any() else 0.0
        m3 = ours_alive & np.isfinite(p_home) & (gt_team >= 0)
        acc = float(np.mean((p_home[m3] > 0.5) == (gt_team[m3] == 0))) if m3.any() else float("nan")
        info.update(alive_agreement=round(agree_alive, 4), gt_team_agreement=round(agree_team, 4),
                    home_away_flipped=flipped, coverage_of_alive=round(cover, 4), team_acc_alive=round(acc, 4))
        np.savez_compressed(out_dir / f"{spec.name}.npz", p_home=p_home, info=np.array(json.dumps(info)))
        rows.append(info)
        print(json.dumps(info))
    bad = [r["game"] for r in rows if not (r["gt_team_agreement"] > 0.97 and r["alive_agreement"] > 0.97 and all(
        r.get(f"half{h}", {}).get("corr", 0) > 0.9 for h in (1, 2)))]
    if bad:
        print(f"CHECK {bad}: frames may be misaligned (expect corr > 0.9 and DFL-flag agreement > 0.97) -- "
              "do not use these predictions until the offsets make sense")
    print(f"wrote {len(rows)} file(s) to {out_dir}")
    return 0 if rows and not bad else 1


# ------------------------------------------------------------------------------------ inner steps (its env)

def _inner_preprocess(argv) -> int:
    import runpy
    sys.path.insert(0, os.getcwd())
    sys.argv = ["datatools/preprocess.py"]
    runpy.run_path("datatools/preprocess.py", run_name="__main__")
    return 0


def _inner_train(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args(argv)
    src = Path("train.py").read_text(encoding="utf-8")
    if src.count(SPLIT_ANCHOR) != 1:
        raise SystemExit("PathCRF train.py changed: the split anchor was not found once -- update SPLIT_ANCHOR")
    Path("train_mqf.py").write_text(src.replace(SPLIT_ANCHOR, SPLIT_PATCH), encoding="utf-8")   # local copy, MPL-2.0
    if Path(f"saved/{a.trial:03d}").exists():
        raise SystemExit(f"saved/{a.trial:03d} exists -- pick another --trial (its runs are never overwritten)")
    args = ["--trial", str(a.trial), *PAPER_ARGS, "--n_epochs", str(a.epochs), "--save_epoch", str(a.epochs),
            "--num_workers", str(a.workers), *a.extra]               # later flags win (argparse)
    print("python train_mqf.py " + " ".join(args), flush=True)
    return subprocess.call([sys.executable, "train_mqf.py", *args])


def _inner_predict(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--games", nargs="*", default=list(VAL_IDS))
    a = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    import pandas as pd
    import torch
    from datatools.sportec_data import SportecData
    from inference import inference
    from models.utils import build_model, load_trial_args
    save = f"saved/{a.trial:03d}"
    targs = load_trial_args(save)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(targs, device=dev)
    ckpt = f"{save}/model/state_dict_{a.epochs}.pt"                # the final epoch: no checkpoint selection
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    fps = 25.0
    for gid in a.games:
        tracking = pd.read_parquet(f"data/sportec/tracking_processed/{gid}.parquet")
        _, _, pred, stats = inference(model, tracking, ball_feature=targs.get("ball_feature", False), use_crf=True,
                                      decode="indep", correct_episode_lasts=False, evaluate=True,
                                      window_seconds=targs.get("window_seconds"), fps=fps,
                                      sample_freq=int(fps / targs.get("target_fps", 5.0)), skip_red_phases=False)
        tr = tracking.set_index("frame_id").sort_index()
        pred = pred.reindex(tr.index)
        team = edges_to_team(pred["edge_src"].to_numpy(object), pred["edge_dst"].to_numpy(object))
        lineup = SportecData(gid, load_tracking=False).lineup
        side = {str(t): (0 if h == "home" else 1) for t, h in lineup[["team_id", "home_away"]].drop_duplicates().to_numpy()}
        gt_team = np.array([side.get(str(t), -1) for t in tr["ball_owning_team_id"].to_numpy(object)], dtype=np.int8)
        np.savez_compressed(f"{OUT_DIR}/{gid}.npz", frame_id=tr.index.to_numpy(np.int64),
                            period_id=tr["period_id"].to_numpy(np.int64), team=team,
                            alive=(tr["ball_state"].astype(str).str.lower() == "alive").to_numpy(),
                            gt_team=gt_team, ball_x=tr["ball_x"].to_numpy(np.float64), ball_y=tr["ball_y"].to_numpy(np.float64))
        keep = {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else None)
                for k, v in stats.items() if k in ("macro_acc", "edge_acc", "src_acc", "dst_acc", "forbid_rate")}
        print(f"{gid}: {keep}  frames with an edge {int((team >= 0).sum())}/{len(team)}", flush=True)
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--inner":
        step, rest = argv[1], argv[2:]
        return {"preprocess": _inner_preprocess, "train": _inner_train, "predict": _inner_predict}[step](rest)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "preprocess", "train", "predict", "align"):
        s = sub.add_parser(name)
        s.add_argument("--pathcrf", type=Path, required=True, help="the PathCRF clone (outside this repo)")
        s.add_argument("--python", help="PathCRF's interpreter (default: this one)")
        if name == "setup":
            s.add_argument("--idsse-raw", type=Path, help="raw IDSSE XMLs (default: external_data/idsse)")
        if name in ("train", "predict"):
            s.add_argument("--trial", type=int, default=901)
            s.add_argument("--epochs", type=int, default=10)
        if name == "train":
            s.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
            s.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                           help="more flags for its train.py, last (e.g. --extra --window_stride 5 for a quick run)")
        if name in ("predict", "align"):
            s.add_argument("--games", nargs="*")
        if name == "align":
            s.add_argument("--out", type=Path)
            s.add_argument("--max-shift", type=int, default=500)
    a = ap.parse_args(argv)
    return {"setup": cmd_setup, "preprocess": cmd_preprocess, "train": cmd_train, "predict": cmd_predict,
            "align": cmd_align}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
