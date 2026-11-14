"""Train a ball model on the other games' bundles (GPU machine).

    python -m mqf.ball.train --arch B --player-mask off --tag v4 --amp --name ball_B_off_v4     # v4 defaults
    python -m mqf.ball.train --arch B --player-mask off --tag v3 --amp --ctx-dim 4 --aux 0     # v3: models/ball_B_off_r.pt
    python -m mqf.ball.train --arch B --player-mask on  --tag v3 --batch 8 --amp
    python -m mqf.ball.train --arch B --player-mask off --ctx off --epochs 30 --tag v1   # the v1 model (ball_B_off.pt)
    python -m mqf.ball.train --arch B --player-mask off --max-windows 64 --epochs 1   # smoke test

Train = IDSSE 5 + Metrica 2 (+ PFF 56 with --sources ... pff, the v5 default), validation = IDSSE 2 +
Metrica 1 (splits.json) -- the same validation games as v1-v4, so best val ADE stays comparable.
Training games are decimated to the model rate in memory (data.decimate). Loss and the validation
ADE only on ALIVE frames whose ball was hidden by the mask. Best epoch by validation ADE is saved to
models/ball_<arch>_<mask>[_r].pt (+ .json with config and history). Aggregates only on stdout.
v3 (--ctx on, default): the model reads the gap context and predicts a correction to the linear
interpolation of the seen ball; its head starts at zero, i.e. training starts AS interpolation.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..config import HARNESS_DIR, MODELS_DIR
from . import data, models

VAL_SEED, VAL_STRIDE_S = 1234, 48.0


class Windows(torch.utils.data.Dataset):
    def __init__(self, games, index, player_mask, seed, train=True, fixed=False):
        self.games, self.index, self.pm, self.seed, self.train, self.fixed = games, index, player_mask, seed, train, fixed
        self.epoch = 0

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        gi, h, s = self.index[i]
        key = [self.seed, i] if self.fixed else [self.seed, self.epoch, i]
        rng = np.random.default_rng(key)
        return data.make_sample(self.games[gi], h, s, rng, self.pm, train=self.train)


def collate(batch):
    out = data.collate(batch)
    t = {k: torch.from_numpy(np.asarray(v)) for k, v in out.items()}
    t["nearest"] = t["nearest"].long()
    for k in ("present", "ball_obs"):
        t[k] = t[k].bool()
    return t


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def run_epoch(model, loader, device, opt=None, sched=None, amp=False, clip=1.0, aux_w=0.0):
    train = opt is not None
    model.train(train)
    tot_loss = tot_ade = tot_w = 0.0
    buckets = {"le2s": [0.0, 0.0], "2-8s": [0.0, 0.0], "gt8s": [0.0, 0.0]}
    hit = {"agent": 0.0, "team": 0.0, "n": 0.0}
    for b in loader:
        b = to_device(b, device)
        with torch.set_grad_enabled(train), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred, logits = models.forward(model, b, with_logits=True)
        pm = models.to_metres(pred.float(), b["pitch"])
        loss, ade = models.loss_and_ade(pm, b["target"], b["w_masked"], b["w_seen"])
        if logits is not None and aux_w > 0:
            loss = loss + aux_w * models.nearest_loss(logits, b["nearest"], b["w_masked"] + 0.1 * b["w_seen"])
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
        w = float(b["w_masked"].sum())
        tot_loss += loss.item() * w
        tot_ade += ade.item() * w
        tot_w += w
        if not train:                                       # ADE by the length of the hidden run
            dist = torch.linalg.norm(pm - torch.nan_to_num(b["target"]), dim=-1).cpu().numpy()
            wm = b["w_masked"].cpu().numpy()
            hidden = (~b["ball_obs"]).cpu().numpy()
            for r in range(len(wm)):
                e = np.diff(np.r_[0, hidden[r].astype(np.int8), 0])
                for s_, t_ in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
                    L = (t_ - s_) / data.HZ
                    k = "le2s" if L <= 2 else "2-8s" if L <= 8 else "gt8s"
                    buckets[k][0] += float((dist[r, s_:t_] * wm[r, s_:t_]).sum())
                    buckets[k][1] += float(wm[r, s_:t_].sum())
            if logits is not None:                          # nearest player / its team on hidden frames
                nr = b["nearest"]
                sel = (b["w_masked"] > 0) & (nr >= 0)
                if sel.any():
                    k = logits.argmax(-1)
                    tm = b["team"]
                    team_pred = torch.gather(tm, 1, k)
                    team_true = torch.gather(tm, 1, nr.clamp(min=0))
                    hit["agent"] += float((k == nr)[sel].sum())
                    hit["team"] += float((team_pred == team_true)[sel].sum())
                    hit["n"] += float(sel.sum())
    out = {"loss": tot_loss / max(tot_w, 1), "ade_m": tot_ade / max(tot_w, 1)}
    if not train:
        out.update({f"ade_{k}": (v[0] / v[1] if v[1] else None) for k, v in buckets.items()})
        out["nearest_acc"] = hit["agent"] / hit["n"] if hit["n"] else None
        out["nearest_team_acc"] = hit["team"] / hit["n"] if hit["n"] else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=["A", "B"], required=True)
    ap.add_argument("--player-mask", choices=["on", "off"], default="off")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=None, help="A: temporal layers (4); B: axial blocks (3)")
    ap.add_argument("--ctx", choices=["on", "off"], default="on", help="v3 gap context + correction head")
    ap.add_argument("--ctx-dim", type=int, default=data.CTX_DIM, choices=[4, 8], help="4 = v3, 8 = v4 (+ edge velocities)")
    ap.add_argument("--aux", type=float, default=0.5, help="weight of the nearest-player head (B only; 0 = off)")
    ap.add_argument("--name", default=None, help="model file stem (default ball_<arch>_<mask>[_r])")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (persistent)")
    ap.add_argument("--sources", nargs="*", default=["idsse", "metrica", "pff"], help="training sources")
    ap.add_argument("--val-sources", nargs="*", default=["idsse", "metrica"], help="validation sources")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-windows", type=int, default=None, help="smoke test: cap train/val windows")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    pm = args.player_mask == "on"
    t0 = time.time()
    log = lambda m: print(m, flush=True)
    tr_games = data.load_split("train", args.sources, decimate_=True, log=log if len(args.sources) > 2 else None)
    va_games = data.load_split("val", args.val_sources)
    if not tr_games or not va_games:
        raise SystemExit("no bundles: python tasks.py bundles")
    tr_idx = [(gi, h, s) for gi, g in enumerate(tr_games) for h, s in data.window_starts(g)]
    va_idx = [(gi, h, s) for gi, g in enumerate(va_games) for h, s in data.window_starts(g, stride_s=VAL_STRIDE_S)]
    if args.max_windows:
        rng = np.random.default_rng(args.seed)
        tr_idx = [tr_idx[i] for i in sorted(rng.choice(len(tr_idx), min(args.max_windows, len(tr_idx)), replace=False))]
        va_idx = va_idx[:max(4, args.max_windows // 4)]
    if device.type == "cpu" and args.device == "auto":
        print(f"WARNING: torch {torch.__version__} sees no CUDA GPU -- training on CPU will be slow "
              "(python tasks.py torch-info)", flush=True)
    print(f"train {len(tr_games)} games / {len(tr_idx)} windows   val {len(va_games)} games / {len(va_idx)} windows   "
          f"device {device}   loaded in {time.time() - t0:.0f} s", flush=True)

    cfg = {"arch": args.arch, "player_mask": args.player_mask, "d": args.d, "heads": args.heads,
           "layers": args.layers or (4 if args.arch == "A" else 3), "set_layers": 2, "dropout": 0.1,
           "hz": data.HZ, "window_s": data.WINDOW_S, "a_max": data.A_MAX, "seed": args.seed,
           "ctx": args.ctx == "on", "context_s": data.CONTEXT_S, "ctx_dim": args.ctx_dim,
           "aux": args.arch == "B" and args.aux > 0, "aux_w": args.aux if args.arch == "B" else 0.0,
           "train_games": [g.name for g in tr_games], "val_games": [g.name for g in va_games],
           "train_sources": sorted({g.source for g in tr_games})}
    model = models.build(cfg)
    if cfg["ctx"]:
        models.zero_correction(model)
    model = model.to(device)
    print(f"model {args.arch}: {models.n_params(model) / 1e6:.2f} M parameters", flush=True)

    tr_ds = Windows(tr_games, tr_idx, pm, args.seed, train=True)
    va_ds = Windows(va_games, va_idx, pm, VAL_SEED, train=False, fixed=True)
    gen = torch.Generator().manual_seed(args.seed)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                                        num_workers=args.workers, generator=gen, drop_last=True)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=args.batch, shuffle=False, collate_fn=collate,
                                        num_workers=args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = max(1, args.epochs * len(tr_dl))
    warm = min(500, total // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.name or f"ball_{args.arch}_{args.player_mask}" + ("_r" if cfg["ctx"] else "")
    best, bad, hist = np.inf, 0, []
    for ep in range(args.epochs):
        tr_ds.epoch = ep
        t1 = time.time()
        tr = run_epoch(model, tr_dl, device, opt, sched, amp=args.amp and device.type == "cuda", aux_w=cfg["aux_w"])
        with torch.no_grad():
            va = run_epoch(model, va_dl, device, aux_w=cfg["aux_w"])
        hist.append({"epoch": ep, "train_loss": tr["loss"], **{f"val_{k}": v for k, v in va.items()}})
        fmt = lambda v: f"{v:.2f}" if v is not None else "-"
        near = (f"  nearest {va['nearest_acc']:.2f} / team {va['nearest_team_acc']:.2f}"
                if va.get("nearest_acc") is not None else "")
        print(f"epoch {ep:>2}  train loss {tr['loss']:.3f}  val ADE {va['ade_m']:.2f} m  "
              f"(<=2 s {fmt(va['ade_le2s'])}  2-8 s {fmt(va['ade_2-8s'])}  >8 s {fmt(va['ade_gt8s'])}){near}  "
              f"{time.time() - t1:.0f} s", flush=True)
        if va["ade_m"] < best - 1e-3:
            best, bad = va["ade_m"], 0
            torch.save({"config": cfg, "state": model.state_dict()}, MODELS_DIR / f"{name}.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break
    report = {"name": name, "tag": args.tag, "best_val_ade_m": best, "config": cfg, "history": hist,
              "params": models.n_params(model)}
    (MODELS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2, default=float))
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    (HARNESS_DIR / f"ball_train_{name}_{args.tag}.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"best val ADE {best:.2f} m -> {MODELS_DIR / (name + '.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
