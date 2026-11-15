"""The learned dead-ball models behind one load / predict.

  gbm  sklearn HistGradientBoostingClassifier with dead_v1's settings (300 iterations, lr 0.05, 31 leaves,
       min 200 per leaf, l2 1) -- NaN-aware, deterministic
  mlp  the same table, standardised, NaN -> 0 plus a missing flag per column; 3 x 256 GELU (mqf.owner's)
  tcn  ASRF-style temporal segmentation (Ishikawa et al. 2021, after MS-TCN, Farha & Gall 2019) on the
       5 Hz rows: mqf.owner's dilated residual TCN (10 layers, dilation 1..512, ~400 s receptive field),
       two class stages and a boundary head -- here P(dead) and P(alive <-> dead change near this step)
  asf  ASFormer-lite (Yi et al. 2021): the same towers with windowed self-attention (64 steps = 12.8 s,
       shifted every other layer) after each dilated convolution -- the backbone swap ASFormer's authors
       report as a drop-in gain for ASRF
Sequence models train on 256 s windows (BCE on every labelled step for each class stage + a positive-
weighted BCE on a soft boundary target) and predict whole halves in one pass, interpolated to every frame.
Outputs are rounded to 1e-4 (reproducible downstream, as dead.predict).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..owner.models import _device, _mlp, _norm_stats, _prep, _seed_all, _state_cpu, _tcn_net

WIN = 1280             # steps (256 s at 5 Hz)
STRIDE = 640
BND_SIGMA = 1.5        # steps
SEQ_KINDS = ("tcn", "asf")
KINDS = ("gbm", "mlp", "tcn", "asf")


def _asf_net(n_in: int, d: int = 64, layers: int = 10, stages: int = 2, heads: int = 4, win: int = 64,
             dropout: float = 0.1):
    import torch
    import torch.nn as nn

    class LocalAttn(nn.Module):
        def __init__(self, shift: int):
            super().__init__()
            self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.shift = shift

        def forward(self, x):                      # (B, d, T)
            B, C, T = x.shape
            h = x.transpose(1, 2)                  # (B, T, d)
            if self.shift:
                h = torch.roll(h, -self.shift, dims=1)
            pad = (-T) % win
            if pad:
                h = torch.cat([h, h.new_zeros(B, pad, C)], dim=1)
            nW = h.shape[1] // win
            hw = self.norm(h.reshape(B * nW, win, C))
            a, _ = self.att(hw, hw, hw, need_weights=False)
            a = a.reshape(B, nW * win, C)[:, :T]
            if self.shift:
                a = torch.roll(a, self.shift, dims=1)
            return a.transpose(1, 2)

    class Block(nn.Module):
        def __init__(self, dil, shift):
            super().__init__()
            self.conv = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
            self.att = LocalAttn(shift)
            self.out = nn.Conv1d(d, d, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = torch.relu(self.conv(x))
            h = h + self.att(h)
            return x + self.drop(self.out(h))

    def tower(c_in):
        return nn.Sequential(nn.Conv1d(c_in, d, 1), *[Block(2 ** i, (win // 2) * (i % 2)) for i in range(layers)])

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = tower(n_in)
            self.cls = nn.Conv1d(d, 1, 1)
            self.bnd = nn.Conv1d(d, 1, 1)
            self.refine = nn.ModuleList([nn.Sequential(tower(1), nn.Conv1d(d, 1, 1)) for _ in range(stages - 1)])

        def forward(self, x):                      # (B, T, F) -> [stage logits (B, T)], boundary logit (B, T)
            h = self.trunk(x.transpose(1, 2))
            outs = [self.cls(h)]
            for r in self.refine:
                outs.append(r(torch.sigmoid(outs[-1])))
            return [o.squeeze(1) for o in outs], self.bnd(h).squeeze(1)

    return Net()


def boundary_target(y: np.ndarray, sigma: float = BND_SIGMA) -> np.ndarray:
    """Soft target per step, exp(-d^2 / 2 sigma^2), d = steps to the nearest alive <-> dead change
    between two labelled steps (both neighbours of the change get 1)."""
    y = np.asarray(y)
    ch = np.flatnonzero((y[1:] >= 0) & (y[:-1] >= 0) & (y[1:] != y[:-1])) + 1
    out = np.zeros(len(y), dtype=np.float32)
    if len(ch) == 0:
        return out
    t = np.arange(len(y), dtype=np.float64)
    mid = ch - 0.5
    k = np.searchsorted(mid, t)
    d = np.minimum(np.abs(t - mid[np.clip(k, 0, len(mid) - 1)]), np.abs(t - mid[np.clip(k - 1, 0, len(mid) - 1)]))
    d = np.maximum(d - 0.5, 0.0)
    return np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)


class AliveModel:
    def __init__(self, kind: str, columns: list[str], feature_set: str, meta: dict, obj=None, state=None, norm=None,
                 arch=None, rule_cfg: dict | None = None):
        self.kind, self.columns, self.feature_set, self.meta = kind, list(columns), feature_set, meta
        self.obj, self.state, self.norm, self.arch = obj, state, norm, arch or {}
        self.rule_cfg = rule_cfg or {}
        self._net = None

    # -------------------------------------------------------------- persistence
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.kind == "gbm":
            import joblib
            joblib.dump({"kind": "gbm", "model": self.obj, "columns": self.columns, "feature_set": self.feature_set,
                         "meta": self.meta, "rule_cfg": self.rule_cfg}, path)
        else:
            import torch
            torch.save({"kind": self.kind, "state": self.state, "columns": self.columns, "feature_set": self.feature_set,
                        "meta": self.meta, "norm": {k: np.asarray(v) for k, v in self.norm.items()}, "arch": self.arch,
                        "rule_cfg": self.rule_cfg}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "AliveModel":
        path = Path(path)
        if path.suffix == ".joblib":
            import joblib
            o = joblib.load(path)
            if "kind" not in o:                     # a mqf.dead model (dead_v1): same contract, v1 columns
                return cls("gbm", o["columns"], "v1", o.get("meta", {}), obj=o["model"])
            return cls("gbm", o["columns"], o["feature_set"], o["meta"], obj=o["model"], rule_cfg=o.get("rule_cfg"))
        import torch
        o = torch.load(path, map_location="cpu", weights_only=False)
        return cls(o["kind"], o["columns"], o["feature_set"], o["meta"], state=o["state"], norm=o["norm"],
                   arch=o["arch"], rule_cfg=o.get("rule_cfg"))

    # -------------------------------------------------------------- inference
    def _torch_net(self, device):
        if self._net is None:
            import torch
            a = self.arch
            if self.kind == "mlp":
                net = _mlp(int(a["n_in"]), a.get("hidden", 256), a.get("depth", 3))
            elif self.kind == "tcn":
                net = _tcn_net(int(a["n_in"]), a.get("d", 64), a.get("layers", 10), a.get("stages", 2))
            else:
                net = _asf_net(int(a["n_in"]), a.get("d", 64), a.get("layers", 10), a.get("stages", 2),
                               a.get("heads", 4), a.get("win", 64))
            net.load_state_dict({k: torch.as_tensor(v) for k, v in self.state.items()})
            net.eval()
            self._net = net.to(device)
        return self._net

    def _p_rows(self, X: np.ndarray, device="cpu", batch: int = 65536) -> np.ndarray:
        if self.kind == "gbm":
            return self.obj.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
        import torch
        dev = _device(device)
        net = self._torch_net(dev)
        Z = _prep(X, self.norm["mu"], self.norm["sd"], self.norm["mask_cols"])
        out = np.empty(len(Z), dtype=np.float64)
        with torch.no_grad():
            for a in range(0, len(Z), batch):
                out[a:a + batch] = torch.sigmoid(net(torch.as_tensor(Z[a:a + batch], device=dev))).squeeze(-1).cpu().numpy()
        return out

    def _seq(self, S: np.ndarray, device="cpu"):
        import torch
        dev = _device(device)
        net = self._torch_net(dev)
        Z = _prep(S, self.norm["mu"], self.norm["sd"], self.norm["mask_cols"])
        with torch.no_grad():
            outs, bnd = net(torch.as_tensor(Z[None], device=dev))
        return outs[-1].squeeze(0).cpu().numpy().astype(np.float64), bnd.squeeze(0).cpu().numpy().astype(np.float64)

    def predict(self, X, halves, n: int, fps: float, device="cpu"):
        """X: the full_rules table (feats.features) as a DataFrame of n rows. Returns ((n,) P(dead), (n,)
        P(boundary) or None), NaN outside the halves, rounded to 1e-4."""
        from .data import step_of
        p = np.full(n, np.nan)
        pb = np.full(n, np.nan) if self.kind in SEQ_KINDS else None
        inside = np.zeros(n, dtype=bool)
        for _, a, b in halves:
            inside[a:b + 1] = True
        if not inside.any():
            return p, pb
        A = X[self.columns].to_numpy(dtype=np.float32)
        if self.kind in ("gbm", "mlp"):
            p[inside] = self._p_rows(A[inside], device)
            return np.round(p, 4), None
        k = step_of(fps)
        for _, a, b in halves:
            idx = np.arange(a, b + 1, k)
            fr = np.arange(a, b + 1)
            lg, lb = self._seq(A[idx], device)
            p[a:b + 1] = 1.0 / (1.0 + np.exp(-np.interp(fr, idx, lg)))
            pb[a:b + 1] = 1.0 / (1.0 + np.exp(-np.interp(fr, idx, lb)))
        return np.round(p, 4), np.round(pb, 4)


# ------------------------------------------------------------------------------------ training

def train_gbm(X, y, columns, feature_set, meta, iters=300, seed=0, rule_cfg=None, log=print) -> AliveModel:
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    model = HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                           l2_regularization=1.0, early_stopping=False, random_state=seed)
    model.fit(X, y)
    log(f"  gbm fit {len(X):,} rows x {X.shape[1]} in {time.time() - t0:.0f} s")
    return AliveModel("gbm", columns, feature_set, meta, obj=model, rule_cfg=rule_cfg)


def train_mlp(X, y, columns, feature_set, meta, epochs=4, batch=4096, lr=1e-3, seed=0, device="auto", hidden=256,
              depth=3, rule_cfg=None, log=print) -> AliveModel:
    import torch
    _seed_all(seed)
    dev = _device(device)
    mu, sd = _norm_stats(X)
    mask_cols = np.isnan(X).any(axis=0)
    n_in = int(X.shape[1] + mask_cols.sum())
    net = _mlp(n_in, hidden, depth).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(X) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    yf = y.astype(np.float32)
    t0 = time.time()
    net.train()
    for ep in range(epochs):
        order = rng.permutation(len(X))
        tot = 0.0
        for a in range(0, len(X), batch):
            ib = order[a:a + batch]
            xb = torch.as_tensor(_prep(X[ib], mu, sd, mask_cols), device=dev)
            yb = torch.as_tensor(yf[ib], device=dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(ib)
        log(f"  mlp epoch {ep + 1}/{epochs} loss {tot / len(X):.4f} ({time.time() - t0:.0f} s)")
    net.eval()
    return AliveModel("mlp", columns, feature_set, meta, state=_state_cpu(net),
                      norm={"mu": mu, "sd": sd, "mask_cols": mask_cols},
                      arch={"n_in": n_in, "hidden": hidden, "depth": depth}, rule_cfg=rule_cfg)


def train_seq(kind: str, blocks, columns, feature_set, meta, epochs=20, batch=8, lr=1e-3, seed=0, device="auto",
              d=64, layers=10, stages=2, heads=4, win=64, w_bnd=1.0, rule_cfg=None, log=print) -> AliveModel:
    """blocks = data.sequences(...): [(S (T, F), y (T,))] per game x view x half."""
    import torch
    import torch.nn.functional as F
    if kind not in SEQ_KINDS:
        raise ValueError(kind)
    _seed_all(seed)
    dev = _device(device)
    Fdim = blocks[0][0].shape[1]
    s1, s2, cnt = np.zeros(Fdim), np.zeros(Fdim), np.zeros(Fdim)
    mask_cols = np.zeros(Fdim, dtype=bool)
    for S, _ in blocks:
        S = S.astype(np.float64)
        ok = ~np.isnan(S)
        mask_cols |= ~ok.all(axis=0)
        S = np.where(ok, S, 0.0)
        s1 += S.sum(axis=0)
        s2 += (S * S).sum(axis=0)
        cnt += ok.sum(axis=0)
    mu = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    sd = np.sqrt(np.maximum(np.where(cnt > 0, s2 / np.maximum(cnt, 1) - mu ** 2, 1.0), 0.0))
    mu, sd = mu.astype(np.float32), np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
    Zs, Ys, Bs, wins = [], [], [], []
    for S, y in blocks:
        if len(S) < 32:
            continue
        Zs.append(_prep(S, mu, sd, mask_cols).astype(np.float16))
        Ys.append(y.astype(np.float32))
        Bs.append(boundary_target(y))
        T = len(S)
        w = min(WIN, T)
        for st in list(range(0, T - w + 1, STRIDE)) or [0]:
            if (y[st:st + w] >= 0).sum() >= 20:
                wins.append((len(Zs) - 1, st, w))
        if T > w and (T - w) % STRIDE:
            wins.append((len(Zs) - 1, T - w, w))
    if not wins:
        raise SystemExit(f"{kind}: no training windows")
    W = max(w[2] for w in wins)
    n_in = Zs[0].shape[1]
    pos_share = float(np.mean([b[y >= 0].mean() for b, y in zip(Bs, Ys) if (y >= 0).any()]))
    pos_weight = float(np.clip((1 - pos_share) / max(pos_share, 1e-4), 1.0, 50.0))
    log(f"  {kind} {len(wins):,} windows of <= {W} steps x {n_in} inputs, boundary share {pos_share:.3f} "
        f"(pos_weight {pos_weight:.1f})")

    def batch_of(ids):
        xb = np.zeros((len(ids), W, n_in), dtype=np.float32)
        yb = np.full((len(ids), W), -1.0, dtype=np.float32)
        bb = np.zeros((len(ids), W), dtype=np.float32)
        for r, i in enumerate(ids):
            k, st, w = wins[i]
            xb[r, :w] = Zs[k][st:st + w]
            yb[r, :w] = Ys[k][st:st + w]
            bb[r, :w] = Bs[k][st:st + w]
        return torch.as_tensor(xb), torch.as_tensor(yb), torch.as_tensor(bb)

    arch = {"n_in": int(n_in), "d": d, "layers": layers, "stages": stages}
    if kind == "asf":
        arch.update(heads=heads, win=win)
        net = _asf_net(n_in, d, layers, stages, heads, win).to(dev)
    else:
        net = _tcn_net(n_in, d, layers, stages).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(wins) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps))
    pw = torch.tensor(pos_weight, device=dev)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    net.train()
    for ep in range(epochs):
        order = rng.permutation(len(wins))
        tot_c = tot_b = 0.0
        cnt_ = 0
        for a in range(0, len(wins), batch):
            xb, yb, bb = (t.to(dev) for t in batch_of(order[a:a + batch]))
            m = yb >= 0
            opt.zero_grad(set_to_none=True)
            outs, lb = net(xb)
            lc = sum(F.binary_cross_entropy_with_logits(o[m], yb[m]) for o in outs) / len(outs)
            lbd = F.binary_cross_entropy_with_logits(lb[m], bb[m], pos_weight=pw)
            loss = lc + w_bnd * lbd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            n_m = int(m.sum())
            tot_c += lc.item() * n_m
            tot_b += lbd.item() * n_m
            cnt_ += n_m
        log(f"  {kind} epoch {ep + 1}/{epochs} class {tot_c / max(cnt_, 1):.4f} boundary {tot_b / max(cnt_, 1):.4f} "
            f"({time.time() - t0:.0f} s)")
    net.eval()
    return AliveModel(kind, columns, feature_set, meta, state=_state_cpu(net),
                      norm={"mu": mu, "sd": sd, "mask_cols": mask_cols}, arch=arch, rule_cfg=rule_cfg)
