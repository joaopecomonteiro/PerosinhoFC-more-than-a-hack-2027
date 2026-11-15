"""The learned owners: GBM, MLP and BiGRU, behind one load / predict.

  gbm  sklearn HistGradientBoostingClassifier, who_v1's settings (300 iterations, lr 0.05, 31
       leaves, min 200 per leaf, l2 1) -- NaN-aware, deterministic
  mlp  the same feature table, standardised, NaN -> 0 plus a missing flag per column; 3 x 256 GELU,
       dropout 0.1, AdamW, BCE. Answers "does a NN beat the trees on the SAME features?"
  gru  5 Hz sequences of the per-frame inputs (feats.seq_inputs: no hand-made clocks or windows, plus
       the 3 nearest players of each team relative to the ball) -> linear 64 -> 2-layer BiGRU (64 per
       direction) -> P(home) per step. Trained on 64 s windows (loss on alive steps only), predicted
       with half-overlapping windows (triangular weights) and interpolated to every frame.
       Answers "can a NN learn the context the clocks hand-code?"

  tcn  ASRF-style temporal segmentation (Ishikawa et al. 2021, after MS-TCN, Farha & Gall 2019) on 5 Hz
       sequences of the FULL table + the nearest players: a dilated residual TCN trunk (10 layers,
       dilation 1..512, receptive field ~400 s), two class stages (the second refines the first's
       probabilities) and a boundary head P(H <-> A change near this step). No smoothing loss: our
       error is under-segmentation (EXPERIMENTS #35-36), and the boundary head is what lowers the
       decoder's switch cost where a change is likely. Trained on 256 s windows, predicted on whole halves.

Every model is applied to both team orders and averaged, P(home) = (p(view) + 1 - p(swapped)) / 2,
so P(H) + P(A) = 1 exactly; rounded to 1e-4 like who.predict.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import rules

GRU_WIN = 320          # steps (64 s at 5 Hz)
GRU_STRIDE_TRAIN = 160
GRU_STRIDE_PRED = 160
TCN_WIN = 1280         # steps (256 s at 5 Hz)
TCN_STRIDE = 640
BND_SIGMA = 1.0        # steps: soft boundary target around an H <-> A change


def _device(name: str = "auto"):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed_all(seed: int):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)


def _norm_stats(X: np.ndarray):
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
    mu = np.nan_to_num(mu, nan=0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0).astype(np.float32)
    return mu, sd


def _prep(X: np.ndarray, mu, sd, mask_cols: np.ndarray) -> np.ndarray:
    """Standardise, clip to +-8, NaN -> 0, append a 0/1 missing flag for the columns that can be NaN."""
    Z = (np.asarray(X, dtype=np.float32) - mu) / sd
    miss = np.isnan(Z)
    Z = np.clip(np.nan_to_num(Z, nan=0.0), -8, 8)
    return np.concatenate([Z, miss[:, mask_cols].astype(np.float32)], axis=1)


# ------------------------------------------------------------------------------------ networks

def _mlp(n_in: int, hidden: int = 256, depth: int = 3, dropout: float = 0.1):
    import torch.nn as nn
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
        d = hidden
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _gru_net(n_in: int, d: int = 64, layers: int = 2, dropout: float = 0.1):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, d), nn.GELU())
            self.gru = nn.GRU(d, d, num_layers=layers, batch_first=True, bidirectional=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.out = nn.Linear(2 * d, 1)

        def forward(self, x):                          # (B, T, F) -> (B, T) logits
            h, _ = self.gru(self.inp(x))
            return self.out(h).squeeze(-1)

    return Net()


def _tcn_net(n_in: int, d: int = 64, layers: int = 10, stages: int = 2, dropout: float = 0.1):
    import torch
    import torch.nn as nn

    class Res(nn.Module):
        def __init__(self, dil):
            super().__init__()
            self.conv = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
            self.out = nn.Conv1d(d, d, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop(self.out(torch.relu(self.conv(x))))

    def tower(c_in):
        return nn.Sequential(nn.Conv1d(c_in, d, 1), *[Res(2 ** i) for i in range(layers)])

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


def boundary_target(sy: np.ndarray, sigma: float = BND_SIGMA) -> np.ndarray:
    """Soft target per step: exp(-d^2 / 2 sigma^2), d = steps to the nearest H <-> A change (a change
    straight between two alive steps; restarts after a dead ball are the alive layer's job)."""
    y = np.asarray(sy)
    ch = np.flatnonzero((y[1:] >= 0) & (y[:-1] >= 0) & (y[1:] != y[:-1])) + 1      # first step of the new team
    out = np.zeros(len(y), dtype=np.float32)
    if len(ch) == 0:
        return out
    t = np.arange(len(y), dtype=np.float64)
    mid = ch - 0.5                                                                # the change lies between two steps
    k = np.searchsorted(mid, t)
    d = np.minimum(np.abs(t - mid[np.clip(k, 0, len(mid) - 1)]), np.abs(t - mid[np.clip(k - 1, 0, len(mid) - 1)]))
    d = np.maximum(d - 0.5, 0.0)                                                  # both neighbours of the change get 1
    return np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)


# ------------------------------------------------------------------------------------ model

class OwnerModel:
    def __init__(self, kind: str, columns: list[str], feature_set: str, rule_cfg: dict, meta: dict,
                 obj=None, state=None, norm=None, arch=None):
        self.kind, self.columns, self.feature_set = kind, list(columns), feature_set
        self.rule_cfg = rules.RuleConfig(**rule_cfg) if isinstance(rule_cfg, dict) else (rule_cfg or rules.RuleConfig())
        self.meta, self.obj, self.state, self.norm, self.arch = meta, obj, state, norm, arch or {}
        self._net = None

    # -------------------------------------------------------------- persistence
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.kind == "gbm":
            import joblib
            joblib.dump({"kind": "gbm", "model": self.obj, "columns": self.columns, "feature_set": self.feature_set,
                         "rule_cfg": asdict(self.rule_cfg), "meta": self.meta}, path)
        else:
            import torch
            torch.save({"kind": self.kind, "state": self.state, "columns": self.columns, "feature_set": self.feature_set,
                        "rule_cfg": asdict(self.rule_cfg), "meta": self.meta,
                        "norm": {k: np.asarray(v) for k, v in self.norm.items()}, "arch": self.arch}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "OwnerModel":
        path = Path(path)
        if path.suffix == ".joblib":
            import joblib
            o = joblib.load(path)
            if "kind" not in o:                         # a who.py model (who_v1): same contract
                return cls("gbm", o["columns"], "who", {}, o.get("meta", {}), obj=o["model"])
            return cls("gbm", o["columns"], o["feature_set"], o["rule_cfg"], o["meta"], obj=o["model"])
        import torch
        o = torch.load(path, map_location="cpu", weights_only=False)
        return cls(o["kind"], o["columns"], o["feature_set"], o["rule_cfg"], o["meta"], state=o["state"],
                   norm=o["norm"], arch=o["arch"])

    # -------------------------------------------------------------- inference
    def _torch_net(self, device):
        if self._net is None:
            import torch
            n_in = int(self.arch["n_in"])
            if self.kind == "mlp":
                net = _mlp(n_in, self.arch.get("hidden", 256), self.arch.get("depth", 3))
            elif self.kind == "tcn":
                net = _tcn_net(n_in, self.arch.get("d", 64), self.arch.get("layers", 10), self.arch.get("stages", 2))
            else:
                net = _gru_net(n_in, self.arch.get("d", 64), self.arch.get("layers", 2))
            net.load_state_dict({k: torch.as_tensor(v) for k, v in self.state.items()})
            net.eval()
            self._net = net.to(device)
        return self._net

    def _p_rows(self, X: np.ndarray, device="cpu", batch: int = 65536) -> np.ndarray:
        """P(home) for rows of the feature table (gbm / mlp), columns already selected."""
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

    def _logit_seq(self, S: np.ndarray, device="cpu") -> np.ndarray:
        """Per-step logits of one half's 5 Hz sequence (gru), overlapping windows averaged."""
        import torch
        dev = _device(device)
        net = self._torch_net(dev)
        Z = _prep(S, self.norm["mu"], self.norm["sd"], self.norm["mask_cols"])
        T = len(Z)
        win = min(GRU_WIN, T)
        starts = list(range(0, max(1, T - win + 1), GRU_STRIDE_PRED))
        if starts[-1] + win < T:
            starts.append(T - win)
        acc = np.zeros(T)
        wsum = np.zeros(T)
        tri = np.minimum(np.arange(1, win + 1), np.arange(win, 0, -1)).astype(np.float64)
        with torch.no_grad():
            for s in starts:
                lg = net(torch.as_tensor(Z[None, s:s + win], device=dev)).squeeze(0).cpu().numpy()
                acc[s:s + win] += tri * lg
                wsum[s:s + win] += tri
        return acc / np.maximum(wsum, 1e-9)

    def _tcn_seq(self, S: np.ndarray, device="cpu"):
        """(home logit of the last stage, boundary logit) per step of one half, in one pass."""
        import torch
        dev = _device(device)
        net = self._torch_net(dev)
        Z = _prep(S, self.norm["mu"], self.norm["sd"], self.norm["mask_cols"])
        with torch.no_grad():
            outs, bnd = net(torch.as_tensor(Z[None], device=dev))
        return outs[-1].squeeze(0).cpu().numpy().astype(np.float64), bnd.squeeze(0).cpu().numpy().astype(np.float64)

    def p_home_and_boundary(self, views: list[dict], halves, n: int, fps: float, device="cpu"):
        """views[k] = {"X": DataFrame (n, F) full_rules table, "S": (n, Fs) seq inputs, "seq_columns"}
        for team order k (0 as delivered, 1 swapped). Returns (n,) P(home) and, for a tcn, (n,)
        P(boundary) (mean of the two orders; None for the other kinds); NaN outside the halves."""
        inside = np.zeros(n, dtype=bool)
        for _, a, b in halves:
            inside[a:b + 1] = True
        p = np.full(n, np.nan)
        if not inside.any():
            return p, None
        ps, bs = [], []
        for v in views:
            if self.kind in ("gbm", "mlp"):
                ps.append(self._p_rows(v["X"].loc[inside, self.columns].to_numpy(dtype=np.float32), device))
                continue
            pos = [v["seq_columns"].index(c) for c in self.columns]
            from .data import step_of
            k = step_of(fps)
            out, bout = np.full(n, np.nan), np.full(n, np.nan)
            for _, a, b in halves:
                idx = np.arange(a, b + 1, k)
                fr = np.arange(a, b + 1)
                if self.kind == "tcn":
                    lg, lb = self._tcn_seq(v["S"][idx][:, pos], device)
                    bout[a:b + 1] = 1.0 / (1.0 + np.exp(-np.interp(fr, idx, lb)))
                else:
                    lg = self._logit_seq(v["S"][idx][:, pos], device)
                out[a:b + 1] = 1.0 / (1.0 + np.exp(-np.interp(fr, idx, lg)))
            ps.append(out[inside])
            bs.append(bout[inside])
        p[inside] = 0.5 * (ps[0] + 1.0 - ps[1])
        pb = None
        if self.kind == "tcn":
            pb = np.full(n, np.nan)
            pb[inside] = 0.5 * (bs[0] + bs[1])
            pb = np.round(pb, 4)
        return np.round(p, 4), pb

    def p_home(self, views: list[dict], halves, n: int, fps: float, device="cpu") -> np.ndarray:
        return self.p_home_and_boundary(views, halves, n, fps, device)[0]


# ------------------------------------------------------------------------------------ training

def train_gbm(X, y, columns, feature_set, rule_cfg, meta, iters=300, seed=0, log=print) -> OwnerModel:
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    model = HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                           l2_regularization=1.0, early_stopping=False, random_state=seed)
    model.fit(X, y)
    log(f"  gbm fit {len(X):,} rows x {X.shape[1]} in {time.time() - t0:.0f} s")
    return OwnerModel("gbm", columns, feature_set, rule_cfg, meta, obj=model)


def _state_cpu(net):
    return {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}


def train_mlp(X, y, columns, feature_set, rule_cfg, meta, epochs=4, batch=4096, lr=1e-3, seed=0, device="auto",
              hidden=256, depth=3, log=print) -> OwnerModel:
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
            xb = torch.as_tensor(_prep(X[ib], mu, sd, mask_cols), device=dev)       # prepared per batch
            yb = torch.as_tensor(yf[ib], device=dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(ib)
        log(f"  mlp epoch {ep + 1}/{epochs} loss {tot / len(X):.4f} ({time.time() - t0:.0f} s)")
    net.eval()
    return OwnerModel("mlp", columns, feature_set, rule_cfg, meta, state=_state_cpu(net),
                      norm={"mu": mu, "sd": sd, "mask_cols": mask_cols},
                      arch={"n_in": n_in, "hidden": hidden, "depth": depth})


def gru_windows(tables, columns: list[str]):
    """[(S block (T, F), labels (T,))] per game x view x half, the raw material of the windows."""
    out = []
    for _, _, t in tables:
        pos = [t["seq_columns"].index(c) for c in columns]
        S, sy, sv, sh = t["S"][:, pos], t["sy"], t["sview"], t["shalf"]
        key = sv.astype(np.int64) * 16 + sh
        s, e = 0, len(key)
        change = np.flatnonzero(key[1:] != key[:-1]) + 1
        for a, b in zip(np.r_[s, change], np.r_[change, e]):
            out.append((S[a:b], sy[a:b]))
    return out


def train_gru(tables, columns, feature_set, rule_cfg, meta, epochs=8, batch=64, lr=2e-3, seed=0, device="auto",
              d=64, layers=2, log=print) -> OwnerModel:
    import torch
    _seed_all(seed)
    dev = _device(device)
    blocks = gru_windows(tables, columns)
    F = blocks[0][0].shape[1]
    s1, s2, cnt = np.zeros(F), np.zeros(F), np.zeros(F)             # streaming mean / std
    mask_cols = np.zeros(F, dtype=bool)
    for S, _ in blocks:
        S = S.astype(np.float64)
        ok = ~np.isnan(S)
        mask_cols |= ~ok.all(axis=0)
        S = np.where(ok, S, 0.0)
        s1 += S.sum(axis=0)
        s2 += (S * S).sum(axis=0)
        cnt += ok.sum(axis=0)
    mu = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    var = np.where(cnt > 0, s2 / np.maximum(cnt, 1) - mu ** 2, 1.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    mu, sd = mu.astype(np.float32), np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
    # one standardised copy of every block; windows are (block, start) pairs gathered per batch
    Zs, Ys, wins = [], [], []
    for S, sy in blocks:
        if len(S) < 16:
            continue
        Zs.append(_prep(S, mu, sd, mask_cols).astype(np.float16))    # half the memory; restored per batch
        Ys.append(sy)
        T = len(S)
        win = min(GRU_WIN, T)
        for s in list(range(0, T - win + 1, GRU_STRIDE_TRAIN)) or [0]:
            if (sy[s:s + win] >= 0).sum() >= 10:
                wins.append((len(Zs) - 1, s, win))
    del blocks
    if not wins:
        raise SystemExit("gru: no training windows")
    W = max(w[2] for w in wins)
    n_in = Zs[0].shape[1]
    log(f"  gru {len(wins):,} windows of <= {W} steps x {n_in} inputs ({sum(z.nbytes for z in Zs) / 1e9:.2f} GB)")

    def batch_of(ids):
        xb = np.zeros((len(ids), W, n_in), dtype=np.float32)
        yb = np.full((len(ids), W), -1.0, dtype=np.float32)
        for r, i in enumerate(ids):
            k, s, w = wins[i]
            xb[r, :w] = Zs[k][s:s + w]
            yb[r, :w] = Ys[k][s:s + w]
        return torch.as_tensor(xb), torch.as_tensor(yb)

    net = _gru_net(n_in, d, layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(wins) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps))
    rng = np.random.default_rng(seed)
    t0 = time.time()
    net.train()
    for ep in range(epochs):
        order = rng.permutation(len(wins))
        tot, cnt = 0.0, 0
        for a in range(0, len(wins), batch):
            xb, yb = batch_of(order[a:a + batch])
            xb, yb = xb.to(dev), yb.to(dev)
            m = yb >= 0
            opt.zero_grad(set_to_none=True)
            lg = net(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lg[m], yb[m])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * int(m.sum())
            cnt += int(m.sum())
        log(f"  gru epoch {ep + 1}/{epochs} loss {tot / max(cnt, 1):.4f} ({time.time() - t0:.0f} s)")
    net.eval()
    return OwnerModel("gru", columns, feature_set, rule_cfg, meta, state=_state_cpu(net),
                      norm={"mu": mu, "sd": sd, "mask_cols": mask_cols}, arch={"n_in": int(n_in), "d": d, "layers": layers})


def train_tcn(tables, columns, feature_set, rule_cfg, meta, epochs=20, batch=8, lr=1e-3, seed=0, device="auto",
              d=64, layers=10, stages=2, w_bnd=1.0, log=print) -> OwnerModel:
    """Windows of TCN_WIN steps (stride TCN_STRIDE) inside each game x view x half; BCE on alive steps
    for every class stage + w_bnd x BCE on the soft boundary target (positive-weighted)."""
    import torch
    import torch.nn.functional as F
    _seed_all(seed)
    dev = _device(device)
    blocks = gru_windows(tables, columns)
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
    for S, sy in blocks:
        if len(S) < 32:
            continue
        Zs.append(_prep(S, mu, sd, mask_cols).astype(np.float16))
        Ys.append(sy)
        Bs.append(boundary_target(sy))
        T = len(S)
        win = min(TCN_WIN, T)
        for st in list(range(0, T - win + 1, TCN_STRIDE)) or [0]:
            if (sy[st:st + win] >= 0).sum() >= 20:
                wins.append((len(Zs) - 1, st, win))
        if T > win and (T - win) % TCN_STRIDE:
            wins.append((len(Zs) - 1, T - win, win))                     # the half's tail
    del blocks
    if not wins:
        raise SystemExit("tcn: no training windows")
    W = max(w[2] for w in wins)
    n_in = Zs[0].shape[1]
    pos_share = float(np.mean([b[y >= 0].mean() for b, y in zip(Bs, Ys) if (y >= 0).any()]))
    pos_weight = float(np.clip((1 - pos_share) / max(pos_share, 1e-4), 1.0, 50.0))
    log(f"  tcn {len(wins):,} windows of <= {W} steps x {n_in} inputs, boundary share {pos_share:.3f} "
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
        cnt = 0
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
            cnt += n_m
        log(f"  tcn epoch {ep + 1}/{epochs} class {tot_c / max(cnt, 1):.4f} boundary {tot_b / max(cnt, 1):.4f} "
            f"({time.time() - t0:.0f} s)")
    net.eval()
    return OwnerModel("tcn", columns, feature_set, rule_cfg, meta, state=_state_cpu(net),
                      norm={"mu": mu, "sd": sd, "mask_cols": mask_cols},
                      arch={"n_in": int(n_in), "d": d, "layers": layers, "stages": stages})
