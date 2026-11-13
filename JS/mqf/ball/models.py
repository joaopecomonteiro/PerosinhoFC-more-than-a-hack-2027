"""Two bidirectional transformers for the ball (torch). Same input layout (see data.py), same output:
the ball's (x, y) per step in pitch-normalised coordinates [-1, 1].

A  SetTemporal -- per step, the visible players are a SET (no identities): a [CLS] token plus the
   players go through set-attention layers; the CLS is the step's summary. Ball token (or a learned
   mask token where hidden) + summary + time encoding -> bidirectional transformer over time.
   Kim et al. (KDD 2023) with a transformer instead of their Bi-LSTM.
B  Axial -- TranSPORTmer-style (Capellera et al., ACCV 2024): the ball and every player / piece are
   agents; blocks alternate attention over time (per agent, non-causal) and over agents (per step,
   permutation-equivariant). Missing tokens are a learned mask token; absent players are ignored by
   the attention over agents.

v3 (cfg["ctx"]): the ball token also reads the gap context (data.ctx: anchor exists, both edges,
time since / until the nearest seen ball) and the interpolated anchor, and the head predicts a
CORRECTION to the anchor: output = anchor + head(...), added in fp32 outside bf16. v1 showed an
absolute-position head ignores the ball around the gap (val ADE flat in the gap length); with the
anchor a zero head is already linear interpolation. The sum is taken in the anchor's dtype (fp32 in
training; fp64 for the reproducible CPU inference of the submission).

v4: ctx has 8 features (+ ball velocity at the gap edges; cfg["ctx_dim"], v3 checkpoints = 4), and
the axial model can carry an auxiliary head (cfg["aux"]): per step, logits over the agents for
"who is nearest to the ball" (cross-entropy in train.py). It shapes the ball token towards the
player-ball relation possession needs, and its softmax is a team probability for later use.
"""
from __future__ import annotations


import torch
from torch import nn


def sinusoid(T: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


def mlp(i: int, d: int, o: int | None = None) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o or d))


def _layer(d: int, heads: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)


def _team_index(team: torch.Tensor) -> torch.Tensor:
    return torch.where(team < 0, torch.full_like(team, 2), team)          # 0 home, 1 away, 2 padding


class SetTemporal(nn.Module):
    def __init__(self, d=128, heads=4, set_layers=2, time_layers=4, dropout=0.1, ctx=False, ctx_dim=4):
        super().__init__()
        self.ctx, self.ctx_dim, self.aux = ctx, ctx_dim, False
        if ctx:
            self.ctx_in = mlp(2 + ctx_dim, d)
        self.agent_in = mlp(4, d)
        self.team_emb = nn.Embedding(3, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.set_enc = nn.TransformerEncoder(_layer(d, heads, dropout), set_layers, enable_nested_tensor=False)
        self.ball_in = mlp(2, d)
        self.ball_mask = nn.Parameter(torch.zeros(d))
        self.time_enc = nn.TransformerEncoder(_layer(d, heads, dropout), time_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = mlp(d, d, 2)
        self.d = d

    def forward(self, agents, present, team, ball_in, ball_obs, anchor=None, ctx=None, return_logits=False):
        B, T, A, _ = agents.shape
        d = self.d
        tok = self.agent_in(agents) + self.team_emb(_team_index(team))[:, None]
        tok = tok.reshape(B * T, A, d)
        x = torch.cat([self.cls.expand(B * T, 1, d), tok], dim=1)
        pad = torch.cat([torch.zeros(B * T, 1, dtype=torch.bool, device=agents.device),
                         ~present.reshape(B * T, A)], dim=1)
        frame = self.set_enc(x, src_key_padding_mask=pad)[:, 0].reshape(B, T, d)
        ball = torch.where(ball_obs[..., None], self.ball_in(ball_in), self.ball_mask)
        if self.ctx:
            ball = ball + self.ctx_in(torch.cat([anchor, ctx[..., :self.ctx_dim]], dim=-1))
        h = self.time_enc(frame + ball + sinusoid(T, d, agents.device))
        out = self.head(self.norm(h))
        return anchor + out.to(anchor.dtype) if self.ctx else out


class AxialBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.time = _layer(d, heads, dropout)
        self.social = _layer(d, heads, dropout)

    def forward(self, x, social_pad):
        B, A1, T, d = x.shape
        x = self.time(x.reshape(B * A1, T, d)).reshape(B, A1, T, d)
        y = x.permute(0, 2, 1, 3).reshape(B * T, A1, d)
        y = self.social(y, src_key_padding_mask=social_pad.reshape(B * T, A1))
        return y.reshape(B, T, A1, d).permute(0, 2, 1, 3)


class Axial(nn.Module):
    def __init__(self, d=128, heads=4, blocks=3, dropout=0.1, ctx=False, ctx_dim=4, aux=False):
        super().__init__()
        self.ctx, self.ctx_dim, self.aux = ctx, ctx_dim, aux
        if ctx:
            self.ctx_in = mlp(2 + ctx_dim, d)
        if aux:
            self.cls_head = mlp(d, d, 1)
        self.agent_in = mlp(4, d)
        self.ball_in = mlp(2, d)
        self.type_emb = nn.Embedding(4, d)                                 # home, away, padding, ball
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([AxialBlock(d, heads, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.head = mlp(d, d, 2)
        self.d = d

    def forward(self, agents, present, team, ball_in, ball_obs, anchor=None, ctx=None, return_logits=False):
        B, T, A, _ = agents.shape
        p = torch.where(present[..., None], self.agent_in(agents), self.mask_tok)
        p = p + self.type_emb(_team_index(team))[:, None]
        b = torch.where(ball_obs[..., None], self.ball_in(ball_in), self.mask_tok) + self.type_emb.weight[3]
        if self.ctx:
            b = b + self.ctx_in(torch.cat([anchor, ctx[..., :self.ctx_dim]], dim=-1))
        x = torch.cat([b[:, :, None], p], dim=2) + sinusoid(T, self.d, agents.device)[None, :, None]
        x = x.permute(0, 2, 1, 3)                                          # (B, A+1, T, d)
        social_pad = torch.cat([torch.zeros(B, T, 1, dtype=torch.bool, device=agents.device), ~present], dim=2)
        for blk in self.blocks:
            x = blk(x, social_pad)
        want = return_logits and self.aux
        h = self.norm(x) if want else self.norm(x[:, :1])   # LayerNorm is per token: same ball token
        out = self.head(h[:, 0])
        pred = anchor + out.to(anchor.dtype) if self.ctx else out
        if not return_logits:
            return pred
        logits = None
        if want:                                            # (B, T, A), absent agents masked
            logits = self.cls_head(h[:, 1:]).squeeze(-1).transpose(1, 2).float()
            logits = logits.masked_fill(~present, -1e9)
        return pred, logits


def build(cfg: dict) -> nn.Module:
    ctx, ctx_dim = bool(cfg.get("ctx", False)), int(cfg.get("ctx_dim", 4))
    if cfg["arch"] == "A":
        return SetTemporal(cfg["d"], cfg["heads"], cfg.get("set_layers", 2), cfg["layers"], cfg.get("dropout", 0.1),
                           ctx, ctx_dim)
    if cfg["arch"] == "B":
        return Axial(cfg["d"], cfg["heads"], cfg["layers"], cfg.get("dropout", 0.1), ctx, ctx_dim,
                     bool(cfg.get("aux", False)))
    raise ValueError(cfg["arch"])


def zero_correction(model: nn.Module) -> nn.Module:
    """Zero the last layer of the head: a v3 model then outputs its anchor (tests; and the init
    for training -- start AS linear interpolation, learn the correction)."""
    last = model.head[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return model


def forward(model: nn.Module, b: dict, with_logits: bool = False):
    """Call any version with a batch dict of tensors. with_logits: (pred, nearest-agent logits or None)."""
    args = (b["agents"], b["present"], b["team"], b["ball_in"], b["ball_obs"])
    kw = {"anchor": b["anchor"], "ctx": b["ctx"]} if getattr(model, "ctx", False) else {}
    if with_logits:
        out = model(*args, return_logits=True, **kw)
        return out if isinstance(out, tuple) else (out, None)
    return model(*args, **kw)


def nearest_loss(logits, nearest, w):
    """Cross-entropy of the nearest-agent head on steps with a target, weighted by w (B, T)."""
    valid = nearest >= 0
    ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), nearest.clamp(min=0).reshape(-1),
                                     reduction="none").reshape(nearest.shape)
    wv = w * valid
    return (ce * wv).sum() / wv.sum().clamp(min=1.0)


def to_metres(pred: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) normalised -> metres with each sample's pitch (B, 2)."""
    return (pred + 1) / 2 * pitch[:, None, :]


def loss_and_ade(pred_m, target, w_masked, w_seen, delta: float = 1.0, seen_weight: float = 0.1):
    """Huber per coordinate on hidden alive frames (+ seen_weight x on visible ones); ADE on hidden."""
    tgt = torch.nan_to_num(target)
    err = nn.functional.smooth_l1_loss(pred_m, tgt, reduction="none", beta=delta).sum(-1)
    dist = torch.linalg.norm(pred_m - tgt, dim=-1)
    nm, ns = w_masked.sum().clamp(min=1.0), w_seen.sum().clamp(min=1.0)
    loss = (err * w_masked).sum() / nm + seen_weight * (err * w_seen).sum() / ns
    ade = (dist * w_masked).sum() / nm
    return loss, ade


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["SetTemporal", "Axial", "build", "forward", "nearest_loss", "zero_correction", "to_metres",
           "loss_and_ade", "n_params"]
