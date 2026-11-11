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
    def __init__(self, d=128, heads=4, set_layers=2, time_layers=4, dropout=0.1):
        super().__init__()
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

    def forward(self, agents, present, team, ball_in, ball_obs):
        B, T, A, _ = agents.shape
        d = self.d
        tok = self.agent_in(agents) + self.team_emb(_team_index(team))[:, None]
        tok = tok.reshape(B * T, A, d)
        x = torch.cat([self.cls.expand(B * T, 1, d), tok], dim=1)
        pad = torch.cat([torch.zeros(B * T, 1, dtype=torch.bool, device=agents.device),
                         ~present.reshape(B * T, A)], dim=1)
        frame = self.set_enc(x, src_key_padding_mask=pad)[:, 0].reshape(B, T, d)
        ball = torch.where(ball_obs[..., None], self.ball_in(ball_in), self.ball_mask)
        h = self.time_enc(frame + ball + sinusoid(T, d, agents.device))
        return self.head(self.norm(h))


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
    def __init__(self, d=128, heads=4, blocks=3, dropout=0.1):
        super().__init__()
        self.agent_in = mlp(4, d)
        self.ball_in = mlp(2, d)
        self.type_emb = nn.Embedding(4, d)                                 # home, away, padding, ball
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([AxialBlock(d, heads, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.head = mlp(d, d, 2)
        self.d = d

    def forward(self, agents, present, team, ball_in, ball_obs):
        B, T, A, _ = agents.shape
        p = torch.where(present[..., None], self.agent_in(agents), self.mask_tok)
        p = p + self.type_emb(_team_index(team))[:, None]
        b = torch.where(ball_obs[..., None], self.ball_in(ball_in), self.mask_tok) + self.type_emb.weight[3]
        x = torch.cat([b[:, :, None], p], dim=2) + sinusoid(T, self.d, agents.device)[None, :, None]
        x = x.permute(0, 2, 1, 3)                                          # (B, A+1, T, d)
        social_pad = torch.cat([torch.zeros(B, T, 1, dtype=torch.bool, device=agents.device), ~present], dim=2)
        for blk in self.blocks:
            x = blk(x, social_pad)
        return self.head(self.norm(x[:, 0]))


def build(cfg: dict) -> nn.Module:
    if cfg["arch"] == "A":
        return SetTemporal(cfg["d"], cfg["heads"], cfg.get("set_layers", 2), cfg["layers"], cfg.get("dropout", 0.1))
    if cfg["arch"] == "B":
        return Axial(cfg["d"], cfg["heads"], cfg["layers"], cfg.get("dropout", 0.1))
    raise ValueError(cfg["arch"])


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


__all__ = ["SetTemporal", "Axial", "build", "to_metres", "loss_and_ade", "n_params"]
