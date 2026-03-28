"""
PatchTST-style encoder for flat sequences (ICLR 2023).

Patches along time; each patch flattens [patch_len * input_dim] → d_model;
Transformer over patch tokens; upsample predictions to per-frame outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from baseline.models.transformer_seq import SinusoidalPE


class PatchTSTGRF(nn.Module):
    def __init__(
        self,
        input_dim: int = 119,
        output_dim: int = 1,
        patch_len: int = 16,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        dim_ff: int | None = None,
        max_patch_positions: int = 2048,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.patch_len = patch_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        dim_ff = dim_ff or d_model * 2
        patch_dim = patch_len * input_dim
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.patch_pe = SinusoidalPE(d_model, max_len=max_patch_positions, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.patch_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_len * output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        pl = self.patch_len
        pad = (pl - T % pl) % pl
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        Tp = x.size(1)
        n_p = Tp // pl
        patches = x[:, : n_p * pl, :].reshape(B, n_p, pl * F)
        h = self.patch_pe(self.patch_proj(patches))
        h = self.enc(h)
        patch_out = self.patch_head(h)  # (B, n_p, pl * out)
        patch_out = patch_out.view(B, n_p * pl, self.output_dim)
        return patch_out[:, :T, :]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
