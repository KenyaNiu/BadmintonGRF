"""
Flat-sequence Transformer (no skeleton graph).
Standard encoder over per-frame feature vectors — common strong baseline for time series.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def _pe_length(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sinusoidal PE for arbitrary T (late fusion / inference may exceed training max_len)."""
        pe = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        pos = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: self.d_model // 2])
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        if t <= self.pe.size(1):
            pe = self.pe[:, :t]
        else:
            pe = self._pe_length(t, x.device, x.dtype)
        return self.drop(x + pe)


class SeqTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 119,
        output_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        dim_ff: int | None = None,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must divide num_heads"
        dim_ff = dim_ff or hidden_dim * 2
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pe = SinusoidalPE(hidden_dim, max_len=256, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.pe(h)
        h = self.enc(h)
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
