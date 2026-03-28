"""
TSMixer-style baseline for per-frame regression.

Based on the "MLP-Mixer for Time Series" family: alternating token-mixing (time)
and channel-mixing (feature) MLP blocks. These models are simple, fast, and
often strong on multivariate time series.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TSMixerGRF(nn.Module):
    """
    Input:  (B, T, F)
    Output: (B, T, out)

    Notes:
      - token mixing mixes along time for each feature independently (via transpose)
      - channel mixing mixes along features for each time step
    """

    def __init__(
        self,
        input_dim: int = 119,
        output_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.15,
        token_mlp_ratio: float = 0.5,
        channel_mlp_ratio: float = 2.0,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.max_len = int(max_len)

        self.in_proj = nn.Linear(self.input_dim, hidden_dim)
        self.blocks = nn.ModuleList()

        token_hidden = max(16, int(self.max_len * float(token_mlp_ratio)))
        channel_hidden = max(16, int(hidden_dim * float(channel_mlp_ratio)))

        for _ in range(int(num_layers)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm_t": nn.LayerNorm(hidden_dim),
                        "token_mlp": _MLP(self.max_len, token_hidden, dropout),
                        "norm_c": nn.LayerNorm(hidden_dim),
                        "channel_mlp": _MLP(hidden_dim, channel_hidden, dropout),
                    }
                )
            )

        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad or crop time axis to max_len for token mixing.
        b, t, _f = x.shape
        if t < self.max_len:
            pad = self.max_len - t
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        elif t > self.max_len:
            x = x[:, : self.max_len, :]

        h = self.in_proj(x)  # (B, L, D)
        for blk in self.blocks:
            # token mixing: (B, L, D) -> (B, D, L) apply MLP over L -> back
            y = blk["norm_t"](h)
            y = y.transpose(1, 2)  # (B, D, L)
            y = blk["token_mlp"](y)  # mixes along L
            y = y.transpose(1, 2)
            h = h + y

            # channel mixing: per token (time step)
            y = blk["norm_c"](h)
            y = blk["channel_mlp"](y)
            h = h + y

        y = self.out_head(h)  # (B, L, out)
        return y[:, :t, :]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

