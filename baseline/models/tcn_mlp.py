"""Pure temporal convolution stack + per-frame MLP (no RNN / no attention)."""

from __future__ import annotations

import torch
import torch.nn as nn

from baseline.models.tcn_blocks import TCNBlock


class TCNMLPBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 119,
        output_dim: int = 1,
        tcn_channels: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
        blocks = []
        c_in, c_out = input_dim, tcn_channels
        for i in range(num_blocks):
            ks = 7 if i == 0 else 5
            blocks.append(TCNBlock(c_in, c_out, kernel_size=ks, dropout=dropout * 0.5))
            c_in = c_out
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(tcn_channels),
            nn.Linear(tcn_channels, tcn_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_channels // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B,T,c_out)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
