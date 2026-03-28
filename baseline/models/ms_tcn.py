"""
Multi-scale temporal CNN for sequence regression.

Parallel dilated conv branches (receptive fields 1,2,4,8,…) + fusion;
strong complementary baseline to single-rate TCN + LSTM.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleTCNGRF(nn.Module):
    def __init__(
        self,
        input_dim: int = 119,
        output_dim: int = 1,
        hidden_dim: int = 128,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        pad = kernel_size // 2
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=pad * d, dilation=d),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad * d, dilation=d),
                    nn.GELU(),
                )
            )
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * len(dilations), hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, output_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> conv1d expects (B, C, T)
        x1 = x.transpose(1, 2)
        outs = [b(x1) for b in self.branches]
        h = torch.cat(outs, dim=1)
        y = self.fuse(h)
        return y.transpose(1, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
