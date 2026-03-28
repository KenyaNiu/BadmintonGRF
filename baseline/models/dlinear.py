"""
DLinear-style decomposition baseline (AAAI 2023).

Trend = temporal moving average along time; seasonal = residual.
Dual linear maps (frame-wise) to the target — supports variable sequence length T.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DLinearGRF(nn.Module):
    def __init__(
        self,
        input_dim: int = 119,
        output_dim: int = 1,
        kernel_size: int = 25,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert kernel_size >= 3 and kernel_size % 2 == 1, "kernel_size should be odd"
        self.kernel_size = kernel_size
        self.fc_trend = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )
        self.fc_season = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        pad = self.kernel_size // 2
        trend = torch.nn.functional.avg_pool1d(x, kernel_size=self.kernel_size, stride=1, padding=pad)
        seasonal = x - trend
        pred_t = self.fc_trend(trend.transpose(1, 2))
        pred_s = self.fc_season(seasonal.transpose(1, 2))
        return pred_t + pred_s

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
