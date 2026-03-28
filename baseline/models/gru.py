"""TCN + BiGRU baseline (flat pose features)."""

from __future__ import annotations

import torch
import torch.nn as nn

from baseline.models.tcn_blocks import TCNBlock


class GRUBaseline(nn.Module):
    """Temporal CNN + bidirectional GRU + frame-wise regression."""

    def __init__(
        self,
        input_dim: int = 119,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        tcn_channels: int = 64,
    ):
        super().__init__()
        d = 2 if bidirectional else 1
        self.tcn = nn.Sequential(
            TCNBlock(input_dim, tcn_channels, kernel_size=7, dropout=dropout * 0.5),
            TCNBlock(tcn_channels, tcn_channels, kernel_size=5, dropout=dropout * 0.5),
        )
        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
