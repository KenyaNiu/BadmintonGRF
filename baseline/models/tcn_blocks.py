"""Shared temporal convolution blocks for sequence baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """Temporal conv residual block with GroupNorm (batch-size independent)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) // 2
        n_groups = min(8, out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.gn1 = nn.GroupNorm(n_groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.gn2 = nn.GroupNorm(n_groups, out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.gn1(self.conv1(x))))
        h = self.drop(F.gelu(self.gn2(self.conv2(h))))
        return h + self.skip(x)
