"""Exponential moving average of model weights (common for better generalization / metrics)."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in self.shadow:
                continue
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def apply_to(self, model: nn.Module) -> None:
        self.backup.clear()
        for n, p in model.named_parameters():
            if n not in self.shadow:
                continue
            self.backup[n] = p.detach().clone()
            p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup.clear()

    def load_shadow(self, state: Dict[str, Any]) -> None:
        for n in self.shadow:
            if n in state:
                self.shadow[n].copy_(state[n].to(self.shadow[n].device))
