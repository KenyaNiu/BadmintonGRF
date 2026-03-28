from __future__ import annotations

import torch
import torch.nn as nn


def contact_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    ev_idx: torch.Tensor,
    alpha: float = 10.0,
    half_win: int = 25,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE with ``alpha`` weight on ±half_win frames around contact index.

    ``valid_mask``: optional ``(B, T)`` float/bool, 1 for real frames and 0 for
    right-padding (variable-length segments in a batch). Padded positions get
    zero weight in the numerator and denominator.
    """
    B, T, D = pred.shape
    w = torch.ones(B, T, 1, device=pred.device, dtype=pred.dtype)
    for b in range(B):
        L_b = T
        if valid_mask is not None:
            L_b = int(valid_mask[b].sum().item())
            L_b = max(L_b, 1)
        ev = int(ev_idx[b].item())
        ev = min(ev, L_b - 1)
        s = max(0, ev - half_win)
        e = min(L_b, ev + half_win)
        w[b, s:e] = alpha
    if valid_mask is not None:
        vm = valid_mask.to(device=pred.device, dtype=pred.dtype)
        w = w * vm.unsqueeze(-1)
    denom = (w.sum() * D).clamp_min(1e-8)
    return (w * (pred - target).pow(2)).sum() / denom
