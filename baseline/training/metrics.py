from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _concat_fz(preds: List[np.ndarray], targets: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    p = np.concatenate([np.asarray(x, dtype=np.float64) for x in preds], axis=0)
    t = np.concatenate([np.asarray(x, dtype=np.float64) for x in targets], axis=0)
    if p.ndim == 3:
        p = p.reshape(-1, p.shape[-1])
        t = t.reshape(-1, t.shape[-1])
    return p[:, -1], t[:, -1]


def fit_fz_linear_calibration(
    all_preds: Sequence[np.ndarray], all_targets: Sequence[np.ndarray]
) -> Tuple[float, float]:
    """
    Fit ``t_fz ≈ a * p_fz + b`` on pooled validation frames (no test labels).
    Standard post-hoc scale/shift correction for regression under domain shift.
    """
    p_fz, t_fz = _concat_fz(list(all_preds), list(all_targets))
    if p_fz.size < 2 or not np.all(np.isfinite(p_fz)) or not np.all(np.isfinite(t_fz)):
        return 1.0, 0.0
    X = np.column_stack([p_fz, np.ones(len(p_fz), dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(X, t_fz, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_fz_linear_calibration(all_preds: List[np.ndarray], a: float, b: float) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for p in all_preds:
        q = np.asarray(p, dtype=np.float32).copy()
        if q.ndim == 1:
            q = a * q + b
        else:
            q[:, -1] = a * q[:, -1] + b
        out.append(q)
    return out


def r2_fz_macro(all_preds: List[np.ndarray], all_targets: List[np.ndarray]) -> float:
    """Mean of per-segment R² on Fz (often higher than pooled global R² when variance is heterogeneous)."""
    r2s: List[float] = []
    for p, t in zip(all_preds, all_targets):
        p = np.asarray(p, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        p_fz = p[:, -1] if p.ndim > 1 else p
        t_fz = t[:, -1] if t.ndim > 1 else t
        ss_res = float(np.sum((t_fz - p_fz) ** 2))
        ss_tot = float(np.sum((t_fz - t_fz.mean()) ** 2))
        if ss_tot < 1e-18:
            continue
        r2s.append(1.0 - ss_res / ss_tot)
    return float(np.mean(r2s)) if r2s else float("nan")


def pearson_r_fz(all_preds: List[np.ndarray], all_targets: List[np.ndarray]) -> float:
    p_fz, t_fz = _concat_fz(all_preds, all_targets)
    if p_fz.size < 2:
        return float("nan")
    r = np.corrcoef(p_fz, t_fz)[0, 1]
    return float(r) if np.isfinite(r) else float("nan")


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    if isinstance(preds, list) and preds and isinstance(preds[0], np.ndarray):
        p = np.concatenate([np.asarray(x, dtype=np.float64) for x in preds], axis=0)
        t = np.concatenate([np.asarray(x, dtype=np.float64) for x in targets], axis=0)
    else:
        p = np.asarray(preds, dtype=np.float64)
        t = np.asarray(targets, dtype=np.float64)
    if p.ndim == 3:
        p = p.reshape(-1, p.shape[-1])
        t = t.reshape(-1, t.shape[-1])
    p_fz = p[:, -1]
    t_fz = t[:, -1]
    ss_res = float(np.sum((t_fz - p_fz) ** 2))
    ss_tot = float(np.sum((t_fz - t_fz.mean()) ** 2))
    return {
        "rmse_fz": float(np.sqrt(np.mean((p_fz - t_fz) ** 2))),
        "r2_fz": float(1.0 - ss_res / (ss_tot + 1e-12)),
        "mae_fz": float(np.mean(np.abs(p_fz - t_fz))),
        "rmse_all": float(np.sqrt(np.mean((p - t) ** 2))),
    }


def compute_metrics_paper_extras(all_preds: List[np.ndarray], all_targets: List[np.ndarray]) -> Dict[str, float]:
    """Macro R² and Pearson metrics (common supplementary / alternative reporting)."""
    pr = pearson_r_fz(all_preds, all_targets)
    return {
        "r2_fz_macro": float(r2_fz_macro(all_preds, all_targets)),
        "pearson_r_fz": float(pr),
        "pearson_r2_fz": float(pr * pr) if np.isfinite(pr) else float("nan"),
    }


def compute_peak_metrics(all_preds, all_targets, ev_indices) -> Dict[str, float]:
    peak_errs, timing_errs = [], []
    for p, t, ev in zip(all_preds, all_targets, ev_indices):
        p_fz = p[:, -1] if p.ndim > 1 else p
        t_fz = t[:, -1] if t.ndim > 1 else t
        T = len(p_fz)
        ws = max(0, ev - 30)
        we = min(T, ev + 30)
        pi = ws + int(np.argmax(p_fz[ws:we]))
        ti = ws + int(np.argmax(t_fz[ws:we]))
        peak_errs.append(abs(float(p_fz[pi]) - float(t_fz[ti])))
        timing_errs.append(abs(pi - ti))
    return {
        "peak_err_bw": float(np.mean(peak_errs)) if peak_errs else 0.0,
        "peak_timing_fr": float(np.mean(timing_errs)) if timing_errs else 0.0,
    }
