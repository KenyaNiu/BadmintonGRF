"""
canonical.py
============
Produce a *canonical* metrics schema across experiments.

Why:
  - E1/E2 (train.py + e2_late_fusion.py) use keys like r2_fz, rmse_fz, peak_err_bw...
  - E4/E5 (ST-GCN + Transformer) use keys like r2, rmse, peak_err, peak_timing...

This module writes an additional JSON file (does NOT modify the original summary.json):
  summary_canonical.json

Training (``--save_report``), fusion, ``aggregate --write-canonical``, and
``python -m baseline paper-export`` call into this for consistent paper tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


CANON_KEYS = [
    "r2_fz",
    "rmse_fz",
    "peak_err_bw",
    "peak_timing_fr",
    "n_test",
    "avg_cams",
]


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _pick(d: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in d:
            v = _as_float(d.get(k))
            if v is not None:
                return v
    return None


def canonicalize_mean(mean: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map mean metrics dict to canonical keys.
    Missing values remain absent.
    """
    out: Dict[str, Any] = {}

    # r2 / rmse
    if (v := _pick(mean, "r2_fz", "r2")) is not None:
        out["r2_fz"] = v
    if (v := _pick(mean, "rmse_fz", "rmse")) is not None:
        out["rmse_fz"] = v

    # peaks
    if (v := _pick(mean, "peak_err_bw", "peak_err")) is not None:
        out["peak_err_bw"] = v
    if (v := _pick(mean, "peak_timing_fr", "peak_timing")) is not None:
        out["peak_timing_fr"] = v

    # counts
    if (v := _pick(mean, "n_test", "n_samples")) is not None:
        out["n_test"] = int(v)
    if (v := _pick(mean, "avg_cams")) is not None:
        out["avg_cams"] = v

    return out


def write_summary_canonical(
    *,
    summary_path: Path,
    experiment: str,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Read an experiment summary.json and write summary_canonical.json next to it.
    Returns the written path.
    """
    d = json.loads(summary_path.read_text(encoding="utf-8"))

    mean = d.get("mean", {})
    canon_mean = canonicalize_mean(mean if isinstance(mean, dict) else {})

    out = {
        "experiment": experiment,
        "source_summary": str(summary_path),
        "canonical_keys": CANON_KEYS,
        "mean": canon_mean,
    }

    out_path = out_path or summary_path.with_name("summary_canonical.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path

