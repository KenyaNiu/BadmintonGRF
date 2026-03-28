"""
DataLoader helpers — limit BLAS threads in workers and multiprocessing start method.

``build_loader_kwargs`` sets ``multiprocessing_context=spawn`` for ``num_workers>0``
(isolated worker processes; avoids fork-after-CUDA / numpy issues in parent). The
global ``BADMINTON_MP_START_METHOD`` still affects ``torch.multiprocessing`` elsewhere
(e.g. ``effective_multiprocessing_start_method()`` logs).

``run_benchmark_resilient.sh`` exports OMP_NUM_THREADS=1; ``worker_init_fn``
forces single-threaded BLAS inside each worker.

Env:
  BADMINTON_MP_START_METHOD unset or ``auto`` — on Unix, use ``forkserver``;
    on Windows, leave default (spawn).
  ``forkserver`` / ``spawn`` — force that method.
  ``fork`` / ``legacy`` / ``default`` — do not call ``set_start_method`` (old Linux fork).
  BADMINTON_PERSISTENT_WORKERS=0 — opt-out long-lived workers (default on for speed; off restarts
    workers each epoch and is very slow with num_workers>0).
  BADMINTON_PREFETCH_FACTOR — optional int (default 4) for DataLoader prefetch when num_workers>0.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any, Callable, Dict, Optional

import torch
import torch.multiprocessing as _mp


def _maybe_set_multiprocessing_start_method() -> None:
    raw = os.environ.get("BADMINTON_MP_START_METHOD", "").strip().lower()
    if raw in ("fork", "legacy", "default", "linux"):
        return
    if raw in ("spawn", "forkserver"):
        target = raw
    elif raw in ("", "auto"):
        if sys.platform == "win32":
            return
        target = "forkserver"
    else:
        return
    current = _mp.get_start_method(allow_none=True)
    if current == target:
        return
    try:
        _mp.set_start_method(target, force=True)
    except RuntimeError:
        pass


_maybe_set_multiprocessing_start_method()


def effective_multiprocessing_start_method() -> str:
    """Resolved start method for logs (None means CPython default before first Process)."""
    m = _mp.get_start_method(allow_none=True)
    if m is not None:
        return m
    if sys.platform == "win32":
        return "spawn"
    return "fork"


def default_num_workers() -> int:
    """Conservative default: feed GPU without spawning too many fork workers."""
    c = os.cpu_count() or 4
    return max(2, min(8, c // 2))


def configure_cuda_training(device: torch.device) -> None:
    """Larger batches + conv: enable cuDNN autotune; Ampere+ benefits from TF32 matmul.

    Some driver/GPU stacks segfault with cuDNN autotune or TF32 under heavy batches.
    If training crashes with SIGSEGV in the main process (even num_workers=0), set:
      export BADMINTON_SAFE_CUDA=1
    or individually:
      BADMINTON_CUDNN_BENCHMARK=0  BADMINTON_MATMUL_PRECISION=medium
    """
    if device.type != "cuda":
        return
    if os.environ.get("BADMINTON_SAFE_CUDA", "").strip() in ("1", "true", "yes"):
        os.environ.setdefault("BADMINTON_CUDNN_BENCHMARK", "0")
        os.environ.setdefault("BADMINTON_MATMUL_PRECISION", "medium")

    bench = os.environ.get("BADMINTON_CUDNN_BENCHMARK", "1").strip().lower()
    torch.backends.cudnn.benchmark = bench not in ("0", "false", "no", "off")

    prec = os.environ.get("BADMINTON_MATMUL_PRECISION", "high").strip().lower()
    if hasattr(torch, "set_float32_matmul_precision"):
        if prec in ("medium", "highest"):
            torch.set_float32_matmul_precision(prec)
        else:
            torch.set_float32_matmul_precision("high")


def worker_init_fn(worker_id: int) -> None:
    """Force single-threaded BLAS in workers (setdefault is not enough after fork)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def build_loader_kwargs(
    *,
    collate_fn: Callable[..., Any],
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: Optional[int] = None,
) -> Dict[str, Any]:
    kw: Dict[str, Any] = {
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        kw["worker_init_fn"] = worker_init_fn
        pf = prefetch_factor
        if pf is None:
            pe = os.environ.get("BADMINTON_PREFETCH_FACTOR", "").strip()
            if pe.isdigit():
                pf = int(pe)
        if pf is None:
            pf = 4
        kw["prefetch_factor"] = max(2, min(32, int(pf)))
        # Match loso_flat: default prefetch 4 is heavy; 2 reduces memory pressure / rare worker crashes.
        if kw.get("prefetch_factor") == 4:
            kw["prefetch_factor"] = 2
        from torch.utils.data import DataLoader as _DL

        sig = inspect.signature(_DL.__init__).parameters
        if "multiprocessing_context" in sig:
            kw["multiprocessing_context"] = _mp.get_context("spawn")
        # persistent_workers: avoids respawning workers each epoch (~10× slower).
        if "persistent_workers" in sig:
            persistent = os.environ.get("BADMINTON_PERSISTENT_WORKERS", "1").strip().lower()
            if persistent not in ("0", "false", "no", "off"):
                kw.setdefault("persistent_workers", True)
    return kw
