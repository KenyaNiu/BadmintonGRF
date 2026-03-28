"""
Training CLI — invoked as ``python -m baseline train …``.

Registered methods live in :mod:`baseline.registry` (``METHODS``).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

from baseline.registry import get_spec, list_method_ids
from baseline.training.dataloader_utils import default_num_workers
from baseline.training.loso_flat import run_flat_loso
from baseline.training.loso_stgcn import run_stgcn_loso


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BadmintonGRF LOSO training")
    p.add_argument(
        "--method",
        required=True,
        choices=list_method_ids(),
        help="Method id (see baseline/registry.py).",
    )
    p.add_argument("--loso_splits", required=True)
    p.add_argument("--run_dir", required=True, help="Output directory for checkpoints + summary.")
    p.add_argument("--fz_only", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cameras", nargs="+", type=int, default=None)
    p.add_argument("--test_subject", default=None, help="Debug: single fold only.")
    p.add_argument("--folds", nargs="*", default=None, help="Run only these held-out subjects.")
    p.add_argument("--save_report", action="store_true")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss_alpha", type=float, default=10.0)
    p.add_argument("--loss_half_win", type=int, default=25)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--tcn_channels", type=int, default=None)
    p.add_argument("--tcn_blocks", type=int, default=None)
    p.add_argument("--unidirectional", action="store_true")
    p.add_argument("--gcn_ch1", type=int, default=None)
    p.add_argument("--gcn_ch2", type=int, default=None)
    p.add_argument("--tf_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--patch_len", type=int, default=None, help="patch_tst: time patch length.")
    p.add_argument(
        "--max_patch_positions",
        type=int,
        default=None,
        help="patch_tst: sinusoidal PE length cap (rarely need to change).",
    )
    p.add_argument("--kernel_size", type=int, default=None, help="dlinear: moving-average kernel (odd).")
    p.add_argument(
        "--dim_ff",
        type=int,
        default=None,
        help="Transformer FFN width (convformer_grf, patch_tst, patch_tst_xl).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=default_num_workers(),
        help="DataLoader workers (default scales with CPU cores, max 8). Use 0 on SIGSEGV.",
    )
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="Batches prefetched per worker when num_workers>0 (default 4).",
    )
    p.add_argument(
        "--no_pin_memory",
        action="store_true",
        help="Disable pin_memory (sometimes avoids worker crashes with CUDA).",
    )
    p.add_argument(
        "--no_resume_folds",
        action="store_true",
        help="Always retrain every fold; ignore existing best_model.pth (default: resume completed folds).",
    )
    p.add_argument(
        "--paper_metrics",
        action="store_true",
        help="Val-set linear Fz calibration (a·Fz+b) + macro R² + Pearson r² + peak_*_cal; no test labels used for fitting.",
    )
    p.add_argument(
        "--report_r2",
        choices=["raw", "calibrated", "macro", "pearson"],
        default="raw",
        help="Which metric populates canonical r2_fz/r2 in summary (calibrated/macro/pearson need --paper_metrics).",
    )
    p.add_argument(
        "--tta_n",
        type=int,
        default=0,
        help="Test-time augmentation: number of extra noisy forward passes (averaged with the first).",
    )
    p.add_argument(
        "--tta_noise",
        type=float,
        default=0.015,
        help="Gaussian noise std on pose for TTA when --tta_n>0.",
    )
    p.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="EMA decay in (0,1) for shadow weights during val/test (0 disables). Saves best_ema_shadow.pt with best checkpoint.",
    )
    return p


def _argv_has_explicit_lr(argv: List[str]) -> bool:
    for a in argv:
        if a == "--lr" or a.startswith("--lr="):
            return True
    return False


def _maybe_scale_lr_for_batch(args: argparse.Namespace, spec_defaults: dict, explicit_lr: bool) -> None:
    """
    If batch_size differs from the method's registry default and the user did not set --lr,
    scale lr for AdamW (default: sqrt batch scaling, common for large-batch training).

    Env:
      BADMINTON_LR_BATCH_SCALE=sqrt|linear|0   (default sqrt; 0 disables)
    """
    if explicit_lr:
        return
    mode = os.environ.get("BADMINTON_LR_BATCH_SCALE", "sqrt").strip().lower()
    if mode in ("0", "false", "no", "off", "none"):
        return
    base_b = float(spec_defaults.get("batch_size", 128))
    base_lr = float(spec_defaults.get("lr", 1e-3))
    cur_b = float(args.batch_size)
    if base_b <= 0 or cur_b <= 0 or not math.isfinite(base_b) or not math.isfinite(cur_b):
        return
    ratio = cur_b / base_b
    if abs(ratio - 1.0) < 1e-9:
        return
    old_lr = float(args.lr)
    if mode == "linear":
        factor = ratio
    else:
        factor = math.sqrt(ratio)
    args.lr = old_lr * factor
    log = logging.getLogger("baseline.training.cli")
    log.info(
        "LR batch scaling (%s): batch_size=%s vs registry=%s → lr %.6g → %.6g",
        mode if mode in ("linear", "sqrt") else "sqrt",
        int(cur_b),
        int(base_b),
        old_lr,
        args.lr,
    )


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)
    # Default on: fold resume saves time on crash/restart. Opt out: --no_resume_folds or BADMINTON_RESUME_FOLDS=0.
    env_r = os.environ.get("BADMINTON_RESUME_FOLDS", "1").strip().lower()
    if env_r in ("0", "false", "no", "off"):
        args.resume_folds = False
    elif args.no_resume_folds:
        args.resume_folds = False
    else:
        args.resume_folds = True
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    spec = get_spec(args.method)
    d = spec.defaults
    explicit_lr = _argv_has_explicit_lr(argv)
    if args.batch_size is None:
        args.batch_size = d.get("batch_size", 128)
    if args.lr is None:
        args.lr = d.get("lr", 1e-3)
    _maybe_scale_lr_for_batch(args, d, explicit_lr)
    if args.hidden_dim is None:
        args.hidden_dim = d.get("hidden_dim", 128)
    if args.dropout is None:
        args.dropout = d.get("dropout", 0.3)
    if args.num_layers is None:
        args.num_layers = d.get("num_layers", 2)
    if args.tcn_channels is None:
        args.tcn_channels = d.get("tcn_channels", 128)
    if args.tcn_blocks is None:
        args.tcn_blocks = d.get("tcn_blocks", 4)
    if args.gcn_ch1 is None:
        args.gcn_ch1 = d.get("gcn_ch1", 32)
    if args.gcn_ch2 is None:
        args.gcn_ch2 = d.get("gcn_ch2", 64)
    if args.tf_layers is None:
        args.tf_layers = d.get("tf_layers", d.get("num_layers", 2))
    if args.num_heads is None:
        args.num_heads = d.get("num_heads", 4)

    Path(args.run_dir).parent.mkdir(parents=True, exist_ok=True)

    if spec.family == "flat":
        run_flat_loso(args)
    elif spec.family == "stgcn":
        run_stgcn_loso(args)
    else:
        raise SystemExit(f"Unknown family: {spec.family}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
