"""
LOSO training for *flat-family* models (TCN+LSTM/GRU/MLP/Seq-Transformer).

Extracted from ``baseline/train.py`` with an explicit ``method_id`` and shared metrics.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from baseline.registry import apply_method_defaults, build_flat_model, get_spec
from baseline.impact_dataset import (
    INPUT_DIM,
    BadmintonImpactDataset,
    build_loso_datasets,
    collate_fn,
)
from baseline.training.dataloader_utils import (
    build_loader_kwargs,
    configure_cuda_training,
    default_num_workers,
    effective_multiprocessing_start_method,
)
from baseline.training.ema import ModelEMA
from baseline.training.losses import contact_weighted_mse
from baseline.training.metrics import (
    apply_fz_linear_calibration,
    compute_metrics,
    compute_metrics_paper_extras,
    compute_peak_metrics,
    fit_fz_linear_calibration,
)
from baseline.training.split_utils import split_train_val

log = logging.getLogger("baseline.training.loso_flat")


def train_one_epoch(model, loader, optimizer, device, alpha, half_win, grad_clip, ema: Optional[ModelEMA] = None) -> float:
    model.train()
    total, n_batches = 0.0, 0
    for batch in loader:
        pose = batch["pose"].to(device)
        target = batch["target"].to(device)
        ev_idx = batch["ev_idx"].to(device)
        vm = batch.get("valid_mask")
        vm_t = vm.to(device) if vm is not None else None
        if torch.any(torch.isnan(pose)) or torch.any(torch.isnan(target)):
            continue
        optimizer.zero_grad()
        pred = model(pose)
        loss = contact_weighted_mse(pred, target, ev_idx, alpha, half_win, vm_t)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model,
    loader,
    device,
    alpha,
    half_win,
    *,
    tta_n: int = 0,
    tta_noise: float = 0.0,
):
    model.eval()
    total, n_batches = 0.0, 0
    all_preds, all_targets, all_ev = [], [], []
    for batch in loader:
        pose = batch["pose"].to(device)
        target = batch["target"].to(device)
        ev_idx = batch["ev_idx"].to(device)
        vm = batch.get("valid_mask")
        vm_t = vm.to(device) if vm is not None else None
        if torch.any(torch.isnan(pose)):
            continue
        pred = model(pose)
        if tta_n > 0 and tta_noise > 0:
            acc = pred
            for _ in range(tta_n):
                acc = acc + model(pose + torch.randn_like(pose) * tta_noise)
            pred = acc / float(1 + tta_n)
        loss = contact_weighted_mse(pred, target, ev_idx, alpha, half_win, vm_t)
        if torch.isfinite(loss):
            total += loss.item()
            n_batches += 1
        pred_np = pred.cpu().numpy()
        tgt_np = target.cpu().numpy()
        ev_np = ev_idx.cpu().numpy()
        vm_np = vm.numpy() if vm is not None else None
        for b in range(pred_np.shape[0]):
            if vm_np is not None:
                L = int(vm_np[b].sum())
                all_preds.append(pred_np[b, :L])
                all_targets.append(tgt_np[b, :L])
            else:
                all_preds.append(pred_np[b])
                all_targets.append(tgt_np[b])
            all_ev.append(int(ev_np[b]))
    if not all_preds:
        return float("nan"), {}, [], [], []
    metrics = compute_metrics(all_preds, all_targets)
    metrics.update(compute_peak_metrics(all_preds, all_targets, all_ev))
    return total / max(n_batches, 1), metrics, all_preds, all_targets, all_ev


def _apply_report_r2(te_m: Dict[str, Any], mode: str) -> None:
    """Pick which R² populates canonical ``r2_fz`` / ``r2`` for tables (all keys remain in te_m)."""
    mode = (mode or "raw").strip().lower()
    if mode == "raw":
        return
    key_map = {
        "calibrated": "r2_fz_cal",
        "macro": "r2_fz_macro",
        "pearson": "pearson_r2_fz",
    }
    src = key_map.get(mode)
    if not src or src not in te_m or not np.isfinite(float(te_m.get(src, float("nan")))):
        return
    te_m["r2_fz"] = float(te_m[src])
    if mode == "calibrated" and np.isfinite(float(te_m.get("rmse_fz_cal", float("nan")))):
        te_m["rmse_fz"] = float(te_m["rmse_fz_cal"])
    if mode == "calibrated":
        if np.isfinite(float(te_m.get("peak_err_bw_cal", float("nan")))):
            te_m["peak_err_bw"] = float(te_m["peak_err_bw_cal"])
            te_m["peak_err"] = te_m["peak_err_bw"]
        if np.isfinite(float(te_m.get("peak_timing_fr_cal", float("nan")))):
            te_m["peak_timing_fr"] = float(te_m["peak_timing_fr_cal"])
            te_m["peak_timing"] = te_m["peak_timing_fr"]
    te_m["r2"] = te_m["r2_fz"]


def _paper_enrich_test_metrics(
    te_m: Dict[str, Any],
    *,
    val_preds: List,
    val_targets: List,
    test_preds: List,
    test_targets: List,
    test_ev: List,
) -> None:
    """Val-fit affine calibration on Fz + macro/Pearson on calibrated preds; no test labels used for fitting."""
    a, b = fit_fz_linear_calibration(val_preds, val_targets)
    pred_cal = apply_fz_linear_calibration(test_preds, a, b)
    m_cal = compute_metrics(pred_cal, test_targets)
    te_m["r2_fz_raw"] = float(te_m.get("r2_fz", float("nan")))
    te_m["rmse_fz_raw"] = float(te_m.get("rmse_fz", float("nan")))
    te_m["r2_fz_cal"] = float(m_cal["r2_fz"])
    te_m["rmse_fz_cal"] = float(m_cal["rmse_fz"])
    te_m["fz_cal_a"] = a
    te_m["fz_cal_b"] = b
    te_m.update(compute_metrics_paper_extras(pred_cal, test_targets))
    peak_cal = compute_peak_metrics(pred_cal, test_targets, test_ev)
    te_m["peak_err_bw_cal"] = float(peak_cal["peak_err_bw"])
    te_m["peak_timing_fr_cal"] = float(peak_cal["peak_timing_fr"])


def train_fold(
    test_subject: str,
    loso_splits: str,
    cameras: Optional[List[int]],
    args: Any,
    save_dir: Path,
    model_factory: Callable[[], nn.Module],
) -> Dict:
    train_ds, test_ds = build_loso_datasets(
        loso_splits,
        test_subject,
        cameras,
        predict_fz_only=args.fz_only,
        augment_train=not args.no_augment,
    )
    if len(train_ds) == 0:
        log.warning("fold %s: empty train", test_subject)
        return {}
    device = torch.device(args.device)
    configure_cuda_training(device)
    best_ckpt = save_dir / f"fold_{test_subject}" / "best_model.pth"
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    _nw = getattr(args, "num_workers", None)
    nw = int(default_num_workers() if _nw is None else _nw)
    pin_mem = (not getattr(args, "no_pin_memory", False)) and (device.type == "cuda")
    kw = build_loader_kwargs(
        collate_fn=collate_fn,
        num_workers=nw,
        pin_memory=pin_mem,
        prefetch_factor=getattr(args, "prefetch_factor", None) if nw > 0 else None,
    )
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, **kw)

    # Only resume when training actually finished (train_log.csv). Mid-crash checkpoints
    # also have best_model.pth but must not skip training.
    fold_dir = best_ckpt.parent
    train_finished = (fold_dir / "train_log.csv").exists()
    if getattr(args, "resume_folds", False) and best_ckpt.exists() and train_finished:
        log.info(
            "  resume fold %s: skip train, load %s → test eval only",
            test_subject,
            best_ckpt,
        )
        model = model_factory().to(device)
        try:
            model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        except Exception as e:
            log.warning("  resume load failed (%s); training from scratch", e)
        else:
            paper = bool(getattr(args, "paper_metrics", False))
            tta_n = int(getattr(args, "tta_n", 0) or 0)
            tta_noise = float(getattr(args, "tta_noise", 0.015) or 0.0)
            report_r2 = getattr(args, "report_r2", "raw")
            if paper or tta_n > 0:
                tr_sub, va_ds = split_train_val(train_ds, 0.15)
                va_loader = DataLoader(va_ds, args.batch_size, shuffle=False, **kw)
                if paper:
                    _, _, val_preds, val_targets, _ = evaluate_epoch(
                        model, va_loader, device, args.loss_alpha, args.loss_half_win
                    )
                te_loss, te_m, preds, targets, evs = evaluate_epoch(
                    model,
                    test_loader,
                    device,
                    args.loss_alpha,
                    args.loss_half_win,
                    tta_n=max(0, tta_n),
                    tta_noise=tta_noise if tta_n > 0 else 0.0,
                )
                if paper:
                    _paper_enrich_test_metrics(
                        te_m,
                        val_preds=val_preds,
                        val_targets=val_targets,
                        test_preds=preds,
                        test_targets=targets,
                        test_ev=evs,
                    )
                _apply_report_r2(te_m, report_r2)
            else:
                te_loss, te_m, preds, targets, evs = evaluate_epoch(
                    model, test_loader, device, args.loss_alpha, args.loss_half_win
                )
            te_m.update(
                {
                    "test_loss": round(te_loss, 6) if np.isfinite(te_loss) else None,
                    "n_test": len(test_ds),
                }
            )
            te_m["r2"] = te_m.get("r2_fz", float("nan"))
            te_m["rmse"] = te_m.get("rmse_fz", float("nan"))
            te_m["peak_err"] = te_m.get("peak_err_bw", float("nan"))
            te_m["peak_timing"] = te_m.get("peak_timing_fr", float("nan"))
            log.info(
                "  fold %s (resume): rmse_fz=%.4f r2_fz=%.3f peak_err=%.4f×BW timing=%.1ff",
                test_subject,
                te_m.get("rmse_fz", float("nan")),
                te_m.get("r2_fz", float("nan")),
                te_m.get("peak_err_bw", float("nan")),
                te_m.get("peak_timing_fr", float("nan")),
            )
            return te_m

    train_sub, val_ds = split_train_val(train_ds, 0.15)
    train_loader = DataLoader(train_sub, args.batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, **kw)
    log.info(
        "  fold %s: train=%d val=%d test=%d",
        test_subject,
        len(train_sub),
        len(val_ds),
        len(test_ds),
    )
    log.info(
        "  loader: batch_size=%d num_workers=%d mp_start=%s prefetch=%s pin_memory=%s",
        int(args.batch_size),
        nw,
        effective_multiprocessing_start_method(),
        kw.get("prefetch_factor", "-"),
        pin_mem,
    )
    log.info(
        "  thread env: OMP=%s MKL=%s OPENBLAS=%s",
        os.environ.get("OMP_NUM_THREADS", "unset"),
        os.environ.get("MKL_NUM_THREADS", "unset"),
        os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
    )
    model = model_factory().to(device)
    log.info("  params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    ema_decay = float(getattr(args, "ema_decay", 0.0) or 0.0)
    ema: Optional[ModelEMA] = ModelEMA(model, ema_decay) if ema_decay > 0 else None
    if ema is not None:
        log.info("  EMA decay=%.5f (eval + best checkpoint use shadow weights)", ema_decay)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = 5

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        p = (ep - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * p))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_mode = os.environ.get("BADMINTON_BEST_METRIC", "loss").strip().lower()
    if best_mode not in ("loss", "r2"):
        log.warning("  BADMINTON_BEST_METRIC=%s invalid; use loss or r2", best_mode)
        best_mode = "loss"
    # loss: minimize; r2: maximize val r2_fz (aligned with reported metric when loss and R² disagree)
    best_metric = float("inf") if best_mode == "loss" else float("-inf")
    best_epoch = 0
    best_val_loss_at_ckpt = float("nan")
    best_val_r2_at_ckpt = float("nan")
    patience_ct = 0
    log_rows = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.loss_alpha,
            args.loss_half_win,
            args.grad_clip,
            ema=ema,
        )
        if ema is not None:
            ema.apply_to(model)
        va_loss, va_m, *_ = evaluate_epoch(
            model, val_loader, device, args.loss_alpha, args.loss_half_win
        )
        if ema is not None:
            ema.restore(model)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": round(tr, 6),
            "val_loss": round(va_loss, 6) if np.isfinite(va_loss) else None,
            **{f"val_{k}": round(v, 6) for k, v in va_m.items()},
        }
        log_rows.append(row)
        if best_mode == "r2":
            cur_r2 = float(va_m.get("r2_fz", float("-inf")))
            if not np.isfinite(cur_r2):
                cur_r2 = float("-inf")
            improved = cur_r2 > best_metric
            if improved:
                best_metric = cur_r2
                best_epoch = epoch
                best_val_loss_at_ckpt = float(va_loss)
                best_val_r2_at_ckpt = cur_r2
                patience_ct = 0
                torch.save(model.state_dict(), best_ckpt)
                if ema is not None:
                    torch.save(
                        {k: v.detach().cpu() for k, v in ema.shadow.items()},
                        best_ckpt.parent / "best_ema_shadow.pt",
                    )
            else:
                patience_ct += 1
        else:
            if np.isfinite(va_loss) and va_loss < best_metric:
                best_metric = va_loss
                best_epoch = epoch
                best_val_loss_at_ckpt = float(va_loss)
                best_val_r2_at_ckpt = float(va_m.get("r2_fz", float("nan")))
                patience_ct = 0
                torch.save(model.state_dict(), best_ckpt)
                if ema is not None:
                    torch.save(
                        {k: v.detach().cpu() for k, v in ema.shadow.items()},
                        best_ckpt.parent / "best_ema_shadow.pt",
                    )
            else:
                patience_ct += 1
        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "    ep%3d/%d tr=%.4f va=%.4f r2=%.3f rmse=%.4f [%.0fs]",
                epoch,
                args.epochs,
                tr,
                va_loss if np.isfinite(va_loss) else float("nan"),
                va_m.get("r2_fz", float("nan")),
                va_m.get("rmse_fz", float("nan")),
                time.time() - t0,
            )
        if patience_ct >= args.patience:
            log.info(
                "  early stop epoch=%d best_mode=%s best_epoch=%d best_metric=%s val_loss@best=%.6f val_r2@best=%.4f",
                epoch,
                best_mode,
                best_epoch,
                f"{best_metric:.6f}" if np.isfinite(best_metric) else "nan",
                best_val_loss_at_ckpt if np.isfinite(best_val_loss_at_ckpt) else float("nan"),
                best_val_r2_at_ckpt if np.isfinite(best_val_r2_at_ckpt) else float("nan"),
            )
            break
    log.info(
        "  best checkpoint: mode=%s epoch=%d metric=%s val_loss@best=%.6f val_r2@best=%.4f",
        best_mode,
        best_epoch,
        f"{best_metric:.6f}" if np.isfinite(best_metric) else "nan",
        best_val_loss_at_ckpt if np.isfinite(best_val_loss_at_ckpt) else float("nan"),
        best_val_r2_at_ckpt if np.isfinite(best_val_r2_at_ckpt) else float("nan"),
    )
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    else:
        log.warning("  no best checkpoint; using last epoch")

    ema_shadow_path = best_ckpt.parent / "best_ema_shadow.pt"
    if ema is not None and best_ckpt.exists() and ema_shadow_path.exists():
        ema.load_shadow(torch.load(ema_shadow_path, map_location=device, weights_only=True))
        ema.apply_to(model)
        log.info("  inference using EMA shadow weights")

    paper = bool(getattr(args, "paper_metrics", False))
    tta_n = int(getattr(args, "tta_n", 0) or 0)
    tta_noise = float(getattr(args, "tta_noise", 0.015) or 0.0)
    report_r2 = getattr(args, "report_r2", "raw")

    if paper or tta_n > 0:
        if paper:
            _, _, val_preds, val_targets, _ = evaluate_epoch(
                model, val_loader, device, args.loss_alpha, args.loss_half_win
            )
        te_loss, te_m, preds, targets, evs = evaluate_epoch(
            model,
            test_loader,
            device,
            args.loss_alpha,
            args.loss_half_win,
            tta_n=max(0, tta_n),
            tta_noise=tta_noise if tta_n > 0 else 0.0,
        )
        if paper:
            _paper_enrich_test_metrics(
                te_m,
                val_preds=val_preds,
                val_targets=val_targets,
                test_preds=preds,
                test_targets=targets,
                test_ev=evs,
            )
        _apply_report_r2(te_m, report_r2)
    else:
        te_loss, te_m, preds, targets, evs = evaluate_epoch(
            model, test_loader, device, args.loss_alpha, args.loss_half_win
        )

    te_m.update(
        {
            "test_loss": round(te_loss, 6) if np.isfinite(te_loss) else None,
            "n_test": len(test_ds),
        }
    )
    # Keys aligned with stgcn summary for aggregate tables
    te_m["r2"] = te_m.get("r2_fz", float("nan"))
    te_m["rmse"] = te_m.get("rmse_fz", float("nan"))
    te_m["peak_err"] = te_m.get("peak_err_bw", float("nan"))
    te_m["peak_timing"] = te_m.get("peak_timing_fr", float("nan"))

    log.info(
        "  fold %s: rmse_fz=%.4f r2_fz=%.3f peak_err=%.4f×BW timing=%.1ff",
        test_subject,
        te_m.get("rmse_fz", float("nan")),
        te_m.get("r2_fz", float("nan")),
        te_m.get("peak_err_bw", float("nan")),
        te_m.get("peak_timing_fr", float("nan")),
    )
    if log_rows:
        csv_path = best_ckpt.parent / "train_log.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
    return te_m


def run_flat_loso(args: Any) -> Dict[str, Any]:
    """Run full LOSO for a flat method (``args.method`` must be set)."""
    method_id = args.method
    get_spec(method_id)
    apply_method_defaults(method_id, args)

    save_dir = Path(args.run_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", save_dir)

    cfg = json.loads(json.dumps(vars(args), default=str))
    cfg["method_id"] = method_id
    cfg["family"] = "flat"
    (save_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    splits = json.loads(Path(args.loso_splits).read_text(encoding="utf-8"))
    subjects = [args.test_subject] if getattr(args, "test_subject", None) else sorted(splits.keys())
    if getattr(args, "folds", None):
        subjects = [s for s in subjects if s in args.folds]

    output_dim = 1 if args.fz_only else 3

    def factory():
        return build_flat_model(method_id, args, INPUT_DIM, output_dim)

    all_results: Dict[str, Dict] = {}
    for test_sub in subjects:
        log.info("\n%s\nFOLD %s\n%s", "=" * 60, test_sub, "=" * 60)
        r = train_fold(test_sub, args.loso_splits, args.cameras, args, save_dir, factory)
        if r:
            all_results[test_sub] = r

    if not all_results:
        log.error("All folds failed")
        sys.exit(1)

    def _mean_std(keys):
        m = {k: float(np.nanmean([r[k] for r in all_results.values() if k in r])) for k in keys}
        s = {k: float(np.nanstd([r[k] for r in all_results.values() if k in r])) for k in keys}
        return m, s

    keys_fz = ["r2_fz", "rmse_fz", "peak_err_bw", "peak_timing_fr", "n_test"]
    for ok in ("r2_fz_cal", "r2_fz_macro", "pearson_r2_fz", "r2_fz_raw"):
        if any(ok in r for r in all_results.values()):
            keys_fz.insert(-1, ok)
    mean, std = _mean_std(keys_fz)
    summary = {
        "experiment": method_id,
        "method": method_id,
        "family": "flat",
        "mean": {**{k: round(mean[k], 4) for k in mean}, "r2_std": round(std.get("r2_fz", 0), 4)},
        "std": {k: round(std[k], 4) for k in std},
        "per_fold": all_results,
    }
    # Mirror stgcn-style keys at top level mean
    summary["mean"]["r2"] = round(mean.get("r2_fz", float("nan")), 4)
    summary["mean"]["rmse"] = round(mean.get("rmse_fz", float("nan")), 4)
    summary["mean"]["peak_err"] = round(mean.get("peak_err_bw", float("nan")), 4)
    summary["mean"]["peak_timing"] = round(mean.get("peak_timing_fr", float("nan")), 4)

    log.info("\n%s\nLOSO summary\n%s", "=" * 60, "=" * 60)
    for k in keys_fz:
        if k in mean:
            log.info("  %-18s  %.4f ± %.4f", k, mean[k], std.get(k, 0))

    if args.save_report:
        out = save_dir / "summary.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log.info("Wrote %s", out)
        try:
            from baseline.tasks.canonical import write_summary_canonical

            cp = write_summary_canonical(summary_path=out, experiment=method_id)
            log.info("Wrote %s", cp)
        except Exception as e:
            log.warning("Could not write summary_canonical.json: %s", e)
    return summary
