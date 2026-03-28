"""
train.py  v2.1
==============
LOSO 交叉验证训练入口。

改进：
  - 接触窗口加权 MSE（alpha=10, half_win=25）
  - 默认使用全部相机（cameras=None）
  - Val 按 subject 安全划分
  - Warmup 5ep + CosineAnnealing
  - 早停（patience=20）
  - NaN 梯度保护（跳过异常 batch）

用法：
  # 调试（单 fold，5 epoch）
  python -m baseline.train \\
      --loso_splits ./data/reports/loso_splits.json \\
      --fz_only --test_subject sub_001 --epochs 5

  # 正式跑全 LOSO（全相机）
  nohup python -m baseline.train \\
      --loso_splits ./data/reports/loso_splits.json \\
      --fz_only --epochs 150 --save_report \\
      > runs/train.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from baseline.impact_dataset import (
    INPUT_DIM,
    BadmintonImpactDataset,
    build_loso_datasets,
    collate_fn,
)
from baseline.training.dataloader_utils import build_loader_kwargs, configure_cuda_training, default_num_workers
from baseline.models.tcn_lstm import LSTMBaseline
from baseline.training.losses import contact_weighted_mse
from baseline.training.metrics import compute_metrics, compute_peak_metrics
from baseline.training.split_utils import split_train_val

log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, alpha, half_win, grad_clip) -> float:
    model.train()
    total, n_batches = 0.0, 0
    for batch in loader:
        pose   = batch["pose"].to(device)
        target = batch["target"].to(device)
        ev_idx = batch["ev_idx"].to(device)
        vm = batch.get("valid_mask")
        vm_t = vm.to(device) if vm is not None else None

        # NaN 保护：跳过含 NaN 的 batch（理论上 v2.1 dataset 已消除）
        if torch.any(torch.isnan(pose)) or torch.any(torch.isnan(target)):
            log.warning("跳过含 NaN 的 batch")
            continue

        optimizer.zero_grad()
        pred = model(pose)
        loss = contact_weighted_mse(pred, target, ev_idx, alpha, half_win, vm_t)

        if not torch.isfinite(loss):
            log.warning("非有限 loss=%.4f，跳过", loss.item())
            continue

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        n_batches += 1

    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, device, alpha, half_win):
    model.eval()
    total, n_batches = 0.0, 0
    all_preds, all_targets, all_ev = [], [], []

    for batch in loader:
        pose   = batch["pose"].to(device)
        target = batch["target"].to(device)
        ev_idx = batch["ev_idx"].to(device)
        vm = batch.get("valid_mask")
        vm_t = vm.to(device) if vm is not None else None

        if torch.any(torch.isnan(pose)):
            continue

        pred = model(pose)
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


# ---------------------------------------------------------------------------
# 单 fold
# ---------------------------------------------------------------------------

def train_fold(test_subject, loso_splits, cameras, args, save_dir) -> Dict:
    train_ds, test_ds = build_loso_datasets(
        loso_splits, test_subject, cameras,
        predict_fz_only=args.fz_only,
        augment_train=not args.no_augment,
    )
    if len(train_ds) == 0:
        log.warning("fold %s: 训练集为空", test_subject); return {}

    device = torch.device(args.device)
    configure_cuda_training(device)
    best_ckpt = save_dir / f"fold_{test_subject}" / "best_model.pth"
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)
    nw = int(getattr(args, "num_workers", default_num_workers()))
    pin_mem = (not getattr(args, "no_pin_memory", False)) and (device.type == "cuda")
    kw = build_loader_kwargs(
        collate_fn=collate_fn,
        num_workers=nw,
        pin_memory=pin_mem,
        prefetch_factor=getattr(args, "prefetch_factor", None),
    )
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, **kw)
    output_dim = 1 if args.fz_only else 3

    env_resume = os.environ.get("BADMINTON_RESUME_FOLDS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    train_done = (best_ckpt.parent / "train_log.csv").exists()
    if env_resume and not getattr(args, "no_resume_folds", False) and best_ckpt.exists() and train_done:
        log.info("  resume fold %s: skip train, test eval only", test_subject)
        model = LSTMBaseline(
            input_dim=INPUT_DIM, hidden_dim=args.hidden_dim,
            output_dim=output_dim, num_layers=args.num_layers,
            dropout=args.dropout, bidirectional=not args.unidirectional,
            tcn_channels=args.tcn_channels,
        ).to(device)
        try:
            model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        except Exception as e:
            log.warning("  resume load failed (%s); training from scratch", e)
        else:
            te_loss, te_m, *_ = evaluate_epoch(
                model, test_loader, device, args.loss_alpha, args.loss_half_win
            )
            te_m.update({
                "test_loss": round(te_loss, 6) if np.isfinite(te_loss) else None,
                "n_test": len(test_ds),
            })
            te_m["r2"] = te_m.get("r2_fz", float("nan"))
            te_m["rmse"] = te_m.get("rmse_fz", float("nan"))
            te_m["peak_err"] = te_m.get("peak_err_bw", float("nan"))
            te_m["peak_timing"] = te_m.get("peak_timing_fr", float("nan"))
            log.info(
                "  ✅ fold %s (resume) → rmse_fz=%.4f  r2_fz=%.3f  peak_err=%.4f×BW  timing=%.1ff",
                test_subject,
                te_m.get("rmse_fz", float("nan")), te_m.get("r2_fz", float("nan")),
                te_m.get("peak_err_bw", float("nan")), te_m.get("peak_timing_fr", float("nan")),
            )
            return te_m

    train_sub, val_ds = split_train_val(train_ds, 0.15)
    train_loader = DataLoader(train_sub, args.batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, **kw)

    log.info("  fold %s: train=%d  val=%d  test=%d",
             test_subject, len(train_sub), len(val_ds), len(test_ds))

    model = LSTMBaseline(
        input_dim=INPUT_DIM, hidden_dim=args.hidden_dim,
        output_dim=output_dim, num_layers=args.num_layers,
        dropout=args.dropout, bidirectional=not args.unidirectional,
        tcn_channels=args.tcn_channels,
    ).to(device)
    log.info("  参数量：%s", f"{model.count_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup: return (ep+1)/warmup
        p = (ep-warmup)/max(args.epochs-warmup, 1)
        return 0.5*(1+np.cos(np.pi*p))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_mode = os.environ.get("BADMINTON_BEST_METRIC", "loss").strip().lower()
    if best_mode not in ("loss", "r2"):
        log.warning("  BADMINTON_BEST_METRIC=%s 无效，使用 loss", best_mode)
        best_mode = "loss"
    best_metric = float("inf") if best_mode == "loss" else float("-inf")
    best_epoch = 0
    best_val_loss_at_ckpt = float("nan")
    best_val_r2_at_ckpt = float("nan")
    patience_ct = 0

    log_rows = []; t0 = time.time()

    for epoch in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_loader, optimizer, device,
                             args.loss_alpha, args.loss_half_win, args.grad_clip)
        va_loss, va_m, *_ = evaluate_epoch(model, val_loader, device,
                                            args.loss_alpha, args.loss_half_win)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": round(tr,6),
               "val_loss": round(va_loss,6) if np.isfinite(va_loss) else None,
               **{f"val_{k}": round(v,6) for k,v in va_m.items()}}
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
            else:
                patience_ct += 1

        if epoch % 10 == 0 or epoch == 1:
            log.info("    ep%3d/%d  tr=%.4f  va=%.4f  r2=%.3f  rmse=%.4f  [%.0fs]",
                     epoch, args.epochs, tr,
                     va_loss if np.isfinite(va_loss) else float("nan"),
                     va_m.get("r2_fz", float("nan")),
                     va_m.get("rmse_fz", float("nan")),
                     time.time()-t0)

        if patience_ct >= args.patience:
            log.info(
                "  早停 epoch=%d  best_mode=%s  best_epoch=%d  metric=%s  val_loss@best=%.6f  val_r2@best=%.4f",
                epoch,
                best_mode,
                best_epoch,
                f"{best_metric:.6f}" if np.isfinite(best_metric) else "nan",
                best_val_loss_at_ckpt if np.isfinite(best_val_loss_at_ckpt) else float("nan"),
                best_val_r2_at_ckpt if np.isfinite(best_val_r2_at_ckpt) else float("nan"),
            )
            break

    log.info(
        "  最佳 checkpoint: mode=%s  epoch=%d  metric=%s  val_loss@best=%.6f  val_r2@best=%.4f",
        best_mode,
        best_epoch,
        f"{best_metric:.6f}" if np.isfinite(best_metric) else "nan",
        best_val_loss_at_ckpt if np.isfinite(best_val_loss_at_ckpt) else float("nan"),
        best_val_r2_at_ckpt if np.isfinite(best_val_r2_at_ckpt) else float("nan"),
    )

    # 加载最佳模型（若不存在则用最后一轮）
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    else:
        log.warning("  未找到最佳模型，使用最后一轮参数")

    te_loss, te_m, preds, targets, evs = evaluate_epoch(
        model, test_loader, device, args.loss_alpha, args.loss_half_win)
    te_m.update({"test_loss": round(te_loss,6) if np.isfinite(te_loss) else None,
                 "n_test": len(test_ds)})

    log.info("  ✅ fold %s → rmse_fz=%.4f  r2_fz=%.3f  peak_err=%.4f×BW  timing=%.1ff",
             test_subject,
             te_m.get("rmse_fz", float("nan")), te_m.get("r2_fz", float("nan")),
             te_m.get("peak_err_bw", float("nan")), te_m.get("peak_timing_fr", float("nan")))

    if log_rows:
        csv_path = best_ckpt.parent / "train_log.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader(); writer.writerows(log_rows)

    return te_m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BadmintonGRF LOSO 训练 v2.1")
    p.add_argument("--loso_splits",   required=True)
    p.add_argument("--cameras",       nargs="+", type=int, default=None,
                   help="相机编号（默认全部）")
    p.add_argument("--fz_only",       action="store_true")
    p.add_argument("--test_subject",  default=None)
    p.add_argument("--hidden_dim",    type=int,   default=64)
    p.add_argument("--num_layers",    type=int,   default=2)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--tcn_channels",  type=int,   default=64)
    p.add_argument("--unidirectional",action="store_true")
    p.add_argument("--epochs",        type=int,   default=150)
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--no_augment",    action="store_true")
    p.add_argument("--loss_alpha",    type=float, default=10.0)
    p.add_argument("--loss_half_win", type=int,   default=25)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_level",     default="INFO")
    p.add_argument("--out_dir",       default="runs")
    p.add_argument(
        "--run_name",
        default=None,
        help="自定义输出目录名（位于 --out_dir 下）。不传则使用默认命名。",
    )
    p.add_argument(
        "--run_dir",
        default=None,
        help="直接指定完整输出目录路径（优先级高于 --out_dir/--run_name）。",
    )
    p.add_argument("--save_report",   action="store_true")
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers (default: scale with CPU). 0 avoids fork+OpenMP SIGSEGV.",
    )
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--no_pin_memory", action="store_true")
    p.add_argument(
        "--no_resume_folds",
        action="store_true",
        help="Ignore existing best_model.pth and retrain every fold (default: resume).",
    )
    return p.parse_args()


def _summarize(results):
    sample = results[next(iter(results))]
    keys   = [k for k in sample if isinstance(sample.get(k), (int, float))]
    mean   = {k: float(np.nanmean([r[k] for r in results.values() if k in r])) for k in keys}
    std    = {k: float(np.nanstd( [r[k] for r in results.values() if k in r])) for k in keys}
    return {"mean": mean, "std": std}


def main():
    args = parse_args()
    if args.num_workers is None:
        args.num_workers = default_num_workers()
    env_r = os.environ.get("BADMINTON_RESUME_FOLDS", "1").strip().lower()
    if env_r in ("0", "false", "no", "off"):
        args.no_resume_folds = True
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cam_tag  = "all" if args.cameras is None else "".join(str(c) for c in args.cameras)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 输出目录可控：
    # - --run_dir：直接指定完整路径（最确定，脚本/集群推荐）
    # - --run_name：在 --out_dir 下自定义目录名
    if args.run_dir:
        save_dir = Path(args.run_dir)
    else:
        run_name = args.run_name or f"{ts}_lstm_tcn_cam{cam_tag}"
        save_dir = Path(args.out_dir) / run_name

    save_dir.mkdir(parents=True, exist_ok=True)
    log.info("输出目录：%s", save_dir)

    splits   = json.loads(Path(args.loso_splits).read_text(encoding="utf-8"))
    subjects = [args.test_subject] if args.test_subject else sorted(splits.keys())
    log.info("LOSO folds：%s  cameras=%s  INPUT_DIM=%d  loss_alpha=%.0f",
             subjects, args.cameras or "all", INPUT_DIM, args.loss_alpha)

    (save_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    all_results = {}
    for test_sub in subjects:
        log.info("\n%s\nFOLD: %s\n%s", "="*60, test_sub, "="*60)
        r = train_fold(test_sub, args.loso_splits, args.cameras, args, save_dir)
        if r: all_results[test_sub] = r

    if not all_results:
        log.error("所有 fold 均失败"); sys.exit(1)

    summary = _summarize(all_results)
    log.info("\n%s\nLOSO 汇总（论文 Table）\n%s", "="*60, "="*60)
    for k, v in summary["mean"].items():
        log.info("  %-22s  %.4f ± %.4f", k, v, summary["std"].get(k,0))

    if args.save_report:
        out = save_dir / "summary.json"
        out.write_text(json.dumps({"per_fold": all_results, **summary},
                                   indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
        log.info("汇总：%s", out)
        try:
            from baseline.tasks.canonical import write_summary_canonical

            cp = write_summary_canonical(summary_path=out, experiment="tcn_bilstm_legacy_trainpy")
            log.info("summary_canonical：%s", cp)
        except Exception as e:
            log.warning("Could not write summary_canonical.json: %s", e)


if __name__ == "__main__":
    main()