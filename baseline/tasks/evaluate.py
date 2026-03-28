"""
evaluate.py
===========
加载训练好的模型，在指定测试集上评估，输出论文所需的所有指标。

支持：
  1. 单 fold 评估（指定 --model_path 和 --test_subject）
  2. 全 LOSO 汇总（从 summary.json 读取已有结果直接展示）
  3. 疲劳分析（--fatigue_analysis：按 stage 分组，展示疲劳对预测误差的影响）

用法：
  # 评估单个 fold
  python -m baseline evaluate \\
      --loso_splits ./data/reports/loso_splits.json \\
      --model_path  runs/.../fold_sub_001/best_model.pth \\
      --test_subject sub_001 \\
      --cameras 2

  # 疲劳分析
  python -m baseline evaluate \\
      --loso_splits ./data/reports/loso_splits.json \\
      --model_path  runs/.../fold_sub_001/best_model.pth \\
      --test_subject sub_001 --cameras 2 \\
      --fatigue_analysis

  # 从 summary.json 直接打印汇总表
  python -m baseline evaluate --summary_path runs/.../summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline.impact_dataset import (
    INPUT_DIM,
    build_loso_datasets,
    collate_fn,
)
from baseline.models.tcn_lstm import LSTMBaseline
from baseline.training.dataloader_utils import build_loader_kwargs

log = logging.getLogger("evaluate")

# ---------------------------------------------------------------------------
# 核心评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
) -> Dict:
    """
    返回完整评估结果，包含每条序列的预测和元数据。
    """
    model.eval()
    criterion = nn.MSELoss()

    records = []   # 每条序列一条记录
    total_loss = 0.0

    for batch in loader:
        pose     = batch["pose"].to(device)
        target   = batch["target"].to(device)
        grf_full = batch.get("grf_full", target).to(device)
        ev_idx   = batch["ev_idx"]
        vm       = batch.get("valid_mask")

        pred = model(pose)
        loss = criterion(pred, target)
        total_loss += loss.item()

        # 逐样本保存（按 valid_mask 截断 padding）
        for i in range(pred.shape[0]):
            L = int(vm[i].sum().item()) if vm is not None else pred.shape[1]
            p_np = pred[i, :L].cpu().numpy()        # (T, D)
            t_np = target[i, :L].cpu().numpy()
            g_np = grf_full[i, :L].cpu().numpy()    # (T, 3) 完整 GRF
            ev   = int(ev_idx[i])

            # Fz 分量
            p_fz = p_np[:, 2] if p_np.shape[-1] >= 3 else p_np[:, 0]
            t_fz = t_np[:, 2] if t_np.shape[-1] >= 3 else t_np[:, 0]
            g_fz = g_np[:, 2]

            # 峰值
            T    = len(p_fz)
            ws   = max(0, ev - 30)
            we   = min(T, ev + 30)
            pi   = ws + int(np.argmax(p_fz[ws:we]))
            ti   = ws + int(np.argmax(t_fz[ws:we]))

            records.append({
                "pred_fz":        p_fz,
                "target_fz":      t_fz,
                "target_grf_full": g_fz,
                "pred_peak":      float(p_fz[pi]),
                "target_peak":    float(t_fz[ti]),
                "peak_err":       abs(float(p_fz[pi]) - float(t_fz[ti])),
                "peak_timing_err": abs(pi - ti),
                "ev_idx":         ev,
                "subject":        batch.get("subject", [""])[i] if "subject" in batch else "",
                "stage":          batch.get("stage",   [""])[i] if "stage"   in batch else "",
                "camera":         int(batch.get("camera", [0])[i]) if "camera" in batch else 0,
                "path":           batch.get("path", [""])[i] if "path" in batch else "",
            })

    # 汇总指标
    p_fz_all = np.concatenate([r["pred_fz"]   for r in records])
    t_fz_all = np.concatenate([r["target_fz"] for r in records])

    ss_res = float(np.sum((t_fz_all - p_fz_all) ** 2))
    ss_tot = float(np.sum((t_fz_all - t_fz_all.mean()) ** 2))

    metrics = {
        "n_samples":        len(records),
        "mse_loss":         round(total_loss / max(len(loader), 1), 6),
        "rmse_fz":          round(float(np.sqrt(np.mean((p_fz_all - t_fz_all)**2))), 4),
        "mae_fz":           round(float(np.mean(np.abs(p_fz_all - t_fz_all))), 4),
        "r2_fz":            round(float(1.0 - ss_res / (ss_tot + 1e-12)), 4),
        "peak_err_mean":    round(float(np.mean([r["peak_err"]        for r in records])), 4),
        "peak_err_std":     round(float(np.std( [r["peak_err"]        for r in records])), 4),
        "peak_timing_mean": round(float(np.mean([r["peak_timing_err"] for r in records])), 2),
    }

    return {"metrics": metrics, "records": records}


# ---------------------------------------------------------------------------
# 疲劳分析（论文 E3）
# ---------------------------------------------------------------------------

def fatigue_analysis(records: List[Dict]) -> Dict:
    """
    按 stage 分组，计算各阶段的预测误差。

    stage 分为三大类：
      fresh  : stage1, stage2, stage3（非疲劳阶段）
      fatigue: fatigue_stage1, fatigue_stage2, fatigue_stage3
      rally  : rally

    返回各组的 rmse_fz 和 peak_err，用于论文 Figure / Table。
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        stage = str(r["stage"]).lower()
        if "fatigue" in stage:
            groups["fatigue"].append(r)
        elif "rally" in stage:
            groups["rally"].append(r)
        elif "stage" in stage:
            groups["fresh"].append(r)
        else:
            groups["other"].append(r)

    # 细粒度分组（stage1/2/3 分开）
    for r in records:
        groups[str(r["stage"])].append(r)

    result = {}
    for group_name, recs in groups.items():
        if not recs:
            continue
        p_fz = np.concatenate([r["pred_fz"]   for r in recs])
        t_fz = np.concatenate([r["target_fz"] for r in recs])
        ss_res = float(np.sum((t_fz - p_fz) ** 2))
        ss_tot = float(np.sum((t_fz - t_fz.mean()) ** 2))
        result[group_name] = {
            "n":         len(recs),
            "rmse_fz":   round(float(np.sqrt(np.mean((p_fz - t_fz)**2))), 4),
            "r2_fz":     round(float(1.0 - ss_res / (ss_tot + 1e-12)), 4),
            "peak_err":  round(float(np.mean([r["peak_err"] for r in recs])), 4),
        }

    return result


# ---------------------------------------------------------------------------
# 打印汇总表（直接从 summary.json）
# ---------------------------------------------------------------------------

def print_summary_table(summary_path: str) -> None:
    d = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    per_fold = d.get("per_fold", {})
    mean     = d.get("mean",     {})
    std      = d.get("std",      {})

    print("\n" + "=" * 70)
    print("LOSO 汇总（论文 Table）")
    print("=" * 70)
    print(f"{'Subject':<12}", end="")

    keys = ["rmse_fz", "r2_fz", "peak_err_bw", "peak_timing_fr", "n_test"]
    for k in keys:
        print(f"  {k:<18}", end="")
    print()
    print("-" * 70)

    for sub, res in sorted(per_fold.items()):
        print(f"{sub:<12}", end="")
        for k in keys:
            v = res.get(k, float("nan"))
            print(f"  {v:<18.4f}", end="")
        print()

    print("-" * 70)
    print(f"{'Mean':<12}", end="")
    for k in keys:
        v = mean.get(k, float("nan"))
        s = std.get(k, 0)
        print(f"  {v:.4f}±{s:.4f}    ", end="")
    print()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BadmintonGRF 模型评估")
    p.add_argument("--loso_splits",   default=None)
    p.add_argument("--model_path",    default=None)
    p.add_argument("--test_subject",  default=None)
    p.add_argument("--cameras",       nargs="+", type=int, default=[2])
    p.add_argument("--fz_only",       action="store_true")
    p.add_argument("--fatigue_analysis", action="store_true",
                   help="输出各疲劳阶段的预测误差（论文 E3）")
    p.add_argument("--summary_path",  default=None,
                   help="直接打印已有 summary.json，不重新推理")
    p.add_argument("--out",           default=None,
                   help="结果保存路径（默认与 model_path 同目录）")
    # 模型超参（需与训练时一致）
    p.add_argument("--hidden_dim",     type=int,   default=128)
    p.add_argument("--num_layers",     type=int,   default=2)
    p.add_argument("--dropout",        type=float, default=0.0)
    p.add_argument("--unidirectional", action="store_true")
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_level",      default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 直接打印 summary
    if args.summary_path:
        print_summary_table(args.summary_path)
        return

    if not args.model_path or not args.loso_splits or not args.test_subject:
        log.error("需要 --model_path、--loso_splits、--test_subject")
        raise SystemExit(1)

    # 数据集
    _, test_ds = build_loso_datasets(
        loso_splits_path = args.loso_splits,
        test_subject     = args.test_subject,
        cameras          = args.cameras,
        predict_fz_only  = args.fz_only,
        augment_train    = False,
    )
    _kw = build_loader_kwargs(
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=None,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **_kw)
    log.info("测试集大小：%d", len(test_ds))

    # 模型
    device     = torch.device(args.device)
    output_dim = 1 if args.fz_only else 3
    model = LSTMBaseline(
        input_dim     = INPUT_DIM,
        hidden_dim    = args.hidden_dim,
        output_dim    = output_dim,
        num_layers    = args.num_layers,
        dropout       = args.dropout,
        bidirectional = not args.unidirectional,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    log.info("已加载模型：%s", args.model_path)

    # 评估
    result = run_evaluation(model, test_loader, device)
    metrics = result["metrics"]
    records = result["records"]

    log.info("\n评估结果（%s）：", args.test_subject)
    for k, v in metrics.items():
        log.info("  %-25s %s", k, v)

    # 疲劳分析
    if args.fatigue_analysis:
        fa = fatigue_analysis(records)
        log.info("\n疲劳分析（各阶段 rmse_fz）：")
        for stage, m in sorted(fa.items()):
            log.info("  %-25s n=%-4d  rmse=%.4f  r2=%.3f  peak_err=%.4f",
                     stage, m["n"], m["rmse_fz"], m["r2_fz"], m["peak_err"])
        metrics["fatigue_analysis"] = fa

    # 保存
    out_path = Path(args.out) if args.out else Path(args.model_path).parent / "eval_results.json"
    out_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("结果已保存：%s", out_path)


if __name__ == "__main__":
    main()