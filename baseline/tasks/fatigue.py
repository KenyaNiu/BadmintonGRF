"""
Fatigue protocol analysis (legacy paper label: E3)
=================================================

功能：
  1. 用 E1 训练好的模型，对每个被试按 stage1/stage2/stage3 分别预测
  2. 计算每个疲劳阶段的 GRF 指标：
     - peak_force_bw   峰值力（体重倍数）
     - loading_rate_bw 加载率（峰值力/到达峰值的帧数）
     - peak_timing_fr  峰值时序误差
     - r2_fz / rmse_fz 预测精度
  3. 输出 per-stage 对比表格 + 统计显著性（Kruskal-Wallis）
  4. 保存可直接贴入论文的 CSV

用法：
  python -m baseline fatigue \\
      --loso_splits /media/nky/Lenovo/data/reports/loso_splits.json \
      --e1_runs_dir  runs/20260316_114243_lstm_tcn_camall \
      --fz_only --save_report
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from baseline.impact_dataset import INPUT_DIM, build_features
from baseline.models.tcn_lstm import LSTMBaseline

log = logging.getLogger("e3_fatigue")

STAGES = ["stage1", "stage2", "stage3"]


# ---------------------------------------------------------------------------
# 辅助：从文件名解析 stage
# ---------------------------------------------------------------------------

def parse_stage(path: str) -> Optional[str]:
    """
    从路径名提取 stage 标签。
    支持：
      sub_001_stage1_01_cam2_impact_001.npz  → stage1
      sub_001_fatigue_stage2_01_...          → stage2
      sub_001_rally_01_...                   → rally（不属于疲劳分期）
    """
    name = Path(path).stem
    m = re.search(r"(fatigue_)?(stage\d)", name)
    if m:
        return m.group(2)
    if "rally" in name:
        return "rally"
    return None


# ---------------------------------------------------------------------------
# 核心：按 stage 分组预测
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_by_stage(
    model:    LSTMBaseline,
    paths:    List[str],
    fz_only:  bool,
    device:   torch.device,
) -> Dict[str, Dict]:
    """
    对 test_paths 里的每个片段做推理，按 stage 汇总指标。
    返回 {stage: {metrics}}
    """
    model.eval()

    by_stage = defaultdict(lambda: {
        "preds":   [],
        "targets": [],
        "evs":     [],
        "peak_forces_true":  [],
        "peak_forces_pred":  [],
        "loading_rates_true": [],
    })

    for p in paths:
        stage = parse_stage(p)
        if stage is None or stage == "rally":
            continue
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue

        kps  = d["keypoints_norm"].astype(np.float32)
        sc   = d["scores"].astype(np.float32)
        grf  = d["grf_normalized"].astype(np.float32)
        ev   = int(d["ev_idx"])

        pose   = torch.from_numpy(build_features(kps, sc)).unsqueeze(0).to(device)
        target = grf[:, 2:3] if fz_only else grf
        pred   = model(pose).squeeze(0).cpu().numpy()

        T  = len(pred)
        ws = max(0, ev - 30)
        we = min(T, ev + 30)

        p_fz = pred[:, -1]
        t_fz = target[:, -1] if target.ndim > 1 else target

        peak_true = float(t_fz[ws:we].max())
        peak_pred = float(p_fz[ws:we].max())

        # 加载率 = 峰值力 / 到达峰值所需帧数（从 ev 开始）
        peak_idx  = ws + int(np.argmax(t_fz[ws:we]))
        frames_to_peak = max(peak_idx - ev, 1)
        loading_rate  = peak_true / frames_to_peak

        by_stage[stage]["preds"].append(pred)
        by_stage[stage]["targets"].append(target)
        by_stage[stage]["evs"].append(ev)
        by_stage[stage]["peak_forces_true"].append(peak_true)
        by_stage[stage]["peak_forces_pred"].append(peak_pred)
        by_stage[stage]["loading_rates_true"].append(loading_rate)

    # 计算每个 stage 的汇总指标
    results = {}
    for stage, data in by_stage.items():
        if not data["preds"]:
            continue
        p_arr = np.array(data["preds"]).reshape(-1, data["preds"][0].shape[-1])
        t_arr = np.array(data["targets"]).reshape(-1, data["targets"][0].shape[-1])
        p_fz  = p_arr[:, -1]; t_fz = t_arr[:, -1]

        ss_res = float(np.sum((t_fz - p_fz)**2))
        ss_tot = float(np.sum((t_fz - t_fz.mean())**2))

        # peak timing
        timing_errs = []
        for pred, target, ev in zip(data["preds"], data["targets"], data["evs"]):
            T  = len(pred); ws = max(0,ev-30); we = min(T,ev+30)
            pi = ws + int(np.argmax(pred[ws:we, -1]))
            ti = ws + int(np.argmax((target[:, -1] if target.ndim>1 else target)[ws:we]))
            timing_errs.append(abs(pi - ti))

        results[stage] = {
            "n":                  len(data["preds"]),
            "r2_fz":              float(1 - ss_res / (ss_tot + 1e-12)),
            "rmse_fz":            float(np.sqrt(np.mean((p_fz - t_fz)**2))),
            "peak_force_true_bw": float(np.mean(data["peak_forces_true"])),
            "peak_force_pred_bw": float(np.mean(data["peak_forces_pred"])),
            "peak_force_err_bw":  float(np.mean(np.abs(
                np.array(data["peak_forces_pred"]) -
                np.array(data["peak_forces_true"])))),
            "loading_rate_mean":  float(np.mean(data["loading_rates_true"])),
            "peak_timing_fr":     float(np.mean(timing_errs)) if timing_errs else float("nan"),
        }

    return results


# ---------------------------------------------------------------------------
# 统计显著性
# ---------------------------------------------------------------------------

def kruskal_wallis_test(by_stage_data: Dict[str, List[float]]) -> Dict:
    """对多组数据做 Kruskal-Wallis 检验"""
    try:
        from scipy.stats import kruskal
        groups = [v for k, v in sorted(by_stage_data.items()) if len(v) > 1]
        if len(groups) < 2:
            return {"h_stat": float("nan"), "p_value": float("nan")}
        h, p = kruskal(*groups)
        return {"h_stat": float(h), "p_value": float(p)}
    except Exception as e:
        return {"h_stat": float("nan"), "p_value": float("nan"), "error": str(e)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_e3(args):
    logging.basicConfig(level="INFO",
                        format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device(args.device)
    splits = json.loads(Path(args.loso_splits).read_text(encoding="utf-8"))

    # 每个 fold 的 per-stage 结果
    fold_stage_results = {}

    for test_sub, fold in splits.items():
        log.info("\n%s\nE3 FOLD: %s\n%s", "="*60, test_sub, "="*60)

        ckpt = Path(args.e1_runs_dir) / f"fold_{test_sub}" / "best_model.pth"
        if not ckpt.exists():
            log.warning("  找不到 %s，跳过", ckpt); continue

        model = LSTMBaseline(
            input_dim=INPUT_DIM, hidden_dim=args.hidden_dim,
            output_dim=1 if args.fz_only else 3,
            num_layers=args.num_layers, dropout=args.dropout,
            bidirectional=not args.unidirectional,
            tcn_channels=args.tcn_channels,
        ).to(device)
        model.load_state_dict(
            torch.load(str(ckpt), map_location=device, weights_only=True))

        stage_results = predict_by_stage(
            model, fold["test"], args.fz_only, device)
        fold_stage_results[test_sub] = stage_results

        for stage in sorted(stage_results.keys()):
            r = stage_results[stage]
            log.info("  %s  n=%3d  r²=%.3f  peak_true=%.3f×BW  peak_err=%.3f×BW  LR=%.4f",
                     stage, r["n"], r["r2_fz"],
                     r["peak_force_true_bw"], r["peak_force_err_bw"],
                     r["loading_rate_mean"])

    if not fold_stage_results:
        log.error("所有 fold 均失败"); return

    # 跨被试汇总：每个 stage 的均值
    agg = defaultdict(lambda: defaultdict(list))
    for sub, stage_res in fold_stage_results.items():
        for stage, metrics in stage_res.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k != "n":
                    agg[stage][k].append(v)

    log.info("\n%s\nE3 跨被试 Stage 汇总（论文 Table）\n%s", "="*60, "="*60)
    log.info("  %-10s  %-6s  %-8s  %-12s  %-12s  %-12s",
             "Stage", "n", "r²", "peak_true×BW", "peak_err×BW", "loading_rate")
    for stage in STAGES:
        if stage not in agg: continue
        d = agg[stage]
        log.info("  %-10s  %-6.0f  %-8.3f  %-12.3f  %-12.3f  %-12.4f",
                 stage,
                 np.mean([fold_stage_results[s].get(stage, {}).get("n", 0)
                          for s in fold_stage_results]),
                 np.nanmean(d["r2_fz"]),
                 np.nanmean(d["peak_force_true_bw"]),
                 np.nanmean(d["peak_force_err_bw"]),
                 np.nanmean(d["loading_rate_mean"]))

    # Kruskal-Wallis：peak_force 是否随疲劳显著变化
    peak_by_stage = {
        s: agg[s]["peak_force_true_bw"] for s in STAGES if s in agg
    }
    kw = kruskal_wallis_test(peak_by_stage)
    log.info("\n  Kruskal-Wallis (peak_force across stages): H=%.3f  p=%.4f",
             kw["h_stat"], kw["p_value"])
    if kw["p_value"] < 0.05:
        log.info("  → ✅ 显著差异（p<0.05）：疲劳对 GRF 有显著影响")
    else:
        log.info("  → 无显著差异（p>=0.05）：需要更多数据")

    if args.save_report:
        out_dir = Path(args.e1_runs_dir) / "e3_fatigue"
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        (out_dir / "per_fold_stage.json").write_text(
            json.dumps(fold_stage_results, indent=2,
                       ensure_ascii=False, default=str), encoding="utf-8")

        # CSV（直接贴入论文）
        csv_path = out_dir / "fatigue_table.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "stage", "r2_fz", "rmse_fz",
                "peak_force_true_bw", "peak_force_err_bw",
                "loading_rate_mean", "peak_timing_fr"
            ])
            for stage in STAGES:
                if stage not in agg: continue
                d = agg[stage]
                writer.writerow([
                    stage,
                    f"{np.nanmean(d['r2_fz']):.3f}±{np.nanstd(d['r2_fz']):.3f}",
                    f"{np.nanmean(d['rmse_fz']):.4f}±{np.nanstd(d['rmse_fz']):.4f}",
                    f"{np.nanmean(d['peak_force_true_bw']):.3f}±{np.nanstd(d['peak_force_true_bw']):.3f}",
                    f"{np.nanmean(d['peak_force_err_bw']):.3f}±{np.nanstd(d['peak_force_err_bw']):.3f}",
                    f"{np.nanmean(d['loading_rate_mean']):.4f}±{np.nanstd(d['loading_rate_mean']):.4f}",
                    f"{np.nanmean(d['peak_timing_fr']):.2f}±{np.nanstd(d['peak_timing_fr']):.2f}",
                ])
        log.info("报告：%s", out_dir)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--loso_splits",   required=True)
    p.add_argument("--e1_runs_dir",   required=True)
    p.add_argument("--fz_only",       action="store_true")
    p.add_argument("--hidden_dim",    type=int,   default=128)
    p.add_argument("--num_layers",    type=int,   default=2)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--tcn_channels",  type=int,   default=128)
    p.add_argument("--unidirectional",action="store_true")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_report",   action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    run_e3(parse_args(argv))


if __name__ == "__main__":
    main()
