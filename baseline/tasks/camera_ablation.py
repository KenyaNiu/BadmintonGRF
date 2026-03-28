"""
e_ablation_cameras.py
=====================
消融实验：各相机角度对 GRF 预测的贡献。

对每个相机单独训练 E1 模型，对比：
  - 8台相机各自的 LOSO 均值 r²
  - Late Fusion vs 最优单相机

用法：
  python -m baseline ablation \
      --loso_splits /media/nky/Lenovo/data/reports/loso_splits.json \
      --fz_only --epochs 300 --save_report \
      --batch_size 256 --hidden_dim 128 --tcn_channels 128 \
      --patience 50
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("e_ablation")

ALL_CAMERAS = [1, 2, 3, 4, 5, 6, 7, 8]


def run_single_camera(cam: int, args, base_out_dir: Path) -> dict:
    """启动子进程跑单相机 E1 训练"""
    out_dir = base_out_dir / f"cam{cam}"
    cmd = [
        sys.executable, "-m", "baseline.train",
        "--loso_splits",  args.loso_splits,
        "--cameras",      str(cam),
        "--fz_only" if args.fz_only else "",
        "--epochs",       str(args.epochs),
        "--patience",     str(args.patience),
        "--batch_size",   str(args.batch_size),
        "--hidden_dim",   str(args.hidden_dim),
        "--tcn_channels", str(args.tcn_channels),
        "--lr",           str(args.lr),
        "--out_dir",      str(out_dir),
        "--save_report",
    ]
    cmd = [c for c in cmd if c]  # 去掉空字符串
    log.info("  启动 cam%d 训练...", cam)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.warning("  cam%d 训练失败", cam)
        return {}

    # 读取 summary
    summary_files = list(out_dir.glob("*/summary.json"))
    if not summary_files:
        return {}
    data = json.loads(summary_files[0].read_text(encoding="utf-8"))
    mean = data.get("mean", {})
    log.info("  cam%d → r²=%.3f  rmse=%.4f  peak_err=%.4f×BW",
             cam,
             mean.get("r2_fz", float("nan")),
             mean.get("rmse_fz", float("nan")),
             mean.get("peak_err_bw", float("nan")))
    return mean


def run_ablation(args):
    logging.basicConfig(level="INFO",
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cameras = args.cameras or ALL_CAMERAS
    base_out = Path(args.out_dir) / "ablation_cameras"
    base_out.mkdir(parents=True, exist_ok=True)

    results = {}
    for cam in cameras:
        results[f"cam{cam}"] = run_single_camera(cam, args, base_out)

    # 打印对比表
    log.info("\n%s\n相机消融实验汇总\n%s", "="*60, "="*60)
    log.info("  %-8s  %-8s  %-10s  %-12s", "Camera", "r²", "RMSE", "peak_err×BW")
    for cam_name, metrics in sorted(results.items()):
        log.info("  %-8s  %-8.3f  %-10.4f  %-12.4f",
                 cam_name,
                 metrics.get("r2_fz", float("nan")),
                 metrics.get("rmse_fz", float("nan")),
                 metrics.get("peak_err_bw", float("nan")))

    if args.save_report:
        out_file = base_out / "camera_ablation_summary.json"
        out_file.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        log.info("报告：%s", out_file)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--loso_splits",   required=True)
    p.add_argument("--cameras",       nargs="+", type=int, default=None)
    p.add_argument("--fz_only",       action="store_true")
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--patience",      type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--hidden_dim",    type=int,   default=128)
    p.add_argument("--tcn_channels",  type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--out_dir",       default="runs")
    p.add_argument("--save_report",   action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    run_ablation(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
