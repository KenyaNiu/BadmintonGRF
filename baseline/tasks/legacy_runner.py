"""
Experiment runner (backward-compatible CLI).

Prefer::

    python -m baseline train --method tcn_bilstm --loso_splits ... --run_dir ...
    python -m baseline fuse --base_run_dir ... --loso_splits ...

Legacy ``e1``…``e5`` map to ``baseline train`` / ``baseline fuse`` / ``baseline fatigue``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from baseline.tasks.canonical import write_summary_canonical


def _run(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(f"Command failed (exit={p.returncode}). See log: {log_path}")


def _require_file(p: Path, label: str) -> None:
    if not p.exists():
        raise SystemExit(f"Missing {label}: {p}")


def _timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _invoke_train_cli(args: argparse.Namespace, extra: List[str], log_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "baseline",
        "train",
        "--method",
        args.method,
        "--loso_splits",
        args.loso_splits,
        "--run_dir",
        args.run_dir,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--save_report",
    ]
    if args.fz_only:
        cmd.append("--fz_only")
    cmd += extra
    _run(cmd, log_path)


def run_e1(args, paths) -> Path:
    _invoke_train_cli(
        argparse.Namespace(
            method="tcn_bilstm",
            loso_splits=args.loso_splits,
            run_dir=args.e1_run_dir,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            fz_only=args.fz_only,
        ),
        [
            "--hidden_dim",
            str(args.hidden_dim),
            "--tcn_channels",
            str(args.tcn_channels),
        ],
        paths.log_dir / "e1.log",
    )
    summ = Path(args.e1_run_dir) / "summary.json"
    _require_file(summ, "e1 summary.json")
    write_summary_canonical(summary_path=summ, experiment="E1")
    return Path(args.e1_run_dir)


def run_e2(args, paths) -> Path:
    _require_file(Path(args.e1_run_dir) / "summary.json", "E1 summary (for e2)")
    cmd = [
        sys.executable,
        "-m",
        "baseline",
        "fuse",
        "--loso_splits",
        args.loso_splits,
        "--base_run_dir",
        args.e1_run_dir,
        "--save_report",
    ]
    if args.fz_only:
        cmd.append("--fz_only")
    _run(cmd, paths.log_dir / "e2.log")
    summ = Path(args.e1_run_dir) / "late_fusion" / "summary.json"
    _require_file(summ, "e2 summary.json")
    write_summary_canonical(summary_path=summ, experiment="E2")
    return summ.parent


def run_e3(args, paths) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "baseline",
        "fatigue",
        "--loso_splits",
        args.loso_splits,
        "--e1_runs_dir",
        args.e1_run_dir,
        "--save_report",
    ]
    if args.fz_only:
        cmd.append("--fz_only")
    _run(cmd, paths.log_dir / "e3.log")
    out_dir = Path(args.e1_run_dir) / "e3_fatigue"
    _require_file(out_dir / "per_fold_stage.json", "E3 per_fold_stage.json")
    return out_dir


def run_e4(args, paths) -> Path:
    extra: List[str] = ["--hidden_dim", str(args.hidden_dim)]
    _invoke_train_cli(
        argparse.Namespace(
            method="stgcn_transformer",
            loso_splits=args.loso_splits,
            run_dir=args.e4_run_dir,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            fz_only=args.fz_only,
        ),
        extra,
        paths.log_dir / "e4.log",
    )
    summ = Path(args.e4_run_dir) / "summary.json"
    _require_file(summ, "e4 summary.json")
    write_summary_canonical(summary_path=summ, experiment="E4")
    return Path(args.e4_run_dir)


def run_e5(args, paths) -> Path:
    _require_file(Path(args.e4_run_dir) / "summary.json", "E4 summary (for e5)")
    cmd = [
        sys.executable,
        "-m",
        "baseline",
        "fuse",
        "--loso_splits",
        args.loso_splits,
        "--base_run_dir",
        args.e4_run_dir,
        "--save_report",
    ]
    if args.fz_only:
        cmd.append("--fz_only")
    _run(cmd, paths.log_dir / "e5.log")
    summ = Path(args.e4_run_dir) / "late_fusion" / "summary.json"
    _require_file(summ, "e5 summary.json")
    write_summary_canonical(summary_path=summ, experiment="E5")
    return summ.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BadmintonGRF experiment runner (legacy aliases)")
    p.add_argument("--runs_root", default=None, help="Base directory for logs (default: ./runs)")

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--loso_splits", required=True)
        sp.add_argument("--fz_only", action="store_true", default=True)

    e1 = sub.add_parser("e1", help="Legacy E1 → tcn_bilstm")
    add_common(e1)
    e1.add_argument("--e1_run_dir", required=True)
    e1.add_argument("--epochs", type=int, default=500)
    e1.add_argument("--patience", type=int, default=80)
    e1.add_argument("--batch_size", type=int, default=256)
    e1.add_argument("--hidden_dim", type=int, default=128)
    e1.add_argument("--tcn_channels", type=int, default=128)
    e1.add_argument("--lr", type=float, default=1e-3)

    e2 = sub.add_parser("e2", help="Legacy E2 → late_fusion on tcn_bilstm run")
    add_common(e2)
    e2.add_argument("--e1_run_dir", required=True)

    e3 = sub.add_parser("e3", help="E3 fatigue analysis")
    add_common(e3)
    e3.add_argument("--e1_run_dir", required=True)

    e4 = sub.add_parser("e4", help="Legacy E4 → stgcn_transformer")
    add_common(e4)
    e4.add_argument("--e4_run_dir", required=True)
    e4.add_argument("--e1_run_dir", default=None)
    e4.add_argument("--epochs", type=int, default=500)
    e4.add_argument("--patience", type=int, default=80)
    e4.add_argument("--batch_size", type=int, default=128)
    e4.add_argument("--hidden_dim", type=int, default=128)
    e4.add_argument("--lr", type=float, default=5e-4)

    e5 = sub.add_parser("e5", help="Legacy E5 → late_fusion on stgcn run")
    add_common(e5)
    e5.add_argument("--e4_run_dir", required=True)
    e5.add_argument("--e1_run_dir", default=None)

    chain = sub.add_parser("chain_e1_e2_e3", help="E1 then E2 then E3")
    add_common(chain)
    chain.add_argument("--e1_run_dir", required=True)
    chain.add_argument("--epochs", type=int, default=500)
    chain.add_argument("--patience", type=int, default=80)
    chain.add_argument("--batch_size", type=int, default=256)
    chain.add_argument("--hidden_dim", type=int, default=128)
    chain.add_argument("--tcn_channels", type=int, default=128)
    chain.add_argument("--lr", type=float, default=1e-3)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runs_root = Path(args.runs_root) if args.runs_root else Path("runs")
    ts = _timestamp()
    paths = type("P", (), {"log_dir": runs_root / "logs" / ts})()
    os.makedirs(paths.log_dir, exist_ok=True)
    (paths.log_dir / "invocation.json").write_text(
        json.dumps({"argv": sys.argv, "cwd": os.getcwd()}, indent=2),
        encoding="utf-8",
    )

    if args.cmd == "e1":
        run_e1(args, paths)
    elif args.cmd == "e2":
        run_e2(args, paths)
    elif args.cmd == "e3":
        run_e3(args, paths)
    elif args.cmd == "e4":
        run_e4(args, paths)
    elif args.cmd == "e5":
        run_e5(args, paths)
    elif args.cmd == "chain_e1_e2_e3":
        run_e1(args, paths)
        run_e2(args, paths)
        run_e3(args, paths)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
