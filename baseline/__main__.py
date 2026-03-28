"""
Unified CLI: ``python -m baseline <command>``.

Recommended workflow (see :file:`README.md` in this package):

1. ``train`` — LOSO training for any registered method
2. ``fuse`` — multi-view late fusion on a finished run
3. ``aggregate`` — merge ``summary.json`` files into a table
4. ``paper-export`` — bundle a benchmark directory into ``paper_bundle.json`` + wide tables
5. ``evaluate`` / ``fatigue`` / ``ablation`` — analysis utilities
6. ``legacy`` — old ``e1``…``e5`` subcommands (compatibility)
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m baseline",
        description="BadmintonGRF baseline toolkit",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="LOSO training (all registered methods)")

    sp = sub.add_parser("fuse", help="Multi-view late fusion on a run directory")
    sp.add_argument("--loso_splits", required=True)
    sp.add_argument("--base_run_dir", required=True)
    sp.add_argument("--cameras", nargs="+", type=int, default=None)
    sp.add_argument("--fz_only", action="store_true", default=True)
    sp.add_argument("--device", default=None)
    sp.add_argument("--save_report", action="store_true")

    sub.add_parser("aggregate", help="Merge summary.json files into CSV/Markdown")

    sub.add_parser("paper-export", help="Paper bundle: canonical JSON + wide table from a benchmark run root")

    sub.add_parser("evaluate", help="Evaluate a checkpoint (TCN+BiLSTM)")

    sub.add_parser("fatigue", help="Stage-stratified fatigue analysis (TCN+BiLSTM run)")

    sub.add_parser("ablation", help="Per-camera training ablation")

    sub.add_parser("legacy", help="Legacy e1–e5 runner (calls subprocess)")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _parser().print_help()
        return 0

    # Dispatch first token without full parse (each subcommand has its own parser)
    cmd = argv[0]
    rest = argv[1:]

    if cmd == "train":
        from baseline.training.cli import main as train_main

        return train_main(rest)

    if cmd == "fuse":
        from baseline.tasks.late_fusion import main as fuse_main

        fuse_main(rest)
        return 0

    if cmd == "aggregate":
        from baseline.tasks.aggregate import main as agg_main

        agg_main(rest)
        return 0

    if cmd in ("paper-export", "paper_export"):
        from baseline.tasks.paper_export import main as paper_main

        return paper_main(rest)

    if cmd == "evaluate":
        from baseline.tasks.evaluate import main as eval_main

        eval_main(rest)
        return 0

    if cmd == "fatigue":
        from baseline.tasks.fatigue import main as fatigue_main

        fatigue_main(rest)
        return 0

    if cmd == "ablation":
        from baseline.tasks.camera_ablation import main as ab_main

        return ab_main(rest)

    if cmd == "legacy":
        from baseline.tasks.legacy_runner import main as legacy_main

        return legacy_main(rest)

    _parser().print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
