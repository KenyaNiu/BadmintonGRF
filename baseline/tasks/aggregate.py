"""
Merge multiple ``summary.json`` files into one comparison table (Markdown + CSV).

Example::

    python -m baseline aggregate \\
        --inputs \\
            runs/a_tcn/summary.json \\
            runs/b_gru/summary.json \\
            runs/c_stgcn/summary.json \\
        --out_csv runs/benchmark_table.csv \\
        --out_md runs/benchmark_table.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _mean_metrics(d: Dict[str, Any]) -> Dict[str, float]:
    m = d.get("mean", {})
    out = {}
    for k in ("r2_fz", "r2", "rmse_fz", "rmse", "peak_err_bw", "peak_err", "peak_timing_fr", "peak_timing"):
        if k in m and isinstance(m[k], (int, float)):
            out[k] = float(m[k])
    # Prefer canonical names
    return {
        "r2": out.get("r2_fz", out.get("r2")),
        "rmse_bw": out.get("rmse_fz", out.get("rmse")),
        "peak_err_bw": out.get("peak_err_bw", out.get("peak_err")),
        "peak_timing_fr": out.get("peak_timing_fr", out.get("peak_timing")),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="summary.json paths")
    p.add_argument("--labels", nargs="*", default=None, help="Row labels (default: method field or path stem)")
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--out_md", type=str, default=None)
    p.add_argument(
        "--write-canonical",
        action="store_true",
        help="Also write summary_canonical.json beside each input (experiment label = row label).",
    )
    args = p.parse_args(argv)

    rows: List[Dict[str, Any]] = []
    for i, path in enumerate(args.inputs):
        pth = Path(path)
        d = json.loads(pth.read_text(encoding="utf-8"))
        label = None
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        if not label:
            label = d.get("method") or d.get("experiment") or pth.parent.name
        mm = _mean_metrics(d)
        rows.append({"label": label, **{k: v for k, v in mm.items() if v is not None}})
        if args.write_canonical:
            from baseline.tasks.canonical import write_summary_canonical

            write_summary_canonical(summary_path=pth, experiment=str(label))

    cols = ["label", "r2", "rmse_bw", "peak_err_bw", "peak_timing_fr"]
    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"Wrote {outp}")

    if args.out_md:
        outp = Path(args.out_md)
        outp.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c, "")
                if c == "label":
                    cells.append(str(v))
                elif isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {outp}")

    if not args.out_csv and not args.out_md:
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()
