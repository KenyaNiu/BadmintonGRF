"""
Bundle experiment outputs for paper writing: canonical JSON + wide tables + manifest.

Run after ``train`` / ``fuse`` (or on an existing benchmark directory)::

    python -m baseline paper-export --run-root runs/benchmark_runs_YYYYMMDD_HHMMSS

Writes under ``--out-dir`` (default: same as ``--run-root``):

- ``paper_bundle.json`` — per-method single-view vs fusion metrics + file paths
- ``paper_table_wide.csv`` / ``paper_table_wide.md`` — one row per method, columns for single vs fusion
- Refreshes ``summary_canonical.json`` next to each ``summary.json`` found
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from baseline.tasks.canonical import canonicalize_mean, write_summary_canonical


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _mean_from_summary(d: Dict[str, Any]) -> Dict[str, Any]:
    m = d.get("mean", {})
    return canonicalize_mean(m if isinstance(m, dict) else {})


def collect_method_artifacts(method_dir: Path) -> Dict[str, Any]:
    mid = method_dir.name
    out: Dict[str, Any] = {"method_id": mid, "run_dir": str(method_dir.resolve())}

    cfg = method_dir / "config.json"
    if cfg.exists():
        out["config_path"] = str(cfg.resolve())
        try:
            out["config"] = _read_json(cfg)
        except Exception:
            out["config"] = None

    sv = method_dir / "summary.json"
    if sv.exists():
        out["single_view_summary"] = str(sv.resolve())
        try:
            raw = _read_json(sv)
            out["single_view_mean_canonical"] = _mean_from_summary(raw)
            write_summary_canonical(summary_path=sv, experiment=f"{mid}_single")
            out["single_view_canonical"] = str(sv.with_name("summary_canonical.json").resolve())
        except Exception as e:
            out["single_view_error"] = str(e)

    lf = method_dir / "late_fusion" / "summary.json"
    if lf.exists():
        out["late_fusion_summary"] = str(lf.resolve())
        try:
            raw = _read_json(lf)
            out["late_fusion_mean_canonical"] = _mean_from_summary(raw)
            write_summary_canonical(summary_path=lf, experiment=f"{mid}_fusion")
            out["late_fusion_canonical"] = str(lf.with_name("summary_canonical.json").resolve())
        except Exception as e:
            out["late_fusion_error"] = str(e)
    else:
        out["late_fusion_summary"] = None

    return out


def build_wide_rows(methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for m in methods:
        mid = m.get("method_id", "")
        s = m.get("single_view_mean_canonical") or {}
        f = m.get("late_fusion_mean_canonical") or {}
        rows.append(
            {
                "method_id": mid,
                "r2_single": s.get("r2_fz"),
                "rmse_single": s.get("rmse_fz"),
                "peak_err_single": s.get("peak_err_bw"),
                "timing_single": s.get("peak_timing_fr"),
                "r2_fusion": f.get("r2_fz"),
                "rmse_fusion": f.get("rmse_fz"),
                "peak_err_fusion": f.get("peak_err_bw"),
                "timing_fusion": f.get("peak_timing_fr"),
                "avg_cams_fusion": f.get("avg_cams"),
            }
        )
    return rows


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_wide_md(rows: List[Dict[str, Any]], path: Path) -> None:
    cols = list(rows[0].keys()) if rows else []
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_wide_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv

    cols = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export paper-ready tables + paper_bundle.json from a benchmark run root.")
    p.add_argument("--run-root", required=True, type=Path, help="Directory containing per-method subfolders (e.g. tcn_bilstm/).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write paper_table_* and paper_bundle.json (default: same as --run-root).",
    )
    args = p.parse_args(argv)

    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"Not a directory: {run_root}")

    out_dir = (args.out_dir or run_root).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    methods_data: List[Dict[str, Any]] = []
    for sub in sorted(run_root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith(".") or sub.name in ("logs", "__pycache__"):
            continue
        methods_data.append(collect_method_artifacts(sub))

    rows = build_wide_rows(methods_data)
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "n_methods": len(methods_data),
        "methods": methods_data,
    }

    bundle_path = out_dir / "paper_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote {bundle_path}")

    if rows:
        md_path = out_dir / "paper_table_wide.md"
        csv_path = out_dir / "paper_table_wide.csv"
        write_wide_md(rows, md_path)
        write_wide_csv(rows, csv_path)
        print(f"Wrote {md_path}")
        print(f"Wrote {csv_path}")
    else:
        print("No method subdirectories with summary.json found; only paper_bundle.json written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
