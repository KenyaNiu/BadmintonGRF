"""
scan_dataset.py
===============
Scan the full BadmintonGRF dataset.

Auto-detects layout:
  Layout A  sub_XXX directly under root         (默认 <repo>/data)
  Layout B  sub_XXX under root/data/
  Layout C  sub_XXX under root/data/pilot/

Outputs:
  - Console table: every subject x trial x camera
  - {root}/reports/scan_report_{timestamp}.json

Usage:
    python scan_dataset.py                        # default: BADMINTON_DATA_ROOT or <repo>/data
    python scan_dataset.py --root ./data
"""

import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import os
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────
#  Layout detection
# ─────────────────────────────────────────────────────────────

def _has_subs(path: Path) -> bool:
    try:
        return path.exists() and any(
            d.is_dir() and d.name.startswith("sub_") for d in path.iterdir()
        )
    except Exception:
        return False


def detect_layout(root: Path) -> Tuple[str, Path]:
    """
    Returns (layout_name, subjects_base_dir).
    Tries three known layouts in order.
    """
    if _has_subs(root):               # F:/BadmintonGRF/sub_001/...
        return "flat", root

    data_dir = root / "data"
    if _has_subs(data_dir):           # E:/BadmintonGRF/data/sub_003/...  (new)
        return "data_flat", data_dir

    pilot_dir = root / "data" / "pilot"
    if _has_subs(pilot_dir):          # E:/BadmintonGRF/data/pilot/sub_003/...  (old)
        return "nested", pilot_dir

    return "unknown", root


# ─────────────────────────────────────────────────────────────
#  Discovery helpers
# ─────────────────────────────────────────────────────────────

def discover_subjects(subjects_base: Path) -> List[str]:
    return sorted(
        d.name for d in subjects_base.iterdir()
        if d.is_dir() and d.name.startswith("sub_")
    )


def discover_trials_for_subject(subjects_base: Path, subject: str) -> List[str]:
    """Discover trials from mocap/*.c3d; fall back to imu/*.csv."""
    sub_dir = subjects_base / subject
    trials  = set()

    mocap_dir = sub_dir / "mocap"
    if mocap_dir.exists():
        for c3d in mocap_dir.glob(f"{subject}_*_mocap.c3d"):
            trials.add(c3d.stem.replace("_mocap", ""))

    if not trials:
        imu_dir = sub_dir / "imu"
        if imu_dir.exists():
            for csv in imu_dir.glob(f"{subject}_*_imu.csv"):
                trials.add(csv.stem.replace("_imu", ""))

    return sorted(trials)


def get_trial_type(trial: str) -> str:
    """sub_003_fatigue_stage1_01 -> fatigue_stage1"""
    parts = trial.split("_")
    return "_".join(parts[2:-1])


# ─────────────────────────────────────────────────────────────
#  Per-trial check
# ─────────────────────────────────────────────────────────────

def check_trial(subjects_base: Path, subject: str, trial: str) -> Dict:
    sub_dir = subjects_base / subject

    result = {
        "trial":        trial,
        "trial_type":   get_trial_type(trial),
        "has_mocap":    False,
        "mocap_mb":     None,
        "has_imu":      False,
        "cams_found":   [],
        "cams_missing": [],
        "num_cams":     0,
        "grf_npy":      None,
    }

    # mocap
    mocap_path = sub_dir / "mocap" / f"{trial}_mocap.c3d"
    if mocap_path.exists():
        result["has_mocap"] = True
        result["mocap_mb"]  = round(mocap_path.stat().st_size / 1024 / 1024, 1)

    # imu
    result["has_imu"] = (sub_dir / "imu" / f"{trial}_imu.csv").exists()

    # cameras
    video_base = sub_dir / "video"
    for cam in range(1, 9):
        mp4 = video_base / f"cam{cam}" / f"{trial}_cam{cam}.mp4"
        if mp4.exists():
            result["cams_found"].append(cam)
        else:
            result["cams_missing"].append(cam)
    result["num_cams"] = len(result["cams_found"])

    # pre-extracted GRF .npy (step0 output)
    for candidate in [
        sub_dir / "labels" / f"{trial}_grf.npy",
        subjects_base / "labels" / f"{trial}_grf.npy",
    ]:
        if candidate.exists():
            result["grf_npy"] = str(candidate)
            break

    return result


# ─────────────────────────────────────────────────────────────
#  Main scan
# ─────────────────────────────────────────────────────────────

def scan_dataset(root: Path) -> Dict:
    SEP  = "=" * 66
    SEP2 = "-" * 66

    print(f"\n{SEP}")
    print(f"  BadmintonGRF Dataset Scanner  v2")
    print(f"  Root : {root}")
    print(f"  Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{SEP}\n")

    report = {
        "scan_time":     datetime.now().isoformat(),
        "root":          str(root),
        "layout":        None,
        "subjects_base": None,
        "subjects":      {},
        "summary":       {},
    }

    if not root.exists():
        print(f"[ERROR] Root not found: {root}")
        return report

    layout, subjects_base = detect_layout(root)
    report["layout"]        = layout
    report["subjects_base"] = str(subjects_base)

    print(f"  Layout detected : {layout}")
    print(f"  Subjects base   : {subjects_base}\n")

    if layout == "unknown":
        print("[ERROR] No sub_XXX directories found.")
        print("  Use --root to point to the correct directory.")
        return report

    subjects = discover_subjects(subjects_base)
    print(f"  Subjects found  : {len(subjects)}")
    print(f"  {subjects}\n")

    # counters
    all_trial_types        = defaultdict(int)
    total_trials           = 0
    total_with_mocap       = 0
    total_with_imu         = 0
    total_grf_npy          = 0
    cam_present_count      = defaultdict(int)
    trials_all8            = 0
    trials_missing_any_cam = 0

    COL = 44

    for subject in subjects:
        trials   = discover_trials_for_subject(subjects_base, subject)
        sub_info = {"num_trials": len(trials), "trials": {}}

        print(f"  [{subject}]  {len(trials)} trials")
        print(f"    {'trial':<{COL}}  mocap    imu   cameras            missing")
        print(f"    {SEP2}")

        for trial in trials:
            t = check_trial(subjects_base, subject, trial)

            mocap_s = f"{t['mocap_mb']:5.0f}MB" if t["has_mocap"] else "   -- "
            imu_s   = " yes " if t["has_imu"] else "  no "
            cams_s  = ",".join(map(str, t["cams_found"])) if t["cams_found"] else "NONE"
            miss_s  = ",".join(map(str, t["cams_missing"])) if t["cams_missing"] else "-"
            npy_s   = " [npy]" if t["grf_npy"] else ""

            print(f"    {trial:<{COL}}  {mocap_s}  {imu_s}  [{cams_s:<17s}] miss:[{miss_s}]{npy_s}")

            sub_info["trials"][trial] = t
            all_trial_types[t["trial_type"]] += 1
            total_trials           += 1
            total_with_mocap       += int(t["has_mocap"])
            total_with_imu         += int(t["has_imu"])
            total_grf_npy          += int(t["grf_npy"] is not None)
            for cam in t["cams_found"]:
                cam_present_count[cam] += 1
            if t["num_cams"] == 8:
                trials_all8 += 1
            if t["cams_missing"]:
                trials_missing_any_cam += 1

        report["subjects"][subject] = sub_info
        print()

    # ── Summary ──────────────────────────────────────────────
    pct = lambda n: f"{100*n//max(total_trials,1):3d}%"

    print(SEP)
    print("  SUMMARY")
    print(SEP2)
    print(f"  Subjects                 : {len(subjects)}")
    print(f"  Total trials             : {total_trials}")
    print(f"  Trials with mocap (.c3d) : {total_with_mocap}  {pct(total_with_mocap)}")
    print(f"  Trials with IMU   (.csv) : {total_with_imu}  {pct(total_with_imu)}")
    print(f"  Trials with GRF   (.npy) : {total_grf_npy}  (step0 pre-extracted)")
    print(f"  Trials with all 8 cams   : {trials_all8}")
    print(f"  Trials missing >=1 cam   : {trials_missing_any_cam}")
    print()
    print("  Trial type distribution:")
    for ttype, cnt in sorted(all_trial_types.items()):
        print(f"    {ttype:<26s}: {cnt:3d}  {'#' * min(cnt, 40)}")
    print()
    print(f"  Camera completeness  (out of {total_trials} trials):")
    for cam in range(1, 9):
        present = cam_present_count.get(cam, 0)
        missing = total_trials - present
        bar     = "#" * present + "." * missing
        print(f"    cam{cam}: {present:3d} present / {missing:3d} missing  [{bar}]")
    print(SEP + "\n")

    report["summary"] = {
        "num_subjects":           len(subjects),
        "total_trials":           total_trials,
        "trials_with_mocap":      total_with_mocap,
        "trials_with_imu":        total_with_imu,
        "trials_with_grf_npy":    total_grf_npy,
        "trials_all8_cams":       trials_all8,
        "trials_missing_any_cam": trials_missing_any_cam,
        "trial_type_counts":      dict(all_trial_types),
        "cam_present_counts":     dict(cam_present_count),
    }
    return report


# ─────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────

def save_report(report: Dict, root: Path):
    out_dir  = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"scan_report_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Report saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
#  Entry
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BadmintonGRF Dataset Scanner v2")
    parser.add_argument("--root",    type=str, default="",
                        help="Dataset root (default: env BADMINTON_DATA_ROOT or <repo>/data)")
    parser.add_argument("--no-save", action="store_true",
                        help="Print only, skip saving JSON")
    args = parser.parse_args()

    _default = str(Path(__file__).resolve().parent / "data")
    root_str = (args.root or os.environ.get("BADMINTON_DATA_ROOT", "").strip()
                or _default)
    if not root_str:
        print("[ERROR] --root required or set BADMINTON_DATA_ROOT")
        sys.exit(1)
    try:
        root   = Path(root_str)
        report = scan_dataset(root)
        if not args.no_save and report.get("layout") != "unknown":
            save_report(report, root)
    except Exception:
        print("\n[FATAL ERROR]")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()