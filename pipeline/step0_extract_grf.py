"""
step0_extract_grf.py
====================
Extract GRF data from mocap .c3d files for ALL subjects in the dataset.

Reads:  {root}/sub_XXX/mocap/{trial}_mocap.c3d
Writes: {root}/sub_XXX/labels/{trial}_grf.npy

Output .npy structure (load with np.load(path, allow_pickle=True).item()):
{
    'timestamps':    ndarray (N,)       -- seconds, 0-based, 1000 Hz
    'combined': {
        'forces':    ndarray (N, 3)     -- [Fx, Fy, Fz] sum of all plates (N)
        'moments':   ndarray (N, 3)     -- [Mx, My, Mz] sum of all plates (N·mm)
    },
    'force_plate_1': { 'forces': (N,3), 'moments': (N,3) },
    'force_plate_2': { ... },
    'force_plate_3': { ... },
    'force_plate_4': { ... },
    'metadata': {
        'trial', 'subject', 'analog_rate', 'point_rate',
        'num_samples', 'duration_sec',
        'channel_layout',          -- which c3d channels were used
        'peak_forces_per_plate',   -- list of 4 peak |Fz| values (N)
        'combined_peak_Fz_N',
        'extraction_time',
    }
}

Note: Fz is stored as-is (negative = downward, Vicon convention).
      To get positive-downward, negate: Fz_pos = -data['combined']['forces'][:, 2]

Usage:
    python step0_extract_grf.py
    python step0_extract_grf.py --subject sub_003
    python step0_extract_grf.py --force   # re-extract all
    python step0_extract_grf.py --dry-run # preview only
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import c3d
except ImportError:
    print("[ERROR] python-c3d not installed.")
    print("  Run: pip install c3d")
    sys.exit(1)

_DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")


# ─────────────────────────────────────────────────────────────
#  C3D channel layout
#
#  Channels 0-23 contain the calibrated force/moment data:
#    plate 1:  0=Fx1  1=Fy1  2=Fz1  3=Mx1  4=My1  5=Mz1
#    plate 2:  6=Fx2  7=Fy2  8=Fz2  9=Mx2 10=My2 11=Mz2
#    plate 3: 12=Fx3 13=Fy3 14=Fz3 15=Mx3 16=My3 17=Mz3
#    plate 4: 18=Fx4 19=Fy4 20=Fz4 21=Mx4 22=My4 23=Mz4
#
#  Channels 24-55 are Raw duplicates (4 sets of 8 raw channels).
#  We only use channels 0-23.
# ─────────────────────────────────────────────────────────────

NUM_PLATES   = 4
CHANS_PER_PLATE = 6   # Fx, Fy, Fz, Mx, My, Mz
FIRST_PLATE_CHAN = 0  # calibrated data starts at channel 0

CHANNEL_LAYOUT = {
    1: {"Fx": 0,  "Fy": 1,  "Fz": 2,  "Mx": 3,  "My": 4,  "Mz": 5},
    2: {"Fx": 6,  "Fy": 7,  "Fz": 8,  "Mx": 9,  "My": 10, "Mz": 11},
    3: {"Fx": 12, "Fy": 13, "Fz": 14, "Mx": 15, "My": 16, "Mz": 17},
    4: {"Fx": 18, "Fy": 19, "Fz": 20, "Mx": 21, "My": 22, "Mz": 23},
}


# ─────────────────────────────────────────────────────────────
#  Layout detection (same as scan_dataset.py)
# ─────────────────────────────────────────────────────────────

def _has_subs(path: Path) -> bool:
    try:
        return path.exists() and any(
            d.is_dir() and d.name.startswith("sub_") for d in path.iterdir()
        )
    except Exception:
        return False


def find_subjects_base(root: Path) -> Path:
    if _has_subs(root):
        return root
    data_dir = root / "data"
    if _has_subs(data_dir):
        return data_dir
    pilot_dir = root / "data" / "pilot"
    if _has_subs(pilot_dir):
        return pilot_dir
    raise FileNotFoundError(
        f"No sub_XXX directories found under {root}\n"
        f"  Tried: {root}, {data_dir}, {pilot_dir}"
    )


# ─────────────────────────────────────────────────────────────
#  Core extraction: one c3d file -> dict
# ─────────────────────────────────────────────────────────────

def extract_grf_from_c3d(c3d_path: Path, trial: str, subject: str) -> dict:
    """
    Parse a single .c3d file and return the GRF data dict.
    Raises on unrecoverable errors.
    """
    with open(c3d_path, 'rb') as f:
        reader = c3d.Reader(f)

        analog_rate  = reader.header.frame_rate * reader.header.analog_per_frame
        point_rate   = float(reader.header.frame_rate)
        num_frames   = reader.header.last_frame - reader.header.first_frame + 1

        # collect all analog frames
        analog_frames = []
        for _, _, analog in reader.read_frames():
            # analog shape: (num_analog_channels, samples_per_frame)
            analog_frames.append(analog)

    # analog_frames: list of (C, S) arrays -> stack to (total_samples, C)
    analog_all = np.hstack(analog_frames).T   # (N_samples, C)

    N = len(analog_all)
    analog_rate_actual = float(analog_rate)

    # sanity: need at least 24 channels
    if analog_all.shape[1] < 24:
        raise ValueError(
            f"Expected >=24 analog channels, got {analog_all.shape[1]}"
        )

    # timestamps (0-based, seconds)
    timestamps = np.arange(N) / analog_rate_actual

    # extract per-plate arrays
    plates = {}
    combined_forces  = np.zeros((N, 3), dtype=np.float64)
    combined_moments = np.zeros((N, 3), dtype=np.float64)
    peak_fz_per_plate = []

    for plate_num, chans in CHANNEL_LAYOUT.items():
        forces  = analog_all[:, [chans["Fx"], chans["Fy"], chans["Fz"]]].astype(np.float64)
        moments = analog_all[:, [chans["Mx"], chans["My"], chans["Mz"]]].astype(np.float64)

        plates[f"force_plate_{plate_num}"] = {
            "forces":  forces,
            "moments": moments,
        }
        combined_forces  += forces
        combined_moments += moments
        peak_fz_per_plate.append(float(np.max(np.abs(forces[:, 2]))))

    combined_peak_fz = float(np.max(np.abs(combined_forces[:, 2])))

    result = {
        "timestamps": timestamps.astype(np.float64),
        "combined": {
            "forces":  combined_forces,
            "moments": combined_moments,
        },
        **plates,
        "metadata": {
            "trial":                 trial,
            "subject":               subject,
            "analog_rate":           analog_rate_actual,
            "point_rate":            point_rate,
            "num_samples":           N,
            "duration_sec":          round(float(timestamps[-1]), 4),
            "channel_layout":        {
                str(p): chans for p, chans in CHANNEL_LAYOUT.items()
            },
            "peak_forces_per_plate": peak_fz_per_plate,
            "combined_peak_Fz_N":    combined_peak_fz,
            "extraction_time":       datetime.now().isoformat(),
        },
    }
    return result


# ─────────────────────────────────────────────────────────────
#  Batch extraction
# ─────────────────────────────────────────────────────────────

def run_extraction(
    root: Path,
    subject_filter: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> Dict:

    SEP = "=" * 64
    print(f"\n{SEP}")
    print(f"  step0_extract_grf.py")
    print(f"  Root    : {root}")
    print(f"  Force   : {force}   DryRun : {dry_run}")
    print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{SEP}\n")

    subjects_base = find_subjects_base(root)
    print(f"  Subjects base: {subjects_base}\n")

    # discover subjects
    all_subjects = sorted(
        d.name for d in subjects_base.iterdir()
        if d.is_dir() and d.name.startswith("sub_")
    )
    if subject_filter:
        all_subjects = [s for s in all_subjects if s == subject_filter]
        if not all_subjects:
            print(f"[ERROR] Subject '{subject_filter}' not found.")
            return {}

    # collect all (subject, trial, c3d_path, npy_path) tuples
    tasks: List[Tuple[str, str, Path, Path]] = []
    for subject in all_subjects:
        sub_dir   = subjects_base / subject
        mocap_dir = sub_dir / "mocap"
        labels_dir = sub_dir / "labels"

        if not mocap_dir.exists():
            print(f"  [{subject}] no mocap/ directory, skipping.")
            continue

        for c3d_path in sorted(mocap_dir.glob(f"{subject}_*_mocap.c3d")):
            trial    = c3d_path.stem.replace("_mocap", "")
            npy_path = labels_dir / f"{trial}_grf.npy"
            tasks.append((subject, trial, c3d_path, npy_path))

    total   = len(tasks)
    skipped = sum(1 for _, _, _, npy in tasks if npy.exists() and not force)
    to_do   = total - skipped

    print(f"  Total tasks  : {total}")
    print(f"  Already done : {skipped}  (use --force to re-extract)")
    print(f"  To process   : {to_do}")
    if dry_run:
        print(f"\n  [DRY RUN] No files will be written.\n")

    results = []
    n_ok  = 0
    n_skip = 0
    n_fail = 0

    current_subject = None

    for subject, trial, c3d_path, npy_path in tasks:

        # section header per subject
        if subject != current_subject:
            if current_subject is not None:
                print()
            print(f"  [{subject}]")
            current_subject = subject

        # skip if already extracted
        if npy_path.exists() and not force:
            print(f"    skip  {trial}  (already exists)")
            n_skip += 1
            results.append({"trial": trial, "subject": subject,
                            "status": "skipped", "npy": str(npy_path)})
            continue

        # dry run
        if dry_run:
            print(f"    would extract  {trial}")
            results.append({"trial": trial, "subject": subject,
                            "status": "dry_run", "c3d": str(c3d_path)})
            continue

        # extract
        try:
            data = extract_grf_from_c3d(c3d_path, trial, subject)

            # create labels dir and save
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, data)

            dur  = data["metadata"]["duration_sec"]
            peak = data["metadata"]["combined_peak_Fz_N"]
            print(f"    OK    {trial:<44s}  {dur:6.1f}s  peak={peak:.0f}N")
            n_ok += 1
            results.append({
                "trial":   trial,
                "subject": subject,
                "status":  "ok",
                "npy":     str(npy_path),
                "duration_sec":       dur,
                "combined_peak_Fz_N": peak,
            })

        except Exception as e:
            print(f"    FAIL  {trial}")
            print(f"          {e}")
            n_fail += 1
            results.append({
                "trial":   trial,
                "subject": subject,
                "status":  "failed",
                "error":   str(e),
            })

    print(f"\n{SEP}")
    print(f"  DONE  ok={n_ok}  skip={n_skip}  fail={n_fail}  total={total}")
    print(f"{SEP}\n")

    return {
        "run_time":   datetime.now().isoformat(),
        "root":       str(root),
        "total":      total,
        "ok":         n_ok,
        "skipped":    n_skip,
        "failed":     n_fail,
        "results":    results,
    }


# ─────────────────────────────────────────────────────────────
#  Save run log
# ─────────────────────────────────────────────────────────────

def save_log(report: Dict, root: Path):
    out_dir  = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"step0_log_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Log saved: {out_path}")


# ─────────────────────────────────────────────────────────────
#  Entry
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BadmintonGRF step0: extract GRF from c3d")
    parser.add_argument("--root",    type=str, default="",
                        help="Dataset root (default: env BADMINTON_DATA_ROOT or <repo>/data)")
    parser.add_argument("--subject", type=str, default=None,
                        help="Process only this subject (e.g. sub_003)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-extract even if .npy already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be done, write nothing")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving the log JSON")
    args = parser.parse_args()

    root_str = (args.root or os.environ.get("BADMINTON_DATA_ROOT", "").strip()
                or _DEFAULT_DATA_ROOT)
    if not root_str:
        print("[ERROR] --root required or set BADMINTON_DATA_ROOT")
        sys.exit(1)

    try:
        root   = Path(root_str)
        report = run_extraction(
            root,
            subject_filter=args.subject,
            force=args.force,
            dry_run=args.dry_run,
        )
        if report and not args.no_save and not args.dry_run:
            save_log(report, root)
    except Exception:
        print("\n[FATAL ERROR]")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()