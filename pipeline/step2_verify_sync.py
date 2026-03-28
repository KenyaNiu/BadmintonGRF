#!/usr/bin/env python3
"""
step2_verify_sync.py — BadmintonGRF Alignment Verification
===========================================================
Loads all *_sync.json files under {base}/sub_XXX/labels/,
computes alignment quality statistics, detects outliers,
and produces publication-quality figures for the ACM MM paper.

Usage
-----
    python step2_verify_sync.py --root ./data
    python step2_verify_sync.py --root ./data --out ./data/verify_output

Outputs
-------
    {out}/
    ├── figures/
    │   ├── fig1_offset_per_subject.pdf      # Box plot: offset distribution per subject
    │   ├── fig2_offset_per_camera.pdf       # Violin: per-camera offset distribution
    │   ├── fig3_consistency_heatmap.pdf     # Subject × Camera intra-consistency
    │   ├── fig4_completion_matrix.pdf       # Annotation progress heatmap
    │   └── fig5_confidence_dist.pdf         # GRF onset confidence distribution
    ├── verify_sync_summary.csv             # Per-trial flat table
    ├── verify_sync_stats.json              # Aggregated statistics
    ├── outliers.json                        # Flagged trials for re-annotation
    ├── whitelist.json                       # User-confirmed OK anomalies (edit manually)
    └── verify_sync_report.txt              # Human-readable report

Author: BadmintonGRF pipeline
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Optional imports with graceful fallback ───────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — figures will be skipped. "
          "pip install matplotlib")

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False
    print("[WARN] pandas not found — CSV output will use built-in writer. "
          "pip install pandas")

# ── Plot style ────────────────────────────────────────────────────────────────
STYLE = {
    # Publication-ready defaults (match ACM MM 2-column aesthetics)
    "figure.dpi":         200,
    "savefig.dpi":        300,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.9,
    "grid.linestyle":     "-",
    "grid.linewidth":     0.6,
    "grid.color":         "#E5E7EB",
    "axes.edgecolor":     "#E5E7EB",
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.0,
}

PALETTE = {
    "good":      "#059669",
    "uncertain": "#F59E0B",
    "bad":       "#DC2626",
    "todo":      "#D1D5DB",
    "blue":      "#2563EB",
    "dark":      "#111827",
    "muted":     "#6B7280",
}

CAM_COLORS = plt.cm.tab10.colors if HAS_MPL else None

# ── Constants ─────────────────────────────────────────────────────────────────
OUTLIER_Z_THRESH   = 3.0    # Z-score threshold per subject-camera group
OUTLIER_IQR_FACTOR = 2.5    # IQR fence multiplier
MIN_GROUP_SIZE     = 3      # Minimum trials to compute group statistics


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_sync(base: Path) -> list[dict]:
    """
    Recursively find and load all *_sync.json files.
    Handles both legacy format (single camera) and new per_cam format.
    Returns list of flat record dicts, one per (trial, camera) pair.
    """
    records = []
    pattern = re.compile(r"^(sub_\d+)_(.+)_sync\.json$")

    for sync_file in sorted(base.rglob("*_sync.json")):
        fname = sync_file.name
        m = pattern.match(fname)
        if not m:
            continue

        subject = m.group(1)
        trial   = fname.replace("_sync.json", "")

        try:
            with open(sync_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Cannot read {sync_file}: {e}")
            continue

        if not isinstance(data, dict):
            print(f"[WARN] Unexpected format in {sync_file}")
            continue

        grf_event_sec = data.get("grf_event_sec")
        quality       = data.get("quality", "good")
        notes         = data.get("notes", "")
        confidence    = data.get("confidence")
        conf_label    = data.get("confidence_label", "")
        annotated_at  = data.get("annotated_at", "")
        tool_version  = data.get("tool_version", "unknown")
        cams_done     = data.get("cams_done", [])

        # ── New per_cam format ─────────────────────────────────────────────
        per_cam = data.get("per_cam", {})
        if per_cam:
            for cam_str, cam_data in per_cam.items():
                try:
                    cam = int(cam_str)
                except ValueError:
                    continue
                if not isinstance(cam_data, dict):
                    continue
                vid_event_sec   = cam_data.get("video_event_sec")
                vid_event_frame = cam_data.get("video_event_frame")
                offset_sec      = cam_data.get("offset_sec")
                cam_annotated   = cam_data.get("annotated_at", annotated_at)

                if offset_sec is None or not _is_finite(offset_sec):
                    continue

                records.append({
                    "subject":          subject,
                    "trial":            trial,
                    "camera":           cam,
                    "grf_event_sec":    grf_event_sec,
                    "video_event_sec":  vid_event_sec,
                    "video_event_frame": vid_event_frame,
                    "offset_sec":       float(offset_sec),
                    "quality":          quality,
                    "notes":            notes,
                    "confidence":       confidence,
                    "confidence_label": conf_label,
                    "annotated_at":     cam_annotated,
                    "tool_version":     tool_version,
                    "cams_done":        cams_done,
                    "sync_file":        str(sync_file),
                    "format":           "per_cam",
                })
        else:
            # ── Legacy single-camera format ────────────────────────────────
            cam            = data.get("video_cam", -1)
            vid_event_sec  = data.get("video_event_sec")
            vid_frame      = data.get("video_event_frame")
            offset_sec     = data.get("offset_sec")

            if offset_sec is None or not _is_finite(offset_sec):
                continue

            records.append({
                "subject":          subject,
                "trial":            trial,
                "camera":           int(cam),
                "grf_event_sec":    grf_event_sec,
                "video_event_sec":  vid_event_sec,
                "video_event_frame": vid_frame,
                "offset_sec":       float(offset_sec),
                "quality":          quality,
                "notes":            notes,
                "confidence":       confidence,
                "confidence_label": conf_label,
                "annotated_at":     annotated_at,
                "tool_version":     tool_version,
                "cams_done":        cams_done,
                "sync_file":        str(sync_file),
                "format":           "legacy",
            })

    return records


def _is_finite(v) -> bool:
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_group_stats(values: list[float]) -> dict:
    """Robust statistics for a group of offset values."""
    if not values:
        return {}
    a = np.array(values, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    return {
        "n":        len(a),
        "mean":     float(np.mean(a)),
        "std":      float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median":   float(np.median(a)),
        "iqr":      float(iqr),
        "q1":       float(q1),
        "q3":       float(q3),
        "min":      float(np.min(a)),
        "max":      float(np.max(a)),
        "range":    float(np.max(a) - np.min(a)),
    }


def load_whitelist(wl_path: Path) -> set:
    """
    Load confirmed-ok entries from whitelist.json.
    Keys are "trial::camera" strings.

    whitelist.json format:
    {
      "confirmed_ok": [
        {"trial": "sub_001_rally_01", "camera": 1,
         "note": "cam started ~78s early; sync stomp at 83.6s. Verified OK."},
        {"trial": "sub_001_rally_01",
         "note": "all cameras started early for this trial (omit camera = all cams)"}
      ]
    }
    """
    if not wl_path.exists():
        return set()
    try:
        with open(wl_path, encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("confirmed_ok", [])
        keys = set()
        for e in entries:
            trial  = e.get("trial", "")
            camera = e.get("camera", None)  # None = all cameras
            if not trial:
                continue
            if camera is not None:
                keys.add(f"{trial}::{int(camera)}")
            else:
                keys.add(f"{trial}::*")  # wildcard: all cameras
        return keys
    except Exception as ex:
        print(f"[WARN] Could not load whitelist.json: {ex}")
        return set()


def is_whitelisted(trial: str, camera: int, whitelist: set) -> bool:
    return (f"{trial}::{camera}" in whitelist or
            f"{trial}::*" in whitelist)


def detect_outliers(records: list[dict],
                    whitelist: set | None = None) -> list[dict]:
    """
    Flag annotation outliers using per-(subject, camera) statistical tests.

    Design rationale
    ----------------
    Each trial's video files are independently started, so offset magnitude
    can legitimately vary between trials for the same camera. However, within
    a continuous recording session, the same camera's start time relative to
    the GRF system should stay approximately constant across trials.

    A large deviation from the group median therefore indicates one of:
      (a) Annotation error — wrong event marked in video or GRF
      (b) Camera was stopped and restarted mid-session
      (c) Camera operator started unusually early/late for this trial

    Cases (b) and (c) are not errors — use whitelist.json to suppress them.
    Case (a) requires re-annotation in align_ui.py.

    Method: IQR fence + Z-score per (subject, camera) group.
    Minimum absolute fence of 5s prevents flagging genuine inter-trial
    timing jitter as outliers.
    """
    if whitelist is None:
        whitelist = set()

    groups = defaultdict(list)
    for r in records:
        groups[(r["subject"], r["camera"])].append(r)

    outliers = []
    for (subj, cam), grp in groups.items():
        if len(grp) < MIN_GROUP_SIZE:
            continue

        offsets  = np.array([r["offset_sec"] for r in grp])
        median_v = float(np.median(offsets))
        q1, q3   = np.percentile(offsets, [25, 75])
        iqr      = q3 - q1
        mean_v   = float(np.mean(offsets))
        std_v    = float(np.std(offsets, ddof=1)) if len(offsets) > 1 else 1e-9

        # Minimum fence of 5s to avoid flagging legitimate timing variation
        fence_half = max(OUTLIER_IQR_FACTOR * iqr, 5.0)
        lo_iqr = q1 - fence_half
        hi_iqr = q3 + fence_half

        for r in grp:
            v   = r["offset_sec"]
            cam_id = r["camera"]

            if is_whitelisted(r["trial"], cam_id, whitelist):
                continue

            z      = abs(v - mean_v) / std_v if std_v > 1e-9 else 0.0
            is_iqr = (v < lo_iqr) or (v > hi_iqr)
            is_z   = z > OUTLIER_Z_THRESH

            if is_iqr or is_z:
                outliers.append({
                    "trial":                    r["trial"],
                    "subject":                  subj,
                    "camera":                   cam_id,
                    "offset_sec":               round(v, 4),
                    "group_median_sec":         round(median_v, 4),
                    "group_mean_sec":           round(mean_v, 4),
                    "group_std_ms":             round(std_v * 1000, 1),
                    "deviation_from_median_ms": round(abs(v - median_v) * 1000, 1),
                    "z_score":                  round(z, 3),
                    "iqr_fence":                [round(lo_iqr, 4), round(hi_iqr, 4)],
                    "reason":                   ("IQR+Z" if (is_iqr and is_z)
                                                 else "IQR" if is_iqr else "Z"),
                    "quality":                  r["quality"],
                    "sync_file":                r["sync_file"],
                    "action": (
                        "Check in align_ui.py: is the sync event correct? "
                        "If offset is valid (camera started early/late), "
                        "add to whitelist.json to suppress this warning."
                    ),
                })

    return outliers
def build_consistency_matrix(records: list[dict]) -> tuple:
    """
    Build a Subject × Camera matrix of intra-group std (ms).
    Returns (matrix, subjects, cameras).
    """
    groups = defaultdict(list)
    for r in records:
        groups[(r["subject"], r["camera"])].append(r["offset_sec"])

    subjects = sorted({r["subject"] for r in records})
    cameras  = sorted({r["camera"]  for r in records})

    matrix = np.full((len(subjects), len(cameras)), np.nan)
    for i, subj in enumerate(subjects):
        for j, cam in enumerate(cameras):
            vals = groups.get((subj, cam), [])
            if len(vals) >= 2:
                matrix[i, j] = np.std(vals, ddof=1) * 1000  # convert to ms

    return matrix, subjects, cameras


def build_completion_matrix(base: Path, records: list[dict]) -> tuple:
    """
    Build a Trial × Camera completion matrix.
    Values: 0=missing, 1=annotated
    Returns (matrix, trials, cameras).
    """
    # Get all known trials from directory
    all_trials = set()
    for sync_file in base.rglob("*_sync.json"):
        name = sync_file.name.replace("_sync.json", "")
        if re.match(r"^sub_\d+_.+$", name):
            all_trials.add(name)

    # Also collect from records
    for r in records:
        all_trials.add(r["trial"])

    trials  = sorted(all_trials)
    cameras = sorted({r["camera"] for r in records} | set(range(1, 9)))

    # Build annotation set
    annotated = {(r["trial"], r["camera"]) for r in records}

    matrix = np.zeros((len(trials), len(cameras)), dtype=int)
    for i, trial in enumerate(trials):
        for j, cam in enumerate(cameras):
            matrix[i, j] = 1 if (trial, cam) in annotated else 0

    return matrix, trials, cameras


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig1_offset_per_subject(records: list[dict], out_dir: Path):
    """Box plot: offset distribution per subject, colored by quality."""
    if not HAS_MPL:
        return

    subjects = sorted({r["subject"] for r in records})
    data_by_subj = defaultdict(list)
    for r in records:
        data_by_subj[r["subject"]].append(r["offset_sec"])

    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 0.9), 5))
    with plt.rc_context(STYLE):

        positions = range(1, len(subjects) + 1)
        box_data  = [data_by_subj[s] for s in subjects]

        bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                        widths=0.55, notch=False,
                        medianprops=dict(color="#e74c3c", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker="o", markersize=4,
                                        markerfacecolor="#e74c3c",
                                        markeredgewidth=0.5, alpha=0.6))

        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE["blue"])
            patch.set_alpha(0.55)

        # Overlay individual points
        for i, (subj, pos) in enumerate(zip(subjects, positions)):
            vals = data_by_subj[subj]
            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(vals))
            ax.scatter([pos + j for j in jitter], vals,
                       s=18, alpha=0.5, color=PALETTE["dark"], zorder=3,
                       linewidths=0)

        # Annotate n
        for subj, pos in zip(subjects, positions):
            n = len(data_by_subj[subj])
            ax.text(pos, ax.get_ylim()[0] if ax.get_ylim()[0] < -1 else
                    min(data_by_subj[subj]) - 0.3,
                    f"n={n}", ha="center", va="top", fontsize=7.5,
                    color="grey")

        ax.set_xticks(list(positions))
        ax.set_xticklabels([s.replace("sub_", "S") for s in subjects],
                           rotation=30, ha="right")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Offset  (GRF − Video)  [s]")
        ax.set_title("Temporal Offset Distribution per Subject")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")

        # Add std annotation
        for subj, pos in zip(subjects, positions):
            vals = data_by_subj[subj]
            if len(vals) > 1:
                std = np.std(vals, ddof=1) * 1000
                ax.text(pos, np.max(vals) + 0.05,
                        f"σ={std:.0f}ms", ha="center", va="bottom",
                        fontsize=7, color="#7f8c8d")

        fig.tight_layout()
        _save_fig(fig, out_dir / "fig1_offset_per_subject.pdf")
        _save_fig(fig, out_dir / "fig1_offset_per_subject.png")
        plt.close(fig)


def fig2_offset_per_camera(records: list[dict], out_dir: Path):
    """Violin plot: offset distribution per camera."""
    if not HAS_MPL:
        return

    cameras  = sorted({r["camera"] for r in records})
    data_by_cam = defaultdict(list)
    for r in records:
        data_by_cam[r["camera"]].append(r["offset_sec"])

    cameras = [c for c in cameras if len(data_by_cam[c]) >= 2]
    if not cameras:
        return

    fig, ax = plt.subplots(figsize=(max(6.8, len(cameras) * 0.85), 2.6))
    with plt.rc_context(STYLE):

        positions = range(1, len(cameras) + 1)
        parts = ax.violinplot(
            [data_by_cam[c] for c in cameras],
            positions=list(positions),
            showmedians=True, showextrema=True,
        )

        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(CAM_COLORS[i % 10])
            pc.set_alpha(0.55)
            pc.set_edgecolor("#FFFFFF")
            pc.set_linewidth(0.6)
        parts["cmedians"].set_colors("#e74c3c")
        parts["cmedians"].set_linewidth(1.6)

        # Overlay scatter
        for cam, pos in zip(cameras, positions):
            vals = data_by_cam[cam]
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter([pos + j for j in jitter], vals,
                       s=14, alpha=0.55, color=PALETTE["dark"], zorder=3,
                       linewidths=0)

        ax.set_xticks(list(positions))
        ax.set_xticklabels([f"C{c}" for c in cameras])
        ax.set_xlabel("Camera")
        ax.set_ylabel("Offset  (GRF − Video)  [s]")
        ax.set_title("Video–GRF offset distribution per camera")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")

        # Annotate stats
        for cam, pos in zip(cameras, positions):
            vals = data_by_cam[cam]
            ax.text(pos, ax.get_ylim()[1] * 0.97,
                    f"n={len(vals)}\nμ={np.mean(vals):.2f}s",
                    ha="center", va="top", fontsize=7.5, color="#555")

        fig.tight_layout()
        _save_fig(fig, out_dir / "fig2_offset_per_camera.pdf")
        _save_fig(fig, out_dir / "fig2_offset_per_camera.png")
        plt.close(fig)


def fig_paper_sync_quality(records: list[dict], out_dir: Path):
    """
    Paper-ready QC figure (single panel):
      per-camera offset distribution (violin + scatter).
    Saved as fig3_sync_quality.{pdf,png} for direct inclusion in the paper.
    """
    if not HAS_MPL:
        return

    # Camera distribution (human-in-the-loop per-camera offsets).
    cameras = sorted({r["camera"] for r in records})
    data_by_cam = defaultdict(list)
    for r in records:
        data_by_cam[r["camera"]].append(r["offset_sec"])
    cameras = [c for c in cameras if len(data_by_cam[c]) >= 2]
    if not cameras:
        return

    tick_fs = 7
    label_fs = 8.5

    fig, ax = plt.subplots(figsize=(3.35, 2.9))
    with plt.rc_context(STYLE):
        fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.22)

        pos = list(range(1, len(cameras) + 1))
        parts = ax.violinplot([data_by_cam[c] for c in cameras],
                              positions=pos, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(CAM_COLORS[i % 10])
            pc.set_alpha(0.55)
            pc.set_edgecolor("#FFFFFF")
            pc.set_linewidth(0.6)
        parts["cmedians"].set_colors(PALETTE["bad"])
        parts["cmedians"].set_linewidth(1.6)

        # Overlay light scatter (deterministic jitter).
        for cam, p in zip(cameras, pos):
            vals = data_by_cam[cam]
            jitter = np.random.default_rng(42).uniform(-0.10, 0.10, len(vals))
            ax.scatter([p + j for j in jitter], vals, s=10, alpha=0.35,
                       color=PALETTE["dark"], linewidths=0)

        ax.set_xticks(pos)
        ax.set_xticklabels([f"C{c}" for c in cameras], fontsize=tick_fs)
        ax.set_xlabel("Camera", fontsize=label_fs, labelpad=6)
        ax.set_ylabel("Offset (s)", fontsize=label_fs, labelpad=4)
        ax.axhline(0, color=PALETTE["muted"], linewidth=0.8, linestyle=":")
        ax.tick_params(axis="both", labelsize=tick_fs, length=0, pad=2)

        # Slightly padded y-range for a clean top/bottom margin.
        all_vals = np.concatenate(
            [np.asarray(data_by_cam[c], dtype=float) for c in cameras if len(data_by_cam[c])]
        )
        if all_vals.size > 0:
            y_min, y_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
            pad = 0.08 * max(1e-6, (y_max - y_min))
            ax.set_ylim(y_min - pad, y_max + pad)

        _save_fig(fig, out_dir / "fig3_sync_quality.pdf")
        _save_fig(fig, out_dir / "fig3_sync_quality.png")
        plt.close(fig)


def fig3_consistency_heatmap(records: list[dict], out_dir: Path):
    """
    Subject × Camera heatmap of intra-subject offset std (ms).
    Low std = consistent annotation = good quality.
    """
    if not HAS_MPL:
        return

    matrix, subjects, cameras = build_consistency_matrix(records)
    if matrix.size == 0:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(cameras) * 0.9 + 1),
                                    max(4, len(subjects) * 0.55 + 1.5)))
    with plt.rc_context(STYLE):

        # Mask NaN cells
        masked = np.ma.masked_invalid(matrix)
        cmap   = plt.cm.RdYlGn_r
        cmap.set_bad(color="#eeeeee")

        im = ax.imshow(masked, cmap=cmap, aspect="auto",
                       vmin=0, vmax=50)

        ax.set_xticks(range(len(cameras)))
        ax.set_xticklabels([f"C{c}" for c in cameras])
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels([s.replace("sub_", "S") for s in subjects])

        # Annotate cells
        for i in range(len(subjects)):
            for j in range(len(cameras)):
                v = matrix[i, j]
                if not np.isnan(v):
                    color = "white" if v > 30 else PALETTE["dark"]
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")
                else:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8, color="#aaa")

        cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("Intra-subject σ (ms)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

        ax.set_title("Annotation Consistency: Intra-subject Offset Std per Camera\n"
                     "(lower = more consistent; grey = insufficient data)",
                     pad=10)
        ax.set_xlabel("Camera")
        ax.set_ylabel("Subject")

        fig.tight_layout()
        _save_fig(fig, out_dir / "fig3_consistency_heatmap.pdf")
        _save_fig(fig, out_dir / "fig3_consistency_heatmap.png")
        plt.close(fig)


def fig4_completion_matrix(base: Path, records: list[dict], out_dir: Path):
    """
    Trial × Camera annotation progress heatmap.
    """
    if not HAS_MPL:
        return

    matrix, trials, cameras = build_completion_matrix(base, records)
    if matrix.size == 0:
        return

    n_trials  = len(trials)
    n_cameras = len(cameras)
    fig_h = max(6, n_trials * 0.22 + 2)
    fig, ax = plt.subplots(figsize=(max(6, n_cameras * 0.9 + 2), fig_h))

    with plt.rc_context(STYLE):
        cmap = matplotlib.colors.ListedColormap(["#f5f5f5", PALETTE["good"]])
        ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1,
                  interpolation="nearest")

        ax.set_xticks(range(n_cameras))
        ax.set_xticklabels([f"C{c}" for c in cameras])
        ax.set_yticks(range(n_trials))

        # Simplify y-axis labels: show subject prefix on first trial of each
        ylabels = []
        prev_subj = None
        for trial in trials:
            subj = "_".join(trial.split("_")[:2])
            short = trial.replace(subj + "_", "")
            if subj != prev_subj:
                ylabels.append(f"{subj.replace('sub_','S')} | {short}")
                prev_subj = subj
            else:
                ylabels.append(f"   {short}")
        ax.set_yticklabels(ylabels, fontsize=7)

        # Per-trial completion percentage on the right
        pct = matrix.sum(axis=1) / n_cameras * 100
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(n_trials))
        ax2.set_yticklabels([f"{p:.0f}%" for p in pct], fontsize=7)
        ax2.spines["left"].set_visible(False)

        # Add subject dividers
        prev_subj = None
        for i, trial in enumerate(trials):
            subj = "_".join(trial.split("_")[:2])
            if subj != prev_subj and i > 0:
                ax.axhline(i - 0.5, color="white", linewidth=2)
            prev_subj = subj

        legend_elements = [
            Patch(facecolor=PALETTE["good"],  label="Annotated"),
            Patch(facecolor="#f5f5f5",         label="Pending"),
        ]
        ax.legend(handles=legend_elements, loc="upper right",
                  bbox_to_anchor=(1.0, 1.02), ncol=2, framealpha=0.9)

        total_done = matrix.sum()
        total_all  = matrix.size
        ax.set_title(f"Annotation Progress: {total_done}/{total_all} "
                     f"({total_done/total_all*100:.1f}%)", pad=10)
        ax.set_xlabel("Camera")
        ax.set_ylabel("Trial")

        fig.tight_layout()
        _save_fig(fig, out_dir / "fig4_completion_matrix.pdf")
        _save_fig(fig, out_dir / "fig4_completion_matrix.png")
        plt.close(fig)


def fig5_confidence_dist(records: list[dict], out_dir: Path):
    """Stacked bar: GRF onset confidence label distribution per subject."""
    if not HAS_MPL:
        return

    # Only use records with confidence info (one per trial, not per cam)
    seen_trials = set()
    trial_conf  = {}
    for r in records:
        if r["trial"] not in seen_trials and r.get("confidence_label"):
            trial_conf[r["trial"]] = {
                "subject":          r["subject"],
                "confidence_label": r["confidence_label"],
                "confidence":       r.get("confidence"),
            }
            seen_trials.add(r["trial"])

    if not trial_conf:
        print("[INFO] No confidence data available — skipping fig5")
        return

    subjects = sorted({v["subject"] for v in trial_conf.values()})
    labels   = ["HIGH", "MEDIUM", "LOW"]
    colors   = [PALETTE["good"], PALETTE["uncertain"], PALETTE["bad"]]

    counts = {subj: {lbl: 0 for lbl in labels} for subj in subjects}
    for v in trial_conf.values():
        subj = v["subject"]
        lbl  = v["confidence_label"].upper()
        if lbl in counts[subj]:
            counts[subj][lbl] += 1

    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 0.9), 4.5))
    with plt.rc_context(STYLE):

        x      = np.arange(len(subjects))
        bottom = np.zeros(len(subjects))

        for lbl, color in zip(labels, colors):
            vals = [counts[s][lbl] for s in subjects]
            ax.bar(x, vals, bottom=bottom, label=lbl, color=color,
                   alpha=0.82, edgecolor="white", linewidth=0.5)
            # Annotate non-zero bars
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0:
                    ax.text(i, b + v / 2, str(v), ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("sub_", "S") for s in subjects],
                           rotation=30, ha="right")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Number of Trials")
        ax.set_title("GRF Onset Detection Confidence per Subject")
        ax.legend(title="Confidence", loc="upper right")

        fig.tight_layout()
        _save_fig(fig, out_dir / "fig5_confidence_dist.pdf")
        _save_fig(fig, out_dir / "fig5_confidence_dist.png")
        plt.close(fig)


def _save_fig(fig, path: Path):
    try:
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"[SAVE] {path.name}")
    except Exception as e:
        print(f"[WARN] Could not save {path}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CSV / JSON OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def save_summary_csv(records: list[dict], path: Path):
    """Write flat per-(trial, camera) CSV."""
    fields = [
        "subject", "trial", "camera",
        "offset_sec", "offset_ms",
        "grf_event_sec", "video_event_sec", "video_event_frame",
        "quality", "confidence_label", "confidence",
        "annotated_at", "tool_version", "notes",
    ]
    rows = []
    for r in records:
        row = {k: r.get(k, "") for k in fields}
        row["offset_ms"] = (round(r["offset_sec"] * 1000, 2)
                            if _is_finite(r.get("offset_sec", None)) else "")
        rows.append(row)

    rows.sort(key=lambda x: (x["subject"], x["trial"], x["camera"]))

    if HAS_PD:
        df = pd.DataFrame(rows, columns=fields)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        import csv
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"[SAVE] {path.name}  ({len(rows)} rows)")


def save_stats_json(records: list[dict], outliers: list[dict],
                    base: Path, path: Path):
    """Save aggregated statistics as JSON."""

    all_offsets = [r["offset_sec"] for r in records]

    # Per-subject stats
    subj_groups = defaultdict(list)
    for r in records:
        subj_groups[r["subject"]].append(r["offset_sec"])

    # Per-camera stats
    cam_groups = defaultdict(list)
    for r in records:
        cam_groups[r["camera"]].append(r["offset_sec"])

    # Quality counts
    quality_counts = defaultdict(int)
    for r in records:
        quality_counts[r["quality"]] += 1

    # Confidence counts (per trial, not per cam)
    seen = set()
    conf_counts = defaultdict(int)
    for r in records:
        if r["trial"] not in seen:
            conf_counts[r.get("confidence_label", "unknown")] += 1
            seen.add(r["trial"])

    # Completion stats
    n_trials_annotated = len({r["trial"] for r in records})
    n_cam_pairs        = len(records)

    stats = {
        "generated_at":        datetime.now().isoformat(),
        "base_path":           str(base),
        "summary": {
            "n_trials_annotated":  n_trials_annotated,
            "n_cam_annotations":   n_cam_pairs,
            "n_subjects":          len(subj_groups),
            "n_outliers_flagged":  len(outliers),
            "quality_counts":      dict(quality_counts),
            "confidence_counts":   dict(conf_counts),
        },
        "global_offset": compute_group_stats(all_offsets),
        "per_subject":   {s: compute_group_stats(v)
                          for s, v in sorted(subj_groups.items())},
        "per_camera":    {str(c): compute_group_stats(v)
                          for c, v in sorted(cam_groups.items())},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {path.name}")
    return stats


def save_outliers_json(outliers: list[dict], path: Path):
    """Save outlier list for re-annotation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outliers, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {path.name}  ({len(outliers)} outliers flagged)")


def _write_whitelist_template(path: Path):
    """
    Create a whitelist.json template if it doesn't exist.
    Users edit this file to suppress known-good anomalies.
    """
    template = {
        "_instructions": [
            "Add entries to confirmed_ok to suppress outlier warnings.",
            "Each entry needs 'trial' and optionally 'camera' (int or omit for all cams).",
            "Add 'note' to explain why this offset is legitimately different.",
            "Example: camera started much earlier than usual for this trial."
        ],
        "confirmed_ok": [
            # Example entry (commented out in JSON via _example key):
        ],
        "_example": {
            "trial":  "sub_001_rally_01",
            "camera": 1,
            "note":   "Camera started ~78s early; sync stomp visible at 83.6s. Verified correct."
        }
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Created whitelist template: {path.name} — edit to suppress known-OK anomalies")
    except OSError as e:
        print(f"[WARN] Could not create whitelist template: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_and_write_report(records: list[dict], outliers: list[dict],
                            stats: dict, path: Path):
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 72)
    p("  BadmintonGRF — Sync Verification Report")
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 72)

    s = stats["summary"]
    p()
    p("── OVERVIEW ────────────────────────────────────────────────────────────")
    p(f"  Trials annotated   : {s['n_trials_annotated']}")
    p(f"  Camera annotations : {s['n_cam_annotations']}")
    p(f"  Subjects           : {s['n_subjects']}")
    p(f"  Outliers flagged   : {s['n_outliers_flagged']}")
    p()

    p("── QUALITY DISTRIBUTION ────────────────────────────────────────────────")
    for q, n in sorted(s["quality_counts"].items()):
        bar = "█" * n
        p(f"  {q:<12}: {n:>4}  {bar}")
    p()

    p("── GRF ONSET CONFIDENCE ────────────────────────────────────────────────")
    for lbl, n in sorted(s["confidence_counts"].items()):
        bar = "█" * n
        p(f"  {lbl:<12}: {n:>4}  {bar}")
    p()

    g = stats["global_offset"]
    if g:
        p("── GLOBAL OFFSET STATISTICS ────────────────────────────────────────────")
        p(f"  Mean   : {g['mean']:+.4f} s  ({g['mean']*1000:+.1f} ms)")
        p(f"  Std    : {g['std']:.4f} s  ({g['std']*1000:.1f} ms)")
        p(f"  Median : {g['median']:+.4f} s  ({g['median']*1000:+.1f} ms)")
        p(f"  IQR    : {g['iqr']:.4f} s  ({g['iqr']*1000:.1f} ms)")
        p(f"  Range  : [{g['min']:+.4f}, {g['max']:+.4f}] s")
        p()

    p("── PER-SUBJECT OFFSET SUMMARY ──────────────────────────────────────────")
    p(f"  {'Subject':<12}  {'n':>4}  {'Mean (ms)':>10}  "
      f"{'Std (ms)':>9}  {'Range (ms)':>12}")
    p("  " + "-" * 60)
    for subj, sv in sorted(stats["per_subject"].items()):
        if not sv:
            continue
        p(f"  {subj:<12}  {sv['n']:>4}  {sv['mean']*1000:>+10.1f}  "
          f"{sv['std']*1000:>9.1f}  "
          f"[{sv['min']*1000:+.0f}, {sv['max']*1000:+.0f}]")
    p()

    p("── PER-CAMERA OFFSET SUMMARY ───────────────────────────────────────────")
    p(f"  {'Camera':<8}  {'n':>4}  {'Mean (ms)':>10}  {'Std (ms)':>9}")
    p("  " + "-" * 40)
    for cam, cv in sorted(stats["per_camera"].items(),
                           key=lambda x: int(x[0])):
        if not cv:
            continue
        p(f"  Cam {cam:<4}  {cv['n']:>4}  {cv['mean']*1000:>+10.1f}  "
          f"{cv['std']*1000:>9.1f}")
    p()

    if outliers:
        p("── OUTLIERS: STATISTICAL ANOMALIES ─────────────────────────────────────")
        p(f"  Offsets that deviate significantly from the (subject, camera) group median.")
        p(f"  May be annotation error OR legitimate camera timing variation.")
        p(f"  Check in align_ui.py, then add to whitelist.json if confirmed OK.")
        p()
        p(f"  {'Trial':<40}  {'Cam':>4}  {'Offset(s)':>10}  "
          f"{'Dev. from median':>18}  {'Z':>5}")
        p("  " + "-" * 85)
        for o in sorted(outliers, key=lambda x: (x["subject"], x["trial"], x["camera"])):
            dev_ms = o.get("deviation_from_median_ms", 0)
            z      = o.get("z_score", 0)
            p(f"  {o['trial']:<40}  {o['camera']:>4}  "
              f"{o['offset_sec']:>+10.4f}  "
              f"{dev_ms:>+17.0f}ms  "
              f"{z:>5.2f}")
        p()
        p('  ► If offset is valid, add to whitelist.json:')
        p('    {"trial": "sub_001_rally_01", "camera": 1, "note": "cam started early"}')
        p('  ► If annotation is wrong, re-annotate in align_ui.py')
    else:
        p("── OUTLIERS ────────────────────────────────────────────────────────────")
        p("  No statistical anomalies detected. Annotation appears consistent.")

    p()
    p("=" * 72)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BadmintonGRF Step 2 — Verify Sync Annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", type=str, default="",
                        help="Data root with sub_XXX dirs (default: BADMINTON_DATA_ROOT or <repo>/data)")
    parser.add_argument("--out",  type=str, default="",
                        help="Output directory (default: {root}/verify_output)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--whitelist", type=str, default="",
                        help="Path to whitelist.json (default: {out}/whitelist.json)")
    args = parser.parse_args()

    _default = str(Path(__file__).resolve().parent.parent / "data")
    root_str = (args.root or os.environ.get("BADMINTON_DATA_ROOT", "").strip()
                or _default)
    if not root_str:
        print("[ERROR] --root required or set BADMINTON_DATA_ROOT")
        sys.exit(1)

    base = Path(root_str).resolve()
    if not base.exists():
        print(f"[ERROR] Root not found: {base}")
        sys.exit(1)

    out_dir = Path(args.out).resolve() if args.out else base / "verify_output"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  BadmintonGRF Sync Verifier")
    print(f"  Root   : {base}")
    print(f"  Output : {out_dir}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("[INFO] Loading sync files...")
    records = load_all_sync(base)

    if not records:
        print("[ERROR] No annotated trials found. "
              "Run step1 (align_ui.py) first.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(records)} (trial, camera) annotations "
          f"from {len({r['trial'] for r in records})} trials")

    # ── Whitelist ─────────────────────────────────────────────────────────────
    wl_path = (Path(args.whitelist).resolve() if args.whitelist
               else out_dir / "whitelist.json")
    whitelist = load_whitelist(wl_path)
    if whitelist:
        print(f"[INFO] Whitelist loaded: {len(whitelist)} entries from {wl_path.name}")
    else:
        # Create template whitelist if it doesn't exist yet
        if not wl_path.exists():
            _write_whitelist_template(wl_path)

    # ── Outlier detection ─────────────────────────────────────────────────────
    print("[INFO] Detecting outliers (cross-camera consistency check)...")
    outliers = detect_outliers(records, whitelist)

    # ── Statistics ────────────────────────────────────────────────────────────
    print("[INFO] Computing statistics...")
    stats = save_stats_json(records, outliers, base,
                             out_dir / "verify_sync_stats.json")

    # ── CSV ───────────────────────────────────────────────────────────────────
    save_summary_csv(records, out_dir / "verify_sync_summary.csv")

    # ── Outliers JSON ─────────────────────────────────────────────────────────
    save_outliers_json(outliers, out_dir / "outliers.json")

    # ── Report ────────────────────────────────────────────────────────────────
    print_and_write_report(records, outliers, stats,
                            out_dir / "verify_sync_report.txt")

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.no_figures and HAS_MPL:
        print("\n[INFO] Generating figures...")
        np.random.seed(42)
        fig1_offset_per_subject(records, fig_dir)
        fig2_offset_per_camera(records, fig_dir)
        fig3_consistency_heatmap(records, fig_dir)
        fig4_completion_matrix(base, records, fig_dir)
        fig5_confidence_dist(records, fig_dir)
        # Paper-ready combined QC (saved alongside other figures)
        fig_paper_sync_quality(records, fig_dir)
        print(f"[INFO] Figures saved to {fig_dir}")
    elif args.no_figures:
        print("[INFO] Figures skipped (--no-figures)")
    else:
        print("[WARN] Figures skipped (matplotlib not available)")

    print(f"\n  Done. All outputs in: {out_dir}\n")


if __name__ == "__main__":
    main()