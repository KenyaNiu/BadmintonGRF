"""
plot_figures_acmmm.py
=====================

统一生成 ACM MM Dataset Track 风格的主要图表。

用法（在仓库根目录执行）::

    python -m analysis.plot_figures_acmmm
    # 或
    python analysis/plot_figures_acmmm.py

输出（默认到 figures/）：
  - fig4_scaling_curve.{pdf,png}
      样本人数 N = {5,6,7,8,9,10} 上，E4 / E5 的 r² 曲线
  - fig5_camera_ablation_e1.{pdf,png}
      E6-E1 相机消融：各相机的 r² / RMSE / peak_err
  - fig6_fatigue_stages.{pdf,png}
      E3 疲劳分析：stage1/2/3 的 rmse_fz 条形图
  - fig4_model_comparison.{pdf,png}
      E1 / E4 / E5 的整体 r² / RMSE 对比（10 人）
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIG_DIR = ROOT / "figures"


# ──────────────────────────── Matplotlib 样式 ────────────────────────────────

_PALETTE = {
    "blue":   "#2563EB",
    "green":  "#059669",
    "orange": "#F59E0B",
    "red":    "#DC2626",
    "ink":    "#111827",
    "muted":  "#6B7280",
    "grid":   "#E5E7EB",
}


def _set_acmmm_style() -> None:
    """
    ACM MM-style, publication-ready Matplotlib defaults.
    - Clean white background
    - Subtle y-grid only
    - Thin spines
    - Consistent font sizes for 2-column papers
    """
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": _PALETTE["grid"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": _PALETTE["grid"],
            "grid.alpha": 0.9,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            # Keep subplot titles modest for ACM 2-column.
            "axes.titlesize": 8.6,
            "axes.titleweight": "regular",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
        }
    )


def _annotate_bar(ax, rects, fmt="{:.2f}", dy=0.015, color=_PALETTE["muted"]):
    for r in rects:
        h = r.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            r.get_x() + r.get_width() / 2,
            h + dy * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
            clip_on=False,
        )


def _annotate_bar_smart(ax, rects, fmt="{:.2f}") -> None:
    """
    Annotate bars without covering data:
    - For positive bars: place label inside bar near the top when possible,
      otherwise place slightly above.
    - For negative bars: place label slightly below the bar.
    """
    y0, y1 = ax.get_ylim()
    yr = max(1e-9, y1 - y0)
    pad_out = 0.040 * yr
    pad_in = 0.022 * yr

    for r in rects:
        h = r.get_height()
        if not np.isfinite(h):
            continue

        x = r.get_x() + r.get_width() / 2
        if h >= 0:
            # If the bar is "tall enough", put label inside (white) to avoid
            # colliding with the top frame or neighboring text.
            if h > 0.12 * yr:
                ax.text(
                    x,
                    h - pad_in,
                    fmt.format(h),
                    ha="center",
                    va="top",
                    fontsize=7.2,
                    color="white",
                    fontweight="bold",
                    clip_on=True,
                )
            else:
                ax.text(
                    x,
                    h + pad_out,
                    fmt.format(h),
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color=_PALETTE["muted"],
                    clip_on=False,
                )
        else:
            ax.text(
                x,
                h - pad_out,
                fmt.format(h),
                ha="center",
                va="top",
                fontsize=7.2,
                color=_PALETTE["muted"],
                clip_on=False,
            )


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[FIG] saved → {out}")
    plt.close(fig)


# ───────────────────────── Canonical summary helpers ────────────────────────

def _load_canonical(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _discover_latest_e1() -> Path:
    """最新的 E1 目录：*_e1_lstm_tcn_camall"""
    cands = sorted(RUNS.glob("*_e1_lstm_tcn_camall"))
    if not cands:
        raise RuntimeError("No E1 runs found under runs/*_e1_lstm_tcn_camall")
    return cands[-1]


def _discover_e4_scaling() -> List[Tuple[int, Path]]:
    """
    返回 [(N, dir), ...]，例如 (5, runs/..._e4_5p_stgcn_transformer)
    """
    out: List[Tuple[int, Path]] = []
    for p in sorted(RUNS.glob("*_e4_*p_stgcn_transformer")):
        name = p.name
        # name 形如 20260317_024228_e4_5p_stgcn_transformer
        try:
            part = name.split("_e4_")[1]
            n_str = part.split("p_")[0]
            n = int(n_str)
        except Exception:
            continue
        out.append((n, p))
    out.sort(key=lambda x: x[0])
    return out


# ────────────────────────────── FIG 1: Scaling ──────────────────────────────

def fig1_scaling_curve() -> None:
    """
    N = {5,6,7,8,9,10} 上，E4/E5 的 r² 曲线。
    使用 canonical summary 中的 r2_fz。
    """
    pairs = _discover_e4_scaling()
    if not pairs:
        print("[FIG1] no E4 scaling runs found, skip.")
        return

    ns, r2_e4, r2_e5 = [], [], []
    s_e4, s_e5 = [], []
    for n, d in pairs:
        try:
            s4 = _load_canonical(d / "summary_canonical.json")
            s5 = _load_canonical(d / "e5_late_fusion" / "summary_canonical.json")
            # Optional std from non-canonical summary (if present)
            s4_full = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            s5_full = json.loads((d / "e5_late_fusion" / "summary.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"[FIG1] missing canonical summary for N={n}, skip.")
            continue
        ns.append(n)
        r2_e4.append(s4["mean"].get("r2_fz", np.nan))
        r2_e5.append(s5["mean"].get("r2_fz", np.nan))
        s_e4.append(float(s4_full.get("mean", {}).get("r2_std", np.nan)))
        s_e5.append(float(s5_full.get("mean", {}).get("r2_std", np.nan)))

    if not ns:
        print("[FIG1] no valid points, skip.")
        return

    _set_acmmm_style()
    # Single-column friendly canvas (avoid being scaled down in LaTeX).
    fig, ax = plt.subplots(figsize=(3.6, 2.35))
    ax.plot(ns, r2_e4, marker="o", color=_PALETTE["blue"], label="E4 (single-view)")
    ax.plot(ns, r2_e5, marker="s", color=_PALETTE["green"], label="E5 (late fusion)")

    # Optional uncertainty bands (if std exists)
    s_e4 = np.asarray(s_e4, dtype=float)
    s_e5 = np.asarray(s_e5, dtype=float)
    if np.isfinite(s_e4).any():
        ax.fill_between(ns, np.array(r2_e4) - s_e4, np.array(r2_e4) + s_e4,
                        color=_PALETTE["blue"], alpha=0.10, linewidth=0)
    if np.isfinite(s_e5).any():
        ax.fill_between(ns, np.array(r2_e5) - s_e5, np.array(r2_e5) + s_e5,
                        color=_PALETTE["green"], alpha=0.10, linewidth=0)

    ax.set_xlabel("Number of subjects (N)")
    ax.set_ylabel(r"$R^2$ on $F_z$")
    ax.set_title("Scaling with subject count")
    ax.set_xticks(ns)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylim(0, max([np.nanmax(r2_e4), np.nanmax(r2_e5)]) * 1.15)
    ax.legend(loc="lower right", handlelength=1.6, borderaxespad=0.2)

    # Annotate each point (top-tier paper style: precise, but unobtrusive).
    # Smart point labels: avoid overlap when the two curves are close.
    label_bbox = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.55)
    for n, y4, y5 in zip(ns, r2_e4, r2_e5):
        if not (np.isfinite(y4) and np.isfinite(y5)):
            continue
        d = abs(y5 - y4)
        # When curves are close, push labels further apart and swap sides.
        if d < 0.055:
            off4, va4 = -0.020, "top"
            off5, va5 = +0.022, "bottom"
        else:
            off4, va4 = +0.014, "bottom"
            off5, va5 = -0.018, "top"
        ax.text(n, y4 + off4, f"{y4:.2f}", ha="center", va=va4,
                fontsize=6.6, color=_PALETTE["blue"], bbox=label_bbox, zorder=5)
        ax.text(n, y5 + off5, f"{y5:.2f}", ha="center", va=va5,
                fontsize=6.6, color=_PALETTE["green"], bbox=label_bbox, zorder=5)
    _save(fig, "fig4_scaling_curve")


# ───────────────────────── FIG 2: Camera ablation (E6-E1) ───────────────────

def fig2_camera_ablation_e1() -> None:
    """
    使用 runs/*_e6_e1base/ablation_cameras/camera_ablation_summary.json
    绘制每个相机的 r² / RMSE / peak_err 条形图。
    """
    cands = sorted(RUNS.glob("*_e6_e1base/ablation_cameras/camera_ablation_summary.json"))
    if not cands:
        print("[FIG2] no camera_ablation_summary.json found, skip.")
        return
    path = cands[-1]
    with path.open(encoding="utf-8") as f:
        data: Dict[str, Dict] = json.load(f)

    cams = sorted(data.keys(), key=lambda x: int(x.replace("cam", "")))
    r2 = [data[c].get("r2_fz", np.nan) for c in cams]
    rmse = [data[c].get("rmse_fz", np.nan) for c in cams]
    peak = [data[c].get("peak_err_bw", np.nan) for c in cams]

    _set_acmmm_style()
    # Stack panels vertically to maximize readability in a single column.
    fig, axes = plt.subplots(3, 1, figsize=(3.6, 4.9), sharex=True)
    x = np.arange(len(cams))

    def _bar(ax, vals, title, color, fmt="{:.2f}"):
        rects = ax.bar(x, vals, color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(title, pad=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_axisbelow(True)
        _annotate_bar_smart(ax, rects, fmt=fmt)
        return rects

    r0 = _bar(axes[0], r2, r"$R^2$ on $F_z$", _PALETTE["blue"], fmt="{:.2f}")
    r1 = _bar(axes[1], rmse, r"nRMSE on $F_z$ (BW)", _PALETTE["green"], fmt="{:.2f}")
    r2b = _bar(axes[2], peak, r"PeakErr (frames)", _PALETTE["orange"], fmt="{:.1f}")

    # Provide headroom so value labels never collide with the top frame.
    # Keep a bit more room for small negative values (e.g., -0.04).
    axes[0].set_ylim(min(-0.10, float(np.nanmin(r2)) * 1.45), float(np.nanmax(r2)) * 1.22)
    axes[1].set_ylim(0.0, np.nanmax(rmse) * 1.15)
    axes[2].set_ylim(0.0, np.nanmax(peak) * 1.20)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([c.replace("cam", "C") for c in cams])
    axes[-1].set_xlabel("Camera")
    fig.suptitle("Camera ablation (E6–E1, E1 backbone)", y=0.995)
    fig.tight_layout(pad=0.6, h_pad=0.7)
    _save(fig, "fig5_camera_ablation_e1")


# ─────────────────────────── FIG 3: Fatigue stages (E3) ─────────────────────

def fig3_fatigue_stages() -> None:
    """
    从 E3 的 fatigue_table.csv 里解析 stage1/2/3 的 rmse_fz，
    绘制条形图（mean ± std）。
    """
    e1_dir = _discover_latest_e1()
    csv_path = e1_dir / "e3_fatigue" / "fatigue_table.csv"
    if not csv_path.exists():
        print("[FIG3] fatigue_table.csv not found, skip.")
        return

    import csv

    stages: List[str] = []
    means: List[float] = []
    stds: List[float] = []

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = row["stage"]
            # row["rmse_fz"] 形如 "0.1234±0.0567"
            val = row["rmse_fz"]
            if "±" in val:
                m_str, s_str = val.split("±")
                try:
                    m = float(m_str)
                    s = float(s_str)
                except ValueError:
                    continue
                stages.append(stage)
                means.append(m)
                stds.append(s)

    if not stages:
        print("[FIG3] no valid fatigue rows, skip.")
        return

    _set_acmmm_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.45))
    x = np.arange(len(stages))
    rects = ax.bar(x, means, yerr=stds, capsize=3, color=_PALETTE["blue"],
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("nRMSE (Fz, ×BW)")
    ax.set_title("Fatigue stages (E3)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylim(0, max(means) * 1.25)
    _annotate_bar(ax, rects, fmt="{:.2f}", dy=0.02)
    _save(fig, "fig6_fatigue_stages")


# ───────────────────── FIG 4: Model comparison (E1/E4/E5) ───────────────────

def fig4_model_comparison() -> None:
    """
    比较 E1 / E4 / E5 在 10 人设置下的 r² / RMSE。
    - E1: latest *_e1_lstm_tcn_camall
    - E4/E5: latest *_e4_10p_stgcn_transformer
    """
    e1_dir = _discover_latest_e1()
    e1_canon = _load_canonical(e1_dir / "summary_canonical.json")

    e4_10 = sorted(RUNS.glob("*_e4_10p_stgcn_transformer"))
    if not e4_10:
        print("[FIG4] no E4 10p run found, skip.")
        return
    e4_dir = e4_10[-1]
    e4_canon = _load_canonical(e4_dir / "summary_canonical.json")
    e5_canon = _load_canonical(e4_dir / "e5_late_fusion" / "summary_canonical.json")

    labels = ["E1 BiLSTM", "E4 ST-GCN+TF", "E5 Fusion"]
    r2 = [
        e1_canon["mean"].get("r2_fz", np.nan),
        e4_canon["mean"].get("r2_fz", np.nan),
        e5_canon["mean"].get("r2_fz", np.nan),
    ]
    rmse = [
        e1_canon["mean"].get("rmse_fz", np.nan),
        e4_canon["mean"].get("rmse_fz", np.nan),
        e5_canon["mean"].get("rmse_fz", np.nan),
    ]

    _set_acmmm_style()
    # Single-column friendly: two horizontal bar panels (readable labels).
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 3.7), sharex=False)
    y = np.arange(len(labels))

    rects0 = axes[0].barh(y, r2, color=_PALETTE["blue"], edgecolor="white", linewidth=0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"$R^2$ on $F_z$")
    axes[0].set_title("Overall accuracy", pad=3)
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
    for r in rects0:
        w = r.get_width()
        axes[0].text(w + 0.01, r.get_y() + r.get_height() / 2, f"{w:.2f}",
                     va="center", ha="left", fontsize=7.5, color=_PALETTE["muted"])

    rects1 = axes[1].barh(y, rmse, color=_PALETTE["green"], edgecolor="white", linewidth=0.6)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("nRMSE on $F_z$ (BW)")
    axes[1].set_title("Overall error", pad=3)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=4))
    for r in rects1:
        w = r.get_width()
        axes[1].text(w + 0.01, r.get_y() + r.get_height() / 2, f"{w:.2f}",
                     va="center", ha="left", fontsize=7.5, color=_PALETTE["muted"])

    fig.suptitle("Model comparison (10-subject)", y=0.995)
    fig.tight_layout(pad=0.6, h_pad=0.8)
    _save(fig, "fig4_model_comparison")


# ─────────────────────────────────── main ────────────────────────────────────

def main() -> None:
    print(f"[INFO] Project root = {ROOT}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig1_scaling_curve()
    fig2_camera_ablation_e1()
    fig3_fatigue_stages()
    fig4_model_comparison()


if __name__ == "__main__":
    main()

