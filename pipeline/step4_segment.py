"""
step4_segment.py  —  Standalone 版本（impact-segment 导出）
=================================================

从 Step3 的 pose.npz + GRF.npy + sync.json 切出以接触事件为中心的窗口，
生成对齐好的训练样本 npz。完全独立，无 pipeline.config.CFG 依赖。

【核心改进 v4】
  旧版每个 trial 只切 1 个片段（以 sync.json 的对齐帧为中心）。
  新版在 pose 覆盖的 ±5s 窗口内自动检测所有 GRF 落地峰值，
  每个峰值切 1 个片段，数据量提升约 3-8 倍/trial。

【Phase A 峰检】默认 peak_mode=adaptive：
  粗检（略低阈值+短 min-distance）→ 时间合并（MERGE_GAP_SEC，合并双脚/双极值）
  → median+K·MAD 严格门控 → 失败则 strict find_peaks；fixed 模式与旧版固定 BW 阈值一致。

  窗口中心 = GRF 峰值（最大接触力）而非初始接触时刻：
  ├─ 物理意义：峰值是损伤风险最高点，是 biomechanics 研究的核心
  ├─ ML 意义：默认 window_mode=adaptive 按 Fz 形状估计前后窗长（夹在 min/max）；
  │           fixed 为对称 ±WINDOW_PRE/POST_SEC
  └─ 工程意义：峰值可从 GRF 精确自动检测，无需手工标注

路径约定（默认 Data Root = 仓库目录下 ``data/``，可用 BADMINTON_DATA_ROOT 覆盖）：
  GRF   : {root}/{sub}/labels/{trial}_grf.npy
  Sync  : {root}/{sub}/labels/{trial}_sync.json
  Pose  : {root}/{sub}/pose/{trial}_cam{N}_pose.npz（step3 v2：GRF 对齐全长，schema pose_npz_v2）
  Output: {root}/{sub}/segments/{trial}_cam{N}_impact_{idx:03d}.npz
  Reports：每次运行默认写入 <repo>/data/reports/step4_summary.json（顶层为全库汇总；
           ``by_subject`` 按 sub_XXX 分类：任务计数、片段统计、errors、rally_low_peak_bw）。
           非 dry_run 另存 step4_report.json 与 loso_splits.json。加 ``--no-save-report`` 可跳过写 JSON。

输出 impact-segment NPZ schema（内部以 schema_version 字符串区分发布格式；见仓库 README）：
  keypoints_norm    float32 (T, 17, 2)   骨骼点，归一化到 [0,1]（除以 W/H）
  keypoints_px      float32 (T, 17, 2)   骨骼点，原始像素坐标
  scores            float32 (T, 17)      关键点置信度
  frame_indices     int32   (T,)         原始视频绝对帧号
  track_status      str     (T,)         tracked/pos_fallback/reidentified/lost
  grf_at_video_fps  float32 (T, 3)       [Fx,Fy,Fz]，单位 N，Fz>0=接触力
  grf_normalized    float32 (T, 3)       grf_at_video_fps / body_weight_N（×BW）
  grf_1200hz        float32 (M, 7)       原始 1200Hz 窗口 [Fx,Fy,Fz,Fz1,Fz2,Fz3,Fz4]
  timestamps_video  float32 (T,)         各帧对应 GRF 时间轴 [s]
  timestamps_grf    float32 (M,)         高频 GRF 时间轴 [s]
  ev_idx            int32                GRF 峰值帧在序列内的局部索引（截断后重算）
  window_pre_sec/post_sec/window_mode  float32/str  本片段实际前后窗长（秒）与模式
  subject/trial/stage/camera/quality     元数据
  impact_idx        int32                本 trial 内的冲击序号（按时间，1-indexed）
  n_impacts_total   int32                本 trial/cam 检测到的总冲击数
  is_sync_impact    bool                 是否为 sync.json 标注的对齐事件
  grf_peak_sec      float32              GRF 峰值的绝对时间 [s]
  body_weight_N/body_weight_kg           体重估算
  video_fps/grf_rate/offset_sec          同步参数
  peak_force_N/peak_force_bw             本片段 Fz 最大值
  image_width/image_height
  stat_lower_body_mean_score/stat_lost_rate  质量统计
  schema_version    str                格式版本标签（写入 npz；论文与主页称 *impact segment*，见 GitHub README）

用法：
  # 单被试单相机
  python step4_segment.py --subjects sub_001 --cameras 2

  # 多被试多相机
  python step4_segment.py --subjects sub_001 sub_002 sub_003 --cameras 2 3 5 8

  # 自动发现所有已有 pose npz
  python step4_segment.py --auto

  # dry run（只打印，不写文件）
  python step4_segment.py --subjects sub_001 --cameras 2 --dry_run

  # 单被试全量 pose（摘要每次自动写入 data/reports/，无需再传开关）
  python step4_segment.py --subjects sub_001 --from-pose --dry_run

  # 覆盖已有文件
  python step4_segment.py --subjects sub_001 --cameras 2 --overwrite

  # 峰值检测：默认 adaptive（粗检+合并+MAD）；复现旧行为加 --peak-mode fixed
  python step4_segment.py --subjects sub_001 --cameras 2 --peak-mode fixed

  # 全库自动发现（同上，自动写 reports）
  python step4_segment.py --auto
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------

def _default_data_root_str() -> str:
    """与 pipeline.config.DEFAULT_DATA_ROOT 一致；本文件不依赖 CFG（便于直接 python 脚本运行）。"""
    env = os.environ.get("BADMINTON_DATA_ROOT", "").strip()
    if env:
        return env
    return str(Path(__file__).resolve().parent.parent / "data")


DEFAULT_ROOT = _default_data_root_str()
# JSON 报告默认写入仓库内 data/reports（与 --root 读取的数据根可分离，避免摘要跟到外置盘）
_REPO_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WINDOW_PRE_SEC    = 0.5      # GRF 峰值前窗口长度 [s]（--window-mode fixed）
WINDOW_POST_SEC   = 0.5      # GRF 峰值后窗口长度 [s]
GRF_RATE_DEFAULT  = 1200.0   # 实测采样率 [Hz]（非 1000）
VIDEO_FPS_DEFAULT = 119.88

# 自适应 segment 窗（相对 GRF 峰值时刻；在 Fz 上估计接触起止，再夹到 [min,max]）
# 下限与 fixed 窗一致，避免「自适应」反而比 ±0.5s 更短
WIN_ADAPT_MIN_PRE_SEC  = 0.5
WIN_ADAPT_MAX_PRE_SEC  = 1.5
WIN_ADAPT_MIN_POST_SEC = 0.5
WIN_ADAPT_MAX_POST_SEC = 2.0
WIN_FLIGHT_BW          = 0.42   # Fz 低于此×BW 视为离地/低位
WIN_ON_PLATE_BW        = 1.14   # 上升沿进入「在台上」
WIN_STANCE_END_BW      = 1.18   # 支撑末：持续低于此×BW
WIN_SUSTAIN_SEC        = 0.045  # 状态判定最短持续时间（通用）
WIN_POST_SUSTAIN_SEC   = 0.080  # 仅用于 post：更长连续低力，避免短振荡误判为「已结束」
# 峰值后先等待再判「支撑末」，否则冲击衰减段会误触发，post 被压到 MIN
WIN_POST_DELAY_SEC     = 0.20
# 支撑末判定：须持续低于 max(1.18×BW, 该峰×以下比例)，高冲击时避免过早判「已卸载」
WIN_POST_REL_PEAK      = 0.50
# grf_peak_sec 对应最近邻采样点，未必是峰顶；在此时间窗内取 raw Fz 最大作为峰值力
WIN_PEAK_LOCAL_HALF_SEC = 0.15  # 略宽，避免 peak_idx 偏一点时漏掉真实峰顶，低估 peak_bw_ratio
# 高冲击（peak/BW 大）时延长 post：仅靠「延迟后首次持续低力」会在 j0 就满足，时间≈0.25s 被 MIN 顶满
WIN_POST_SCALE_START_BW = 2.0   # 超过此 ×BW 开始追加 post
WIN_POST_EXTRA_PER_BW   = 0.18  # 每多 1×BW 追加的秒数（在 [MIN,MAX] 内）


def _peak_bw_post_floor_sec(peak_bw_ratio: float) -> float:
    """peak/BW 超过 WIN_POST_SCALE_START_BW 时，返回建议的 post 下限（秒）；否则 0。"""
    if peak_bw_ratio <= WIN_POST_SCALE_START_BW:
        return 0.0
    return min(
        WIN_ADAPT_MAX_POST_SEC,
        WINDOW_POST_SEC + WIN_POST_EXTRA_PER_BW * (peak_bw_ratio - WIN_POST_SCALE_START_BW),
    )


# 体重估算：Fz 在此范围内认为是静止站立 [N]
BW_FZ_MIN = 300.0
BW_FZ_MAX = 900.0

# GRF 峰值检测参数（基于 BW 的倍数，适合羽毛球落地）
PEAK_HEIGHT_BW     = 0.5   # 至少 0.5×BW 才算有效落地
PEAK_DISTANCE_SEC  = 0.25  # 两个峰值最短间隔 [s]（strict 单-pass）
PEAK_PROMINENCE_BW = 0.2   # 峰值显著度（相对周围）

# ── Phase A：粗检 + 时间合并 + 窗口内自适应阈值 ──
# 粗检：略低阈值、略短 min-distance，提高召回；再按 MERGE_GAP_SEC 合并双脚/双极值
PEAK_HEIGHT_COARSE_BW     = 0.35
PEAK_PROMINENCE_COARSE_BW = 0.12
PEAK_DISTANCE_COARSE_SEC  = 0.12
MERGE_GAP_SEC             = 0.18  # 同一冲击簇内多峰合并 [s]
# 自适应：height ≥ max(固定×BW, median(Fz)+K·MAD)，抑制 rally 基线漂移与噪声
ADAPT_HEIGHT_MAD_K = 4.0
ADAPT_PROM_MAD_K   = 2.0

# Butterworth 低通滤波截止频率（仅用于峰值检测，不影响保存的原始数据）
BUTTER_CUTOFF_HZ = 50.0

# Rally trial：全场移动、Fz 基线漂移与非周期性扰动大，median+K·MAD 门控易把真峰全部压掉。
# 策略：更低截止频率平滑 → 粗检峰 + 时间合并 → 仅用「绝对 Fz 下限」过滤（不再做 strict gate）。
RALLY_BUTTER_CUTOFF_HZ       = 12.0
RALLY_PEAK_MIN_FZ_BW         = 0.24   # 合并后保留峰：原始 Fz ≥ 此×BW
RALLY_PEAK_HEIGHT_COARSE_BW   = 0.22   # find_peaks height 下限（与 min_fz 二选一较松者）
RALLY_PEAK_PROMINENCE_COARSE  = 0.06   # 相对 BW 的 prominence 下限（再与 ~0.3·MAD 取 max）
RALLY_PEAK_DISTANCE_COARSE_SEC = 0.10
RALLY_MERGE_GAP_SEC_DEFAULT   = 0.22
# Summary：rally 片段中 peak×BW 低于此值的条目单列，便于抽查 visualize_segment
RALLY_LOW_PEAK_BW_SUMMARY_THRESHOLD = 1.0

# COCO-17 下肢关键点索引
LOWER_BODY_KP = [11, 12, 13, 14, 15, 16]

log = logging.getLogger("step4")


# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------

def subject_of(trial: str) -> str:
    m = re.match(r"(sub_\d+)", trial)
    if not m:
        raise ValueError(f"无法从 trial 名称解析 subject：{trial!r}")
    return m.group(1)


def stage_of(trial: str) -> str:
    t = trial.lower()
    for s in ["1", "2", "3"]:
        if f"fatigue_stage{s}" in t or f"fatigue_stage_{s}" in t:
            return f"fatigue_stage{s}"
    for s in ["1", "2", "3"]:
        if f"stage{s}" in t or f"stage_{s}" in t:
            return f"stage{s}"
    return "rally" if "rally" in t else "unknown"


def is_rally_trial(trial: str) -> bool:
    """trial 名含 rally（大小写不敏感）→ 使用 rally 专用峰检路径。"""
    return "rally" in (trial or "").lower()


def p_grf(root: Path, trial: str) -> Path:
    return root / subject_of(trial) / "labels" / f"{trial}_grf.npy"

def p_sync(root: Path, trial: str) -> Path:
    return root / subject_of(trial) / "labels" / f"{trial}_sync.json"

def p_pose(root: Path, trial: str, cam: int) -> Path:
    """step3 v2 唯一约定：{trial}_cam{N}_pose.npz（GRF 对齐全长）。"""
    return root / subject_of(trial) / "pose" / f"{trial}_cam{cam}_pose.npz"

def p_out(root: Path, trial: str, cam: int, idx: int) -> Path:
    return root / subject_of(trial) / "segments" / f"{trial}_cam{cam}_impact_{idx:03d}.npz"


def discover_tasks(root: Path) -> List[Tuple[str, int]]:
    """自动发现 v2 pose：*_cam{N}_pose.npz（排除误匹配）。"""
    tasks = []
    for p in sorted(root.glob("sub_*/pose/*_cam*_pose.npz")):
        m = re.match(r"(.+)_cam(\d+)_pose\.npz$", p.name)
        if m:
            tasks.append((m.group(1), int(m.group(2))))
    return tasks


def discover_trials_for_subject(root: Path, sub: str) -> List[str]:
    labels_dir = root / sub / "labels"
    if not labels_dir.exists():
        return []
    return sorted({
        p.stem[:-5]  # 去掉 _sync 后缀
        for p in labels_dir.glob(f"{sub}_*_sync.json")
    })


def discover_tasks_for_subject_pose(root: Path, sub: str) -> List[Tuple[str, int]]:
    """按实际存在的 pose npz 枚举 (trial, cam)，用于全量切分统计（不依赖 labels 列表）。"""
    pose_dir = root / sub / "pose"
    if not pose_dir.exists():
        return []
    tasks: List[Tuple[str, int]] = []
    for p in sorted(pose_dir.glob("*_cam*_pose.npz")):
        m = re.match(r"(.+)_cam(\d+)_pose\.npz$", p.name)
        if m:
            tasks.append((m.group(1), int(m.group(2))))
    return tasks


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class SyncInfo:
    def __init__(self, trial, subject, grf_event_sec, video_event_sec,
                 video_event_frame, video_fps, offset_sec,
                 offset_uncertainty_sec, quality):
        self.trial                  = trial
        self.subject                = subject
        self.grf_event_sec          = grf_event_sec
        self.video_event_sec        = video_event_sec
        self.video_event_frame      = video_event_frame
        self.video_fps              = video_fps
        self.offset_sec             = offset_sec
        self.offset_uncertainty_sec = offset_uncertainty_sec
        self.quality                = quality


class ImpactEvent:
    """一个检测到的落地冲击事件"""
    def __init__(self, grf_peak_sec, grf_peak_N, grf_peak_bw,
                 pose_ev_local, pose_ev_abs, impact_idx, is_sync_impact):
        self.grf_peak_sec   = grf_peak_sec
        self.grf_peak_N     = grf_peak_N
        self.grf_peak_bw    = grf_peak_bw
        self.pose_ev_local  = pose_ev_local
        self.pose_ev_abs    = pose_ev_abs
        self.impact_idx     = impact_idx
        self.is_sync_impact = is_sync_impact


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_sync(root: Path, trial: str, cam: int
              ) -> Tuple[Optional[SyncInfo], Optional[str]]:
    """读取 sync.json，优先使用 per_cam[cam] 的标注"""
    p = p_sync(root, trial)
    if not p.exists():
        return None, f"sync.json 不存在：{p}"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"sync.json 读取失败：{e}"

    grf_ev = d.get("grf_event_sec")
    if grf_ev is None:
        return None, "sync.json 缺少 grf_event_sec"

    cam_data = d.get("per_cam", {}).get(str(cam))
    if cam_data:
        vef = int(cam_data.get("video_event_frame", d.get("video_event_frame", 0)) or 0)
        ves = float(cam_data.get("video_event_sec",   d.get("video_event_sec",   0.0)) or 0.0)
        off = float(cam_data.get("offset_sec",        d.get("offset_sec",        0.0)) or 0.0)
    else:
        vef = int(d.get("video_event_frame", 0) or 0)
        ves = float(d.get("video_event_sec",   0.0) or 0.0)
        off = float(d.get("offset_sec",        0.0) or 0.0)

    fps = float(d.get("video_fps", VIDEO_FPS_DEFAULT) or VIDEO_FPS_DEFAULT)
    if fps <= 0:
        fps = VIDEO_FPS_DEFAULT
    unc = float(d.get("offset_uncertainty_sec", 1.0 / fps) or (1.0 / fps))

    return SyncInfo(
        trial=trial,
        subject=subject_of(trial),
        grf_event_sec=float(grf_ev),
        video_event_sec=ves,
        video_event_frame=vef,
        video_fps=fps,
        offset_sec=off,
        offset_uncertainty_sec=unc,
        quality=str(d.get("quality", "good")),
    ), None


def load_grf(root: Path, trial: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    加载 .npy GRF 文件。

    力台坐标系：Z 轴向上为正，运动员踩下为负。
    本函数统一取负，使 fz_contact 正值=接触力。

    返回 dict：
      timestamps   (N,) float64   [s]
      fz_contact   (N,) float64   合并 Fz，正值=接触力 [N]
      fx_contact   (N,) float64   合并 Fx
      fy_contact   (N,) float64   合并 Fy
      plates       {"plate1"~"plate4": (N,) float64}  各板 Fz（已取负）
      rate         float           采样率 Hz（实测 1200）
      bw_N         float           体重估算 [N]
    """
    path = p_grf(root, trial)
    if not path.exists():
        return None, f"GRF 文件不存在：{path}"
    try:
        raw = np.load(str(path), allow_pickle=True)
        d = raw.item() if raw.dtype == object else None
    except Exception as e:
        return None, f"GRF 读取失败：{e}"

    if not isinstance(d, dict):
        return None, f"GRF 格式异常（期望 dict）：{path}"
    if "timestamps" not in d:
        return None, f"GRF 缺少 timestamps：{path}"

    ts = np.asarray(d["timestamps"], dtype=np.float64)
    N  = len(ts)

    combined = d.get("combined", {})
    forces   = np.asarray(combined.get("forces", np.zeros((N, 3))), dtype=np.float64)
    fz = -forces[:, 2]
    fx =  forces[:, 0]
    fy =  forces[:, 1]

    plates = {}
    for i in range(1, 5):
        key = f"force_plate_{i}"
        if key in d:
            fp = np.asarray(d[key].get("forces", np.zeros((N, 3))), dtype=np.float64)
            plates[f"plate{i}"] = -fp[:, 2]
        else:
            plates[f"plate{i}"] = np.zeros(N, dtype=np.float64)

    meta = d.get("metadata", {})
    rate = float(meta.get("analog_rate", meta.get("rate", GRF_RATE_DEFAULT)) or GRF_RATE_DEFAULT)
    if rate <= 0:
        rate = GRF_RATE_DEFAULT

    # 体重估算：取静止站立段（Fz 在合理体重范围内）中位数
    active = (fz > BW_FZ_MIN) & (fz < BW_FZ_MAX)
    if active.sum() > int(rate * 0.3):
        bw = float(np.median(fz[active]))
    else:
        nz = fz[fz > 50]
        bw = float(np.median(nz)) if len(nz) > 0 else 700.0

    return dict(
        timestamps=ts, fz_contact=fz, fx_contact=fx, fy_contact=fy,
        plates=plates, rate=rate, bw_N=bw,
    ), None


def load_pose(root: Path, trial: str, cam: int) -> Tuple[Optional[Dict], Optional[str]]:
    path = p_pose(root, trial, cam)
    if not path.exists():
        return None, f"pose npz 不存在：{path}"
    try:
        d = dict(np.load(str(path), allow_pickle=True))
    except Exception as e:
        return None, f"pose npz 读取失败：{e}"
    for key in ("keypoints", "scores", "frame_indices",
                "event_frame_abs", "event_frame_local"):
        if key not in d:
            return None, f"pose npz 缺少字段 {key!r}：{path}"
    ver = str(d.get("schema_version", "") or "")
    cov = str(d.get("pose_coverage", "") or "")
    if ver != "pose_npz_v2" or cov != "grf_aligned":
        return None, (
            f"pose 须为 step3 v2（schema_version=pose_npz_v2, pose_coverage=grf_aligned）：{path}"
        )
    return d, None


# ---------------------------------------------------------------------------
# Phase A：稳健 MAD、粗检峰合并
# ---------------------------------------------------------------------------

def _robust_mad(x: np.ndarray) -> float:
    """Median absolute deviation（与 median 同尺度，用于窗口内相对噪声）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _merge_peaks_by_time(
    ts_local: np.ndarray,
    fz_local: np.ndarray,
    peak_idx: np.ndarray,
    merge_gap_sec: float,
) -> np.ndarray:
    """
    将时间间隔 ≤ merge_gap_sec 的峰合并为一次冲击，保留 |Fz| 最大（此处 fz 为接触力，取最大）。
    """
    peak_idx = np.asarray(peak_idx, dtype=np.int64)
    if peak_idx.size <= 1:
        return peak_idx
    order = np.argsort(ts_local[peak_idx])
    idx_sorted = peak_idx[order]
    merged: List[int] = []
    cur_group: List[int] = [int(idx_sorted[0])]
    for i in range(1, len(idx_sorted)):
        t_prev = float(ts_local[cur_group[-1]])
        t_cur = float(ts_local[int(idx_sorted[i])])
        if t_cur - t_prev <= merge_gap_sec:
            cur_group.append(int(idx_sorted[i]))
        else:
            best = max(cur_group, key=lambda ix: float(fz_local[ix]))
            merged.append(best)
            cur_group = [int(idx_sorted[i])]
    merged.append(max(cur_group, key=lambda ix: float(fz_local[ix])))
    return np.array(sorted(set(merged)), dtype=np.int64)


def _adaptive_thresholds(fz_smooth: np.ndarray, bw: float) -> Tuple[float, float]:
    """
    返回 (height_min, prominence_min)，单位 N。
    height ≥ max(PEAK_HEIGHT_BW×BW, median+K_h·MAD)，prominence 同理。
    """
    mad = max(_robust_mad(fz_smooth), 1.0)
    med = float(np.median(fz_smooth))
    height = max(bw * PEAK_HEIGHT_BW, med + ADAPT_HEIGHT_MAD_K * mad)
    prom   = max(bw * PEAK_PROMINENCE_BW, ADAPT_PROM_MAD_K * mad)
    return height, prom


def _coarse_thresholds(fz_smooth: np.ndarray, bw: float) -> Tuple[float, float]:
    """粗检：略低固定倍数 + 略松的 median+MAD，提高召回。"""
    mad = max(_robust_mad(fz_smooth), 1.0)
    med = float(np.median(fz_smooth))
    kh = max(0.5, ADAPT_HEIGHT_MAD_K - 1.5)
    kp = max(0.5, ADAPT_PROM_MAD_K - 0.5)
    height = max(bw * PEAK_HEIGHT_COARSE_BW, med + kh * mad)
    prom   = max(bw * PEAK_PROMINENCE_COARSE_BW, kp * mad)
    return height, prom


def compute_adaptive_pre_post_sec(
    ts: np.ndarray,
    fz: np.ndarray,
    peak_sec: float,
    bw: float,
    rate: float,
    *,
    grf_peak_N: Optional[float] = None,
) -> Tuple[float, float]:
    """
    在完整 GRF 时间序列上，根据 Fz 形状估计峰值前/后应保留的秒数（相对峰值）。

    - 前：峰前窗口内最后一次「低位→上台面」交叉；无则回退 WINDOW_PRE_SEC。
    - 后：峰值后延迟，再判持续低于 max(stance_end×BW, WIN_POST_REL_PEAK×峰力)。
      峰力：在 grf_peak_sec±WIN_PEAK_LOCAL_HALF_SEC 内对 raw Fz 取 max，并与 grf_peak_N 取大（避免最近邻采样偏离峰顶）。
    - 再按 peak_fz/BW 追加 post（高冲击需更长恢复段；否则曲线法常在 j0 即满足而被 MIN 顶满）。
    边界用 50Hz 低通 Fz，与峰检一致；结果夹在 [MIN_PRE/POST, MAX_PRE/POST]。
    """
    from scipy.signal import butter, filtfilt

    ts = np.asarray(ts, dtype=np.float64)
    fz = np.asarray(fz, dtype=np.float64)
    if ts.size < 128 or bw <= 1e-6 or not np.isfinite(peak_sec) or peak_sec <= 1e-6:
        return WINDOW_PRE_SEC, WINDOW_POST_SEC

    try:
        b, a = butter(4, BUTTER_CUTOFF_HZ / (rate / 2.0), btype="low")
        fz_s = filtfilt(b, a, fz)
    except Exception:
        fz_s = fz

    peak_idx = int(np.argmin(np.abs(ts - peak_sec)))
    peak_idx = max(0, min(peak_idx, len(fz_s) - 1))

    flight = WIN_FLIGHT_BW * bw
    on_edge = WIN_ON_PLATE_BW * bw * 0.55
    stance_end = WIN_STANCE_END_BW * bw
    sustain = max(3, int(WIN_SUSTAIN_SEC * rate))
    sustain_post = max(3, int(WIN_POST_SUSTAIN_SEC * rate))
    t_lo = float(peak_sec) - WIN_PEAK_LOCAL_HALF_SEC
    t_hi = float(peak_sec) + WIN_PEAK_LOCAL_HALF_SEC
    tmask = (ts >= t_lo) & (ts <= t_hi)
    if np.any(tmask):
        peak_fz = float(max(float(np.max(fz[tmask])), stance_end))
    else:
        peak_fz = float(max(float(fz[peak_idx]), stance_end))
    if grf_peak_N is not None and np.isfinite(grf_peak_N) and grf_peak_N > stance_end:
        peak_fz = max(peak_fz, float(grf_peak_N))
    thr_post = max(stance_end, WIN_POST_REL_PEAK * peak_fz)

    t_peak = float(ts[peak_idx])

    # ── 前：峰前窗口内最后一次 flight→plate 交叉 ───────────────────────────
    lo = max(0, peak_idx - int(WIN_ADAPT_MAX_PRE_SEC * rate))
    onset_idx = max(lo, peak_idx - int(WINDOW_PRE_SEC * rate))
    last_onset: Optional[int] = None
    for i in range(lo, max(lo, peak_idx - 1)):
        if fz_s[i] < flight and fz_s[i + 1] >= on_edge:
            last_onset = i + 1
    if last_onset is not None:
        onset_idx = last_onset
    t_onset = float(ts[onset_idx])
    pre_sec = t_peak - t_onset + 0.03
    pre_sec = max(WIN_ADAPT_MIN_PRE_SEC, min(WIN_ADAPT_MAX_PRE_SEC, pre_sec))

    # ── 后：峰值后延迟再判支撑末（跳过冲击主衰减）────────────────────────────
    hi = min(len(fz_s) - sustain_post - 1, peak_idx + int(WIN_ADAPT_MAX_POST_SEC * rate))
    j0 = peak_idx + max(1, int(WIN_POST_DELAY_SEC * rate))
    post_sec = 0.88
    for j in range(j0, hi + 1):
        if j + sustain_post <= len(fz_s) and bool(
            np.all(fz_s[j : j + sustain_post] < thr_post)
        ):
            t_off = float(ts[j + sustain_post // 2])
            post_sec = t_off - t_peak + 0.04
            break
    post_sec = max(WIN_ADAPT_MIN_POST_SEC, min(WIN_ADAPT_MAX_POST_SEC, post_sec))

    peak_bw_ratio = peak_fz / max(bw, 1.0)
    floor_sec = _peak_bw_post_floor_sec(peak_bw_ratio)
    if floor_sec > 0:
        post_sec = max(post_sec, min(WIN_ADAPT_MAX_POST_SEC, floor_sec))
        post_sec = max(WIN_ADAPT_MIN_POST_SEC, min(WIN_ADAPT_MAX_POST_SEC, post_sec))

    return float(pre_sec), float(post_sec)


def _detect_grf_impacts_rally(
    ts_win: np.ndarray,
    fz_win: np.ndarray,
    bw: float,
    rate: float,
    sync: SyncInfo,
    pose: Dict,
    fps: float,
    *,
    window_mode: str,
) -> List[ImpactEvent]:
    """
    Rally 专用：乱波形下不用 strict median+MAD 门控。
    12Hz 低通 → 宽松 find_peaks → 时间合并 → 仅保留 Fz≥RALLY_PEAK_MIN_FZ_BW×BW 的峰。
    """
    from scipy.signal import butter, filtfilt, find_peaks

    ts_win = np.asarray(ts_win, dtype=np.float64)
    fz_win = np.asarray(fz_win, dtype=np.float64)
    try:
        b, a = butter(4, RALLY_BUTTER_CUTOFF_HZ / (rate / 2.0), btype="low")
        fz_s = filtfilt(b, a, fz_win)
    except Exception:
        fz_s = fz_win

    mad = max(_robust_mad(fz_s), 1.0)
    med = float(np.median(fz_s))
    dist_c = max(2, int(RALLY_PEAK_DISTANCE_COARSE_SEC * rate))
    prom_floor = max(bw * RALLY_PEAK_PROMINENCE_COARSE, 0.35 * mad)
    h_floor = max(bw * RALLY_PEAK_HEIGHT_COARSE_BW, med + 0.9 * mad)

    coarse_idx, _ = find_peaks(
        fz_s,
        height=h_floor,
        distance=dist_c,
        prominence=prom_floor,
    )
    if coarse_idx.size == 0:
        h2 = max(bw * 0.18, med + 0.35 * mad)
        p2 = max(bw * 0.04, 0.22 * mad)
        coarse_idx, _ = find_peaks(
            fz_s,
            height=h2,
            distance=dist_c,
            prominence=p2,
        )
    if coarse_idx.size == 0:
        coarse_idx, _ = find_peaks(
            fz_s,
            height=max(bw * 0.15, 45.0),
            distance=dist_c,
            prominence=max(2.5, 0.12 * mad),
        )

    merged_idx = _merge_peaks_by_time(
        ts_win, fz_win, coarse_idx, RALLY_MERGE_GAP_SEC_DEFAULT,
    )
    min_fz = bw * RALLY_PEAK_MIN_FZ_BW
    peaks_local = np.asarray(
        [int(i) for i in merged_idx if float(fz_win[int(i)]) >= min_fz],
        dtype=np.int64,
    )
    if peaks_local.size == 0:
        return []

    if window_mode == "fixed":
        pre_max = int(round(WINDOW_PRE_SEC * fps))
        post_max = int(round(WINDOW_POST_SEC * fps))
    else:
        pre_max = int(round(WIN_ADAPT_MAX_PRE_SEC * fps))
        post_max = int(round(WIN_ADAPT_MAX_POST_SEC * fps))

    fidx = np.asarray(pose["frame_indices"], dtype=np.int64)
    n_total = len(fidx)
    ev_abs = int(pose["event_frame_abs"])

    impacts_raw = []
    for pk_l in peaks_local:
        grf_peak_sec = float(ts_win[pk_l])
        grf_peak_N = float(fz_win[pk_l])
        t_video = grf_peak_sec - sync.offset_sec
        frame_float = ev_abs + (t_video - sync.video_event_sec) * fps
        diffs = np.abs(fidx.astype(float) - frame_float)
        pose_ev_local = int(np.argmin(diffs))
        if pose_ev_local - pre_max < 0 or pose_ev_local + post_max > n_total:
            log.debug("  [rally] 峰值 t=%.3fs 离 pose 边界太近，跳过", grf_peak_sec)
            continue
        impacts_raw.append(dict(
            grf_peak_sec=grf_peak_sec,
            grf_peak_N=grf_peak_N,
            grf_peak_bw=grf_peak_N / max(bw, 1.0),
            pose_ev_local=pose_ev_local,
            pose_ev_abs=int(fidx[pose_ev_local]),
        ))

    if not impacts_raw:
        return []

    impacts_raw.sort(key=lambda x: x["grf_peak_sec"])
    min_sep = pre_max + post_max
    deduped: List[Dict] = []
    for imp in impacts_raw:
        merged = False
        for k in deduped:
            if abs(imp["pose_ev_local"] - k["pose_ev_local"]) < min_sep:
                if imp["grf_peak_N"] > k["grf_peak_N"]:
                    deduped[deduped.index(k)] = imp
                merged = True
                break
        if not merged:
            deduped.append(imp)
    deduped.sort(key=lambda x: x["grf_peak_sec"])

    sync_t = sync.grf_event_sec
    sync_match = min(
        range(len(deduped)),
        key=lambda i: abs(deduped[i]["grf_peak_sec"] - sync_t),
    )
    events = []
    for rank, imp in enumerate(deduped):
        events.append(ImpactEvent(
            grf_peak_sec=float(imp["grf_peak_sec"]),
            grf_peak_N=float(imp["grf_peak_N"]),
            grf_peak_bw=float(imp["grf_peak_bw"]),
            pose_ev_local=int(imp["pose_ev_local"]),
            pose_ev_abs=int(imp["pose_ev_abs"]),
            impact_idx=rank + 1,
            is_sync_impact=(rank == sync_match),
        ))
    return events


# ---------------------------------------------------------------------------
# GRF 峰值检测（核心新增）
# ---------------------------------------------------------------------------

def detect_grf_impacts(
    grf:   Dict,
    sync:  SyncInfo,
    pose:  Dict,
    fps:   float,
    *,
    peak_mode: str = "adaptive",
    merge_gap_sec: float = MERGE_GAP_SEC,
    window_mode: str = "adaptive",
    trial: Optional[str] = None,
    rally_special: bool = True,
) -> List[ImpactEvent]:
    """
    在 pose.npz 覆盖的视频窗口内，自动检测所有有效落地冲击峰值。

    peak_mode=fixed：与旧版相同（固定 0.5×BW / 250ms / 0.2×BW），无粗检合并。
    peak_mode=adaptive（默认）：
      1. 同下：时间窗 + 50Hz 低通（仅用于检测）
      2. 粗检 find_peaks（略低阈值、短 min-distance）
      3. 按 MERGE_GAP_SEC 合并时间邻近峰（双脚/双极值）
      4. 用「严格」自适应阈值（median+K·MAD 与固定 BW 下限取 max）过滤假峰
      5. 若仍无峰：再跑一次 strict find_peaks（distance=250ms）
      6. 映射 pose 帧、边界、窗口去重、标记 sync

    若未检测到任何有效峰值，回退到 sync event 单峰模式。

    trial 名含 ``rally`` 且 ``rally_special`` 为真时，先走 ``_detect_grf_impacts_rally``，
    失败再使用下方标准 adaptive 流程。
    """
    from scipy.signal import butter, filtfilt, find_peaks

    ts   = grf["timestamps"]
    fz   = grf["fz_contact"]
    bw   = grf["bw_N"]
    rate = grf["rate"]

    fidx    = np.asarray(pose["frame_indices"], dtype=np.int64)
    n_total = len(fidx)
    ev_abs  = int(pose["event_frame_abs"])   # sync 事件的视频绝对帧号

    # ── 1. pose 覆盖的 GRF 时间范围 ──────────────────────────────────────────
    # 视频时间：t_video = sync.video_event_sec + (frame - ev_abs) / fps
    # GRF 时间：t_grf  = t_video + sync.offset_sec
    t_grf_start = sync.video_event_sec + (float(fidx[0])  - ev_abs) / fps + sync.offset_sec
    t_grf_end   = sync.video_event_sec + (float(fidx[-1]) - ev_abs) / fps + sync.offset_sec

    # 加小裕度，防止边界浮点舍入
    mask = (ts >= t_grf_start - 0.05) & (ts <= t_grf_end + 0.05)
    if not mask.any():
        log.warning("  GRF 时间窗口内无数据，退回 sync event 单峰模式")
        return _fallback_sync_impact(pose, fps, bw)

    ts_win = ts[mask]
    fz_win = fz[mask]

    if rally_special and trial and is_rally_trial(trial) and peak_mode == "adaptive":
        rally_ev = _detect_grf_impacts_rally(
            ts_win, fz_win, bw, rate, sync, pose, fps,
            window_mode=window_mode,
        )
        if rally_ev:
            log.info("  rally 专用峰检：%d 个冲击", len(rally_ev))
            return rally_ev
        log.info("  rally 专用峰检未得到有效峰，回退标准 adaptive 峰检")

    # ── 2. Butterworth 低通滤波（仅用于检测） ────────────────────────────────
    try:
        b, a = butter(4, BUTTER_CUTOFF_HZ / (rate / 2.0), btype="low")
        fz_smooth = filtfilt(b, a, fz_win)
    except Exception:
        fz_smooth = fz_win  # 滤波失败则用原始信号

    peaks_local: np.ndarray
    if peak_mode == "fixed" or len(fz_smooth) < 16:
        # 与旧版一致（短窗不用自适应）
        peaks_local, _ = find_peaks(
            fz_smooth,
            height     = bw * PEAK_HEIGHT_BW,
            distance   = max(1, int(PEAK_DISTANCE_SEC * rate)),
            prominence = bw * PEAK_PROMINENCE_BW,
        )
        if peak_mode == "adaptive" and len(fz_smooth) < 16:
            log.debug("  Fz 窗口过短，peak_mode 退化为 fixed 单-pass")
    else:
        h_strict, p_strict = _adaptive_thresholds(fz_smooth, bw)
        h_coarse, p_coarse = _coarse_thresholds(fz_smooth, bw)
        dist_coarse = max(2, int(PEAK_DISTANCE_COARSE_SEC * rate))

        coarse_idx, _ = find_peaks(
            fz_smooth,
            height=h_coarse,
            distance=dist_coarse,
            prominence=p_coarse,
        )
        merged_idx = _merge_peaks_by_time(ts_win, fz_win, coarse_idx, merge_gap_sec)
        # 严格门控：抑制粗检引入的踏步噪声
        keep = [int(i) for i in merged_idx if float(fz_win[int(i)]) >= h_strict]
        peaks_local = np.asarray(keep, dtype=np.int64)
        n_gate = len(peaks_local)

        if peaks_local.size == 0:
            peaks_local, _ = find_peaks(
                fz_smooth,
                height=h_strict,
                distance=max(1, int(PEAK_DISTANCE_SEC * rate)),
                prominence=p_strict,
            )

        med = float(np.median(fz_smooth))
        mad = _robust_mad(fz_smooth)
        log.info(
            "  GRF峰检 adaptive: med=%.0fN MAD=%.1fN  strict_h≥%.0fN  "
            "coarse=%d merge=%d gate=%d final=%d",
            med, mad, h_strict,
            len(coarse_idx), len(merged_idx), n_gate, len(peaks_local),
        )

    if len(peaks_local) == 0:
        log.warning(
            "  未检测到有效 GRF 峰值（mode=%s threshold≈%.0fN），退回 sync event 单峰",
            peak_mode, bw * PEAK_HEIGHT_BW,
        )
        return _fallback_sync_impact(pose, fps, bw)

    # ── 4. 映射到 pose 帧，验证裕度 ─────────────────────────────────────────
    if window_mode == "fixed":
        pre_max  = int(round(WINDOW_PRE_SEC * fps))
        post_max = int(round(WINDOW_POST_SEC * fps))
    else:
        pre_max  = int(round(WIN_ADAPT_MAX_PRE_SEC * fps))
        post_max = int(round(WIN_ADAPT_MAX_POST_SEC * fps))

    impacts_raw = []
    for pk_l in peaks_local:
        grf_peak_sec = float(ts_win[pk_l])
        grf_peak_N   = float(fz_win[pk_l])   # 使用原始（未平滑）值

        # GRF 时间 → 视频时间 → 最近 pose 帧
        t_video     = grf_peak_sec - sync.offset_sec
        frame_float = ev_abs + (t_video - sync.video_event_sec) * fps
        diffs       = np.abs(fidx.astype(float) - frame_float)
        pose_ev_local = int(np.argmin(diffs))

        # 边界检查（两侧均需有足够帧数；adaptive 按最大可能窗）
        if pose_ev_local - pre_max < 0 or pose_ev_local + post_max > n_total:
            log.debug("  峰值 t=%.3fs 离边界太近，跳过", grf_peak_sec)
            continue

        impacts_raw.append(dict(
            grf_peak_sec  = grf_peak_sec,
            grf_peak_N    = grf_peak_N,
            grf_peak_bw   = grf_peak_N / max(bw, 1.0),
            pose_ev_local = pose_ev_local,
            pose_ev_abs   = int(fidx[pose_ev_local]),
        ))

    if not impacts_raw:
        log.warning("  所有峰值均因边界不足被跳过，退回 sync event 单峰模式")
        return _fallback_sync_impact(pose, fps, bw)

    # ── 5. 按时间排序 ────────────────────────────────────────────────────────
    impacts_raw.sort(key=lambda x: x["grf_peak_sec"])

    # ── 6. 窗口重叠去重（核心修复） ──────────────────────────────────────────
    #
    # 判断标准：pose 窗口是否实质重叠（而非峰值数值是否相同）
    #
    # 若两者的 pose_ev_local 差距 < min_sep，说明采样窗口大量重叠，
    # 属于同一次落地被重复检测（通常发生在双脚依次落地间隔 <250ms，
    # 或 scipy 在同一峰值的上升沿/下降沿产生两个相邻极值时）。
    # 保留峰值更高的那个（更有代表性）。
    #
    min_sep = pre_max + post_max

    deduped: List[Dict] = []
    for imp in impacts_raw:
        merged = False
        for k in deduped:
            if abs(imp["pose_ev_local"] - k["pose_ev_local"]) < min_sep:
                # 窗口重叠 → 同一事件，保留峰值更高的
                if imp["grf_peak_N"] > k["grf_peak_N"]:
                    deduped[deduped.index(k)] = imp
                    log.debug(
                        "  去重：t=%.3fs(%.0fN) 替换 t=%.3fs(%.0fN)（窗口重叠）",
                        imp["grf_peak_sec"], imp["grf_peak_N"],
                        k["grf_peak_sec"],  k["grf_peak_N"],
                    )
                else:
                    log.debug(
                        "  去重：跳过 t=%.3fs(%.0fN)（与 t=%.3fs 窗口重叠）",
                        imp["grf_peak_sec"], imp["grf_peak_N"],
                        k["grf_peak_sec"],
                    )
                merged = True
                break
        if not merged:
            deduped.append(imp)

    n_removed = len(impacts_raw) - len(deduped)
    if n_removed > 0:
        log.info("  窗口去重：移除 %d 个重复峰值（%d → %d）",
                 n_removed, len(impacts_raw), len(deduped))

    # 去重后重新按时间排序（替换操作可能打乱顺序）
    deduped.sort(key=lambda x: x["grf_peak_sec"])

    # ── 7. 标记 sync event ───────────────────────────────────────────────────
    sync_t     = sync.grf_event_sec
    sync_match = min(
        range(len(deduped)),
        key=lambda i: abs(deduped[i]["grf_peak_sec"] - sync_t),
    )

    events = []
    for rank, imp in enumerate(deduped):
        events.append(ImpactEvent(
            grf_peak_sec   = imp["grf_peak_sec"],
            grf_peak_N     = imp["grf_peak_N"],
            grf_peak_bw    = imp["grf_peak_bw"],
            pose_ev_local  = imp["pose_ev_local"],
            pose_ev_abs    = imp["pose_ev_abs"],
            impact_idx     = rank + 1,
            is_sync_impact = (rank == sync_match),
        ))

    return events


def _fallback_sync_impact(pose: Dict, fps: float, bw: float) -> List[ImpactEvent]:
    """回退：使用 sync event 作为唯一冲击事件；必要时收缩前后窗以适配 pose 长度。"""
    ev_local = int(pose["event_frame_local"])
    fidx     = np.asarray(pose["frame_indices"], dtype=np.int64)
    for scale in (1.0, 0.78, 0.6, 0.45, 0.32, 0.24):
        pre_sec = WINDOW_PRE_SEC * scale
        post_sec = WINDOW_POST_SEC * scale
        pre = int(round(pre_sec * fps))
        post = int(round(post_sec * fps))
        if ev_local - pre >= 0 and ev_local + post <= len(fidx):
            if scale < 1.0:
                log.info(
                    "  sync 单峰回退：窗长收缩为 pre≈%.2fs post≈%.2fs（scale=%.2f）",
                    pre_sec, post_sec, scale,
                )
            return [ImpactEvent(
                grf_peak_sec=0.0,
                grf_peak_N=0.0,
                grf_peak_bw=0.0,
                pose_ev_local=ev_local,
                pose_ev_abs=int(fidx[ev_local]),
                impact_idx=1,
                is_sync_impact=True,
            )]
    return []


# ---------------------------------------------------------------------------
# 单冲击切片
# ---------------------------------------------------------------------------

def _interp_grf(
    sync:  SyncInfo,
    pose:  Dict,
    grf:   Dict,
    start: int,
    end:   int,
    fps:   float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 GRF [Fx,Fy,Fz] 插值到视频帧时间点。
    返回 grf_at_fps (T,3)、ts_video (T,)。

    视频帧 → 绝对视频时间：
      t_video = sync.video_event_sec + (frame - ev_abs) / fps
    视频时间 → GRF 时间：
      t_grf = t_video + sync.offset_sec
    """
    fidx   = np.asarray(pose["frame_indices"], dtype=np.int64)[start:end]
    ev_abs = int(pose["event_frame_abs"])

    ts_video  = sync.video_event_sec + (fidx - ev_abs) / fps
    ts_grf_q  = ts_video + sync.offset_sec
    ts_full   = grf["timestamps"]

    def _interp(sig: np.ndarray) -> np.ndarray:
        return np.interp(
            ts_grf_q, ts_full, sig,
            left=float(sig[0]), right=float(sig[-1]),
        ).astype(np.float32)

    return (
        np.stack([_interp(grf["fx_contact"]),
                  _interp(grf["fy_contact"]),
                  _interp(grf["fz_contact"])], axis=1),
        ts_video.astype(np.float32),
    )


def _build_grf_hf_window(
    grf_peak_sec: float,
    grf:          Dict,
    *,
    pre_sec: float = WINDOW_PRE_SEC,
    post_sec: float = WINDOW_POST_SEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    裁出以 GRF 峰值为中心的高频窗口。
    返回 grf_1200hz (M,7) = [Fx,Fy,Fz,Fz1,Fz2,Fz3,Fz4]、timestamps_grf (M,)。
    """
    ts = grf["timestamps"]

    if grf_peak_sec > 0:
        t0 = grf_peak_sec - pre_sec
        t1 = grf_peak_sec + post_sec
        m  = (ts >= t0) & (ts <= t1)
    else:
        # 回退：用整段
        m = np.ones(len(ts), bool)

    if not m.any():
        m = np.ones(len(ts), bool)

    cols = [
        grf["fx_contact"][m],
        grf["fy_contact"][m],
        grf["fz_contact"][m],
        grf["plates"]["plate1"][m],
        grf["plates"]["plate2"][m],
        grf["plates"]["plate3"][m],
        grf["plates"]["plate4"][m],
    ]
    return np.stack(cols, axis=1).astype(np.float32), ts[m].astype(np.float32)


def _segment_one_impact(
    root:    Path,
    trial:   str,
    cam:     int,
    sync:    SyncInfo,
    grf:     Dict,
    pose:    Dict,
    impact:  ImpactEvent,
    n_total_impacts: int,
    fps:     float,
    W:       int,
    H:       int,
    dry_run: bool,
    *,
    window_mode: str = "adaptive",
) -> Dict:
    """
    切出单个冲击事件的窗口并保存 npz。
    窗口以 GRF 峰值为中心（impact.pose_ev_local）。
    """
    result = {"trial": trial, "cam": cam,
              "impact_idx": impact.impact_idx, "status": "error"}

    ev_local = int(impact.pose_ev_local)
    bw = float(grf["bw_N"])
    rate = float(grf["rate"])
    if window_mode == "fixed" or impact.grf_peak_sec <= 1e-6:
        pre_sec = WINDOW_PRE_SEC
        post_sec = WINDOW_POST_SEC
    else:
        pre_sec, post_sec = compute_adaptive_pre_post_sec(
            grf["timestamps"],
            grf["fz_contact"],
            float(impact.grf_peak_sec),
            bw,
            rate,
            grf_peak_N=float(impact.grf_peak_N),
        )

    pre_f  = int(round(pre_sec * fps))
    post_f = int(round(post_sec * fps))
    n_pose = len(pose["frame_indices"])
    start = max(0, ev_local - pre_f)
    end   = min(n_pose, ev_local + post_f)
    ev_idx = ev_local - start
    T = end - start

    if T <= 0 or ev_idx < 0 or ev_idx >= T:
        return {**result, "message": "空窗口或峰值越界"}

    # ── Pose 切片 + 视频率 GRF（插值峰可能比 raw 更贴近你看到的 peak×BW）──────────
    kps    = np.asarray(pose["keypoints"],    np.float32)[start:end]
    scores = np.asarray(pose["scores"],       np.float32)[start:end]
    fidx   = np.asarray(pose["frame_indices"],np.int32  )[start:end]
    tstat  = np.asarray(
        pose.get("track_status",
                 np.full(len(pose["frame_indices"]), "unknown")),
        dtype="U20",
    )[start:end]

    kps_norm = kps / np.array([W, H], np.float32)[None, None, :]

    grf_fps, ts_video = _interp_grf(sync, pose, grf, start, end, fps)
    if window_mode == "adaptive" and impact.grf_peak_sec > 1e-6:
        peak_bw_interp = float(np.max(grf_fps[:, 2])) / max(bw, 1.0)
        floor_sec = _peak_bw_post_floor_sec(peak_bw_interp)
        if floor_sec > 0:
            new_post = max(post_sec, min(WIN_ADAPT_MAX_POST_SEC, floor_sec))
            new_post = max(WIN_ADAPT_MIN_POST_SEC, min(WIN_ADAPT_MAX_POST_SEC, new_post))
            if new_post > post_sec + 1e-5:
                post_sec = new_post
                post_f = int(round(post_sec * fps))
                end = min(n_pose, ev_local + post_f)
                T = end - start
                if T <= 0 or ev_idx < 0 or ev_idx >= T:
                    return {**result, "message": "空窗口或峰值越界"}
                kps = np.asarray(pose["keypoints"], np.float32)[start:end]
                scores = np.asarray(pose["scores"], np.float32)[start:end]
                fidx = np.asarray(pose["frame_indices"], np.int32)[start:end]
                tstat = np.asarray(
                    pose.get(
                        "track_status",
                        np.full(len(pose["frame_indices"]), "unknown"),
                    ),
                    dtype="U20",
                )[start:end]
                kps_norm = kps / np.array([W, H], np.float32)[None, None, :]
                grf_fps, ts_video = _interp_grf(sync, pose, grf, start, end, fps)

    grf_hf, ts_grf = _build_grf_hf_window(
        impact.grf_peak_sec, grf, pre_sec=pre_sec, post_sec=post_sec,
    )

    grf_norm = grf_fps / max(bw, 1.0)

    fz_win     = grf_fps[:, 2]
    peak_force = float(np.max(fz_win))
    peak_bw    = peak_force / max(bw, 1.0)

    # ── 质量统计 ─────────────────────────────────────────────────────────────
    lower_mean = float(scores[:, LOWER_BODY_KP].mean())
    lost_rate  = float((tstat == "lost").mean())

    # ── Dry run ──────────────────────────────────────────────────────────────
    if dry_run:
        log.info(
            "  [DRY] #%03d  T=%d  ev_idx=%d  pre=%.2fs post=%.2fs  grf_peak=%.3fs"
            "  peak=%dN(%.2f×BW)  lower=%.3f  lost=%.1f%%  sync=%s",
            impact.impact_idx, T, ev_idx, pre_sec, post_sec,
            impact.grf_peak_sec, peak_force, peak_bw,
            lower_mean, lost_rate * 100, impact.is_sync_impact,
        )
        return {**result, "status": "dry_run", "T": T,
                "peak_bw": round(peak_bw, 3),
                "window_pre_sec": pre_sec, "window_post_sec": post_sec,
                "window_mode": window_mode}

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out = p_out(root, trial, cam, impact.impact_idx)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(out),
        # pose
        keypoints_norm   = kps_norm,
        keypoints_px     = kps,
        scores           = scores,
        frame_indices    = fidx,
        track_status     = tstat,
        # GRF
        grf_at_video_fps = grf_fps,
        grf_normalized   = grf_norm,
        grf_1200hz       = grf_hf,
        timestamps_video = ts_video,
        timestamps_grf   = ts_grf,
        # 事件索引（GRF 峰值在序列内的局部索引）
        ev_idx           = np.int32(ev_idx),
        # 元数据
        trial            = np.str_(trial),
        subject          = np.str_(subject_of(trial)),
        stage            = np.str_(stage_of(trial)),
        camera           = np.int32(cam),
        quality          = np.str_(sync.quality),
        video_fps        = np.float32(fps),
        grf_rate         = np.float32(grf["rate"]),
        offset_sec       = np.float32(sync.offset_sec),
        offset_uncertainty_sec = np.float32(sync.offset_uncertainty_sec),
        image_width      = np.int32(W),
        image_height     = np.int32(H),
        body_weight_N    = np.float32(bw),
        body_weight_kg   = np.float32(bw / 9.81),
        peak_force_N     = np.float32(peak_force),
        peak_force_bw    = np.float32(peak_bw),
        window_pre_sec   = np.float32(pre_sec),
        window_post_sec  = np.float32(post_sec),
        window_mode      = np.str_(window_mode),
        # 新增：冲击事件元数据
        impact_idx       = np.int32(impact.impact_idx),
        n_impacts_total  = np.int32(n_total_impacts),
        is_sync_impact   = np.bool_(impact.is_sync_impact),
        grf_peak_sec     = np.float32(impact.grf_peak_sec),
        grf_peak_N_detected = np.float32(impact.grf_peak_N),
        # 质量统计
        stat_lower_body_mean_score = np.float32(lower_mean),
        stat_lost_rate             = np.float32(lost_rate),
        # schema
        schema_version   = np.str_("segment_v4"),
        grf_columns_hf   = np.str_("[Fx,Fy,Fz,Fz1,Fz2,Fz3,Fz4] positive_contact"),
        grf_columns_fps  = np.str_("[Fx,Fy,Fz] positive_contact"),
    )

    log.info(
        "  ✓ #%03d  T=%d  ev=%d  pre=%.2fs post=%.2fs  peak=%dN(%.2f×BW)"
        "  lower=%.3f  lost=%.1f%%  → %s",
        impact.impact_idx, T, ev_idx, pre_sec, post_sec,
        peak_force, peak_bw, lower_mean, lost_rate * 100, out.name,
    )

    return {**result, "status": "ok", "path": str(out),
            "T": T, "peak_bw": round(peak_bw, 3),
            "is_sync_impact": impact.is_sync_impact,
            "window_pre_sec": pre_sec, "window_post_sec": post_sec}


# ---------------------------------------------------------------------------
# 单 trial 处理入口
# ---------------------------------------------------------------------------

def segment_one(
    root:      Path,
    trial:     str,
    cam:       int,
    dry_run:   bool = False,
    overwrite: bool = False,
    *,
    peak_mode: str = "adaptive",
    merge_gap_sec: float = MERGE_GAP_SEC,
    window_mode: str = "adaptive",
    rally_special: bool = True,
) -> Dict:
    """处理单个 (trial, cam)，返回 result dict（含所有冲击事件的子结果）"""
    result: Dict = {"trial": trial, "cam": cam, "status": "error"}

    # ── 加载 ────────────────────────────────────────────────────────────────
    sync, err = load_sync(root, trial, cam)
    if err:
        return {**result, "message": err}

    if sync.quality.lower() not in ("good", "ok", ""):
        return {**result, "status": "skip",
                "message": f"quality={sync.quality!r}，跳过"}

    grf_data, err = load_grf(root, trial)
    if err:
        return {**result, "message": err}

    pose, err = load_pose(root, trial, cam)
    if err:
        return {**result, "message": err}

    fps = float(pose.get("fps", sync.video_fps) or sync.video_fps)
    if fps <= 0:
        fps = VIDEO_FPS_DEFAULT
    W = int(pose.get("width",  1920))
    H = int(pose.get("height", 1080))

    # ── 峰值检测 ─────────────────────────────────────────────────────────────
    impacts = detect_grf_impacts(
        grf_data, sync, pose, fps,
        peak_mode=peak_mode,
        merge_gap_sec=merge_gap_sec,
        window_mode=window_mode,
        trial=trial,
        rally_special=rally_special,
    )
    if not impacts:
        return {**result, "message": "无有效冲击事件（pose 帧数不足或边界问题）"}

    n = len(impacts)
    log.info(
        "[%s cam%d] 检测到 %d 个冲击峰值（bw=%.0fN=%.1fkg）",
        trial, cam, n, grf_data["bw_N"], grf_data["bw_N"] / 9.81,
    )

    # ── 覆盖检查 ─────────────────────────────────────────────────────────────
    if not overwrite and not dry_run:
        all_exist = all(p_out(root, trial, cam, imp.impact_idx).exists()
                        for imp in impacts)
        if all_exist:
            log.info("  全部 %d 个片段已存在，跳过", n)
            return {**result, "status": "skip",
                    "message": f"已完成 {n} 个片段", "n_impacts": n}

    # ── 逐峰切片 ─────────────────────────────────────────────────────────────
    sub_results = []
    n_ok = n_err = 0
    for impact in impacts:
        r = _segment_one_impact(
            root=root, trial=trial, cam=cam,
            sync=sync, grf=grf_data, pose=pose,
            impact=impact, n_total_impacts=n,
            fps=fps, W=W, H=H, dry_run=dry_run,
            window_mode=window_mode,
        )
        sub_results.append(r)
        if r["status"] in ("ok", "dry_run"):
            n_ok += 1
        else:
            n_err += 1

    status = "ok" if n_err == 0 else ("partial" if n_ok > 0 else "error")
    return {
        **result, "status": status,
        "n_impacts": n, "n_ok": n_ok, "n_err": n_err,
        "segments": sub_results,
    }


# ---------------------------------------------------------------------------
# LOSO 划分
# ---------------------------------------------------------------------------

def make_loso_splits(results: List[Dict]) -> Dict:
    """
    留一被试交叉验证（LOSO）划分。

    ⚠️  同一 trial 的多个冲击片段必须整体分配到同一侧（train 或 test），
    防止数据泄漏。本函数按 subject 分组，同一 subject 的所有片段进入同一侧。
    """
    by_sub: Dict[str, List[str]] = defaultdict(list)
    for r in results:
        if r.get("status") in ("ok", "partial"):
            sub = subject_of(r["trial"])
            for seg in r.get("segments", []):
                if seg.get("status") == "ok" and "path" in seg:
                    by_sub[sub].append(seg["path"])

    subjects = sorted(by_sub.keys())
    splits = {}
    for test_sub in subjects:
        splits[test_sub] = {
            "test":  by_sub[test_sub],
            "train": [p for s in subjects if s != test_sub
                      for p in by_sub[s]],
        }
    return splits


def _stats_list(values: List[float]) -> Optional[Dict[str, float]]:
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
    }


def _step4_summary_block(results: List[Dict]) -> Dict[str, Any]:
    """对一组 (trial,cam) 结果做汇总（单被试或全库）。"""
    status_counts: Counter = Counter()
    errs: List[Dict[str, Any]] = []
    Ts: List[float] = []
    pres: List[float] = []
    posts: List[float] = []
    peaks: List[float] = []
    n_impacts_total = 0
    rally_low_bw: List[Dict[str, Any]] = []
    thr = float(RALLY_LOW_PEAK_BW_SUMMARY_THRESHOLD)

    for r in results:
        st = r.get("status", "unknown")
        status_counts[st] += 1
        if st == "error":
            errs.append({
                "trial": r.get("trial"),
                "cam": r.get("cam"),
                "message": r.get("message"),
            })
        ni = r.get("n_impacts")
        if ni is not None:
            n_impacts_total += int(ni)

        trial = str(r.get("trial") or "")
        is_rally = "rally" in trial.lower()
        cam = r.get("cam")

        for seg in r.get("segments", []):
            if seg.get("status") not in ("ok", "dry_run"):
                continue
            if "T" in seg and seg["T"] is not None:
                Ts.append(float(seg["T"]))
            if seg.get("window_pre_sec") is not None:
                pres.append(float(seg["window_pre_sec"]))
            if seg.get("window_post_sec") is not None:
                posts.append(float(seg["window_post_sec"]))
            if seg.get("peak_bw") is not None:
                peaks.append(float(seg["peak_bw"]))

            if is_rally:
                pb = seg.get("peak_bw")
                if pb is not None and float(pb) < thr:
                    rally_low_bw.append({
                        "trial": trial,
                        "cam": cam,
                        "impact_idx": seg.get("impact_idx"),
                        "peak_bw": float(pb),
                        "T": seg.get("T"),
                        "path": seg.get("path"),
                    })

    rally_low_bw.sort(
        key=lambda x: (str(x["trial"]), int(x["cam"] or 0), int(x["impact_idx"] or 0)),
    )

    return {
        "task_status_counts": dict(status_counts),
        "tasks_total": len(results),
        "n_impacts_labelled_tasks": n_impacts_total,
        "segments_ok_or_dry": len(Ts),
        "segment_stats": {
            "T_frames": _stats_list(Ts),
            "window_pre_sec": _stats_list(pres),
            "window_post_sec": _stats_list(posts),
            "peak_bw": _stats_list(peaks),
        },
        "errors": errs,
        "rally_low_peak_bw": {
            "threshold": thr,
            "description": (
                "trial 名含 rally 且 peak_bw（×BW）低于 threshold 的片段，"
                "供 visualize_segment --npz path 抽查；dry_run 时 path 可能为 null"
            ),
            "n_segments": len(rally_low_bw),
            "segments": rally_low_bw,
        },
    }


def summarize_step4_results(results: List[Dict]) -> Dict[str, Any]:
    """聚合所有 (trial,cam) 的片段级指标；多被试时按 subject 分类写入 by_subject。"""
    by_sub_lists: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        trial = r.get("trial")
        try:
            sub = subject_of(str(trial)) if trial else "unknown"
        except ValueError:
            sub = "unknown"
        by_sub_lists[sub].append(r)

    out = _step4_summary_block(results)
    subs_sorted = sorted(by_sub_lists.keys())
    out["subjects"] = subs_sorted
    out["n_subjects"] = len(subs_sorted)
    out["by_subject"] = {s: _step4_summary_block(by_sub_lists[s]) for s in subs_sorted}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step4: 切片 pose+GRF，生成训练样本 npz（多峰版 v4）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--subjects", nargs="+", metavar="SUB")
    p.add_argument("--cameras",  nargs="+", type=int, metavar="CAM")
    p.add_argument("--auto",      action="store_true",
                   help="自动发现所有已有 pose npz")
    p.add_argument(
        "--from-pose",
        action="store_true",
        help="与 --subjects 联用：按 {sub}/pose/*_cam*_pose.npz 枚举任务"
             "（可选 --cameras 再筛选机位；不传则包含数据中全部机位）",
    )
    p.add_argument("--dry_run",   action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="覆盖已存在的片段文件")
    p.add_argument(
        "--no-save-report",
        action="store_true",
        help="不写入 reports/*.json（默认每次运行都写 step4_summary.json；"
             "非 dry_run 时另写 step4_report.json 与 loso_splits.json）",
    )
    p.add_argument(
        "--save_report",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="JSON 报告目录（默认 <repo>/data/reports，与 --root 数据位置无关；"
             "需与数据报告同盘时可显式设为 {root}/reports）",
    )
    p.add_argument(
        "--peak-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="adaptive：粗检+时间合并+MAD 自适应门控（默认）；fixed：旧版固定 BW 阈值",
    )
    p.add_argument(
        "--merge-gap-sec",
        type=float,
        default=MERGE_GAP_SEC,
        metavar="S",
        help="粗检后合并峰的最大时间间隔 [s]（仅 adaptive，默认 %.2f）" % MERGE_GAP_SEC,
    )
    p.add_argument(
        "--window-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="adaptive：按 Fz 形状估计前后窗长（默认）；fixed：固定前 %.2fs 后 %.2fs"
        % (WINDOW_PRE_SEC, WINDOW_POST_SEC),
    )
    p.add_argument(
        "--disable-rally-special",
        action="store_true",
        help="trial 名含 rally 时也使用与 stage 相同的峰检（默认启用 rally 专用逻辑）",
    )
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    root = Path(args.root)
    if not root.exists():
        log.error("数据根目录不存在：%s", root)
        sys.exit(1)

    if args.auto and args.from_pose:
        log.error("--auto 与 --from-pose 不能同时使用")
        sys.exit(1)

    if args.auto:
        tasks = discover_tasks(root)
        log.info("自动发现 %d 个 (trial, cam) 任务", len(tasks))
    elif args.from_pose:
        subjects = args.subjects or []
        if not subjects:
            log.error("--from-pose 需要指定 --subjects")
            sys.exit(1)
        tasks = []
        for sub in subjects:
            tasks.extend(discover_tasks_for_subject_pose(root, sub))
        cam_note = ""
        if args.cameras:
            cs = set(args.cameras)
            tasks = [(t, c) for t, c in tasks if c in cs]
            cam_note = f"（机位 {sorted(cs)}）"
        log.info("--from-pose：%d 个 (trial, cam)%s", len(tasks), cam_note)
    else:
        subjects = args.subjects or []
        cameras  = args.cameras  or [2]
        if not subjects:
            log.error("请指定 --subjects 或使用 --auto")
            sys.exit(1)
        tasks = []
        for sub in subjects:
            for trial in discover_trials_for_subject(root, sub):
                for cam in cameras:
                    tasks.append((trial, cam))

    if not tasks:
        log.warning("没有找到任何任务，退出")
        sys.exit(0)

    log.info("共 %d 个任务，dry_run=%s  overwrite=%s",
             len(tasks), args.dry_run, args.overwrite)

    results = []
    n_ok = n_skip = n_err = 0
    total_segments = 0

    for trial, cam in tasks:
        r = segment_one(
            root, trial, cam,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            peak_mode=args.peak_mode,
            merge_gap_sec=args.merge_gap_sec,
            window_mode=args.window_mode,
            rally_special=not args.disable_rally_special,
        )
        results.append(r)
        s = r["status"]
        if s in ("ok", "partial"):
            n_ok += 1
            total_segments += r.get("n_ok", 0)
        elif s == "skip":
            n_skip += 1
        else:
            n_err += 1
            log.warning("  ✗ %s cam%d: %s", trial, cam, r.get("message", ""))

    log.info(
        "完成 — trial_ok=%d  skip=%d  error=%d  总片段=%d",
        n_ok, n_skip, n_err, total_segments,
    )

    summary = summarize_step4_results(results)
    wp = summary["segment_stats"].get("window_post_sec")
    pb = summary["segment_stats"].get("peak_bw")
    if wp:
        log.info(
            "片段聚合: n=%d  post_sec mean=%.3fs std=%.3fs p50=%.3fs p90=%.3fs  "
            "peak_bw mean=%.2f",
            wp["n"], wp["mean"], wp["std"], wp["p50"], wp["p90"],
            pb["mean"] if pb else float("nan"),
        )
    rlb = summary.get("rally_low_peak_bw") or {}
    n_rally_low = int(rlb.get("n_segments") or 0)
    if n_rally_low:
        log.info(
            "rally 且 peak_bw<%.1f×：%d 条（摘要字段 rally_low_peak_bw.segments）",
            RALLY_LOW_PEAK_BW_SUMMARY_THRESHOLD,
            n_rally_low,
        )

    n_sub = int(summary.get("n_subjects") or 0)
    if n_sub > 1:
        log.info("按被试汇总（共 %d 个 subject，详见 step4_summary.json → by_subject）", n_sub)
        for sub in summary.get("subjects") or []:
            blk = (summary.get("by_subject") or {}).get(sub) or {}
            tc = blk.get("task_status_counts") or {}
            log.info(
                "  %s  tasks=%d  ok=%s  skip=%s  err=%s  segments=%d",
                sub,
                blk.get("tasks_total", 0),
                tc.get("ok", 0),
                tc.get("skip", 0),
                tc.get("error", 0),
                blk.get("segments_ok_or_dry", 0),
            )

    if args.no_save_report:
        log.info("已跳过写入 reports（--no-save-report）")
    else:
        report_dir = (
            args.report_dir.expanduser().resolve()
            if args.report_dir is not None
            else (_REPO_DATA_DIR / "reports").resolve()
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        summary["root"] = str(root)
        summary["generated_at"] = (
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        summary["dry_run"] = bool(args.dry_run)
        summary["cli"] = {k: v for k, v in vars(args).items()}
        summary["report_dir"] = str(report_dir)
        (report_dir / "step4_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        log.info("摘要已写入 %s", report_dir / "step4_summary.json")

        if not args.dry_run:
            (report_dir / "step4_report.json").write_text(
                json.dumps(
                    {"total_tasks": len(tasks),
                     "n_ok": n_ok, "n_skip": n_skip, "n_err": n_err,
                     "total_segments": total_segments,
                     "results": results},
                    indent=2, ensure_ascii=False, default=str,
                ), encoding="utf-8",
            )

            loso = make_loso_splits(results)
            (report_dir / "loso_splits.json").write_text(
                json.dumps(loso, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            log.info("完整报告与 LOSO 已保存至 %s", report_dir)


if __name__ == "__main__":
    main()