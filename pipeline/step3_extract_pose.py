"""
BadmintonGRF — Step 3: 骨骼点提取
=====================================
使用 YOLO26-pose + ByteTrack 提取运动员骨骼点序列。

安装（Ubuntu，一行搞定）：
    pip install ultralytics opencv-python numpy

运行：
    conda activate badminton_grf
    python step3_extract_pose.py --root ./data   # 或 BADMINTON_DATA_ROOT
    python step3_extract_pose.py --subjects sub_001 sub_002 --dry_run

★ 核心设计思路（2026-03-15 定稿）：

  【识别策略：ev_local 竖向冲击特征】
  ev_local 是 GRF 传感器记录的落地接触帧，是整个 clip 里唯一的物理地面真值。
  在 ev_local 这一帧，运动员正在落地，其 bbox_center_y 相对前 10 帧基线有
  明显的向下位移（竖向冲击），而所有助手/观察者都是静止或慢速移动的。
  这个特征不依赖颜色检测，不依赖相机角度，不依赖场地布局，是数据集独有的先验。

  【三个锚点：全部来自检测结果，不依赖任何外部信号】
  anchor_pos   = ev_local 帧运动员 bbox_center   ← 替代颜色检测的 plate_center
  anchor_area  = ev_local 帧运动员 bbox_area     ← 面积参考（双向过滤）
  anchor_ratio = ev_local 帧运动员 bbox h/w      ← 直接排除坐姿/蹲姿/横行路人

  【Fix-1】bytetrack_badminton.yaml  track_buffer=600（≈5s @ 119.88fps）
  【Fix-3】pos_fallback 位置最近邻，MAX=300px，同步更新 cur_tid
  【Fix-5】连续丢失≥30帧强制重识别，用 anchor 三锚点过滤 + 距 cur_pos 最近选人

  【删除的逻辑】
  - plate_center 不再参与任何识别或过滤，也不再在可视化中显示
  - MAX_PLATE_DIST_FOR_REID 整个删除
  - identify_athlete（ROI+坐姿+运动量）整个删除

单一约定（v2 系统性重构）：
  每个 trial×相机只输出一个 {trial}_cam{N}_pose.npz：
  帧率由 resolve_video_fps 决定（sync 标注 > ffprobe > OpenCV），缓解 4K60 / 1080p120 元数据不一致。
  覆盖区间由 resolve_full_frame_range 决定：
  - 时间轴上仅使用 GRF timestamps 有采样的部分（视频可长于力台，末端按 t_grf_hi 截断）；
  - 起点取 max(GRF 映射起点, per_cam 的 video_event_frame)：仅保证从人工标注的
    sync 帧起视频与力台对齐，片头再长也不纳入。
  在 sync 对齐帧上用竖向冲击锁定主运动员。

输出 .npz 字段：
    keypoints         float32 (N, 17, 2)   像素坐标 (x, y)，COCO-17
    scores            float32 (N, 17)       关键点置信度 [0,1]
    bbox              float32 (N, 5)        (x1,y1,x2,y2,conf)
    frame_indices     int32   (N,)          原始视频绝对帧号
    track_status      U20     (N,)          tracked/pos_fallback/reidentified/lost
    event_frame_abs   int32                 接触事件绝对帧号
    event_frame_local int32   ★            接触事件局部索引（Step4直接用）
    fps / width / height / n_frames
    plate_center_px   float32 (2,)          力台中心（仅可视化）
    model / subject / trial / camera / extracted_at / schema_version (pose_npz_v2)
    pose_coverage     str   固定 "grf_aligned"
    stat_*                                  质量统计
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _default_data_root() -> Path:
    env = os.environ.get("BADMINTON_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


def _npz_is_valid(path: Path) -> bool:
    """仅承认 pose_npz_v2 + pose_coverage=grf_aligned（GRF 对齐全长）。"""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as d:
            required = [
                "keypoints", "scores", "bbox", "frame_indices",
                "event_frame_abs", "event_frame_local", "fps",
                "width", "height", "n_frames", "subject", "trial", "camera",
            ]
            if not all(k in d for k in required):
                return False
            if str(d.get("schema_version", "") or "") != "pose_npz_v2":
                return False
            if str(d.get("pose_coverage", "") or "") != "grf_aligned":
                return False
            return True
    except Exception:
        return False


def _write_npz_atomic(path: Path, **payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用不带原始后缀的 .tmp 文件，并通过文件句柄写入，
    # 避免 np.savez_compressed 自动追加 .npz 后缀导致找不到临时文件
    # 例如: target=sub_004_foo_pose.npz → tmp=sub_004_foo_pose.tmp
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp_path, path)


def _load_grf_timestamp_bounds(grf_path: Path) -> Optional[tuple[float, float]]:
    """返回 GRF 文件 timestamps 的 [t_min, t_max]（秒），失败则 None。"""
    if not grf_path.exists():
        return None
    try:
        raw = np.load(str(grf_path), allow_pickle=True)
        d = raw.item() if raw.dtype == object else None
        if not isinstance(d, dict) or "timestamps" not in d:
            return None
        ts = np.asarray(d["timestamps"], dtype=np.float64)
        if ts.size == 0:
            return None
        return float(ts.min()), float(ts.max())
    except Exception:
        return None


def _grf_aligned_frame_bounds(
    t_grf_lo: float,
    t_grf_hi: float,
    video_event_sec: float,
    video_event_frame: int,
    offset_sec: float,
    fps: float,
    total_frames: int,
) -> tuple[int, int]:
    """
    与 step4 一致：t_video = video_event_sec + (f - video_event_frame) / fps，
    t_grf = t_video + offset_sec。求使 t_grf ∈ [t_grf_lo, t_grf_hi] 的整数帧范围。
    """
    if fps <= 0:
        return 0, max(0, total_frames - 1)
    # f = video_event_frame + (t_grf - offset_sec - video_event_sec) * fps
    f_lo = video_event_frame + (t_grf_lo - offset_sec - video_event_sec) * fps
    f_hi = video_event_frame + (t_grf_hi - offset_sec - video_event_sec) * fps
    start_f = int(np.ceil(f_lo - 1e-6))
    end_f   = int(np.floor(f_hi + 1e-6))
    start_f = max(0, start_f)
    end_f   = min(total_frames - 1, end_f)
    return start_f, end_f


def _parse_rate_fraction(s: str) -> Optional[float]:
    """解析 ffprobe 的 '120/1'、'60000/1001' 等为 float Hz。"""
    s = (s or "").strip()
    if not s or s == "0/0":
        return None
    if "/" in s:
        a, _, b = s.partition("/")
        try:
            v = float(a) / float(b)
            return v if v > 1e-6 else None
        except (ValueError, ZeroDivisionError):
            return None
    try:
        v = float(s)
        return v if v > 1e-6 else None
    except ValueError:
        return None


def _ffprobe_avg_fps(video_path: Path) -> Optional[float]:
    """用 ffprobe 读视频流 avg_frame_rate（对 4K/高帧率容器通常比 OpenCV 可靠）。"""
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return None
    return _parse_rate_fraction(line[0])


def resolve_video_fps(
    video_path: Path,
    cap,
    sync:  dict,
    cam:   int,
    logger: logging.Logger,
) -> float:
    """
    确定用于对齐与写入 npz 的帧率（Hz）。

    优先级：
      1) sync：per_cam[cam].video_fps（若有）
      2) sync：顶层 video_fps（对齐 UI 写入）
      3) ffprobe avg_frame_rate（4K60 / 1080p120 元数据常与 OpenCV 不一致）
      4) OpenCV CAP_PROP_FPS
      5) 119.88
    """
    cam_data = (sync.get("per_cam") or {}).get(str(cam), {}) or {}
    v = float(cam_data.get("video_fps", 0) or 0)
    src = "per_cam.video_fps"
    if v <= 1e-6:
        v = float(sync.get("video_fps", 0) or 0)
        src = "sync.video_fps"
    if v > 1e-6:
        logger.info("  帧率采用 %s: %.4f Hz", src, v)
        return v

    cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    ff_fps = _ffprobe_avg_fps(video_path)

    if ff_fps and ff_fps > 1e-6:
        if cap_fps > 1e-6 and abs(ff_fps - cap_fps) / max(ff_fps, cap_fps) > 0.07:
            logger.info(
                "  帧率: OpenCV=%.4f  ffprobe=%.4f → 采用 ffprobe",
                cap_fps,
                ff_fps,
            )
        else:
            logger.info("  帧率采用 ffprobe: %.4f Hz", ff_fps)
        return ff_fps

    if cap_fps > 1e-6:
        logger.info("  帧率采用 OpenCV: %.4f Hz", cap_fps)
        return cap_fps

    logger.warning("  无法确定帧率，使用默认 119.88 Hz")
    return 119.88


# ──────────────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────────────

# 竖向冲击识别：接触帧前多少帧作为"基线"
IMPACT_BASELINE_FRAMES = 10

# Fix-5：连续真正丢失超过此帧数后触发强制重识别（≈0.25s @ 119.88fps）
REID_THRESHOLD   = 30

# Fix-3：位置最近邻最大允许距离（像素）
MAX_POS_FALLBACK_DIST = 300

# 力台蓝色 HSV（仅用于 detect_plate_center，结果存入 npz 但不显示）
PLATE_HSV_LO = np.array([100, 60,  60],  dtype=np.uint8)
PLATE_HSV_HI = np.array([130, 255, 255], dtype=np.uint8)

# COCO-17 关键点索引
LOWER_BODY_IDX = [11, 12, 13, 14, 15, 16]

# cam1 误配 4K 时不纳入 pose（与论文/benchmark「cam1 部分 4K 已排除」一致）。
# 1) 手工名单：历史上已知误配的 subject×trial（扫描阶段不生成 cam1 任务）。
# 2) 自动识别：扫描与 process_task 均按视频分辨率判定（见 _is_cam1_4k_resolution），
#    不产出该 cam 的 npz，无需再维护完整名单。
CAM1_4K_SKIP: dict[str, list[str]] = {
    "sub_001": [
        "fatigue_stage1_01", "fatigue_stage2_01", "fatigue_stage3_01",
        "rally_01", "rally_02", "rally_03", "rally_04",
        "stage1_02", "stage1_03", "stage1_04", "stage1_05",
        "stage2_01", "stage2_02", "stage2_03",
        "stage3_01", "stage3_02", "stage3_04",
    ],
    "sub_002": ["rally_02", "rally_03"],
    "sub_008": [
        "fatigue_stage1_01", "fatigue_stage2_01", "fatigue_stage3_01",
        "rally_02", "rally_03", "rally_04",
        "stage1_01", "stage1_03", "stage1_04",
        "stage2_01", "stage2_02", "stage2_03",
        "stage3_01", "stage3_02", "stage3_03",
    ],
    "sub_009": [
        "fatigue_stage1_01", "fatigue_stage2_01", "fatigue_stage3_01",
        "rally_01", "rally_02", "rally_03",
        "stage1_01", "stage1_02", "stage1_03",
        "stage2_01", "stage2_02", "stage2_03",
        "stage3_01", "stage3_02", "stage3_03",
    ],
    "sub_011": [
        "fatigue_stage1_02", "fatigue_stage2_02", "fatigue_stage3_02",
        "stage3_04",
    ],
    "sub_012": [
        "fatigue_stage1_01", "fatigue_stage2_01", "fatigue_stage3_01",
        "rally_01", "rally_02", "rally_03",
        "stage1_01", "stage1_02", "stage1_03",
        "stage2_01", "stage2_02", "stage2_03",
        "stage3_01", "stage3_02", "stage3_03",
    ],
    "sub_014": [
        "fatigue_stage1_01", "fatigue_stage2_01", "fatigue_stage3_01",
        "rally_02", "rally_03",
        "stage1_01", "stage1_02", "stage1_03",
        "stage2_01", "stage2_02", "stage2_03",
        "stage3_01", "stage3_02", "stage3_03",
    ],
    "sub_017": [
        "fatigue_stage2_01", "fatigue_stage3_01",
        "rally_01", "rally_02", "rally_03",
        "stage1_01", "stage1_02", "stage1_03",
        "stage2_01", "stage2_02", "stage2_03",
        "stage3_01", "stage3_03",
    ],
}


def _ffprobe_stream_wh(video_path: Path) -> Optional[tuple[int, int]]:
    """ffprobe 读首路视频宽高（cv2 不可用时供 scan_tasks 使用）。"""
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = line[0].split(",")
    if len(parts) != 2:
        return None
    try:
        w, h = int(parts[0]), int(parts[1])
        return (w, h) if w > 0 and h > 0 else None
    except ValueError:
        return None


def _video_wh_quick(vpath: Path) -> Optional[tuple[int, int]]:
    """快速读视频分辨率（不重解码）；失败时回退 ffprobe。"""
    if cv2 is not None:
        cap = cv2.VideoCapture(str(vpath))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return (w, h)
    return _ffprobe_stream_wh(vpath)


def _is_cam1_4k_resolution(w: int, h: int) -> bool:
    """
    判定 cam1 是否为 4K 档分辨率（误配时与 1080p 其它机位不对齐，不纳入 pose）。
    规则：长边≥3600 或 短边≥2160（覆盖 3840×2160、3840×1600、4096×2160 等）。
    """
    if w <= 0 or h <= 0:
        return False
    lo, hi = (w, h) if w <= h else (h, w)
    return hi >= 3600 or lo >= 2160


# ──────────────────────────────────────────────────────────────────────────────
# 辅助工具
# ──────────────────────────────────────────────────────────────────────────────

def bbox_area(bb: np.ndarray) -> float:
    return float(max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1]))


def bbox_center(bb: np.ndarray) -> np.ndarray:
    return np.array([(bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0],
                    dtype=np.float32)


def bbox_ratio(bb: np.ndarray) -> float:
    """bbox 高宽比 h/w，站立人物约 2.5-4.0，坐/蹲约 0.8-1.5。"""
    w = max(1.0, float(bb[2] - bb[0]))
    h = max(1.0, float(bb[3] - bb[1]))
    return h / w


def detect_plate_center(frames: list[np.ndarray]) -> Optional[np.ndarray]:
    """
    检测力台中心，仅用于可视化绿圈，不参与任何识别逻辑。
    Fix-4：只接受下半画面（cy > height*0.4），避免顶部灯具误检。
    """
    if not frames:
        return None
    frame_h = frames[0].shape[0]
    accum   = None
    for frame in frames[:min(10, len(frames))]:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, PLATE_HSV_LO, PLATE_HSV_HI)
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        accum = mask.astype(np.float32) if accum is None \
                else accum + mask.astype(np.float32)
    if accum is None:
        return None
    bin_mask = (accum > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    valid_c = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000:
            continue
        M_tmp = cv2.moments(c)
        if M_tmp["m00"] == 0:
            continue
        if M_tmp["m01"] / M_tmp["m00"] > frame_h * 0.4:
            valid_c.append((area, c))
    if not valid_c:
        return None
    _, largest = max(valid_c, key=lambda x: x[0])
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]],
                    dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# YOLO 推理封装
# ──────────────────────────────────────────────────────────────────────────────

def _disable_torch_dynamo_runtime(torch_mod) -> None:
    """
    在 import ultralytics 之前调用：先 import torch 并关闭 dynamo。
    勿设置 TORCHDYNAMO_DISABLE 环境变量——会与 trace_rules 里巨型 re.compile 冲突，
    出现 ValueError: too many values to unpack（子进程偶发 import 失败）。
    """
    try:
        import torch._dynamo as dynamo
        dynamo.config.suppress_errors = True
        if hasattr(dynamo.config, "disable"):
            dynamo.config.disable = True
    except Exception:
        pass


def _apply_cudnn_safe(torch_mod, logger: logging.Logger) -> None:
    """
    须在「已成功 import torch」（例如已通过 ultralytics 加载）之后调用。
    若在子进程里先于 YOLO 单独 import torch 且中途失败，会留下半初始化状态，
    再次 import 时可能触发 TORCH_LIBRARY / prims 重复注册。
    """
    try:
        if not torch_mod.cuda.is_available():
            return
        torch_mod.backends.cudnn.benchmark = False
        logger.info(
            "  CUDA 安全模式：cudnn.benchmark=False（缓解部分环境下的随机段错误）"
        )
    except Exception as e:
        logger.warning("  CUDA 安全模式配置失败：%s", e)


def build_model(
    device: str = "cuda:0",
    *,
    half: bool = True,
    cuda_safe: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """
    half：CUDA 上默认 FP16，加速推理；显存未必显著增大（track 仍为逐帧）。

    不在加载后改写各层 nn.Module（例如替换共享 SiLU）：那会破坏 fuse 与
    nn.Conv2d 的 parameter 注册，随机出现 conv2d bias 变成 Conv2d、
    _apply_instance_norm 等异常。若遇 SiLU/conv 相关错误，请对齐 ultralytics 与 torch 版本。
    """
    log = logger or logging.getLogger("step3")
    import torch
    _disable_torch_dynamo_runtime(torch)
    from ultralytics import YOLO

    if cuda_safe:
        _apply_cudnn_safe(torch, log)
    use_half = bool(half) and ("cuda" in str(device).lower())
    pred_kw = {"device": device, "verbose": False}
    if use_half:
        pred_kw["half"] = True
    for name in ("yolo26l-pose.pt", "yolo11l-pose.pt"):
        try:
            model = YOLO(name)
            model.predict(np.zeros((64, 64, 3), dtype=np.uint8), **pred_kw)
            log.info(f"模型加载成功：{name}")
            return model, name
        except Exception as e:
            log.warning(f"  {name} 失败：{e}")
    raise RuntimeError("yolo26l-pose 和 yolo11l-pose 均加载失败")


def parse_yolo_result(result) -> list[dict]:
    preds = []
    if result.keypoints is None or result.boxes is None:
        return preds
    kps_xy = result.keypoints.xy.cpu().numpy()
    conf   = result.keypoints.conf
    boxes  = result.boxes
    n      = len(kps_xy)
    sc_all = conf.cpu().numpy().astype(np.float32) if conf is not None \
             else np.full((n, 17), 0.9, dtype=np.float32)
    tids   = boxes.id.int().cpu().numpy() if boxes.id is not None \
             else np.arange(n, dtype=np.int32)
    for i in range(n):
        preds.append({
            "track_id":        int(tids[i]),
            "keypoints":       kps_xy[i].astype(np.float32),
            "keypoint_scores": sc_all[i],
            "bbox":            boxes.xyxy[i].cpu().numpy().astype(np.float32),
            "bbox_score":      float(boxes.conf[i].cpu().numpy()),
        })
    return preds


def reset_tracker(model):
    try:
        if hasattr(model, "predictor") and model.predictor is not None:
            if hasattr(model.predictor, "trackers"):
                del model.predictor.trackers
            if hasattr(model.predictor, "vid_path"):
                del model.predictor.vid_path
    except Exception:
        pass


def _bytetrack_yaml_path() -> str:
    """Fix-1：track_buffer=600（≈5s @ 119.88fps），防止腾空时 track 被删除。"""
    yaml_path = Path(__file__).parent / "bytetrack_badminton.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(
            "tracker_type: bytetrack\n"
            "track_buffer: 600\n"
            "track_high_thresh: 0.25\n"
            "track_low_thresh: 0.1\n"
            "new_track_thresh: 0.25\n"
            "match_thresh: 0.8\n"
            "fuse_score: True\n",
            encoding="utf-8",
        )
        logging.getLogger("step3").info("  已创建 bytetrack_badminton.yaml（track_buffer=600）")
    return str(yaml_path)


# ──────────────────────────────────────────────────────────────────────────────
# 核心识别：ev_local 竖向冲击特征
# ──────────────────────────────────────────────────────────────────────────────

def identify_athlete_at_ev_local(
    all_preds: list[list[dict]],
    ev_local:  int,
    width:     int,
    logger:    logging.Logger,
) -> tuple[Optional[int], Optional[np.ndarray]]:
    """
    利用 ev_local（GRF 接触帧）的两个互补信号识别运动员：

    信号一：竖向冲击（impact_score）
      ev_local 是力台传感器落地峰值帧，运动员正在落地，
      bbox_center_y 相对前 IMPACT_BASELINE_FRAMES 帧基线有明显正增量。
      ⚠️ 单独使用的局限：举手、走路也会产生竖向位移。

    信号二：水平中心度（center_score）
      所有摄像机对准力台架设，力台接近每个相机视野的水平中心。
      运动员在力台上 → 接近画面水平中心（center_score ≈ 1）。
      边缘助手/路人 → 远离中心（center_score ≈ 0）。
      center_score = 1 - |cx - width/2| / (width/2)

    综合评分（加权融合）：
      impact_norm  = clip(max(0, impact_y) / (est_height * 0.08), 0, 1)
      score = 0.55 * impact_norm + 0.45 * center_score

    权重设计：冲击稍高权重（对应物理事件），中心度强力纠错（防边缘干扰）。

    Fallback（ev_local 无检测）：
      在 ev_local ± 20 帧里用「频率 × 面积 × 中心度」选人。
    """
    n = len(all_preds)

    # ── 主策略：竖向冲击 + 水平中心度 ──
    ev_preds = all_preds[ev_local] if ev_local < n else []

    if ev_preds:
        # 每个 tid 在 ev_local 前的 bbox_center_y 基线
        baseline_y: dict[int, float] = {}
        for p in ev_preds:
            tid = p["track_id"]
            ys  = []
            for fi in range(max(0, ev_local - IMPACT_BASELINE_FRAMES), ev_local):
                for pp in all_preds[fi]:
                    if pp["track_id"] == tid:
                        ys.append(float(bbox_center(pp["bbox"])[1]))
                        break
            baseline_y[tid] = float(np.mean(ys)) if ys                               else float(bbox_center(p["bbox"])[1])

        # 竖向冲击量（y 增大 = 身体向下 = 落地）
        impacts: dict[int, float] = {}
        for p in ev_preds:
            tid   = p["track_id"]
            cur_y = float(bbox_center(p["bbox"])[1])
            impacts[tid] = cur_y - baseline_y[tid]

        # 估算画面高度（用 bbox y2 最大值，无需额外参数）
        est_height = max(float(p["bbox"][3]) for p in ev_preds) * 1.1
        est_height = max(est_height, 100.0)

        # 综合评分
        scores_combined: dict[int, float] = {}
        for p in ev_preds:
            tid = p["track_id"]
            cx  = float(bbox_center(p["bbox"])[0])
            impact_norm  = float(np.clip(impacts[tid] / (est_height * 0.08), 0.0, 1.0))
            center_score = float(np.clip(1.0 - abs(cx - width / 2.0) / (width / 2.0), 0.0, 1.0))
            scores_combined[tid] = 0.55 * impact_norm + 0.45 * center_score

        # 打印所有候选
        logger.info("  ★ 竖向冲击+中心度识别（ev_local=%d）：", ev_local)
        for p in sorted(ev_preds, key=lambda x: -scores_combined.get(x["track_id"], 0)):
            tid = p["track_id"]
            cx  = float(bbox_center(p["bbox"])[0])
            impact_norm  = float(np.clip(impacts[tid] / (est_height * 0.08), 0.0, 1.0))
            center_score = float(np.clip(1.0 - abs(cx - width/2.0) / (width/2.0), 0.0, 1.0))
            logger.info(
                "    tid=%-4d  score=%.3f  impact=%.1fpx(%.2f)  "
                "cx=%.0f  center=%.2f  area=%.0fpx²  ratio=%.2f",
                tid, scores_combined[tid],
                impacts[tid], impact_norm,
                cx, center_score,
                bbox_area(p["bbox"]), bbox_ratio(p["bbox"]),
            )

        best_tid = max(scores_combined, key=scores_combined.get)
        best_p   = next(p for p in ev_preds if p["track_id"] == best_tid)
        logger.info(
            "  ★ 选定 track_id=%d  score=%.3f  impact=%.1fpx  "
            "area=%.0fpx²  ratio=%.2f",
            best_tid, scores_combined[best_tid], impacts[best_tid],
            bbox_area(best_p["bbox"]), bbox_ratio(best_p["bbox"]),
        )
        if impacts[best_tid] < 2.0:
            logger.warning(
                "  ⚠️  冲击量仅 %.1fpx（<2px），主要依赖中心度选人，"
                "建议可视化核查", impacts[best_tid]
            )
        return best_tid, best_p["bbox"].copy()

    # ── Fallback：ev_local 无检测，用频率×面积×中心度 ──
    logger.warning("  ev_local 帧无检测结果，退回频率×面积×中心度策略")
    sample_s = max(0, ev_local - 20)
    sample_e = min(n, ev_local + 20)
    count:  dict[int, int]   = defaultdict(int)
    area:   dict[int, float] = defaultdict(float)
    cx_sum: dict[int, float] = defaultdict(float)
    last_bbox_per_tid: dict[int, np.ndarray] = {}

    for fi in range(sample_s, sample_e):
        for p in all_preds[fi]:
            tid = p["track_id"]
            count[tid]  += 1
            area[tid]   += bbox_area(p["bbox"])
            cx_sum[tid] += float(bbox_center(p["bbox"])[0])
            last_bbox_per_tid[tid] = p["bbox"].copy()

    if not count:
        logger.error("  ev_local 附近无任何检测结果，无法识别运动员")
        return None, None

    fb_scores: dict[int, float] = {}
    for tid in count:
        avg_area  = area[tid] / count[tid]
        avg_cx    = cx_sum[tid] / count[tid]
        center_s  = float(np.clip(1.0 - abs(avg_cx - width/2.0) / (width/2.0), 0.0, 1.0))
        fb_scores[tid] = count[tid] * avg_area * (0.5 + 0.5 * center_s)

    best_tid = max(fb_scores, key=fb_scores.get)
    logger.info("  Fallback 选定 track_id=%d  score=%.0f", best_tid, fb_scores[best_tid])
    return best_tid, last_bbox_per_tid.get(best_tid)


# ──────────────────────────────────────────────────────────────────────────────
# 逐帧过滤：Fix-3 位置兜底 + Fix-5 强制重识别
# ──────────────────────────────────────────────────────────────────────────────

def choose_athlete_per_frame(
    all_preds:    list[list[dict]],
    athlete_tid:  int,
    anchor_bbox:  np.ndarray,
    logger:       logging.Logger,
) -> tuple[list[int], list[str]]:
    """
    按帧决定追踪哪个 track_id。

    三个锚点来自 identify_athlete_at_ev_local 返回的 anchor_bbox，
    全部源自 YOLO 检测结果，不依赖任何外部信号（颜色、ROI、plate_center）：

      anchor_pos   = bbox_center(anchor_bbox)   位置锚点（代替 plate_center）
      anchor_area  = bbox_area(anchor_bbox)     面积锚点
      anchor_ratio = h/w of anchor_bbox         姿态锚点（站立≈3，坐/蹲≈1）

    策略（按优先级）：
      1. ByteTrack track_id 匹配 → tracked
         cur_pos 做 EMA 缓慢跟随（α=0.05），跟随运动员真实位置漂移
      2. 连续丢失 ≥ REID_THRESHOLD → 强制重识别（Fix-5）
         过滤：面积 ∈ [0.3, 2.5] × anchor_area
         过滤：h/w  ∈ [0.5, 2.0] × anchor_ratio
         选取：距 cur_pos 最近（不是最大面积！前景路人面积更大但位置更远）
         → reidentified
      3. 短暂丢失 → 位置最近邻兜底（Fix-3）
         在距 last_bbox 中心 MAX_POS_FALLBACK_DIST=300px 内选最近的人
         更新 cur_tid（适应 ByteTrack 重分配漂移）
         → pos_fallback
      4. 全部失败 → lost

    anchor_pos / anchor_area / anchor_ratio 永远不更新（模板保持接触帧状态）。
    cur_pos 随运动员移动缓慢更新。
    """
    # 从 anchor_bbox 提取三个不变的锚点
    anchor_pos   = bbox_center(anchor_bbox)
    anchor_area  = bbox_area(anchor_bbox)
    anchor_ratio = bbox_ratio(anchor_bbox)

    n                = len(all_preds)
    chosen_tids      = [-1] * n
    statuses         = ["lost"] * n
    cur_tid          = athlete_tid
    last_bbox        = anchor_bbox.copy()
    # cur_pos 跟随运动员实际位置，初始值为 anchor_pos
    cur_pos          = anchor_pos.copy()
    consecutive_lost = 0
    fallback_cnt     = 0
    lost_cnt         = 0
    reid_cnt         = 0
    EMA_ALPHA        = 0.05   # tracked 帧 cur_pos 的平滑系数（小=慢漂移）

    for fi, preds in enumerate(all_preds):
        if not preds:
            lost_cnt         += 1
            consecutive_lost += 1
            continue

        # ── Fix-5：连续丢失达门槛，强制重识别 ──
        if consecutive_lost >= REID_THRESHOLD:
            # 面积双向过滤：下限排除极小路过者，上限排除靠近镜头的大框前景人
            candidates = [
                p for p in preds
                if anchor_area * 0.3 < bbox_area(p["bbox"]) < anchor_area * 2.5
                and anchor_ratio * 0.5 < bbox_ratio(p["bbox"]) < anchor_ratio * 2.0
            ]
            if not candidates:
                # 降级：只用面积过滤，放弃高宽比约束
                candidates = [
                    p for p in preds
                    if anchor_area * 0.3 < bbox_area(p["bbox"]) < anchor_area * 2.5
                ]
            if not candidates:
                candidates = preds  # 最终降级：用全部

            # 选距 cur_pos 最近的人（不是最大面积！）
            best_p = min(candidates,
                         key=lambda p: float(np.linalg.norm(
                             bbox_center(p["bbox"]) - cur_pos)))
            cur_tid          = best_p["track_id"]
            consecutive_lost = 0
            chosen_tids[fi]  = cur_tid
            statuses[fi]     = "reidentified"
            last_bbox        = best_p["bbox"].copy()
            cur_pos          = bbox_center(best_p["bbox"]).copy()
            reid_cnt        += 1
            logger.info(
                "  帧%d 强制重识别 → tid=%d  area=%.0fpx²  ratio=%.2f  "
                "dist_to_cur=%.1fpx",
                fi, cur_tid, bbox_area(best_p["bbox"]), bbox_ratio(best_p["bbox"]),
                float(np.linalg.norm(bbox_center(best_p["bbox"]) - cur_pos)),
            )
            continue

        # ── 主策略：ByteTrack track_id 匹配 ──
        target = next((p for p in preds if p["track_id"] == cur_tid), None)
        if target is not None:
            chosen_tids[fi]  = cur_tid
            statuses[fi]     = "tracked"
            last_bbox        = target["bbox"].copy()
            # EMA 慢速更新 cur_pos，跟随运动员真实位置漂移
            cur_pos = (1 - EMA_ALPHA) * cur_pos + EMA_ALPHA * bbox_center(target["bbox"])
            consecutive_lost = 0
            continue

        # ── Fix-3：短暂丢失，位置最近邻兜底 ──
        last_ctr = bbox_center(last_bbox)
        best_p, best_d = None, float("inf")
        for p in preds:
            d = float(np.linalg.norm(bbox_center(p["bbox"]) - last_ctr))
            if d < best_d:
                best_d, best_p = d, p

        if best_p is not None and best_d <= MAX_POS_FALLBACK_DIST:
            cur_tid          = best_p["track_id"]  # 跟随 ByteTrack 新 id
            chosen_tids[fi]  = cur_tid
            statuses[fi]     = "pos_fallback"
            last_bbox        = best_p["bbox"].copy()
            cur_pos          = bbox_center(best_p["bbox"]).copy()
            consecutive_lost = 0
            fallback_cnt    += 1
            continue

        # ── 真正丢失 ──
        statuses[fi]     = "lost"
        lost_cnt        += 1
        consecutive_lost += 1

    if lost_cnt:
        logger.info("  真正丢失帧：%d", lost_cnt)
    if fallback_cnt:
        logger.info("  位置兜底帧：%d", fallback_cnt)
    if reid_cnt:
        logger.info("  强制重识别次数：%d", reid_cnt)

    return chosen_tids, statuses


def extract_values_by_tid(
    all_preds:   list[list[dict]],
    chosen_tids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n         = len(all_preds)
    keypoints = np.full((n, 17, 2), np.nan, dtype=np.float32)
    scores    = np.zeros((n, 17),   dtype=np.float32)
    bboxes    = np.full((n, 5),     np.nan, dtype=np.float32)
    for fi, (preds, tid) in enumerate(zip(all_preds, chosen_tids)):
        if tid == -1:
            continue
        target = next((p for p in preds if p["track_id"] == tid), None)
        if target is None:
            continue
        keypoints[fi] = target["keypoints"]
        scores[fi]    = target["keypoint_scores"]
        bboxes[fi]    = np.append(target["bbox"], target["bbox_score"])
    return keypoints, scores, bboxes


# ──────────────────────────────────────────────────────────────────────────────
# 质量统计
# ──────────────────────────────────────────────────────────────────────────────

def _quality_stats(
    keypoints: np.ndarray, scores: np.ndarray,
    bboxes: np.ndarray, statuses: list[str],
    fps: float,
) -> dict:
    valid    = ~np.isnan(bboxes[:, 0])
    lower_sc = scores[:, LOWER_BODY_IDX]
    lost_n   = statuses.count("lost")
    fall_n   = statuses.count("pos_fallback")
    reid_n   = statuses.count("reidentified")
    n        = len(statuses)

    stats = {
        "stat_n_frames":              int(n),
        "stat_lost_frames":           int(lost_n),
        "stat_lost_rate":             round(lost_n / max(n, 1), 4),
        "stat_pos_fallback_frames":   int(fall_n),
        "stat_reidentified_frames":   int(reid_n),
        "stat_mean_kp_score":         round(
            float(np.nanmean(scores[valid])) if valid.any() else 0.0, 4),
        "stat_lower_body_mean_score": round(
            float(np.nanmean(lower_sc[valid])) if valid.any() else 0.0, 4),
        "stat_high_conf_frame_rate":  round(
            float(np.mean(np.all(lower_sc[valid] > 0.5, axis=1)))
            if valid.any() else 0.0, 4),
    }

    if not valid.any():
        stats.update({
            "stat_bbox_area_mean": 0.0,
            "stat_bbox_area_cv": 0.0,
            "stat_bbox_aspect_mean": 0.0,
            "stat_bbox_area_outlier_rate": 0.0,
            "stat_speed_mean": 0.0,
            "stat_speed_p95": 0.0,
            "stat_accel_mean": 0.0,
            "stat_accel_p95": 0.0,
        })
        return stats

    bbox_w = np.clip(bboxes[:, 2] - bboxes[:, 0], 1.0, None)
    bbox_h = np.clip(bboxes[:, 3] - bboxes[:, 1], 1.0, None)
    bbox_area = bbox_w * bbox_h
    bbox_aspect = bbox_h / bbox_w

    area_valid = bbox_area[valid]
    stats["stat_bbox_area_mean"] = round(float(np.nanmean(area_valid)), 2)
    stats["stat_bbox_area_cv"] = round(float(np.nanstd(area_valid) / (np.nanmean(area_valid) + 1e-6)), 4)
    stats["stat_bbox_aspect_mean"] = round(float(np.nanmean(bbox_aspect[valid])), 4)

    med = float(np.nanmedian(area_valid))
    iqr = float(np.nanpercentile(area_valid, 75) - np.nanpercentile(area_valid, 25))
    lo = med - 2.5 * iqr
    hi = med + 2.5 * iqr
    stats["stat_bbox_area_outlier_rate"] = round(
        float(np.mean((area_valid < lo) | (area_valid > hi))), 4
    )

    centers = np.stack([
        (bboxes[:, 0] + bboxes[:, 2]) * 0.5,
        (bboxes[:, 1] + bboxes[:, 3]) * 0.5,
    ], axis=1)
    centers[~valid] = np.nan
    diffs = np.diff(centers, axis=0)
    speed = np.linalg.norm(diffs, axis=1)
    speed = speed[np.isfinite(speed)] * fps
    if speed.size:
        stats["stat_speed_mean"] = round(float(np.mean(speed)), 2)
        stats["stat_speed_p95"] = round(float(np.percentile(speed, 95)), 2)
        accel = np.diff(speed) * fps
        accel = accel[np.isfinite(accel)]
        if accel.size:
            stats["stat_accel_mean"] = round(float(np.mean(np.abs(accel))), 2)
            stats["stat_accel_p95"] = round(float(np.percentile(np.abs(accel), 95)), 2)
        else:
            stats["stat_accel_mean"] = 0.0
            stats["stat_accel_p95"] = 0.0
    else:
        stats["stat_speed_mean"] = 0.0
        stats["stat_speed_p95"] = 0.0
        stats["stat_accel_mean"] = 0.0
        stats["stat_accel_p95"] = 0.0

    return stats


def resolve_full_frame_range(
    root:          Path,
    full_trial:    str,
    subject:       str,
    cam:           int,
    per_cam:       dict,
    event_frame:   int,
    fps:           float,
    total_frames:  int,
    logger:        logging.Logger,
) -> Optional[tuple[int, int]]:
    """
    按 GRF 有效时间与 sync 计算可提取的视频帧闭区间 [start_f, end_f]。

    - 末端：由 GRF timestamps 上限映射到视频帧，且不超过视频总长（视频长于 GRF 时截断）。
    - 起点：由 GRF 下限映射到视频帧后，再与 **sync 标注帧 video_event_frame** 取 max：
      只保证「从标注帧起」与力台对齐；标注之前的片头再长也不处理。
    - event_frame 须落在最终 [start_f, end_f] 内（用于身份锚点）。
    """
    cam_data = per_cam.get(str(cam), {})
    ves = float(cam_data.get("video_event_sec", 0.0) or 0.0)
    vef = int(cam_data.get("video_event_frame", event_frame) or event_frame)
    off = float(cam_data.get("offset_sec", 0.0) or 0.0)
    grf_path = root / subject / "labels" / f"{full_trial}_grf.npy"
    bounds = _load_grf_timestamp_bounds(grf_path)
    if bounds is None:
        logger.error("  无法读取 GRF 时间范围：%s", grf_path)
        return None
    t_lo, t_hi = bounds
    start_f, end_f = _grf_aligned_frame_bounds(
        t_lo, t_hi, ves, vef, off, fps, total_frames,
    )
    if start_f > end_f:
        logger.error("  GRF 对齐帧范围为空")
        return None

    # 仅信任从 sync 标注帧起的对齐；片头可能长于 GRF 起点映射，但不纳入
    trust_from = max(0, min(vef, total_frames - 1))
    if start_f < trust_from:
        logger.info(
            "  start_f %d → %d（仅处理 sync 帧及之后；此前视频不对齐）",
            start_f, trust_from,
        )
        start_f = trust_from

    if start_f > end_f:
        logger.error("  截断 sync 信任起点后帧范围为空 [%d,%d]", start_f, end_f)
        return None
    if event_frame < start_f or event_frame > end_f:
        logger.error(
            "  sync 帧 %d 不在可用区间 [%d,%d] 内",
            event_frame, start_f, end_f,
        )
        return None
    logger.info(
        "  最终帧范围：[%d,%d]（GRF t∈[%.4f,%.4f]s，信任起点≥sync 帧 %d）",
        start_f, end_f, t_lo, t_hi, trust_from,
    )
    return start_f, end_f


# ──────────────────────────────────────────────────────────────────────────────
# 核心提取
# ──────────────────────────────────────────────────────────────────────────────

def extract_window(
    video_path:  Path,
    event_frame: int,
    start_f:     int,
    end_f:       int,
    video_fps:   float,
    model,
    model_name:  str,
    device:      str,
    logger:      logging.Logger,
    *,
    infer_half: bool = True,
    imgsz: Optional[int] = None,
    cuda_sync_every: int = 0,
) -> Optional[dict]:
    """
    在 [start_f, end_f] 闭区间上提取骨骼点（须包含 sync 的 event_frame）。

    video_fps：已由 resolve_video_fps 给出（sync / ffprobe / OpenCV），写入 npz，用于对齐。

    流程：
      1. 流式推理（单次遍历；ByteTrack 须按时间序，无法像 detect 那样大批次堆帧）
      2. identify_athlete_at_ev_local：竖向冲击 + 水平居中
      3. choose_athlete_per_frame：三锚点全程过滤

    infer_half / imgsz：与 build_model 一致；imgsz 越小通常越快（略降精度）。
    cuda_sync_every：>0 时每 N 帧 cuda.synchronize + empty_cache（配合 --cuda-safe，默认 200，见 --cuda-sync-every）。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"  无法打开：{video_path}")
        return None

    fps = float(video_fps)
    if fps <= 1e-6:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 119.88
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_f = max(0, int(start_f))
    end_f   = min(total_frames - 1, int(end_f))
    if start_f > end_f:
        logger.error("  无效帧范围：start_f=%d end_f=%d", start_f, end_f)
        cap.release()
        return None
    if event_frame < start_f or event_frame > end_f:
        logger.error(
            "  sync 帧 %d 不在 GRF 对齐范围 [%d,%d] 内",
            event_frame, start_f, end_f,
        )
        cap.release()
        return None
    ev_local = event_frame - start_f
    logger.info(
        f"  {video_path.name}  {width}×{height}@{fps:.2f}fps  "
        f"GRF对齐[{start_f}~{end_f}]  共{end_f-start_f+1}帧  ev_local={ev_local}"
    )

    reset_tracker(model)
    tracker_yaml = _bytetrack_yaml_path()  # Fix-1: track_buffer=600
    use_half = bool(infer_half) and ("cuda" in str(device).lower())
    track_kw = dict(
        persist=True,
        device=device,
        verbose=False,
        conf=0.3,
        iou=0.7,
        tracker=tracker_yaml,
    )
    if use_half:
        track_kw["half"] = True
    if imgsz is not None:
        track_kw["imgsz"] = int(imgsz)

    all_preds:     list[list[dict]] = []
    frame_indices: list[int]        = []
    plate_sample:  list[np.ndarray] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    t0 = time.time()

    torch_mod = None
    if cuda_sync_every > 0 and "cuda" in str(device).lower():
        try:
            import torch as _torch
            torch_mod = _torch
        except Exception:
            torch_mod = None

    for abs_fi in range(start_f, end_f + 1):
        ret, frame = cap.read()
        if not ret:
            break
        local_i = abs_fi - start_f

        if local_i < 10:
            plate_sample.append(frame.copy())

        results = model.track(frame, **track_kw)
        all_preds.append(parse_yolo_result(results[0]))
        frame_indices.append(abs_fi)

        if (
            torch_mod is not None
            and (local_i + 1) % cuda_sync_every == 0
        ):
            try:
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.synchronize()
                    torch_mod.cuda.empty_cache()
            except Exception:
                pass

        if (local_i + 1) % 300 == 0:
            elapsed  = time.time() - t0
            fps_proc = (local_i + 1) / elapsed
            eta      = (end_f - abs_fi) / max(fps_proc, 1e-6)
            logger.info(f"    {local_i+1}/{end_f-start_f+1}帧  "
                        f"{fps_proc:.1f}fps  ETA={eta:.0f}s")

    cap.release()

    if not all_preds:
        logger.error("  读帧失败")
        return None

    ev_local = min(ev_local, len(all_preds) - 1)

    # 力台中心（仅可视化，不参与逻辑）
    plate_center = detect_plate_center(plate_sample)
    if plate_center is None:
        plate_center = np.array([float("nan"), float("nan")], dtype=np.float32)
    else:
        logger.info(f"  力台中心（仅可视化）：({plate_center[0]:.0f}, {plate_center[1]:.0f})")

    # 竖向冲击识别
    athlete_tid, anchor_bbox = identify_athlete_at_ev_local(
        all_preds, ev_local, width, logger
    )
    if athlete_tid is None or anchor_bbox is None:
        logger.error("  无法识别运动员，跳过此 trial/cam")
        return None

    # 逐帧过滤
    chosen_tids, statuses = choose_athlete_per_frame(
        all_preds, athlete_tid, anchor_bbox, logger
    )
    keypoints, scores, bboxes = extract_values_by_tid(all_preds, chosen_tids)
    stats = _quality_stats(keypoints, scores, bboxes, statuses, fps)

    return {
        "keypoints":          keypoints,
        "scores":             scores,
        "bbox":               bboxes,
        "frame_indices":      np.array(frame_indices, dtype=np.int32),
        "track_status":       np.array(statuses, dtype="U20"),
        "event_frame_abs":    np.int32(event_frame),
        "event_frame_local":  np.int32(ev_local),
        "fps":                np.float32(fps),
        "width":              np.int32(width),
        "height":             np.int32(height),
        "n_frames":           np.int32(len(all_preds)),
        "plate_center_px":    plate_center,
        "model_name":         model_name,
        **stats,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 任务扫描
# ──────────────────────────────────────────────────────────────────────────────

def scan_tasks(
    root:     Path,
    subjects: Optional[list[str]] = None,
    cameras:  Optional[list[int]] = None,
) -> tuple[list[dict], list[str]]:
    """
    每任务对应一个 {trial}_cam{N}_pose.npz（GRF 对齐全长，schema v2）。

    返回 (tasks, skipped_cam1_4k_auto)：后者为「自动识别为 4K 而排除的 cam1」说明串列表，
    便于日志汇总（手工名单 CAM1_4K_SKIP 不计入此列表）。
    """
    all_cams = cameras or list(range(1, 9))
    tasks: list[dict] = []
    skipped_cam1_4k_auto: list[str] = []
    for sync_file in sorted(root.glob("*/labels/*_sync.json")):
        subject = sync_file.parts[-3]
        trial   = sync_file.stem.replace("_sync", "")
        if trial.startswith(f"{subject}_"):
            trial = trial[len(f"{subject}_"):]
        if subjects and subject not in subjects:
            continue
        try:
            sync = json.loads(sync_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        per_cam    = sync.get("per_cam", {})
        if not per_cam:
            continue
        full_trial = sync.get("trial", f"{subject}_{trial}")
        for cam in all_cams:
            cam_str = str(cam)
            if cam_str not in per_cam:
                continue
            if cam == 1 and trial in CAM1_4K_SKIP.get(subject, []):
                continue
            ev_frame = per_cam[cam_str].get("video_event_frame")
            if ev_frame is None:
                continue
            vpath = (root / subject / "video" / f"cam{cam}"
                     / f"{full_trial}_cam{cam}.mp4")
            if not vpath.exists():
                continue
            if cam == 1:
                wh = _video_wh_quick(vpath)
                if wh is not None and _is_cam1_4k_resolution(*wh):
                    w, h = wh
                    skipped_cam1_4k_auto.append(
                        f"{subject}/{trial} cam1 {w}×{h}"
                    )
                    continue
            out_path = root / subject / "pose" / f"{full_trial}_cam{cam}_pose.npz"
            tasks.append({
                "root":         root,
                "subject":      subject,
                "trial":        trial,
                "cam":          cam,
                "event_frame":  int(ev_frame),
                "video_path":   vpath,
                "sync_file":    sync_file,
                "full_trial":   full_trial,
                "video_mtime":  vpath.stat().st_mtime,
                "out_path":     out_path,
                "done":         _npz_is_valid(out_path),
            })
    return tasks, skipped_cam1_4k_auto


# ──────────────────────────────────────────────────────────────────────────────
# 单任务处理
# ──────────────────────────────────────────────────────────────────────────────

def process_task(task: dict, model, model_name: str,
                 device: str, logger: logging.Logger,
                 *, overwrite: bool = False,
                 infer_half: bool = True,
                 imgsz: Optional[int] = None,
                 cuda_sync_every: int = 0) -> dict:
    subject = task["subject"]
    trial   = task["trial"]
    cam     = task["cam"]
    out_p   = task["out_path"]
    root    = task["root"]

    if not overwrite and _npz_is_valid(out_p):
        with np.load(out_p, allow_pickle=True) as d:
            entry = {k: (v.item() if hasattr(v, "item") else str(v))
                     for k, v in d.items() if k.startswith("stat_")}
        entry.update({"subject": subject, "trial": trial,
                      "cam": cam, "status": "already_done"})
        logger.info(f"  已完成，跳过：{out_p.name}")
        return entry

    if out_p.exists() and not overwrite:
        logger.warning("  已有 npz 可能损坏/不完整，重新提取")

    if cv2 is None:
        return {"subject": subject, "trial": trial, "cam": cam,
                "status": "error_no_cv2"}

    t0 = time.time()
    sync = json.loads(task["sync_file"].read_text(encoding="utf-8"))
    per_cam = sync.get("per_cam", {})
    full_trial = task.get("full_trial") or sync.get("trial", f"{subject}_{trial}")
    vpath = Path(task["video_path"])
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return {"subject": subject, "trial": trial,
                "cam": cam, "status": "error_video_open"}
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cam == 1 and _is_cam1_4k_resolution(vw, vh):
        cap.release()
        logger.info(
            "  跳过 cam1（%d×%d 判定为 4K），不产出 pose",
            vw, vh,
        )
        return {"subject": subject, "trial": trial, "cam": cam,
                "status": "skipped_cam1_4k"}

    fps = resolve_video_fps(vpath, cap, sync, cam, logger)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    rng = resolve_full_frame_range(
        root, full_trial, subject, cam, per_cam,
        task["event_frame"], fps, total_frames, logger,
    )
    if rng is None:
        return {"subject": subject, "trial": trial, "cam": cam,
                "status": "error_full_range"}
    start_f, end_f = rng
    try:
        data = extract_window(
            video_path=task["video_path"],
            event_frame=task["event_frame"],
            start_f=start_f,
            end_f=end_f,
            video_fps=fps,
            model=model,
            model_name=model_name,
            device=device,
            logger=logger,
            infer_half=infer_half,
            imgsz=imgsz,
            cuda_sync_every=cuda_sync_every,
        )
    except (TypeError, RuntimeError) as e:
        # Some torch/ultralytics builds mis-handle FP16 on certain frames (conv2d
        # "invalid combination" with a str in the signature). Retry whole clip FP32.
        err_s = str(e)
        if infer_half and ("cuda" in str(device).lower()) and (
            "conv2d" in err_s or "invalid combination" in err_s
        ):
            logger.warning(
                "  本任务 FP16 推理异常，整段改用 FP32 重试：%s",
                err_s[:160],
            )
            data = extract_window(
                video_path=task["video_path"],
                event_frame=task["event_frame"],
                start_f=start_f,
                end_f=end_f,
                video_fps=fps,
                model=model,
                model_name=model_name,
                device=device,
                logger=logger,
                infer_half=False,
                imgsz=imgsz,
                cuda_sync_every=cuda_sync_every,
            )
        else:
            raise

    if data is None:
        return {"subject": subject, "trial": trial,
                "cam": cam, "status": "error_extract_failed"}

    out_p.parent.mkdir(parents=True, exist_ok=True)

    stat_arrays = {
        k: np.float32(v) if isinstance(v, float) else np.int32(v)
        for k, v in data.items() if k.startswith("stat_")
    }

    _write_npz_atomic(
        out_p,
        keypoints         = data["keypoints"],
        scores            = data["scores"],
        bbox              = data["bbox"],
        frame_indices     = data["frame_indices"],
        track_status      = data["track_status"],
        event_frame_abs   = data["event_frame_abs"],
        event_frame_local = data["event_frame_local"],
        fps               = data["fps"],
        width             = data["width"],
        height            = data["height"],
        n_frames          = data["n_frames"],
        plate_center_px   = data["plate_center_px"],
        model             = np.str_(data["model_name"]),
        subject           = np.str_(subject),
        trial             = np.str_(trial),
        camera            = np.int32(cam),
        extracted_at      = np.str_(datetime.now().isoformat()),
        schema_version    = np.str_("pose_npz_v2"),
        pose_coverage     = np.str_("grf_aligned"),
        **stat_arrays,
    )

    elapsed = time.time() - t0
    logger.info(
        f"  ✅ {out_p.name}  "
        f"({int(data['n_frames'])}帧  {elapsed/60:.1f}min)  "
        f"下肢置信度={data['stat_lower_body_mean_score']:.3f}  "
        f"丢失率={data['stat_lost_rate']:.1%}"
    )

    entry = {k: (v.item() if hasattr(v, "item") else v)
             for k, v in data.items() if k.startswith("stat_")}
    entry.update({"subject": subject, "trial": trial, "cam": cam,
                  "status": "done", "elapsed_min": round(elapsed / 60, 2)})
    return entry


# ──────────────────────────────────────────────────────────────────────────────
# 日志工具
# ──────────────────────────────────────────────────────────────────────────────

def _log_path(root: Path) -> Path:
    return root / "pose_extraction_log.json"


def load_log(root: Path) -> list:
    p = _log_path(root)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("entries", [])
        except Exception:
            pass
    return []


def save_log(root: Path, entries: list):
    _log_path(root).write_text(
        json.dumps({"updated_at": datetime.now().isoformat(),
                    "entries": entries},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_summary_csv(path: Path, entries: list[dict]):
    headers = sorted({k for e in entries for k in e.keys()})
    if not headers:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in entries:
            writer.writerow(row)


def _serialize_task(task: dict) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in task.items()}


def _deserialize_task(d: dict) -> dict:
    out = dict(d)
    for key in ("root", "video_path", "sync_file", "out_path"):
        if key in out and out[key] is not None:
            out[key] = Path(out[key])
    return out


def _run_single_task_worker(args) -> None:
    """
    单个子进程入口：处理一个 task 并将结果写入 --result-json。
    供 --isolate-tasks 使用；段错误时不会执行到写文件。
    """
    root = args.root if args.root is not None else _default_data_root()
    if not args.single_task_json or not args.result_json:
        print("--single-task-json 与 --result-json 必须同时给出", file=sys.stderr)
        sys.exit(2)

    task = _deserialize_task(
        json.loads(args.single_task_json.read_text(encoding="utf-8"))
    )

    log_file = root / "pose_extraction.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("step3")
    infer_half = (not args.fp32) and ("cuda" in str(args.device).lower())
    cuda_sync_every = _cuda_sync_every_from_args(args)
    logger.info(
        f"[子进程] {task['subject']}/{task['trial']} cam{task['cam']}  "
        f"infer={'fp16' if infer_half else 'fp32'}  "
        f"cuda_sync_every={cuda_sync_every if cuda_sync_every else 'off'}"
    )
    model, model_name = build_model(
        args.device,
        half=infer_half,
        cuda_safe=bool(args.cuda_safe),
        logger=logger,
    )
    try:
        entry = process_task(
            task, model, model_name, args.device, logger,
            overwrite=args.overwrite,
            infer_half=infer_half,
            imgsz=args.imgsz,
            cuda_sync_every=cuda_sync_every,
        )
    except Exception as e:
        logger.exception("子进程单任务异常")
        entry = {
            "subject": task["subject"],
            "trial":   task["trial"],
            "cam":     task["cam"],
            "status":  "exception",
            "error":   str(e),
        }
    args.result_json.write_text(
        json.dumps(entry, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def print_summary(root: Path, logger: logging.Logger):
    entries = load_log(root)
    done    = [e for e in entries
               if e.get("status") in ("done", "already_done")
               and "stat_lower_body_mean_score" in e]
    if not done:
        return
    lower = [e["stat_lower_body_mean_score"] for e in done]
    hcr   = [e["stat_high_conf_frame_rate"]  for e in done]
    lost  = [e["stat_lost_rate"]             for e in done]
    speed = [e.get("stat_speed_mean", 0)      for e in done]
    accel = [e.get("stat_accel_p95", 0)       for e in done]
    logger.info("\n── 质量汇总（论文 Table 参考）──")
    logger.info(f"  完成样本：          {len(done)}")
    logger.info(f"  下肢关键点置信度：  {np.mean(lower):.3f} ± {np.std(lower):.3f}")
    logger.info(f"  高置信帧率(>0.5)：  {np.mean(hcr):.1%} ± {np.std(hcr):.1%}")
    logger.info(f"  运动员丢失率：      {np.mean(lost):.2%} ± {np.std(lost):.2%}")
    logger.info(f"  中心速度均值(px/s)：{np.mean(speed):.1f} ± {np.std(speed):.1f}")
    logger.info(f"  加速度P95(px/s²)：  {np.mean(accel):.1f} ± {np.std(accel):.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def _cuda_sync_every_from_args(args) -> int:
    """--cuda-safe 时按帧同步；默认 200（原为 500），可用 --cuda-sync-every 覆盖。"""
    if not getattr(args, "cuda_safe", False):
        return 0
    v = getattr(args, "cuda_sync_every", None)
    if v is not None:
        return max(1, int(v))
    return 200


def main():
    ap = argparse.ArgumentParser(description="BadmintonGRF Step 3: 骨骼点提取")
    ap.add_argument("--root",     type=Path, default=None)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--cameras",  nargs="+", type=int, default=None)
    ap.add_argument("--device",   default="cuda:0")
    ap.add_argument(
        "--fp32",
        action="store_true",
        help="禁用 FP16，全精度推理（更慢；若出现 Segmentation fault 建议与 --cuda-safe 同用）",
    )
    ap.add_argument(
        "--cuda-safe",
        action="store_true",
        help="关闭 cudnn.benchmark，并定期 cuda.synchronize+empty_cache；无法捕获段错误，但可降低部分 GPU 崩溃概率",
    )
    ap.add_argument(
        "--cuda-sync-every",
        type=int,
        default=None,
        metavar="N",
        help="与 --cuda-safe 同用：每 N 帧同步一次；默认 200，更长视频若仍 SIGSEGV 可试 100 或 64",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=None,
        metavar="N",
        help="YOLO 推理边长（如 960、640）；越小越快、略降精度，默认用模型配置",
    )
    ap.add_argument("--dry_run",  action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing npz even if valid")
    ap.add_argument("--export_qc_fig", action="store_true",
                    help="Export a paper-ready QC figure for one pose npz, then exit")
    ap.add_argument("--qc_npz", type=Path, default=None,
                    help="Path to a *_pose.npz file (default: auto-pick one under --root)")
    ap.add_argument("--qc_out", type=Path, default=None,
                    help="Output PDF path (default: paper/figures/fig_pose_tracking_qc.pdf)")
    ap.add_argument(
        "--isolate-tasks",
        action="store_true",
        help="每个任务在独立子进程中运行；CUDA 段错误只影响当前任务，不终止整批（每次子进程会重新加载模型，更慢）",
    )
    ap.add_argument(
        "--one-task-per-run",
        action="store_true",
        help="每次进程仅处理 1 个待处理任务（成功或失败都退出）；配合外层 while 自动重启可降低进程污染累积",
    )
    ap.add_argument("--single-task-json", type=Path, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--result-json", type=Path, default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    root = args.root if args.root is not None else _default_data_root()
    if not root.exists():
        print(f"错误：数据根目录不存在: {root}", file=sys.stderr)
        sys.exit(1)

    if args.single_task_json is not None or args.result_json is not None:
        if args.single_task_json is None or args.result_json is None:
            print("错误：--single-task-json 与 --result-json 必须同时指定", file=sys.stderr)
            sys.exit(2)
        _run_single_task_worker(args)
        return

    if args.export_qc_fig:
        npz_path = args.qc_npz
        if npz_path is None:
            cands = sorted(root.rglob("*_pose.npz"))
            npz_path = next((p for p in cands if _npz_is_valid(p)), None)
        if npz_path is None or not npz_path.exists():
            raise RuntimeError("未找到可用的 pose npz（请指定 --qc_npz 或先运行提取）")

        out_pdf = args.qc_out or (Path(__file__).resolve().parents[1] / "paper" / "figures" / "fig_pose_tracking_qc.pdf")
        export_pose_qc_figure(npz_path, out_pdf)
        print(f"[QC] saved → {out_pdf}")
        return

    log_file = root / "pose_extraction.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("step3")
    logger.info("=" * 60)
    logger.info("BadmintonGRF Step 3: 骨骼点提取 (YOLO26-pose)")
    infer_half = (not args.fp32) and ("cuda" in str(args.device).lower())
    cuda_sync_every = _cuda_sync_every_from_args(args)
    logger.info(
        f"根目录：{root}  device：{args.device}  "
        f"infer={'fp16' if infer_half else 'fp32'}  "
        f"imgsz={args.imgsz if args.imgsz is not None else 'default'}  "
        f"cuda_safe={bool(args.cuda_safe)}  "
        f"cuda_sync_every={cuda_sync_every if cuda_sync_every else 'off'}  "
        f"isolate_tasks={bool(args.isolate_tasks)}"
    )

    tasks, skipped_4k = scan_tasks(root, args.subjects, args.cameras)
    if skipped_4k:
        logger.info(
            "cam1 4K 自动跳过 %d 个（不产出 pose，与 benchmark 排除策略一致）",
            len(skipped_4k),
        )
        for line in skipped_4k[:40]:
            logger.info("  自动跳过 %s", line)
        if len(skipped_4k) > 40:
            logger.info("  … 另有 %d 个未列出", len(skipped_4k) - 40)
    done_cnt = sum(t["done"] for t in tasks)
    pending  = [t for t in tasks if not t["done"] or args.overwrite]
    logger.info(f"找到任务：{len(tasks)}  已完成：{done_cnt}  待处理：{len(pending)}")

    by_sub = defaultdict(lambda: {"total": 0, "done": 0})
    for t in tasks:
        by_sub[t["subject"]]["total"] += 1
        by_sub[t["subject"]]["done"]  += int(t["done"])
    for sub, v in sorted(by_sub.items()):
        flag = "✅" if v["done"] == v["total"] else "🔄"
        logger.info(f"  {flag} {sub}: {v['done']}/{v['total']}")

    if args.dry_run:
        logger.info("dry_run 退出")
        return
    if not pending:
        logger.info("全部已完成 ✅")
        write_summary_csv(root / "pose_extraction_summary.csv", load_log(root))
        print_summary(root, logger)
        return

    if not args.isolate_tasks:
        logger.info(f"加载模型（{args.device}）...")
        model, model_name = build_model(
            args.device,
            half=infer_half,
            cuda_safe=bool(args.cuda_safe),
            logger=logger,
        )
        logger.info(f"模型就绪：{model_name} ✅")
    else:
        logger.info("isolate-tasks：每个任务在独立子进程中运行（每任务重新加载模型，较慢）")
        model = model_name = None

    logger.info("=" * 60)

    log_entries = load_log(root)
    ok = fail = 0

    script_path = Path(__file__).resolve()

    def _iter_exception_chain(exc: BaseException):
        """遍历 __cause__ / __context__，避免链式异常里只看最外层字符串漏判。"""
        seen = set()
        stack = [exc]
        while stack:
            e = stack.pop()
            if e is None or id(e) in seen:
                continue
            seen.add(id(e))
            yield e
            stack.append(getattr(e, "__cause__", None))
            stack.append(getattr(e, "__context__", None))

    def _is_fatal_runtime_corruption(exc: BaseException) -> bool:
        """
        运行时污染类错误：继续在同一进程里跑很容易触发更严重的崩溃（如 segfault）。
        触发条件越“窄”，越不影响正常任务推进。
        """
        for exc in _iter_exception_chain(exc):
            # Conv.forward_fuse 访问 self.act 时 str(KeyError) 仅为 'act'，无更长文案
            if isinstance(exc, KeyError) and exc.args and exc.args[0] == "act":
                return True
            msg_low = str(exc).lower()
            # SiLU 被错当成 set 等（污染后 __getattr__ 走到 Module 的 discard 路径）
            if "silu" in msg_low and "discard" in msg_low:
                return True
            # ultralytics/torch 层出现模块属性被污染：act 变成 tuple 等。
            if "tuple" in msg_low and "object is not callable" in msg_low:
                return True
            # torch 内部（如 F.silu → has_torch_function_unary）被污染成 Tensor 等
            if "tensor" in msg_low and "object is not callable" in msg_low:
                return True
            # 模块被污染为 None/list/SiLU 等非预期类型
            if "nonetype" in msg_low and "object is not callable" in msg_low:
                return True
            if "list" in msg_low and "has no attribute 'groups'" in msg_low:
                return True
            if "silu_()" in msg_low and "must be tensor, not silu" in msg_low:
                return True
            # CUDAGuard 失败后偶见：silu_ / C 扩展侧报空属性名（进程内 CUDA 状态已不可靠）
            if "attribute name must be string" in msg_low:
                return True
            # Module.__getattr__ / code 对象损坏（如 XXX lineno: -1, opcode: 139）
            if "unknown opcode" in msg_low:
                return True
            # nn.Module._parameters 被污染成 Conv2d，`name in _parameters` → not iterable
            if "not iterable" in msg_low and "conv2d" in msg_low:
                return True
            # 模块/容器结构被污染：Conv/Dict 等缺少正常属性；或 docstring 被当成属性名
            # （如 'Conv' object has no attribute 'Forward pass for training mode.\n\n Args:...')
            if "object has no attribute" in msg_low and (
                "'torch'" in msg_low
                or "'iter'" in msg_low
                or "forward pass" in msg_low
            ):
                return True
            # 典型的“运行时图结构/层结构损坏”表现
            if "tuple index out of range" in msg_low:
                return True
            # conv2d 签名里出现 str/tuple 错位（常见于 nn.Conv2d 内部字段被污染后的 F.conv2d 调用）
            if "conv2d" in msg_low and "invalid combination" in msg_low:
                return True
            # Module._call_impl 展开 *args 时期望序列却得到 Conv（图/子模块列表损坏）
            if "must be an iterable" in msg_low and "not conv" in msg_low:
                return True
            # PyTorch CUDA guard 内部断言（CUDAGuardImpl.h）：
            if "cudaguard" in msg_low and "internal assert failed" in msg_low:
                return True
            if "d.is_cuda()" in msg_low and "internal assert failed" in msg_low:
                return True
            # 仍留一点保护：遇到“INTERNAL ASSERT FAILED”但不确定是哪类，也优先让 watchdog 重启。
            if "internal assert failed" in msg_low and ("cuda" in msg_low or "cudaguard" in msg_low):
                return True
        return False

    for i, task in enumerate(pending):
        logger.info(
            f"\n[{i+1}/{len(pending)}] "
            f"{task['subject']}/{task['trial']} cam{task['cam']}  "
            f"event={task['event_frame']}"
        )
        try:
            if args.isolate_tasks:
                fd, task_path = tempfile.mkstemp(suffix=".json")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(_serialize_task(task), f, ensure_ascii=False)
                fd2, result_path = tempfile.mkstemp(suffix=".json")
                os.close(fd2)
                result_path = Path(result_path)
                cmd = [
                    sys.executable,
                    "-u",
                    str(script_path),
                    "--root",
                    str(root),
                    "--device",
                    args.device,
                    "--single-task-json",
                    task_path,
                    "--result-json",
                    str(result_path),
                ]
                if args.fp32:
                    cmd.append("--fp32")
                if args.cuda_safe:
                    cmd.append("--cuda-safe")
                    cmd.extend(
                        ["--cuda-sync-every", str(_cuda_sync_every_from_args(args))]
                    )
                if args.overwrite:
                    cmd.append("--overwrite")
                if args.imgsz is not None:
                    cmd.extend(["--imgsz", str(args.imgsz)])
                try:
                    r = subprocess.run(cmd)
                finally:
                    try:
                        os.unlink(task_path)
                    except OSError:
                        pass
                entry = None
                if result_path.exists() and result_path.stat().st_size > 0:
                    try:
                        entry = json.loads(
                            result_path.read_text(encoding="utf-8")
                        )
                    except Exception as ex:
                        logger.error("  读取子进程结果 JSON 失败：%s", ex)
                try:
                    result_path.unlink(missing_ok=True)
                except OSError:
                    pass
                if entry is None:
                    rc = r.returncode
                    sig = None
                    if rc is not None:
                        if rc < 0:
                            # Linux: 被信号杀死时常为 -SIG（如 -11 = SIGSEGV）
                            sig = -rc
                        elif rc >= 128:
                            sig = rc - 128
                    is_segv = sig == 11
                    if is_segv:
                        logger.error(
                            "  子进程 SIGSEGV（信号 11，returncode=%s），多为 CUDA/cuDNN/驱动问题",
                            rc,
                        )
                    else:
                        logger.error(
                            "  子进程异常退出（returncode=%s，signal=%s）",
                            rc,
                            sig,
                        )
                    entry = {
                        "subject":      task["subject"],
                        "trial":        task["trial"],
                        "cam":          task["cam"],
                        "status":       "segfault_or_crash" if is_segv else "subprocess_failed",
                        "returncode":   rc,
                        "signal":       sig,
                    }
                log_entries.append(entry)
            else:
                entry = process_task(
                    task, model, model_name, args.device, logger,
                    overwrite=args.overwrite,
                    infer_half=infer_half,
                    imgsz=args.imgsz,
                    cuda_sync_every=cuda_sync_every,
                )
                log_entries.append(entry)
            if entry["status"] in ("done", "already_done", "skipped_cam1_4k"):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            # 运行时污染类错误：不再让进程继续沿用被污染的 model。
            # 在非 isolate 模式下，我们尝试“重建模型并重试当前任务”，以尽量避免
            # 这种污染向后持续扩散（这相当于进程级 watchdog，但在脚本内部完成）。
            if (not args.isolate_tasks) and _is_fatal_runtime_corruption(e):
                logger.error("  ❌ 运行时污染疑似错误：%s；重建模型并重试一次", str(e))
                try:
                    try:
                        import torch as _torch
                        if ("cuda" in str(args.device).lower()) and _torch.cuda.is_available():
                            _torch.cuda.synchronize()
                            _torch.cuda.empty_cache()
                    except Exception:
                        pass
                    model, model_name = build_model(
                        args.device,
                        half=infer_half,
                        cuda_safe=bool(args.cuda_safe),
                        logger=logger,
                    )
                    entry = process_task(
                        task, model, model_name, args.device, logger,
                        overwrite=args.overwrite,
                        infer_half=infer_half,
                        imgsz=args.imgsz,
                        cuda_sync_every=cuda_sync_every,
                    )
                    log_entries.append(entry)
                    if entry["status"] in ("done", "already_done", "skipped_cam1_4k"):
                        ok += 1
                    else:
                        fail += 1
                    continue
                except Exception as e2:
                    logger.error(
                        "  ❌ 重建/重试失败：%s",
                        str(e2),
                        exc_info=True,
                    )

            logger.error(f"  ❌ {e}", exc_info=True)
            log_entries.append({
                "subject": task["subject"], "trial": task["trial"],
                "cam": task["cam"], "status": "exception", "error": str(e),
            })
            fail += 1

        if (i + 1) % 10 == 0:
            save_log(root, log_entries)

        if args.one_task_per_run:
            save_log(root, log_entries)
            summary_path = root / "pose_extraction_summary.csv"
            write_summary_csv(summary_path, log_entries)
            logger.info("one-task-per-run：本轮仅处理 1 个任务，按设定提前退出")
            logger.info(f"  已写出 CSV 汇总：{summary_path}")
            logger.info("=" * 60)
            logger.info(f"当前累计  成功：{ok}  失败：{fail}")
            return

    save_log(root, log_entries)
    summary_path = root / "pose_extraction_summary.csv"
    write_summary_csv(summary_path, log_entries)
    logger.info(f"  已写出 CSV 汇总：{summary_path}")
    logger.info("=" * 60)
    logger.info(f"完成  成功：{ok}  失败：{fail}")
    print_summary(root, logger)


# ──────────────────────────────────────────────────────────────────────────────
# 可视化（debug / 论文 Figure）
# ──────────────────────────────────────────────────────────────────────────────

def visualize_npz(
    npz_path:   str,
    video_path: str,
    n_frames:   int = 0,
    out_path:   Optional[str] = None,
    start_local: int = 0,
    event_window: int = 30,
    playback_fps: Optional[float] = None,
):
    """
    骨骼点叠加渲染，用于 debug 和论文定性展示。
    n_frames=0 播完整个窗口；out_path 指定时保存为 mp4。
    start_local 可用于从指定局部帧开始播放。
    event_window 会高亮接触帧附近的窗口范围。
    按 q 退出，Space 暂停。

    播放/写 mp4 帧率：优先 playback_fps；否则优先 npz 内 fps（与 step3 对齐一致，避免容器
    CAP_PROP_FPS 误报成 60/30 导致慢放）；再退回视频 CAP_PROP_FPS；最后默认 119.88。

    颜色：
      绿色  tracked      = ByteTrack 正常追踪
      蓝色  pos_fallback = 位置最近邻兜底（Fix-3）
      橙色  reidentified = 强制重识别（Fix-5）
      红色  lost         = 丢失且兜底失败
    """
    if cv2 is None:
        raise RuntimeError("cv2 not available; install opencv-python to visualize pose overlays.")
    SKELETON = [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16),
    ]
    d        = dict(np.load(npz_path, allow_pickle=True))
    kps_all  = d["keypoints"]
    sc_all   = d["scores"]
    fidx     = d["frame_indices"]
    ev_local = int(d["event_frame_local"])
    statuses = d.get("track_status", None)
    npz_fps  = float(d.get("fps", 0)) or 0.0

    # 可视化轻量平滑（窗口均值，不改 npz）
    if kps_all.ndim == 3 and len(kps_all) > 1:
        kps_vis = kps_all.copy()
        hw = 2
        for i in range(len(kps_all)):
            s, e = max(0, i-hw), min(len(kps_all), i+hw+1)
            with np.errstate(invalid="ignore"):
                kps_vis[i] = np.nanmean(kps_all[s:e], axis=0)
    else:
        kps_vis = kps_all

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    cap_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if playback_fps is not None and playback_fps > 1e-6:
        fps_play = float(playback_fps)
    elif npz_fps > 1e-3:
        fps_play = npz_fps
    elif cap_fps > 1e-3:
        fps_play = cap_fps
    else:
        fps_play = 119.88

    if npz_fps > 1e-3 and cap_fps > 1e-3:
        rel = abs(npz_fps - cap_fps) / max(npz_fps, cap_fps)
        if rel > 0.12:
            print(
                f"[visualize_npz] fps: npz={npz_fps:.3f}  video_cap={cap_fps:.3f}  "
                f"→ 使用 {fps_play:.3f}（与 npz 对齐；若仍不对请传 playback_fps）",
                file=sys.stderr,
            )

    start_local = max(0, min(int(start_local), len(kps_all) - 1))
    start_abs   = int(fidx[0]) + start_local
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_abs)

    w = int(d.get("width", 1920))
    h = int(d.get("height", 1080))
    scale        = min(1280/w, 720/h, 1.0)
    display_size = (int(w*scale), int(h*scale))
    delay_ms     = max(1, int(1000.0 / fps_play))
    end_total    = len(kps_all) - start_local
    end          = end_total if n_frames <= 0 else min(n_frames, end_total)
    paused       = False
    col_map      = {
        "tracked":      (0,   200, 0),
        "pos_fallback": (0,   165, 255),
        "reidentified": (255, 165, 0),
        "lost":         (0,   0,   200),
    }

    writer = None
    if out_path is not None:
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"),
            fps_play, display_size,
        )

    contact_lo = max(0, ev_local - event_window)
    contact_hi = min(len(kps_all) - 1, ev_local + event_window)

    for i in range(end):
        li = start_local + i
        ret, frame = cap.read()
        if not ret:
            break
        kps = kps_vis[li]
        sc  = sc_all[li]

        for a, b in SKELETON:
            if sc[a]>0.3 and sc[b]>0.3 and not np.isnan(kps[a,0]):
                cv2.line(frame, tuple(kps[a].astype(int)),
                         tuple(kps[b].astype(int)), (30,200,30), 2)

        for ki, (kp, s) in enumerate(zip(kps, sc)):
            if s>0.3 and not np.isnan(kp[0]):
                col = (0,220,220) if ki in LOWER_BODY_IDX else (220,200,0)
                cv2.circle(frame, tuple(kp.astype(int)), 5, col, -1)

        if li == ev_local:
            cv2.rectangle(frame, (0,0),
                          (frame.shape[1]-1, frame.shape[0]-1), (0,0,255), 4)
            cv2.putText(frame, "★ CONTACT", (20,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,255), 3)
        elif contact_lo <= li <= contact_hi:
            cv2.rectangle(frame, (0,0),
                          (frame.shape[1]-1, frame.shape[0]-1), (40,140,255), 2)

        status_txt = str(statuses[li]) if statuses is not None else ""
        txt_col    = col_map.get(status_txt, (180,180,180))
        cv2.putText(frame, f"local={li}  status={status_txt}",
                    (10, frame.shape[0]-38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_col, 1)

        if li != ev_local:
            dt = li - ev_local
            cv2.putText(frame, f"ev_idx={ev_local}  dt={dt:+d}",
                        (10, frame.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
        else:
            cv2.putText(frame, f"ev_idx={ev_local}",
                        (10, frame.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        show = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
        if writer is not None:
            writer.write(show)
        else:
            cv2.imshow("BadmintonGRF Step3", show)
            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
        print(f"  已保存：{out_path}  ({fps_play:.2f} fps)")
    else:
        cv2.destroyAllWindows()


def export_pose_qc_figure(npz_path: Path, out_pdf: Path) -> Path:
    """
    Export a paper-friendly QC figure for Step3 (pose extraction).
    Panels:
      (a) bbox center_y around the contact event (impact anchor evidence)
      (b) tracking status timeline (tracked / fallback / reidentified / lost)
      (c) keypoint score distribution (privacy-friendly quality signal)
    """
    if plt is None:
        raise RuntimeError("matplotlib not available; install matplotlib to export QC figure.")

    d = dict(np.load(npz_path, allow_pickle=True))
    sc = d["scores"].astype(np.float32)            # (N,17)
    bbox = d["bbox"].astype(np.float32)            # (N,5) x1,y1,x2,y2,conf
    fi = d["frame_indices"].astype(np.int32)       # (N,)
    ev_local = int(d["event_frame_local"])
    statuses = d.get("track_status", np.array(["tracked"] * len(fi), dtype="U20"))

    cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
    x = np.arange(len(fi), dtype=int)

    # Window around event for display
    w = 60
    lo = max(0, ev_local - w)
    hi = min(len(fi), ev_local + w + 1)
    xw = x[lo:hi] - ev_local
    cyw = cy[lo:hi]

    # Baseline median before event
    b0 = max(0, ev_local - IMPACT_BASELINE_FRAMES)
    b1 = ev_local
    baseline = float(np.median(cy[b0:b1])) if b1 > b0 else float(np.median(cyw))

    uniq = ["tracked", "pos_fallback", "reidentified", "lost"]
    cmap = {
        "tracked": "#2563EB",
        "pos_fallback": "#F59E0B",
        "reidentified": "#059669",
        "lost": "#DC2626",
    }
    status_idx = {s: i for i, s in enumerate(uniq)}
    ysw = np.array([status_idx.get(str(s), 0) for s in statuses[lo:hi]], dtype=int)

    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Single-column friendly layout: stack panels vertically for readability.
    fig = plt.figure(figsize=(3.6, 4.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.70, 1.0], hspace=0.45)

    # (a) impact anchor evidence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(xw, cyw, color="#111827", linewidth=1.3)
    ax1.axvline(0, color="#DC2626", linestyle="--", linewidth=1.0)
    ax1.axhline(baseline, color="#6B7280", linestyle=":", linewidth=1.0)
    ax1.set_title("(a) Impact anchor", pad=3)
    ax1.set_xlabel("")
    ax1.set_ylabel("bbox center_y (px)")

    # (b) tracking status timeline
    ax2 = fig.add_subplot(gs[1, 0])
    for s in uniq:
        m = (ysw == status_idx[s])
        if m.any():
            ax2.scatter(xw[m], np.zeros_like(xw[m]), s=14,
                        color=cmap[s], alpha=0.9, linewidths=0, label=s)
    ax2.axvline(0, color="#DC2626", linestyle="--", linewidth=1.0)
    ax2.set_yticks([])
    ax2.set_ylim(-1, 1)
    ax2.set_title("(b) Tracking status", pad=3)
    ax2.set_xlabel("")
    ax2.legend(ncol=2, fontsize=7.5, frameon=False, loc="upper left",
               handletextpad=0.3, columnspacing=0.8)

    # (c) keypoint score distribution
    ax3 = fig.add_subplot(gs[2, 0])
    scores_flat = sc.reshape(-1)
    scores_flat = scores_flat[np.isfinite(scores_flat)]
    ax3.hist(scores_flat, bins=16, range=(0, 1.0), color="#2563EB", alpha=0.85,
             edgecolor="white", linewidth=0.5)
    ax3.set_title("(c) Keypoint scores", pad=3)
    ax3.set_xlabel("Confidence / frame")
    ax3.set_ylabel("Count")
    ax3.set_xlim(0, 1.0)
    # Shared x meaning for (a)/(b): show once at bottom.
    ax3.text(0.0, -0.36, "Frames relative to contact", transform=ax3.transAxes,
             ha="left", va="top", fontsize=8, color="#6B7280")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    main()