"""
数据读写工具
============
统一管理GRF、JSON、视频路径的加载和保存。
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from pipeline.config import CFG


def load_grf(trial: str) -> dict:
    """
    加载GRF数据。

    返回结构:
        {
            'timestamps': ndarray (N,),
            'combined': {'forces': ndarray (N,3), 'moments': ndarray (N,3)},
            'force_plate_1'...'force_plate_4': {...},
            'fps': 1000.0,
            'metadata': {...}
        }
    """
    path = CFG.grf_path(trial)
    if not path.exists():
        raise FileNotFoundError(f"GRF文件不存在: {path}")
    return np.load(path, allow_pickle=True).item()


def get_grf_fz(grf_data: dict, positive_down: bool = True) -> np.ndarray:
    """
    提取垂直力分量。
    Fz原始为负（向下），positive_down=True时取反为正值。
    """
    fz = grf_data['combined']['forces'][:, 2]
    return -fz if positive_down else fz


def load_detection(trial: str) -> dict:
    """加载峰值检测结果（detection.json）"""
    path = CFG.detection_path(trial)
    if not path.exists():
        raise FileNotFoundError(f"检测文件不存在: {path}")
    return load_json(path)


def load_annotated(trial: str) -> Optional[dict]:
    """加载标注结果（annotated.json），不存在则返回None"""
    path = CFG.annotated_path(trial)
    return load_json(path) if path.exists() else None


def load_alignment(trial: str) -> Optional[dict]:
    """加载对齐结果（alignment.json），不存在则返回None"""
    path = CFG.alignment_path(trial)
    return load_json(path) if path.exists() else None


def load_json(path) -> dict:
    """通用JSON加载"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, path, indent: int = 2):
    """通用JSON保存（自动创建父目录）"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def make_output_header(trial: str, step: str) -> dict:
    """生成标准JSON输出头"""
    return {
        "trial_name": trial,
        "created_at": datetime.now().isoformat(),
        "pipeline_version": "2.0",
        "step": step,
    }


def list_available_cameras(trial: str) -> List[int]:
    """列出某个trial可用的摄像机编号"""
    subject = CFG.trial_to_subject(trial)
    video_base = CFG.subject_dir(subject) / "video"
    cams = []
    for cam_dir in sorted(video_base.glob("cam*")):
        cam_id = int(cam_dir.name.replace("cam", ""))
        if (cam_dir / f"{trial}_cam{cam_id}.mp4").exists():
            cams.append(cam_id)
    return cams
