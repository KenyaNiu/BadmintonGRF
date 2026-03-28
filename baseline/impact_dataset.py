"""
PyTorch ``Dataset``：加载 step4 导出的 impact-segment ``.npz``。

环境变量 ``BADMINTON_IMPACT_RAM_CACHE=1``：在 ``__init__`` 末尾将全部样本读入内存
（``self._cache``），``__getitem__`` 不再访问磁盘；可规避 DataLoader worker 内
``np.load``/mmap 相关的偶发崩溃。注意：使用 ``spawn`` 多进程时每个 worker 会反序列化
一份 Dataset，RAM 占用 ≈ ×(1+num_workers) 量级，请确保内存充足。

输入特征 (T, 119)：
  pos   (T, 34)  归一化骨骼点坐标
  vel   (T, 34)  速度（一阶差分）
  acc   (T, 34)  加速度（二阶差分）
  score (T, 17)  关键点置信度

NaN 处理策略：
  lost 帧的 keypoints 可能为 NaN（step3 输出）。
  处理顺序：先用置信度 mask（score<0.1 的帧置零），再 nan_to_num，
  最后差分时遇到边界零值不会产生虚假速度。
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
INPUT_DIM = 119   # 34 pos + 34 vel + 34 acc + 17 scores

FILTER_LOWER_MIN   = 0.70
FILTER_LOST_MAX    = 0.05
FILTER_PEAK_BW_MAX = 3.0

_FLIP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

# Largest/smallest finite float32 (do not call np.finfo inside workers: spawn + NumPy 2.x
# can break finfo/getlimits cache and raise AttributeError on _finfo_cache).
_F32_MAX = 3.4028234663852886e38
_F32_MIN = -3.4028234663852886e38


def _replace_nonfinite_like_nan_to_num(
    a: np.ndarray,
    *,
    nan_fill: float = 0.0,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> np.ndarray:
    """
    ``np.nan_to_num`` 的等价替换，避免 NumPy 2.x 在 DataLoader worker 内触发
    ``nan_to_num`` / ``finfo`` / ``getlimits`` 的多进程竞态。
    """
    a = np.asarray(a, dtype=np.float32)
    if posinf is None:
        posinf = _F32_MAX
    if neginf is None:
        neginf = _F32_MIN
    x = np.where(np.isnan(a), np.float32(nan_fill), a)
    # Avoid np.isposinf/isneginf in worker processes (NumPy edge-case in some envs).
    inf_mask = np.isinf(a)
    pos_mask = inf_mask & (a > 0)
    neg_mask = inf_mask & (a < 0)
    x = np.where(pos_mask, np.float32(posinf), x)
    x = np.where(neg_mask, np.float32(neginf), x)
    return x.astype(np.float32, copy=False)


def build_features(
    kps_norm: np.ndarray,  # (T, 17, 2) float32，可能含 NaN
    scores:   np.ndarray,  # (T, 17)    float32
) -> np.ndarray:
    """
    构造 (T, 119) 特征向量，保证输出无 NaN/Inf。

    处理流程：
      1. 将低置信度帧（score < 0.1）的坐标置零，视为"无效帧"
      2. nan_to_num：残余 NaN/Inf → 0
      3. 差分计算 vel/acc
      4. 对 vel/acc 也做 nan_to_num（防止边缘情况）
      5. clip 到合理范围（位置[0,1]，速度/加速度[-0.5,0.5]）
    """
    kps_norm = np.ascontiguousarray(np.asarray(kps_norm, dtype=np.float32))
    scores = np.ascontiguousarray(np.asarray(scores, dtype=np.float32))
    if kps_norm.ndim != 3 or kps_norm.shape[1:] != (17, 2):
        raise ValueError(
            "build_features: expected keypoints (T,17,2), got %s" % (kps_norm.shape,)
        )
    n_frames = int(kps_norm.shape[0])
    if scores.shape != (n_frames, 17):
        raise ValueError(
            "build_features: scores shape %s != (%d, 17)"
            % (scores.shape, n_frames)
        )

    # step1：低置信度帧坐标置零
    sc_mask = (scores < 0.1)[:, :, None]           # (n_frames, 17, 1) bool
    kps = kps_norm.copy()
    kps[np.broadcast_to(sc_mask, kps.shape)] = 0.0

    # step2：nan/inf → 0（不用 np.nan_to_num：见 _replace_nonfinite_like_nan_to_num 说明）
    kps = _replace_nonfinite_like_nan_to_num(kps, nan_fill=0.0, posinf=1.0, neginf=0.0)
    kps = np.ascontiguousarray(
        np.minimum(np.maximum(kps, np.float32(0.0)), np.float32(1.0))
        .reshape(n_frames, 34)
        .astype(np.float32, copy=False)
    )

    sc = _replace_nonfinite_like_nan_to_num(scores, nan_fill=0.0).astype(np.float32)  # (n_frames,17)
    sc = np.minimum(np.maximum(sc, np.float32(0.0)), np.float32(1.0))

    # step3：速度（前向差分，首帧=0）
    vel = np.zeros((n_frames, 34), dtype=np.float32)
    if n_frames >= 2:
        np.subtract(kps[1:], kps[:-1], out=vel[1:])
    vel = _replace_nonfinite_like_nan_to_num(vel, nan_fill=0.0)
    vel = np.minimum(np.maximum(vel, np.float32(-0.5)), np.float32(0.5))

    # step4：加速度
    acc = np.zeros((n_frames, 34), dtype=np.float32)
    if n_frames >= 2:
        np.subtract(vel[1:], vel[:-1], out=acc[1:])
    acc = _replace_nonfinite_like_nan_to_num(acc, nan_fill=0.0)
    acc = np.minimum(np.maximum(acc, np.float32(-0.5)), np.float32(0.5))

    feat = np.concatenate([kps, vel, acc, sc], axis=1)  # (n_frames, 119)
    assert np.isfinite(feat).all(), "build_features: 输出仍含 NaN/Inf"
    return feat


# ---------------------------------------------------------------------------

class BadmintonImpactDataset(Dataset):
    def __init__(
        self,
        paths:           List[str],
        predict_fz_only: bool  = True,
        augment:         bool  = False,
        lower_min:       float = FILTER_LOWER_MIN,
        lost_max:        float = FILTER_LOST_MAX,
        peak_bw_max:     float = FILTER_PEAK_BW_MAX,
        ram_cache:       Optional[bool] = None,
    ):
        self.predict_fz_only = predict_fz_only
        self.augment         = augment

        if ram_cache is None:
            ram_cache = os.environ.get("BADMINTON_IMPACT_RAM_CACHE", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
        self._ram_cache = bool(ram_cache)
        self._cache: Optional[List[Dict[str, Any]]] = None

        valid_paths: List[str] = []
        meta: List[Dict[str, Any]] = []
        n_qual, n_miss = 0, 0
        for p in paths:
            pp = Path(p)
            if not pp.exists():
                n_miss += 1
                continue
            try:
                with np.load(str(pp), allow_pickle=True) as d:
                    lower = float(np.asarray(d.get("stat_lower_body_mean_score", 1.0)))
                    lost = float(np.asarray(d.get("stat_lost_rate", 0.0)))
                    pbw = float(np.asarray(d.get("peak_force_bw", 1.0)))

                    if lower < lower_min or lost > lost_max or pbw > peak_bw_max:
                        n_qual += 1
                        continue
                    valid_paths.append(str(pp))
                    meta.append(
                        {
                            "ev_idx": int(np.asarray(d["ev_idx"])),
                            "subject": str(d.get("subject", "")),
                            "stage": str(d.get("stage", "")),
                            "camera": int(np.asarray(d.get("camera", 0))),
                            "trial": str(d.get("trial", "")),
                        }
                    )
            except Exception:
                n_miss += 1
                continue

        self.paths = valid_paths
        self._meta = meta
        if n_qual or n_miss:
            warnings.warn(
                f"Dataset: {len(paths)} -> {len(valid_paths)} "
                f"(质量-{n_qual} 缺失-{n_miss})",
                stacklevel=2,
            )

        if self._ram_cache:
            self._cache = [self._materialize_numpy(i) for i in range(len(self.paths))]

    def _materialize_numpy(self, idx: int) -> Dict[str, Any]:
        """Load one sample from disk into float32 numpy (no augment). Used by RAM cache and disk path."""
        p = self.paths[idx]
        m = self._meta[idx]
        ev_idx = m["ev_idx"]

        try:
            dctx = np.load(p, mmap_mode="r", allow_pickle=False)
        except Exception:
            dctx = np.load(p, allow_pickle=True)

        with dctx:
            kps_norm = np.ascontiguousarray(
                np.array(dctx["keypoints_norm"], dtype=np.float32, copy=True)
            )
            scores = np.ascontiguousarray(
                np.array(dctx["scores"], dtype=np.float32, copy=True)
            )
            grf_norm = np.ascontiguousarray(
                np.array(dctx["grf_normalized"], dtype=np.float32, copy=True)
            )

        pose = build_features(kps_norm, scores)
        grf_full = np.ascontiguousarray(grf_norm.copy(), dtype=np.float32)
        target = grf_norm[:, 2:3] if self.predict_fz_only else grf_full
        if self.predict_fz_only:
            target = np.ascontiguousarray(target, dtype=np.float32)
        else:
            target = grf_full

        pose = np.ascontiguousarray(pose, dtype=np.float32)
        target = np.ascontiguousarray(target, dtype=np.float32)
        grf_full = np.ascontiguousarray(grf_full, dtype=np.float32)

        return {
            "pose": pose,
            "target": target,
            "grf_full": grf_full,
            "ev_idx": int(ev_idx),
            "path": p,
            "subject": m["subject"],
            "stage": m["stage"],
            "camera": m["camera"],
            "trial": m["trial"],
        }

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        """Return one sample from RAM cache or disk.

        ``BADMINTON_IMPACT_RAM_CACHE=1`` (or ``ram_cache=True``): preloads all
        samples in ``__init__`` so workers do **not** call ``np.load`` / mmap.
        Training augment still runs each epoch (copies from cache first).

        Default ``ram_cache=False``: mmap + disk reads per access; avoids
        multi-GB RAM and duplicate copies when DataLoader uses **spawn** workers
        (each worker unpickles a full cache).
        """
        if self._cache is not None:
            item = self._cache[idx]
            pose = item["pose"].copy()
            target = item["target"].copy()
            grf_full = item["grf_full"].copy()
            ev_idx = int(item["ev_idx"])
            p = item["path"]
            m = self._meta[idx]
        else:
            item = self._materialize_numpy(idx)
            pose = item["pose"]
            target = item["target"]
            grf_full = item["grf_full"]
            ev_idx = int(item["ev_idx"])
            p = item["path"]
            m = self._meta[idx]

        if self.augment:
            pose, target, grf_full, ev_idx = self._augment(
                pose, target, grf_full, ev_idx, pose.shape[0]
            )

        pose = np.ascontiguousarray(pose, dtype=np.float32)
        target = np.ascontiguousarray(target, dtype=np.float32)
        grf_full = np.ascontiguousarray(grf_full, dtype=np.float32)

        return {
            "pose": torch.from_numpy(pose),
            "target": torch.from_numpy(target),
            "grf_full": torch.from_numpy(grf_full),
            "ev_idx": torch.tensor(ev_idx, dtype=torch.long),
            "path": p,
            "subject": m["subject"],
            "stage": m["stage"],
            "camera": m["camera"],
            "trial": m["trial"],
        }

    def _augment(self, pose, target, grf_full, ev_idx, T):
        # pose 中 pos 段 = [:34]，score 段 = [102:119]
        pos = pose[:, :34].reshape(T, 17, 2).copy()
        sc  = pose[:, 102:].copy()

        # 1. 坐标噪声
        if np.random.rand() < 0.5:
            pos = np.clip(pos + np.random.randn(T,17,2).astype(np.float32)*0.003, 0, 1)

        # 2. 水平翻转
        if np.random.rand() < 0.5:
            pos[:, :, 0] = 1.0 - pos[:, :, 0]
            for l, r in _FLIP_PAIRS:
                pos[:, [l,r]] = pos[:, [r,l]]
                sc[:,  [l,r]] = sc[:,  [r,l]]
            grf_full = grf_full.copy(); grf_full[:, 0] = -grf_full[:, 0]
            if target.shape[-1] == 3:
                target = target.copy(); target[:, 0] = -target[:, 0]

        # 3. 时间偏移
        if np.random.rand() < 0.3:
            shift = int(np.random.randint(-5, 6))
            if shift:
                pos      = np.roll(pos,      shift, axis=0)
                sc       = np.roll(sc,       shift, axis=0)
                target   = np.roll(target,   shift, axis=0)
                grf_full = np.roll(grf_full, shift, axis=0)
                ev_idx   = int(np.clip(ev_idx - shift, 0, T-1))

        # 4. 置信度 dropout
        if np.random.rand() < 0.3:
            sc[np.random.rand(T, 17) < 0.10] = 0.0

        pose = build_features(pos, sc)
        return pose, target, grf_full, ev_idx

    def stage_distribution(self):
        return dict(Counter(m["stage"] for m in self._meta))

    def subject_distribution(self):
        return dict(Counter(m["subject"] for m in self._meta))


class BadmintonDataset(BadmintonImpactDataset):
    """Backward-compatible alias for older experiments."""

    def __init__(
        self,
        paths: List[str],
        fz_only: bool = True,
        augment: bool = False,
        ram_cache: Optional[bool] = None,
    ):
        super().__init__(
            paths=paths,
            predict_fz_only=fz_only,
            augment=augment,
            lower_min=FILTER_LOWER_MIN,
            lost_max=FILTER_LOST_MAX,
            peak_bw_max=FILTER_PEAK_BW_MAX,
            ram_cache=ram_cache,
        )



def build_loso_datasets(
    loso_splits_path: str,
    test_subject:     str,
    cameras:          Optional[List[int]] = None,
    predict_fz_only:  bool  = True,
    augment_train:    bool  = True,
    lower_min:        float = FILTER_LOWER_MIN,
    lost_max:         float = FILTER_LOST_MAX,
    peak_bw_max:      float = FILTER_PEAK_BW_MAX,
    ram_cache:          Optional[bool] = None,
) -> Tuple[BadmintonImpactDataset, BadmintonImpactDataset]:
    import json

    if ram_cache is None:
        ram_cache = os.environ.get("BADMINTON_IMPACT_RAM_CACHE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    splits = json.loads(Path(loso_splits_path).read_text(encoding="utf-8"))
    if test_subject not in splits:
        raise ValueError(f"{test_subject!r} 不在 splits 中，可用：{sorted(splits.keys())}")

    train_paths = splits[test_subject]["train"]
    test_paths  = splits[test_subject]["test"]

    if cameras is not None:
        cam_set     = {str(c) for c in cameras}
        train_paths = _filter_by_camera(train_paths, cam_set)
        test_paths  = _filter_by_camera(test_paths,  cam_set)

    train_ds = BadmintonImpactDataset(
        train_paths,
        predict_fz_only,
        augment=augment_train,
        lower_min=lower_min,
        lost_max=lost_max,
        peak_bw_max=peak_bw_max,
        ram_cache=ram_cache,
    )
    test_ds = BadmintonImpactDataset(
        test_paths,
        predict_fz_only,
        augment=False,
        lower_min=lower_min,
        lost_max=lost_max,
        peak_bw_max=peak_bw_max,
        ram_cache=ram_cache,
    )
    return train_ds, test_ds


def _filter_by_camera(paths, cam_set):
    return [p for p in paths
            if (m := re.search(r"_cam(\d+)_", Path(p).name)) and m.group(1) in cam_set]


def collate_fn(batch):
    """Pad variable-length segments to ``max T`` in batch; ``valid_mask`` is 1 on real frames."""
    if not batch:
        return {}
    max_t = max(int(item["pose"].shape[0]) for item in batch)
    B = len(batch)
    f = batch[0]["pose"].shape[1]
    td = batch[0]["target"].shape[1]
    gd = batch[0]["grf_full"].shape[1]

    pose = torch.zeros(B, max_t, f, dtype=batch[0]["pose"].dtype)
    target = torch.zeros(B, max_t, td, dtype=batch[0]["target"].dtype)
    grf_full = torch.zeros(B, max_t, gd, dtype=batch[0]["grf_full"].dtype)
    valid = torch.zeros(B, max_t, dtype=torch.float32)
    ev_idx = torch.zeros(B, dtype=torch.long)

    out: Dict = {}
    for i, item in enumerate(batch):
        t = int(item["pose"].shape[0])
        pose[i, :t] = item["pose"]
        target[i, :t] = item["target"]
        grf_full[i, :t] = item["grf_full"]
        valid[i, :t] = 1.0
        ev_idx[i] = item["ev_idx"]

    out["pose"] = pose
    out["target"] = target
    out["grf_full"] = grf_full
    out["ev_idx"] = ev_idx
    out["valid_mask"] = valid
    for k in ("path", "subject", "stage", "camera", "trial"):
        if k in batch[0]:
            out[k] = [item[k] for item in batch]
    return out


def collate_fn_multiview(batch):
    out = {}
    for k in ("poses", "target", "grf_full", "cam_mask", "ev_idx"):
        if k in batch[0]:
            out[k] = torch.stack([item[k] for item in batch])
    for k in ("subject", "stage", "trial"):
        if k in batch[0]:
            out[k] = [item[k] for item in batch]
    return out


if __name__ == "__main__":
    import sys

    from torch.utils.data import DataLoader

    from baseline.training.dataloader_utils import build_loader_kwargs

    _repo = Path(__file__).resolve().parent.parent
    loso_path = str(_repo / "data" / "reports" / "loso_splits.json")
    if not Path(loso_path).exists():
        sys.exit(0)
    train_ds, test_ds = build_loso_datasets(loso_path, "sub_002",
                                             predict_fz_only=True, augment_train=True)
    print(f"train={len(train_ds)}  test={len(test_ds)}  INPUT_DIM={INPUT_DIM}")
    s = train_ds[0]
    assert not torch.any(torch.isnan(s['pose'])), "NaN in pose!"
    print(f"pose={s['pose'].shape}  target={s['target'].shape}  no NaN ✅")
    _kw = build_loader_kwargs(collate_fn=collate_fn, num_workers=0, pin_memory=False)
    loader = DataLoader(train_ds, batch_size=16, **_kw)
    b = next(iter(loader))
    assert not torch.any(torch.isnan(b['pose'])), "NaN in batch!"
    print(f"batch pose={b['pose'].shape}  no NaN ✅")