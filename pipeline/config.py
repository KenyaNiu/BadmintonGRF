"""
全局配置
========
所有路径、参数、阈值集中管理。
其他模块统一使用: from pipeline.config import CFG

默认数据根为仓库目录下的 ``data/``（例如 ``/path/to/BadmintonGRF/data``）。
可通过环境变量 BADMINTON_DATA_ROOT 指向外置硬盘等其它位置。

通过 CFG.switch_root(...) 或 use_full() / use_pilot() 切换。
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import os


def _default_root() -> Path:
    """自动推断默认根目录: config.py 所在位置的上一级"""
    return Path(__file__).resolve().parent.parent


# 未设置 BADMINTON_DATA_ROOT 时的本仓库默认数据目录（与 .gitignore 中 data/ 一致）
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


def _default_data_root() -> Path:
    """默认 ``{repo}/data``；可通过 BADMINTON_DATA_ROOT 覆盖。"""
    env = os.environ.get("BADMINTON_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    return DEFAULT_DATA_ROOT


def _has_subjects(path: Path) -> bool:
    """路径下是否存在 sub_XXX 目录"""
    try:
        return path.exists() and any(
            d.is_dir() and d.name.startswith("sub_") for d in path.iterdir()
        )
    except (OSError, PermissionError):
        return False


def _find_subjects_base(root: Path) -> Optional[Path]:
    """
    从根目录推断被试根目录 (subjects_base)。
    支持三种布局: root/sub_XXX, root/data/sub_XXX, root/data/pilot/sub_XXX
    """
    if _has_subjects(root):
        return root
    if _has_subjects(root / "data"):
        return root / "data"
    if _has_subjects(root / "data" / "pilot"):
        return root / "data" / "pilot"
    return None


@dataclass
class Config:
    """项目全局配置"""

    # ── 路径 ──
    project_root: Path = field(default_factory=_default_root)

    # 完整数据集根目录，默认仓库内 data/
    full_data_root: Path = field(default_factory=_default_data_root)

    # 当前使用哪个根目录（None = 使用 project_root）
    _active_root: Optional[Path] = field(default=None, repr=False)

    @property
    def root(self) -> Path:
        """当前激活的数据根目录"""
        return self._active_root if self._active_root is not None else self.project_root

    def switch_root(self, path):
        """切换到指定根目录"""
        self._active_root = Path(path)
        return self

    def use_pilot(self):
        """使用项目根目录（Pilot 开发）"""
        self._active_root = None
        return self

    def use_full(self):
        """使用完整数据集根目录"""
        self._active_root = self.full_data_root
        return self

    # ── 数据路径（自动检测 layout：flat / data / data/pilot）──
    @property
    def data_dir(self) -> Path:
        base = _find_subjects_base(self.root)
        if base is not None:
            return base
        return self.root / "data" / "pilot"

    @property
    def labels_dir(self) -> Path:
        """单一 labels 目录（旧布局 data_dir/labels）；当前实际为每 subject 下 labels/。"""
        return self.data_dir / "labels"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments" / "pilot"

    @property
    def annotations_dir(self) -> Path:
        return self.experiments_dir / "annotations"

    @property
    def alignment_dir(self) -> Path:
        return self.experiments_dir / "alignment"

    @property
    def reports_dir(self) -> Path:
        return self.experiments_dir / "reports"

    @property
    def segments_dir(self) -> Path:
        return self.root / "segments"

    # ── GRF 参数 ──
    grf_sample_rate: int = 1000          # Hz
    grf_force_threshold: float = 20.0    # N，接触判定阈值
    peak_min_prominence: float = 400.0   # N
    peak_min_height: float = 200.0       # N
    peak_min_distance: float = 1.0       # 秒

    # ── 视频参数 ──
    num_cameras: int = 8
    default_cam: int = 2                  # 对齐默认 cam2

    # ── 对齐参数 ──
    align_slope_prior: float = 1.0
    align_slope_tol: float = 0.005        # |slope-1| 容许范围
    align_snap_threshold: float = 0.5     # 秒，GRF候选吸附距离

    # ── 样本切割参数 ──
    segment_pre_contact: float = 0.5      # 秒，IC 前
    segment_post_contact: float = 1.0     # 秒，IC 后

    @property
    def segment_duration(self) -> float:
        return self.segment_pre_contact + self.segment_post_contact

    # ── 质量阈值 ──
    quality_rmse_excellent: float = 0.05
    quality_rmse_good: float = 0.10
    quality_rmse_fair: float = 0.15

    # ── 工具方法：路径构建 ──

    def subject_dir(self, subject: str) -> Path:
        return self.data_dir / subject

    def video_dir(self, subject: str, cam: int) -> Path:
        return self.subject_dir(subject) / "video" / f"cam{cam}"

    def video_path(self, trial: str, cam: int) -> Path:
        """trial名 → 视频路径: data/pilot/{subject}/video/cam{N}/{trial}_cam{N}.mp4"""
        subject = self.trial_to_subject(trial)
        return self.video_dir(subject, cam) / f"{trial}_cam{cam}.mp4"

    def grf_path(self, trial: str) -> Path:
        """GRF 文件路径：按被试分目录，即 {data_dir}/{subject}/labels/{trial}_grf.npy"""
        subject = self.trial_to_subject(trial)
        return self.subject_dir(subject) / "labels" / f"{trial}_grf.npy"

    def mocap_path(self, trial: str) -> Path:
        subject = self.trial_to_subject(trial)
        return self.subject_dir(subject) / "mocap" / f"{trial}_mocap.c3d"

    def detection_path(self, trial: str) -> Path:
        return self.annotations_dir / f"{trial}_detection.json"

    def annotated_path(self, trial: str) -> Path:
        return self.annotations_dir / f"{trial}_annotated.json"

    def alignment_path(self, trial: str) -> Path:
        return self.alignment_dir / f"{trial}_alignment.json"

    def verification_path(self, trial: str) -> Path:
        return self.alignment_dir / f"{trial}_verification.json"

    def segment_path(self, trial: str, seg_id: int) -> Path:
        # 与 step1_align_ui 中对 step4 输出的文档保持一致:
        # {trial}_impact_{idx:03d}.npz
        return self.segments_dir / f"{trial}_impact_{seg_id:03d}.npz"

    # ── 工具方法：发现数据 ──

    @staticmethod
    def trial_to_subject(trial: str) -> str:
        """sub_003_stage1_01 → sub_003"""
        parts = trial.split("_")
        return f"{parts[0]}_{parts[1]}"

    def discover_subjects(self) -> List[str]:
        """从 data_dir 下发现所有 sub_XXX 目录（与每 subject 下 labels/ 布局一致）"""
        if not self.data_dir.exists():
            return []
        return sorted(
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith("sub_")
        )

    def discover_trials(self, subject: str = None) -> List[str]:
        """从各 subject 的 labels/ 目录发现 trial（可按被试过滤）"""
        subjects = self.discover_subjects()
        trials = []
        for s in subjects:
            if subject and s != subject:
                continue
            labels_dir = self.data_dir / s / "labels"
            if not labels_dir.exists():
                continue
            for p in sorted(labels_dir.glob("*_grf.npy")):
                trials.append(p.stem.replace("_grf", ""))
        return sorted(trials)

    def list_cameras(self, trial: str) -> List[int]:
        """列出某个 trial 实际存在的摄像机编号"""
        subject = self.trial_to_subject(trial)
        video_base = self.subject_dir(subject) / "video"
        cams = []
        if not video_base.exists():
            return cams
        for cam_dir in sorted(video_base.glob("cam*")):
            try:
                cam_id = int(cam_dir.name.replace("cam", ""))
            except ValueError:
                continue
            if (cam_dir / f"{trial}_cam{cam_id}.mp4").exists():
                cams.append(cam_id)
        return cams

    def ensure_dirs(self):
        """创建所有必要的输出目录"""
        for d in [self.annotations_dir, self.alignment_dir,
                  self.reports_dir, self.segments_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── 工具方法：质量判断 ──

    def quality_grade(self, rmse: float) -> str:
        if rmse < self.quality_rmse_excellent:
            return "Excellent"
        if rmse < self.quality_rmse_good:
            return "Good"
        if rmse < self.quality_rmse_fair:
            return "Fair"
        return "Poor"

    # ── 诊断 ──

    def validate(self, verbose: bool = True) -> bool:
        """检查关键路径是否存在，打印诊断信息"""
        if verbose:
            print(f"Active root   : {self.root}")
            print(f"Project root  : {self.project_root}")
            print(f"Full data root: {self.full_data_root}")
        ok = True
        if verbose:
            print(f"  {'✓' if self.data_dir.exists() else '✗'} subjects_base (data_dir): {self.data_dir}")
        if not self.data_dir.exists():
            ok = False
        # labels: 支持单一 data_dir/labels 或每 subject 下 labels/
        has_labels = self.labels_dir.exists() or any(
            (self.data_dir / s / "labels").exists() for s in self.discover_subjects()
        )
        if verbose:
            print(f"  {'✓' if has_labels else '✗'} labels (flat or per-subject)")
        if not has_labels:
            ok = False
        if verbose and self.annotations_dir != self.data_dir:
            print(f"  {'✓' if self.annotations_dir.exists() else '✗'} annotations: {self.annotations_dir}")
        subjects = self.discover_subjects()
        trials   = self.discover_trials()
        if verbose:
            print(f"  Subjects found : {len(subjects)} — {subjects}")
            print(f"  Trials found   : {len(trials)}")
            if trials:
                print(f"    e.g. {trials[0]} ... {trials[-1]}")
        return ok


# ── 全局单例 ──
CFG = Config()