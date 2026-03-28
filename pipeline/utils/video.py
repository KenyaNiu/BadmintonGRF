"""
视频读取工具
============
基于OpenCV的视频读取器，带LRU帧缓存。
"""

import cv2
import numpy as np
from pathlib import Path


class VideoReader:
    """
    视频帧读取器。

    特性:
    - 基于seek的随机访问
    - LRU帧缓存（默认150帧）
    - 自动BGR→RGB转换

    用法:
        with VideoReader("path/to/video.mp4") as v:
            frame = v.get_frame(100)
    """

    def __init__(self, path, cache_size: int = 150):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"视频不存在: {path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"无法打开视频: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.n_frames / self.fps if self.fps > 0 else 0

        self._cache = {}
        self._order = []
        self._cache_max = cache_size

    def get_frame(self, idx: int) -> np.ndarray:
        """获取指定帧（RGB, uint8）"""
        idx = max(0, min(idx, self.n_frames - 1))

        if idx in self._cache:
            self._order.remove(idx)
            self._order.append(idx)
            return self._cache[idx]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if len(self._cache) >= self._cache_max:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[idx] = frame
        self._order.append(idx)
        return frame

    def get_frames_range(self, start: int, end: int) -> np.ndarray:
        """批量获取连续帧，返回 (N, H, W, 3)"""
        frames = [self.get_frame(i) for i in range(start, min(end, self.n_frames))]
        return np.array(frames) if frames else np.empty((0, self.height, self.width, 3), dtype=np.uint8)

    def frame_to_time(self, idx: int) -> float:
        return idx / self.fps

    def time_to_frame(self, t: float) -> int:
        return int(t * self.fps)

    def close(self):
        self.cap.release()
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return (f"VideoReader({self.path.name}, {self.n_frames}frames, "
                f"{self.fps:.2f}fps, {self.width}x{self.height})")
