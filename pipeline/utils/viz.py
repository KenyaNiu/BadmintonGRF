"""
可视化工具函数
==============
GRF曲线绘制、对齐结果可视化等复用逻辑。
"""

import numpy as np


def plot_grf_with_candidates(ax, timestamps, fz, candidates,
                              correspondences=None, title=None):
    """
    在axes上绘制GRF曲线 + 候选动作标记 + 已配对点。

    参数:
        ax: matplotlib axes
        timestamps: 时间序列 (N,)
        fz: 垂直力，正值表示向下 (N,)
        candidates: detection.json中的候选列表
        correspondences: 已标记的对应点列表（可选）
        title: 图标题
    """
    # 降采样（1000Hz太密集）
    step = max(1, len(timestamps) // 5000)
    ax.plot(timestamps[::step], fz[::step],
            color='steelblue', lw=0.5, alpha=0.8)

    # 候选（红色三角）
    for c in candidates:
        ax.axvline(c['time'], color='red', alpha=0.15, lw=0.8)
    cand_t = [c['time'] for c in candidates]
    cand_f = [c['peak_force'] for c in candidates]
    ax.plot(cand_t, cand_f, 'rv', ms=5, alpha=0.6,
            label=f'Candidates ({len(candidates)})')

    # 对应点（绿色）
    if correspondences:
        for i, cp in enumerate(correspondences):
            ax.axvline(cp['grf_time'], color='green', lw=1.5, alpha=0.7)
            force = cp.get('grf_force', 100)
            ax.plot(cp['grf_time'], force, 'go', ms=8, zorder=5)
            ax.annotate(f"P{i+1}", (cp['grf_time'], force),
                        xytext=(5, 12), textcoords='offset points',
                        fontsize=8, color='green', fontweight='bold')

    ax.set_xlabel("GRF Time (s)")
    ax.set_ylabel("|Fz| (N)")
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.2)
    if title:
        ax.set_title(title, fontsize=9)
