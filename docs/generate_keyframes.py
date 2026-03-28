#!/usr/bin/env python3
"""
BadmintonGRF Demo - 关键帧生成脚本
生成动画视频所需的关键帧图像
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'

DEMO_DIR = Path(__file__).parent
OUT_DIR = DEMO_DIR / "keyframes"
OUT_DIR.mkdir(exist_ok=True, parents=True)

COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12],
    [11, 12], [5, 11], [6, 12], [5, 6],
    [5, 7], [6, 8], [7, 9], [8, 10],
    [0, 1], [0, 2], [1, 3], [2, 4],
]

COCO_KEYPOINTS = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'shoulder_l', 'shoulder_r', 'elbow_l', 'elbow_r',
    'wrist_l', 'wrist_r', 'hip_l', 'hip_r',
    'knee_l', 'knee_r', 'ankle_l', 'ankle_r'
]

def draw_skeleton(ax, keypoints, scores, color='red', alpha=0.8):
    keypoints = np.array(keypoints)
    scores = np.array(scores)

    for i, (p1, p2) in enumerate(COCO_SKELETON):
        if scores[p1] > 0.3 and scores[p2] > 0.3:
            ax.plot([keypoints[p1, 0], keypoints[p2, 0]],
                    [keypoints[p1, 1], keypoints[p2, 1]],
                    color=color, linewidth=3, alpha=alpha, solid_capstyle='round')

    for j, (x, y) in enumerate(keypoints):
        if scores[j] > 0.3:
            circle = Circle((x, y), radius=4, color=color, alpha=alpha, zorder=5)
            ax.add_patch(circle)

def generate_frame_01_setup():
    """Frame 1: 数据采集现场示意"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    ax.text(5, 6.5, 'BadmintonGRF Data Collection', fontsize=20, ha='center', fontweight='bold', color='#2c3e50')
    ax.text(5, 6.0, 'Multi-modal Sensing Setup', fontsize=14, ha='center', color='#7f8c8d')

    court = FancyBboxPatch((1, 2), 8, 4, boxstyle="round,pad=0.05",
                            linewidth=3, edgecolor='#27ae60', facecolor='#d5f5e3', alpha=0.7)
    ax.add_patch(court)
    ax.text(5, 4, 'Badminton Court', fontsize=12, ha='center', color='#27ae60', fontweight='bold')

    for i, (x, y) in enumerate([(2.5, 3.5), (7.5, 3.5), (2.5, 4.5), (7.5, 4.5)]):
        plate = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                               boxstyle="round,pad=0.02",
                               linewidth=2, edgecolor='#e74c3c', facecolor='#fadbd8', alpha=0.9)
        ax.add_patch(plate)
    ax.text(5, 2.5, '4x Kistler Force Plates', fontsize=10, ha='center', color='#c0392b')

    camera_positions = [(0.5, 5), (9.5, 5), (0.5, 3), (9.5, 3), (0.5, 1), (9.5, 1), (4, 0.3), (6, 0.3)]
    for i, (x, y) in enumerate(camera_positions):
        camera = plt.Circle((x, y), 0.25, color='#3498db', alpha=0.9, zorder=5)
        ax.add_patch(camera)
        ax.annotate(f'CAM{i+1}', (x, y+0.4), fontsize=7, ha='center', color='#2980b9', fontweight='bold')

    ax.text(5, 1.2, '8x DJI Osmo Action 4 (~120 FPS)', fontsize=10, ha='center', color='#2980b9')

    plt.tight_layout()
    path = OUT_DIR / "frame_01_setup.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Generated: {path}")
    return path

def generate_frame_02_pose():
    """Frame 2: 姿态估计"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1, ax2 = axes
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    fig.patch.set_facecolor('#1a1a2e')

    np.random.seed(42)

    keypoints_running = np.array([
        [5.0, 8.5], [4.7, 7.8], [5.3, 7.8], [4.5, 7.2], [5.5, 7.2],
        [4.2, 5.5], [5.8, 5.5], [3.8, 4.0], [6.2, 4.0],
        [3.5, 2.8], [6.5, 2.8], [4.5, 3.5], [5.5, 3.5],
        [4.0, 2.0], [6.0, 2.0], [3.8, 1.0], [6.2, 1.0]
    ]) + np.random.randn(17, 2) * 0.15
    scores_running = np.clip(0.7 + np.random.rand(17) * 0.3, 0, 1)

    draw_skeleton(ax1, keypoints_running, scores_running, color='#00d4ff', alpha=0.9)
    ax1.text(5, 9.2, 'YOLO26-pose + ByteTrack', fontsize=14, ha='center',
             color='#00d4ff', fontweight='bold')
    ax1.text(5, 0.3, 'COCO-17 Keypoints', fontsize=10, ha='center', color='#7f8c8d')

    keypoints_landing = np.array([
        [5.0, 9.0], [4.6, 8.4], [5.4, 8.4], [4.4, 7.8], [5.6, 7.8],
        [3.8, 5.8], [6.2, 5.8], [3.2, 4.2], [6.8, 4.2],
        [2.8, 3.0], [7.2, 3.0], [4.0, 4.0], [6.0, 4.0],
        [4.2, 2.5], [5.8, 2.5], [4.0, 1.2], [6.0, 1.2]
    ])
    scores_landing = np.clip(0.8 + np.random.rand(17) * 0.2, 0, 1)

    draw_skeleton(ax2, keypoints_landing, scores_landing, color='#ff6b6b', alpha=0.9)
    ax2.text(5, 9.2, 'Impact Frame Detection', fontsize=14, ha='center',
             color='#ff6b6b', fontweight='bold')
    ax2.text(5, 0.3, 'Landing Impact', fontsize=10, ha='center', color='#7f8c8d')

    plt.tight_layout()
    path = OUT_DIR / "frame_02_pose.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Generated: {path}")
    return path

def generate_frame_03_grf_prediction():
    """Frame 3: GRF预测对比"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.2, 1])

    fig.patch.set_facecolor('#0f0f23')
    for ax in axes:
        ax.set_facecolor('#0f0f23')
        ax.spines['bottom'].set_color('#4a4a6a')
        ax.spines['left'].set_color('#4a4a6a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#aaaaaa')
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')

    t = np.linspace(0, 1, 500)
    impact_center = 0.5
    impact_idx = (np.abs(t - impact_center)).argmin()

    fz_true = np.zeros_like(t)
    fz_true[:impact_idx] = 100 * np.exp(-((t[:impact_idx] - 0.3) ** 2) / 0.02)
    fz_true[impact_idx:] = 800 * np.exp(-((t[impact_idx:] - impact_center) ** 2) / 0.005)
    fz_true = np.clip(fz_true, 0, 1200)

    noise = np.random.randn(len(t)) * 25
    fz_pred = fz_true + noise
    fz_pred = np.clip(fz_pred, 0, 1500)

    ax = axes[0]
    ax.fill_between(t, 0, fz_true, alpha=0.3, color='#00ff88', label='Ground Truth GRF')
    ax.plot(t, fz_true, color='#00ff88', linewidth=2)
    ax.fill_between(t, 0, fz_pred, alpha=0.3, color='#ff6b6b', label='Predicted GRF')
    ax.plot(t, fz_pred, color='#ff6b6b', linewidth=2, linestyle='--')

    ax.axvline(impact_center, color='#ffffff', linewidth=1.5, linestyle=':', alpha=0.7)
    ax.annotate('Impact Peak', xy=(impact_center, 1100), fontsize=10, color='#ffffff',
                ha='center', fontweight='bold')

    ax.set_ylabel('Vertical Force Fz (N)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1400)
    ax.set_title('Ground Reaction Force: Prediction vs Ground Truth', fontsize=14,
                 color='#ffffff', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.1, color='#4a4a6a')

    ax = axes[1]
    error = fz_pred - fz_true
    colors = ['#00ff88' if e < 50 else '#ff6b6b' for e in error]
    ax.bar(t[::5], error[::5], width=0.008, color=colors[::5], alpha=0.7)
    ax.axhline(0, color='#ffffff', linewidth=1)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Prediction Error (N)', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(-150, 150)
    ax.set_title('Prediction Error', fontsize=12, color='#ffffff', pad=10)
    ax.grid(True, alpha=0.1, color='#4a4a6a')

    plt.tight_layout()
    path = OUT_DIR / "frame_03_grf_prediction.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Generated: {path}")
    return path

def generate_frame_04_metrics():
    """Frame 4: 性能指标展示"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    metrics = [
        ('R² Score', '0.847', '#00d4ff'),
        ('RMSE', '0.193 BW', '#00ff88'),
        ('Peak Timing Error', '1.03 fr', '#ff6b6b'),
        ('Peak Force Error', '0.534 BW', '#ffd93d'),
    ]

    ax.text(5, 9.0, 'Benchmark Results', fontsize=22, ha='center',
            fontweight='bold', color='#ffffff')

    for i, (name, value, color) in enumerate(metrics):
        y_pos = 7.0 - i * 1.5

        bg = FancyBboxPatch((1.5, y_pos - 0.5), 7, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor='#2d2d44', alpha=0.8, edgecolor=color, linewidth=2)
        ax.add_patch(bg)

        ax.text(2.5, y_pos + 0.1, name, fontsize=14, ha='left',
                color='#aaaaaa', fontweight='bold')
        ax.text(7.5, y_pos + 0.1, value, fontsize=18, ha='right',
                color=color, fontweight='bold')

    ax.text(5, 1.0, '10 Subjects | LOSO Cross-Validation', fontsize=11,
            ha='center', color='#7f8c8d')

    plt.tight_layout()
    path = OUT_DIR / "frame_04_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Generated: {path}")
    return path

def generate_frame_05_pipeline():
    """Frame 5: 完整流程展示"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')

    stages = [
        ('Multi-View\nVideo', '#3498db', 'CAM1-8'),
        ('Pose\nEstimation', '#00d4ff', 'YOLO26'),
        ('Impact\nSegmentation', '#ffd93d', '±0.5s'),
        ('GRF\nPrediction', '#00ff88', 'TCN+BiLSTM'),
        ('Benchmark\nOutput', '#ff6b6b', 'LOSO'),
    ]

    for i, (name, color, sub) in enumerate(stages):
        x = 1.5 + i * 2.6
        y = 2.5

        box = FancyBboxPatch((x - 0.9, y - 0.8), 1.8, 1.6,
                              boxstyle="round,pad=0.1",
                              facecolor='#2d2d44', edgecolor=color, linewidth=3, alpha=0.9)
        ax.add_patch(box)

        ax.text(x, y + 0.1, name, fontsize=11, ha='center', va='center',
                color='#ffffff', fontweight='bold')
        ax.text(x, y - 0.5, sub, fontsize=9, ha='center', va='center',
                color=color, fontweight='bold')

        if i < len(stages) - 1:
            ax.annotate('', xy=(x + 1.1, y), xytext=(x + 0.9, y),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))

    ax.text(7, 4.5, 'BadmintonGRF Pipeline', fontsize=18, ha='center',
            fontweight='bold', color='#ffffff')

    ax.text(7, 0.8, '17,425 segments  →  12,867 after QA  →  1,732 unique impacts',
            fontsize=11, ha='center', color='#7f8c8d')

    plt.tight_layout()
    path = OUT_DIR / "frame_05_pipeline.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Generated: {path}")
    return path

def main():
    print("Generating keyframes for BadmintonGRF demo...")
    print("=" * 50)

    frames = []
    frames.append(generate_frame_01_setup())
    frames.append(generate_frame_02_pose())
    frames.append(generate_frame_03_grf_prediction())
    frames.append(generate_frame_04_metrics())
    frames.append(generate_frame_05_pipeline())

    print("=" * 50)
    print(f"All frames saved to: {OUT_DIR}")
    for i, f in enumerate(frames):
        print(f"  {i+1}. {f.name}")

    print("\nNext step: Run FFmpeg to create video:")
    print(f"  cd {DEMO_DIR}")
    print("  ffmpeg -framerate 1/3 -i keyframes/frame_%02d.png -c:v libx264 -r 30 -pix_fmt yuv420p demo.mp4")

if __name__ == "__main__":
    main()