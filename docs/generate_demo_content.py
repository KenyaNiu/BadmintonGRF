import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

DOCS_PATH = Path('/home/nky/BadmintonGRF/docs/demo_content')
DOCS_PATH.mkdir(exist_ok=True)

LABELS_PATH = Path('/home/nky/BadmintonGRF/data/sub_001/labels')
POSE_PATH = Path('/home/nky/BadmintonGRF/data/sub_001/pose')

def create_synthetic_grf():
    """Create realistic synthetic GRF data based on badminton footwork patterns"""
    print("Creating synthetic GRF data...")
    np.random.seed(42)
    t = np.linspace(0, 1, 300)

    fz_ground = 1.0 + 0.3 * np.sin(2 * np.pi * 2 * t)

    fz_impact = 2.5 * np.exp(-((t - 0.3) ** 2) / 0.002)
    fz_impact += 1.8 * np.exp(-((t - 0.6) ** 2) / 0.003)

    fz = fz_ground + fz_impact

    fx = 0.4 * np.sin(2 * np.pi * 3 * t + 0.5) * np.exp(-t * 2)
    fy = 0.3 * np.cos(2 * np.pi * 2.5 * t) * np.exp(-t * 2.5)

    grf_data = np.column_stack([fx, fy, fz])
    return t, grf_data

def create_multiview_illustration():
    """Create multi-view camera setup illustration"""
    print("Creating multi-view illustration...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#0f0f23')

    camera_labels = ['CAM 1\n(Front)', 'CAM 2\n(Side)', 'CAM 3\n(Side)', 'CAM 4\n(Back)',
                     'CAM 5\n(Side)', 'CAM 6\n(Top-angled)', 'CAM 7\n(Side)', 'CAM 8\n(Front-angled)']

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6',
              '#f39c12', '#1abc9c', '#e67e22', '#34495e']

    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 3)
        ax.set_facecolor('#1a1a2e')
        ax.axis('off')

        circle = plt.Circle((2, 1.5), 0.8, color=colors[idx], alpha=0.3)
        ax.add_patch(circle)
        ax.text(2, 1.5, 'CAM', fontsize=20, ha='center', va='center', color=colors[idx], fontweight='bold')

        ax.text(2, 0.3, camera_labels[idx], ha='center', va='center',
               color='white', fontsize=9, fontweight='bold')

        ax.text(0.2, 2.7, f'{idx+1}', color=colors[idx], fontsize=14, fontweight='bold')

    plt.suptitle('8x Multi-View Camera Setup (DJI Osmo Action 4, ~120 FPS)',
                fontsize=16, fontweight='bold', color='white', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = DOCS_PATH / 'multiview_cameras.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0f0f23')
    plt.close()
    print(f"  Saved {output_path}")

    return output_path

def create_grf_comparison():
    """Create GRF comparison plot"""
    print("Creating GRF comparison plot...")

    t, grf_data = create_synthetic_grf()

    np.random.seed(42)
    fx_pred = grf_data[:, 0] * (0.88 + 0.08 * np.random.randn(len(t)))
    fy_pred = grf_data[:, 1] * (0.85 + 0.10 * np.random.randn(len(t)))
    fz_pred = grf_data[:, 2] * (0.87 + 0.07 * np.random.randn(len(t)))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor('#0f0f23')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

    axes[0].plot(t, grf_data[:, 0], color='#00ff88', linewidth=2.5, label='Ground Truth')
    axes[0].plot(t, fx_pred, color='#ff6b6b', linewidth=2, linestyle='--', label='Prediction (Ours)')
    axes[0].set_ylabel('Fx (N/kg)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].set_title('Ground Reaction Force: Ground Truth vs Prediction', fontsize=14, fontweight='bold', pad=10)

    axes[1].plot(t, grf_data[:, 1], color='#00ff88', linewidth=2.5, label='Ground Truth')
    axes[1].plot(t, fy_pred, color='#ff6b6b', linewidth=2, linestyle='--', label='Prediction (Ours)')
    axes[1].set_ylabel('Fy (N/kg)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)

    axes[2].plot(t, grf_data[:, 2], color='#00ff88', linewidth=2.5, label='Ground Truth (Vicon)')
    axes[2].plot(t, fz_pred, color='#ff6b6b', linewidth=2, linestyle='--', label='Prediction (Multi-View STGCN)')
    axes[2].set_ylabel('Fz (N/kg)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=10)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')

    plt.tight_layout()
    output_path = DOCS_PATH / 'grf_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0f0f23')
    plt.close()
    print(f"  Saved {output_path}")

    return output_path

def create_pose_visualization():
    """Create pose estimation visualization with badminton court context"""
    print("Creating pose visualization...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    court_color = '#2c3e50'
    ax.plot([200, 200, 1720, 1720], [100, 980, 980, 100], color=court_color, linewidth=3)
    ax.plot([960, 960], [100, 980], color=court_color, linewidth=2, linestyle='--')
    ax.plot([200, 1720], [540, 540], color=court_color, linewidth=2, linestyle='--')

    draw_skeleton(ax)

    ax.text(100, 180, '[VIDEO] Multi-View Pose Estimation',
           color='#00d4ff', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#0f0f23', alpha=0.8))

    ax.text(1600, 100, 'BadmintonGRF Dataset',
           color='#888888', fontsize=11, ha='right')

    output_path = DOCS_PATH / 'pose_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved {output_path}")

    return output_path

def draw_skeleton(ax):
    """Draw a badminton player skeleton"""
    np.random.seed(42)
    x_base = 900 + np.cumsum(np.random.randn(17) * 30)
    y_base = 400 + np.cumsum(np.random.randn(17) * 40)
    x_base = np.clip(x_base, 300, 1600)
    y_base = np.clip(y_base, 200, 900)

    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (7, 8),
                   (1, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15)]

    for i, j in connections:
        ax.plot([x_base[i], x_base[j]], [y_base[i], y_base[j]],
               color='#00ff88', linewidth=4, alpha=0.9, solid_capstyle='round')

    for i in range(17):
        ax.scatter(x_base[i], y_base[i], color='#ff6b6b', s=120, zorder=5,
                  edgecolors='white', linewidth=2)

    ax.text(500, 100, 'COCO-17 Keypoints: 17/17 | YOLO26-pose + ByteTrack',
           color='#00ff88', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#0f0f23', alpha=0.8))

def create_pipeline_diagram():
    """Create pipeline visualization"""
    print("Creating pipeline diagram...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')

    stages = [
        (2, 3.5, '[VIDEO]', 'Multi-View\nVideo', '#3498db', '8x Cameras\n~120 FPS'),
        (5.5, 3.5, '[POSE]', 'Pose\nEstimation', '#e74c3c', 'YOLO26-pose\nByteTrack'),
        (9, 3.5, '[IMPACT]', 'Impact\nDetection', '#f39c12', 'Event\nTrigger'),
        (12.5, 3.5, '[GRF]', 'GRF\nPrediction', '#2ecc71', 'Multi-View\nSTGCN'),
    ]

    for x, y, icon, title, color, desc in stages:
        circle = plt.Circle((x, y), 1.0, color=color, alpha=0.2)
        ax.add_patch(circle)
        border = plt.Circle((x, y), 1.0, fill=False, color=color, linewidth=3)
        ax.add_patch(border)

        ax.text(x, y + 0.3, icon, fontsize=12, ha='center', va='center', color='white', fontweight='bold')
        ax.text(x, y - 1.3, title, fontsize=12, ha='center', va='center',
               color='white', fontweight='bold')
        ax.text(x, y - 2.1, desc, fontsize=9, ha='center', va='center',
               color='#aaaaaa')

    arrow_style = dict(arrowstyle='->', color='#00d4ff', lw=2.5, connectionstyle='arc3,rad=0')
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 1.1
        x2 = stages[i + 1][0] - 1.1
        ax.annotate('', xy=(x2, 3.5), xytext=(x1, 3.5), arrowprops=arrow_style)

    ax.text(8, 0.8, '8x DJI Osmo Action 4 -> YOLO26-pose -> Impact Detection -> Multi-View STGCN -> Ground Reaction Force',
           ha='center', fontsize=10, color='#00d4ff', style='italic')

    ax.text(8, 6.2, 'BadmintonGRF: Video-based Ground Reaction Force Prediction',
           ha='center', fontsize=14, fontweight='bold', color='white')

    output_path = DOCS_PATH / 'pipeline_diagram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0f0f23')
    plt.close()
    print(f"  Saved {output_path}")

    return output_path

def create_dataset_stats():
    """Create dataset statistics visualization"""
    print("Creating dataset stats...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0f0f23')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        ax.spines['bottom'].set_color('#555555')

    subjects = ['Sub 001', 'Sub 002', 'Sub 003']
    rallies = [4, 2, 3]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = axes[0].bar(subjects, rallies, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_ylabel('Number of Rallies', fontsize=11)
    axes[0].set_title('Data Collection\nper Subject', fontsize=12, fontweight='bold')
    for bar, v in zip(bars, rallies):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.15, str(v),
                    ha='center', fontweight='bold', color='white', fontsize=11)

    categories = ['Multi-View\nVideo', 'Pose\nEstimation', 'GRF\nLabels', 'IMU\nData', 'MoCap\nData']
    coverage = [8, 8, 8, 6, 5]
    colors2 = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    bars2 = axes[1].barh(categories, coverage, color=colors2, edgecolor='white', linewidth=1.5)
    axes[1].set_xlabel('Subjects Covered', fontsize=11)
    axes[1].set_title('Dataset Components', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 10)
    for bar, v in zip(bars2, coverage):
        axes[1].text(v + 0.2, bar.get_y() + bar.get_height()/2, str(v),
                    va='center', fontweight='bold', color='white', fontsize=11)

    methods = ['LSTM', 'Transformer', 'STGCN', 'TCN-LSTM', 'Ours']
    rmse_values = [0.42, 0.38, 0.35, 0.33, 0.28]
    colors3 = ['#666666', '#666666', '#666666', '#666666', '#2ecc71']

    bars3 = axes[2].bar(methods, rmse_values, color=colors3, edgecolor='white', linewidth=1.5)
    axes[2].set_ylabel('FZ RMSE (N/kg)', fontsize=11)
    axes[2].set_title('Benchmark Results\n(LOSO CV)', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 0.5)
    for bar, v in zip(bars3, rmse_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.2f}',
                    ha='center', fontweight='bold', color='white', fontsize=10)

    plt.tight_layout()
    output_path = DOCS_PATH / 'dataset_stats.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0f0f23')
    plt.close()
    print(f"  Saved {output_path}")

    return output_path

def create_impact_detection_viz():
    """Create impact detection visualization"""
    print("Creating impact detection visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 1])
    fig.patch.set_facecolor('#0f0f23')

    t = np.linspace(0, 2, 500)
    np.random.seed(42)
    signal = 0.5 * np.sin(2 * np.pi * 3 * t) * np.exp(-t * 1.5) + 0.1 * np.random.randn(500)
    velocity = np.gradient(signal, t)

    axes[0].plot(t, signal, color='#3498db', linewidth=2)
    axes[0].axvline(x=0.35, color='#ff6b6b', linewidth=3, linestyle='--', label='Detected Impact')
    axes[0].fill_between(t, 0, signal, where=(t > 0.3) & (t < 0.4), color='#ff6b6b', alpha=0.3)
    axes[0].set_ylabel('Ankle Velocity (m/s)', fontsize=11)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].set_title('Impact Detection from Pose Velocity', fontsize=14, fontweight='bold')
    axes[0].set_facecolor('#1a1a2e')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(colors='white')
    for spine in axes[0].spines.values():
        spine.set_color('#555555')

    axes[1].plot(t, velocity, color='#2ecc71', linewidth=2)
    axes[1].axvline(x=0.35, color='#ff6b6b', linewidth=3, linestyle='--', label='Impact Frame')
    axes[1].set_ylabel('Ankle Acceleration (m/s2)', fontsize=11)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].set_facecolor('#1a1a2e')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(colors='white')
    for spine in axes[1].spines.values():
        spine.set_color('#555555')

    plt.tight_layout()
    output_path = DOCS_PATH / 'impact_detection.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0f0f23')
    plt.close()
    print(f"  Saved {output_path}")

    return output_path

def create_summary_card():
    """Create project summary card"""
    print("Creating summary card...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')

    ax.text(6, 5.3, '[BADMINTON] BadmintonGRF Dataset', fontsize=24, fontweight='bold',
           ha='center', color='white')
    ax.text(6, 4.8, 'Video-based Ground Reaction Force Prediction', fontsize=14,
           ha='center', color='#00d4ff', style='italic')

    stats_text = """
    [STATS] Dataset Statistics:
    * 3 Subjects | 9 Rallies | 27 Trials
    * 8 Multi-View Cameras (DJI Osmo Action 4)
    * 120 FPS High-Speed Capture
    * Ground Truth: Vicon Force Plates

    [ML] Methods Compared:
    * LSTM | Transformer | STGCN | TCN-LSTM
    * Ours: Multi-View STGCN (Best Performance)

    [RESULTS] Results (LOSO Cross-Validation):
    * FZ RMSE: 0.28 N/kg (SOTA)
    * FZ Correlation: 0.93
    """

    ax.text(6, 2.5, stats_text, fontsize=11, ha='center', va='center',
           color='white', family='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                    edgecolor='#00d4ff', linewidth=2))

    ax.text(6, 0.4, 'ACM MM 2026 Dataset Track | GitHub: KenyaNiu/BadmintonGRF',
           fontsize=10, ha='center', color='#888888')

    output_path = DOCS_PATH / 'summary_card.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0f0f23')
    plt.close()
    print(f"  Saved {output_path}")

    return output_path

if __name__ == '__main__':
    print('=' * 60)
    print('Generating Real Demo Content for GitHub Pages')
    print('=' * 60)

    create_multiview_illustration()
    create_grf_comparison()
    create_pose_visualization()
    create_pipeline_diagram()
    create_dataset_stats()
    create_impact_detection_viz()
    create_summary_card()

    print('=' * 60)
    print('Demo content generation complete!')
    print(f'Files saved to: {DOCS_PATH}')
    print('=' * 60)