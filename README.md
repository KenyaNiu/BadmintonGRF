<div align="center">
  <img src="docs/demo_content/teaser.png" width="100%" alt="BadmintonGRF Teaser">
</div>

<div align="center">
  <h1>🏸 BadmintonGRF</h1>
  <h3>A Multimodal Dataset and Benchmark for Markerless Ground Reaction Force Estimation in Badminton</h3>
  
  [![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-blue.svg)](https://acmmm2026.org/)
  [![Zenodo](https://img.shields.io/badge/Dataset-Zenodo-173670.svg)](https://doi.org/10.5281/zenodo.19277566)
  [![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
  [![Data License](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-orange.svg)](LICENSE-DATA-CC-BY-NC-4.0.txt)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

  [**Project Page**](https://KenyaNiu.github.io/BadmintonGRF/) |
  [**Dataset (Tier 1)**](https://doi.org/10.5281/zenodo.19277566) |
  [**Paper**](#-citation) 
</div>

<br>

**BadmintonGRF** is a study-grade, large-scale multimodal dataset designed to advance non-contact Ground Reaction Force (GRF) estimation and biomechanics research in intermittent sports. It pairs high-frame-rate multi-view video with laboratory-grade in-floor force plates and Vicon motion capture, featuring human-verified alignment and rigorous benchmark protocols.

## 🌟 Highlights

- 🎥 **Multi-View High-Speed RGB**: 8 fixed cameras recording at ~120 FPS.
- ⚖️ **Laboratory Ground Truth**: 4 Kistler 6-axis force plates (1000/1200 Hz) and 8-camera Vicon C3D mocap.
- 🏸 **Impact-Centric Badminton Protocol**: Focuses on high-intensity, non-periodic footwork and landings, with fatigue-stage stratification.
- ⏱️ **Software-Level Alignment**: Human-in-the-loop video-GRF alignment with automatic QA, avoiding the need for expensive hardware genlock.
- 📊 **Comprehensive Benchmark**: 10 reproducible baseline models (e.g., PatchTST, ST-GCN, TSMixer) with a rigorous Leave-One-Subject-Out (LOSO) evaluation protocol and optional late fusion.
- 🔒 **Tiered Privacy Release**: CC BY-NC 4.0 for processed pose+GRF (Tier 1), and controlled access for raw RGB/C3D (Tier 2).

## 📢 News
- **[2026-03]** 🏸 BadmintonGRF dataset and benchmark code are officially released!

## 📊 Dataset Overview

We provide 17,425 impact-segment archives collected from elite athletes. After applying strict quality gates, the benchmark features **12,867 valid instances** corresponding to **1,732 unique impacts**. 

| Modality | Description |
|---|---|
| **Video** | 8 views × ~120 FPS (DJI Osmo Action 4) |
| **Pose** | 2D COCO-17 via YOLO26-pose + ByteTrack |
| **GRF** | 6-axis forces from 4 Kistler plates (1000/1200 Hz) |
| **Mocap** | Vicon C3D (~52 markers, 240/250 Hz) |

*For full dataset schema and statistics, please refer to our [Project Page](https://KenyaNiu.github.io/BadmintonGRF/).*

## 🏆 Benchmark Results

We benchmark 10 distinct models under a fixed **Leave-One-Subject-Out (LOSO)** protocol to assess cross-subject generalization. The target is predicting the body-weight (BW) normalized vertical force ($F_z$).

| Model | $r^2$ $\uparrow$ | RMSE (BW) $\downarrow$ | Peak Error (BW) $\downarrow$ | Peak Timing (frames) $\downarrow$ |
|:---|:---:|:---:|:---:|:---:|
| **PatchTST** | **0.403** | **0.510** | 0.226 | 1.07 |
| **ST-GCN+Transformer** | 0.394 | 0.514 | **0.221** | **0.96** |
| **TCN+BiGRU** | 0.390 | 0.514 | 0.348 | 3.79 |
| **TSMixer** | 0.351 | 0.531 | 0.218 | 1.85 |
| **Seq-Transformer** | 0.345 | 0.533 | 0.281 | 1.82 |

*(Results shown for LOSO Single-View. For multi-view late fusion and within-trial diagnostic results, see the main paper.)*

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/KenyaNiu/BadmintonGRF.git
cd BadmintonGRF

# Create and activate conda environment
conda env create -f environment.yml
conda activate badminton_grf
```

### 2. Data Preparation

1. Download the **Tier 1 (Public)** dataset from [Zenodo](https://doi.org/10.5281/zenodo.19277566).
2. Extract the data and set the environment variable:
   ```bash
   export BADMINTON_DATA_ROOT=/path/to/extracted/data
   ```

### 3. Running Baselines

We provide a unified CLI and bash scripts to easily train and evaluate all baselines.

```bash
# Run all baseline models sequentially
bash run_all_baselines.sh

# Or run a specific baseline (e.g., PatchTST)
bash run_all_baselines.sh train patch_tst

# Using the Python CLI directly for advanced options
python -m baseline train --method patch_tst \
    --loso_splits $BADMINTON_DATA_ROOT/reports/loso_splits_10p.json \
    --run_dir runs/patch_tst_loso
```

## 📁 Repository Structure

```text
BadmintonGRF/
├── baseline/              # 10 baseline models, LOSO training loops, and tasks
├── pipeline/              # End-to-end data processing (extract GRF, align, pose, segment)
├── tools/                 # Dataset scanning and validation scripts
├── docs/                  # Project page assets and HTML
├── paper/                 # ACM MM 2026 paper LaTeX source
└── run_all_baselines.sh   # Automated benchmarking script
```

## 📜 License & Access

- **Code**: Released under the [MIT License](LICENSE).
- **Data (Tier 1)**: Processed segments (Pose + Aligned GRF) are released under [CC BY-NC 4.0](LICENSE-DATA-CC-BY-NC-4.0.txt).
- **Data (Tier 2)**: Raw RGB videos and C3D mocap data are available under controlled access for privacy protection. Please check the project page for application details.

## 📖 Citation

If you find our dataset or code useful in your research, please consider citing:

```bibtex
@inproceedings{niu2026badmintongrf,
  title={BadmintonGRF: A Multimodal Dataset and Benchmark for Markerless Ground Reaction Force Estimation in Badminton},
  author={Niu, Kuoye and Li, Jianwei and Cai, Shengze and Ma, Yong and Jia, Mengyao and Shen, Lishun and Zhang, Zhenheng and Peng, Yuxin and Song, Xian},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026}
}
```

## 🤝 Acknowledgments

We extend our deepest gratitude to all the elite athletes from **Wuhan Sports University** who participated in this study, and the experimental staff for their professional support during the data collection process.
