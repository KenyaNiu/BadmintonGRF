# BadmintonGRF

[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-blue.svg)](https://doi.org/10.5281/zenodo.19277566)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/data%20license-CC%20BY--NC%204.0-orange.svg)](LICENSE-DATA-CC-BY-NC-4.0.txt)

> **ACM MM 2026 Dataset Track** - A multimodal dataset and benchmark for markerless ground reaction force estimation in badminton

## Overview

BadmintonGRF is the first multimodal dataset that provides synchronized multi-view high-frame-rate videos, 6-axis ground reaction force (GRF), full-body Vicon motion capture (C3D), and fatigue-stage annotations for badminton biomechanics research. It enables contactless GRF estimation from consumer cameras with human-in-the-loop alignment, addressing the challenge of hardware synchronization between consumer cameras and laboratory equipment.

### Key Features

- **17 elite badminton athletes** (national second-tier and above)
- **8 multi-view cameras** (1080p, ~120 FPS)
- **4 Kistler force plates** (6-axis, 1000/1200 Hz)
- **Vicon motion capture** (8 IR cameras, ~52 markers, 240/250 Hz)
- **Fatigue-stage annotations** for protocol stratification
- **Human-in-the-loop alignment tool** for camera-GRF synchronization
- **17,425 impact-segment samples** (12,867 after quality gates)
- **1,732 unique impacts** (multi-view collapsed)
- **156 trials** with diverse badminton movements
- **LOSO (Leave-One-Subject-Out) evaluation protocol** with fixed splits
- **10 reproducible baseline models** for GRF estimation

## Dataset Access

- **Tier 1 (Public)**: Processed pose, GRF, metadata, and splits available on [Zenodo](https://doi.org/10.5281/zenodo.19277566)
- **Tier 2 (Controlled)**: Raw RGB videos + C3D under application access

## 📖 Citation

```bibtex
@dataset{badmintongrf2026,
  title={BadmintonGRF v1.0: A Multimodal Dataset and Benchmark for Markerless Ground Reaction Force Estimation in Badminton},
  author={Niu, Kuoye and Li, Jianwei and Cai, Shengze and Ma, Yong and Jia, Mengyao and Shen, Lishun and Zhang, Zhenheng and Peng, Yuxin and Song, Xian},
  year={2026},
  doi={10.5281/zenodo.19277566},
  organization={Zenodo}
}
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/KenyaNiu/BadmintonGRF.git
cd BadmintonGRF

# Create conda environment
conda env create -f environment.yml
conda activate badminton_grf
```

### Data Preparation

1. Download the dataset from Zenodo
2. Set the data root:
   ```bash
   export BADMINTON_DATA_ROOT=/path/to/data
   ```

### Run Baselines

```bash
# Run all baselines
bash run_all_baselines.sh

# Or run a single baseline
bash run_all_baselines.sh train tcn_bilstm

# Use the unified CLI
python -m baseline --help
python -m baseline train --method tcn_bilstm --loso_splits $BADMINTON_DATA_ROOT/reports/loso_splits_10p.json --run_dir runs/my_run
```

## Project Structure

```
BadmintonGRF/
├── baseline/              # Baseline models and training code
│   ├── models/           # Model architectures
│   ├── tasks/            # Analysis and fusion tasks
│   ├── training/         # Training loops and utilities
│   └── README.md         # Detailed baseline documentation
├── pipeline/             # Data processing pipeline
│   ├── step0_extract_grf.py
│   ├── step1_align_ui.py
│   ├── step2_verify_sync.py
│   ├── step3_extract_pose.py
│   └── step4_segment.py
├── paper/                # ACM MM 2026 paper
├── tools/                # Utility scripts
├── environment.yml       # Conda environment
└── run_all_baselines.sh  # Baseline runner
```

## License

- **Code**: [MIT License](LICENSE)
- **Data**: [CC BY-NC 4.0](LICENSE-DATA-CC-BY-NC-4.0.txt)

## Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{niu2026badmintongrf,
  title={BadmintonGRF: A Multimodal Dataset and Benchmark for Markerless Ground Reaction Force Estimation in Badminton},
  author={Niu, Kuoye and Li, Jianwei and Cai, Shengze and Ma, Yong and Jia, Mengyao and Shen, Lishun and Zhang, Zhenheng and Peng, Yuxin and Song, Xian},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026}
}
```

## Contact

For questions, please contact:
- Kuoye Niu: kuoyeniu@163.com

## Acknowledgments

We thank all the athletes who participated in this study.
