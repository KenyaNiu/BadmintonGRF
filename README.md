# BadmintonGRF

[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-blue.svg)](https://doi.org/10.5281/zenodo.19277566)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/data%20license-CC%20BY--NC%204.0-orange.svg)](LICENSE-DATA-CC-BY-NC-4.0.txt)

> **ACM MM 2026 Dataset Track** - A multi-modal dataset for ground reaction force estimation from badminton videos

## Overview

BadmintonGRF is the first multi-modal dataset that provides synchronized multi-view high-frame-rate videos, 6-axis ground reaction force (GRF), full-body Vicon motion capture (C3D), and fatigue-stage annotations for badminton biomechanics research. It enables contactless GRF estimation from consumer cameras for the first time.

### Key Features

- **17 elite badminton athletes** (national second-tier and above)
- **8 multi-view cameras** (1080p, ~120 FPS)
- **4 Kistler force plates** (6-axis, 1000/1200 Hz)
- **Vicon motion capture** (8 IR cameras, ~52 markers, 240/250 Hz)
- **3-stage fatigue annotations**
- **Human-in-the-loop alignment tool** for camera-GRF synchronization
- **Impact-segment samples** with standardized schema

## Dataset Access

- **Pose, GRF, C3D, and IMU data**: Publicly available on [Zenodo](https://doi.org/10.5281/zenodo.19277566)
- **Raw video**: Access via request (see [Video Access Policy](docs/video_access_policy.md))

## 📖 Citation

```bibtex
@dataset{badmintongrf2026,
  title={BadmintonGRF v1.0: A Multimodal Dataset and Benchmark for Non-Contact Ground Reaction Force Estimation in Badminton},
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
3. (Optional) Request raw video access using the [form](docs/video_access_request_form.md)

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
├── analysis/             # Analysis and visualization
├── docs/                 # Documentation
├── paper/                # ACM MM 2026 paper
├── tools/                # Utility scripts
├── environment.yml       # Conda environment
└── run_all_baselines.sh  # Baseline runner
```

## Documentation

- [Dataset Release Checklist](docs/dataset_release_checklist.md)
- [Zenodo Release Plan](docs/zenodo_release_plan.md)
- [Video Access Policy](docs/video_access_policy.md)
- [Video Access Request Form](docs/video_access_request_form.md)
- [Paper Supplementary Material](docs/paper_supplementary.md)
- [Project Overview](docs/project_overview_acmmm.md)

## License

- **Code**: [MIT License](LICENSE)
- **Data**: [CC BY-NC 4.0](LICENSE-DATA-CC-BY-NC-4.0.txt)

## Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{cai2026badmintongrf,
  title={BadmintonGRF: A Multi-Modal Dataset for Ground Reaction Force Estimation from Badminton Videos},
  author={Cai, Shengze and ... and Song, Xian},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026}
}
```

## Contact

For questions, please contact:
- Xian Song (corresponding author): sx1993@zju.edu.cn

## Acknowledgments

We thank all the athletes who participated in this study.
