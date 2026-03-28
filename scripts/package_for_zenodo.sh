#!/bin/bash
set -euo
set -o pipefail

# =============================================================================
# BadmintonGRF v1.0 - Zenodo Package Script
# =============================================================================

PROJECT_ROOT="/home/nky/BadmintonGRF"
OUTPUT_DIR="${PROJECT_ROOT}/zenodo_upload"

echo "============================================================================"
echo "BadmintonGRF v1.0 - Zenodo Package Script"
echo "============================================================================"
echo ""
echo "Usage: bash $0 [data_root_path]"
echo ""
echo "Example: bash $0 /media/nky/Lenovo/data"
echo ""
echo "If no argument provided, will use: /media/nky/Lenovo/data"
echo "============================================================================"
echo ""

# Use command line argument if provided, otherwise use default
DATA_ROOT="${1:-/media/nky/Lenovo/data}"

# Check if data root exists
if [ ! -d "${DATA_ROOT}" ]; then
  echo "Error: Data root directory not found: ${DATA_ROOT}"
  echo "Please provide correct path as argument"
  exit 1
fi

echo "Using data root: ${DATA_ROOT}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/processed_core"
mkdir -p "${OUTPUT_DIR}/processed_core/segments"
mkdir -p "${OUTPUT_DIR}/processed_core/labels"
mkdir -p "${OUTPUT_DIR}/processed_core/reports"
mkdir -p "${OUTPUT_DIR}/processed_core/manifest"
mkdir -p "${OUTPUT_DIR}/processed_core/docs"
mkdir -p "${OUTPUT_DIR}/mocap"
mkdir -p "${OUTPUT_DIR}/code"

echo "[1/6] Packaging processed_core.zip (segments + labels + reports + manifest + docs)..."
echo ""

# Copy segments (sub_001 to sub_010 only)
for i in {001..010}; do
  SUBJECT_DIR="${DATA_ROOT}/sub_${i}"
  if [ -d "${SUBJECT_DIR}/segments" ]; then
    echo "  Copying sub_${i}/segments..."
    cp -r "${SUBJECT_DIR}/segments" "${OUTPUT_DIR}/processed_core/segments/"
  else
    echo "  Warning: sub_${i}/segments not found, skipping..."
  fi
done

# Copy labels (sub_001 to sub_010 only)
for i in {001..010}; do
  SUBJECT_DIR="${DATA_ROOT}/sub_${i}"
  if [ -d "${SUBJECT_DIR}/labels" ]; then
    echo "  Copying sub_${i}/labels..."
    cp -r "${SUBJECT_DIR}/labels" "${OUTPUT_DIR}/processed_core/labels/"
  else
    echo "  Warning: sub_${i}/labels not found, skipping..."
  fi
done

# Copy reports (LOSO splits)
echo "  Copying reports (LOSO splits)..."
if [ -d "${DATA_ROOT}/reports" ]; then
  cp "${DATA_ROOT}/reports/loso_splits_10p.json" "${OUTPUT_DIR}/processed_core/reports/" 2>/dev/null || echo "  Warning: loso_splits_10p.json not found"
  cp "${DATA_ROOT}/reports/loso_splits_5p.json" "${OUTPUT_DIR}/processed_core/reports/" 2>/dev/null || echo "  Warning: loso_splits_5p.json not found"
  cp "${DATA_ROOT}/reports/loso_splits_6p.json" "${OUTPUT_DIR}/processed_core/reports/" 2>/dev/null || echo "  Warning: loso_splits_6p.json not found"
  cp "${DATA_ROOT}/reports/loso_splits_7p.json" "${OUTPUT_DIR}/processed_core/reports/" 2>/dev/null || echo "  Warning: loso_splits_7p.json not found"
  cp "${DATA_ROOT}/reports/loso_splits_8p.json" "${OUTPUT_DIR}/processed_core/reports/" 2>/dev/null || echo "  Warning: loso_splits_8p.json not found"
  cp "${DATA_ROOT}/reports/loso_splits_9p.json" "${OUTPUT_DIR}/processed_core/reports/" 2>/dev/null || echo "  Warning: loso_splits_9p.json not found"
  
  # Copy manifest
  if [ -f "${DATA_ROOT}/reports/badmintongrf_manifest.json" ]; then
    echo "  Copying manifest..."
    cp "${DATA_ROOT}/reports/badmintongrf_manifest.json" "${OUTPUT_DIR}/processed_core/manifest/"
  else
    echo "  Warning: manifest not found, skipping..."
  fi
else
  echo "  Warning: reports directory not found, skipping..."
fi

# Copy docs
echo "  Copying documentation..."
if [ -d "${PROJECT_ROOT}/docs" ]; then
  cp "${PROJECT_ROOT}/docs/dataset_release_checklist.md" "${OUTPUT_DIR}/processed_core/docs/" 2>/dev/null || echo "  Warning: dataset_release_checklist.md not found"
  cp "${PROJECT_ROOT}/docs/paper_supplementary.md" "${OUTPUT_DIR}/processed_core/docs/" 2>/dev/null || echo "  Warning: paper_supplementary.md not found"
  cp "${PROJECT_ROOT}/docs/video_access_policy.md" "${OUTPUT_DIR}/processed_core/docs/" 2>/dev/null || echo "  Warning: video_access_policy.md not found"
  cp "${PROJECT_ROOT}/docs/video_access_request_form.md" "${OUTPUT_DIR}/processed_core/docs/" 2>/dev/null || echo "  Warning: video_access_request_form.md not found"
  cp "${PROJECT_ROOT}/docs/zenodo_release_plan.md" "${OUTPUT_DIR}/processed_core/docs/" 2>/dev/null || echo "  Warning: zenodo_release_plan.md not found"
else
  echo "  Warning: docs directory not found, skipping..."
fi

# Create dataset_card.md and quickstart.md
cat > "${OUTPUT_DIR}/processed_core/docs/dataset_card.md" << 'EOF'
# BadmintonGRF Dataset Card

## Overview
BadmintonGRF is a multi-modal dataset for ground reaction force (GRF) estimation from badminton videos.

## Dataset Statistics
- **Subjects**: 10 (public benchmark)
- **Trials**: 156
- **Impact Segments**: ~17,425 (on-disk), ~12,867 (after quality gates)
- **Cameras**: 8 per trial (1080p, ~120 FPS)
- **Force Plates**: 4 Kistler 6-axis plates (1000/1200 Hz)
- **Mocap**: Vicon system with ~52 markers (240/250 Hz)

## Data Format
- **Segments**: Impact-segment .npz files with pose, GRF, and metadata
- **Labels**: Trial-level GRF .npy and sync .json
- **Splits**: LOSO splits for 5-10 subjects

## License
- **Processed Data**: CC BY-NC 4.0
- **Code**: MIT
- **Raw Videos**: Application-based access

## Citation
```bibtex
@dataset{badmintongrf2026,
  title={BadmintonGRF v1.0: A Multimodal Dataset and Benchmark for Non-Contact GRF Estimation in Badminton},
  author={Cai, Shengze and ...},
  year={2026},
  doi={10.5281/zenodo.XXXXXX}
}
```
EOF

cat > "${OUTPUT_DIR}/processed_core/docs/quickstart.md" << 'EOF'
# BadmintonGRF Quick Start

## Installation
```bash
# Clone repository
git clone https://github.com/KenyaNiu/BadmintonGRF.git
cd BadmintonGRF

# Create conda environment
conda env create -f environment.yml
conda activate badminton_grf
```

## Data Download
Download from Zenodo: [DOI链接]

## Run Baselines
```bash
export BADMINTON_DATA_ROOT=/path/to/downloaded/data
bash run_all_baselines.sh
```

## Documentation
- [Dataset Card](dataset_card.md)
- [Supplementary Material](paper_supplementary.md)
- [Project Overview](project_overview_acmmm.md)
EOF

echo ""
echo "[2/6] Compressing processed_core.zip..."
cd "${OUTPUT_DIR}"
zip -r BadmintonGRF_v1.0_processed_core.zip processed_core
echo "  Created: ${OUTPUT_DIR}/BadmintonGRF_v1.0_processed_core.zip"

echo ""
echo "[3/6] Packaging mocap_c3d_10subjects.zip..."
for i in {001..010}; do
  SUBJECT_DIR="${DATA_ROOT}/sub_${i}"
  if [ -d "${SUBJECT_DIR}/mocap" ]; then
    echo "  Copying sub_${i}/mocap..."
    cp -r "${SUBJECT_DIR}/mocap" "${OUTPUT_DIR}/mocap/"
  else
    echo "  Warning: sub_${i}/mocap not found, skipping..."
  fi
done

cd "${OUTPUT_DIR}"
zip -r BadmintonGRF_v1.0_mocap_c3d_10subjects.zip mocap
echo "  Created: ${OUTPUT_DIR}/BadmintonGRF_v1.0_mocap_c3d_10subjects.zip"

echo ""
echo "[4/6] Packaging code.zip..."
cd "${PROJECT_ROOT}"
git archive --format=zip --output="${OUTPUT_DIR}/BadmintonGRF_v1.0_code.zip" HEAD
echo "  Created: ${OUTPUT_DIR}/BadmintonGRF_v1.0_code.zip"

echo ""
echo "============================================================================"
echo "Packaging Complete!"
echo "============================================================================"
echo ""
echo "Files created in: ${OUTPUT_DIR}"
echo ""
ls -lh "${OUTPUT_DIR}"/*.zip
echo ""
echo "Next steps:"
echo "1. Upload to Zenodo: https://zenodo.org/deposit/new"
echo "2. After getting DOI, update README.md and paper"
echo "============================================================================"
