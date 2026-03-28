## BadmintonGRF Final Release Checklist (ACM MM Dataset Track)

This checklist is designed to ensure the final release is **reviewer-friendly**, **legally safe**, and **reproducible**.

### A) Split the release into two tiers
- **Tier-1 (public, Zenodo/DOI)**: processed benchmark data + documentation + evaluation scripts
- **Tier-2 (application-based)**: raw multi-view videos

### B) Tier-1 (Public) — what to upload

#### B.1 Core data (minimum to run all baselines)
- **Impact segments** as `.npz` (format documented in repository)
  - Contains: normalized 2D keypoints, joint scores, GRF aligned at video FPS, timestamps, event index, subject/trial/stage/camera metadata.
- **Trial-level labels**
  - `*_grf.npy` (raw/high-rate GRF + timestamps + metadata)
  - `*_sync.json` (per-camera offsets and alignment provenance)
- **Mocap**
  - `*_mocap.c3d` (Vicon C3D). If file size is too large for Tier-1, provide a separate DOI shard.
- **LOSO split files**
  - `loso_splits_5p.json ... loso_splits_10p.json`
- **Dataset indices / manifests**
  - A global manifest listing all samples and paths (recommended).
  - A schema document for impact-segment fields in the repository README (recommended).

#### B.2 Reproducibility package (must-have)
- Code repository (this repo)
- `environment.yml`
- `run_all_baselines.sh`（或 `python -m baseline train` / `fuse`，见 `baseline/README.md`）
- A single command that regenerates figures from `runs/` (e.g., `python -m analysis.plot_figures_acmmm`)
- A short “quickstart” section in the dataset website: download → run one experiment → reproduce one figure

#### B.3 Documentation (must-have)
- **Dataset card** (task, modalities, splits, metrics, known limitations)
- **Data access page** (Tier-1 download + Tier-2 application)
- **License page**
  - Processed data: CC BY-NC 4.0 (recommended)
  - Code: MIT (recommended)
  - Raw videos: application-based agreement
- **Ethics/privacy statement**

#### B.4 Integrity and versioning (strongly recommended)
- Checksums: `sha256sum` for each archive
- Version tag: `v1.0` and a changelog
- A stable DOI (Zenodo)

### C) Tier-2 (Videos) — recommended packaging
- Provide **one archive per subject** (or per trial) for manageable access control.
- Provide a mapping from public sample IDs to video files (without exposing video publicly).
- Optionally provide a **small redacted demo video pack** (faces blurred / low resolution) for public preview.

### D) Suggested Zenodo layout (example)
- `BadmintonGRF_v1.0_processed_core.zip` (segments + splits + manifest + docs)
- `BadmintonGRF_v1.0_grf_raw.zip` (trial-level GRF)
- `BadmintonGRF_v1.0_mocap_c3d.zip` (C3D)
- `BadmintonGRF_v1.0_code_snapshot.zip` (optional if you also use GitHub release)

### E) Paper consistency checks (before submission)
- Paper must include a **reviewer-accessible link** (DOI/website)
- Paper must explicitly state:
  - what is public vs by-request
  - the exact license identifiers
  - how to reproduce baselines and figures

