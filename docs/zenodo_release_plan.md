## Zenodo Release Plan (Recommended) — BadmintonGRF v1.0

This plan is optimized for **ACM MM Dataset Track** review: reviewer-friendly access, reproducibility, and clear licensing.

### Summary (what we will do)
- **Tier-1 (public, Zenodo DOI, CC BY-NC 4.0)**: processed benchmark data + splits + manifests + documentation (+ optionally C3D if size permits).
- **Tier-2 (by-request)**: raw multi-view videos under an application-based agreement.
- **Code**: GitHub repo under MIT; optionally archive a release snapshot.

### A. Create Zenodo records (best practice)

#### Record 1 (primary DOI): `BadmintonGRF v1.0 — Processed Benchmark`
**License**: CC BY-NC 4.0  
**Must include** (reviewer can run baselines without videos):
- `segments/` (impact-segment `.npz`, ideally zipped per subject)
- `labels/` (trial-level GRF `.npy` + sync `.json`)
- `reports/` (LOSO splits 5p..10p)
- `manifest/` (global manifest + schema; if not yet present, include a `README` describing fields)
- `docs/` (dataset card, ethics/privacy statement, license statement, quickstart)

**Recommended filenames**:
- `BadmintonGRF_v1.0_processed_core.zip` (segments + splits + schema + manifest + docs)
- `BadmintonGRF_v1.0_trial_labels.zip` (grf.npy + sync.json)

#### Record 2 (optional): `BadmintonGRF v1.0 — Mocap C3D (Public)`
Use this only if C3D size is large or you want separate downloads.

**License**: CC BY-NC 4.0  
**Include**:
- `mocap/` C3D files for the **benchmark subset** (recommended: 10 subjects first).

**Recommended filename**:
- `BadmintonGRF_v1.0_mocap_c3d_10subjects.zip`

### B. GitHub ↔ Zenodo (optional but recommended)
If you connect Zenodo to the GitHub repo, Zenodo can mint a DOI for code releases.
This is not required for the Dataset Track, but improves reproducibility signaling.

### C. Tier-2 videos (by request)
Do not upload raw videos to Zenodo.
Instead, publish:
- `docs/video_access_policy.md`
- `docs/video_access_request_form.md`

### D. Metadata to put on Zenodo (copy/paste)
- **Title**: BadmintonGRF v1.0: A Multimodal Dataset and Benchmark for Non-Contact GRF Estimation in Badminton
- **Creators**: match paper author list
- **Description**: 5–8 lines describing modalities, task, benchmark, and access tiers
- **License**: CC BY-NC 4.0
- **Related identifiers**:
  - GitHub repo URL
  - Paper preprint (if any)
- **Keywords**: dataset, biomechanics, badminton, ground reaction force, pose estimation, fatigue

### E. What remains TBD (you must decide)
- Whether to include **C3D** in Tier-1 (Record 1) or split to Record 2.
- Whether to publish **10 subjects only** (recommended for v1.0) or include all 17.

