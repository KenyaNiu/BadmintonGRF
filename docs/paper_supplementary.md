# BadmintonGRF — Supplementary Material (ACM MM 2026 Dataset Track)

This file is the **Supplementary Material** referenced in `paper/badmintongrf-mm2026.tex` (the in-paper macro `\emph{Supplementary Material}` points here). It extends the main PDF with reproducibility detail, deferred tables, and schema documentation—without duplicating the narrative introduction, related work, or conclusions.

**Hosting.** Ship this document with the public repository (e.g. `docs/paper_supplementary.md`) or as a PDF/HTML appendix on the project site, per the Dataset Track instructions. Set the canonical project URL once in the TeX source via `\newcommand{\BadmintonGRFGitHubBase}{...}`.

**Convention.** Top-level blocks **A–G** below are *supplementary* sections (dataset through maintenance checklist); references such as “Sec. 3” mean sections in the **main paper**.

---

## A. Dataset (extends main Sec. 3; Figs. 2–3; Tables 1–2)

### A.1 Tiering and cohort

| Item | Main-paper pointer | Detail |
|------|-------------------|--------|
| Tier 1 (public) | Sec. 3, Data Access | Processed impact segments: 2D pose, aligned GRF, sync metadata, LOSO splits. |
| Tier 2 (controlled) | Sec. 3, Data Access | Raw multi-view video and Vicon C3D; application-based access. |
| Benchmark subjects | Table 2; Sec. 3.2 | **10** athletes (`sub_001`–`sub_010`) with packaged LOSO splits. |
| Full collection | Table 2; Sec. 3.2 | **17** athletes instrumented to date (`sub_001`–`sub_017`); additional subjects may be added after QA. |

### A.2 Protocol tags and fatigue stages

Non-fatigue capture follows `rally` → `stage1` → `stage2` → `stage3` (upper-net footwork, smash-oriented footwork, six-point reaction). Fatigue is induced with a full-court protocol; the same three drills are repeated under `fatigue_stage1`, `fatigue_stage2`, `fatigue_stage3`. Each impact-segment `.npz` inherits a `stage` string parsed from the parent trial id (see metadata fields in §A.6). The **primary benchmark table in the main paper pools all protocol conditions**; researchers may stratify by `stage` or fatigue tags for exploratory analysis.

### A.3 Impact-segment inventory (canonical counts)

Counts below match a full scan of **`sub_*/segments/*_impact_*.npz`** on the Tier-1 data root for **`sub_001`–`sub_010`**, using the same discovery rules as `tools/scan_dataset.py`. Regenerate after any packaging change:

```bash
python tools/audit_segment_coverage.py --root "$BADMINTON_DATA_ROOT"   # or --quick for a faster scan
```

| Statistic | Value |
|-----------|------:|
| On-disk impact-segment archives (10-subject Tier-1 tree) | **17,425** |
| After benchmark loader quality gates (§A.5), view-specific instances | **12,867** |
| Distinct trials with ≥1 on-disk segment | **156** |
| Unique landing impacts `(trial, impact_idx)` after gates (multi-view deduplicated) | **1,732** |
| Approximate loader drop rate \((1 - 12867/17425)\) | **~26%** |
| `is_sync_impact` rate (pre-gate audit; sync-event anchor when auto peak search infeasible) | **~6.9%** |
| `stage` label segment counts: max/min ratio (audit) | **~4.1** |
| Gini coefficient (segment counts across the 10 benchmark subjects) | **~0.085** |

**Note.** `is_sync_impact == true` marks windows anchored at the human-aligned event in `sync.json`; it is **not** an indicator of failed video–GRF synchronization.

### A.4 Synchronization QA snapshot (extends Sec. 3.3)

For the 10-subject Tier-1 release:

- **156** instrumented trials with usable packaging.
- **1,247** annotated `(trial, camera)` offset records.
- Automatic verification **flagged 65** offsets (**5.2%**) for manual review; the remainder were accepted as consistent.

Aggregate offset statistics, outlier lists, and per-camera QC plots are produced by the released sync verification tooling under `pipeline/` (e.g. `step2_verify_sync.py` workflow). Bundle those plots in the supplementary ZIP or project site as **video–GRF alignment diagnostics**.

### A.5 Segment construction, loader gates, and features

**Export (pipeline).** Impact-centric windows are built in `pipeline/step4_segment.py`: GRF peak finding on contact-positive vertical force (default adaptive mode with rally-specific branch), ±0.5 s windowing around the peak, alignment with pose `frame_indices`, and metadata write to compressed `.npz`. Authoritative numeric constants (filters, Butterworth cutoff for peak *detection only*, minimum peak spacing, sync quality gates) live in that module—**keep this supplementary section aligned with the code when thresholds change**.

**Benchmark loader (training/evaluation).** Implemented in `baseline/impact_dataset.py`. A segment is **excluded** if any holds:

- `stat_lower_body_mean_score` **< 0.70**
- `stat_lost_rate` **> 0.05**
- `peak_force_bw` **> 3.0**

**Model inputs** (Tier-1 task): COCO-17 `keypoints_norm` and `scores` → 119-D per-frame vector (positions + finite-difference velocity/acceleration + confidences). Low-confidence frames use **`scores < 0.1`** to zero coordinates before finite differences; values are sanitized and clipped per the released `build_features` implementation.

### A.6 Impact-segment NPZ schema (release format)

One file = one `(trial, camera, impact)` window. The **`schema_version`** string in each file identifies the export revision; the **authoritative key list and write order** are defined in `pipeline/step4_segment.py` (`np.savez_compressed`).

#### Minimal groups (benchmark contract)

| Keys | Role |
|------|------|
| `keypoints_norm`, `keypoints_px`, `scores` | COCO-17 pose; shapes \((T,17,2)\), \((T,17,2)\), \((T,17)\). |
| `frame_indices`, `track_status` | Absolute frame ids; tracking state per frame. |
| `grf_at_video_fps`, `grf_normalized` | Video-rate \([F_x,F_y,F_z]\) (N) and BW-normalized targets; \((T,3)\). |
| `timestamps_video`, `timestamps_grf`, `grf_1200hz` | Time axes; high-rate stack \((M,7)\); nominal rate in `grf_rate`. Key name `grf_1200hz` is fixed; actual rate may be 1000 or 1200 Hz per trial. |
| `offset_sec`, `offset_uncertainty_sec`, `ev_idx`, `grf_peak_sec` | Sync offset (s), optional uncertainty, peak index in window, absolute GRF peak time. |
| `trial`, `subject`, `stage`, `camera`, `quality` | Protocol / QC metadata. |
| `impact_idx`, `n_impacts_total`, `is_sync_impact` | Impact indexing and sync-anchor flag. |
| `body_weight_N`, `body_weight_kg`, `peak_force_N`, `peak_force_bw` | Scaling and peak summaries. |
| `stat_lower_body_mean_score`, `stat_lost_rate` | Loader QC statistics. |
| `schema_version`, `grf_columns_hf`, `grf_columns_fps` | Format revision and channel-order documentation. |

**Normalization.** `keypoints_norm = srcpx / [image_width, image_height]` element-wise.

**Interpolation.** Video-rate GRF samples are obtained by linear interpolation (`np.interp`) onto per-frame GRF-axis times derived from `sync.json` and `offset_sec`, as described in the main paper’s archived technical block (contact-positive \(F_z\) for vertical load).

### A.7 Video–GRF interpolation (authoritative; `pipeline/step4_segment.py`)

Let `frame_indices` be the absolute video frame indices in an impact window, `event_frame_abs` the sync event frame from pose metadata, `video_fps` the effective frame rate, and `offset_sec` the per-camera offset from `sync.json`. Define:

\[
t_{\text{video}}(i)=\texttt{video\_event\_sec}+\frac{f_i-\texttt{event\_frame\_abs}}{\texttt{video\_fps}},\qquad
t_{\text{grf}}(i)=t_{\text{video}}(i)+\texttt{offset\_sec}.
\]

For each channel \(Fx\), \(Fy\), \(Fz\) on the original GRF time axis, the implementation uses `numpy.interp` to sample at \(\{t_{\text{grf}}(i)\}\) with **constant endpoint extension** (`left=sig[0]`, `right=sig[-1]`) outside the original timestamp range—matching the main-text contract. The exported arrays `grf_at_video_fps`, `timestamps_video`, and `timestamps_grf` follow this convention.

**Heterogeneous clocks.** Vicon and force plates share a laboratory timebase; RGB streams are independently clocked and aligned only through `offset_sec` (no hardware genlock). Optional `offset_uncertainty_sec` is stored when available (default floor \(\sim 1/\texttt{video\_fps}\) if omitted in JSON).

### A.8 Impact detection constants (reference; default `peak_mode=adaptive`)

Authoritative numeric constants live in `pipeline/step4_segment.py` (bump this supplementary section when those constants change).

| Role | Symbol / mode | Value (release defaults) |
|------|-----------------|---------------------------|
| Fixed window (symmetric) | `WINDOW_PRE_SEC`, `WINDOW_POST_SEC` | 0.5 s each |
| Butterworth low-pass (detection only) | `BUTTER_CUTOFF_HZ` | 50 Hz |
| Strict `find_peaks` (fixed / fallback pass) | height / prominence / min distance | \(0.5\times\)BW, \(0.2\times\)BW, 250 ms |
| Adaptive coarse pass | `PEAK_HEIGHT_COARSE_BW`, `PEAK_PROMINENCE_COARSE_BW`, `PEAK_DISTANCE_COARSE_SEC` | 0.35 / 0.12 / 0.12 s |
| Merge nearby peaks | `MERGE_GAP_SEC` | 0.18 s |
| Rally-specific branch | `RALLY_BUTTER_CUTOFF_HZ`, etc. | 12 Hz cutoff + separate thresholds (see code) |
| `sync.json` quality gate | skip (trial,cam) if quality ∉ | `{"good","ok",""}` |

Rally trials (`rally` in trial id) use `_detect_grf_impacts_rally` first when `peak_mode=adaptive`; otherwise the standard adaptive pipeline runs, with **fallback to the human-aligned sync event** when no valid peak is found (`is_sync_impact` metadata).

#### Full `np.savez` field order (reference)

The following table mirrors the exporter write order for reproducibility audits (36 fields as in the current `segment_v4`-family release; if the exporter adds keys, extend this list when bumping `schema_version`):

| # | Field | Short definition |
|---|--------|------------------|
| 1–5 | `keypoints_norm`, `keypoints_px`, `scores`, `frame_indices`, `track_status` | Pose stream and coverage. |
| 6–10 | `grf_at_video_fps`, `grf_normalized`, `grf_1200hz`, `timestamps_video`, `timestamps_grf` | GRF and time bases. |
| 11–20 | `ev_idx`, `trial`, `subject`, `stage`, `camera`, `quality`, `video_fps`, `grf_rate`, `offset_sec`, `offset_uncertainty_sec` | Event index and metadata. |
| 21–26 | `image_width`, `image_height`, `body_weight_N`, `body_weight_kg`, `peak_force_N`, `peak_force_bw` | Geometry and scaling. |
| 27–33 | `impact_idx`, `n_impacts_total`, `is_sync_impact`, `grf_peak_sec`, `grf_peak_N_detected`, `stat_lower_body_mean_score`, `stat_lost_rate` | Impact bookkeeping and QC. |
| 34–36 | `schema_version`, `grf_columns_hf`, `grf_columns_fps` | Versioning and column labels. |

---

## B. Benchmark (extends main Sec. 4–5; Table 3)

### B.1 Public Tier-1 task

**Input:** per-segment streams of COCO-17 2D keypoints and confidences (optionally derived from Tier-2 video offline). **Target:** BW-normalized vertical GRF \(F_z\) at video rate, strictly time-aligned to the same impact window. The task is **conditional** pose-to-GRF regression given **known** impact alignment (not end-to-end impact detection from pose alone).

### B.2 LOSO protocol and metrics

- **Split:** Leave-one-subject-out on the 10-subject Tier-1 tree; **within training subjects**, **15%** of loader-retained segments are held out for validation (see `baseline/training/split_utils.py`).
- **Reported test inference:** one deterministic forward pass from the **best validation checkpoint**; **test-time augmentation is disabled** for the main table.
- **Metrics** (macro-averaged over LOSO folds): \(r^2\), RMSE of \(F_z\) in BW units, mean absolute peak error (BW), mean absolute peak timing error (video frames). Canonical keys in exported JSON: `r2_fz`, `rmse_fz`, `peak_err_bw`, `peak_timing_fr`.

### B.3 Within-trial diagnostic

Held-out **impacts inside the same trials** as training data, with identical models and metrics. Reported **in the same Table 3** as “Within SV / Within Fus”. This block is an upper-bound-style diagnostic under **weaker subject shift**; it is **not** a second independent benchmark track.

### B.4 Multi-view late fusion

Single-view models are trained per architecture; a **reference** late-fusion stage aggregates predictions across cameras that observe the **same physical impact**, using **confidence-based weights**. Fusion is **not** jointly optimized for RMSE; \(r^2\) can improve while scalar errors worsen—interpret the four reported metrics together.

### B.5 Reference implementations and method identifiers

Ten architectures are registered in `baseline/registry.py`. Literature mapping: `docs/baseline_literature.md`. Train / fuse / aggregate via `python -m baseline …` (see `baseline/README.md`).

### B.6 Reproducing Table 3 (macro means)

The main paper’s **Table 3** concatenates:

| Table block | Role | Frozen export (example; ship equivalents with your release tag) |
|-------------|------|-------------------------------------------------------------------|
| LOSO SV + LOSO Fus | 10-subject LOSO, single-view metrics + late fusion | `runs/benchmark_bundle_20260325/paper_table_wide.csv` |
| Within SV + Within Fus | Within-trial split, same metric columns | `runs/trial_generalization_20260326_201509/within/paper_table_wide.csv` |

**Regenerate** wide tables and `paper_bundle.json` after training:

```bash
python -m baseline paper-export --run-root <path_to_bundle_directory>
```

**Method subfolder id → paper row name**

| `method_id` (directory) | Paper label |
|-------------------------|-------------|
| `patch_tst` | PatchTST |
| `patch_tst_xl` | PatchTST-XL |
| `stgcn_transformer` | ST-GCN+Transformer |
| `gru_bigru` | TCN+BiGRU |
| `tcn_bilstm` | TCN+BiLSTM |
| `tsmixer_grf` | TSMixer |
| `seq_transformer` | Seq-Transformer |
| `tcn_mlp` | TCN+MLP |
| `ms_tcn` | MS-TCN |
| `dlinear` | DLinear |

**Decimal policy for the LaTeX table.** Print \(r^2\), RMSE (**R**), and peak error (**P**) with **three** decimal places; peak timing (**T**) with **two**—matching the exported CSV after the same fixed-width formatting.

**Per-fold dispersion.** Standard deviations and per-fold metrics are stored in each method’s `summary.json` (see §B.9); `paper_bundle.json` aggregates paths for scripting.

### B.7 Deferred quantitative materials (cited from the main paper)

The main text states that **LOSO cohort scaling** (\(N \in \{5,10\}\)), **per-camera ablations**, and **detailed synchronization QA plots** are provided in the supplementary / project materials. Regenerate or attach:

| Material | How to produce / where |
|----------|-------------------------|
| **LOSO cohort scaling** | Use LOSO split bundles for \(N\in\{5,10\}\) (see `tools/regen_loso_from_disk.py` and repository guides); run `python -m baseline train` with `--loso_splits` pointing at the desired JSON; compare aggregated `summary.json` metrics. |
| **Per-camera ablations** | `python -m baseline ablation` (see `baseline/tasks/camera_ablation.py` and `python -m baseline --help`). |
| **Fatigue / protocol stratification** | `python -m baseline fatigue` (optional; see `baseline/tasks/fatigue.py`). |
| **Sync QC figures** | `python pipeline/step2_verify_sync.py --root "$BADMINTON_DATA_ROOT" --out <verify_dir>` (default under `data/verify_output/`). Expected artifacts are listed in §E.1. |

### B.8 Default training hyperparameters and reproducibility (`baseline/`)

**CLI defaults** (`baseline/training/cli.py`, `python -m baseline train`): `epochs=500`, `patience=80` (early stopping on validation), AdamW with `weight_decay=1e-4`, `grad_clip=1.0`, contact-weighted MSE with `loss_alpha=10.0` and `loss_half_win=25`, **no test-time augmentation** for the main table (`tta_n=0`). Optional `--no_augment` disables training-time pose jitter. Best checkpoint = best validation metric per fold (same rule for all methods).

**Per-method defaults** are centralized in `baseline/registry.py` (learning rate, batch size, architecture widths). The trainer applies **sqrt batch–LR scaling** when `batch_size` differs from the registry default and `--lr` is not explicitly passed (`BADMINTON_LR_BATCH_SCALE`, default `sqrt`; set `0` to disable).

| `method_id` | Default `lr` | Default `batch_size` | Notes |
|-------------|-------------|----------------------|--------|
| `tcn_bilstm`, `gru_bigru`, `tcn_mlp`, `ms_tcn` | 1e-3 | 512 | TCN+RNN/MLP/MS-TCN |
| `seq_transformer` | 5e-4 | 256 | |
| `dlinear` | 1e-3 | 512 | `kernel_size=25` |
| `patch_tst` | 5e-4 | 256 | `patch_len=16`, `tf_layers=2` |
| `patch_tst_xl` | 3e-4 | 128 | wider PatchTST; `weight_decay=1e-4` |
| `tsmixer_grf` | 8e-4 | 512 | `hidden_dim=256`, `weight_decay=1e-4` |
| `stgcn_transformer` | 5e-4 | 256 | `weight_decay=1e-4` |

**Train/validation split.** Within each LOSO fold, `baseline/training/split_utils.split_train_val` holds out **15%** of training-subject segments per subject using `numpy.random.default_rng(42)`—so **validation indices are deterministic** given the segment list. Training shuffling uses PyTorch DataLoader randomness; for bit-wise reproducible runs, set `torch.manual_seed` / `torch.cuda.manual_seed_all` and worker seeds in your job script (not forced inside the current trainer).

**Artifacts.** After training with `--save_report`, the method run directory contains `config.json`, `summary.json`, and `summary_canonical.json`. Each fold writes `fold_<test_subject>/` (e.g. `fold_sub_001/`) with `best_model.pth` and `train_log.csv`. Run `python -m baseline paper-export --run-root <bundle_dir>` to emit `paper_table_wide.csv`, `paper_bundle.json`, and companion markdown for the paper table.

**Late fusion (LOSO Fus / Within Fus).** After single-view LOSO runs exist under a common layout, run `python -m baseline fuse --loso_splits … --base_run_dir …` (see `baseline/README.md`); fusion outputs `late_fusion/summary.json` when `--save_report` is used.

### B.9 Per-fold metrics (standard deviations and JSON layout)

The main paper reports **macro means** over LOSO folds. After `python -m baseline train --save_report`, each method’s `summary.json` includes:

- **`mean`**: macro-averaged metrics (e.g. `r2_fz`, `rmse_fz`, `peak_err_bw`, `peak_timing_fr`).
- **`std`**: **standard deviations across folds** for the same keys (computed in `baseline/training/loso_flat.py` from `per_fold`).
- **`per_fold`**: a dict mapping each held-out subject id (e.g. `sub_001`) to that fold’s test metrics.

**On-disk layout (single method, one LOSO run):**

```
<run_dir>/config.json
<run_dir>/summary.json              # aggregated mean/std/per_fold (+ save_report)
<run_dir>/summary_canonical.json    # canonical metric keys
<run_dir>/fold_sub_001/best_model.pth
<run_dir>/fold_sub_001/train_log.csv
…
<run_dir>/late_fusion/summary.json  # after fuse --save_report
```

Use `python -m baseline paper-export --run-root <bundle_dir>` to refresh aggregates; the exported bundle lists paths used for the frozen paper table. Metric keys align with `baseline/tasks/canonical.py`.

When preparing the camera-ready supplementary ZIP, **copy the archived `paper_bundle.json` + `summary.json` files** that exactly match the LaTeX Table~3 build.

---

## C. Extended multi-sport comparison (extends main Table 1 footnote)

Abbreviations: **MV** = multi-view video, **MC** = marker mocap, **GRF** = laboratory in-floor force plates, **Ftg** = fatigue/protocol tags, **BP** = public benchmark recipe.

| Dataset | Sport | Videos | Rallies/Clips | Actions/Evts. | MV | MC | GRF | Ftg | BP |
|--------|-------|--------|---------------|---------------|----|----|-----|-----|-----|
| FineGym (CVPR'20) | Gymnastics | 303 | 4,883 | 32,697 | ✗ | ✗ | ✗ | ✗ | ✓ |
| SoccerNet-v2 (CVPR'21) | Soccer | 500 | — | 110,458 | ✗ | ✗ | ✗ | ✗ | ✓ |
| FineDiving (CVPR'22) | Diving | 135 | — | 3,000 | ✗ | ✗ | ✗ | ✗ | ✓ |
| ShuttleNet (AAAI'22) | Badminton | 75 | 4,325 | 43,191 | ✗ | ✗ | ✗ | ✗ | ✓ |
| ShuttleSet (KDD'23) | Badminton | 44 | 3,685 | 36,492 | ✗ | ✗ | ✗ | ✗ | ✓ |
| ShuttleSet22 (IJCAI'23) | Badminton | 58 | 3,992 | 33,612 | ✗ | ✗ | ✗ | ✗ | ✓ |
| LOGO (CVPR'23) | Artistic swimming | 200 | — | 15,764 | ✗ | ✗ | ✗ | ✗ | ✓ |
| FineSports (CVPR'24) | Basketball | — | 10,000 | 16,000 | ✗ | ✗ | ✗ | ✗ | ✓ |
| P2ANet (TOMM'24) | Table tennis | 200 | 2,721 | 139,075 | ✗ | ✗ | ✗ | ✗ | ✓ |
| SportsHHI (CVPR'24) | Ball sports | 160 | — | 50,649 | ✗ | ✗ | ✗ | ✗ | ✓ |
| F³Set (ICLR'25) | Various | 114 | 11,584 | 42,846 | ✗ | ✗ | ✗ | ✗ | ✓ |
| FineBadminton (MM'25) | Badminton | 120 | 3,215 | 33,325 | ✗ | ✗ | ✗ | ✗ | ✓ |
| **BadmintonGRF (ours)** | **Badminton** | **8-view** | **156 trials (Tier-1)** | **17.4k on-disk; 12.9k loader; 1.73k uniq. impacts** | ✓ | ✓ | ✓ | ✓ | ✓ |

**Semantics for our row.** *Videos* denotes the **layout** (eight fixed RGB streams per instrumented trial), not a clip-count shorthand. *Rallies/Clips* counts Tier-1 trials with at least one exported segment. *Actions/Evts.* summarizes on-disk archives vs loader-retained vs deduplicated impacts (see §A.3). Citations follow `paper/ACM_MM_2026.bib`.

---

## D. Optional figures and tooling (project site / ZIP)

The main PDF uses **three** figures (teaser, pipeline, court schematic). Additional assets for talks or reviewer packs can be produced from repository tools—for example:

- **Teaser / layout renders:** `tools/render_fig1_teaser.py` (requires data root, optional trained checkpoints; see `--help`).
- **Top-\(k\) GRF visualization packs:** `tools/export_grf_topk_pack.py` and related utilities in `tools/`.

Exact command lines and checkpoint-selection logic evolve with the codebase; treat **`tools/*.py --help`** and the repository `README` as the live contract rather than freezing long command paragraphs here.

---

## E. Tier-2 access, sync diagnostics, and ethics (extends main Sec. 3 Data Access)

### E.1 Video–GRF alignment verification bundle (`pipeline/step2_verify_sync.py`)

Running:

```bash
python pipeline/step2_verify_sync.py --root "$BADMINTON_DATA_ROOT" --out <verify_dir>
```

produces (under `<verify_dir>`) at minimum:

| Artifact | Role |
|----------|------|
| `figures/fig1_offset_per_subject.pdf` | Box plot: offset distribution per subject |
| `figures/fig2_offset_per_camera.pdf` | Per-camera offset distribution |
| `figures/fig3_consistency_heatmap.pdf` | Subject × camera consistency |
| `figures/fig4_completion_matrix.pdf` | Annotation progress |
| `figures/fig5_confidence_dist.pdf` | GRF-onset confidence distribution |
| `verify_sync_summary.csv` | Per-trial flat table |
| `verify_sync_stats.json` | Aggregated statistics |
| `outliers.json` | Flagged trials for review |
| `verify_sync_report.txt` | Human-readable report |

Bundle **`figures/*.pdf`** (or PNG exports if you convert) with the supplementary ZIP for reviewers who require offline artifacts. Matplotlib is required for figures.

### E.2 Tier-2 (raw RGB + C3D) access workflow

Raw multi-view video and Vicon C3D are **not** in the public Tier-1 drop. Access is **application-based**, research-only, with no redistribution and no re-identification. Full policy and eligibility are in:

- `docs/video_access_policy.md`
- Request template: `docs/video_access_request_form.md`

The dataset landing page (canonical URL set via `\BadmintonGRFGitHubBase` in the TeX source) should list the same workflow for camera-ready.

### E.3 Ethics and consent (summary)

Participants provided **informed consent** under host-institution review for research capture and tiered release. Tier-1 processed signals (pose, GRF, metadata) are redistributable under the project license; Tier-2 raw streams require the controlled-access terms above.

---

## F. Software environment (reproducibility)

**Conda (reference).** `environment.yml` defines conda env name **`badminton_grf`** with Python 3.10, NumPy/SciPy/Matplotlib, OpenCV, `c3d`, PyMuPDF, etc. **PyTorch** is not pinned in that file; install a CUDA build compatible with your GPU (e.g. from [pytorch.org](https://pytorch.org)) before running `python -m baseline train`.

**Pinned runs.** For paper-grade reproduction, archive:

1. `conda list --explicit` or `pip freeze` from the machine that produced Table~3.
2. Git **commit hash** of this repository.
3. Paths or bundle ids to the frozen `paper_bundle.json` / `paper_table_wide.csv` (§B.6).

---

## G. Maintenance checklist (camera-ready)

1. **Refresh §A.3** counts with `tools/audit_segment_coverage.py` on the exact Tier-1 shard you will freeze on Zenodo.
2. **Re-run `paper-export`** on the archived benchmark directories that back Table 3; verify the LaTeX table matches the CSV under the decimal policy in §B.6.
3. **Update** `paper/badmintongrf-mm2026.tex` `\BadmintonGRFGitHubBase` (and any dataset DOI) so in-paper links resolve.
4. **Attach** sync-QC figures from §E.1 and scaling/ablation outputs from §B.7 if the venue expects binary artifacts rather than links only.
5. **Align §A.7–A.8** with `pipeline/step4_segment.py` after any peak-detection or interpolation change.
6. **Align §B.8** with `baseline/registry.py` and `baseline/training/cli.py` after any default training change.
7. **Archive** `summary.json` / `paper_bundle.json` for all ten methods + fusion that match the published Table~3.

---

*End of supplementary material structure aligned with `badmintongrf-mm2026.tex` (Dataset Track draft).*
