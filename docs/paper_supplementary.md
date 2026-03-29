# BadmintonGRF — Supplementary Material (ACM MM 2026 Dataset Track)

This document is the **Supplementary Material** referenced by `paper/badmintongrf-mm2026.tex` (in-paper macro: `\emph{Supplementary Material}`). It extends the main PDF with reproducibility details, deferred tables, and schema-level documentation, without repeating the introduction, related work, or closing discussion.

**Hosting.** Publish with the repository (e.g. `docs/paper_supplementary.md`) or export to PDF/HTML. The canonical repository is **`https://github.com/KenyaNiu/BadmintonGRF`**: the main PDF spells this URL explicitly the first time (Impact segments / schema pointer); later mentions use the clickable **project page** shortcut in the TeX source.

**Convention.** Top-level blocks **A–H** are supplementary sections. A phrase like “Sec. 3” means a section in the **main paper** PDF (not this file).

---

### Main-paper traceability map

| Main-paper pointer | Supplementary landing |
|--------------------|------------------------|
| Interpolation / continuous timeline (Sec. 3, Synchronization and GRF) | §A.7 |
| Offset uncertainty, sync QA figures | §A.4, §E.1 |
| Tier 1 / Tier 2, Zenodo, license | §A.1, §E.2 |
| Tier-1 counts (17,425; 12,867; 1,732; 156 trials; ~7% sync-anchor) | §A.3 (aligned with abstract + Table 2/3 in main PDF) |
| Loader gates + feature defaults | §A.5 |
| NPZ schema contract | §A.6, §A.8 |
| **Table 2** `tab:qa_audit` — Panel A (timing stress) & Panel B (loader audit) | §A.3, §A.4, **§H** |
| **Table 4** `tab:benchmark_all` — benchmark metrics | §B.1–§B.10 |
| LOSO protocol (15% val, no TTA, best-val checkpoint) | §B.2, §B.8 |
| Macro means, per-fold std, `paper-export` bundles (also cited from the Table 4 caption / evaluation text on the **project page** release) | §B.6, §B.9, §B.10 |
| LOSO scaling \(N\in\{5,10\}\), camera / fatigue stratification | §B.7 |
| Extended multi-sport comparison (main Table 1 footnote) | §C |
| Ethics / Tier-2 access | §E.2, §E.3 |
| Camera-ready checklist | §G |

**Table numbering in the compiled main paper** (typical `sigconf` float order): **Table 1** = modality comparison (`tab:modality_compare`); **Table 2** = QA audit (`tab:qa_audit`); **Table 3** = scope summary (`tab:stats`); **Table 4** = reference benchmark (`tab:benchmark_all`). This file uses those numbers when referring to the PDF.

---

## A. Dataset (main Sec. 3; Figs. 2–3)

**Reviewer quick-check.**

- **Tier 1:** Processed impact segments (2D pose, time-aligned GRF, sync metadata, LOSO splits). Public record: **`https://doi.org/10.5281/zenodo.19277566`** under **CC BY-NC 4.0** (see main paper, Data Access).
- **Tier 2:** Raw multi-view RGB and Vicon C3D — application-based access only.
- **Benchmark cohort:** 10 subjects (`sub_001`–`sub_010`) for fixed LOSO; **17** instrumented subjects in the collected pool (`sub_001`–`sub_017` in the main text).
- **Scale (main-paper anchors):** **17,425** on-disk impact segments → **12,867** after loader gates → **1,732** unique physical impacts (post-gate multi-view dedup.) → **156** packaged trials.

### A.1 Tiering, cohort, access

| Item | Detail |
|------|--------|
| Tier 1 | Pose + aligned GRF + metadata + splits; no raw RGB/C3D in the public benchmark tree. |
| Tier 2 | Raw RGB + C3D; controlled access; no redistribution. |
| Zenodo | `https://doi.org/10.5281/zenodo.19277566` |
| LOSO subset | Ten subjects with frozen manifests in the release. |

### A.2 Protocol tags and fatigue stages

Non-fatigue capture follows **`rally` → `stage1` → `stage2` → `stage3`** (upper-net footwork, smash-oriented footwork, six-point reaction). After whole-court fatigue induction, the same three drills repeat as **`fatigue_stage1`**, **`fatigue_stage2`**, **`fatigue_stage3`**. Each segment `.npz` carries a **`stage`** string parsed from the parent trial id (metadata in §A.6). The **headline benchmark table in the main paper pools all protocol conditions**; stratify by `stage` or fatigue tags for exploratory analysis.

### A.3 Impact-segment inventory (canonical counts)

The following **must match** the frozen Tier-1 shard archived with the Zenodo release and the statements in the main-paper abstract and Sec. 3.

| Statistic | Value |
|-----------|------:|
| On-disk impact-segment archives (10-subject Tier-1 tree) | **17,425** |
| After benchmark loader quality gates (§A.5), view-specific instances | **12,867** |
| Packaged instrumented trials (Tier-1 tree) | **156** |
| Unique landing impacts `(trial, impact_idx)` after gates (multi-view dedup.) | **1,732** |
| Loader drop rate \((1 - 12{,}867/17{,}425)\) | **~26.1%** |
| `is_sync_impact` rate (pre-gate; sync-event anchor when auto peak search is ill-conditioned) | **~6.9%** (main text ~7%) |

**Notes.**

- `is_sync_impact == true` marks windows anchored at the human-aligned event in `sync.json`; it does **not** mean synchronization failed.
- **Do not** treat protocol-stage imbalance statistics as fixed constants unless you recompute them from the release; if you publish stage histograms, regenerate from the packaged `.npz` trial/stage fields (see `tools/scan_dataset.py` for structural scans over your local data root).

### A.4 Synchronization QA (main Sec. 3; Table 2 Panel B context)

For the released 10-subject Tier-1 packaging:

- **156** instrumented trials with packaged segments.
- **1,247** `(trial, camera)` offset records.
- **65** (**5.2%**) flagged for manual reconciliation; the rest passed automated checks.

Aggregate tables, outlier lists, and figures come from `pipeline/step2_verify_sync.py` (§E.1).

### A.5 Segment construction, loader gates, and features

**Export.** Impact windows are produced by `pipeline/step4_segment.py` (peak finding on contact-positive \(F_z\), \(\pm 0.5\,\mathrm{s}\) symmetric windows, alignment with pose `frame_indices`, `np.savez_compressed`). Authoritative numeric thresholds (detection filter, peak spacing, sync quality allowed values) live in that module—**keep §A.7–A.8 aligned with code** when they change.

**Benchmark loader** (reference: `baseline/impact_dataset.py`). Exclude a segment if **any** holds:

- `stat_lower_body_mean_score` **&lt; 0.70**
- `stat_lost_rate` **&gt; 0.05**
- `peak_force_bw` **&gt; 3.0**

**Model inputs (Tier-1 task):** COCO-17 `keypoints_norm` and `scores` → **119-D** per-frame vector (positions + finite-difference velocity/acceleration + confidences). Frames with **`scores < 0.1`** zero the coordinates before differences; then sanitize/clamp per the released `build_features` implementation.

### A.6 Impact-segment NPZ schema (release format)

One file \(\equiv\) one `(trial, camera, impact)` window. **`schema_version`** identifies the export revision; key **order** is defined by `pipeline/step4_segment.py`.

#### Minimal groups (benchmark contract)

| Keys | Role |
|------|------|
| `keypoints_norm`, `keypoints_px`, `scores` | COCO-17 pose; \((T,17,2)\), \((T,17,2)\), \((T,17)\). |
| `frame_indices`, `track_status` | Absolute frame ids; per-frame tracking state. |
| `grf_at_video_fps`, `grf_normalized` | Video-rate \([F_x,F_y,F_z]\) (N) and BW-normalized targets; \((T,3)\). |
| `timestamps_video`, `timestamps_grf`, `grf_1200hz` | Time axes; high-rate stack \((M,7)\); nominal Hz in `grf_rate`. Name `grf_1200hz` is fixed; rate may be 1000 or 1200 Hz. |
| `offset_sec`, `offset_uncertainty_sec`, `ev_idx`, `grf_peak_sec` | Sync offset (s), optional uncertainty, peak index, absolute peak time. |
| `trial`, `subject`, `stage`, `camera`, `quality` | Protocol / QC metadata. |
| `impact_idx`, `n_impacts_total`, `is_sync_impact` | Impact indexing and sync-anchor flag. |
| `body_weight_N`, `body_weight_kg`, `peak_force_N`, `peak_force_bw` | Scaling and peak summaries. |
| `stat_lower_body_mean_score`, `stat_lost_rate` | Loader QC statistics. |
| `schema_version`, `grf_columns_hf`, `grf_columns_fps` | Versioning and channel-order labels. |

**Normalization.** `keypoints_norm = keypoints_px / [image_width, image_height]` element-wise.

**Interpolation.** Video-rate GRF uses `numpy.interp` onto per-frame GRF times from `sync.json` and `offset_sec` (contact-positive \(F_z\); constant endpoint extension)—§A.7.

### A.7 Video–GRF interpolation (authoritative)

Let `frame_indices` be absolute video indices, `event_frame_abs` the sync event frame, `video_fps` effective fps, `offset_sec` from `sync.json`:

\[
t_{\text{video}}(i)=\texttt{video\_event\_sec}+\frac{f_i-\texttt{event\_frame\_abs}}{\texttt{video\_fps}},\qquad
t_{\text{grf}}(i)=t_{\text{video}}(i)+\texttt{offset\_sec}.
\]

Sample each \(F_x,F_y,F_z\) channel with **`numpy.interp`** at \(\{t_{\text{grf}}(i)\}\) with **constant endpoints** outside the native GRF timestamp span (matches the archived technical note in the TeX source).

**Clocks.** Vicon and plates share a lab timebase; RGB streams are **not** hardware-genlocked—alignment is **`offset_sec`** plus QA in §A.4 / §E.1.

### A.8 Impact detection defaults (reference)

Constants are defined in `pipeline/step4_segment.py`. Typical release defaults (verify in code):

| Role | Value |
|------|--------|
| Window | 0.5 s pre + 0.5 s post |
| Butterworth (detection path) | 50 Hz cutoff (primary pass); rally branch may use a lower cutoff |
| Peak thresholds (fixed/fallback pass) | height \(\ge 0.5\times\) BW, prominence \(\ge 0.2\times\) BW, min distance 250 ms |
| Adaptive / merge knobs | See `step4_segment.py` (`PEAK_*`, `MERGE_GAP_SEC`, etc.) |
| `sync.json` quality | accept `good`, `ok`, or empty string; otherwise skip `(trial, cam)` |

Rally trials may call a rally-specific detector first when `peak_mode=adaptive`; otherwise fall back to the sync-event-anchored window when no valid peak exists (`is_sync_impact`).

#### Full `np.savez` field order (36 keys, `segment_v4`-family)

When bumping the exporter, extend this list and **`schema_version`** together:

| # | Fields |
|---|--------|
| 1–5 | `keypoints_norm`, `keypoints_px`, `scores`, `frame_indices`, `track_status` |
| 6–10 | `grf_at_video_fps`, `grf_normalized`, `grf_1200hz`, `timestamps_video`, `timestamps_grf` |
| 11–20 | `ev_idx`, `trial`, `subject`, `stage`, `camera`, `quality`, `video_fps`, `grf_rate`, `offset_sec`, `offset_uncertainty_sec` |
| 21–26 | `image_width`, `image_height`, `body_weight_N`, `body_weight_kg`, `peak_force_N`, `peak_force_bw` |
| 27–33 | `impact_idx`, `n_impacts_total`, `is_sync_impact`, `grf_peak_sec`, `grf_peak_N_detected`, `stat_lower_body_mean_score`, `stat_lost_rate` |
| 34–36 | `schema_version`, `grf_columns_hf`, `grf_columns_fps` |

---

## B. Benchmark (main Sec. 4–5; **Table 4** `tab:benchmark_all`)

**Reviewer quick-check.**

- **Primary evaluation:** 10-subject LOSO, **15%** validation carve-out inside training subjects, **no test-time augmentation** for the published table.
- **Table 4** reports macro means for **`r2_fz`**, **`rmse_fz`**, **`peak_err_bw`**, **`peak_timing_fr`** (four blocks: LOSO SV, LOSO Fus, Within SV, Within Fus).
- **Fusion:** confidence-weighted late fusion can raise \(r^2\) while hurting RMSE / peak metrics—interpret all four columns.
- **Exports:** `python -m baseline paper-export --run-root <bundle_dir>` produces `paper_bundle.json`, `paper_table_wide.csv`, and companion markdown (see `baseline/README.md`). **Per-fold** tables for the LOSO bundle referenced in the main paper ship with the **project page** release (same frozen `summary.json` / export layout; §B.6–B.10).

### B.1 Tier-1 task

**Input:** COCO-17 2D keypoints + confidences per segment (from Tier-1 files; pose may be recomputed from Tier-2 video offline). **Target:** BW-normalized \(F_z\) at video rate in the **same** impact window. The task assumes **known** impact alignment (not blind impact detection from pose).

### B.2 LOSO protocol and metrics

- **Split:** LOSO on ten subjects; **15%** of gated training-subject segments per fold for validation (`baseline/training/split_utils.py`, deterministic RNG seed **42** for held-out indices).
- **Inference:** one forward pass from the **best validation** checkpoint; **TTA off** for Table 4.
- **Metrics:** macro fold means of \(r^2\), RMSE (\(F_z\), BW), mean abs peak error (BW), mean abs peak timing error (frames)—keys above in exported JSON.

### B.3 Within-trial diagnostic

Held-out impacts **inside trials that also appear in training**; reported in **Table 4** as Within SV / Within Fus. Upper-bound-style diagnostic under reduced subject shift; **not** a second challenge track.

### B.4 Multi-view late fusion

Per-camera models; **reference** fusion averages/weights predictions per physical impact using **pose confidence** weights (not jointly trained for RMSE). Read **R**, **P**, **T**, and \(r^2\) together.

### B.5 Implementations

Ten methods registered in **`baseline/registry.py`**. Entry points: **`python -m baseline`** (`train`, `fuse`, `paper-export`, etc.).

### B.6 Reproducing **Table 4**

After LOSO (and optional fusion) runs with **`--save_report`**, point `paper-export` at the **parent directory** that contains one subfolder per method (see `baseline/tasks/paper_export.py`).

| Block in Table 4 | Typical layout (your run roots will differ) |
|------------------|---------------------------------------------|
| LOSO SV + LOSO Fus | Parent folder whose children include `patch_tst/`, `stgcn_transformer/`, … and `late_fusion/` after `fuse`. |
| Within SV + Within Fus | Separate parent for within-trial runs, same method ids. |

**Method directory → paper row** (must match `baseline/registry.py`):

| `method_id` | Paper label |
|-------------|-------------|
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

**Decimal formatting for LaTeX:** match `paper_table_wide.csv` after the same rounding (main PDF uses three decimals for \(r^2\)/RMSE/peak error and two for timing).

### B.7 Deferred plots / auxiliary experiments

Materials cited from the main paper (LOSO scaling \(N\in\{5,10\}\), camera slices, fatigue stratification, sync QC PDFs):

| Material | Suggestion |
|----------|------------|
| LOSO subset scaling | Train with released split JSONs for \(N\in\{5,10\}\) if bundled; compare `summary.json`. |
| Camera ablations | `python -m baseline` ablation entry points if enabled in your branch (`baseline/tasks/`). |
| Fatigue / stage slices | Stratify by `stage` / fatigue tags from `.npz` before aggregating metrics. |
| Sync QC figures | `python pipeline/step2_verify_sync.py --root "$BADMINTON_DATA_ROOT" --out <verify_dir>` (§E.1). |

### B.8 Training defaults (`baseline/`)

Documented in **`baseline/training/cli.py`** and **`baseline/registry.py`** (epochs, patience, AdamW, contact-weighted loss, **`tta_n=0`** for the paper table, etc.). **Refresh §B.8 in this file whenever defaults change.**

**Artifacts per method run:** `config.json`, `summary.json`, `summary_canonical.json`, `fold_<subject>/best_model.pth`, `fold_<subject>/train_log.csv`, and `late_fusion/summary.json` after fusion.

### B.9 Per-fold JSON

`summary.json` contains **`mean`**, **`std`** (across folds), and **`per_fold`** metrics. Use these for uncertainty and for **§B.10** statistics.

### B.10 Statistical note (Top-2 LOSO-SV; main Table 4 + evaluation text)

The main paper treats the top **LOSO SV** rows as overlapping under **cross-fold variability** (Table 4 caption) and notes that **paired fold-level comparisons** on the archived LOSO bundle do not show a stable separation (main Benchmark / Evaluation text). This section is the **method landing** for those claims: **do not ship hard-coded means, \(p\)-values, or intervals in this Markdown file** unless they match a recomputation from the **frozen** `per_fold` blocks in the same `summary.json` files used to build Table 4 (release with the **project page** / Zenodo bundle).

**Recommended workflow (camera-ready):**

1. Load `per_fold` for PatchTST and ST-GCN+Transformer (LOSO SV block).
2. Report fold-wise **mean ± std** for each metric.
3. Compute **95% paired t-intervals** on fold differences (or bootstrap CIs); check overlap.
4. Optional paired \(t\)-test on fold differences; report \(p\) values **only** if they match that export.
5. Archive a small JSON + README next to the supplementary PDF listing the exact command line, git commit, and paths used.

If you maintain a reviewer-facing boxplot or table, add it under `data/reports/<your_audit_tag>/` and describe it here **only after** files exist in the release.

---

## C. Extended multi-sport comparison (main Table 1 footnote)

Abbreviations: **MV** multi-view video, **MC** marker mocap, **GRF** laboratory force plates, **Ftg** fatigue/protocol tags, **BP** public benchmark recipe.

**Scope.** These rows **extend** the main modality table for positioning. Counts and venues are taken from the **original dataset papers** at preparation time—**verify each against the primary citation** before camera-ready (especially rally / clip / action definitions, which differ by dataset).

| Dataset | Sport | Notes (verify in source) | MV | MC | GRF | Ftg | BP |
|---------|-------|---------------------------|----|----|-----|-----|-----|
| FineGym [CVPR’20] | Gymnastics | Fine-grained gymnastics video benchmark | ✗ | ✗ | ✗ | ✗ | ✓ |
| SoccerNet-v2 [CVPR’21] | Soccer | Broadcast video understanding benchmark | ✗ | ✗ | ✗ | ✗ | ✓ |
| FineDiving [CVPR’22] | Diving | Action quality / procedure labels | ✗ | ✗ | ✗ | ✗ | ✓ |
| ShuttleNet [AAAI’22] | Badminton | Rally / stroke forecasting scale per paper | ✗ | ✗ | ✗ | ✗ | ✓ |
| ShuttleSet [KDD’23] | Badminton | Stroke-level singles tactics | ✗ | ✗ | ✗ | ✗ | ✓ |
| LOGO [CVPR’23] | Artistic swimming | Long-form group actions | ✗ | ✗ | ✗ | ✗ | ✓ |
| FineSports [CVPR’24] | Basketball | Multi-person fine-grained video | ✗ | ✗ | ✗ | ✗ | ✓ |
| P2ANet [TOMM’24] | Table tennis | Dense action detection from broadcast | ✗ | ✗ | ✗ | ✗ | ✓ |
| SportsHHI [CVPR’24] | Ball sports | Human–human interaction detection | ✗ | ✗ | ✗ | ✗ | ✓ |
| F³Set [ICLR’25] | Various | Fast/frequent fine-grained events | ✗ | ✗ | ✗ | ✗ | ✓ |
| FineBadminton [MM’25] | Badminton | Hierarchical semantics / MLLM tasks | ✗ | ✗ | ✗ | ✗ | ✓ |
| **BadmintonGRF (ours)** | **Badminton** | **156 instrumented trials; 17,425 segments → 12,867 gated → 1,732 uniq. impacts; 8 RGB streams when fully instrumented** | ✓ | ✓ | ✓ | ✓ | ✓ |

**BadmintonGRF row semantics.** “Videos” refers to **up to eight fixed RGB streams per instrumented trial**, not a YouTube clip count. **BP** means the LOSO + loader + metric recipe in the main paper.

---

## D. Optional figures (project site)

Maintained scripts in this repository (check `--help` per file):

- `tools/generate_final_teaser.py`
- `tools/generate_long_video.py`
- `tools/generate_cool_grf_slider.py`
- `tools/prepare_ms_demo.py`
- `tools/scan_dataset.py`
- `docs/generate_demo_content.py`, `docs/generate_keyframes.py`

Prefer linking to **`docs/project_overview_acmmm.md`** or the README for volatile command lines.

---

## E. Tier-2 access, sync diagnostics, ethics

### E.1 `pipeline/step2_verify_sync.py`

```bash
python pipeline/step2_verify_sync.py --root "$BADMINTON_DATA_ROOT" --out <verify_dir>
```

**Expected outputs** (when Matplotlib is available):

| Artifact | Role |
|----------|------|
| `figures/fig1_offset_per_subject.pdf` | Offsets by subject |
| `figures/fig2_offset_per_camera.pdf` | Offsets by camera |
| `figures/fig3_consistency_heatmap.pdf` | Subject × camera consistency |
| `figures/fig4_completion_matrix.pdf` | Completion / coverage |
| `figures/fig5_confidence_dist.pdf` | Only if `confidence_label` exists in metadata |
| `verify_sync_summary.csv`, `verify_sync_stats.json`, `outliers.json`, `verify_sync_report.txt` | Tabular audit |

A typical local mirror is `data/verify_output_sync/` (may be **gitignored** when it contains data paths; ship separately or attach as supplementary ZIP).

### E.2 Tier-2 workflow

Policies: **`docs/video_access_policy.md`**, request template **`docs/video_access_request_form.md`**.

### E.3 Ethics

Tiered informed consent and IRB approval as described in the main paper; Tier-1 redistribution under **CC BY-NC 4.0**; Tier-2 under controlled terms.

---

## F. Software environment

**Conda.** Root `environment.yml`, env name **`badminton_grf`**, Python 3.10; PyTorch listed via **pip** with a **lower bound**—**pin** `torch` / CUDA wheels for exact Table-4 reproduction and save `pip freeze` or `conda list --explicit` with the frozen benchmark bundle.

---

## G. Camera-ready checklist

1. Recompute §A.3 counts on the **exact** Zenodo drop; update main paper if they change.
2. Re-run **`paper-export`** on the bundle that backs **Table 4**; diff CSV vs LaTeX.
3. Confirm **Zenodo DOI**, GitHub URL, and license text match the proceedings PDF.
4. Attach sync QC from §E.1 and any §B.7 plots expected by reviewers.
5. Regenerate **§H** machine-readable JSON (if used) from the same build as **Table 2**.
6. Archive `summary.json` / `paper_bundle.json` for **all ten** methods + fusion runs (include per-fold exports if the Table 4 caption points reviewers to the **project page** bundle).

---

## H. Machine-readable audits for **Table 2** (`tab:qa_audit`)

The main paper’s **Table 2** has:

- **Panel A** — synthetic frame-shift stress test (\(n=17{,}425\) released windows; self-consistency of the pose-to-\(F_z\) pipeline).
- **Panel B** — loader gate audit (pre- vs post-gate counts and mean \(|\mathrm{lag}|\) in frames).

Older drafts used placeholder labels `tab:sync_sensitivity` / `tab:filter_audit`; **the submitted paper consolidates these into `tab:qa_audit`.**

**Optional release bundle** (for reviewers who want JSON provenance): mirror the structure documented in **`docs/supp_artifact_inventory.md`**, e.g. `data/reports/paper_minipack/` with `sync_sensitivity/summary.json`, `filter_audit/summary.json`, checksum manifest, and README files **generated from the same experiment commit as Table 2**. If those paths are absent in a public GitHub snapshot (large or private artifacts), ship them via Zenodo or a supplementary ZIP instead—**do not claim files exist on GitHub unless they are actually committed or linked.**

---

*End of supplementary material (aligned with `paper/badmintongrf-mm2026.tex` as of the Dataset Track draft).*
