# BadmintonGRF Supplementary Artifact Inventory (Camera-Ready)

This checklist maps supplementary artifacts to main-paper claims and current repository paths.

## A. Sync QA bundle (main text: sync QA plots / alignment diagnostics)

| Status | Artifact path | Main-paper / supplementary link | Notes |
|---|---|---|---|
| ✅ | `data/verify_output_sync/figures/fig1_offset_per_subject.pdf` | Supplementary §E.1 | Generated |
| ✅ | `data/verify_output_sync/figures/fig2_offset_per_camera.pdf` | Supplementary §E.1 | Generated |
| ✅ | `data/verify_output_sync/figures/fig3_consistency_heatmap.pdf` | Supplementary §E.1 | Generated |
| ✅ | `data/verify_output_sync/figures/fig4_completion_matrix.pdf` | Supplementary §E.1 | Generated |
| ✅ | `data/verify_output_sync/figures/fig3_sync_quality.pdf` | Supplementary §E.1 (paper-ready QC figure) | Generated |
| ⚠️ (conditional) | `data/verify_output_sync/figures/fig5_confidence_dist.pdf` | Supplementary §E.1 | Not generated in current release because `confidence_label` is absent in current sync metadata |
| ✅ | `data/verify_output_sync/verify_sync_summary.csv` | Supplementary §E.1 | Generated |
| ✅ | `data/verify_output_sync/verify_sync_stats.json` | Supplementary §E.1 | Generated |
| ✅ | `data/verify_output_sync/outliers.json` | Supplementary §E.1 | Generated |
| ✅ | `data/verify_output_sync/verify_sync_report.txt` | Supplementary §E.1 | Generated |

## B. Main-paper mini-tables audit bundle (optional JSON for **Table 2** `tab:qa_audit` panels; legacy names `tab:sync_sensitivity`, `tab:filter_audit`)

| Status | Artifact path | Main-paper / supplementary link | Notes |
|---|---|---|---|
| ✅ | `data/reports/paper_minipack/sync_sensitivity/summary.json` | Supplementary §H (`tab:sync_sensitivity`) | Generated from existing minipack JSON |
| ✅ | `data/reports/paper_minipack/sync_sensitivity/README.md` | Supplementary §H | Generated |
| ✅ | `data/reports/paper_minipack/filter_audit/summary.json` | Supplementary §H (`tab:filter_audit`) | Generated from existing minipack JSON |
| ✅ | `data/reports/paper_minipack/filter_audit/README.md` | Supplementary §H | Generated |
| ✅ | `data/reports/paper_minipack/artifact_manifest.json` | Supplementary §H | Generated (SHA256 + byte size) |
| ✅ | `data/reports/paper_minipack/sync_sensitivity_table.json` | Source for `tab:sync_sensitivity` | Existing source |
| ✅ | `data/reports/paper_minipack/filter_bias_audit_table.json` | Source for `tab:filter_audit` | Existing source |
| ✅ | `data/reports/paper_minipack/mini_tables_summary.md` | Human-readable summary | Existing source |

## C. Benchmark reproduction artifacts (Table 3 support)

| Status | Artifact path | Main-paper / supplementary link | Notes |
|---|---|---|---|
| ✅ | `runs/benchmark_bundle_20260325/paper_table_wide.csv` | Supplementary §B.6 | LOSO block |
| ✅ | `runs/benchmark_bundle_20260325/paper_bundle.json` | Supplementary §B.6 / §B.9 | LOSO bundle index |
| ✅ | `runs/trial_generalization_20260326_201509/within/paper_table_wide.csv` | Supplementary §B.6 | Within-trial block |
| ✅ | `runs/trial_generalization_20260326_201509/within/paper_bundle.json` | Supplementary §B.6 / §B.9 | Within-trial bundle index |

## D. Access/ethics policy artifacts (Tier-2 workflow)

| Status | Artifact path | Main-paper / supplementary link | Notes |
|---|---|---|---|
| ✅ | `docs/video_access_policy.md` | Supplementary §E.2 | Existing |
| ✅ | `docs/video_access_request_form.md` | Supplementary §E.2 | Existing |

---

## Packaging checklist for submission

- [ ] Include `docs/paper_supplementary.md` and this inventory file.
- [ ] Include full `data/verify_output_sync/` folder (or equivalent mirrored paths).
- [ ] Include full `data/reports/paper_minipack/` folder (including `sync_sensitivity/`, `filter_audit/`, and manifest).
- [ ] Include frozen Table-4 bundles referenced in Supplementary §B.6.
- [ ] If bundle root is renamed, keep internal relative paths unchanged.

---

Last updated: 2026-03-28
