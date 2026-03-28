# BadmintonGRF — `baseline/` layout

Library / training / tasks split: **data → models → training loops → analysis CLIs**.

## Commands (single entry)

```bash
python -m baseline --help
python -m baseline train   --method tcn_bilstm --loso_splits … --run_dir …
python -m baseline fuse    --loso_splits … --base_run_dir …
python -m baseline aggregate --inputs runs/*/summary.json --out_md table.md
python -m baseline paper-export --run-root runs/benchmark_runs_…   # paper_bundle + wide table
python -m baseline evaluate | fatigue | ablation
python -m baseline legacy e1 …   # optional: old e1–e5 names
```

### Outputs useful for writing / revising the paper

| Artifact | When |
|----------|------|
| `summary.json` | `train` / `fuse` with `--save_report` |
| `summary_canonical.json` | **Automatically** next to each `summary.json` (unified metric keys: `r2_fz`, `rmse_fz`, …) |
| `late_fusion/summary.json` + canonical | After `fuse --save_report` |
| `paper_bundle.json`, `paper_table_wide.csv`, `paper_table_wide.md` | `python -m baseline paper-export --run-root <benchmark_dir>` (also run at end of `run_all_baselines.sh`) |
| `aggregate` + `--write-canonical` | Optional canonical refresh when merging arbitrary `summary.json` paths |

## Directory map

```
baseline/
├── README.md
├── __init__.py
├── __main__.py                 # python -m baseline <subcommand>
├── train.py                    # Legacy TCN+BiLSTM-only trainer (camera ablation still calls this)
├── registry.py                 # METHODS + build_flat_model + default hparams
├── impact_dataset.py           # BadmintonImpactDataset, build_loso_datasets, …
│
├── recipes/                    # YAML templates only (not auto-loaded by code)
│   ├── README.md
│   └── *.yaml
│
├── models/
│   ├── __init__.py
│   ├── tcn_blocks.py
│   ├── tcn_lstm.py             # LSTMBaseline / TCNBiLSTM
│   ├── gru.py
│   ├── tcn_mlp.py
│   ├── dlinear.py              # DLinear-style trend/seasonal
│   ├── patch_tst.py            # Patch + Transformer (PatchTST family)
│   ├── ms_tcn.py               # Multi-scale dilated TCN
│   ├── transformer_seq.py
│   └── stgcn_transformer.py
│
├── training/
│   ├── __init__.py
│   ├── cli.py                  # argparse for ``baseline train``
│   ├── losses.py
│   ├── metrics.py
│   ├── loso_flat.py
│   └── loso_stgcn.py
│
└── tasks/
    ├── late_fusion.py
    ├── aggregate.py
    ├── paper_export.py       # paper_bundle.json + paper_table_wide.*
    ├── evaluate.py
    ├── fatigue.py
    ├── camera_ablation.py
    ├── canonical.py
    └── legacy_runner.py
```

## Conventions

- **`registry.py`** = method ids, default hyperparameters, `build_flat_model`.
- **`recipes/`** = optional YAML templates for humans / papers (not read by training code unless you wire it in).
- **Literature ↔ baselines** (what we cite vs what is implemented): see `docs/baseline_literature.md`.
- **LOSO JSON (10 subjects, all impact npz on disk):** `python tools/regen_loso_from_disk.py --data-root "$BADMINTON_DATA_ROOT" --check --out data/reports/loso_splits_10p.json` then point `LOSO_SPLITS` / `--loso_splits` at that file.
- **Imports** go downward: `tasks/` may import `training/` and `models/`; `models/` does not import `tasks/`.
