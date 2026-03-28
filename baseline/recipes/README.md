## `baseline/recipes/` (reference-only)

这里的 `*.yaml` **目前不会被代码自动读取**。

- **实际生效的配置来源**：命令行参数（首选 ``python -m baseline <train|fuse|…>``，见 ``baseline/README.md``）；`train.py` / `tasks/*.py` 为具体实现。
- **这些 YAML 的用途**：作为“实验配方/模板”，便于复现实验、写论文与团队沟通。

### 建议用法

- **统一入口（推荐）**：`python -m baseline train ...` / `fuse` / `aggregate`；旧版 e1–e5 名称：`python -m baseline legacy e1 ...`（见 `tasks/legacy_runner.py`）。
- **跑实验时**：把 YAML 里的参数转写成 CLI 命令，并把命令（或日志）保存在 `runs/<exp>/config.json` / `train_log.csv` 等产物中。
- **改配方时**：同步更新你实际使用的 CLI 命令，避免“yaml 写了但没生效”的误解。

### 文件说明

- **`lstm.yaml`**：E1 单相机（TCN + BiLSTM）训练参数模板。
- **`transformer.yaml`**：E4（ST-GCN + Temporal Transformer）相关参数模板（若采用）。
- **`multiview.yaml`**：多相机/多视角实验参数模板（如 E2 Late Fusion 或后续扩展）。

