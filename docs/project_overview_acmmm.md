## BadmintonGRF 项目现状与结构总览（投稿前总结）

> 本文是给“未来的我们”和新对话窗口看的项目总览文档。  
> 目标：在不重新翻仓库的情况下，能迅速恢复上下文、继续基于当前进度做 **ACM MM Dataset Track 级别的论文开发**。

---

### 0. 立刻上手（新对话窗口必读）

- **冲刺目标**：ACM MM 2026 **Dataset Track**，定位 **Dataset/Engineering** 叙事，目标冲 Best Paper（强调：数据复杂度/多样性、可复现 pipeline、质量验证、社区价值、合规开放策略）。
- **论文主文件**：`paper/badmintongrf-mm2026.tex`（ACM `acmart`，双栏；正文 **严格 6 页**，参考文献不计页数）。
- **参考文献主库**：`paper/ACM MM 2026.bib`（主库不删），编译使用无空格副本 `paper/ACM_MM_2026.bib`。
- **论文编译（推荐唯一方式）**：
  - 一次编译：在 `paper/` 下运行  
    `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build badmintongrf-mm2026.tex`
  - 实时编译：  
    `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -pvc badmintongrf-mm2026.tex`
  - 输出 PDF：`paper/build/badmintongrf-mm2026.pdf`
- **Fig2（采集布局）已升级为“示意图 + 场景照”组合**：
  - 示意图蓝本：`paper/figures/Fig_acquisition_setup.svg`
  - 已导出 PDF：`paper/figures/Fig_acquisition_setup.pdf`（用于论文嵌入）
  - 场景照（已匿名/待继续优化）：`paper/figures/fig2_scene.png`
- **关键事实（写论文/答审稿的“硬信息”）**：
  - 视频：8 台 DJI Osmo Action 4，1080p，\(\sim\)120 FPS（cam1 4K/60 在大量 trial 误配，benchmark 中排除）。
  - 力台：4 块 Kistler（6-axis），1000/1200 Hz（per-trial metadata）。
  - Mocap：Vicon **8 台**红外相机（T40，UK），C3D；点频 240/250 Hz（per-trial metadata）。
  - 被试：已采集 17 人（sub\_001–017），论文/公开 benchmark 以 10 人（sub\_001–010）为主（LOSO splits）。

---

### 1. 项目基本目的与问题设定

- **核心任务**：从多机位 2D 人体关键点序列（羽毛球运动录像）预测地面反作用力（GRF），重点是垂直分量 Fz。
- **研究动机**：
  - 实际场景中很难在所有场地布置力板（昂贵 / 限制动作区域），但视频+关键点相对容易获取。
  - 如果能从视频端准确预测 GRF，就可以在大规模真实训练/比赛环境下做 **负荷评估、疲劳监测、受伤风险评估**。
- **数据特点**：
  - 17 位羽毛球运动员（sub_001–sub_017），包含 **疲劳阶段**（fatigue_stage1/2/3）、普通阶段（stage1/2/3）和比赛/对抗（rally）。
  - 每个被试有：
    - 多相机视频（最多 8 台，cam1–cam8）
    - Vicon mocap（c3d）
    - 力板数据（从 c3d 提取 GRF）
- **项目目标**：
  1. 构建高质量的 **BadmintonGRF** 数据集（包含对齐良好的 video–GRF 对）。
  2. 构建一条稳定的 **数据处理流水线**：c3d → GRF → video–GRF 对齐 → 姿态 → 片段 → 模型训练。
  3. 做出一套有说服力的 **baseline 与消融实验**，满足 ACM MM dataset track 的审稿标准。

#### 1.1 一句话 Pitch 与差异化卖点（写论文必用）

- **一句话**：首个同步提供多视角高帧率视频、6 分量 GRF、Vicon 全身 mocap（C3D）与疲劳分期标注的羽毛球生物力学多模态数据集，使「从普通摄像机无接触估算 GRF」第一次有了真实标注数据。
- **相对 FineBadminton 等数据集的差异化要点（可做 Table 1）**：
  - GRF：我们有 **6 分量 GRF（1000/1200 Hz）**，FineBadminton 无 GRF。
  - 视角：我们是 **8 机位高帧率视频**，现有 GRF 数据集多为单视角或无视频。
  - Mocap：提供 Vicon C3D（约 52 markers），可做 3D biomechanics 研究。
  - 疲劳：有明确的 **3 阶段疲劳分期标注**，其他羽毛球数据集通常无。
  - 开源策略：10 人骨骼点 + GRF + C3D +（可选 IMU）开放，原始视频走申请制。

#### 1.2 ACM MM 2026 Dataset Track 关键约束（防踩坑）

- **版式与页数**：ACM 双栏，正文 **严格 6 页**（不含参考文献），超出直接 desk-reject。
- **评审与投稿**：
  - 单盲评审（作者信息可见），通过 OpenReview 投稿。
  - 投稿时必须提供 **可访问的数据集链接**（建议 Zenodo，带 DOI）。
  - 审稿侧重：数据集的 **原创性 / 复杂度与多样性 / 社区价值 / 描述质量**，baseline 算法只是配套，不是主贡献。
- **定位提醒**：论文必须明确是 **“dataset track” + 基准解**，E1–E5 属于 benchmark pipeline，而不是“新 SOTA 算法”。

---

### 2. 当前整体进展（高层）

- **数据准备侧**：
  - 已完成：
    - 从 mocap `.c3d` 中提取 GRF（step0）。
    - 人工 UI 标注 GRF–视频对齐（step1 + Flask 工具）。
    - 自动验证同步质量 + outlier 检测（step2）。
    - YOLO/ByteTrack + 姿态提取（step3），生成 per-frame 关键点与 track 信息。
    - 自动基于 GRF 峰值做切片、生成 **impact-segment** 样本（step4，`.npz`；实现细节见仓库）。
    - 生成多种规模的 LOSO splits（5/6/7/8/9/10 人）。
  - 数据质量检查：
    - `mocap_inspection.json` 显示每个 subject 的 `point_rate_hz` / `analog_rate_hz` 与 marker 数量；
    - 两套采样率组合：240/1200 Hz 与 250/1000 Hz（均为整倍数，利于对齐）。
- **模型/实验侧**（E1–E6 系列）：
  - **E1**：单相机 TCN+BiLSTM baseline（10 人 LOSO）已重训完成。
  - **E2**：基于 E1 checkpoint 的多机位 Late Fusion（无须重训）已完成。
  - **E3**：疲劳分析（stage1/2/3 + Kruskal-Wallis）已完成。
  - **E4**：ST-GCN + Temporal Transformer（STGCNTransformer），在不同人数 N = 5…10 上的 scaling 实验已完成。
  - **E5**：基于 E4 checkpoint 的多机位 Late Fusion（更强模型 base）已完成（同样 5…10 人）。
  - **E6-E1**：以 E1（BiLSTM）为 base 的相机消融实验（cam1–cam8 独立训练）已完成。
  - **E6-E4**：以 E4 为 base 的相机消融 **尚未实现**（目前是“可选扩展”，不是硬要求）。
- **绘图/论文准备侧**：
  - 新增统一绘图脚本 `analysis/plot_figures_acmmm.py`，自动根据 `runs/` 生成 ACM MM 风格的主要图：
    - scaling curve（E4/E5）
    - 相机消融（E6-E1）
    - 疲劳阶段对比（E3）
    - 模型整体对比（E1/E4/E5）
  - LaTeX 写作工程化（2026-03 迭代中）：
    - `paper/badmintongrf-mm2026.tex`：ACM `acmart`（`sigconf,screen,review`）论文主文件（含 Dataset Track 常见章节：Data Access/Privacy、Licensing、Ethics、Reproducibility 等）。
    - 作者区块：当前 **不再标注共同一作脚注**（Shengze Cai 转为普通共同作者）；宋宪为通讯作者（`sx1993@zju.edu.cn`）。
    - 编译链：以 `latexmk -outdir=build` 为唯一推荐（避免 LaTeX Workshop 的 sample bib 混淆）；输出在 `paper/build/`。
    - `paper/figures/` 已包含论文图和素材（见 0 节）。

现在已经进入 **结果整理 + 论文撰写** 阶段。

#### 2.1 采集系统与数据规格速览

- **多模态同步采集**：
  - 摄像机：8 台 @ 119.88 fps，1080p；角点布局（每角一高一低），部分 cam1 为 4K，已在数据集中排除。
  - 力台：4 块 Kistler 6 分量力板，纵向排列；analog 采样率有两档：
    - 1200 Hz：sub_001 / 004 / 005 / 007 / 008 / 009 / 011 / 013 / 015 / 016
    - 1000 Hz：sub_002 / 003 / 006 / 010 / 012 / 014 / 017
  - Mocap：Vicon 系统（**8 台 IR 相机**），约 52 markers（个别 session 略有增减），point rate 为 240 或 250 Hz，对应各自的力台 session。
  - IMU：腰骶骨 + 双大腿 + 双小腿一共 5 个，质量一般，**仅作为“可选辅助模态”简单说明，不作为卖点**。
- **同步与工具**：
  - 手工标注 `grf_event_sec + video_event_frame`，写入 `*_sync.json`。
  - 自研 Web GRF–Video 对齐工具（Flask + 前端单页），精度约 ±8.3 ms，可作为论文 Section 中的一小节与 Figure。
- **被试与开源范围**：
  - 总采集：17 人（sub_001–017），持续扩展中。
  - 论文与开源基准：当前以 **10 人（sub_001–010）** 为主（LOSO splits 已生成）。
  - 人群特征：国家二级及以上羽毛球运动员，8 男 2 女；视频因隐私采用“申请制访问”，骨骼点 / GRF / C3D / IMU 计划公开。
- **整体数据布局速记**：
  - 代码根：`/home/nky/BadmintonGRF/`
  - 数据根：`/media/nky/Lenovo/data/`，下含 `sub_XXX/labels, video, mocap, pose, segments` 以及 `reports/loso_splits_*.json`、`mocap_inspection.json`。

#### 2.2 数据质量统计与 quick facts（写 Method / Dataset 时直接抄）

- 下肢关键点置信度：约 **0.980 ± 0.061**。
- 高置信帧率（score > 0.5）：约 **98.6% ± 8.4%**。
- 主运动员丢失率：约 **0.60% ± 1.62%**。
- 当前 10 人数据集在所有相机上的 **总 impact 片段数约 2,880**，适合作为“数据规模”一行写进正文。

---

### 3. 代码与文件结构概览

#### 3.1 顶层目录

- `environment.yml`  
  项目环境定义。当前主要依赖：
  - Python 3.10
  - numpy, scipy, matplotlib
  - pip 包：`opencv-python`、`c3d`（注意：旧名 `python-c3d` 已不可用，需用 `c3d`）、`pymupdf`（用于从 PDF 抽取摘要补全 bib）。
- `tools/scan_dataset.py`  
  扫描数据根目录（`BADMINTON_DATA_ROOT` 或仓库内 `data/`），统计被试和 trial 分布。
- `run_all_baselines.sh`  
  顺序跑注册表中的基线（`train` → 可选 `fuse`；见脚本内说明）。
- `python -m baseline …`  
  统一实验入口：`train` / `fuse` / `aggregate` / `fatigue` / `legacy e1`…（见 `baseline/README.md`）。旧版「全量 E1→…→E6」若需可自建 shell 包装上述子命令。

#### 3.2 `pipeline/`（预处理流水线）

- `pipeline/config.py`  
  全局配置 `CFG`：
  - 数据根：`root` / `data_dir` / `subjects_base`
  - 路径构造：`video_path(trial, cam)`、`grf_path(trial)`、`mocap_path(trial)`、`segment_path(...)`
  - 质量阈值：RMSE 分档、peak 检测阈值等。
- `pipeline/step0_extract_grf.py`  
  - 输入：`{sub}/mocap/{trial}_mocap.c3d`
  - 输出：`{sub}/labels/{trial}_grf.npy`  
  - 存储结构包含：
    - `timestamps`（秒）
    - `combined.forces`（Fx, Fy, Fz）
    - per-plate forces、元数据（如 analog_rate）。
- `pipeline/step1_align_ui.py`（Flask + 前端单页）  
  - 作用：
    - 人工标注 GRF–视频对齐（点击 GRF onset + 视频事件帧）。
    - 计算并写出：
      - `offset_sec = grf_event_sec - video_event_sec`
      - `offset_uncertainty_sec = 1 / fps`
      - `per_cam[cam]` 结构（每个相机有独立的 `video_event_frame/sec` 和 `offset`）。
    - 生成 `badmintongrf_manifest.json`：
      - Index：每个 trial 的视频路径、GRF 路径、对齐质量等。
      - `npz_schema` 文档：描述 step4 输出的 impact-segment 样本格式（见仓库 README / `step4_segment.py`）。
  - 主要输出：
    - `{sub}/labels/{trial}_sync.json`
    - `subjects_base/badmintongrf_manifest.json`
- `pipeline/step2_verify_sync.py`  
  - 全量扫描 `*_sync.json`，兼容旧版单相机与新版 `per_cam` 格式。
  - 统计 offset 分布，生成：
    - 统计 CSV / JSON
    - outliers/whitelist 列表
    - 多张 publication-ready 图（偏移分布、进度矩阵等）。
- `pipeline/step3_extract_pose.py`  
  - 使用 YOLO + ByteTrack 识别运动员并提取姿态。
  - 逻辑亮点：
    - 识别主运动员（基于 vertical impulse + 水平位置）。
    - ByteTrack 配置 + 丢失恢复策略。
  - 输出：`{sub}/pose/{trial}_camN_pose.npz`
    - `keypoints`、`scores`、`bbox`
    - `event_frame_local`、`track_status` 等。
- `pipeline/step4_segment.py`  
  - 核心逻辑：  
    - 在 pose 覆盖的 ±5s 窗口内，基于 GRF 峰值检测所有落地冲击；
    - 以每个 GRF 峰值为中心，切出固定窗口（默认 0.5 s pre + 0.5 s post）。
  - 输入：
    - `*_grf.npy`、`*_sync.json`、`*_camN_pose.npz`
  - 输出：`{sub}/segments/{trial}_camN_impact_{idx:03d}.npz`（impact-segment）  
    - 关键字段：
      - `keypoints_norm (T, 17, 2)`、`scores (T, 17)`
      - `grf_at_video_fps (T, 3)`、`grf_normalized (T, 3)`
      - `timestamps_video / timestamps_grf`
      - `ev_idx`（窗口内事件帧）
      - `subject/trial/stage/camera/quality/peak_force_bw/stat_lost_rate` 等。
- 可视化辅助脚本：
  - `pipeline/visualize_segment.py`
  - `pipeline/visualize_pose.py`
  - `pipeline/scan_suspicious_main_athlete.py` 等，用于 debugging 和质检。

#### 3.3 `baseline/`（模型与训练）

- `baseline/impact_dataset.py`
  - 输入：impact-segment `.npz` 文件路径列表。
  - 策略：
    - 加载 `keypoints_norm` 与 `scores`，构造 `(T, 119)` 特征：
      - pos (T, 34)、vel (T, 34)、acc (T, 34)、score (T, 17)
    - 质量过滤：
      - `stat_lower_body_mean_score`
      - `stat_lost_rate`
      - `peak_force_bw` 上限
    - 支持数据增强（时间偏移、水平翻转、坐标噪声、score dropout）。
  - LOSO 构造：
    - `build_loso_datasets(loso_splits_path, test_subject, cameras=None, ...)`。
- `baseline/models/tcn_lstm.py`（E1 基线）
  - 架构：
    - TCNBlock ×2（GroupNorm + same padding）
    - BiLSTM×N
    - MLP head → `(B, T, 1或3)`
  - 典型输入输出：`(B, T, 119) → (B, T, 1)`（仅预测 Fz）。
- `baseline/models/stgcn_transformer.py`（E4）
  - 输入先 reshape：
    - `(B, T, 119) → (B, T, 17, 7)`（pos/vel/acc/score）
  - 结构：
    - Spatial GCN（COCO-17 skeleton graph）×2
    - 对关节平均池化 → `(B, T, 64)`  
    - 线性投影到 `hidden_dim`
    - Sinusoidal PE
    - TransformerEncoder（pre-norm，batch_first）
    - MLP head → `(B, T)`
- `baseline/train.py`（E1 训练主入口）
  - CLI 支持：
    - `--loso_splits path`
    - `--cameras ...`
    - `--run_dir` / `--run_name` / `--out_dir`
    - `--fz_only` `--epochs` `--patience` `--batch_size` 等。
  - 核心特性：
    - 接触窗口加权 MSE（ev_idx±half_win 权重=alpha）。
    - 按 subject 安全切分 train/val。
    - warmup + CosineAnnealing，早停，NaN 防护。
  - 输出：
    - `runs/<ts>_lstm_tcn_cam.../`：
      - `fold_sub_xxx/best_model.pth`
      - `summary.json`（per_fold + mean/std）
      - `summary_canonical.json`（后续由 canonical 工具新增）。
- `baseline/tasks/`（训练后分析 / 融合 / 消融）
  - `late_fusion.py`：多机位 Late Fusion（`python -m baseline fuse`）。
  - `fatigue.py`：按 stage 分组 + Kruskal-Wallis（`python -m baseline fatigue`）。
  - `camera_ablation.py`：相机消融（`python -m baseline ablation`）。
  - `aggregate.py`：合并多份 `summary.json`（`python -m baseline aggregate`）。
  - `canonical.py`：生成 `summary_canonical.json`。
  - `legacy_runner.py`：旧 e1–e5 命名（`python -m baseline legacy …`）。

#### 3.4 `analysis/`（分析与绘图）

- `analysis/compare_sub001_grf.py`  
  对单被试（如 sub_001）进行 predicted vs ground truth GRF 曲线对比，输出图像 + summary。
- `analysis/dataset_stats.py`  
  统计 trial、camera 分布，生成 dataset 统计图。
- `analysis/fatigue_analysis.py`  
  直接在 GRF 层面比较 fatigue stages 的峰值与平均曲线（不涉及预测模型）。
- `analysis/verify_sync.py`  
  可视化 step2 验证结果（目前为占位/示例）。
- `analysis/plot_figures_acmmm.py`  
  统一生成 ACM MM 风格图表（Fig1–Fig4）。

#### 3.5 `docs/`（文档）

- `docs/exp.md`  
  实验编号与脚本路径对照 + 状态：
  - E1：`baseline/train.py`
  - E2：`baseline/tasks/late_fusion.py`
  - E3：`baseline/tasks/fatigue.py`
  - E4：`baseline/training/loso_stgcn.py`
  - E5：`late_fusion` + ST-GCN run
  - E6：`baseline/tasks/camera_ablation.py`（E1 base，E4 base 版本尚未实现）

#### 3.6 `paper/`（论文写作目录，ACM MM Dataset Track）

- `paper/badmintongrf-mm2026.tex`
  - ACM `acmart` 模板（`sigconf,screen,review`）主稿骨架。
  - 约定：编译输出统一在 `paper/build/`，PDF 预览以这份为准。
- `paper/ACM MM 2026.bib`
  - 当前 related work / 方法引用库（Zotero 导出并清洗/补全）。
  - 已通过脚本把所有条目补齐 `abstract`（用于后续自动分组/写 related work）。
- `paper/abstract_verify_report.json`
  - `abstract` 与 `file` 指向 PDF 的一致性校验报告（自动生成）。

#### 3.7 `tools/`（写作与可复现辅助脚本）

- `tools/fill_bib_abstracts_from_pdfs.py`
  - 从 `.bib` 的 `file={...pdf}` 路径读取 PDF，抽取前若干页文本，自动补全缺失 `abstract` 字段。
  - 默认会写 `.bak` 备份。
- `tools/verify_bib_abstracts.py`
  - 校验 `.bib` 中的 `abstract` 是否能与对应 PDF 前几页文本匹配（防止错配/瞎编）。

---

### 4. 已完成实验与结果概览（按 E1–E6）

> 这里只给“结构级”概览，具体数值可以从 `summary_canonical.json` 和图表中提取。

#### 4.0 主结果一眼看完（Table/Scaling 用）

- **主表（10 人 LOSO）核心指标**（具体数值最终以 `summary_canonical.json` 为准）：
  - E1（TCN-BiLSTM，单相机）：r² ≈ 0.20，RMSE ≈ 0.42×BW，peak_timing 误差约 2.6 帧。
  - E4（ST-GCN+Transformer，单相机）：r² ≈ 0.26，RMSE ≈ 0.41×BW，peak_timing 误差显著降低到约 0.9 帧。
  - E5（E4 base，多机位 Late Fusion）：r² ≈ **0.35**，其余指标从 `e5_late_fusion/summary.json` 读取后补全。
- **Scaling curve（5–10 人）**：E4/E5 的 r² 随被试数单调总体提升（个别点如 9 人 < 8 人，是因为 sub_009 缺少一台相机，轻微拉低了多机位融合质量）。
- **相机消融（E6-E1）关键结论**：
  - 网前角 A 视角整体最好，其中 **cam8（高位）r²≈0.21 为最优相机**，相同角的低位 cam3 次之。
  - 底线角 D 视角最差，特别是高位 cam6，r² 甚至接近 0 或略负。
  - 旧结论“低位系统性优于高位”在 10 人新数据下 **不再成立**，需要在论文中用更新后的表格支持新说法。
  - E6-E4 版本相机消融已明确 **不做**，简要在“局限性 / future work”中说明理由即可。

- **E1：单相机 baseline（TCN+BiLSTM）**
  - 10 被试 LOSO，输入 `(T,119)`，输出 GRF Fz。
  - 输出：
    - `runs/20260317_015200_e1_lstm_tcn_camall/summary{,_canonical}.json`
    - 每个 fold 的 best_model + train_log。
- **E2：E1 的多机位 Late Fusion**
  - 不重新训练，只在测试时对多相机 impact 进行置信度加权。
  - 结果在 `.../e2_late_fusion/summary{,_canonical}.json`。
- **E3：疲劳分析（GRF + 预测性能随 fatigue stage 的变化）**
  - 分组：stage1 / stage2 / stage3（以及 rally/其他作为对照）。
  - 输出：E3 的表格 CSV（便于直接贴入论文）。
- **E4：ST-GCN + Transformer（E4）**
  - 在 N = 5/6/7/8/9/10 人设置下分别训练 LOSO。
  - 输出：
    - `runs/20260317_024228_e4_{Np}_stgcn_transformer/summary{,_canonical}.json`
    - 各 fold 的 best_model。
- **E5：E4 的多机位 Late Fusion**
  - 类似 E2，但 base 模型是 E4。
  - 输出：
    - `.../e5_late_fusion/summary{,_canonical}.json`。
- **E6-E1：相机消融（BiLSTM base）**
  - 每个 camN 单独重训一套 E1 模型，统计各相机的平衡表现。
  - 输出：
    - `runs/20260317_035957_e6_e1base/ablation_cameras/camera_ablation_summary.json`  
      + 各 camN 对应的 runs。
- **E6-E4：相机消融（ST-GCN base）**
  - 尚未实现（作为 optional 的后续扩展）。

---

### 5. 统一绘图脚本：自动生成 ACM MM 风格图

脚本：`analysis/plot_figures_acmmm.py`

- **Fig1：Scaling curve**  
  - 输入：`runs/*_e4_{Np}_stgcn_transformer/summary_canonical.json` 与对应 E5 的 canonical。
  - 输出：`figures/fig4_scaling_curve.{pdf,png}`  
  - 内容：N ∈ {5,6,7,8,9,10} 上，E4/E5 的 r² 曲线。

- **Fig2：Camera ablation (E6-E1)**  
  - 输入：`runs/*_e6_e1base/ablation_cameras/camera_ablation_summary.json`。
  - 输出：`figures/fig5_camera_ablation_e1.{pdf,png}`
  - 内容：cam1…8 的 r²/ RMSE / peak_err 对比条形图。

- **Fig3：Fatigue stages (E3)**  
  - 输入：`runs/*_e1_lstm_tcn_camall/e3_fatigue/fatigue_table.csv`。
  - 输出：`figures/fig6_fatigue_stages.{pdf,png}`
  - 内容：stage1/2/3 的 rmse_fz mean±std 条形图。

- **Fig4：Model comparison (E1/E4/E5)**  
  - 输入：
    - 最新 E1（10p）canonical summary
    - 最新 E4 10p canonical summary
    - 对应 E5 canonical summary
  - 输出：`figures/fig4_model_comparison.{pdf,png}`
  - 内容：三种模型在 10 人设置下的 r² / RMSE 条形对比。

运行方式：

```bash
(badminton_grf) python -m analysis.plot_figures_acmmm
# 输出会打印到 stdout 并写入 figures/*.pdf / *.png
```

> 旧版根目录下的 `plot_scaling_curve.py` / `plot_grf_prediction.py` 逻辑，现已收敛到本脚本中；如有分歧，以 `analysis/plot_figures_acmmm.py` 为准。

---

### 6. 关键技术细节与“坑点”速记（务必看一眼）

- **时序与采样率处理**：
  - `VIDEO_FPS = 119.88`，窗口长度 `WINDOW_SEC = ±0.5 s`，因此视频侧 `T ≈ 120` 帧，事件中心在 `ev_idx ≈ 60` 附近。
  - `GRF_RATE` 不能写死：**必须从 `.npy` metadata 中读取（1000 或 1200 Hz）**，step4 已按“秒”做对齐，混合采样率是安全的。
  - Mocap point rate 为 240 / 250 Hz，对齐也统一在“秒”域中完成。
- **Dataset 返回格式**：
  - `BadmintonImpactDataset.__getitem__` 返回的是 **Dict**，而非 tuple：
    - 主要键：`pose (T,119)`、`target (T,1)`、`ev_idx`、`path`、`subject`、`stage`、`camera`、`trial`。
    - 下游模型使用时注意 `target` 需要 `.squeeze(-1)`，否则维度会多一维。
- **E4（ST-GCN+Transformer）实现要点**：
  - `pad_collate` 返回 `(feats, targets, ev_idxs, pad_mask)` 四元组，Transformer 需要 `pad_mask` 正确掩蔽 padding。
  - Sinusoidal positional encoding 是在 **线性投影之后、Transformer 之前** 注入；若省略或放错位置，`peak_timing` 误差会退化到 ~18 帧（实验已验证）。
  - 优化器与训练技巧：`AdamW + CosineAnnealing`，`max_norm = 1.0` 的梯度裁剪，能够稳定收敛。
- **Late Fusion 细节（E2 vs E5，论文中要说明差异）**：
  - E2（BiLSTM base）置信度权重：`sc.mean()`，即 **全身 17 关节的平均 score**。
  - E5（ST-GCN base）置信度权重：`sc[:, 11:].mean()`，即 **下肢关节 11–16 的平均 score**，更契合“落地冲击主要由下肢承担”的物理先验。
- **Analog 双采样率与 MoCap markers 的特殊情况**：
  - Analog 采样率分布已在 2.1 节给出；再次强调：**任何地方不要假设统一 1000/1200Hz，必须信任 metadata**。
  - MoCap markers：大多数 session 为 52 markers，但：
    - `sub_006 = 51`，`sub_007/008 = 53`，`sub_011 = 65`（换了协议）。
    - C3D 中 marker 名称为 `*0~*51` 等未命名编号，无法在论文中给出标准表格，只能描述为“约 52 markers，个别 session 略有变体（详见公开 C3D）”。
- **相机与视角的特殊情况**：
  - cam1 在部分被试为 4K，已在数据集构建和实验中排除；相机消融表中因此记为 `N/A`。
  - 相机布局：四角点，每角一高一低；最终结论是 **视角朝向（尤其是网前角 A）远比安装高度重要**，这点在撰写 Discussion 时可点明。
- **论文图表与补充材料策略**：
  - 正文保留 4 张主图（采集系统 schematic、同步工具截图、GRF 预测曲线、scaling curve），疲劳分析曲线与更细致的 E6 结果建议放补充材料/项目网站。
  - 任何需要占据较多篇幅的 per-subject 统计表、更多 ablation 也倾向放 Supplement。

---

### 7. 面向下一轮开发与论文撰写的提示

> 下面是为了“新对话窗口”仍能游刃有余地继续推进而准备的 checklist。

- **环境复现**
  - 优先用 `environment.yml` 创建 conda 环境，然后用 pip 安装最新版 `c3d` 替代 `python-c3d`。
  - 核心依赖：
    - `torch` + CUDA（按当前机器配置，不在 `environment.yml` 中强绑定）。
    - `c3d`、`opencv-python`、`matplotlib`。
- **关键入口总结**
  - 数据 pipeline：
    - `step0_extract_grf.py` → `step1_align_ui.py` → `step2_verify_sync.py` → `step3_extract_pose.py` → `step4_segment.py`
  - 模型训练/实验：
    - E1/E2/E3/E4/E5/E6：`baseline/train.py`、`baseline/training/`、`baseline/tasks/`（见 `baseline/README.md`）
    - 统一入口：`python -m baseline train` / `fuse` / `aggregate`（见 `baseline/README.md`）
  - 实验批量重跑：
    - `bash run_all_baselines.sh`（或按 `baseline/README.md` 用 `python -m baseline …` 分步）。
- **论文贡献结构建议**
  - **数据集与任务定义**：
    - BadmintonGRF 数据集规模 / trial 类型 / 标签设计。
    - GRF 提取、对齐与 impact-segment 样本设计（含 schema，见仓库）。
  - **方法部分**：
    - E1（BiLSTM baseline）：架构 + 特征工程（pos/vel/acc/score）。
    - E4（ST-GCN + Transformer）：空间/时间建模方式与动机。
    - Late Fusion 策略（E2/E5）：置信度加权 + 多相机事件对齐。
  - **实验部分**：
    - 主表：E1 vs E4 vs E5 在 10 人 LOSO 下的指标（用 canonical summary）。
    - scaling curve：N vs r² / RMSE 曲线。
    - 相机消融：哪台相机“最好”、多机位带来的收益。
    - 疲劳分析：不同疲劳阶段的 GRF 与预测误差差异（附统计检验）。
- **未来/可选扩展**
  - 实现 E6-E4（ST-GCN base 的相机消融）。
  - 探索更轻量模型用于实际部署（剪枝 / distillation）。
  - 引入更多物理约束（如 “flight phase Fz ≈ 0” penalty 已在注释中提到）。

---

以上这个 markdown（`项目总览 + 结构 + 实验状态 + 绘图入口`）已经足够让新的对话窗口在不知道前情的情况下，快速恢复上下文、继续做：

- 细致的结果分析；
- 论文图表与表格的微调；
- 新的 ablation / 扩展实验（如 E6-E4）。

后续如果你想，我们可以下一轮专门针对“论文写作”组织一个 `paper.md`，把 Introduction / Related Work / Method / Experiment 逐章梳理出来。

