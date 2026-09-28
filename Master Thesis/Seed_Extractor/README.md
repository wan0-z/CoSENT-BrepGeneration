# Feature Seed Extractor

本目录包含两个彼此独立的处理入口，并共享同一套 attributed FAG、basic matching、generalized expansion 和最终边界验证逻辑。

外部边界验证接受任意二面角类型的 `convex` 边，以及 dihedral type 不是 `acute` 的 `smooth` 边；`concave` 边不被接受。该规则同时用于延迟 potential-cosurface 边之后的 basic 边界检查、扩展候选的逐面检查以及 generalized feature 的最终完整边界检查；seed 内部 relation 仍严格匹配 edge type、convexity 和 dihedral type。

```text
Seed_Extractor/
├── data/
│   ├── feature_seeds.json
│   └── single/<sample_name>/<sample_name>.step|json
├── output/
│   ├── single/<sample_name>/
│   │   ├── cache.json
│   │   ├── attributed_fag.json
│   │   └── evaluation.json
│   └── batch/
│       ├── web_cache/<sample_name>.json
│       ├── summary.json
│       └── per_class_metrics.csv
├── copy_data.py
├── extract_one_step.py
├── extract_batch.py
├── analyze_ground_truth_boundaries.py
├── evaluation.py
└── app.py
```

## 复制单个样本

在 `copy_data.py` 顶部修改 `DATASET_ROOT` 和 `SAMPLE_NAME`，或使用参数运行。目标样本已存在时脚本会停止，不会覆盖或删除原有样本。

```bash
python copy_data.py \
  --data-root ../MF_Explorer/data/mfinstseg \
  --sample 20221123_142528_1
```

## 处理单个样本

`extract_one_step.py` 顶部的 `DEFAULT_SINGLE_SAMPLE_NAME` 控制无参数运行的样本，也可以通过 `--sample` 选择 `data/single` 下的其他样本。

```bash
python extract_one_step.py --sample 20221123_142528_0
```

## 批量处理

在 `extract_batch.py` 顶部修改 `DATASET_ROOT` 和 `SAMPLE_LIMIT`。`DATASET_ROOT` 必须包含 `steps` 和 `labels` 子目录。当前默认读取 `MF_Explorer/data/mfinstseg`，并按文件名自然排序处理前 200 对 STEP/label。

```bash
python extract_batch.py
```

也可以临时覆盖配置。

```bash
python extract_batch.py \
  --data-root ../MF_Explorer/data/mfinstseg \
  --limit 200
```

批量总体准确率、累计混淆矩阵、feature-face detection 指标和 exact-instance 指标保存在 `output/batch/summary.json`；分类别指标同时写入 `per_class_metrics.csv`。

## 真实实例外部边分析

`analyze_ground_truth_boundaries.py` 直接从 STEP 和真实标签实例矩阵构建完整 attributed FAG，并统计真实加工特征实例外部边的 `convexity × dihedral_type` 组合以及候选边界规则的边级、实例级覆盖率。默认分析与 batch 相同的前 200 个样本，忽略 stock、chamfer 和 round/fillet，但保留评估中映射到 rectangular through step 的两类 step。

```bash
python analyze_ground_truth_boundaries.py --limit 200
```

结果写入 `output/batch/ground_truth_boundary_analysis`，包括总体汇总、组合频率、规则覆盖率、分类别覆盖率和逐样本真实实例记录。

## 统一 Dash 前面板

```bash
python app.py
```

前面板默认进入 Batch mode，也可以切换到 Single mode。标题、两行 mode 选择和三个筛选框位于同一个紧凑顶栏，顶栏不保留额外外部上边距，内部上下留白使用相同的小间距。Batch mode 依次按预测结果、加工特征类别和匹配样本进行筛选，各筛选框默认选择当前有效候选列表的第一项。Prediction result 包含 True positive、False negative、False positive 和 Mixed class；Mixed class 表示真实值与预测值均为加工特征、但加工特征类别错误，此时第二个筛选框使用“真实类别 → 预测类别”的组合。加工特征候选会根据 Prediction result 动态过滤，没有出现该预测结果的类别或类别组合不会显示。Single mode 的样本框仍直接列出全部单独处理过的样本。页面保持三列布局，左侧为 3D 模型，中间为上下两张 attributed FAG，右侧为所选面的连接属性和特征实例列表；3D 的小型 Rotate/Pan 按钮位于模型面板右上角，两张 FAG 默认使用 pan mode。FAG 边悬停和所选面连接列表显示 edge type、convexity、dihedral、角度与 seam，但不显示 curve support。悬停 3D 面时只显示基础标签信息，单击面后才在右栏显示与其他面的连接边属性。

3D 模型及两张 FAG 均按照 category ID 使用与 `MF_Explorer/5_prepare_ui_cache.py` 完全相同的类别色表。界面在显示时直接按类别编号解析颜色，因此无须重新生成已有 single 或 batch 缓存。

## 评估约定

评估将 `slanted_through_step` 和 `triangular_blind_step` 归一化为 `rectangular_through_step`，并忽略 ground-truth chamfer 与 round/fillet。混淆矩阵的行是真实类别，列是预测类别。

## 测试

```bash
python -m unittest -v test_seed_extractor.py
```
