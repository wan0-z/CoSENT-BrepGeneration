# Transformer demo

本目录是一个可运行的 STEP → 特征识别 → 特征空间 → Coedge 序列 → 离散词表 → 因果 Transformer 训练实验。它验证序列化和训练管线，不把短时 token 训练等同于已经能够生成有效、封闭的 STEP 实体。原始数据、Seed Extractor 和 Geometry Reconstruction Viewer 的源文件均不修改。

## 三维特征浏览页面

运行 `.venv/bin/python app.py --port 8057`，打开 [Dash 特征面与特征空间](http://127.0.0.1:8057/)。页面只有一个三维视图，可切换全部 100 个样本以及单个／全部特征实例。实色显示原始 STEP 的特征拓扑面，半透明显示对应的 feature-space 支撑边界，箭头沿拓扑外法向指向 void side；背景模型、特征面、空间和法向可以分别开关，也可调节显示延展范围和透明度。

拓扑面绘制复用 `geometry_reconstruct_viewer` 所依赖的 OCC face triangulation 和 trimmed edge sampling，以及同样的 Plotly Mesh3d 光照方式。面编号严格沿用 Seed Extractor 的遍历顺序。网格精度随模型大小调整，避免原 viewer 固定绝对公差使本数据集的小面无法三角化。缓存仅写入 `data/viewer_cache`，不改变 STEP、extracted JSON、训练序列或权重。

空间边界读取本 demo 的 `config/feature_space_seeds.json` 和已提取的 role/orientation/angular-domain 信息。平面边界按其他角色的有向半空间裁剪；圆柱边界保留角域，按轴线延展，并接受其他角色的约束裁剪。显示范围仅是数值可视化窗口，不增加封口面，也不把各个完整无限支撑面简单重叠。平面裁剪为解析计算，涉及曲面的裁剪交线采用网格近似；该页面不是复杂 piecewise cell、连通分量和 collision 的精确布尔认证器。

`.venv/bin/python check_viewer.py` 会检查全部 100 个样本的网格、执行 7 项测试并保存 `logs/viewer_audit.json` 和 `logs/viewer_tests.txt`。本次发现 16 个已识别实例的平面 void-side 约束互斥（13 个 rectangular_through_slot，其余为 v_circular_end_blind_slot、rectangular_blind_slot、rectangular_passage 各 1 个）。这类实例保留原拓扑面着色并显示明确提示，不翻转法向来伪造空间；它们也仍保留在既有训练数据中，本次查看器任务未重新筛选或训练语料。

页面接口参考 [Dash Dropdown 文档](https://dash.plotly.com/dash-core-components/dropdown) 与 [Plotly Mesh3d 文档](https://plotly.com/python/3d-mesh/)。

## 数据与两个 seed 文件的职责

`prepare.py` 从 `../datasets/eccv2026-cad-challenge-data/train/target_step` 按文件名顺序复制 100 个 STEP 到 `data/steps`。`data/manifest.json` 保留来源、目标和 SHA-256。训练／验证按零件随机种子 42 划分为 90／10，两个集合之间不共享同一个零件的窗口。

识别严格调用 `Seed_Extractor/extract_one_step.py` 的 `build_attributed_fag`、`load_feature_seeds` 和 `extract_features_in_priority_order`，与 `extract_batch.py` 的识别流程相同，使用 `config/feature_seeds.json` 的全部 20 类规则。原批量流程的外部标签用于评估，而非识别，因此本 demo 不需要伪造标签文件。识别结果是原提取器的预测，不是人工真值，也不能通过这份未标注数据报告 feature-recognition accuracy。

识别完成以后，`config/feature_space_seeds.json` 才用于将每个已识别实例映射成 surface roles、法向／sigma、有向 void-side 谓词、相互裁剪角色和圆柱角域。它不参与实例匹配。原 JSON 本身是 discussion draft；其要求的全部 Boolean collision、recession escape、cavity evidence 和 connected-component certification 并未全部重新实现。原提取器已经执行的 collision 结果被保留，其他规则也保留在空间记录中，不将“记录规则”误称为“已验证规则”。

平面空间采用 `n·x ≥ d` 的交集，surface patch 由其他角色的约束相互裁剪，开口方向不由有限 bbox 封闭。圆柱采用 `sigma·(distance_to_axis² − r²) ≥ 0`。partial-cylinder 和 circular-end 情况保留源 patch 的角域，并且仅在对应角域内施加该圆柱约束，不能以整圆柱交集替代。`prepare.contains` 提供这些隐式空间的可执行点包含测试。对非矩形 UV trims／复杂 piecewise cell 的角域表达是受限 demo，尚不构成通用精确 B-Rep feature-cell Boolean 重建。

## 实际序列约定

先输出可选 condition；没有条件时输出 `<NO_CONDITION>`。Condition 只包含特征类别、supporting surface 参数、角色和 oriented/clipping 信息，不包含生成目标的 coedge 列表。多个 condition 共用一个 supporting surface 时，该 surface 只生成一次，并携带多个 condition/role reference。

随后按 Viewer 的规则合并 analytic cosurface faces，先 condition surfaces 后 context surfaces。同一 surface 内采用 outer loop 优先，再按原 face、loop 和 loop-local 顺序稳定排序；每个 edge 的首次 occurrence 放入 NEW 段，剩余 occurrences 放入 MATE 段。这里修正了 Viewer 的静态分组细节，seam 或同一 surface 中出现第二次的 coedge 也必须成为 mate，而不是两个 NEW。

每个 coedge 有全局递增 ID、face/loop 所属关系与方向、surface type、edge type、凹凸性、二面角类型和离散角度、起终点 reference，以及到所有已生成关联 coedge 的连接记录。连接向量顺序与 Viewer 一致，依次是 mate、next、previous、cocurve、cosurface、coface、share_start、share_end，编码为 `<REL_0>` 至 `<REL_255>`。只引用过去的 coedge，不提前暴露未来 ID。

顶点首次出现时输出 `<NEW_POINT> ID <XYZ> Qx Qy Qz </XYZ>`，之后只输出 `<POINT_REF> ID`。精确原始坐标保存在 extracted JSON 中，模型并不同时看到精确浮点数和它的量化答案。第二个 coedge 出现时输出 `<MATE> prior_coedge_id <EDGE_REF> edge_id <EDGE_BBOX> <P_MIN> Qx Qy Qz </P_MIN> <P_MAX> Qx Qy Qz </P_MAX> </EDGE_BBOX>`。bbox 通过 OCC `AddOptimal(..., useTriangulation=False, useShapeTolerance=False)` 从实际 trimmed edge 几何计算，不是由两个端点或可视化折线估算。

NEW/MATE 排序不是几何上的 loop 环绕顺序，因此在所有 coedges 之后额外保存 `<LOOP_ORDER>`，其中只引用已经生成的 coedge ID，恢复真实 cyclic order。然后才输出 `<SURFACE_GEOMETRY>` 和 `</SURFACE>`。全模型以 `<END_MODEL><EOS>` 结束。非流形 edge 明确报错；没有 mate 的单 occurrence edge 不伪造 bbox，并在计数中单列。

## 几何离散化与词表

第一版不使用 VAE。模型空间使用逐零件的统一中心和等比例缩放，缩放范围同时包住实际 vertices、edge bbox、support origins 和 spline control points，避免无限 support 的原点位于实体 bbox 外时被静默 clamp。变换与误差界保存在每个 sequence 的 `quantization` 中；推理时该模型坐标框架须作为外部预处理约定提供。这个实验还没有学习模型的绝对单位／尺度分布。

坐标、方向、长度、角度、节点和权重使用有语义的字段标记，共享 1,024 个 `<Qn>` 数值 token；长度按模型尺度归一化，方向映射到 [-1,1]，角度按固定角区间量化。每轴坐标量化误差不超过 `scale / 1023`。解码方向需要重新单位化并正交化。浮点原始数据无额外小数位截断，但离散化后的几何当然不再逐位精确。

支持 plane、cylinder、cone、sphere、torus、B-spline 和 Bézier surface。数据中的一般 spline 保留真实 degree、pole grid、weights、knots 和 multiplicities；节点作 affine 参数域归一化，权重除以共同最大值，这两项不改变底层曲面。quasi-uniform 被单独标记，但不会把一般 spline 硬压成 3×3。未知 surface family 明确拒绝，不生成只有 origin 的假完整几何。

`data/vocabulary.json` 固定包含全部 20 类特征名称、对应角色、支持的曲面／曲线类型、全部量化 bin 和关系 bitmask。整数 ID 是序列中的 ordinal symbols，不是浮点 geometry code。`data/corpus_report.json` 给出实际类别覆盖、surface 类型、token 长度和词表大小。词表构建包含预定义语法和本 demo 所需 ID 范围，不包含基于验证集拟合的数值量化器。

## Transformer 与损失

模型为 2 层、64 维、4 个 attention heads 的小型因果 Transformer。目标 token 的 query 始终可以访问完整 condition prefix 和当前窗口内的过去 token，不能访问未来目标。为了在 CPU 上处理最长约 8k 的 condition，prefix 内部采用局部因果 attention，目标对 prefix 使用全局 attention。长模型本体采用 128 个历史 token 加 256 个目标 token 的窗口，不声称保留 156k 序列的完整拓扑历史。

训练使用 teacher forcing 下的 next-token cross entropy。Condition/prefix 和纯历史上下文不计算目标损失，避免模型仅仅背诵输入 condition。10% 的样本访问使用真正的无条件版本；该版本不会残留引用不存在 condition 的 token。TensorBoard 记录 train/step_loss、train/step_accuracy、四个字段组的 loss、固定 held-out 窗口的 validation loss/accuracy、gradient norm 和 learning rate。字段损失是诊断指标，默认总损失仍是所有目标 token 的标准均值，不引入尚未定义的几何一致性 loss。

默认 3 个 round，每个 round 每个训练零件抽取 2 个窗口，合计 540 次更新。round 不是完整语料 epoch。窗口在同一零件内采样，几何位置会得到额外抽样，不把文件尾部直接丢掉。`--full-pass` 则遍历每个目标位置一次；它保留同样的有限历史上下文，但训练时间更长。验证集使用固定、可复现的 held-out 窗口，验证准确率是 token-level teacher-forced accuracy，不是整序列完全一致率。

## 运行

当前 `.venv` 使用已有 `aagnet-mac` 的 OCC、PyTorch 和数值库，TensorBoard 仅安装于这个 demo 的独立环境，没有更新原环境。以下命令从本目录运行。

```bash
.venv/bin/python prepare.py
.venv/bin/python serialize.py
.venv/bin/python -m unittest -v test_demo
.venv/bin/python train.py --run demo_v2 --rounds 3
.venv/bin/python verify.py --run demo_v2
.venv/bin/python -m tensorboard.main --logdir runs/demo_v2 --host 127.0.0.1 --port 6006
```

新实验请使用不同的 `--run` 名称，脚本不会覆盖已有 checkpoint。继续某次实验时同时指定匹配的 `--run` 和 `--resume checkpoints/<run>/last.pt`，模型尺寸参数也需要一致。每次正式 run 内保存独立的 vocabulary 快照与配置。首次 `demo_v1` 是补齐未出现类别 vocabulary 之前的开发试跑，正式结果使用 `demo_v2`，不要将 v1 的旧词表 checkpoint 与当前词表混用。

`show_sequence.py 000000` 输出可读的 XML 式 token 文件。`checkpoints/<run>/teacher_forced_reconstruction.json` 对照真实目标和一步预测；`free_running_continuation.json` 是不读取后续真值的自由生成，可能不满足 grammar。demo 目前没有把生成 token 解码回有效 STEP；bbox、端点和 surface 对一般 seam/cosurface/ambiguous-intersection 曲线也不保证唯一恢复。

打开 [TensorBoard](http://127.0.0.1:6006/#scalars) 查看曲线。TensorBoard 日志接口参考 [PyTorch SummaryWriter 官方文档](https://docs.pytorch.org/docs/stable/tensorboard.html)。这次真实实验的数字写入 `RESULTS.md`，测试结果和 TensorBoard event 审计保存在 `logs`。
