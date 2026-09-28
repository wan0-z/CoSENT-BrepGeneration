# Demo 实验记录

2026-08-30，正式 run 为 `demo_v2`。本实验已实际运行，不是仅提供训练脚本。TensorBoard 当前地址为 [本机损失曲线](http://127.0.0.1:6006/#scalars)，事件文件保存在 `runs/demo_v2`；关闭服务不会丢失历史曲线。

## 数据与序列

从原始训练数据按文件名顺序复制 `000000.step` 至 `000099.step`，100 个文件均成功处理，来源和副本的 SHA-256 检查通过。90 个零件用于训练，10 个零件用于固定验证，不在训练与验证之间共享零件。

识别使用 `feature_seeds.json` 的全部 20 类规则，得到 2,331 个预测实例，实际覆盖 13 类；其中 circular_through_slot 有 2,010 个实例，类别分布明显偏斜。through_hole、triangular_passage、6sides_passage、Oring、blind_hole、triangular_pocket、circular_end_pocket 未被识别到。这是原提取器的输出，不是特征人工真值；特别是高频圆柱类结果需要后续人工抽检。未为了凑齐类别而替换样本或更改匹配规则。

100 条有条件序列合计 2,334,907 tokens，词表包含 2,922 个 token，其中 1,024 个为数值量化 bins。最长序列 156,064 tokens，最长 condition prefix 8,126 tokens。另为每个零件生成无条件版本，用于训练时 condition dropout。

总计 9,742 个 vertices、30,289 个 coedges、15,177 个 topological edges。15,112 个 edge 各有一对 coedges，均只在 mate 后生成一次 bbox。其余 65 个均为 OCC 标记的退化 edge，只有一个 coedge，不伪造 mate 或 bbox。它们分布在 12 个零件中；这类极点退化情况与普通开放边不同，后续完整几何解码需要单独约定。原始退化标志与坐标保留在 extracted JSON。

## 实际训练结果

CPU 上训练 477,034 参数的两层 Transformer，完成 540 次更新，约 53 秒（不包含语料加载和最后预测示例的导出）。每个训练零件访问 6 个目标窗口；3 个 round 是窗口抽样轮次，不是整个 230 万 token 语料的 3 次全量 epoch。每个窗口最多 256 个目标 token，另附 128 个历史 token 和完整 condition。共计算 137,568 个训练目标 token，存在重复抽样，不代表唯一覆盖率。

验证采用 10 个未参与训练零件的固定窗口，每次共评估 15,360 个目标 token。

| 训练位置 | 训练交叉熵 | 验证交叉熵 | 验证 token 准确率 |
| --- | ---: | ---: | ---: |
| 随机初始化 | — | 7.9909 | 0.013% |
| 180 updates | 5.7085 | 4.6403 | 22.845% |
| 360 updates | 3.6494 | 3.3736 | 37.135% |
| 540 updates | 3.1026 | 3.0738 | 38.796% |

最后一次验证中，topology、vertex、edge bbox、surface 四组交叉熵分别为 2.5564、4.5676、4.8163、4.6461。这些字段组包含结构分隔符和数值 token，因此不是纯坐标误差。几何组损失仍明显高于拓扑组；当前结果不能证明精确几何重建能力。全部训练损失、验证损失、分组损失、准确率、梯度范数和学习率已经写入 TensorBoard，并通过其 HTTP 接口核实这些曲线确实可用。

## 可检查的产物

`checkpoints/demo_v2/best.pt` 和 `last.pt` 为真实权重及优化器状态，旁边保存配置、该次运行的独立词表与逐轮指标。`teacher_forced_reconstruction.json` 展示给定真实前文时的一步预测，不能当作自主生成；`free_running_continuation.json` 展示不读取未来真值的 96-token greedy 续写，目前没有语法约束或有效实体保证。

`examples/000000.tokens.txt` 是可读序列。`logs/tests.txt` 记录 9 项通过的测试，覆盖全部 100 个样本的引用、几何出现时机、new-first 顺序、同面 seam 配对、无条件引用、量化误差、特征空间半无限性、圆柱角域和因果 attention。`logs/demo_v2_verification.json` 核实事件指标有限、所有训练样本被访问、540 个 step 已记录，并确认 checkpoint 与词表一致。

## 仍需方法论定稿的边界

第一，bbox 的对角点是 edge 的范围约束，不是一般空间曲线的唯一参数化。这不影响本次 token 训练，但会影响将生成序列解码成精确 B-Rep 的充分性；65 个退化 edge 的闭合规则也需要单独定义。

第二，feature-space JSON 是草案。本 demo 已实现平面有向半空间、角色相互裁剪和带角域保护的圆柱约束，未完成任意非矩形 trim、piecewise cell 的连通分量和全部 collision／escape／cavity 认证。因此不能声称已完成该草案的全部精确空间布尔实现。

第三，当前选择直接离散化、1,024 bins、逐模型等比例归一化和有限历史窗口。这些是 demo 的明确默认值，并非论文最终方案。完整长程 coedge 引用记忆、condition 子集采样、数值量化精度和几何一致性约束仍需通过更大规模实验确定；尤其不应将这个短窗口模型解释成能记住完整 156k-token 序列的全局拓扑。
