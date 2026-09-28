# Space 提取器诊断与 UI 更新

本次仅修改 Seed_Extractor；MF_Explorer 数据只读。未改动识别判据，没有重跑 200 个样本。真实标签仅用于提取结束后的评估和诊断重放，不参与识别。

## 三个样本的实际结果

| 样本后缀 | 提取实例数 | Face accuracy | Feature face accuracy |
| --- | ---: | ---: | ---: |
| 10 | 2 | 51.85% | 38.10% |
| 10030 | 5 | 74.29% | 68.97% |
| 1 | 6 | 75.76% | 70.37% |

完整结果和逐阶段诊断保存在 `output_space/single/20221123_142528_<后缀>/`，其中 `diagnostic_report.json` 包含 surface 关系、trim 条件、group constraint、有效边界采样和碰撞重放结果。GT 重放是解释性对照，不表示运行时曾接受该候选。

## 圆柱相关失败

样本 10 的 circular_blind_step 为 face 2、3。Surface 关系、90° 圆柱 trim、联合约束和有效边界检查全部通过。实际拒绝原因是碰撞体积 1481.686723，容差约 0.0000535。当前 trimmed 分支将所有候选面的采样点投影后取凸包，沿圆柱轴向两端各延伸 10%，直接返回碰撞结果，遗漏底面半空间裁切。对照实验保留同一个凸包棱柱，只增加平面半空间裁切，碰撞体积变为 0。

样本 10 的真实标签没有 v_circular_end_blind_slot；它包含两个 circular_end_pocket，分别为 face 11、17、18、19、26 和 face 12、13、14、15、16、25。两组 surface 关系（包括相切）和 trim 均通过，但有效边界筛选失败，正常流程尚未进入对应组合的碰撞检测。代码将圆弧端的完整圆柱内侧也作为全局交集约束，这不是由直边和圆弧拼接的 trimmed cell，因而会排除正常直侧面。强制重放碰撞分别为 4.711828 和 3.607942；补上平面裁切后均为 0，但这不能替代对整个 trimmed cell 构造的修正。

样本 10030 的 v_circular_end_blind_slot 为 face 10、11、12、25。相切关系和 trim 均通过。两个直侧面分别仅 7/24 个采样点满足完整圆柱的内侧约束，低于程序要求的 17/24，因此在碰撞检测之前被拒绝。这不是相切条件没有满足。

现存 200 个 space 样本汇总中，circular_end_pocket、v_circular_end_blind_slot、h_circular_end_blind_slot 的预测面数均为 0，真实支持面数分别为 383、71、57。此次三个样本不含独立的 h 类验证实例，不能据此宣称已逐实例验证 h 类的全部失败原因。

## 平面特征失败

样本 10030 的 rectangular_pocket 为 face 3、4、5、19、21。Surface 匹配和联合约束通过；face 3 只有 25/42 个采样点满足其他面的半空间，程序要求至少 30 个，因此失败。绕过这一检查，仅对原候选重放碰撞，体积为 0。

样本 1 的 triangular_passage 为 face 1、3、19。Surface 匹配和闭合截面联合约束通过。face 1 通过 25/42 点（要求 30），face 3 通过 37/54 点（要求 38），两者不满足的点均违反 face 19 的半空间。该实例邻接 face 2（round）。正常流程在有效边界阶段结束；对原实例单独重放碰撞体积为 0。故这里不能把漏检解释为圆角挡住通道，直接原因是目前的 70% 离散采样比例门槛。该比例也不等于有效裁切面积比例，受三角化和边采样密度影响。

样本 1 另有 circular_end_pocket（face 15、16、17、18、30）同样被错误的全圆柱有效边界约束拒绝。

## Viewer 与实际碰撞体不相同

Seed_Viewer 是理想种子的独立几何绘图。其盲特征的下端停在底面，上端才向开口延伸；当前提取器的 trimmed 凸包棱柱却向轴向两端延伸，并遗漏平面裁切。不能使用 seed viewer 的图形作为当前提取器碰撞体的证据。

现在已接受实例的 `collision.mesh` 保存提取器碰撞输入体的三角网格（含人工封口），主 app 的 Recession cone 勾选框直接显示这些缓存，透明红色，不在浏览器端重新计算几何。该图层显示已提取实例，不把仅供 GT 诊断的候选混入预测结果；被拒绝的候选目前在诊断 JSON 中查看数值。旧缓存没有网格时会提示重新运行 space extractor。

## 指标显示与使用

Face accuracy 和 Feature face accuracy 显示当前样本值 / 当前 normal 或 space 模式下的 batch 样本算术平均值。不是将所有面合并后的加权准确率。当前 space batch 平均为 79.98% / 72.07%。Class F1 根据 machining feature 选项更新，统计范围为当前样本；mixed class 的 A→B 选项显示预测类别 B 的 F1，10、20 映射到 8，chamfer、round 映射到 stock，保持既有评估口径。

重启主 app 后，在 Single + Space 下查看三个样本。Normal 结果和原布局保留。诊断脚本 `diagnose_space_sample.py` 与 `extractor_one_step_space.py` 使用相同参数，例如 `--sample 20221123_142528_10 --step <STEP路径> --label <JSON路径>`。

验证使用 `/opt/anaconda3/envs/aagnet-mac/bin/python Seed_Extractor/test_space_viewer.py`，覆盖透明网格与相机状态、类别 F1 更新、Dash 布局和回调连接，3 项测试通过。尚未对运行中的浏览器做人工视觉验收。

后续修正应先统一 seed viewer 与 extractor 的 trimmed cell 几何构造，确保仅向开放方向延伸并应用底面限制；再用实际裁切后的有效边界判断代替 70% 点数比例。不能仅降低采样比例或放宽碰撞容差来掩盖这两个问题。
