# FAG SENT Dash Viewer

一个三栏 Dash 应用：

1. 左侧：STEP/BREP 3D 展示，鼠标 hover 显示 FAG 面编号或边编号。
2. 中间：根据 FAG 抽取的 SENT 图，红色有向线显示 SENT trail，并标注第几段。
3. 右侧：按照 SENT 生成顺序逐步展示 `面-边-面-边...` 的 3D 恢复过程。

## 安装

建议使用 conda 安装 OpenCascade 后端：

```bash
conda create -n fag_dash python=3.10 -y
conda activate fag_dash
conda install -c conda-forge pythonocc-core -y
pip install -r requirements.txt
```

## 先生成 webcache

```bash
python fag_sent_dash/build_cache.py --force
```

默认读取 `../Generation_Viewer/data/graphs`，并将缓存写入本项目的
`web_cache` 文件夹。

## 启动 Dash

```bash
python fag_sent_dash/app.py
```

浏览器打开 <http://127.0.0.1:8056>。

## JSON 格式假设

每个 JSON 至少包含：

```json
{
  "step_path": ".../part.stp",
  "n_faces": 4,
  "nodes": [1,2,3,4],
  "edges": [{"u":1,"v":4,"shared_edges":1}]
}
```

FAG 的边编号按 JSON `edges` 数组顺序从 1 开始。STEP 中的 face 编号按 OpenCascade 遍历顺序从 1 开始。若你的 FAG face id 与 STEP face 顺序不是同一个顺序，需要在 JSON 中额外保存映射，或修改 `step_cache.py` 中的 face mapping 逻辑。
