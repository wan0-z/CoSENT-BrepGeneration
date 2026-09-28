# CoAG Generation Viewer

This folder contains the complete single-sample pipeline and Dash viewer. The
three matching source files are resolved from `data/steps`, `data/labels`, and
`data/graphs` using the same basename.

## Prepare one sample

Use the existing `aagnet-mac` environment, which already contains OpenCascade,
Dash, Plotly, and NetworkX:

```bash
/opt/anaconda3/envs/aagnet-mac/bin/python prepare_sample.py 20221123_142528_10 --force
```

The command writes `web_cache/<sample>.webcache.json`. It extracts the labeled
CoAG, repeats boundary nodes while splitting feature-instance subgraphs, creates
one CoSENT per subgraph, and stores a globally deduplicated generation order.
Chamfer (label 0) and round/fillet (label 23) are treated as completion nodes,
not feature seeds.

## Run the Dash app

```bash
/opt/anaconda3/envs/aagnet-mac/bin/python app.py 20221123_142528_10
```

Open <http://127.0.0.1:8050>. The right panel starts with locally cropped analytic
support surfaces around each machining-feature instance. Each arrow step adds one CoAG
node's parent B-rep edge: its first oriented occurrence is red, and after its
mate appears the edge becomes black.

Use **Auto** in the right-hand controls to play all generation steps
automatically. The same button pauses playback and becomes **Replay** at the
final frame.
