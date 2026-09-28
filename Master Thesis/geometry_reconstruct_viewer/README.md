# Geometry Reconstruction Viewer

This local Dash application demonstrates the thesis reconstruction contract:
the autoregressive sequence emits exact vertex coordinates and exact supporting-
surface parameters, while edge geometry is absent from the generated vocabulary.
After all surfaces are known, a post-processing stage obtains candidate
surface-pair intersections and uses CoAG vertex references, coedge order, and
loop ownership to select and trim the final B-Rep boundaries.

The current baseline stores a transparent fixed-bin scalar quantization next to
every exact value. It does not use a VAE or VQ-VAE.

## Prepare the ten copied samples

```bash
/opt/anaconda3/envs/aagnet-mac/bin/python extract_geometry_sequence.py --force
```

## Run

Build the independent reconstruction cache first. This script reads only exact
vertex XYZ, exact supporting-surface parameters, and edge/coedge/loop incidence
from each extracted sample. It deliberately ignores the original STEP edge
curve and trimmed edge samples when calculating intersections and trims.

```bash
/opt/anaconda3/envs/aagnet-mac/bin/python reconstruct_brep_from_known_inputs.py --force
```

Then start the dashboard.

```bash
/opt/anaconda3/envs/aagnet-mac/bin/python app.py --port 8030
```

Open `http://127.0.0.1:8030`. The feature-space and surface-helper switches can
be changed at any frame. Surface helpers are turned off automatically when the
animation enters loop reconstruction, but can immediately be re-enabled.

The top segmented control selects either `Ground truth`, which retains the
original explanatory playback, or `Reconstruction`, which plays the independent
surface-intersection and topology-driven trim result. In reconstruction mode,
outer loops precede inner loops. Playback pauses at every non-unique trim. The
candidate button displays all topology-consistent loop hypotheses; pressing the
exit button commits the deterministic shortest hypothesis and continues.
When an edge's mate coedge is emitted, the sequence inserts the edge AABB's
minimum and maximum model-coordinate corners as two consecutive geometry steps.
They appear as blue spheres with the same marker size as generated vertices.
At an ambiguous trim, candidate mode also displays the current loop edges' blue
corner markers and translucent blue bounding regions. A full-rank AABB is drawn
as a closed cuboid; a planar, linear, or point-degenerate AABB is drawn in its
corresponding lower-dimensional form. These boxes are extracted from the source
STEP only as known reconstruction inputs and are not derived from edge curves.
When playback is running, dragging the reconstruction camera pauses frame
updates for the duration of the gesture and resumes automatically after the new
camera has been stored. A camera adjustment made after an explicit Pause remains
paused until the user starts playback again.

Feature-space meshes are cached before the Dash app starts. Polygonal passages
and pockets use the `closed_oriented_prismatic_section` rule from
`Seed_Extractor/data/feature_space_seeds.json`: oriented planar halfspaces form
the closed cross-section, and only the mutually clipped side polygons are
extruded along the recession axis. Other planar spaces are clipped jointly by
all active halfspaces and a finite visualization envelope. Trimmed cylindrical
and mixed cells preserve their connected trimmed boundary domains.

The right Plotly scene is created once per selected sample. Animation frames use
Dash partial-property updates to change trace visibility and highlights without
recreating the WebGL graph, while the left STEP figure updates only when the
sample changes.

## Interpretation boundary

For analytic lines and conics, the viewer expands the underlying candidate curve
inside the model box. For general curves, it uses the STEP-selected intersection
branch as an explicitly labelled visualization fallback. These cached curves are
post-processing ground truth for the demonstration and are never generated edge
tokens.

The independent reconstruction mode supports the analytic surface families in
the ten samples. Seam and cosurface edges are intentionally skipped, as required
by the first-version experiment. Exact OCC wires are built where the selected
curve branches form valid wires. A surface-UV triangulation derived from the
recalculated loops supplies a visual fallback for faces whose exact OCC wire
cannot be assembled because a seam was skipped or the deterministic ambiguity
hypothesis is incompatible. This fallback is a visualization of the recovered
trim domain; it is not claimed to be an exported watertight STEP solid.
Candidate loops are explicitly checked for cyclic vertex continuity. If a
candidate loop contains a deliberately skipped seam/cosurface edge, its original
STEP trimmed segment is shown in gray only to close the candidate visualization;
that gray segment is not used to reconstruct any non-seam edge.
