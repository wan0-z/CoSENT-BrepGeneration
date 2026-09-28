from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from .sent import SentResult


def _bbox_diag(cache: dict) -> float:
    bbox = cache.get("geometry", {}).get("bbox", [0, 0, 0, 1, 1, 1])
    lo = np.array(bbox[:3], dtype=float)
    hi = np.array(bbox[3:], dtype=float)
    d = float(np.linalg.norm(hi - lo))
    return d if d > 1e-9 else 1.0


def _face_trace(face: dict, opacity: float = 0.75) -> go.Mesh3d | None:
    verts = face.get("vertices") or []
    tris = face.get("triangles") or []
    if not verts or not tris:
        return None
    x, y, z = zip(*verts)
    i, j, k = zip(*tris)
    fid = face.get("id")
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        opacity=opacity,
        name=f"face {fid}",
        hovertemplate=f"generated face {fid}<extra></extra>",
        flatshading=True,
        showscale=False,
    )


def _edge_between_faces_as_intersection_hint(cache: dict, a: int, b: int) -> go.Scatter3d | None:
    # Without explicit JSON face-edge geometry mapping, use the nearest STEP curve to both face centroids as a fast visual proxy.
    geom = cache.get("geometry", {})
    faces = {int(f["id"]): f for f in geom.get("faces", [])}
    if a not in faces or b not in faces:
        return None
    def centroid(f):
        v = np.array(f.get("vertices") or [], dtype=float)
        return v.mean(axis=0) if len(v) else np.zeros(3)
    ca, cb = centroid(faces[a]), centroid(faces[b])
    target = (ca + cb) / 2
    best, best_d = None, float("inf")
    for e in geom.get("step_edges", []):
        pts = np.array(e.get("points") or [], dtype=float)
        if not len(pts):
            continue
        d = float(np.linalg.norm(pts.mean(axis=0) - target))
        if d < best_d:
            best, best_d = e, d
    if best is None:
        return None
    pts = np.array(best["points"], dtype=float)
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines",
        line=dict(width=8),
        name=f"edge {a}-{b}",
        hovertemplate=f"generated FAG edge between face {a} and {b}<br>STEP curve proxy: {best.get('id')}<extra></extra>",
    )


def make_generation_figure(cache: dict, sent: SentResult, step_index: int) -> tuple[go.Figure, str]:
    seq = sent.alternating_sequence
    if not seq:
        fig = go.Figure()
        fig.update_layout(title="Generation", scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=35, b=0))
        return fig, "empty"
    step_index = max(0, min(step_index, len(seq) - 1))
    visible = seq[: step_index + 1]
    faces = {int(f["id"]): f for f in cache.get("geometry", {}).get("faces", [])}
    fig = go.Figure()
    for item in visible:
        if item["type"] == "face":
            tr = _face_trace(faces.get(int(item["id"]), {}), opacity=0.65)
            if tr is not None:
                fig.add_trace(tr)
        elif item["type"] == "edge":
            tr = _edge_between_faces_as_intersection_hint(cache, int(item["from"]), int(item["to"]))
            if tr is not None:
                fig.add_trace(tr)
    current = visible[-1]
    label = f"Step {step_index + 1}/{len(seq)}: {current['type']} {current.get('id')}"
    fig.update_layout(title=label, scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=35, b=0), showlegend=False)
    return fig, label
