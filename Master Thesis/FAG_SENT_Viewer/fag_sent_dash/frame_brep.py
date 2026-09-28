from __future__ import annotations
import plotly.graph_objects as go


def _empty_3d(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=35, b=0))
    return fig


def make_brep_figure(cache: dict, title: str = "BREP / STEP") -> go.Figure:
    fig = go.Figure()
    geom = cache.get("geometry", {})
    for face in geom.get("faces", []):
        verts = face.get("vertices") or []
        tris = face.get("triangles") or []
        if not verts or not tris:
            continue
        x, y, z = zip(*verts)
        i, j, k = zip(*tris)
        fid = face.get("id")
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            opacity=0.75,
            name=f"face {fid}",
            hovertemplate=f"FAG face: {fid}<br>surface: {face.get('surface_type','')}<extra></extra>",
            flatshading=True,
            showscale=False,
        ))
    for e in cache.get("fag_edges", []):
        # FAG graph edges are topological adjacencies. Precise geometric edge mapping is unavailable
        # unless saved in JSON. We display them in the SENT/generation frames using graph ids.
        pass
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=35, b=0),
        hovermode="closest",
        showlegend=False,
    )
    return fig if fig.data else _empty_3d(title)
