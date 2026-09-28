from __future__ import annotations
import math
import networkx as nx
import plotly.graph_objects as go
from .models import PartRecord
from .sent import SentResult, build_graph


def make_sent_figure(part: PartRecord, sent: SentResult, seed: int = 0) -> go.Figure:
    g = build_graph(part)
    if len(g) == 0:
        return go.Figure()
    pos = nx.spring_layout(g, seed=seed, dim=2)
    fig = go.Figure()

    # base FAG edges
    edge_x, edge_y, edge_text = [], [], []
    mid_x, mid_y, mid_text = [], [], []
    for u, v, data in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        eid = data.get("edge_id")
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text += [f"FAG edge {eid}: {u}-{v}", f"FAG edge {eid}: {u}-{v}", None]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_text.append(f"e{eid}")
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hovertext=edge_text, hoverinfo="text", name="FAG edges"))
    fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="text", text=mid_text, textposition="middle center", hoverinfo="skip", name="edge labels"))

    # SENT trail arrows as red annotations
    annotations = []
    for s in sent.steps:
        if s.prev_face_id is None:
            continue
        x0, y0 = pos[s.prev_face_id]
        x1, y1 = pos[s.face_id]
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) or 1.0
        shrink = 0.08
        ax, ay = x0 + shrink * dx / norm, y0 + shrink * dy / norm
        bx, by = x1 - shrink * dx / norm, y1 - shrink * dy / norm
        annotations.append(dict(x=bx, y=by, ax=ax, ay=ay, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=3, arrowcolor="red", text=f"{s.segment_id}", font=dict(color="red")))

    node_x = [pos[n][0] for n in g.nodes]
    node_y = [pos[n][1] for n in g.nodes]
    node_text = [str(n) for n in g.nodes]
    hover = [f"FAG face node {n}" for n in g.nodes]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="middle center", marker=dict(size=34), hovertext=hover, hoverinfo="text", name="faces"))

    fig.update_layout(
        title="FAG + SENT trail",
        annotations=annotations,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        showlegend=False,
    )
    return fig
