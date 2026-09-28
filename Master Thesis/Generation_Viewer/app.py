"""Three-panel Dash viewer for labeled B-rep, CoAG/CoSENT, and generation."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

from coag_pipeline import feature_color, feature_name, load_cache


GRAPH_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _scene(bbox: list[float], *, camera: dict[str, Any] | None = None) -> dict[str, Any]:
    scene = {
        "xaxis": {"visible": False, "range": [bbox[0], bbox[3]]},
        "yaxis": {"visible": False, "range": [bbox[1], bbox[4]]},
        "zaxis": {"visible": False, "range": [bbox[2], bbox[5]]},
        "aspectmode": "data",
        "bgcolor": "#f8fafc",
    }
    if camera:
        scene["camera"] = camera
    return scene


def _mesh_trace(mesh: dict[str, Any], *, color: str, opacity: float, name: str, hover: str) -> go.Mesh3d | None:
    vertices, triangles = mesh.get("vertices") or [], mesh.get("triangles") or []
    if not vertices or not triangles:
        return None
    x, y, z = zip(*vertices)
    i, j, k = zip(*triangles)
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity,
        name=name, hovertemplate=hover + "<extra></extra>", flatshading=True,
        lighting={"ambient": 0.65, "diffuse": 0.7, "specular": 0.15},
        lightposition={"x": 100, "y": 200, "z": 150}, showscale=False,
    )


def make_labeled_brep_figure(cache: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    shown_labels = set()
    for face in cache["face_meshes"]:
        label_id = face.get("face_label")
        label_name = face.get("face_label_name", feature_name(label_id))
        instance = face.get("feature_instance_id")
        trace = _mesh_trace(
            face, color=feature_color(label_id), opacity=0.94,
            name=f"{label_id}: {label_name}",
            hover=(
                f"STEP face ID: {face['face_id']}<br>Feature: {label_id} / {label_name}"
                f"<br>Instance: {instance if instance is not None else 'none'}"
                f"<br>Surface: {face.get('face_type', 'unknown')}<br>Bottom: {face.get('bottom', 0)}"
            ),
        )
        if trace is None:
            continue
        trace.legendgroup = str(label_id)
        trace.showlegend = label_id not in shown_labels
        shown_labels.add(label_id)
        fig.add_trace(trace)
    fig.update_layout(
        title={"text": "Labeled STEP model", "x": 0.03, "xanchor": "left"},
        scene=_scene(cache["view_bbox"]), margin={"l": 0, "r": 0, "t": 46, "b": 0},
        paper_bgcolor="#f8fafc", uirevision=f"left-{cache['sample_name']}",
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,.78)", "font": {"size": 10}},
    )
    return fig


def _relations_text(edge: dict[str, Any], relation_order: list[str]) -> str:
    return ", ".join(name for name in relation_order if edge["relations"].get(name)) or "none"


def make_coag_figure(cache: dict[str, Any]) -> go.Figure:
    positions = {int(k): v for k, v in cache["graph_layout"].items()}
    fig = go.Figure()

    edge_x, edge_y = [], []
    midpoint_x, midpoint_y, midpoint_text = [], [], []
    for index, edge in enumerate(cache["edges"]):
        x0, y0 = positions[edge["u"]]
        x1, y1 = positions[edge["v"]]
        edge_x.extend((x0, x1, None))
        edge_y.extend((y0, y1, None))
        midpoint_x.append((x0 + x1) / 2)
        midpoint_y.append((y0 + y1) / 2)
        midpoint_text.append(f"CoAG edge {index}<br>{_relations_text(edge, cache['relation_order'])}")
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line={"color": "#cbd5e1", "width": 0.8},
        hoverinfo="skip", name="CoAG relations", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=midpoint_x, y=midpoint_y, mode="markers", marker={"size": 8, "opacity": 0},
        hovertext=midpoint_text, hoverinfo="text", name="relation details", showlegend=False,
    ))

    # Each subgraph gets a separate arrow color. Small perpendicular offsets keep
    # repeated boundary paths legible when two CoSENT trails share an edge.
    annotations = []
    for subgraph_index, subgraph in enumerate(cache["subgraphs"]):
        color = subgraph["path_color"]
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", line={"color": color, "width": 4},
            name=subgraph["title"], hoverinfo="skip",
        ))
        for step in subgraph["cosent"]["steps"]:
            previous, current = step.get("prev_node_id"), step["node_id"]
            if previous is None or previous not in positions or current not in positions:
                continue
            x0, y0 = positions[previous]
            x1, y1 = positions[current]
            dx, dy = x1 - x0, y1 - y0
            norm = math.hypot(dx, dy) or 1.0
            # Center the offset set around the base edge.
            offset = ((subgraph_index % 5) - 2) * 0.006
            ox, oy = -dy / norm * offset, dx / norm * offset
            shrink = 0.022
            annotations.append({
                "x": x1 - dx / norm * shrink + ox, "y": y1 - dy / norm * shrink + oy,
                "ax": x0 + dx / norm * shrink + ox, "ay": y0 + dy / norm * shrink + oy,
                "xref": "x", "yref": "y", "axref": "x", "ayref": "y",
                "showarrow": True, "arrowhead": 3, "arrowsize": 0.85,
                "arrowwidth": 2.6, "arrowcolor": color, "text": "",
            })

    node_ids = [node["id"] for node in cache["nodes"]]
    node_x = [positions[node_id][0] for node_id in node_ids]
    node_y = [positions[node_id][1] for node_id in node_ids]
    node_colors = [feature_color(cache["nodes"][node_id].get("face_label")) for node_id in node_ids]
    hover = []
    for node in cache["nodes"]:
        hover.append(
            f"Coedge c{node['id']}<br>STEP edge: {node.get('edge_id')}<br>STEP face: {node.get('face_id')}"
            f"<br>Feature: {node.get('face_label')} / {node.get('face_label_name')}"
            f"<br>Instance: {node.get('feature_instance_id')}<br>Face surface: {node.get('face_type')}"
            f"<br>Edge curve: {node.get('edge_type')}<br>Loop: {node.get('loop_type')} {node.get('loop_id')}"
        )
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=[f"c{node_id}" for node_id in node_ids],
        textposition="middle center", textfont={"size": 8, "color": "#0f172a"},
        marker={"size": 22, "color": node_colors, "line": {"color": "white", "width": 1.4}},
        hovertext=hover, hoverinfo="text", name="CoAG nodes", showlegend=False,
    ))
    fig.update_layout(
        title={"text": "CoAG with per-subgraph CoSENT trails", "x": 0.03, "xanchor": "left"},
        annotations=annotations, xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 5, "r": 5, "t": 46, "b": 5}, paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc", hovermode="closest",
        legend={"orientation": "h", "x": 0, "y": -0.02, "font": {"size": 9}, "bgcolor": "rgba(255,255,255,.8)"},
        uirevision=f"coag-{cache['sample_name']}",
    )
    return fig


def _surface_traces(cache: dict[str, Any]) -> list[go.Mesh3d]:
    traces = []
    for surface in cache["support_surfaces"]:
        trace = _mesh_trace(
            surface, color=feature_color(surface.get("face_label")), opacity=0.17,
            name=f"support: {surface.get('face_label_name')}",
            hover=(
                f"Feature support surface<br>Source STEP face: {surface.get('source_face_id')}"
                f"<br>Feature: {surface.get('face_label')} / {surface.get('face_label_name')}"
                f"<br>Instance: {surface.get('feature_instance_id')}<br>Surface: {surface.get('surface_type')}"
            ),
        )
        if trace is not None:
            trace.showlegend = False
            traces.append(trace)
    return traces


def make_generation_figure(cache: dict[str, Any], generation_step: int, camera: dict[str, Any] | None = None) -> tuple[go.Figure, str, float]:
    order = cache["generation_order"]
    step = max(0, min(int(generation_step), len(order)))
    visible_node_ids = order[:step]
    nodes = cache["nodes"]
    counts = Counter(nodes[node_id].get("edge_id") for node_id in visible_node_ids)
    seen_coedges: dict[int, list[int]] = defaultdict(list)
    for node_id in visible_node_ids:
        edge_id = nodes[node_id].get("edge_id")
        if edge_id is not None:
            seen_coedges[int(edge_id)].append(node_id)

    fig = go.Figure()
    for trace in _surface_traces(cache):
        fig.add_trace(trace)
    for edge in cache["brep_edges"]:
        edge_id = int(edge["edge_id"])
        count = counts.get(edge_id, 0)
        if count <= 0 or not edge.get("points"):
            continue
        points = edge["points"]
        x, y, z = zip(*points)
        complete = count >= 2
        color = "#111827" if complete else "#ef233c"
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", line={"color": color, "width": 6 if not complete else 4},
            name="confirmed edge" if complete else "first coedge",
            hovertemplate=(
                f"STEP edge: {edge_id}<br>Curve: {edge.get('edge_type')}<br>Seen coedges: {seen_coedges[edge_id]}"
                f"<br>Status: {'black / confirmed' if complete else 'red / first occurrence'}<extra></extra>"
            ), showlegend=False,
        ))

    if step == 0:
        label = f"Step 0/{len(order)} · prerequisite feature support surfaces"
    else:
        current = nodes[order[step - 1]]
        label = (
            f"Step {step}/{len(order)} · coedge c{current['id']} · STEP edge {current.get('edge_id')} · "
            f"{current.get('face_label_name')}"
        )
    complete_edges = sum(1 for count in counts.values() if count >= 2)
    visible_edges = len(counts)
    progress = 100.0 * step / max(len(order), 1)
    fig.update_layout(
        title={"text": f"B-rep generation · {visible_edges} visible / {complete_edges} confirmed", "x": 0.03, "xanchor": "left"},
        scene=_scene(cache["view_bbox"], camera=camera),
        margin={"l": 0, "r": 0, "t": 46, "b": 0}, paper_bgcolor="#f8fafc",
        showlegend=False, uirevision=f"generation-{cache['sample_name']}",
    )
    return fig, label, progress


def make_app(sample_name: str) -> Dash:
    cache = load_cache(sample_name)
    app = Dash(__name__, assets_folder=str(Path(__file__).parent / "assets"))
    app.title = f"CoAG Generation · {cache['sample_name']}"
    left_figure = make_labeled_brep_figure(cache)
    middle_figure = make_coag_figure(cache)

    chips = []
    for subgraph in cache["subgraphs"]:
        chips.append(html.Span(
            f"{subgraph['title']} · {len(subgraph['core_nodes'])}+{len(subgraph['boundary_nodes'])}",
            className="subgraph-chip", style={"borderColor": subgraph["path_color"], "color": subgraph["path_color"]},
        ))

    app.layout = html.Div([
        dcc.Store(id="generation-step", data=0),
        dcc.Interval(id="autoplay-interval", interval=320, n_intervals=0, disabled=True),
        html.Header([
            html.Div([
                html.H1("CoAG Generation Viewer"),
                html.P(f"Sample {cache['sample_name']} · {cache['n_faces']} faces · {cache['n_coedges']} coedges"),
            ]),
            html.Div(chips, className="subgraph-chips"),
        ], className="topbar"),
        html.Main([
            html.Section([
                html.Div([html.Span("01"), html.Div([html.Strong("Labeled B-rep"), html.Small("Hover for STEP face and feature")])], className="panel-heading"),
                dcc.Graph(id="brep-graph", figure=left_figure, config=GRAPH_CONFIG, className="viewer-graph"),
            ], className="viewer-panel"),
            html.Section([
                html.Div([html.Span("02"), html.Div([html.Strong("CoAG + multi-CoSENT"), html.Small("Colored arrows show each subgraph sequence")])], className="panel-heading"),
                dcc.Graph(id="coag-graph", figure=middle_figure, config=GRAPH_CONFIG, className="viewer-graph"),
            ], className="viewer-panel"),
            html.Section([
                html.Div([html.Span("03"), html.Div([html.Strong("B-rep generation"), html.Small("Red first occurrence · black mate-confirmed")])], className="panel-heading"),
                html.Div([
                    html.Button("←", id="previous-step", n_clicks=0, title="Previous generation step"),
                    html.Button("→", id="next-step", n_clicks=0, title="Next generation step"),
                    html.Button("▶ Auto", id="autoplay-button", n_clicks=0, className="autoplay-button", title="Automatically play the B-rep generation"),
                    html.Button("Final", id="final-step", n_clicks=0, className="final-button", title="Jump to final wireframe"),
                ], className="generation-controls"),
                dcc.Graph(id="generation-graph", config=GRAPH_CONFIG, className="viewer-graph"),
                html.Div([
                    html.Div(id="generation-label", className="generation-label"),
                    html.Div(html.Div(id="progress-fill"), className="progress-track"),
                ], className="generation-status"),
            ], className="viewer-panel generation-panel"),
        ], className="viewer-grid"),
    ], className="app-shell")

    @app.callback(
        Output("generation-step", "data"),
        Input("previous-step", "n_clicks"), Input("next-step", "n_clicks"),
        Input("autoplay-button", "n_clicks"), Input("final-step", "n_clicks"),
        Input("autoplay-interval", "n_intervals"),
        State("generation-step", "data"), prevent_initial_call=True,
    )
    def change_step(_previous: int, _next: int, _autoplay: int, _final: int, _tick: int, current: int) -> int:
        current = int(current or 0)
        if ctx.triggered_id == "previous-step":
            return max(0, current - 1)
        if ctx.triggered_id == "next-step":
            return min(len(cache["generation_order"]), current + 1)
        if ctx.triggered_id == "autoplay-button":
            # Clicking Replay at the final frame starts a fresh run.
            return 0 if current >= len(cache["generation_order"]) else current
        if ctx.triggered_id == "final-step":
            return len(cache["generation_order"])
        if ctx.triggered_id == "autoplay-interval":
            return min(len(cache["generation_order"]), current + 1)
        return current

    @app.callback(
        Output("autoplay-interval", "disabled"), Output("autoplay-button", "children"),
        Input("autoplay-button", "n_clicks"), Input("generation-step", "data"),
        Input("previous-step", "n_clicks"), Input("next-step", "n_clicks"), Input("final-step", "n_clicks"),
        State("autoplay-interval", "disabled"), prevent_initial_call=True,
    )
    def control_autoplay(
        _autoplay: int, generation_step: int, _previous: int, _next: int,
        _final: int, disabled: bool,
    ) -> tuple[bool, str]:
        total = len(cache["generation_order"])
        if ctx.triggered_id == "autoplay-button":
            will_play = bool(disabled)
            return (not will_play), ("⏸ Pause" if will_play else "▶ Auto")
        if ctx.triggered_id in {"previous-step", "next-step", "final-step"}:
            return True, "↻ Replay" if int(generation_step or 0) >= total else "▶ Auto"
        if int(generation_step or 0) >= total:
            return True, "↻ Replay"
        return bool(disabled), "▶ Auto" if disabled else "⏸ Pause"

    @app.callback(
        Output("generation-graph", "figure"), Output("generation-label", "children"),
        Output("progress-fill", "style"), Input("generation-step", "data"),
        State("generation-graph", "relayoutData"),
    )
    def render_generation(step: int, relayout: dict[str, Any] | None):
        camera = (relayout or {}).get("scene.camera")
        figure, label, progress = make_generation_figure(cache, int(step or 0), camera=camera)
        return figure, label, {"width": f"{progress:.3f}%"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", help="Sample basename already prepared with prepare_sample.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    make_app(args.sample).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
