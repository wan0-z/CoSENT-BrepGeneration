# app_coag.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dash
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go


# ============================================================
# 默认路径
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR
COAG_CACHE_DIR = APP_DIR / "coag_cache"
INDEX_PATH = COAG_CACHE_DIR / "index.json"

APP_TITLE = "CoAG Viewer"
HOST = "127.0.0.1"
PORT = 8055
DEBUG = False


FACE_CATEGORIES = [
    'chamfer',                      # 0
    'through_hole',                 # 1
    'triangular_passage',           # 2
    'rectangular_passage',          # 3
    '6sides_passage',               # 4
    'triangular_through_slot',      # 5
    'rectangular_through_slot',     # 6
    'circular_through_slot',        # 7
    'rectangular_through_step',     # 8
    '2sides_through_step',          # 9
    'slanted_through_step',         # 10
    'Oring',                        # 11
    'blind_hole',                   # 12
    'triangular_pocket',            # 13
    'rectangular_pocket',           # 14
    '6sides_pocket',                # 15
    'circular_end_pocket',          # 16
    'rectangular_blind_slot',       # 17
    'v_circular_end_blind_slot',    # 18
    'h_circular_end_blind_slot',    # 19
    'triangular_blind_step',        # 20
    'circular_blind_step',          # 21
    'rectangular_blind_step',       # 22
    'round',                        # 23
    'stock'                         # 24
]

STOCK_LABEL_ID = 24

RELATION_ORDER = [
    "mate",
    "next",
    "previous",
    "cocurve",
    "cosurface",
    "coface",
]


# ============================================================
# 颜色
# ============================================================

FEATURE_COLORS = {
    0:  "#e41a1c",
    1:  "#377eb8",
    2:  "#4daf4a",
    3:  "#984ea3",
    4:  "#ff7f00",
    5:  "#ffff33",
    6:  "#a65628",
    7:  "#f781bf",
    8:  "#999999",
    9:  "#66c2a5",
    10: "#fc8d62",
    11: "#8da0cb",
    12: "#e78ac3",
    13: "#a6d854",
    14: "#ffd92f",
    15: "#e5c494",
    16: "#b3b3b3",
    17: "#1b9e77",
    18: "#d95f02",
    19: "#7570b3",
    20: "#e7298a",
    21: "#66a61e",
    22: "#e6ab02",
    23: "#a6761d",
    24: "#c9c9c9",
}

GRAPH_DEFAULT_NODE_COLOR = "#d0d0d0"
GRAPH_EDGE_COLOR = "#9a9a9a"


def feature_name(label_id: Any) -> str:
    try:
        lid = int(label_id)
    except Exception:
        return "unknown"
    if 0 <= lid < len(FACE_CATEGORIES):
        return FACE_CATEGORIES[lid]
    return f"unknown_{lid}"


def feature_color(label_id: Any) -> str:
    try:
        lid = int(label_id)
    except Exception:
        return "#bbbbbb"
    return FEATURE_COLORS.get(lid, "#bbbbbb")


# ============================================================
# 读取 cache
# ============================================================

def load_index() -> Dict[str, Any]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"找不到 index.json: {INDEX_PATH}\n"
            f"请先运行 build_coag_cache.py。"
        )
    with INDEX_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_coag_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


INDEX = load_index()
ITEMS = INDEX.get("items", [])

if len(ITEMS) == 0:
    raise RuntimeError(
        f"index.json 里没有成功样本。请检查 build_coag_cache.py 是否成功生成 cache: {INDEX_PATH}"
    )


# ============================================================
# 3D figure
# ============================================================

def build_3d_figure(data: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()

    face_meshes = data.get("face_meshes", [])

    # 按 feature label 分组，减少 trace 数量
    grouped: Dict[int, Dict[str, Any]] = {}

    for fm in face_meshes:
        label = fm.get("face_label")
        if label is None:
            label = -1
        label = int(label)

        if label not in grouped:
            grouped[label] = {
                "x": [],
                "y": [],
                "z": [],
                "i": [],
                "j": [],
                "k": [],
                "hover": [],
                "offset": 0,
                "face_count": 0,
            }

        g = grouped[label]
        offset = g["offset"]

        vertices = fm.get("vertices", [])
        triangles = fm.get("triangles", [])

        for v in vertices:
            if len(v) >= 3:
                g["x"].append(v[0])
                g["y"].append(v[1])
                g["z"].append(v[2])

        for tri in triangles:
            if len(tri) >= 3:
                g["i"].append(offset + tri[0])
                g["j"].append(offset + tri[1])
                g["k"].append(offset + tri[2])

        hover_text = (
            f"face_id: {fm.get('face_id')}<br>"
            f"feature: {fm.get('face_label')} / {fm.get('face_label_name')}<br>"
            f"instance: {fm.get('face_instance_ids')}<br>"
            f"face_type: {fm.get('face_type')}<br>"
            f"bottom: {fm.get('bottom')}"
        )
        for _ in triangles:
            g["hover"].append(hover_text)

        g["offset"] += len(vertices)
        g["face_count"] += 1

    for label, g in sorted(grouped.items(), key=lambda kv: kv[0]):
        if len(g["x"]) == 0 or len(g["i"]) == 0:
            continue

        name = f"{label}: {feature_name(label)}"

        fig.add_trace(go.Mesh3d(
            x=g["x"],
            y=g["y"],
            z=g["z"],
            i=g["i"],
            j=g["j"],
            k=g["k"],
            name=name,
            color=feature_color(label),
            opacity=1.0,
            flatshading=True,
            hoverinfo="text",
            text=g["hover"],
            showscale=False,
        ))

    fig.update_layout(
        title=f"3D B-Rep colored by feature label — {data.get('part_id')}",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.65)",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


# ============================================================
# graph layout
# ============================================================

def circular_layout(node_ids: List[int]) -> Dict[int, Tuple[float, float]]:
    n = len(node_ids)
    if n == 0:
        return {}
    pos = {}
    for idx, nid in enumerate(node_ids):
        theta = 2.0 * math.pi * idx / n
        pos[nid] = (math.cos(theta), math.sin(theta))
    return pos


def spring_layout_simple(
    node_ids: List[int],
    edges: List[Tuple[int, int]],
    iterations: int = 80,
) -> Dict[int, Tuple[float, float]]:
    """
    不依赖 networkx 的简单 force layout。
    节点数量很大时，也能跑，只是不追求科研画图精度。
    """
    n = len(node_ids)
    if n == 0:
        return {}

    pos = circular_layout(node_ids)
    node_set = set(node_ids)
    edge_list = [(u, v) for u, v in edges if u in node_set and v in node_set]

    area = 4.0
    k = math.sqrt(area / max(n, 1))

    disp = {nid: [0.0, 0.0] for nid in node_ids}

    for it in range(iterations):
        for nid in node_ids:
            disp[nid][0] = 0.0
            disp[nid][1] = 0.0

        # repulsion
        for a_idx in range(n):
            v = node_ids[a_idx]
            vx, vy = pos[v]
            for b_idx in range(a_idx + 1, n):
                u = node_ids[b_idx]
                ux, uy = pos[u]
                dx = vx - ux
                dy = vy - uy
                dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                force = k * k / dist

                fx = dx / dist * force
                fy = dy / dist * force

                disp[v][0] += fx
                disp[v][1] += fy
                disp[u][0] -= fx
                disp[u][1] -= fy

        # attraction
        for v, u in edge_list:
            vx, vy = pos[v]
            ux, uy = pos[u]
            dx = vx - ux
            dy = vy - uy
            dist = math.sqrt(dx * dx + dy * dy) + 1e-6
            force = dist * dist / k

            fx = dx / dist * force
            fy = dy / dist * force

            disp[v][0] -= fx
            disp[v][1] -= fy
            disp[u][0] += fx
            disp[u][1] += fy

        temp = 0.08 * (1.0 - it / max(iterations, 1))

        for nid in node_ids:
            dx, dy = disp[nid]
            length = math.sqrt(dx * dx + dy * dy) + 1e-6
            x, y = pos[nid]
            x += dx / length * min(length, temp)
            y += dy / length * min(length, temp)
            pos[nid] = (x, y)

    return pos


def edge_matches_relation_filter(edge: Dict[str, Any], selected_relations: List[str]) -> bool:
    if not selected_relations:
        return True

    relations = edge.get("relations", {})
    for r in selected_relations:
        if bool(relations.get(r, False)):
            return True
    return False


def build_graph_figure(
    data: Dict[str, Any],
    selected_relations: List[str],
    show_only_internal_nodes: bool,
) -> go.Figure:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if show_only_internal_nodes:
        visible_node_ids = [
            int(n["id"]) for n in nodes
            if bool(n.get("is_feature_internal", False))
        ]
        visible_node_set = set(visible_node_ids)
    else:
        visible_node_ids = [int(n["id"]) for n in nodes]
        visible_node_set = set(visible_node_ids)

    visible_edges = []
    for e in edges:
        u = int(e["u"])
        v = int(e["v"])
        if u not in visible_node_set or v not in visible_node_set:
            continue
        if edge_matches_relation_filter(e, selected_relations):
            visible_edges.append((u, v, e))

    layout_edges = [(u, v) for u, v, _ in visible_edges]
    pos = spring_layout_simple(visible_node_ids, layout_edges)

    edge_x: List[float] = []
    edge_y: List[float] = []
    edge_hover_x: List[float] = []
    edge_hover_y: List[float] = []
    edge_hover_text: List[str] = []

    for u, v, e in visible_edges:
        x0, y0 = pos.get(u, (0, 0))
        x1, y1 = pos.get(v, (0, 0))

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        rels = [
            name for name, val in e.get("relations", {}).items()
            if bool(val)
        ]
        edge_hover_x.append((x0 + x1) / 2.0)
        edge_hover_y.append((y0 + y1) / 2.0)
        edge_hover_text.append(
            f"edge: {u} - {v}<br>"
            f"relations: {', '.join(rels)}<br>"
            f"vector: {e.get('relation_vector')}"
        )

    node_x: List[float] = []
    node_y: List[float] = []
    node_color: List[str] = []
    node_size: List[int] = []
    node_text: List[str] = []

    node_by_id = {int(n["id"]): n for n in nodes}

    for nid in visible_node_ids:
        n = node_by_id[nid]
        x, y = pos.get(nid, (0, 0))

        node_x.append(x)
        node_y.append(y)

        if bool(n.get("is_feature_internal", False)):
            label = n.get("internal_feature_label")
            node_color.append(feature_color(label))
            node_size.append(12)
        else:
            node_color.append(GRAPH_DEFAULT_NODE_COLOR)
            node_size.append(7)

        node_text.append(
            f"coedge id: {nid}<br>"
            f"face id: {n.get('face_id')}<br>"
            f"face feature: {n.get('face_label')} / {n.get('face_label_name')}<br>"
            f"face instance: {n.get('face_instance_ids')}<br>"
            f"feature internal: {n.get('is_feature_internal')}<br>"
            f"internal feature: {n.get('internal_feature_label')} / {n.get('internal_feature_name')}<br>"
            f"internal instance: {n.get('internal_feature_instance_ids')}<br>"
            f"face_type: {n.get('face_type')}<br>"
            f"edge_type: {n.get('edge_type')}<br>"
            f"loop_type: {n.get('loop_type')}<br>"
            f"orientation: {n.get('orientation')}<br>"
            f"edge_id: {n.get('edge_id')}"
        )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color=GRAPH_EDGE_COLOR),
        hoverinfo="skip",
        name="CoAG edges",
    ))

    # 额外放透明 marker 用于 hover edge relation
    fig.add_trace(go.Scatter(
        x=edge_hover_x,
        y=edge_hover_y,
        mode="markers",
        marker=dict(size=8, color="rgba(0,0,0,0)"),
        hoverinfo="text",
        text=edge_hover_text,
        showlegend=False,
        name="edge relation",
    ))

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=0.5, color="#333333"),
        ),
        hoverinfo="text",
        text=node_text,
        name="Coedge nodes",
    ))

    fig.update_layout(
        title=(
            f"CoAG graph — {data.get('part_id')} "
            f"nodes={len(visible_node_ids)}, edges={len(visible_edges)}"
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=45, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    return fig


# ============================================================
# Dash app
# ============================================================

def make_options() -> List[Dict[str, str]]:
    options = []
    for item in ITEMS:
        label = (
            f"{item.get('part_id')} | "
            f"faces={item.get('n_faces')} "
            f"coedges={item.get('n_coedges')} "
            f"edges={item.get('n_graph_edges')}"
        )
        options.append({
            "label": label,
            "value": item.get("coag_json"),
        })
    return options


app: Dash = dash.Dash(__name__)
app.title = APP_TITLE

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "height": "100vh",
        "display": "flex",
        "flexDirection": "column",
        "background": "#f7f7f7",
    },
    children=[
        html.Div(
            style={
                "padding": "10px 14px",
                "background": "#222",
                "color": "white",
                "display": "flex",
                "alignItems": "center",
                "gap": "14px",
            },
            children=[
                html.Div(
                    "CoAG Viewer",
                    style={
                        "fontSize": "20px",
                        "fontWeight": "bold",
                        "whiteSpace": "nowrap",
                    },
                ),
                html.Div(
                    style={"flex": "1"},
                    children=[
                        dcc.Dropdown(
                            id="part-dropdown",
                            options=make_options(),
                            value=ITEMS[0].get("coag_json"),
                            clearable=False,
                            style={
                                "color": "#222",
                                "width": "100%",
                            },
                        )
                    ],
                ),
            ],
        ),

        html.Div(
            style={
                "padding": "8px 12px",
                "background": "white",
                "borderBottom": "1px solid #ddd",
                "display": "flex",
                "alignItems": "center",
                "gap": "20px",
                "flexWrap": "wrap",
            },
            children=[
                html.Div(
                    children=[
                        html.Span("Graph relation filter: ", style={"fontWeight": "bold"}),
                        dcc.Checklist(
                            id="relation-checklist",
                            options=[
                                {"label": r, "value": r}
                                for r in RELATION_ORDER
                            ],
                            value=RELATION_ORDER,
                            inline=True,
                            style={"display": "inline-block"},
                            inputStyle={"marginLeft": "10px", "marginRight": "4px"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        dcc.Checklist(
                            id="internal-only-check",
                            options=[
                                {
                                    "label": "show only feature-internal coedges",
                                    "value": "internal_only",
                                }
                            ],
                            value=[],
                            inline=True,
                        )
                    ]
                ),
            ],
        ),

        dcc.Store(id="coag-data-store"),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.1fr 0.9fr",
                "gap": "8px",
                "padding": "8px",
                "flex": "1",
                "minHeight": 0,
            },
            children=[
                html.Div(
                    style={
                        "background": "white",
                        "border": "1px solid #ddd",
                        "borderRadius": "8px",
                        "overflow": "hidden",
                        "minHeight": 0,
                    },
                    children=[
                        dcc.Graph(
                            id="brep-3d-graph",
                            style={"height": "calc(100vh - 140px)"},
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                            },
                        )
                    ],
                ),
                html.Div(
                    style={
                        "background": "white",
                        "border": "1px solid #ddd",
                        "borderRadius": "8px",
                        "overflow": "hidden",
                        "minHeight": 0,
                    },
                    children=[
                        dcc.Graph(
                            id="coag-graph",
                            style={"height": "calc(100vh - 140px)"},
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                            },
                        )
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("coag-data-store", "data"),
    Input("part-dropdown", "value"),
)
def load_selected_part(coag_json_path: str) -> Dict[str, Any]:
    return load_coag_json(coag_json_path)


@app.callback(
    Output("brep-3d-graph", "figure"),
    Input("coag-data-store", "data"),
)
def update_3d(data: Dict[str, Any]) -> go.Figure:
    if not data:
        return go.Figure()
    return build_3d_figure(data)


@app.callback(
    Output("coag-graph", "figure"),
    Input("coag-data-store", "data"),
    Input("relation-checklist", "value"),
    Input("internal-only-check", "value"),
)
def update_coag_graph(
    data: Dict[str, Any],
    selected_relations: List[str],
    internal_only_value: List[str],
) -> go.Figure:
    if not data:
        return go.Figure()

    show_only_internal_nodes = "internal_only" in (internal_only_value or [])

    return build_graph_figure(
        data=data,
        selected_relations=selected_relations or [],
        show_only_internal_nodes=show_only_internal_nodes,
    )


if __name__ == "__main__":
    print(f"[INFO] ROOT_DIR       = {ROOT_DIR}")
    print(f"[INFO] COAG_CACHE_DIR = {COAG_CACHE_DIR}")
    print(f"[INFO] INDEX_PATH     = {INDEX_PATH}")
    print(f"[INFO] samples        = {len(ITEMS)}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
