# view_fag_topology_groups.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, List

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, dash_table
from tqdm import tqdm

# pythonocc-core / OpenCascade
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location


# ============================================================
# Config
# 只需要改这里
# ============================================================

CONFIG = {
    # topology group 脚本输出的 latest 文件
    "TOPOLOGY_GROUPS_CSV": Path(
        "output/fusion_fag_topology_groups_cached/topology_groups_latest.csv"
    ),

    "TOPOLOGY_MEMBERS_CSV": Path(
        "output/fusion_fag_topology_groups_cached/topology_group_members_latest.csv"
    ),

    # FAG 拓扑缓存目录
    # 这里要和 grouping 脚本里的 GRAPH_CACHE_DIR 一致
    "GRAPH_CACHE_DIR": Path(
        "output/fusion_fag_topology_groups_cached/fag_cache"
    ),

    # mesh 缓存目录
    "MESH_CACHE_DIR": Path(
        "output/fusion_fag_topology_groups_cached/mesh_cache"
    ),

    # STEP 三角化精度
    "LINEAR_DEFLECTION": 0.15,

    # 启动网页时是否提前缓存所有 topology group 成员的 STEP mesh
    "PREBUILD_CACHE_ON_START": False,

    # 默认最多自动勾选多少个模型
    "DEFAULT_SELECTED_MODELS": 2,

    # FAG 拓扑图 viewer 高度
    "FAG_VIEWER_HEIGHT": 330,

    # Dash 端口
    "HOST": "127.0.0.1",
    "PORT": 8040,
}


# ============================================================
# STEP loading / meshing
# ============================================================

def read_step_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))

    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")

    reader.TransferRoots()
    return reader.OneShape()


def triangle_get_indices(triangle):
    return triangle.Get()


def step_to_mesh_arrays(
    step_path: Path,
    linear_deflection: float = 0.15,
) -> Dict[str, np.ndarray]:
    shape = read_step_shape(step_path)

    BRepMesh_IncrementalMesh(shape, linear_deflection)

    xs, ys, zs = [], [], []
    I, J, K = [], [], []

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)

    while face_exp.More():
        face = face_exp.Current()

        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)

        if triangulation is not None:
            trsf = loc.Transformation()
            base_index = len(xs)

            nb_nodes = triangulation.NbNodes()
            for idx in range(1, nb_nodes + 1):
                p = triangulation.Node(idx).Transformed(trsf)
                xs.append(p.X())
                ys.append(p.Y())
                zs.append(p.Z())

            nb_triangles = triangulation.NbTriangles()
            reversed_face = face.Orientation() == TopAbs_REVERSED

            for tidx in range(1, nb_triangles + 1):
                tri = triangulation.Triangle(tidx)
                n1, n2, n3 = triangle_get_indices(tri)

                if reversed_face:
                    n2, n3 = n3, n2

                I.append(base_index + n1 - 1)
                J.append(base_index + n2 - 1)
                K.append(base_index + n3 - 1)

        face_exp.Next()

    if len(xs) == 0 or len(I) == 0:
        raise RuntimeError(f"No mesh generated from STEP: {step_path}")

    return {
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "z": np.asarray(zs, dtype=np.float64),
        "i": np.asarray(I, dtype=np.int32),
        "j": np.asarray(J, dtype=np.int32),
        "k": np.asarray(K, dtype=np.int32),
    }


# ============================================================
# Mesh cache
# ============================================================

def safe_mesh_cache_name(step_path: Path, linear_deflection: float) -> str:
    text = f"{step_path.resolve()}|deflection={linear_deflection}"
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{step_path.stem}_{h}.npz"


def get_mesh_cache_path(step_path: Path) -> Path:
    return CONFIG["MESH_CACHE_DIR"] / safe_mesh_cache_name(
        step_path,
        CONFIG["LINEAR_DEFLECTION"],
    )


def load_or_build_mesh(step_path: Path) -> Dict[str, np.ndarray]:
    step_path = Path(step_path)
    cache_path = get_mesh_cache_path(step_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        data = np.load(cache_path)
        return {
            "x": data["x"],
            "y": data["y"],
            "z": data["z"],
            "i": data["i"],
            "j": data["j"],
            "k": data["k"],
        }

    mesh = step_to_mesh_arrays(
        step_path,
        linear_deflection=CONFIG["LINEAR_DEFLECTION"],
    )

    np.savez_compressed(
        cache_path,
        x=mesh["x"],
        y=mesh["y"],
        z=mesh["z"],
        i=mesh["i"],
        j=mesh["j"],
        k=mesh["k"],
    )

    return mesh


def prebuild_cache_for_members(members_df: pd.DataFrame):
    paths = [Path(p) for p in members_df["path"].dropna().tolist()]
    unique_paths = sorted(set(paths), key=lambda p: str(p))

    print(f"[INFO] Prebuilding mesh cache for {len(unique_paths)} STEP files...")

    failed = []

    for step_path in tqdm(unique_paths, desc="Prebuilding mesh cache"):
        try:
            load_or_build_mesh(step_path)
        except Exception as e:
            failed.append(
                {
                    "path": str(step_path),
                    "error": repr(e),
                }
            )

    if failed:
        failed_path = CONFIG["MESH_CACHE_DIR"] / "failed_mesh_cache.json"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text(
            json.dumps(failed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[WARN] Mesh cache failed: {len(failed)}. See {failed_path}")

    print("[DONE] Mesh cache prebuild finished.")


# ============================================================
# FAG graph cache / drawing
# ============================================================

def safe_graph_cache_name(step_path: Path) -> str:
    """
    必须和 grouping 脚本里的 make_graph_cache_name() 保持一致：
      h = md5(str(step_path.resolve()))
      filename = f"{step_path.stem}_{h}.json"
    """
    abs_path = str(step_path.resolve())
    h = hashlib.md5(abs_path.encode("utf-8")).hexdigest()
    return f"{step_path.stem}_{h}.json"


def get_graph_cache_path(step_path: Path) -> Path:
    return CONFIG["GRAPH_CACHE_DIR"] / safe_graph_cache_name(step_path)


def load_fag_from_cache(step_path: Path) -> nx.Graph:
    cache_path = get_graph_cache_path(step_path)

    if not cache_path.exists():
        raise FileNotFoundError(f"FAG cache not found: {cache_path}")

    data = json.loads(cache_path.read_text(encoding="utf-8"))

    G = nx.Graph()

    for n in data["nodes"]:
        G.add_node(int(n))

    for e in data["edges"]:
        G.add_edge(
            int(e["u"]),
            int(e["v"]),
            shared_edges=int(e.get("shared_edges", 1)),
        )

    G.graph["step_path"] = data.get("step_path", str(step_path))
    G.graph["n_faces"] = G.number_of_nodes()
    G.graph["n_adjacencies"] = G.number_of_edges()

    return G


def make_fag_figure(
    G: nx.Graph,
    title: str = "FAG topology",
    height: int = 330,
) -> go.Figure:
    """
    用 Plotly 画 FAG。
    这个 figure 支持：
      - scrollZoom=True 时滚轮缩放
      - dragmode='pan' 时拖拽平移
    """
    if G.number_of_nodes() == 0:
        return make_error_figure("Empty FAG graph.", height=height)

    # 固定 seed，保证同一个 topology 每次布局稳定
    pos = nx.spring_layout(G, seed=42, k=None)

    edge_x = []
    edge_y = []
    edge_hover_x = []
    edge_hover_y = []
    edge_hover_text = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        edge_hover_x.append((x0 + x1) / 2)
        edge_hover_y.append((y0 + y1) / 2)
        edge_hover_text.append(
            f"face {u} - face {v}<br>"
            f"shared_edges={data.get('shared_edges', 1)}"
        )

    node_x = []
    node_y = []
    node_text = []
    node_degree = []

    degrees = dict(G.degree())

    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_degree.append(degrees[n])
        node_text.append(
            f"face {n}<br>"
            f"degree={degrees[n]}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # edge hover invisible markers
    fig.add_trace(
        go.Scatter(
            x=edge_hover_x,
            y=edge_hover_y,
            mode="markers",
            marker=dict(size=8, opacity=0),
            text=edge_hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(n) for n in G.nodes()],
            textposition="middle center",
            textfont=dict(size=10),
            marker=dict(
                size=[max(18, 12 + d * 2) for d in node_degree],
                line=dict(width=1),
            ),
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=0, r=0, t=45, b=0),
        dragmode="pan",
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False,
        ),
    )

    return fig


# ============================================================
# Plotly 3D figure
# ============================================================

def make_mesh_figure(
    mesh: Dict[str, np.ndarray],
    title: str,
    step_path: Optional[Path] = None,
    height: int = 430,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=mesh["x"],
            y=mesh["y"],
            z=mesh["z"],
            i=mesh["i"],
            j=mesh["j"],
            k=mesh["k"],
            opacity=1.0,
            flatshading=True,
            lighting=dict(
                ambient=0.45,
                diffuse=0.75,
                specular=0.2,
                roughness=0.8,
                fresnel=0.1,
            ),
            lightposition=dict(x=100, y=200, z=300),
            hoverinfo="skip",
        )
    )

    subtitle = ""
    if step_path is not None:
        subtitle = f"<br><span style='font-size:12px'>{step_path.name}</span>"

    fig.update_layout(
        title=f"{title}{subtitle}",
        height=height,
        margin=dict(l=0, r=0, t=58, b=0),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        showlegend=False,
    )

    return fig


def make_error_figure(message: str, height: int = 430) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="Failed to load",
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12),
            )
        ],
        height=height,
        margin=dict(l=0, r=0, t=58, b=0),
    )
    return fig


# ============================================================
# Load topology CSVs
# ============================================================

def load_topology_groups_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Topology groups CSV not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"topology_id", "num_samples", "n_faces", "n_adjacencies"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Topology groups CSV missing columns: {missing}")

    df = df.copy()

    if "topology_mode" not in df.columns:
        df["topology_mode"] = ""

    return df


def load_topology_members_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Topology members CSV not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"topology_id", "sample_index", "file_name", "path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Topology members CSV missing columns: {missing}")

    df = df.copy()

    if "n_faces" not in df.columns:
        df["n_faces"] = ""
    if "n_adjacencies" not in df.columns:
        df["n_adjacencies"] = ""
    if "topology_mode" not in df.columns:
        df["topology_mode"] = ""
    if "graph_source" not in df.columns:
        df["graph_source"] = ""

    return df


GROUPS_DF = load_topology_groups_csv(CONFIG["TOPOLOGY_GROUPS_CSV"])
MEMBERS_DF = load_topology_members_csv(CONFIG["TOPOLOGY_MEMBERS_CSV"])

if CONFIG["PREBUILD_CACHE_ON_START"]:
    prebuild_cache_for_members(MEMBERS_DF)


# ============================================================
# Dash app helpers
# ============================================================

app = Dash(__name__)
app.title = "FAG Topology Group Viewer"


def make_topology_table_data(df: pd.DataFrame):
    cols = [
        "topology_id",
        "num_samples",
        "n_faces",
        "n_adjacencies",
        "topology_mode",
    ]
    return df[cols].to_dict("records")


def get_members_for_topology(topology_id: str) -> pd.DataFrame:
    sub = MEMBERS_DF[MEMBERS_DF["topology_id"] == topology_id].copy()
    sub = sub.sort_values(["sample_index", "file_name"])
    return sub


def make_member_options(members: pd.DataFrame):
    options = []

    for _, row in members.iterrows():
        label = (
            f"[{int(row['sample_index'])}] "
            f"{row['file_name']} "
            f"(faces={row.get('n_faces', '')}, adj={row.get('n_adjacencies', '')})"
        )

        options.append(
            {
                "label": label,
                "value": str(row["path"]),
            }
        )

    return options


def make_empty_viewer_message(message: str):
    return html.Div(
        message,
        style={
            "height": "calc(100vh - 120px)",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontSize": "18px",
            "color": "#777",
            "backgroundColor": "#ffffff",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
        },
    )


def make_viewer_card(
    step_path: Path,
    index: int,
    total: int,
):
    title = f"Model {index}/{total}"

    try:
        mesh = load_or_build_mesh(step_path)
        fig = make_mesh_figure(mesh, title, step_path)
    except Exception as e:
        fig = make_error_figure(f"{step_path}\n\n{repr(e)}")

    return html.Div(
        style={
            "backgroundColor": "#ffffff",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "overflow": "hidden",
            "minHeight": "430px",
        },
        children=[
            dcc.Graph(
                figure=fig,
                style={"height": "100%"},
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )
        ],
    )


def grid_template_for_count(n: int) -> str:
    if n <= 1:
        return "1fr"
    if n == 2:
        return "1fr 1fr"
    return "1fr 1fr 1fr"


# ============================================================
# Layout
# ============================================================

app.layout = html.Div(
    style={
        "height": "100vh",
        "display": "flex",
        "flexDirection": "row",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f5f5f5",
    },
    children=[
        # Left panel: topology list
        html.Div(
            style={
                "width": "430px",
                "padding": "12px",
                "boxSizing": "border-box",
                "backgroundColor": "#ffffff",
                "borderRight": "1px solid #ddd",
                "overflowY": "auto",
            },
            children=[
                html.H2(
                    "拓扑类型列表",
                    style={
                        "margin": "0 0 8px 0",
                        "fontSize": "22px",
                    },
                ),
                html.Div(
                    f"Groups CSV: {CONFIG['TOPOLOGY_GROUPS_CSV']}",
                    style={
                        "fontSize": "12px",
                        "color": "#666",
                        "wordBreak": "break-all",
                        "marginBottom": "6px",
                    },
                ),
                html.Div(
                    f"Members CSV: {CONFIG['TOPOLOGY_MEMBERS_CSV']}",
                    style={
                        "fontSize": "12px",
                        "color": "#666",
                        "wordBreak": "break-all",
                        "marginBottom": "10px",
                    },
                ),
                html.Div(
                    f"Topology count: {len(GROUPS_DF)}",
                    style={
                        "fontSize": "14px",
                        "marginBottom": "12px",
                    },
                ),
                html.Button(
                    "Build mesh cache for selected topology",
                    id="build-cache-selected-button",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "8px",
                        "marginBottom": "8px",
                        "cursor": "pointer",
                    },
                ),
                html.Button(
                    "Build mesh cache for all topology members",
                    id="build-cache-all-button",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "8px",
                        "marginBottom": "8px",
                        "cursor": "pointer",
                    },
                ),
                html.Div(
                    id="cache-status",
                    style={
                        "fontSize": "12px",
                        "color": "#555",
                        "marginBottom": "12px",
                        "whiteSpace": "pre-wrap",
                    },
                ),
                dash_table.DataTable(
                    id="topology-table",
                    data=make_topology_table_data(GROUPS_DF),
                    columns=[
                        {"name": "topology", "id": "topology_id"},
                        {"name": "samples", "id": "num_samples"},
                        {"name": "faces", "id": "n_faces"},
                        {"name": "adj", "id": "n_adjacencies"},
                        {"name": "mode", "id": "topology_mode"},
                    ],
                    page_size=20,
                    row_selectable="single",
                    selected_rows=[0] if len(GROUPS_DF) > 0 else [],
                    sort_action="native",
                    filter_action="native",
                    style_table={
                        "overflowX": "auto",
                    },
                    style_cell={
                        "fontSize": "12px",
                        "padding": "6px",
                        "textAlign": "left",
                        "whiteSpace": "normal",
                        "height": "auto",
                        "maxWidth": "150px",
                    },
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": "#f0f0f0",
                    },
                    style_data_conditional=[
                        {
                            "if": {"state": "selected"},
                            "backgroundColor": "#d8ecff",
                            "border": "1px solid #74a9d8",
                        }
                    ],
                ),
                html.Div(
                    id="topology-detail",
                    style={
                        "marginTop": "12px",
                        "fontSize": "12px",
                        "whiteSpace": "pre-wrap",
                        "wordBreak": "break-all",
                        "backgroundColor": "#f7f7f7",
                        "padding": "8px",
                        "border": "1px solid #e0e0e0",
                    },
                ),
            ],
        ),

        # Middle panel: FAG viewer + member selection
        html.Div(
            style={
                "width": "390px",
                "padding": "12px",
                "boxSizing": "border-box",
                "backgroundColor": "#fafafa",
                "borderRight": "1px solid #ddd",
                "overflowY": "auto",
            },
            children=[
                html.H2(
                    "FAG 拓扑图",
                    style={
                        "margin": "0 0 8px 0",
                        "fontSize": "20px",
                    },
                ),
                html.Div(
                    style={
                        "backgroundColor": "#ffffff",
                        "border": "1px solid #ddd",
                        "borderRadius": "6px",
                        "overflow": "hidden",
                        "marginBottom": "12px",
                    },
                    children=[
                        dcc.Graph(
                            id="fag-graph",
                            style={"height": f"{CONFIG['FAG_VIEWER_HEIGHT']}px"},
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                                "doubleClick": "reset",
                                "modeBarButtonsToRemove": [
                                    "select2d",
                                    "lasso2d",
                                ],
                            },
                        )
                    ],
                ),
                html.H2(
                    "该拓扑下的样本",
                    style={
                        "margin": "0 0 8px 0",
                        "fontSize": "20px",
                    },
                ),
                html.Div(
                    id="member-count-info",
                    style={
                        "fontSize": "13px",
                        "marginBottom": "8px",
                        "color": "#555",
                    },
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "8px",
                        "marginBottom": "8px",
                    },
                    children=[
                        html.Button(
                            "Select all",
                            id="select-all-button",
                            n_clicks=0,
                            style={
                                "padding": "8px",
                                "cursor": "pointer",
                            },
                        ),
                        html.Button(
                            "Clear",
                            id="clear-selection-button",
                            n_clicks=0,
                            style={
                                "padding": "8px",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                ),
                dcc.Checklist(
                    id="member-checklist",
                    options=[],
                    value=[],
                    labelStyle={
                        "display": "block",
                        "fontSize": "12px",
                        "marginBottom": "8px",
                        "lineHeight": "1.35",
                        "cursor": "pointer",
                    },
                    inputStyle={
                        "marginRight": "6px",
                    },
                    style={
                        "backgroundColor": "#ffffff",
                        "border": "1px solid #ddd",
                        "borderRadius": "6px",
                        "padding": "8px",
                    },
                ),
            ],
        ),

        # Right panel: dynamic 3D viewers
        html.Div(
            style={
                "flex": "1",
                "padding": "12px",
                "boxSizing": "border-box",
                "overflow": "hidden",
            },
            children=[
                html.Div(
                    id="viewer-title",
                    style={
                        "fontSize": "20px",
                        "fontWeight": "bold",
                        "marginBottom": "8px",
                    },
                ),
                dcc.Loading(
                    type="default",
                    children=[
                        html.Div(
                            id="viewer-grid",
                            style={
                                "height": "calc(100vh - 64px)",
                                "overflowY": "auto",
                            },
                        )
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================
# Callbacks
# ============================================================

@app.callback(
    Output("member-checklist", "options"),
    Output("member-checklist", "value"),
    Output("member-count-info", "children"),
    Output("topology-detail", "children"),
    Output("fag-graph", "figure"),
    Input("topology-table", "selected_rows"),
)
def update_members_and_fag_for_topology(selected_rows):
    if not selected_rows:
        return (
            [],
            [],
            "No topology selected.",
            "",
            make_error_figure("No topology selected.", height=CONFIG["FAG_VIEWER_HEIGHT"]),
        )

    row_idx = selected_rows[0]
    if row_idx >= len(GROUPS_DF):
        return (
            [],
            [],
            "Invalid topology selection.",
            "",
            make_error_figure("Invalid topology selection.", height=CONFIG["FAG_VIEWER_HEIGHT"]),
        )

    topo_row = GROUPS_DF.iloc[row_idx]
    topology_id = topo_row["topology_id"]

    members = get_members_for_topology(topology_id)
    options = make_member_options(members)

    default_n = min(CONFIG["DEFAULT_SELECTED_MODELS"], len(options))
    default_values = [opt["value"] for opt in options[:default_n]]

    detail = (
        f"topology_id: {topology_id}\n"
        f"num_samples: {topo_row.get('num_samples', '')}\n"
        f"n_faces: {topo_row.get('n_faces', '')}\n"
        f"n_adjacencies: {topo_row.get('n_adjacencies', '')}\n"
        f"topology_mode: {topo_row.get('topology_mode', '')}\n"
    )

    if "sample_indices" in topo_row.index:
        detail += f"sample_indices: {topo_row.get('sample_indices', '')}\n"

    info = f"{topology_id}: {len(options)} samples. Select models to display."

    # 用该 topology 的第一个成员作为代表，读取它的 FAG cache 来画拓扑
    if len(members) > 0:
        representative_path = Path(members.iloc[0]["path"])
        try:
            G = load_fag_from_cache(representative_path)
            fag_fig = make_fag_figure(
                G,
                title=(
                    f"{topology_id} FAG "
                    f"(faces={G.number_of_nodes()}, adj={G.number_of_edges()})"
                ),
                height=CONFIG["FAG_VIEWER_HEIGHT"],
            )
        except Exception as e:
            fag_fig = make_error_figure(
                f"Failed to load FAG cache:\n{representative_path}\n\n{repr(e)}",
                height=CONFIG["FAG_VIEWER_HEIGHT"],
            )
    else:
        fag_fig = make_error_figure(
            "No members in this topology.",
            height=CONFIG["FAG_VIEWER_HEIGHT"],
        )

    return options, default_values, info, detail, fag_fig


@app.callback(
    Output("member-checklist", "value", allow_duplicate=True),
    Input("select-all-button", "n_clicks"),
    Input("clear-selection-button", "n_clicks"),
    State("member-checklist", "options"),
    prevent_initial_call=True,
)
def select_or_clear_members(select_all_clicks, clear_clicks, options):
    from dash import callback_context

    if not callback_context.triggered:
        return []

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "select-all-button":
        return [opt["value"] for opt in options]

    if trigger_id == "clear-selection-button":
        return []

    return []


@app.callback(
    Output("viewer-title", "children"),
    Output("viewer-grid", "children"),
    Output("viewer-grid", "style"),
    Input("topology-table", "selected_rows"),
    Input("member-checklist", "value"),
)
def update_dynamic_viewers(selected_rows, selected_member_paths):
    base_style = {
        "height": "calc(100vh - 64px)",
        "overflowY": "auto",
    }

    if not selected_rows:
        return (
            "请选择左侧一个 topology",
            make_empty_viewer_message("No topology selected."),
            base_style,
        )

    row_idx = selected_rows[0]
    if row_idx >= len(GROUPS_DF):
        return (
            "Invalid topology selection",
            make_empty_viewer_message("Invalid topology selection."),
            base_style,
        )

    topo_row = GROUPS_DF.iloc[row_idx]
    topology_id = topo_row["topology_id"]

    if not selected_member_paths:
        return (
            f"{topology_id} | no model selected",
            make_empty_viewer_message("No model selected. Use the checklist to select models."),
            base_style,
        )

    selected_paths = [Path(p) for p in selected_member_paths]
    n = len(selected_paths)

    grid_style = {
        **base_style,
        "display": "grid",
        "gridTemplateColumns": grid_template_for_count(n),
        "gap": "12px",
        "alignContent": "start",
    }

    cards = [
        make_viewer_card(
            step_path=step_path,
            index=i,
            total=n,
        )
        for i, step_path in enumerate(selected_paths, start=1)
    ]

    title = (
        f"{topology_id} | selected {n} / "
        f"{int(topo_row.get('num_samples', n))} models | "
        f"faces={topo_row.get('n_faces', '')}, "
        f"adj={topo_row.get('n_adjacencies', '')}, "
        f"mode={topo_row.get('topology_mode', '')}"
    )

    return title, cards, grid_style


@app.callback(
    Output("cache-status", "children"),
    Input("build-cache-selected-button", "n_clicks"),
    Input("build-cache-all-button", "n_clicks"),
    State("topology-table", "selected_rows"),
    prevent_initial_call=True,
)
def build_cache_buttons(selected_clicks, all_clicks, selected_rows):
    from dash import callback_context

    if not callback_context.triggered:
        return ""

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "build-cache-selected-button":
        if not selected_rows:
            return "No topology selected."

        row_idx = selected_rows[0]
        if row_idx >= len(GROUPS_DF):
            return "Invalid topology selection."

        topology_id = GROUPS_DF.iloc[row_idx]["topology_id"]
        members = get_members_for_topology(topology_id)
        paths = [Path(p) for p in members["path"].dropna().tolist()]
        label = f"selected topology {topology_id}"

    else:
        paths = [Path(p) for p in MEMBERS_DF["path"].dropna().tolist()]
        label = "all topology members"

    unique_paths = sorted(set(paths), key=lambda p: str(p))

    failed = []
    built = 0
    existed = 0

    for step_path in unique_paths:
        try:
            cache_path = get_mesh_cache_path(step_path)
            if cache_path.exists():
                existed += 1
            else:
                load_or_build_mesh(step_path)
                built += 1
        except Exception as e:
            failed.append(
                {
                    "path": str(step_path),
                    "error": repr(e),
                }
            )

    if failed:
        failed_path = CONFIG["MESH_CACHE_DIR"] / "failed_mesh_cache_from_web.json"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text(
            json.dumps(failed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return (
            f"Cache finished for {label} with failures.\n"
            f"Existing: {existed}\n"
            f"Newly built: {built}\n"
            f"Failed: {len(failed)}\n"
            f"Failure log: {failed_path}"
        )

    return (
        f"Cache finished for {label}.\n"
        f"Existing: {existed}\n"
        f"Newly built: {built}\n"
        f"Failed: 0"
    )


# ============================================================
# Main
# ============================================================

def main():
    print(f"[INFO] Groups CSV: {CONFIG['TOPOLOGY_GROUPS_CSV']}")
    print(f"[INFO] Members CSV: {CONFIG['TOPOLOGY_MEMBERS_CSV']}")
    print(f"[INFO] Graph cache dir: {CONFIG['GRAPH_CACHE_DIR']}")
    print(f"[INFO] Mesh cache dir: {CONFIG['MESH_CACHE_DIR']}")
    print(f"[INFO] Topology count: {len(GROUPS_DF)}")
    print(f"[INFO] Member count: {len(MEMBERS_DF)}")
    print(f"[INFO] Open: http://{CONFIG['HOST']}:{CONFIG['PORT']}")

    app.run(
        host=CONFIG["HOST"],
        port=CONFIG["PORT"],
        debug=True,
    )


if __name__ == "__main__":
    main()