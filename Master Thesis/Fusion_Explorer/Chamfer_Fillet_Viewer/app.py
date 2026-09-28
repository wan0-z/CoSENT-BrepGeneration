import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update


# ============================================================
# 0. 你只需要改这里
# ============================================================

CONFIG = {
    # 这里填你的缓存根目录
    # 下面应该有 compare_mesh_cache/manifest.json
    "CACHE_ROOT": r"E:\fusion_cache_50",

    "HOST": "127.0.0.1",
    "PORT": 8050,
    "DEBUG": True,

    # True：一个 B-Rep face 一个 trace，hover 能看到 face index 和 surface type，但更慢
    # False：一个 feature 一个 trace，更快，但 hover 不能精确到 face index
    "ONE_TRACE_PER_FACE": True,
}


DEFAULT_FEATURE_COLORS = {
    "Chamfer": "#39FF14",
    "Fillet": "#FFFF33",
    "ExtrudeSide": "#1f77b4",
    "ExtrudeEnd": "#ff7f0e",
    "CutSide": "#d62728",
    "CutEnd": "#9467bd",
    "RevolveSide": "#17becf",
    "RevolveEnd": "#e377c2",
    "RecoveredPatch": "#BDBDBD",
    "Unknown": "#808080",
}

DEFAULT_SURFACE_TYPE_NAMES = {
    -1: "UnknownSurface",
}


# ============================================================
# 1. 读取 manifest / mesh
# ============================================================

def load_manifest(cache_root: Path):
    manifest_path = cache_root / "compare_mesh_cache" / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cannot find manifest.json: {manifest_path}\n"
            f"请先运行 prepare_defeature_compare_cache.py"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz_mesh(cache_root: Path, mesh_rel_path: str):
    path = cache_root / mesh_rel_path

    if not path.exists():
        raise FileNotFoundError(f"Cannot find mesh: {path}")

    data = np.load(path)

    vertices = data["vertices"]
    triangles = data["triangles"]
    triangle_face = data["triangle_face"]
    face_label = data["face_label"]

    # 兼容旧 cache：如果没有 face_surface_type，就全部标成 -1
    if "face_surface_type" in data:
        face_surface_type = data["face_surface_type"]
    else:
        face_surface_type = np.full((len(face_label),), -1, dtype=np.int32)

    return (
        vertices,
        triangles,
        triangle_face,
        face_label,
        face_surface_type,
    )


def build_label_map(manifest):
    segment_names = manifest.get("segment_names", {})

    if isinstance(segment_names, list):
        return {i: str(v) for i, v in enumerate(segment_names)}

    if isinstance(segment_names, dict):
        return {int(k): str(v) for k, v in segment_names.items()}

    return {
        0: "ExtrudeSide",
        1: "ExtrudeEnd",
        2: "CutSide",
        3: "CutEnd",
        4: "Fillet",
        5: "Chamfer",
        6: "RevolveSide",
        7: "RevolveEnd",
        8: "RecoveredPatch",
    }


def build_surface_type_map(manifest):
    surface_type_names = manifest.get("surface_type_names", {})

    if isinstance(surface_type_names, dict):
        result = {int(k): str(v) for k, v in surface_type_names.items()}
        result[-1] = "UnknownSurface"
        return result

    return DEFAULT_SURFACE_TYPE_NAMES


# ============================================================
# 2. mesh 工具
# ============================================================

def make_submesh(vertices, triangles, tri_indices):
    sub_tri_global = triangles[tri_indices]

    used_vertices, inverse = np.unique(
        sub_tri_global.reshape(-1),
        return_inverse=True,
    )

    sub_vertices = vertices[used_vertices]
    sub_tri = inverse.reshape((-1, 3))

    x = sub_vertices[:, 0]
    y = sub_vertices[:, 1]
    z = sub_vertices[:, 2]

    i = sub_tri[:, 0]
    j = sub_tri[:, 1]
    k = sub_tri[:, 2]

    return x, y, z, i, j, k


# ============================================================
# 3. figure：一个 face 一个 trace
# ============================================================

def make_figure_one_trace_per_face(
    title,
    vertices,
    triangles,
    triangle_face,
    face_label,
    face_surface_type,
    label_map,
    surface_type_map,
    feature_colors,
):
    traces = []

    for face_idx in range(len(face_label)):
        tri_indices = np.where(triangle_face == face_idx)[0]

        if len(tri_indices) == 0:
            continue

        label_idx = int(face_label[face_idx])
        feature = label_map.get(label_idx, "Unknown")
        color = feature_colors.get(
            feature,
            DEFAULT_FEATURE_COLORS.get(feature, "#808080"),
        )

        surface_type_id = int(face_surface_type[face_idx])
        surface_type_name = surface_type_map.get(
            surface_type_id,
            f"UnknownSurfaceType_{surface_type_id}",
        )

        x, y, z, i, j, k = make_submesh(vertices, triangles, tri_indices)

        hover_text = (
            f"<b>Feature:</b> {feature}<br>"
            f"<b>Surface type:</b> {surface_type_name}<br>"
            f"<b>Face index:</b> {face_idx}<br>"
            f"<b>Label index:</b> {label_idx}<br>"
            f"<b>Surface type id:</b> {surface_type_id}<br>"
            f"<b>Triangles:</b> {len(tri_indices)}"
        )

        traces.append(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=color,
                name=f"Face {face_idx}: {feature} | {surface_type_name}",
                opacity=1.0,
                flatshading=True,
                hovertemplate=hover_text + "<extra></extra>",
                showscale=False,
                showlegend=False,
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.75,
                    specular=0.25,
                    roughness=0.8,
                    fresnel=0.1,
                ),
            )
        )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
    )

    return fig


# ============================================================
# 4. figure：一个 feature 一个 trace，更快
# ============================================================

def make_figure_one_trace_per_feature(
    title,
    vertices,
    triangles,
    triangle_face,
    face_label,
    face_surface_type,
    label_map,
    surface_type_map,
    feature_colors,
):
    traces = []

    triangle_labels = face_label[triangle_face]
    unique_labels = sorted(set(int(x) for x in triangle_labels))

    for label_idx in unique_labels:
        feature = label_map.get(label_idx, "Unknown")
        color = feature_colors.get(
            feature,
            DEFAULT_FEATURE_COLORS.get(feature, "#808080"),
        )

        tri_indices = np.where(triangle_labels == label_idx)[0]

        if len(tri_indices) == 0:
            continue

        face_indices = np.unique(triangle_face[tri_indices])
        surface_ids = sorted(set(int(face_surface_type[f]) for f in face_indices))
        surface_names = [
            surface_type_map.get(sid, f"UnknownSurfaceType_{sid}")
            for sid in surface_ids
        ]
        surface_text = ", ".join(surface_names)

        x, y, z, i, j, k = make_submesh(vertices, triangles, tri_indices)

        hover_text = (
            f"<b>Feature:</b> {feature}<br>"
            f"<b>Label index:</b> {label_idx}<br>"
            f"<b>Surface types:</b> {surface_text}<br>"
            f"<b>Faces:</b> {len(face_indices)}<br>"
            f"<b>Triangles:</b> {len(tri_indices)}"
        )

        traces.append(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=color,
                name=f"{feature} | {surface_text}",
                opacity=1.0,
                flatshading=True,
                hovertemplate=hover_text + "<extra></extra>",
                showscale=False,
                showlegend=True,
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.75,
                    specular=0.25,
                    roughness=0.8,
                    fresnel=0.1,
                ),
            )
        )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )

    return fig


def make_figure(title, cache_root, mesh_rel_path, manifest):
    label_map = build_label_map(manifest)
    surface_type_map = build_surface_type_map(manifest)
    feature_colors = manifest.get("feature_colors", DEFAULT_FEATURE_COLORS)

    vertices, triangles, triangle_face, face_label, face_surface_type = load_npz_mesh(
        cache_root,
        mesh_rel_path,
    )

    if CONFIG["ONE_TRACE_PER_FACE"]:
        return make_figure_one_trace_per_face(
            title,
            vertices,
            triangles,
            triangle_face,
            face_label,
            face_surface_type,
            label_map,
            surface_type_map,
            feature_colors,
        )

    return make_figure_one_trace_per_feature(
        title,
        vertices,
        triangles,
        triangle_face,
        face_label,
        face_surface_type,
        label_map,
        surface_type_map,
        feature_colors,
    )


# ============================================================
# 5. UI 辅助函数
# ============================================================

def make_legend(feature_colors):
    ordered = [
        "ExtrudeSide",
        "ExtrudeEnd",
        "CutSide",
        "CutEnd",
        "Fillet",
        "Chamfer",
        "RevolveSide",
        "RevolveEnd",
        "RecoveredPatch",
    ]

    rows = []

    for name in ordered:
        color = feature_colors.get(name, DEFAULT_FEATURE_COLORS.get(name, "#808080"))

        rows.append(
            html.Div(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "16px",
                            "height": "16px",
                            "backgroundColor": color,
                            "border": "1px solid #333",
                            "marginRight": "8px",
                        }
                    ),
                    html.Span(name),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "7px",
                    "fontSize": "14px",
                },
            )
        )

    return html.Div(rows)


def format_feature_count(feature_count):
    if not feature_count:
        return "No feature count."

    ordered = [
        "ExtrudeSide",
        "ExtrudeEnd",
        "CutSide",
        "CutEnd",
        "Fillet",
        "Chamfer",
        "RevolveSide",
        "RevolveEnd",
        "RecoveredPatch",
        "Unknown",
    ]

    parts = []

    for name in ordered:
        if name in feature_count:
            parts.append(f"{name}: {feature_count[name]}")

    for name, value in feature_count.items():
        if name not in ordered:
            parts.append(f"{name}: {value}")

    return " | ".join(parts)


def format_simple_count_dict(d):
    if not d:
        return "None"

    return " | ".join(f"{k}: {v}" for k, v in sorted(d.items()))


def format_nested_count_dict(d):
    if not d:
        return html.Div("None")

    blocks = []

    for feature_name, sub in sorted(d.items()):
        if not isinstance(sub, dict):
            continue

        inner = ", ".join(
            f"{surface}: {count}"
            for surface, count in sorted(sub.items())
        )
        blocks.append(html.Div(f"{feature_name} -> {inner}"))

    if not blocks:
        return html.Div("None")

    return html.Div(blocks)


# ============================================================
# 6. Dash app
# ============================================================

def create_app():
    app = Dash(__name__)

    cache_root = Path(CONFIG["CACHE_ROOT"]).expanduser().resolve()

    try:
        manifest = load_manifest(cache_root)
        feature_colors = manifest.get("feature_colors", DEFAULT_FEATURE_COLORS)
    except Exception:
        manifest = {
            "items": [],
            "failures": [],
            "feature_colors": DEFAULT_FEATURE_COLORS,
        }
        feature_colors = DEFAULT_FEATURE_COLORS

    app.layout = html.Div(
        style={
            "fontFamily": "Arial, sans-serif",
            "padding": "16px",
            "backgroundColor": "#f5f5f5",
        },
        children=[
            html.H2("Fillet / Chamfer Defeaturing Comparison with Surface Types"),

            html.Div(
                [
                    html.Div(f"Cache root: {cache_root}"),
                    html.Div(
                        f"Mode: {'one trace per face' if CONFIG['ONE_TRACE_PER_FACE'] else 'one trace per feature'}"
                    ),
                    html.Button(
                        "Load comparison cache",
                        id="load-cache-button",
                        n_clicks=0,
                        style={
                            "marginTop": "8px",
                            "padding": "7px 14px",
                        },
                    ),
                ],
                style={
                    "marginBottom": "12px",
                    "padding": "10px",
                    "backgroundColor": "white",
                    "border": "1px solid #ddd",
                },
            ),

            html.Div(
                [
                    html.Label("Model:"),
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=[],
                        value=None,
                        placeholder="Select model",
                        style={
                            "width": "75%",
                            "display": "inline-block",
                            "marginLeft": "12px",
                        },
                    ),
                ],
                style={"marginBottom": "12px"},
            ),

            html.Div(
                id="status",
                style={
                    "marginBottom": "10px",
                    "color": "#333",
                    "fontSize": "14px",
                },
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Before: original B-Rep"),
                            dcc.Graph(
                                id="before-graph",
                                figure=go.Figure(),
                                style={
                                    "height": "680px",
                                    "backgroundColor": "white",
                                    "border": "1px solid #ddd",
                                },
                                config={
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "responsive": True,
                                },
                            ),
                        ],
                        style={
                            "width": "49%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                        },
                    ),

                    html.Div(
                        [
                            html.H3("After: fillet/chamfer removed"),
                            dcc.Graph(
                                id="after-graph",
                                figure=go.Figure(),
                                style={
                                    "height": "680px",
                                    "backgroundColor": "white",
                                    "border": "1px solid #ddd",
                                },
                                config={
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "responsive": True,
                                },
                            ),
                        ],
                        style={
                            "width": "49%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "marginLeft": "1%",
                        },
                    ),
                ],
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Feature colors"),
                            make_legend(feature_colors),
                        ],
                        style={
                            "width": "24%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "10px",
                            "backgroundColor": "white",
                            "border": "1px solid #ddd",
                            "marginTop": "12px",
                        },
                    ),

                    html.Div(
                        [
                            html.H4("Model info"),
                            html.Div(id="model-info", children="No model loaded."),
                        ],
                        style={
                            "width": "73%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "10px",
                            "backgroundColor": "white",
                            "border": "1px solid #ddd",
                            "marginLeft": "1%",
                            "marginTop": "12px",
                            "fontSize": "14px",
                            "lineHeight": "1.6",
                        },
                    ),
                ]
            ),

            dcc.Store(id="manifest-store"),
        ],
    )

    @app.callback(
        Output("manifest-store", "data"),
        Output("model-dropdown", "options"),
        Output("model-dropdown", "value"),
        Output("status", "children"),
        Input("load-cache-button", "n_clicks"),
        prevent_initial_call=False,
    )
    def load_cache(n_clicks):
        try:
            manifest = load_manifest(cache_root)
        except Exception as e:
            return {}, [], None, f"加载 compare cache 失败：{e}"

        items = manifest.get("items", [])
        failures = manifest.get("failures", [])

        if not items:
            return manifest, [], None, f"没有成功样本。失败数：{len(failures)}"

        options = []

        for item in items:
            before_fc = item.get("before_feature_count", {})
            after_fc = item.get("after_feature_count", {})

            label = (
                f"{item['stem']} | "
                f"removed={item.get('num_removed_faces', '?')} | "
                f"before F={before_fc.get('Fillet', 0)}, C={before_fc.get('Chamfer', 0)} | "
                f"after recovered={item.get('num_recovered_patch_faces', '?')}"
            )

            options.append(
                {
                    "label": label,
                    "value": item["stem"],
                }
            )

        return (
            manifest,
            options,
            options[0]["value"],
            f"成功样本：{len(items)} | 失败样本：{len(failures)}",
        )

    @app.callback(
        Output("before-graph", "figure"),
        Output("after-graph", "figure"),
        Output("status", "children", allow_duplicate=True),
        Output("model-info", "children"),
        Input("model-dropdown", "value"),
        State("manifest-store", "data"),
        prevent_initial_call=True,
    )
    def show_model(selected_stem, manifest):
        if not selected_stem or not manifest:
            return no_update, no_update, "请选择模型。", no_update

        items = manifest.get("items", [])

        selected = None
        for item in items:
            if item["stem"] == selected_stem:
                selected = item
                break

        if selected is None:
            return no_update, no_update, f"找不到模型：{selected_stem}", no_update

        try:
            before_fig = make_figure(
                title=f"Before | {selected['stem']}",
                cache_root=cache_root,
                mesh_rel_path=selected["before_mesh"],
                manifest=manifest,
            )

            after_fig = make_figure(
                title=f"After | {selected['stem']}",
                cache_root=cache_root,
                mesh_rel_path=selected["after_mesh"],
                manifest=manifest,
            )

            before_count = format_feature_count(selected.get("before_feature_count", {}))
            after_count = format_feature_count(selected.get("after_feature_count", {}))

            before_surface_count = selected.get("before_surface_type_count", {})
            after_surface_count = selected.get("after_surface_type_count", {})

            before_feature_surface_count = selected.get("before_feature_surface_count", {})
            after_feature_surface_count = selected.get("after_feature_surface_count", {})

            status = (
                f"已加载：{selected['stem']} | "
                f"removed={selected.get('num_removed_faces', '?')} | "
                f"before_faces={selected.get('original_num_faces', '?')} | "
                f"after_faces={selected.get('result_num_faces', '?')} | "
                f"valid={selected.get('is_result_valid', '?')}"
            )

            info = html.Div(
                [
                    html.Div(f"Name: {selected['stem']}"),
                    html.Div(f"Original STEP: {selected.get('original_step', '')}"),
                    html.Div(f"Defeatured STEP: {selected.get('defeatured_step', '')}"),
                    html.Div(f"SEG: {selected.get('seg', '')}"),
                    html.Div(f"Removed Fillet/Chamfer faces: {selected.get('num_removed_faces', '?')}"),
                    html.Div(f"Original faces: {selected.get('original_num_faces', '?')}"),
                    html.Div(f"After faces: {selected.get('result_num_faces', '?')}"),
                    html.Div(f"RecoveredPatch faces: {selected.get('num_recovered_patch_faces', '?')}"),
                    html.Div(f"Result valid by BRepCheck_Analyzer: {selected.get('is_result_valid', '?')}"),

                    html.Hr(),
                    html.H4("Feature count"),
                    html.Div(f"Before features: {before_count}"),
                    html.Div(f"After features: {after_count}"),

                    html.Hr(),
                    html.H4("Surface type count"),
                    html.Div(f"Before surface types: {format_simple_count_dict(before_surface_count)}"),
                    html.Div(f"After surface types: {format_simple_count_dict(after_surface_count)}"),

                    html.Hr(),
                    html.H4("Feature × surface type"),
                    html.Div("Before:"),
                    format_nested_count_dict(before_feature_surface_count),
                    html.Br(),
                    html.Div("After:"),
                    format_nested_count_dict(after_feature_surface_count),
                ]
            )

            return before_fig, after_fig, status, info

        except Exception as e:
            before_fig = go.Figure()
            after_fig = go.Figure()
            before_fig.update_layout(title="Failed")
            after_fig.update_layout(title="Failed")
            return before_fig, after_fig, f"加载失败：{e}", "Failed."

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        debug=CONFIG["DEBUG"],
        host=CONFIG["HOST"],
        port=CONFIG["PORT"],
    )