# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dash
from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask import send_from_directory


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


parser = argparse.ArgumentParser()
dataset_name = "mfinstseg"
APP_DIR = Path(__file__).resolve().parent
parser.add_argument(
    "--cache_dir",
    type=str,
    default=str(APP_DIR / "web_cache" / dataset_name),
)
parser.add_argument(
    "--topology_summary",
    type=str,
    default=str(APP_DIR / "output" / dataset_name / "topology_type_summary.json"),
)
args, _ = parser.parse_known_args()

CACHE_DIR = Path(args.cache_dir).resolve()
INDEX_PATH = CACHE_DIR / "samples_index.json"

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {INDEX_PATH}. Please run prepare_ui_cache.py first."
    )

INDEX = load_json(INDEX_PATH)
SAMPLES = INDEX["samples"]
SAMPLE_NAMES = [x["sample_name"] for x in SAMPLES]
SAMPLE_MAP = {x["sample_name"]: x for x in SAMPLES}
DEFAULT_SAMPLE = SAMPLE_NAMES[0] if SAMPLE_NAMES else None

# -----------------------------------------------------------------------------
# Variant-aware left filters
# -----------------------------------------------------------------------------
# prepare_topology_variant_web_cache.py augments each sample_meta["subgraphs"] with:
#   sg["topology_variant"]["relation"] in {"main", "variant", "non_variant"}
# If the cache has not been augmented yet, relation falls back to "unknown".

VARIANT_RELATION_OPTIONS = [
    {"label": "Belong to main member", "value": "main"},
    {"label": "Belong to main member variants", "value": "variant"},
    {"label": "Non-variants", "value": "non_variant"},
]

RELATION_LABELS = {
    "main": "belong to main type",
    "variant": "belong to main type variant",
    "non_variant": "not variant",
    "unknown": "topology variant information not found",
}


def get_sg_relation(sg: dict) -> str:
    return str((sg.get("topology_variant") or {}).get("relation", "unknown"))


def build_variant_filter_indices(samples):
    """
    Build two-level filter indices:

    Level 1:
        topology variant relation: main / variant / non_variant

    Level 2:
        machining feature category among instances matching level 1
    """
    relation_to_samples = {"main": set(), "variant": set(), "non_variant": set()}
    relation_to_feature_options = {"main": {}, "variant": {}, "non_variant": {}}
    relation_feature_to_samples = {}

    for sample in samples:
        sample_name = sample["sample_name"]
        for sg in sample.get("subgraphs", []):
            relation = get_sg_relation(sg)
            if relation not in relation_to_samples:
                continue

            cat_id = int(sg["category_id"])
            cat_name = sg["category_name"]
            feature_key = str(cat_id)
            pair_key = f"{relation}|{feature_key}"

            relation_to_samples[relation].add(sample_name)
            relation_feature_to_samples.setdefault(pair_key, set()).add(sample_name)

            if feature_key not in relation_to_feature_options[relation]:
                relation_to_feature_options[relation][feature_key] = {
                    "label": f"{cat_id}: {cat_name}",
                    "value": feature_key,
                    "_category_id": cat_id,
                }

    options_by_relation = {}
    for relation, option_map in relation_to_feature_options.items():
        options_by_relation[relation] = [
            {"label": x["label"], "value": x["value"]}
            for x in sorted(option_map.values(), key=lambda item: item["_category_id"])
        ]

    return relation_to_samples, options_by_relation, relation_feature_to_samples


(
    RELATION_TO_SAMPLES,
    RELATION_TO_FEATURE_OPTIONS,
    RELATION_FEATURE_TO_SAMPLES,
) = build_variant_filter_indices(SAMPLES)


def get_filtered_sample_names(variant_relation_value=None, feature_category_value=None):
    """
    Return sample names after applying new filters.

    No filter:
        return all samples

    Only relation selected:
        return samples containing at least one instance in this relation group

    Relation + machining feature selected:
        return samples containing at least one instance satisfying both conditions
    """
    if variant_relation_value and feature_category_value:
        sample_set = RELATION_FEATURE_TO_SAMPLES.get(
            f"{variant_relation_value}|{feature_category_value}",
            set(),
        )
        return [name for name in SAMPLE_NAMES if name in sample_set]

    if variant_relation_value:
        sample_set = RELATION_TO_SAMPLES.get(str(variant_relation_value), set())
        return [name for name in SAMPLE_NAMES if name in sample_set]

    return SAMPLE_NAMES

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server


@server.route("/cache/<path:subpath>")
def serve_cache_file(subpath):
    return send_from_directory(str(CACHE_DIR), subpath)


def cache_url(rel_path: str) -> str:
    return f"/cache/{rel_path}"


def load_face_meshes(sample_meta: dict):
    mesh_json = CACHE_DIR / sample_meta["mesh_json"]
    return load_json(mesh_json)


def make_step_figure(sample_meta: dict):
    face_meshes = load_face_meshes(sample_meta)

    fig = go.Figure()
    trace_face_meta = []

    for face in face_meshes:
        if not face["x"] or not face["i"]:
            # 跳过没成功三角化的面
            continue

        face_id = face["face_id"]
        cat_name = face["category_name"]
        cat_id = face["category_id"]
        inst_id = face["instance_id"]
        inst_text = f"I{inst_id}" if inst_id is not None else "None"

        hover_text = (
            f"Face node id: {face_id}<br>"
            f"Category: {cat_id} - {cat_name}<br>"
            f"Instance: {inst_text}"
        )

        fig.add_trace(
            go.Mesh3d(
                x=face["x"],
                y=face["y"],
                z=face["z"],
                i=face["i"],
                j=face["j"],
                k=face["k"],
                color=face["color"],
                opacity=1.0,
                flatshading=True,
                name=f"face_{face_id}",

                # 关键：不要用 hoverinfo="skip"
                hoverinfo="text",
                text=hover_text,
                hovertemplate=hover_text + "<extra></extra>",

                showlegend=False,
            )
        )

        trace_face_meta.append({
            "face_id": face_id,
            "category_id": cat_id,
            "category_name": cat_name,
            "instance_id": inst_id,
        })

    fig.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        bgcolor="white",

        # 关键：固定 3D scene 的交互状态
        uirevision=sample_meta["sample_name"],
        dragmode="orbit",
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    hovermode="closest",

    # 顶层也保留
    uirevision=sample_meta["sample_name"],
)

    return fig, trace_face_meta


def make_sample_buttons(sample_names=None, selected_sample=None):
    if sample_names is None:
        sample_names = SAMPLE_NAMES

    if not sample_names:
        return [
            html.Div(
                "No samples match current filters.",
                style={
                    "fontSize": "13px",
                    "color": "#777",
                    "padding": "8px",
                },
            )
        ]

    buttons = []

    for name in sample_names:
        is_selected = name == selected_sample

        buttons.append(
            dbc.Button(
                name,
                id={"type": "sample-btn", "index": name},
                color="primary" if is_selected else "secondary",
                outline=not is_selected,
                className="mb-2",
                style={
                    "width": "100%",
                    "textAlign": "left",
                    "fontSize": "13px",
                },
            )
        )

    return buttons



def make_sample_dropdown_options(sample_names=None):
    if sample_names is None:
        sample_names = SAMPLE_NAMES

    return [
        {"label": name, "value": name}
        for name in (sample_names or [])
    ]


def subgraph_options(sample_meta: dict):
    options = []
    for sg in sample_meta["subgraphs"]:
        relation = get_sg_relation(sg)
        relation_label = RELATION_LABELS.get(relation, relation)
        label = (
            f"I{sg['instance_id']} | "
            f"{sg['category_id']}:{sg['category_name']} | "
            f"{relation_label} | "
            f"faces={sg['nodes']}"
        )
        options.append({
            "label": label,
            "value": str(sg["instance_id"]),
        })
    return options


def get_subgraph_entry(sample_meta: dict, instance_id_str: str):
    for sg in sample_meta["subgraphs"]:
        if str(sg["instance_id"]) == str(instance_id_str):
            return sg
    return None


def subgraph_matches_current_filter(sg: dict, variant_relation_value=None, feature_category_value=None) -> bool:
    """
    Whether a subgraph instance satisfies the current left sidebar filters.

    This is used only to choose the right-side subgraph dropdown default.
    The sample list itself is still controlled by get_filtered_sample_names().
    """
    if variant_relation_value and get_sg_relation(sg) != str(variant_relation_value):
        return False

    if feature_category_value and str(sg.get("category_id")) != str(feature_category_value):
        return False

    return True


def choose_default_subgraph_for_filter(sample_meta: dict, variant_relation_value=None, feature_category_value=None):
    """
    Prefer the first feature instance in this sample that matches the current
    relation/category filters. Fall back to the first subgraph if nothing matches.
    """
    subgraphs = sample_meta.get("subgraphs", [])

    for sg in subgraphs:
        if subgraph_matches_current_filter(
            sg,
            variant_relation_value=variant_relation_value,
            feature_category_value=feature_category_value,
        ):
            return str(sg["instance_id"])

    return str(subgraphs[0]["instance_id"]) if subgraphs else None


app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id="selected-sample-store", data=DEFAULT_SAMPLE),
        dcc.Store(id="face-meta-store", data=[]),
        dcc.Store(id="filtered-sample-names-store", data=SAMPLE_NAMES),

        dbc.Row([
            dbc.Col(html.H3("MFInstSeg Interactive Viewer"), width=12)
        ], className="mt-2 mb-2"),

        dbc.Row([
            # =============================
            # Left: horizontal filters + 3D STEP view
            # =============================
            dbc.Col(
                [
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                "Topology variant relation",
                                                style={"fontWeight": "bold", "fontSize": "12px", "marginBottom": "4px"},
                                            ),
                                            dcc.Dropdown(
                                                id="variant-relation-filter",
                                                options=VARIANT_RELATION_OPTIONS,
                                                value=None,
                                                placeholder="All relations",
                                                clearable=True,
                                                style={"fontSize": "12px"},
                                            ),
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                "Machining feature category",
                                                style={"fontWeight": "bold", "fontSize": "12px", "marginBottom": "4px"},
                                            ),
                                            dcc.Dropdown(
                                                id="feature-category-filter",
                                                options=[],
                                                value=None,
                                                placeholder="All categories",
                                                clearable=True,
                                                style={"fontSize": "12px"},
                                            ),
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                "Matched samples",
                                                style={"fontWeight": "bold", "fontSize": "12px", "marginBottom": "4px"},
                                            ),
                                            dcc.Dropdown(
                                                id="sample-dropdown",
                                                options=make_sample_dropdown_options(SAMPLE_NAMES),
                                                value=DEFAULT_SAMPLE,
                                                placeholder="Select a matched sample...",
                                                clearable=False,
                                                style={"fontSize": "12px"},
                                            ),
                                        ],
                                        width=4,
                                    ),
                                ],
                                className="g-2",
                            ),
                        ],
                        style={
                            "border": "1px solid #DDD",
                            "padding": "8px",
                            "borderRadius": "8px",
                            "backgroundColor": "#FAFAFA",
                            "marginBottom": "10px",
                        },
                    ),

                    html.H5("3D STEP View"),
                    dcc.Graph(
                        id="step-view",
                        style={"height": "70vh", "border": "1px solid #DDD", "borderRadius": "8px"},
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "doubleClick": "reset",
                            "modeBarButtonsToRemove": [
                                "resetCameraDefault3d",
                                "resetCameraLastSave3d",
                            ],
                        },
                    ),
                    html.Div(
                        id="hover-info-box",
                        style={
                            "marginTop": "8px",
                            "border": "1px solid #DDD",
                            "padding": "10px",
                            "borderRadius": "8px",
                            "minHeight": "72px",
                            "backgroundColor": "#FCFCFC",
                        },
                    ),
                ],
                width=5,
            ),

            # =============================
            # Middle: full graph + subgraph
            # =============================
            dbc.Col(
                [
                    html.H5("Graphs"),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Full Graph", style={"fontWeight": "bold"}),
                                    dbc.Checklist(
                                        id="full-graph-toggle",
                                        options=[{"label": "Show full graph", "value": "show"}],
                                        value=["show"],
                                        switch=True,
                                        style={"fontSize": "14px"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "marginBottom": "6px",
                                },
                            ),

                            html.Div(
                                id="full-graph-frame",
                                className="panzoom-frame",
                                children=[
                                    html.Img(
                                        id="full-graph-img",
                                        className="panzoom-img",
                                        title="Mouse wheel to zoom, drag to pan, double click to reset",
                                        style={
                                            "maxWidth": "100%",
                                            "maxHeight": "100%",
                                            "objectFit": "contain",
                                            "transformOrigin": "0 0",
                                            "cursor": "grab",
                                            "userSelect": "none",
                                            "WebkitUserDrag": "none",
                                        },
                                        draggable="false",
                                    ),
                                ],
                                style={
                                    "width": "100%",
                                    "height": "320px",
                                    "border": "1px solid #DDD",
                                    "borderRadius": "8px",
                                    "padding": "8px",
                                    "backgroundColor": "white",
                                    "overflow": "hidden",
                                    "marginBottom": "12px",
                                    "position": "relative",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                },
                            ),
                        ]
                    ),

                    html.Div(
                        [
                            html.Div("Subgraph", style={"fontWeight": "bold", "marginBottom": "6px"}),

                            dcc.Dropdown(
                                id="subgraph-dropdown",
                                options=[],
                                value=None,
                                placeholder="Select one instance subgraph...",
                                style={"marginBottom": "8px", "fontSize": "12px"},
                            ),

                            dbc.Checklist(
                                id="context-toggle",
                                options=[
                                    {"label": "Show one-hop external nodes", "value": "with_context"},
                                ],
                                value=["with_context"],
                                switch=True,
                                style={"marginBottom": "10px"},
                            ),

                            html.Div(
                                id="subgraph-frame",
                                className="panzoom-frame",
                                children=[
                                    html.Img(
                                        id="subgraph-img",
                                        className="panzoom-img",
                                        title="Mouse wheel to zoom, drag to pan, double click to reset",
                                        style={
                                            "maxWidth": "100%",
                                            "maxHeight": "100%",
                                            "objectFit": "contain",
                                            "transformOrigin": "0 0",
                                            "cursor": "grab",
                                            "userSelect": "none",
                                            "WebkitUserDrag": "none",
                                        },
                                        draggable="false",
                                    ),
                                ],
                                style={
                                    "width": "100%",
                                    "height": "400px",
                                    "border": "1px solid #DDD",
                                    "borderRadius": "8px",
                                    "padding": "8px",
                                    "backgroundColor": "white",
                                    "overflow": "hidden",
                                    "position": "relative",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                },
                            ),
                        ]
                    ),
                ],
                width=4,
            ),

            # =============================
            # Right: always-visible topology history
            # =============================
            dbc.Col(
                [
                    html.Div(
                        id="topology-variant-history-wrapper",
                        children=[
                            html.Div(
                                [
                                    html.Div("Topology Variant History", style={"fontWeight": "bold", "fontSize": "16px"}),
                                    dbc.Checklist(
                                        id="history-detail-toggle",
                                        options=[{"label": "Show detail skipped-node paths", "value": "show_detail"}],
                                        value=["show_detail"],
                                        switch=True,
                                        style={"fontSize": "13px"},
                                    ),
                                ],
                                style={"marginBottom": "10px"},
                            ),
                            html.Div(id="topology-variant-history-panel"),
                        ],
                        style={
                            "display": "block",
                            "border": "1px solid #DDD",
                            "borderRadius": "8px",
                            "padding": "10px",
                            "backgroundColor": "#FCFCFC",
                            "height": "86vh",
                            "overflowY": "auto",
                        },
                    ),
                ],
                width=3,
            ),
        ]),
    ],
)

def make_history_step_card(step: dict, idx: int, show_detail: bool):
    image = step.get("image_detail") if show_detail else step.get("image_simple")
    if not image:
        image = step.get("image")

    return html.Div(
        [
            html.Div(
                f"{idx + 1}. {step.get('title', 'History step')}",
                style={"fontWeight": "bold", "fontSize": "13px", "marginBottom": "6px"},
            ),
            html.Div(
                className="panzoom-frame",
                children=[
                    html.Img(
                        src=cache_url(image) if image else None,
                        className="panzoom-img",
                        title="Mouse wheel to zoom, drag to pan, double click to reset",
                        style={
                            "maxWidth": "100%",
                            "maxHeight": "100%",
                            "objectFit": "contain",
                            "transformOrigin": "0 0",
                            "cursor": "grab",
                            "userSelect": "none",
                            "WebkitUserDrag": "none",
                        },
                        draggable="false",
                    ),
                ],
                style={
                    "width": "100%",
                    "height": "220px",
                    "border": "1px solid #DDD",
                    "borderRadius": "8px",
                    "padding": "8px",
                    "backgroundColor": "white",
                    "overflow": "hidden",
                    "position": "relative",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            ) if image else html.Div("No image for this step."),
            html.Div(
                step.get("note", ""),
                style={"fontSize": "12px", "color": "#555", "marginTop": "6px", "maxWidth": "100%", "whiteSpace": "normal"},
            ),
        ],
        style={
            "display": "block",
            "verticalAlign": "top",
            "marginBottom": "12px",
            "padding": "8px",
            "border": "1px solid #E5E5E5",
            "borderRadius": "8px",
            "backgroundColor": "white",
            "whiteSpace": "normal",
        },
    )


def render_topology_variant_history(sample_name, subgraph_instance_id, show_detail=False):
    if sample_name is None or subgraph_instance_id is None:
        return html.Div("Select one feature instance first.")

    sample_meta = SAMPLE_MAP.get(sample_name)
    if not sample_meta:
        return html.Div("No sample metadata found.")

    sg = get_subgraph_entry(sample_meta, subgraph_instance_id)
    if sg is None:
        return html.Div("No subgraph metadata found.")

    tv = sg.get("topology_variant") or {}
    relation = str(tv.get("relation", "unknown"))
    relation_label = tv.get("relation_label") or RELATION_LABELS.get(relation, relation)
    reason_label = tv.get("reason_label") or tv.get("reason") or ""
    variant_type = tv.get("variant_type") or ""
    steps = tv.get("history_steps") or []

    header = html.Div(
        [
            html.Span(
                f"I{sg['instance_id']} | {sg['category_id']}:{sg['category_name']} | ",
                style={"fontWeight": "bold"},
            ),
            html.Span(relation_label),
            html.Span(f" | {variant_type}" if variant_type else ""),
        ],
        style={"marginBottom": "8px"},
    )

    if relation == "main":
        return html.Div([header, html.Div("belong to main type", style={"fontSize": "18px", "fontWeight": "bold"})])

    if relation == "non_variant":
        prefix = "✅ " if tv.get("green_check") else ""
        return html.Div([
            header,
            html.Div(
                prefix + (reason_label or "No detailed reason was cached."),
                style={"fontSize": "15px", "fontWeight": "bold", "color": "#1B7F3A" if tv.get("green_check") else "#333"},
            ),
        ])

    if relation == "variant":
        cards = [make_history_step_card(step, idx, show_detail=show_detail) for idx, step in enumerate(steps)]
        if not cards:
            cards = [html.Div("This instance is a variant, but no history images were cached.")]
        return html.Div([
            header,
            html.Div(
                cards,
                style={
                    "whiteSpace": "normal",
                    "overflowX": "hidden",
                    "paddingBottom": "6px",
                },
            ),
        ])

    return html.Div([header, html.Div(reason_label or "Topology variant cache was not found for this instance.")])


# 1) variant relation 改变后，更新 machining feature 下拉框
@app.callback(
    Output("feature-category-filter", "options"),
    Output("feature-category-filter", "value"),
    Input("variant-relation-filter", "value"),
)
def update_feature_category_filter(variant_relation_value):
    if not variant_relation_value:
        return [], None

    options = RELATION_TO_FEATURE_OPTIONS.get(str(variant_relation_value), [])
    return options, None


# 2) 只根据 filter 计算 filtered sample names
# 注意：这里不能依赖 selected-sample-store，否则会形成循环依赖
@app.callback(
    Output("filtered-sample-names-store", "data"),
    Input("variant-relation-filter", "value"),
    Input("feature-category-filter", "value"),
)
def update_filtered_sample_names(variant_relation_value, feature_category_value):
    filtered_names = get_filtered_sample_names(
        variant_relation_value=variant_relation_value,
        feature_category_value=feature_category_value,
    )
    return filtered_names


# 3) 根据 filtered samples 更新顶部 matched samples 下拉框
@app.callback(
    Output("sample-dropdown", "options"),
    Output("sample-dropdown", "value"),
    Input("filtered-sample-names-store", "data"),
    State("selected-sample-store", "data"),
)
def update_sample_dropdown(filtered_sample_names, current_sample):
    filtered_sample_names = filtered_sample_names or []
    options = make_sample_dropdown_options(filtered_sample_names)

    if not filtered_sample_names:
        return options, None

    if current_sample in filtered_sample_names:
        return options, current_sample

    return options, filtered_sample_names[0]


# 4) sample dropdown 或 filter 改变后，更新 selected sample
@app.callback(
    Output("selected-sample-store", "data"),
    Input("sample-dropdown", "value"),
    Input("filtered-sample-names-store", "data"),
    State("selected-sample-store", "data"),
)
def update_selected_sample(sample_dropdown_value, filtered_sample_names, current_sample):
    filtered_sample_names = filtered_sample_names or []

    if not filtered_sample_names:
        return None

    if sample_dropdown_value in filtered_sample_names:
        return sample_dropdown_value

    if current_sample in filtered_sample_names:
        return current_sample

    return filtered_sample_names[0]

# 2) 根据 sample 更新 3D 图 / hover meta / full graph / subgraph dropdown
@app.callback(
    Output("step-view", "figure"),
    Output("face-meta-store", "data"),
    Output("full-graph-img", "src"),
    Output("subgraph-dropdown", "options"),
    Output("subgraph-dropdown", "value"),
    Input("selected-sample-store", "data"),
    Input("variant-relation-filter", "value"),
    Input("feature-category-filter", "value"),
)
def update_main_views(sample_name, variant_relation_value, feature_category_value):
    if sample_name is None:
        return go.Figure(), [], None, [], None

    sample_meta = SAMPLE_MAP[sample_name]

    fig, trace_face_meta = make_step_figure(sample_meta)
    full_graph_src = cache_url(sample_meta["full_graph_image"])

    options = subgraph_options(sample_meta)
    default_subgraph = choose_default_subgraph_for_filter(
        sample_meta,
        variant_relation_value=variant_relation_value,
        feature_category_value=feature_category_value,
    )

    return fig, trace_face_meta, full_graph_src, options, default_subgraph


# 3) subgraph image
@app.callback(
    Output("subgraph-img", "src"),
    Input("selected-sample-store", "data"),
    Input("subgraph-dropdown", "value"),
    Input("context-toggle", "value"),
)
def update_subgraph_image(sample_name, subgraph_instance_id, toggle_values):
    if sample_name is None or subgraph_instance_id is None:
        return None

    sample_meta = SAMPLE_MAP[sample_name]
    sg = get_subgraph_entry(sample_meta, subgraph_instance_id)
    if sg is None:
        return None

    with_context = "with_context" in (toggle_values or [])
    rel_path = sg["image_with_context"] if with_context else sg["image_no_context"]
    return cache_url(rel_path)


# 3b) topology variant history panel
@app.callback(
    Output("topology-variant-history-panel", "children"),
    Input("selected-sample-store", "data"),
    Input("subgraph-dropdown", "value"),
    Input("history-detail-toggle", "value"),
)
def update_topology_variant_history_panel(sample_name, subgraph_instance_id, detail_toggle_values):
    show_detail = "show_detail" in (detail_toggle_values or [])
    return render_topology_variant_history(sample_name, subgraph_instance_id, show_detail=show_detail)


# 4) hover 信息框
@app.callback(
    Output("hover-info-box", "children"),
    Input("step-view", "hoverData"),
    State("face-meta-store", "data"),
)
def update_hover_box(hover_data, face_meta_store):
    if not hover_data or "points" not in hover_data or not hover_data["points"]:
        return html.Div([
            html.Div("Hover on a face in the 3D STEP view."),
            html.Div("You will see: face node id / feature category / instance id."),
        ])

    point = hover_data["points"][0]
    curve_number = point.get("curveNumber", None)
    if curve_number is None:
        return "No hover info."

    if curve_number >= len(face_meta_store):
        return "No hover info."

    meta = face_meta_store[curve_number]
    face_id = meta["face_id"]
    category_name = meta["category_name"]
    category_id = meta["category_id"]
    instance_id = meta["instance_id"]

    return html.Div([
        html.Div(f"Face node id: {face_id}"),
        html.Div(f"Feature category: {category_id} - {category_name}"),
        html.Div(f"Instance: {'I' + str(instance_id) if instance_id is not None else 'None'}"),
    ])

@app.callback(
    Output("full-graph-frame", "style"),
    Input("full-graph-toggle", "value"),
)
def toggle_full_graph_frame(toggle_values):
    base_style = {
        "width": "100%",
        "height": "360px",
        "border": "1px solid #DDD",
        "borderRadius": "8px",
        "padding": "8px",
        "backgroundColor": "white",
        "overflow": "hidden",
        "marginBottom": "14px",
        "position": "relative",
        "alignItems": "center",
        "justifyContent": "center",
    }

    if "show" in (toggle_values or []):
        base_style["display"] = "flex"
    else:
        base_style["display"] = "none"

    return base_style


def main():
    app.run(debug=True, host="127.0.0.1", port=8060)


if __name__ == "__main__":
    main()
