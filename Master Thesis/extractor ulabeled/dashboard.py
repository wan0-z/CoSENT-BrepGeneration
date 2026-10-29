"""Minimal two-filter Dash viewer for unlabelled extraction caches."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


HERE = Path(__file__).resolve().parent
SEED_ROOT = HERE.parent / "Seed_Extractor"
if str(SEED_ROOT) not in sys.path:
    sys.path.insert(0, str(SEED_ROOT))
import extract_one_step as legacy  # type: ignore # noqa: E402


CACHE_ROOT = HERE / "output" / "web_cache"
HOST = "127.0.0.1"
PORT = 8051
FEATURE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22]
CATEGORY_TO_COLOR = legacy.CATEGORY_TO_COLOR


def rgb_css(color: List[float]) -> str:
    return "rgb({},{},{})".format(*(round(float(channel) * 255) for channel in color))


def cache_paths() -> List[Path]:
    return sorted(CACHE_ROOT.glob("*.json")) if CACHE_ROOT.exists() else []


@lru_cache(maxsize=256)
def load_cache(path_string: str) -> Dict[str, Any]:
    return json.loads(Path(path_string).read_text(encoding="utf-8"))


def build_index() -> Dict[int, List[Dict[str, str]]]:
    index = {category_id: [] for category_id in FEATURE_IDS}
    for path in cache_paths():
        try:
            cache = load_cache(str(path))
        except (OSError, json.JSONDecodeError):
            continue
        present = {int(value) for value in cache.get("present_category_ids", [])}
        for category_id in FEATURE_IDS:
            if category_id in present:
                index[category_id].append({"label": cache.get("part_name", path.stem), "value": str(path)})
    return index


FILTER_INDEX = build_index()
FEATURE_OPTIONS = [
    {
        "label": f"{legacy.FACE_CATEGORIES[category_id]} ({category_id})",
        "value": category_id,
    }
    for category_id in FEATURE_IDS
]
INITIAL_FEATURE = next((value for value in FEATURE_IDS if FILTER_INDEX[value]), FEATURE_IDS[0])
INITIAL_STEPS = FILTER_INDEX[INITIAL_FEATURE]
INITIAL_STEP = INITIAL_STEPS[0]["value"] if INITIAL_STEPS else None


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font={"size": 18, "color": "#64748b"})
    figure.update_layout(
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def build_figure(cache: Dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    for face in cache.get("faces", []):
        vertices, triangles = face.get("vertices", []), face.get("triangles", [])
        if not vertices or not triangles:
            continue
        category_id = int(face.get("predicted_category_id", legacy.STOCK_CATEGORY_ID))
        category_name = face.get("predicted_category_name", legacy.FACE_CATEGORIES[category_id])
        hover = (
            f"<b>Face {face['face_id']}</b><br>Surface: {face.get('face_type')}"
            f"<br>Feature: {category_name} ({category_id})"
            f"<br>Instance: {face.get('predicted_instance_id')}"
            f"<br>Seam edges: {face.get('seam_edge_count', 0)}"
        )
        figure.add_trace(
            go.Mesh3d(
                x=[vertex[0] for vertex in vertices],
                y=[vertex[1] for vertex in vertices],
                z=[vertex[2] for vertex in vertices],
                i=[triangle[0] for triangle in triangles],
                j=[triangle[1] for triangle in triangles],
                k=[triangle[2] for triangle in triangles],
                color=rgb_css(CATEGORY_TO_COLOR[category_id]),
                opacity=0.45 if category_id == legacy.STOCK_CATEGORY_ID else 1.0,
                flatshading=True,
                text=[hover] * len(vertices),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
                name=f"Face {face['face_id']}",
            )
        )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        paper_bgcolor="white",
        scene={
            "aspectmode": "data",
            "dragmode": "orbit",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "bgcolor": "white",
        },
        uirevision=cache.get("part_name"),
    )
    return figure


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Unlabelled STEP Feature Viewer"
app.layout = dbc.Container(
    fluid=True,
    className="p-3",
    children=[
        dbc.Row(
            className="g-3 mb-3",
            children=[
                dbc.Col(
                    [
                        html.Label("Feature", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="feature-filter",
                            options=FEATURE_OPTIONS,
                            value=INITIAL_FEATURE,
                            clearable=False,
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.Label("STEP file", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id="step-filter",
                            options=INITIAL_STEPS,
                            value=INITIAL_STEP,
                            clearable=False,
                        ),
                    ],
                    md=6,
                ),
            ],
        ),
        dcc.Graph(
            id="model-viewer",
            figure=build_figure(load_cache(INITIAL_STEP)) if INITIAL_STEP else empty_figure("No matching STEP file"),
            style={"height": "calc(100vh - 125px)", "minHeight": "520px"},
            config={"displaylogo": False, "scrollZoom": True},
        ),
    ],
)


@app.callback(
    Output("step-filter", "options"),
    Output("step-filter", "value"),
    Input("feature-filter", "value"),
)
def update_step_filter(category_id: int):
    options = FILTER_INDEX.get(int(category_id), [])
    return options, options[0]["value"] if options else None


@app.callback(Output("model-viewer", "figure"), Input("step-filter", "value"))
def update_viewer(cache_path: str | None):
    if not cache_path:
        return empty_figure("No STEP file contains the selected feature")
    return build_figure(load_cache(cache_path))


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)

