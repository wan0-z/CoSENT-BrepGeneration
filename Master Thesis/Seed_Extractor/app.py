# Unified single/batch Dash viewer
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import itertools
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import dash
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html


# ============================================================
# 手动配置
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SINGLE_CACHE_ROOT = SCRIPT_DIRECTORY / "output" / "single"
SPACE_SINGLE_CACHE_ROOT = SCRIPT_DIRECTORY / "output_space" / "single"
BATCH_CACHE_ROOT = SCRIPT_DIRECTORY / "output" / "batch" / "web_cache"
SPACE_BATCH_CACHE_ROOT = SCRIPT_DIRECTORY / "output_space" / "batch" / "web_cache"
BATCH_SUMMARY_PATH = SCRIPT_DIRECTORY / "output" / "batch" / "summary.json"
SPACE_BATCH_SUMMARY_PATH = SCRIPT_DIRECTORY / "output_space" / "batch" / "summary.json"
HOST = "127.0.0.1"
PORT = 8040
DEBUG = False

INITIAL_DRAG_MODE = "orbit"
STOCK_OPACITY = 1.0
FEATURE_OPACITY = 1.0
ERROR_FACE_COLOR = [1.0, 0.0, 0.0]
FAG_LAYOUT_SEED = 42
FAG_NODE_SIZE = 22
FAG_FEATURE_NODE_SIZE = 34
FAG_EDGE_WIDTH = 1.3
FAG_MULTI_EDGE_CURVATURE_STEP = 0.10
FAG_SELF_LOOP_RADIUS = 0.055

# Keep the viewer palette identical to MF_Explorer/5_prepare_ui_cache.py.
# Colors are resolved from category IDs at display time so that old caches also
# use the current canonical palette.
CATEGORY_TO_COLOR = {
    0: [0.0, 1.0, 0.0],
    1: [0.0, 0.0, 1.0],
    2: [1.0, 1.0, 0.0],
    3: [1.0, 0.5, 0.0],
    4: [0.0, 1.0, 1.0],
    5: [1.0, 0.0, 1.0],
    6: [0.5, 0.5, 0.5],
    7: [0.6, 0.4, 0.2],
    8: [0.5, 0.0, 0.5],
    9: [0.4, 0.8, 0.8],
    10: [0.9, 0.7, 0.5],
    11: [0.7, 0.3, 0.0],
    12: [0.6, 0.6, 0.0],
    13: [0.8, 0.0, 0.1],
    14: [0.0, 0.5, 0.9],
    15: [0.8, 0.4, 0.7],
    16: [1.0, 0.8, 0.0],
    17: [0.3, 0.9, 0.3],
    18: [0.7, 0.0, 1.0],
    19: [0.5, 1.0, 0.5],
    20: [0.2, 0.8, 1.0],
    21: [0.8, 0.5, 0.0],
    22: [0.6, 0.8, 0.4],
    23: [0.9, 0.5, 0.5],
    24: [0.5, 0.7, 0.9],
}


# ============================================================
# Cache索引
# ============================================================

def cache_root_for_mode(mode: str, extraction_mode: str = "normal") -> Path:
    if mode == "single":
        return SPACE_SINGLE_CACHE_ROOT if extraction_mode == "space" else SINGLE_CACHE_ROOT
    return SPACE_BATCH_CACHE_ROOT if extraction_mode == "space" else BATCH_CACHE_ROOT


def list_cache_files(mode: str, extraction_mode: str = "normal") -> List[Path]:
    cache_root = cache_root_for_mode(mode, extraction_mode)
    if not cache_root.exists():
        return []
    return sorted(
        path
        for path in cache_root.rglob("*.json")
        if (
            (mode == "single" and path.name == "cache.json")
            or (mode == "batch" and path.name != "summary.json")
        )
    )


def load_cache(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def cache_options(mode: str, extraction_mode: str = "normal") -> List[Dict[str, str]]:
    cache_root = cache_root_for_mode(mode, extraction_mode)
    options = []
    for path in list_cache_files(mode, extraction_mode):
        if mode == "single":
            label = path.parent.name
        else:
            label = str(path.relative_to(cache_root).with_suffix(""))
        options.append({"label": label, "value": str(path)})
    return options


OUTCOME_OPTIONS = [
    {"label": "True positive", "value": "true_positive"},
    {"label": "False negative", "value": "false_negative"},
    {"label": "False positive", "value": "false_positive"},
    {"label": "Mixed class", "value": "mixed_class"},
]
FEATURE_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 21, 22,
]
SPECIAL_SLANTED_KEY = "slanted_through_step_as_rectangular_through_step"
SPECIAL_TRIANGULAR_BLIND_KEY = "triangular_blind_step_as_rectangular_through_step"
MIXED_CLASS_PREFIX = "mixed"
SPECIAL_FEATURE_LABELS = {
    SPECIAL_SLANTED_KEY: "slanted_through_step → rectangular_through_step",
    SPECIAL_TRIANGULAR_BLIND_KEY: "triangular_blind_step → rectangular_through_step",
}
INITIAL_MODE = "batch"
INITIAL_EXTRACTION_MODE = "normal"
INITIAL_OUTCOME = "true_positive"


def category_names_from_caches() -> List[str]:
    for mode in ("batch", "single"):
        files = list_cache_files(mode, "normal")
        if files:
            return list(load_cache(files[0]).get("category_names", []))
    return []


CATEGORY_NAMES = category_names_from_caches()


def category_label(category_id: int) -> str:
    if 0 <= category_id < len(CATEGORY_NAMES):
        return f"{CATEGORY_NAMES[category_id]} ({category_id})"
    return str(category_id)


FEATURE_OPTION_ORDER = [str(category_id) for category_id in FEATURE_CATEGORY_IDS] + [
    SPECIAL_SLANTED_KEY,
    SPECIAL_TRIANGULAR_BLIND_KEY,
]
FEATURE_LABELS = {
    str(category_id): category_label(category_id)
    for category_id in FEATURE_CATEGORY_IDS
}
FEATURE_LABELS.update(SPECIAL_FEATURE_LABELS)


def mixed_class_key(ground_truth_id: int, predicted_id: int) -> str:
    return f"{MIXED_CLASS_PREFIX}:{int(ground_truth_id)}:{int(predicted_id)}"


def mixed_class_ids(feature_key: str) -> Tuple[int, int]:
    prefix, ground_truth_id, predicted_id = str(feature_key).split(":", 2)
    if prefix != MIXED_CLASS_PREFIX:
        raise ValueError(f"Not a mixed-class key: {feature_key}")
    return int(ground_truth_id), int(predicted_id)


def mixed_class_label(feature_key: str) -> str:
    ground_truth_id, predicted_id = mixed_class_ids(feature_key)
    return f"{category_label(ground_truth_id)} → {category_label(predicted_id)}"


def face_filter_pair(face: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    ground_truth_id = int(face.get("ground_truth_category_id", 24))
    predicted_id = int(face.get("predicted_category_id", 24))

    # Chamfer and round/fillet are excluded extraction targets, so their
    # expected extractor output is stock. Predicting a feature is an error.
    if ground_truth_id in {0, 23}:
        if predicted_id != 24:
            return "false_positive", str(predicted_id)
        return None

    if ground_truth_id in {10, 20}:
        special_key = (
            SPECIAL_SLANTED_KEY
            if ground_truth_id == 10
            else SPECIAL_TRIANGULAR_BLIND_KEY
        )
        if predicted_id == 8:
            return "true_positive", special_key
        if predicted_id == 24:
            return "false_negative", special_key
        return "mixed_class", mixed_class_key(ground_truth_id, predicted_id)

    if ground_truth_id == 24:
        if predicted_id != 24:
            return "false_positive", str(predicted_id)
        return None

    if predicted_id == 24:
        return "false_negative", str(ground_truth_id)

    if ground_truth_id == predicted_id:
        return "true_positive", str(ground_truth_id)
    return "mixed_class", mixed_class_key(ground_truth_id, predicted_id)


def build_filter_index() -> Dict[str, Set[Tuple[str, str]]]:
    index: Dict[str, Set[Tuple[str, str]]] = {}
    for mode, extraction_mode in itertools.product(("single", "batch"), ("normal", "space")):
        for path in list_cache_files(mode, extraction_mode):
            pairs = {
                pair
                for face in load_cache(path).get("faces", [])
                if (pair := face_filter_pair(face)) is not None
            }
            index[str(path)] = pairs
    return index


CACHE_FILTER_INDEX = build_filter_index()


def feature_options(mode: str, outcome: str, extraction_mode: str = "normal") -> List[Dict[str, str]]:
    available = {
        feature_key
        for path in list_cache_files(mode, extraction_mode)
        for indexed_outcome, feature_key in CACHE_FILTER_INDEX.get(str(path), set())
        if indexed_outcome == outcome
    }
    if outcome == "mixed_class":
        ordered_keys = sorted(available, key=mixed_class_ids)
        return [
            {"label": mixed_class_label(key), "value": key}
            for key in ordered_keys
        ]
    return [
        {"label": FEATURE_LABELS[key], "value": key}
        for key in FEATURE_OPTION_ORDER
        if key in available
    ]


def filtered_cache_options(mode: str, outcome: str, feature_key: str, extraction_mode: str = "normal") -> List[Dict[str, str]]:
    return [
        option
        for option in cache_options(mode, extraction_mode)
        if (outcome, feature_key) in CACHE_FILTER_INDEX.get(option["value"], set())
    ]


INITIAL_FEATURE_OPTIONS = feature_options(INITIAL_MODE, INITIAL_OUTCOME)
INITIAL_FEATURE_VALUE = (
    INITIAL_FEATURE_OPTIONS[0]["value"]
    if INITIAL_FEATURE_OPTIONS
    else None
)
INITIAL_CACHE_OPTIONS = filtered_cache_options(
    INITIAL_MODE,
    INITIAL_OUTCOME,
    INITIAL_FEATURE_VALUE,
)


# ============================================================
# 颜色和格式
# ============================================================

def rgb_float_to_css(color: List[float]) -> str:
    values = [int(round(float(value) * 255.0)) for value in color]
    return f"rgb({values[0]},{values[1]},{values[2]})"


def format_angle(angle: Optional[float]) -> str:
    if angle is None:
        return "unknown"
    return f"{float(angle):.2f}°"


def category_color(category_id: int) -> List[float]:
    return CATEGORY_TO_COLOR.get(int(category_id), CATEGORY_TO_COLOR[24])


def face_display_color(face: Dict[str, Any]) -> List[float]:
    filter_pair = face_filter_pair(face)
    if filter_pair is not None and filter_pair[0] != "true_positive":
        return ERROR_FACE_COLOR
    return category_color(int(face.get("predicted_category_id", 24)))


# ============================================================
# 图构建
# ============================================================

def build_multigraph(cache: Dict[str, Any]) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for node in cache.get("fag", {}).get("nodes", []):
        face_id = int(node["face_id"])
        graph.add_node(face_id, **node)
    for edge in cache.get("fag", {}).get("edges", []):
        source = int(edge["source"])
        target = int(edge["target"])
        key = int(edge["edge_key"])
        graph.add_edge(source, target, key=key, **edge)
    return graph


def build_shared_positions(graph: nx.MultiGraph) -> Dict[int, Tuple[float, float]]:
    simple_graph = nx.Graph()
    simple_graph.add_nodes_from(graph.nodes)
    for source, target in graph.edges():
        if source != target:
            simple_graph.add_edge(source, target)
    node_count = max(simple_graph.number_of_nodes(), 1)
    k_value = max(0.24, 2.8 / math.sqrt(node_count))
    positions = nx.spring_layout(
        simple_graph,
        seed=FAG_LAYOUT_SEED,
        k=k_value,
        iterations=700,
    )
    return {
        int(node_id): (float(position[0]), float(position[1]))
        for node_id, position in positions.items()
    }


def build_face_lookup(cache: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(face["face_id"]): face for face in cache.get("faces", [])}


# ============================================================
# 3D Viewer
# ============================================================

@lru_cache(maxsize=1)
def boundary_marker_sphere():
    vertices = [(math.sin(math.pi * row / 10) * math.cos(2 * math.pi * col / 16),
                 math.sin(math.pi * row / 10) * math.sin(2 * math.pi * col / 16),
                 math.cos(math.pi * row / 10)) for row in range(11) for col in range(16)]
    triangles = []
    for row in range(10):
        for col in range(16):
            a, b = row * 16 + col, row * 16 + (col + 1) % 16
            triangles.extend([(a, b, a + 16), (b, b + 16, a + 16)])
    return vertices, triangles


def build_3d_figure(cache: Dict[str, Any], drag_mode: str, space_layers=None) -> go.Figure:
    figure = go.Figure()
    diagnostics = cache.get("space_diagnostics", {}).get("gt_instances", [])
    boundary_by_face = {
        int(check["face_id"]): check
        for instance in diagnostics
        for check in (instance.get("active_boundary") or {}).get("faces", [])
    }
    for face in cache.get("faces", []):
        vertices = face.get("vertices", [])
        triangles = face.get("triangles", [])
        if not vertices or not triangles:
            continue
        category_id = int(face.get("predicted_category_id", 24))
        color = rgb_float_to_css(face_display_color(face))
        hover = (
            f"<b>Face {face['face_id']}</b>"
            f"<br>Face type: {face.get('face_type')}"
            f"<br>Predicted: {face.get('predicted_category_name')} ({category_id})"
            f"<br>Predicted instance: {face.get('predicted_instance_id')}"
            f"<br>Ground truth: {face.get('ground_truth_category_name')} "
            f"({face.get('ground_truth_category_id')})"
            f"<br>Ground-truth instance: {face.get('ground_truth_instance_id')}"
            f"<br>Face seam count: {face.get('seam_edge_count')}"
        )
        boundary_check = boundary_by_face.get(int(face["face_id"]))
        if boundary_check is not None:
            fraction = boundary_check.get("active_fraction")
            overlap = f"{100. * fraction:.6f}%" if fraction is not None else "N/A"
            hover += f"<br>Boundary overlap: {overlap}"
        figure.add_trace(
            go.Mesh3d(
                x=[v[0] for v in vertices],
                y=[v[1] for v in vertices],
                z=[v[2] for v in vertices],
                i=[t[0] for t in triangles],
                j=[t[1] for t in triangles],
                k=[t[2] for t in triangles],
                color=color,
                opacity=1.0 if space_layers else STOCK_OPACITY if category_id == 24 else FEATURE_OPACITY,
                flatshading=True,
                text=[hover for _ in vertices],
                customdata=[int(face["face_id"]) for _ in vertices],
                hoverinfo="text",
                hovertemplate="%{text}<extra></extra>",
                name=f"Face {face['face_id']}",
                showlegend=False,
            )
        )
    space_layers = space_layers or []
    if "finite_collision_domain" in space_layers:
        for instance in diagnostics:
            collision = instance.get("finite_collision_domain") or {}
            mesh = collision.get("mesh", {})
            vertices, triangles = mesh.get("vertices", []), mesh.get("triangles", [])
            if not vertices or not triangles:
                continue
            figure.add_trace(go.Mesh3d(
                x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices],
                i=[t[0] for t in triangles], j=[t[1] for t in triangles], k=[t[2] for t in triangles],
                color="#00cc66" if instance["stage"] == "extracted" else "#ff0000" if instance["stage"] == "collision" else "#a0a0a0",
                opacity=0.22, flatshading=True, showlegend=False,
                name=f"Finite collision domain · GT {instance['gt_instance_id']}",
                hovertemplate=f"Finite collision domain · {instance.get('category_name')}<br>GT instance {instance['gt_instance_id']}<br>Stage: {instance['stage']}<br>{collision.get('method', '')}<br>Collision volume: {collision.get('collision_volume', 'not tested')}<extra></extra>",
            ))
    if space_layers:
        coordinates = [[], [], []]
        for edge in cache.get("topological_edges", []):
            for axis in range(3):
                coordinates[axis].extend([point[axis] for point in edge["points"]] + [None])
        if coordinates[0]:
            figure.add_trace(go.Scatter3d(
                x=coordinates[0], y=coordinates[1], z=coordinates[2], mode="lines",
                line=dict(color="#0066ff", width=3), opacity=1., hoverinfo="skip",
                name="Model topological edges", showlegend=False))
    if "active_boundary" in space_layers:
        model_points = [v for face in cache.get("faces", []) for v in face.get("vertices", [])]
        diagonal = math.sqrt(sum((max(p[a] for p in model_points) - min(p[a] for p in model_points)) ** 2
                                 for a in range(3))) if model_points else 1.
        radius = max(diagonal * .004, 1.e-6)
        unit_vertices, triangles = boundary_marker_sphere()
        for instance in diagnostics:
            boundary = instance.get("active_boundary") or {}
            for face in boundary.get("faces", []):
                center = face.get("center")
                if center is None:
                    continue  # Regenerate old projection caches rather than relabelling their sample counts.
                vertices = [[center[a] + radius * v[a] for a in range(3)] for v in unit_vertices]
                passed = face.get("passed")
                color = ("#a0a0a0" if passed is None else "#39ff14" if instance["stage"] == "extracted"
                         else "#ffff00" if passed else "#ff2020")
                figure.add_trace(go.Mesh3d(
                    x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices],
                    i=[t[0] for t in triangles], j=[t[1] for t in triangles], k=[t[2] for t in triangles],
                    color=color, opacity=1., flatshading=False, showlegend=False,
                    customdata=[int(face["face_id"])] * len(vertices),
                    name=f"Active boundary · GT {instance['gt_instance_id']} · Face {face['face_id']}",
                    hoverinfo="skip", hovertemplate=None,
                ))
    figure.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            aspectmode="data",
            dragmode=drag_mode,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="white",
            uirevision=cache.get("part_name"),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_size=12),
        clickmode="event+select",
        uirevision=cache.get("part_name"),
    )
    return figure


# ============================================================
# FAG边曲线
# ============================================================

def quadratic_bezier_points(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    curvature: float,
    point_count: int = 24,
) -> Tuple[List[float], List[float]]:
    middle_x = (start_x + end_x) * 0.5
    middle_y = (start_y + end_y) * 0.5
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.sqrt(dx * dx + dy * dy)
    normal_x, normal_y = ((0.0, 1.0) if length <= 1.0e-12 else (-dy / length, dx / length))
    control_x = middle_x + normal_x * curvature
    control_y = middle_y + normal_y * curvature
    x_values, y_values = [], []
    for index in range(point_count):
        t = index / (point_count - 1)
        s = 1.0 - t
        x_values.append(s * s * start_x + 2 * s * t * control_x + t * t * end_x)
        y_values.append(s * s * start_y + 2 * s * t * control_y + t * t * end_y)
    return x_values, y_values


def self_loop_points(center_x: float, center_y: float, loop_index: int) -> Tuple[List[float], List[float]]:
    radius = FAG_SELF_LOOP_RADIUS + loop_index * 0.018
    offset_y = radius * 1.15
    x_values, y_values = [], []
    for index in range(31):
        angle = 2.0 * math.pi * index / 30.0
        x_values.append(center_x + radius * math.cos(angle))
        y_values.append(center_y + offset_y + radius * math.sin(angle))
    return x_values, y_values


def build_edge_hover(source: int, target: int, key: int, attrs: Dict[str, Any]) -> str:
    return (
        f"<b>Topological edge {key}</b>"
        f"<br>Faces: {source} ↔ {target}"
        f"<br>Edge type: {attrs.get('edge_type')}"
        f"<br>Convexity: {attrs.get('convexity')}"
        f"<br>Dihedral type: {attrs.get('dihedral_type')}"
        f"<br>Dihedral angle: {format_angle(attrs.get('dihedral_angle_degrees'))}"
        f"<br>Seam: {attrs.get('is_seam')}"
    )


# ============================================================
# FAG figure
# ============================================================

def build_fag_figure(
    cache: Dict[str, Any],
    graph: nx.MultiGraph,
    positions: Dict[int, Tuple[float, float]],
    label_mode: str,
) -> go.Figure:
    figure = go.Figure()
    face_lookup = build_face_lookup(cache)

    pair_edges: Dict[Tuple[int, int], List[Tuple[int, Dict[str, Any]]]] = {}
    self_edges: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        source, target, key = int(source), int(target), int(key)
        if source == target:
            self_edges.setdefault(source, []).append((key, attrs))
        else:
            pair_edges.setdefault(tuple(sorted((source, target))), []).append((key, attrs))

    for (source, target), edges in pair_edges.items():
        sx, sy = positions[source]
        tx, ty = positions[target]
        count = len(edges)
        for index, (key, attrs) in enumerate(sorted(edges, key=lambda item: item[0])):
            centered = index - (count - 1) * 0.5
            xs, ys = quadratic_bezier_points(sx, sy, tx, ty, centered * FAG_MULTI_EDGE_CURVATURE_STEP)
            figure.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=FAG_EDGE_WIDTH, color="rgba(55,55,55,0.70)"),
                text=[build_edge_hover(source, target, key, attrs) for _ in xs],
                hoverinfo="text",
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ))
            mid = len(xs) // 2
            figure.add_trace(go.Scatter(
                x=[xs[mid]], y=[ys[mid]], mode="markers",
                marker=dict(size=15, color="rgba(0,0,0,0.01)"),
                hovertext=[build_edge_hover(source, target, key, attrs)],
                hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
            ))

    for face_id, edges in self_edges.items():
        cx, cy = positions[face_id]
        for loop_index, (key, attrs) in enumerate(sorted(edges, key=lambda item: item[0])):
            xs, ys = self_loop_points(cx, cy, loop_index)
            figure.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=FAG_EDGE_WIDTH, color="rgba(55,55,55,0.75)"),
                text=[build_edge_hover(face_id, face_id, key, attrs) for _ in xs],
                hoverinfo="text",
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ))
            top_index = max(range(len(ys)), key=lambda i: ys[i])
            figure.add_trace(go.Scatter(
                x=[xs[top_index]], y=[ys[top_index]], mode="markers",
                marker=dict(size=15, color="rgba(0,0,0,0.01)"),
                hovertext=[build_edge_hover(face_id, face_id, key, attrs)],
                hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
            ))

    node_x, node_y, node_colors, node_sizes, node_text, node_hover = [], [], [], [], [], []
    for face_id in sorted(graph.nodes):
        face = face_lookup[face_id]
        x, y = positions[face_id]
        if label_mode == "ground_truth":
            category_id = int(face.get("ground_truth_category_id", 24))
            category_name = face.get("ground_truth_category_name")
            instance_id = face.get("ground_truth_instance_id")
            color = category_color(category_id)
        else:
            category_id = int(face.get("predicted_category_id", 24))
            category_name = face.get("predicted_category_name")
            instance_id = face.get("predicted_instance_id")
            color = category_color(category_id)
        node_x.append(x)
        node_y.append(y)
        node_colors.append(rgb_float_to_css(color))
        node_sizes.append(FAG_NODE_SIZE if category_id == 24 else FAG_FEATURE_NODE_SIZE)
        node_text.append(str(face_id))
        node_hover.append(
            f"<b>Face {face_id}</b>"
            f"<br>Face type: {face.get('face_type')}"
            f"<br>Label: {category_name} ({category_id})"
            f"<br>Instance: {instance_id}"
            f"<br>Bottom: {face.get('ground_truth_bottom')}"
            f"<br>Seam edge count: {face.get('seam_edge_count')}"
            f"<br>Graph degree: {graph.degree(face_id)}"
        )

    figure.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1.5, color="black")),
        text=node_text, textposition="middle center", textfont=dict(size=9),
        hovertext=node_hover, hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
    ))
    title = "Ground-truth attributed FAG" if label_mode == "ground_truth" else "Seed-extractor attributed FAG"
    figure.update_layout(
        title=dict(text=title, x=0.5), template="plotly_white",
        margin=dict(l=5, r=5, t=42, b=5), hovermode="closest", dragmode="pan",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        uirevision=f"fag-{cache.get('part_name')}",
    )
    return figure


# ============================================================
# 页面
# ============================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Feature Seed Extractor"

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col([
            html.H5("Feature Seed Extractor", className="mb-0"),
        ], width=2, className="d-flex align-items-center"),
        dbc.Col([
            dcc.RadioItems(
                id="mode-selector",
                options=[
                    {"label": "Single mode", "value": "single"},
                    {"label": "Batch mode", "value": "batch"},
                ],
                value=INITIAL_MODE,
                inline=False,
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "2px"},
            ),
        ], width=1, className="d-flex align-items-center"),
        dbc.Col([
            dcc.RadioItems(
                id="extraction-mode-selector",
                options=[
                    {"label": "Normal", "value": "normal"},
                    {"label": "Space", "value": "space"},
                ],
                value=INITIAL_EXTRACTION_MODE,
                inline=False,
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "2px"},
            ),
        ], width=1, className="d-flex align-items-center"),
        dbc.Col([
            html.Div("Prediction result", className="small fw-bold mb-1"),
            dcc.Dropdown(
                id="outcome-selector",
                options=OUTCOME_OPTIONS,
                value=INITIAL_OUTCOME,
                clearable=False,
                maxHeight=300,
                optionHeight=35,
            ),
        ], width=2),
        dbc.Col([
            html.Div("Machining feature", className="small fw-bold mb-1"),
            dcc.Dropdown(
                id="feature-selector",
                options=INITIAL_FEATURE_OPTIONS,
                value=INITIAL_FEATURE_VALUE,
                clearable=False,
                maxHeight=420,
                optionHeight=35,
            ),
        ], width=2),
        dbc.Col([
            html.Div("Matched samples", className="small fw-bold mb-1"),
            dcc.Dropdown(
                id="part-selector",
                options=INITIAL_CACHE_OPTIONS,
                value=(INITIAL_CACHE_OPTIONS[0]["value"] if INITIAL_CACHE_OPTIONS else None),
                clearable=False,
                maxHeight=520,
                optionHeight=35,
                searchable=True,
            ),
        ], width=4),
    ], className="align-items-center gx-2 px-2 py-1 border rounded bg-light"),
    dbc.Row([
        dbc.Col([
            html.Div(id="part-summary", className="mt-1"),
            html.Div(id="batch-summary", className="small text-muted"),
        ], width=12),
    ]),
    dcc.Store(id="drag-mode-store", data=INITIAL_DRAG_MODE),
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(
                    id="part-viewer",
                    style={"height": "86vh"},
                    config={"displaylogo": False, "scrollZoom": True, "responsive": True},
                ),
                dbc.ButtonGroup([
                    dbc.Button(
                        "Rotate mode",
                        id="rotate-button",
                        color="primary",
                        size="sm",
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Pan mode",
                        id="pan-button",
                        color="secondary",
                        size="sm",
                        n_clicks=0,
                    ),
                ], style={
                    "position": "absolute",
                    "top": "44px",
                    "right": "12px",
                    "zIndex": 5,
                }),
                dcc.Checklist(id="space-layers", options=[{"label": " Finite collision domain", "value": "finite_collision_domain"}, {"label": " Active boundary", "value": "active_boundary"}], value=[], inline=True,
                              style={"position": "absolute", "top": "12px", "right": "12px", "zIndex": 5, "backgroundColor": "rgba(255,255,255,0.9)", "fontSize": "13px"}),
            ], style={"position": "relative"}),
        ], width=5),
        dbc.Col([
            dcc.Graph(
                id="ground-truth-fag",
                style={"height": "42vh"},
                config={"displaylogo": False, "scrollZoom": True, "responsive": True},
            ),
            dcc.Graph(
                id="predicted-fag",
                style={"height": "42vh"},
                config={"displaylogo": False, "scrollZoom": True, "responsive": True},
            ),
        ], width=4),
        dbc.Col([
            html.H6("Selected face connections", className="mt-2"),
            html.Div(
                id="selected-face-panel",
                children=dbc.Alert(
                    "Click a face in the 3D model to inspect its incident edges.",
                    color="light",
                ),
                style={"maxHeight": "27vh", "overflowY": "auto"},
            ),
            html.Div([
                html.H6(
                    "Extracted feature instances",
                    className="mb-0",
                    style={"fontSize": "0.82rem", "whiteSpace": "nowrap"},
                ),
                dbc.Select(
                    id="instance-variant-selector",
                    options=[
                        {"label": "Basic", "value": "basic"},
                        {"label": "Generalized", "value": "generalized"},
                    ],
                    value="basic",
                    size="sm",
                    style={"width": "105px"},
                ),
            ], className="d-flex align-items-center justify-content-between mt-2"),
            html.Div(
                id="instance-panel",
                style={"maxHeight": "52vh", "overflowY": "auto"},
            ),
        ], width=3),
    ]),
])


@app.callback(
    Output("feature-selector", "options"),
    Output("feature-selector", "value"),
    Input("mode-selector", "value"),
    Input("extraction-mode-selector", "value"),
    Input("outcome-selector", "value"),
)
def update_feature_filter(mode: str, extraction_mode: str, outcome: str):
    options = feature_options(mode, outcome, extraction_mode)
    return options, options[0]["value"] if options else None


@app.callback(
    Output("part-selector", "options"),
    Output("part-selector", "value"),
    Output("batch-summary", "children"),
    Input("mode-selector", "value"),
    Input("extraction-mode-selector", "value"),
    Input("outcome-selector", "value"),
    Input("feature-selector", "value"),
    State("part-selector", "value"),
)
def update_sample_filter(
    mode: str,
    extraction_mode: str,
    outcome: str,
    feature_key: str,
    current_sample: str,
):
    if mode == "single":
        # Single mode remains a direct browser for every individually processed sample.
        options = cache_options(mode, extraction_mode)
    else:
        options = filtered_cache_options(mode, outcome, feature_key, extraction_mode) if feature_key else []
    values = {option["value"] for option in options}
    selected_value = current_sample if current_sample in values else (
        options[0]["value"] if options else None
    )
    matched_text = f"Matched samples: {len(options)}"

    if mode != "batch":
        return options, selected_value, f"Single {extraction_mode} samples: {len(options)}"

    summary_path = (
        SPACE_BATCH_SUMMARY_PATH
        if extraction_mode == "space"
        else BATCH_SUMMARY_PATH
    )
    if not summary_path.exists():
        return options, selected_value, f"{matched_text} | Batch summary is not available yet."

    summary = load_cache(summary_path)
    evaluation = summary.get("evaluation", {})
    exact_instance = evaluation.get("exact_instance", {})
    text = (
        f"{matched_text} | Batch samples: {summary.get('successful_sample_count', 0)}/"
        f"{summary.get('selected_sample_count', 0)} | "
        f"Batch exact-instance precision/recall: {exact_instance.get('precision', 0.0):.4f}/"
        f"{exact_instance.get('recall', 0.0):.4f}"
    )
    return options, selected_value, text


@app.callback(
    Output("drag-mode-store", "data"),
    Input("rotate-button", "n_clicks"),
    Input("pan-button", "n_clicks"),
    State("drag-mode-store", "data"),
    prevent_initial_call=True,
)
def update_drag_mode(rotate_clicks: int, pan_clicks: int, current_mode: str) -> str:
    if dash.ctx.triggered_id == "rotate-button":
        return "orbit"
    if dash.ctx.triggered_id == "pan-button":
        return "pan"
    return current_mode


def build_instance_panel(
    cache: Dict[str, Any],
    variant_filter: str,
) -> List[Any]:
    instances = cache.get("instances", [])
    if cache.get("extraction_mode") == "space":
        instances = list(instances)
    elif variant_filter == "generalized":
        instances = [
            instance
            for instance in instances
            if instance.get("variant") == "generalized"
        ]
    else:
        instances = [
            instance
            for instance in instances
            if instance.get("variant") in {"basic", "single_face"}
        ]
    if not instances:
        label = "generalized" if variant_filter == "generalized" else "basic"
        return [dbc.Alert(f"No {label} feature instances were recognized.", color="secondary")]
    cards = []
    for instance in instances:
        role_rows = [
            html.Div([html.Strong(f"{role}: "), html.Span(str(face_ids))])
            for role, face_ids in instance.get("role_groups", {}).items()
        ]
        accepted_expansion = sum(
            bool(record.get("accepted"))
            for record in instance.get("expansion_records", [])
        )
        cards.append(dbc.Card(dbc.CardBody([
            html.H6(f"Instance {instance.get('instance_id')}: {instance.get('category_name')}"),
            html.Div(f"Category ID: {instance.get('category_id')}"),
            html.Div(f"Variant: {instance.get('variant')}"),
            html.Div(f"Basic faces: {instance.get('basic_face_ids')}"),
            html.Div(f"Final faces: {instance.get('face_ids')}"),
            html.Div(f"Accepted expansion faces: {accepted_expansion}"),
            html.Hr(),
            html.Div(role_rows),
        ]), className="mb-2"))
    return cards


def build_selected_face_panel(cache: Dict[str, Any], face_id: int) -> Any:
    face = build_face_lookup(cache).get(face_id)
    if face is None:
        return dbc.Alert("The selected face is not present in this sample.", color="warning")
    incident_edges = sorted(
        face.get("incident_edges", []),
        key=lambda edge: int(edge.get("edge_key", -1)),
    )
    if not incident_edges:
        return dbc.Alert(f"Face {face_id} has no incident edge records.", color="secondary")

    rows = []
    for edge in incident_edges:
        rows.append(dbc.ListGroupItem([
            html.Div([
                html.Strong(f"E{edge.get('edge_key')}"),
                html.Span(
                    f"Face {face_id} ↔ Face {edge.get('other_face_id')}",
                    className="ms-2",
                ),
            ]),
            html.Div(
                f"{edge.get('edge_type')} | {edge.get('convexity')} | "
                f"{edge.get('dihedral_type')} | "
                f"{format_angle(edge.get('dihedral_angle_degrees'))}",
                className="small",
            ),
            html.Div(
                f"Seam: {edge.get('is_seam')}",
                className="small text-muted",
            ),
        ], className="py-2"))
    return html.Div([
        html.Div(f"Face {face_id} · {face.get('face_type')}", className="fw-bold mb-1"),
        dbc.ListGroup(rows, flush=True),
    ])


@app.callback(
    Output("selected-face-panel", "children"),
    Input("part-viewer", "clickData"),
    Input("part-selector", "value"),
)
def update_selected_face(click_data: Dict[str, Any], cache_path: str):
    if not cache_path:
        return dbc.Alert("No sample is selected.", color="secondary")
    if dash.ctx.triggered_id == "part-selector" or not click_data or not click_data.get("points"):
        return dbc.Alert(
            "Click a face in the 3D model to inspect its incident edges.",
            color="light",
        )
    face_id = click_data["points"][0].get("customdata")
    if face_id is None:
        return dbc.Alert("Click a rendered face to inspect its incident edges.", color="light")
    return build_selected_face_panel(load_cache(Path(cache_path)), int(face_id))


@lru_cache(maxsize=4)
def batch_sample_averages(summary_path: str, modified: int):
    """Arithmetic mean across completed samples, not pooled face accuracy."""
    summary = load_cache(Path(summary_path))
    values = {"face_accuracy": [], "machining_feature_face_accuracy": []}
    for result in summary.get("results", []):
        if result.get("status") not in {"success", "reused"}:
            continue
        evaluation = result.get("evaluation", {})
        cache_path = Path(result.get("cache_path", ""))
        if cache_path.is_file():
            evaluation = load_cache(cache_path).get("evaluation", evaluation)
        for key in values:
            value = evaluation.get(key, result.get(key))
            if value is not None:
                values[key].append(float(value))
    return {key: sum(items) / len(items) if items else None for key, items in values.items()}


def accuracy_summary(cache, extraction_mode, feature_key):
    path = SPACE_BATCH_SUMMARY_PATH if extraction_mode == "space" else BATCH_SUMMARY_PATH
    averages = batch_sample_averages(str(path), path.stat().st_mtime_ns) if path.is_file() else {}
    evaluation = cache.get("evaluation", {})
    percent = lambda value: "N/A" if value is None else f"{100 * value:.2f}%"
    text = (f"Face accuracy (sample / batch mean): {percent(evaluation.get('face_accuracy'))} / {percent(averages.get('face_accuracy'))} | "
            f"Feature face accuracy (sample / batch mean): {percent(evaluation.get('machining_feature_face_accuracy'))} / {percent(averages.get('machining_feature_face_accuracy'))}")
    try:
        category = mixed_class_ids(feature_key)[1] if str(feature_key).startswith("mixed:") else int(feature_key)
        category = 8 if category in {10, 20} else 24 if category in {0, 23} else category
        metric = next((item for item in evaluation.get("per_class", []) if item['category_id'] == category), None)
    except (ValueError, TypeError):
        metric = None
    text += f" | Class F1 (sample): {metric['category_name']} {percent(metric['f1'])}" if metric else " | Class F1 (sample): N/A"
    return text


@app.callback(
    Output("part-viewer", "figure"),
    Output("ground-truth-fag", "figure"),
    Output("predicted-fag", "figure"),
    Output("part-summary", "children"),
    Input("part-selector", "value"),
    Input("drag-mode-store", "data"),
    Input("space-layers", "value"),
    Input("feature-selector", "value"),
    Input("extraction-mode-selector", "value"),
)
def update_all(cache_path: str, drag_mode: str, space_layers=None, feature_key=None, extraction_mode="normal"):
    if not cache_path:
        empty_figure = go.Figure()
        empty_figure.update_layout(template="plotly_white")
        return (
            empty_figure,
            empty_figure,
            empty_figure,
            "No cache is available for the selected mode.",
        )

    cache = load_cache(Path(cache_path))
    graph = build_multigraph(cache)
    positions = build_shared_positions(graph)
    summary = (
        f"Part: {cache.get('part_name')} | Faces: {cache.get('face_count')} | "
        f"Predicted instances: {cache.get('feature_instance_count')} | "
        f"FAG edges: {graph.number_of_edges()}"
    )
    summary = html.Div([html.Div(summary), html.Div(accuracy_summary(cache, extraction_mode, feature_key))])
    if space_layers:
        diagnostics = cache.get("space_diagnostics", {}).get("gt_instances", [])
        if not diagnostics:
            summary.children.append(html.Div("No GT diagnostic layers cached. Re-run the single space extractor for this sample.", className="text-muted small"))
        else:
            counts = {}
            for instance in diagnostics:
                counts[instance["stage"]] = counts.get(instance["stage"], 0) + 1
            summary.children.append(html.Div("GT diagnostics: " + ", ".join(f"{stage}={count}" for stage, count in counts.items()), className="small"))
            summary.children.append(html.Div("Domains: green = exact extraction, red = collision rejection, grey = other diagnostic stage. Face-center spheres: green = exact extraction, red = overlap below threshold, yellow = passed face in an unrecovered instance, grey = unavailable. Hover shows area overlap to six decimals. Collision failures are replayed for diagnostics only; missing cells have no fabricated percentage.", className="text-muted small"))
    return (
        build_3d_figure(cache, drag_mode, space_layers),
        build_fag_figure(cache, graph, positions, "ground_truth"),
        build_fag_figure(cache, graph, positions, "predicted"),
        summary,
    )


@app.callback(
    Output("instance-panel", "children"),
    Input("part-selector", "value"),
    Input("instance-variant-selector", "value"),
)
def update_instance_panel(cache_path: str, variant_filter: str):
    if not cache_path:
        return [dbc.Alert("No extracted samples are available.", color="secondary")]
    return build_instance_panel(load_cache(Path(cache_path)), variant_filter)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
