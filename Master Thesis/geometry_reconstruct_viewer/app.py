"""Dash viewer for exact-vertex / exact-surface B-Rep reconstruction."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "web_cache"
RECONSTRUCTION_CACHE_DIR = APP_DIR / "reconstruction_cache"
GRAPH_CONFIG = {
    "scrollZoom": True, "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
FEATURE_COLORS = {
    0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3", 4: "#ff7f00",
    5: "#ffff33", 6: "#a65628", 7: "#f781bf", 8: "#999999", 9: "#66c2a5",
    10: "#fc8d62", 11: "#8da0cb", 12: "#e78ac3", 13: "#a6d854", 14: "#ffd92f",
    15: "#e5c494", 16: "#b3b3b3", 17: "#1b9e77", 18: "#d95f02", 19: "#7570b3",
    20: "#e7298a", 21: "#66a61e", 22: "#e6ab02", 23: "#a6761d", 24: "#c9c9c9",
}


@lru_cache(maxsize=16)
def load_cache(sample: str) -> dict[str, Any]:
    path = CACHE_DIR / f"{Path(sample).stem}.geometry.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache {path}; run extract_geometry_sequence.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def load_true_reconstruction(sample: str) -> dict[str, Any]:
    path = RECONSTRUCTION_CACHE_DIR / f"{Path(sample).stem}.reconstruction.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing reconstruction cache {path}; run reconstruct_brep_from_known_inputs.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _scene(bbox: list[float], camera: dict[str, Any] | None = None) -> dict[str, Any]:
    spans = [max(float(bbox[index + 3]) - float(bbox[index]), 1e-9) for index in range(3)]
    longest = max(spans)
    scene: dict[str, Any] = {
        "xaxis": {"visible": False, "range": [bbox[0], bbox[3]], "autorange": False},
        "yaxis": {"visible": False, "range": [bbox[1], bbox[4]], "autorange": False},
        "zaxis": {"visible": False, "range": [bbox[2], bbox[5]], "autorange": False},
        "aspectmode": "manual",
        "aspectratio": {"x": spans[0] / longest, "y": spans[1] / longest, "z": spans[2] / longest},
        "bgcolor": "#f7f9fc", "dragmode": "orbit",
    }
    if camera:
        scene["camera"] = camera
    return scene


def _mesh(mesh: dict[str, Any], color: str, opacity: float, name: str, hover: str) -> go.Mesh3d | None:
    vertices, triangles = mesh.get("vertices") or [], mesh.get("triangles") or []
    if not vertices or not triangles:
        return None
    x, y, z = zip(*vertices)
    i, j, k = zip(*triangles)
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity,
        flatshading=True, name=name, hovertemplate=hover + "<extra></extra>",
        lighting={"ambient": 0.68, "diffuse": 0.72, "specular": 0.12},
        lightposition={"x": 120, "y": 180, "z": 160}, showscale=False,
    )


def _has_mesh(mesh: dict[str, Any]) -> bool:
    return bool(mesh.get("vertices") and mesh.get("triangles"))


def _bbox_trace(edge: dict[str, Any]) -> go.BaseTraceType | None:
    bbox = edge.get("bbox") or {}
    minimum, maximum = bbox.get("min_corner"), bbox.get("max_corner")
    if not minimum or not maximum:
        return None
    flat_axes = set(bbox.get("flat_axes") or [])
    varying_axes = [axis for axis in range(3) if axis not in flat_axes]
    hover = (
        f"Edge e{edge['edge_id']} AABB<br>min {minimum}<br>max {maximum}"
        f"<br>dimension {bbox.get('dimension')}"
    )
    if len(varying_axes) >= 3:
        vertices = [
            [maximum[0] if x else minimum[0], maximum[1] if y else minimum[1], maximum[2] if z else minimum[2]]
            for x, y, z in itertools.product((0, 1), repeat=3)
        ]
        triangles = [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ]
        return _mesh({"vertices": vertices, "triangles": triangles}, "#2563eb", 0.20, "edge AABB", hover)
    if len(varying_axes) == 2:
        first, second = varying_axes
        vertices = []
        for first_high, second_high in ((0, 0), (1, 0), (1, 1), (0, 1)):
            point = list(minimum)
            point[first] = maximum[first] if first_high else minimum[first]
            point[second] = maximum[second] if second_high else minimum[second]
            vertices.append(point)
        return _mesh(
            {"vertices": vertices, "triangles": [[0, 1, 2], [0, 2, 3]]},
            "#2563eb", 0.24, "planar edge AABB", hover,
        )
    if len(varying_axes) == 1:
        return _line([minimum, maximum], "#2563eb", 6, "linear edge AABB", hover, 0.55)
    point = minimum
    return go.Scatter3d(
        x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
        marker={"size": 5, "color": "#2563eb"}, name="point edge AABB",
        showlegend=False, hovertemplate=hover + "<extra></extra>",
    )


def make_original_figure(cache: dict[str, Any], camera: dict[str, Any] | None = None) -> go.Figure:
    """Mirror MF Explorer's left viewer: one colored mesh per labeled STEP face."""
    fig = go.Figure()
    categories = cache["feature_categories"]
    shown: set[int] = set()
    for face in cache["face_meshes"]:
        label = face.get("face_label")
        label_name = categories[label] if label is not None and 0 <= label < len(categories) else "unknown"
        trace = _mesh(
            face, FEATURE_COLORS.get(label, "#94a3b8"), 1.0, f"{label}: {label_name}",
            f"STEP face {face['face_id']}<br>Feature {label}: {label_name}"
            f"<br>Instance {face.get('feature_instance_id')}<br>Surface {face.get('face_type')}",
        )
        if trace is None:
            continue
        trace.legendgroup = str(label)
        trace.showlegend = label not in shown
        shown.add(label)
        fig.add_trace(trace)
    fig.update_layout(
        scene=_scene(cache["view_bbox"], camera), margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="#f7f9fc", hovermode="closest", uirevision=f"original-{cache['sample_name']}",
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,.76)", "font": {"size": 9}},
    )
    return fig


def _line(points: list[list[float]], color: str, width: float, name: str, hover: str, opacity: float = 1.0) -> go.Scatter3d | None:
    if len(points) < 2:
        return None
    x, y, z = zip(*points)
    return go.Scatter3d(
        x=x, y=y, z=z, mode="lines", line={"color": color, "width": width}, opacity=opacity,
        name=name, showlegend=False, hovertemplate=hover + "<extra></extra>",
    )


def _axis_trace(origin: list[float], direction: list[float], length: float, color: str, name: str) -> go.Scatter3d:
    end = [origin[i] + direction[i] * length for i in range(3)]
    return go.Scatter3d(
        x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]], mode="lines+markers",
        line={"color": color, "width": 6}, marker={"size": [2, 5], "color": color},
        name=name, showlegend=False, hovertemplate=f"{name}<extra></extra>",
    )


def _event_state(cache: dict[str, Any], step: int) -> dict[str, Any]:
    events = cache["events"]
    step = max(0, min(int(step), len(events) - 1))
    prefix = events[: step + 1]
    state: dict[str, Any] = {
        "event": events[step], "vertices": set(), "surface_parts": {}, "intersections": set(),
        "control_points": {}, "trimmed": set(), "hide_helpers_event": False, "final_edges": False, "final_faces": False,
    }
    for event in prefix:
        kind = event["kind"]
        if kind == "vertex":
            state["vertices"].add(event["vertex_id"])
        elif kind in {"origin", "axis_z", "axis_x", "axis_y", "surface"}:
            state["surface_parts"].setdefault(event["surface_id"], set()).add(kind)
        elif kind == "control_point":
            state["control_points"].setdefault(event["surface_id"], []).append(event["point"])
        elif kind == "intersection":
            state["intersections"].add(event["edge_id"])
        elif kind == "loop_trimmed":
            loop = next(loop for loop in cache["loops"] if loop["key"] == event["loop_key"])
            state["trimmed"].update(loop["edge_ids"])
        elif kind == "hide_helpers":
            state["hide_helpers_event"] = True
        elif kind == "trimmed_edges":
            state["final_edges"] = True
        elif kind == "trimmed_faces":
            state["final_edges"] = True
            state["final_faces"] = True
    return state


def _trace_manifest(cache: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = []
    for index, feature_space in enumerate(cache.get("feature_spaces", [])):
        if _has_mesh(feature_space["mesh"]):
            manifest.append({"role": "condition", "id": index})
    for surface in cache["surfaces"]:
        surface_id = surface["id"]
        if _has_mesh(surface["mesh"]):
            manifest.append({"role": "surface", "id": surface_id})
        parameters = surface["parameters"]
        if "origin" in parameters:
            manifest.append({"role": "origin", "id": surface_id})
            for axis_name in ("axis_z", "axis_x", "axis_y"):
                if axis_name in parameters:
                    manifest.append({"role": axis_name, "id": surface_id})
        for pole_index, _point_value in enumerate(parameters.get("poles", [])):
            manifest.append({"role": "control_point", "id": surface_id, "index": pole_index})
    for vertex in cache["vertices"]:
        manifest.append({"role": "vertex", "id": vertex["id"]})
    for edge in cache["edges"]:
        manifest.append({"role": "untrimmed", "id": edge["edge_id"]})
    for edge in cache["edges"]:
        manifest.append({"role": "trimmed", "id": edge["edge_id"]})
    for face in cache["face_meshes"]:
        if _has_mesh(face):
            manifest.append({"role": "final_face", "id": face["face_id"]})
    return manifest


def _presentation(cache: dict[str, Any], step: int, show_conditions: bool, show_helpers: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = _event_state(cache, step)
    current = state["event"]
    intersection_surface_ids: set[int] = set()
    if current["kind"] == "intersection":
        edge = next(edge for edge in cache["edges"] if edge["edge_id"] == current["edge_id"])
        intersection_surface_ids = set(edge.get("surface_ids") or [])
    current_loop_edges: set[int] = set()
    if current["kind"] in {"loop_candidates", "loop_trimmed"}:
        loop = next(loop for loop in cache["loops"] if loop["key"] == current["loop_key"])
        current_loop_edges = set(loop["edge_ids"])
    trimmed_ids = {edge["edge_id"] for edge in cache["edges"]} if state["final_edges"] else state["trimmed"]
    styles = []
    for item in _trace_manifest(cache):
        role, item_id = item["role"], item["id"]
        style: dict[str, Any] = {"visible": False}
        if role == "condition":
            style["visible"] = show_conditions and not intersection_surface_ids
        elif role == "surface":
            if intersection_surface_ids:
                style["visible"] = item_id in intersection_surface_ids
                style.update({"mesh_color": "#f7d774", "opacity": 0.42})
            else:
                style["visible"] = show_helpers and "surface" in state["surface_parts"].get(item_id, set())
                style.update({"mesh_color": "#f29a91", "opacity": 0.18})
        elif role in {"origin", "axis_z", "axis_x", "axis_y"}:
            style["visible"] = (
                not intersection_surface_ids and show_helpers
                and role in state["surface_parts"].get(item_id, set())
            )
        elif role == "control_point":
            style["visible"] = show_helpers and item["index"] < len(state["control_points"].get(item_id, []))
        elif role == "vertex":
            style["visible"] = not intersection_surface_ids and item_id in state["vertices"]
        elif role == "untrimmed":
            style["visible"] = not state["final_edges"] and item_id in state["intersections"]
            style.update({
                "color": "#f6c344" if item_id in current_loop_edges else "#64748b",
                "width": 8 if item_id in current_loop_edges else 3,
                "opacity": 1.0 if item_id in current_loop_edges else 0.56,
            })
        elif role == "trimmed":
            style["visible"] = item_id in trimmed_ids
            highlighted = current["kind"] == "loop_trimmed" and item_id in current_loop_edges
            style.update({"color": "#10b9b0" if highlighted else "#222b3a", "width": 9 if highlighted else 5})
        elif role == "final_face":
            style["visible"] = state["final_faces"]
        styles.append(style)
    return styles, state


def make_master_reconstruction_figure(
    cache: dict[str, Any], step: int, show_conditions: bool, show_helpers: bool,
) -> tuple[go.Figure, dict[str, Any]]:
    fig = go.Figure()
    diagonal = math.sqrt(sum((cache["bbox"][i + 3] - cache["bbox"][i]) ** 2 for i in range(3))) or 1.0
    for feature_space in cache.get("feature_spaces", []):
        trace = _mesh(
            feature_space["mesh"], "#69d19c", 0.30, "feature-space cell",
            f"Feature-space cell {feature_space['feature_instance_id']}<br>{feature_space['category_name']}"
            f"<br>{feature_space['geometry_kind']}",
        )
        if trace:
            fig.add_trace(trace)
    for surface in cache["surfaces"]:
        surface_id, parameters = surface["id"], surface["parameters"]
        trace = _mesh(
            surface["mesh"], "#f29a91", 0.18, f"generated support s{surface_id}",
            f"Generated support s{surface_id}<br>{parameters['type']}<br>Exact parameters are available in the event panel.",
        )
        if trace:
            fig.add_trace(trace)
        if "origin" in parameters:
            origin = parameters["origin"]
            fig.add_trace(go.Scatter3d(
                x=[origin[0]], y=[origin[1]], z=[origin[2]], mode="markers",
                marker={"size": 6, "color": "#f59e0b"}, name="surface origin", showlegend=False,
                hovertemplate=f"s{surface_id} origin {origin}<extra></extra>",
            ))
            for key, color, label in (("axis_z", "#2563eb", "z"), ("axis_x", "#ef4444", "x"), ("axis_y", "#16a34a", "y")):
                if key in parameters:
                    fig.add_trace(_axis_trace(origin, parameters[key], diagonal * 0.09, color, f"s{surface_id} {label}-axis"))
        for pole_index, point in enumerate(parameters.get("poles", [])):
            fig.add_trace(go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
                marker={"size": 5, "color": "#7c3aed"}, name="surface control point", showlegend=False,
                hovertemplate=f"s{surface_id} control point {pole_index}<br>{point}<extra></extra>",
            ))
    for vertex in cache["vertices"]:
        point = vertex["xyz"]
        fig.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
            marker={"size": 4.8, "color": "#e11d48", "line": {"color": "white", "width": 0.7}},
            name="generated vertex", showlegend=False,
            hovertemplate=f"v{vertex['id']}<br>XYZ {point}<br>Q {vertex['quantized']}<extra></extra>",
        ))
    for edge in cache["edges"]:
        trace = _line(
            edge["untrimmed_points"], "#64748b", 3, "untrimmed intersection",
            f"Untrimmed candidate e{edge['edge_id']}<br>Surface pair {edge['surface_ids']}<br>{edge['derivation_mode']}", 0.56,
        )
        if trace:
            fig.add_trace(trace)
    for edge in cache["edges"]:
        trace = _line(
            edge["trimmed_points"], "#222b3a", 5, "trimmed edge",
            f"Trimmed result e{edge['edge_id']}<br>Curve {edge['edge_type']}<br>Surface pair {edge['surface_ids']}",
        )
        if trace:
            fig.add_trace(trace)
    for face in cache["face_meshes"]:
        trace = _mesh(
            face, "#e98f85", 0.72, "trimmed face",
            f"Final trimmed STEP face {face['face_id']}<br>Surface {face.get('face_type')}",
        )
        if trace:
            fig.add_trace(trace)
    styles, state = _presentation(cache, step, show_conditions, show_helpers)
    for trace, style in zip(fig.data, styles):
        trace.visible = style["visible"]
        if "opacity" in style:
            trace.opacity = style["opacity"]
        if "mesh_color" in style:
            trace.color = style["mesh_color"]
        if "color" in style:
            trace.line.color = style["color"]
            trace.line.width = style["width"]
    fig.update_layout(
        scene=_scene(cache["view_bbox"]), margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="#f7f9fc", showlegend=False, hovermode="closest",
        uirevision=f"reconstruction-{cache['sample_name']}",
    )
    return fig, state


def patch_reconstruction_figure(cache: dict[str, Any], step: int, show_conditions: bool, show_helpers: bool) -> tuple[Patch, dict[str, Any]]:
    styles, state = _presentation(cache, step, show_conditions, show_helpers)
    patch = Patch()
    for index, style in enumerate(styles):
        patch["data"][index]["visible"] = style["visible"]
        if "opacity" in style:
            patch["data"][index]["opacity"] = style["opacity"]
        if "mesh_color" in style:
            patch["data"][index]["color"] = style["mesh_color"]
        if "color" in style:
            patch["data"][index]["line"]["color"] = style["color"]
            patch["data"][index]["line"]["width"] = style["width"]
    return patch, state


def _true_manifest(source: dict[str, Any], rebuilt: dict[str, Any]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for index, feature_space in enumerate(source.get("feature_spaces", [])):
        if _has_mesh(feature_space["mesh"]):
            manifest.append({"role": "condition", "id": index})
    for surface in source["surfaces"]:
        surface_id = surface["id"]
        if _has_mesh(surface["mesh"]):
            manifest.append({"role": "surface", "id": surface_id})
        parameters = surface["parameters"]
        if "origin" in parameters:
            manifest.append({"role": "origin", "id": surface_id})
            for axis_name in ("axis_z", "axis_x", "axis_y"):
                if axis_name in parameters:
                    manifest.append({"role": axis_name, "id": surface_id})
    for vertex in rebuilt["vertices"]:
        manifest.append({"role": "vertex", "id": vertex["id"]})
    for edge in rebuilt["edges"]:
        if edge.get("bbox"):
            manifest.append({"role": "bbox_corner", "id": edge["edge_id"], "corner": 0})
            manifest.append({"role": "bbox_corner", "id": edge["edge_id"], "corner": 1})
            manifest.append({"role": "bbox_geometry", "id": edge["edge_id"]})
    for group in rebuilt["intersection_groups"]:
        for branch in group["branches"]:
            manifest.append({"role": "intersection", "id": group["id"], "branch": branch["id"]})
    for edge in rebuilt["edges"]:
        if edge["candidates"]:
            manifest.append({"role": "committed_edge", "id": edge["edge_id"]})
    for loop_index, loop in enumerate(rebuilt["loop_results"]):
        if not loop["ambiguous"]:
            continue
        for solution in loop["solutions"]:
            manifest.append({"role": "candidate_solution", "id": loop_index, "solution": solution["id"]})
        seam_edge_ids = sorted({
            occurrence["edge_id"] for occurrence in loop["occurrences"]
            if occurrence.get("status") == "seam_or_cosurface_skipped"
        })
        for edge_id in seam_edge_ids:
            manifest.append({"role": "candidate_seam", "id": loop_index, "edge_id": edge_id})
    for face in rebuilt["reconstructed_face_meshes"]:
        if _has_mesh(face):
            manifest.append({"role": "rebuilt_face", "id": face["face_id"]})
    return manifest


def _true_event_state(rebuilt: dict[str, Any], step: int, candidate_view: bool) -> dict[str, Any]:
    events = rebuilt["events"]
    step = max(0, min(int(step), len(events) - 1))
    state: dict[str, Any] = {
        "event": events[step], "groups": set(), "committed_loops": set(),
        "helpers_hidden": False, "final": False, "candidate_loop": None,
        "candidate_view": bool(candidate_view), "vertices": set(),
        "surface_parts": {}, "bbox_corners": set(),
    }
    for event in events[: step + 1]:
        kind = event["kind"]
        if kind == "intersection_group":
            state["groups"].add(event["intersection_group_id"])
        elif kind == "vertex":
            state["vertices"].add(event["vertex_id"])
        elif kind in {"origin", "axis_z", "axis_x", "axis_y", "surface"}:
            state["surface_parts"].setdefault(event["surface_id"], set()).add(kind)
        elif kind == "edge_bbox_corner":
            state["bbox_corners"].add((event["edge_id"], event["corner_index"]))
        elif kind == "hide_surface_helpers":
            state["helpers_hidden"] = True
        elif kind == "loop_commit":
            state["committed_loops"].add(event["loop_index"])
        elif kind == "reconstruction_final":
            state["final"] = True
    if state["event"]["kind"] in {"loop_candidates", "ambiguity", "loop_commit"}:
        state["candidate_loop"] = state["event"]["loop_index"]
    return state


def _true_presentation(
    source: dict[str, Any], rebuilt: dict[str, Any], step: int, show_conditions: bool,
    show_helpers: bool, candidate_view: bool, selected_candidate: int | None = None,
    show_current_highlight: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = _true_event_state(rebuilt, step, candidate_view)
    current_loop = state["candidate_loop"]
    intersection_surface_ids: set[int] = set()
    if state["event"]["kind"] == "intersection_group":
        group = rebuilt["intersection_groups"][state["event"]["intersection_group_id"]]
        intersection_surface_ids = set(group["surface_ids"])
    current_edge_ids: set[int] = set()
    if current_loop is not None:
        current_edge_ids = {item["edge_id"] for item in rebuilt["loop_results"][current_loop]["occurrences"]}
    committed_edges: set[int] = set()
    for loop_index in state["committed_loops"]:
        committed_edges.update(item["edge_id"] for item in rebuilt["loop_results"][loop_index]["occurrences"])
    styles: list[dict[str, Any]] = []
    for item in _true_manifest(source, rebuilt):
        role, item_id = item["role"], item["id"]
        style: dict[str, Any] = {"visible": False}
        if role == "condition":
            style["visible"] = show_conditions and not intersection_surface_ids
        elif role == "surface":
            if intersection_surface_ids:
                style["visible"] = item_id in intersection_surface_ids
                style.update({"mesh_color": "#f7d774", "opacity": 0.42})
            else:
                style["visible"] = (
                    show_helpers and not state["helpers_hidden"]
                    and "surface" in state["surface_parts"].get(item_id, set())
                )
                style.update({"mesh_color": "#f29a91", "opacity": 0.16})
        elif role in {"origin", "axis_z", "axis_x", "axis_y"}:
            style["visible"] = (
                show_helpers and not state["helpers_hidden"] and not intersection_surface_ids
                and role in state["surface_parts"].get(item_id, set())
            )
        elif role == "vertex":
            style["visible"] = not intersection_surface_ids and item_id in state["vertices"]
        elif role == "bbox_corner":
            generated_now = (
                state["event"].get("phase") == "geometry"
                and (item_id, item["corner"]) in state["bbox_corners"]
            )
            candidate_relevant = (
                state["event"]["kind"] == "ambiguity" and state["candidate_view"]
                and item_id in current_edge_ids
            )
            style["visible"] = generated_now or candidate_relevant
        elif role == "bbox_geometry":
            style["visible"] = (
                state["event"]["kind"] == "ambiguity" and state["candidate_view"]
                and item_id in current_edge_ids
            )
        elif role == "intersection":
            group_edges = set(rebuilt["intersection_groups"][item_id]["edge_ids"])
            highlighted = bool(group_edges & current_edge_ids)
            replace_highlight = highlighted and (
                selected_candidate is not None or not show_current_highlight
            )
            style["visible"] = not replace_highlight and not state["final"] and item_id in state["groups"]
            style.update({
                "color": "#f59e0b" if highlighted else "#718096",
                "width": 8 if highlighted else 3,
                "opacity": 1.0 if highlighted else 0.46,
            })
        elif role == "committed_edge":
            style["visible"] = state["final"] or item_id in committed_edges
            style.update({"color": "#0f766e", "width": 5})
        elif role == "candidate_solution":
            in_current_list = state["event"]["kind"] == "ambiguity" and state["candidate_view"] and item_id == current_loop
            if in_current_list:
                style["visible"] = True if item["solution"] == selected_candidate else "legendonly"
        elif role == "candidate_seam":
            style["visible"] = (
                state["event"]["kind"] == "ambiguity" and state["candidate_view"] and item_id == current_loop
            )
        elif role == "rebuilt_face":
            style["visible"] = state["final"]
        styles.append(style)
    return styles, state


def make_master_true_reconstruction_figure(
    source: dict[str, Any], rebuilt: dict[str, Any], step: int, show_conditions: bool,
    show_helpers: bool, candidate_view: bool, selected_candidate: int | None = None,
    show_current_highlight: bool = True,
) -> tuple[go.Figure, dict[str, Any]]:
    fig = go.Figure()
    diagonal = math.sqrt(sum((source["bbox"][i + 3] - source["bbox"][i]) ** 2 for i in range(3))) or 1.0
    for feature_space in source.get("feature_spaces", []):
        trace = _mesh(feature_space["mesh"], "#69d19c", 0.30, "feature-space cell", "Condition feature-space cell")
        if trace:
            fig.add_trace(trace)
    for surface in source["surfaces"]:
        surface_id, parameters = surface["id"], surface["parameters"]
        trace = _mesh(surface["mesh"], "#f29a91", 0.16, f"support s{surface_id}", f"Exact supporting surface s{surface_id}")
        if trace:
            fig.add_trace(trace)
        if "origin" in parameters:
            origin = parameters["origin"]
            fig.add_trace(go.Scatter3d(
                x=[origin[0]], y=[origin[1]], z=[origin[2]], mode="markers",
                marker={"size": 6, "color": "#f59e0b"}, showlegend=False,
                hovertemplate=f"s{surface_id} origin {origin}<extra></extra>",
            ))
            for key, color, label in (("axis_z", "#2563eb", "z"), ("axis_x", "#ef4444", "x"), ("axis_y", "#16a34a", "y")):
                if key in parameters:
                    fig.add_trace(_axis_trace(origin, parameters[key], diagonal * 0.09, color, f"s{surface_id} {label}-axis"))
    for vertex in rebuilt["vertices"]:
        point = vertex["xyz"]
        fig.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
            marker={"size": 4.8, "color": "#e11d48", "line": {"color": "white", "width": 0.7}},
            showlegend=False, hovertemplate=f"Known v{vertex['id']}<br>XYZ {point}<extra></extra>",
        ))
    for edge in rebuilt["edges"]:
        bbox = edge.get("bbox")
        if not bbox:
            continue
        for corner_index, corner_name in enumerate(("min_corner", "max_corner")):
            point = bbox[corner_name]
            fig.add_trace(go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
                marker={"size": 4.8, "color": "#2563eb", "line": {"color": "white", "width": 0.7}},
                name="edge bbox corner", showlegend=False,
                hovertemplate=(
                    f"e{edge['edge_id']} bbox corner {corner_index + 1}<br>XYZ {point}<extra></extra>"
                ),
            ))
        bbox_trace = _bbox_trace(edge)
        if bbox_trace:
            fig.add_trace(bbox_trace)
    for group in rebuilt["intersection_groups"]:
        for branch in group["branches"]:
            trace = _line(
                branch["points"], "#718096", 3, "calculated full intersection",
                f"Calculated surfaces {group['surface_ids']}<br>Branch {branch['id']}<br>{group['method']}", 0.46,
            )
            if trace:
                fig.add_trace(trace)
    for edge in rebuilt["edges"]:
        if not edge["candidates"]:
            continue
        candidate = edge["candidates"][0]
        trace = _line(
            candidate["points"], "#0f766e", 5, "previously committed default edge",
            f"Previously committed e{edge['edge_id']}<br>Default trim hypothesis: candidate 0 (shortest)",
        )
        if trace:
            fig.add_trace(trace)
    palette = ["#7c3aed", "#db2777", "#0891b2", "#65a30d", "#ea580c", "#4f46e5", "#be123c", "#0d9488"]
    for loop_index, loop in enumerate(rebuilt["loop_results"]):
        if not loop["ambiguous"]:
            continue
        for solution in loop["solutions"]:
            is_selected = solution["id"] == loop.get("selected_solution_id", 0)
            x: list[Any] = []; y: list[Any] = []; z: list[Any] = []
            for segment in solution["segments"]:
                for point in segment["points"]:
                    x.append(point[0]); y.append(point[1]); z.append(point[2])
                x.append(None); y.append(None); z.append(None)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines",
                line={"color": palette[solution["id"] % len(palette)], "width": 8},
                name=f"{'★ ' if is_selected else ''}candidate {solution['id'] + 1}", showlegend=True,
                hovertemplate=(
                    f"Loop {loop['key']} candidate {solution['id'] + 1}"
                    f"{' — selected' if is_selected else ''}<br>"
                    f"Length {solution['total_length']}<extra></extra>"
                ),
            ))
        source_edges = {edge["edge_id"]: edge for edge in source["edges"]}
        seam_edge_ids = sorted({
            occurrence["edge_id"] for occurrence in loop["occurrences"]
            if occurrence.get("status") == "seam_or_cosurface_skipped"
        })
        for edge_id in seam_edge_ids:
            ground_truth = source_edges.get(edge_id, {})
            trace = _line(
                ground_truth.get("trimmed_points") or [], "#7b8494", 7,
                "ground-truth seam closure",
                f"Seam e{edge_id}<br>Ground-truth geometry used only to close the displayed candidate loop",
            )
            if trace:
                fig.add_trace(trace)
    for face in rebuilt["reconstructed_face_meshes"]:
        trace = _mesh(
            face, "#e98f85", 0.76, "reconstructed trimmed face",
            f"Reconstructed face {face['face_id']}<br>Surface {face['surface_id']}<br>{face.get('reconstruction_method', 'OCC wire build')}",
        )
        if trace:
            fig.add_trace(trace)
    styles, state = _true_presentation(
        source, rebuilt, step, show_conditions, show_helpers, candidate_view, selected_candidate,
        show_current_highlight,
    )
    for trace, style in zip(fig.data, styles):
        trace.visible = style["visible"]
        if "opacity" in style:
            trace.opacity = style["opacity"]
        if "mesh_color" in style:
            trace.color = style["mesh_color"]
        if "color" in style:
            trace.line.color = style["color"]
            trace.line.width = style["width"]
    fig.update_layout(
        scene=_scene(source["view_bbox"]), margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="#f7f9fc", showlegend=False, hovermode="closest",
        uirevision=f"true-reconstruction-{rebuilt['sample_name']}",
        legend={
            "x": 0.985, "y": 0.5, "xanchor": "right", "yanchor": "middle",
            "bgcolor": "rgba(255,255,255,.88)", "bordercolor": "#d8dee8", "borderwidth": 1,
            "itemclick": "toggle", "itemdoubleclick": False,
        },
    )
    return fig, state


def patch_true_reconstruction_figure(
    source: dict[str, Any], rebuilt: dict[str, Any], step: int, show_conditions: bool,
    show_helpers: bool, candidate_view: bool, selected_candidate: int | None = None,
    show_current_highlight: bool = True,
) -> tuple[Patch, dict[str, Any]]:
    styles, state = _true_presentation(
        source, rebuilt, step, show_conditions, show_helpers, candidate_view, selected_candidate,
        show_current_highlight,
    )
    patch = Patch()
    for index, style in enumerate(styles):
        patch["data"][index]["visible"] = style["visible"]
        if "opacity" in style:
            patch["data"][index]["opacity"] = style["opacity"]
        if "mesh_color" in style:
            patch["data"][index]["color"] = style["mesh_color"]
        if "color" in style:
            patch["data"][index]["line"]["color"] = style["color"]
            patch["data"][index]["line"]["width"] = style["width"]
    patch["layout"]["showlegend"] = bool(candidate_view)
    return patch, state


def make_app() -> Dash:
    samples = sorted(path.name.removesuffix(".geometry.json") for path in CACHE_DIR.glob("*.geometry.json"))
    if not samples:
        raise FileNotFoundError("No extracted samples. Run extract_geometry_sequence.py first.")
    app = Dash(__name__, assets_folder=str(APP_DIR / "assets"))
    app.title = "Geometry Reconstruction Viewer"
    initial_source = load_cache(samples[0])
    initial_right_figure, _initial_state = make_master_reconstruction_figure(
        initial_source, 0, True, True,
    )
    app.layout = html.Div([
        dcc.Store(id="generation-step", data=0),
        dcc.Store(id="candidate-view", data=False),
        dcc.Store(id="selected-candidate", data=None),
        dcc.Store(id="show-current-highlight", data=True),
        dcc.Store(id="saved-camera", data=None),
        dcc.Interval(id="autoplay-interval", interval=720, n_intervals=0, disabled=True),
        html.Header([
            html.Div([
                html.Div("EXACT B-REP LAB", className="eyebrow"),
                html.H1("Geometry Reconstruction Viewer"),
                html.P("Exact vertices + exact supporting surfaces + CoAG → post-processed edges, loops, and trimmed faces"),
            ]),
            html.Div([
                html.Div([
                    html.Label("Playback data"),
                    dcc.RadioItems(
                        id="viewer-mode",
                        options=[
                            {"label": "Ground truth", "value": "ground_truth"},
                            {"label": "Reconstruction", "value": "reconstruction"},
                        ], value="ground_truth", inline=True, className="mode-toggle",
                    ),
                ], className="mode-picker"),
                html.Div([
                    html.Label("STEP sample"),
                    dcc.Dropdown(id="sample-picker", options=[{"label": sample, "value": sample} for sample in samples], value=samples[0], clearable=False),
                ], className="sample-picker"),
            ], className="topbar-controls"),
        ], className="topbar"),
        html.Main([
            html.Section([
                html.Div([html.Span("01"), html.Div([html.Strong("Ground-truth STEP"), html.Small("MF Explorer face rendering and labels")])], className="panel-heading"),
                dcc.Graph(id="original-graph", config=GRAPH_CONFIG, className="viewer-graph"),
            ], className="viewer-panel"),
            html.Section([
                html.Div([
                    html.Div([html.Span("02"), html.Div([html.Strong(id="right-panel-title"), html.Small(id="right-panel-subtitle")])], className="panel-heading"),
                    html.Div([
                        dcc.Checklist(id="show-conditions", options=[{"label": "Feature-space supports", "value": "show"}], value=["show"], className="switch"),
                        dcc.Checklist(id="show-helpers", options=[{"label": "Surface parameters", "value": "show"}], value=["show"], className="switch"),
                    ], className="visibility-controls"),
                ], className="right-heading"),
                dcc.Graph(
                    id="reconstruction-graph", figure=initial_right_figure,
                    config=GRAPH_CONFIG, className="viewer-graph",
                ),
                html.Div([
                    html.Button("←", id="previous-step", n_clicks=0, title="Previous frame"),
                    html.Button("→", id="next-step", n_clicks=0, title="Next frame"),
                    html.Button("▶ Auto", id="autoplay-button", n_clicks=0, className="primary-button"),
                    html.Button("Final", id="final-step", n_clicks=0),
                    dcc.RadioItems(
                        id="playback-speed",
                        options=[{"label": "1×", "value": 1}, {"label": "2×", "value": 2}],
                        value=1, inline=True, className="speed-control",
                    ),
                    dcc.Slider(id="step-slider", min=0, max=1, step=1, value=0, tooltip={"placement": "top"}, className="step-slider"),
                ], className="playback-controls"),
                html.Div([
                    html.Button("Show candidates", id="candidate-button", n_clicks=0, className="candidate-button"),
                    html.Button(
                        "Hide highlighted intersections", id="highlight-button", n_clicks=0,
                        className="highlight-button",
                    ),
                    html.Span(id="candidate-hint"),
                ], id="candidate-controls", className="candidate-controls", style={"display": "none"}),
                html.Div([
                    html.Div([html.Div(id="event-title", className="event-title"), html.Div(id="event-detail", className="event-detail")]),
                    html.Div(id="frame-counter", className="frame-counter"),
                ], className="event-card"),
            ], className="viewer-panel reconstruction-panel"),
        ], className="viewer-grid"),
    ], className="app-shell")

    @app.callback(
        Output("generation-step", "data"), Output("show-helpers", "value"), Output("step-slider", "value"),
        Output("candidate-view", "data"),
        Input("viewer-mode", "value"), Input("sample-picker", "value"),
        Input("previous-step", "n_clicks"), Input("next-step", "n_clicks"),
        Input("autoplay-button", "n_clicks"), Input("final-step", "n_clicks"), Input("autoplay-interval", "n_intervals"),
        Input("step-slider", "value"), Input("candidate-button", "n_clicks"),
        State("generation-step", "data"), State("show-helpers", "value"), State("candidate-view", "data"),
        prevent_initial_call=True,
    )
    def change_step(
        mode: str, sample: str, _prev: int, _next: int, _auto: int, _final: int, _tick: int,
        slider: int, _candidate_clicks: int, current: int, helpers: list[str], candidate_view: bool,
    ):
        cache = load_cache(sample) if mode == "ground_truth" else load_true_reconstruction(sample)
        total = len(cache["events"])
        current = max(0, min(int(current or 0), total - 1))
        trigger = ctx.triggered_id
        if trigger in {"sample-picker", "viewer-mode"}:
            return 0, ["show"], 0, False
        current_event = cache["events"][current]
        if trigger == "candidate-button" and mode == "reconstruction" and current_event["kind"] == "ambiguity":
            if not candidate_view:
                return current, helpers, current, True
            target = min(total - 1, current + 1)
            return target, helpers, target, False
        if trigger == "previous-step":
            target = max(0, current - 1)
        elif trigger in {"next-step", "autoplay-interval"}:
            if trigger == "autoplay-interval" and mode == "reconstruction" and current_event["kind"] == "ambiguity":
                return current, helpers, current, candidate_view
            target = min(total - 1, current + 1)
        elif trigger == "autoplay-button":
            target = 0 if current >= total - 1 else current
        elif trigger == "final-step":
            target = total - 1
        elif trigger == "step-slider":
            target = max(0, min(int(slider or 0), total - 1))
        else:
            target = current
        first_intersection = next(
            (index for index, event in enumerate(cache["events"]) if event["kind"] in {"intersection", "intersection_group"}),
            total,
        )
        if trigger in {"next-step", "autoplay-interval", "step-slider", "final-step"} and current < first_intersection <= target:
            helpers = []
        return target, helpers, target, False

    @app.callback(
        Output("autoplay-interval", "disabled"), Output("autoplay-button", "children"),
        Input("autoplay-button", "n_clicks"), Input("generation-step", "data"),
        Input("previous-step", "n_clicks"), Input("next-step", "n_clicks"), Input("final-step", "n_clicks"),
        Input("sample-picker", "value"), Input("viewer-mode", "value"),
        State("autoplay-interval", "disabled"), prevent_initial_call=True,
    )
    def control_autoplay(_auto: int, step: int, _prev: int, _next: int, _final: int, sample: str, mode: str, disabled: bool):
        cache = load_cache(sample) if mode == "ground_truth" else load_true_reconstruction(sample)
        total = len(cache["events"])
        step = max(0, min(int(step or 0), total - 1))
        if mode == "reconstruction" and cache["events"][step]["kind"] == "ambiguity":
            return True, "Paused: ambiguity"
        if ctx.triggered_id == "autoplay-button":
            play = bool(disabled)
            return not play, "⏸ Pause" if play else "▶ Auto"
        if ctx.triggered_id in {"previous-step", "next-step", "final-step", "sample-picker", "viewer-mode"} or step >= total - 1:
            return True, "↻ Replay" if int(step or 0) >= total - 1 else "▶ Auto"
        return disabled, "▶ Auto" if disabled else "⏸ Pause"

    @app.callback(Output("autoplay-interval", "interval"), Input("playback-speed", "value"))
    def set_playback_speed(speed: int):
        return 360 if int(speed or 1) == 2 else 720

    @app.callback(
        Output("show-current-highlight", "data"),
        Input("highlight-button", "n_clicks"), Input("generation-step", "data"),
        Input("sample-picker", "value"), Input("viewer-mode", "value"),
        State("show-current-highlight", "data"), prevent_initial_call=True,
    )
    def toggle_current_highlight(_clicks: int, _step: int, _sample: str, _mode: str, current: bool):
        if ctx.triggered_id == "highlight-button":
            return not bool(current)
        return True

    @app.callback(
        Output("saved-camera", "data"),
        Input("reconstruction-graph", "relayoutData"), State("saved-camera", "data"),
        prevent_initial_call=True,
    )
    def remember_camera(relayout_data: Any, current_camera: dict[str, Any] | None):
        if not relayout_data:
            return no_update
        if isinstance(relayout_data.get("scene.camera"), dict):
            return relayout_data["scene.camera"]
        camera = dict(current_camera or {})
        changed = False
        for key, value in relayout_data.items():
            if not key.startswith("scene.camera."):
                continue
            path = key.removeprefix("scene.camera.").split(".")
            target = camera
            for part in path[:-1]:
                target = target.setdefault(part, {})
            target[path[-1]] = value
            changed = True
        return camera if changed else no_update

    @app.callback(Output("original-graph", "figure"), Input("sample-picker", "value"))
    def render_original(sample: str):
        return make_original_figure(load_cache(sample))

    @app.callback(
        Output("selected-candidate", "data"),
        Input("reconstruction-graph", "restyleData"), Input("candidate-view", "data"),
        Input("viewer-mode", "value"), Input("sample-picker", "value"), Input("generation-step", "data"),
        prevent_initial_call=True,
    )
    def select_candidate(restyle_data: Any, candidate_view: bool, mode: str, sample: str, step: int):
        if ctx.triggered_id != "reconstruction-graph" or mode != "reconstruction" or not candidate_view:
            return None
        rebuilt = load_true_reconstruction(sample)
        step = max(0, min(int(step or 0), len(rebuilt["events"]) - 1))
        if rebuilt["events"][step]["kind"] != "ambiguity" or not restyle_data or len(restyle_data) < 2:
            return no_update
        indices = restyle_data[1]
        trace_index = indices[0] if isinstance(indices, list) else indices
        manifest = _true_manifest(load_cache(sample), rebuilt)
        if not isinstance(trace_index, int) or trace_index < 0 or trace_index >= len(manifest):
            return no_update
        item = manifest[trace_index]
        if item["role"] != "candidate_solution" or item["id"] != rebuilt["events"][step]["loop_index"]:
            return no_update
        visibility = restyle_data[0].get("visible")
        if isinstance(visibility, list):
            visibility = visibility[0] if visibility else None
        return item["solution"] if visibility is True else None

    @app.callback(
        Output("reconstruction-graph", "figure"),
        Input("viewer-mode", "value"), Input("sample-picker", "value"), Input("generation-step", "data"),
        Input("show-conditions", "value"), Input("show-helpers", "value"), Input("candidate-view", "data"),
        Input("selected-candidate", "data"), Input("show-current-highlight", "data"),
        Input("saved-camera", "data"),
    )
    def render_reconstruction(
        mode: str, sample: str, step: int, conditions: list[str], helpers: list[str],
        candidate_view: bool, selected_candidate: int | None, show_current_highlight: bool,
        saved_camera: dict[str, Any] | None,
    ):
        source = load_cache(sample)
        cache = source if mode == "ground_truth" else load_true_reconstruction(sample)
        step = max(0, min(int(step or 0), len(cache["events"]) - 1))
        show_conditions = "show" in (conditions or [])
        show_helpers = "show" in (helpers or [])
        if ctx.triggered_id in (None, "sample-picker", "viewer-mode"):
            if mode == "ground_truth":
                figure, _state = make_master_reconstruction_figure(source, step, show_conditions, show_helpers)
            else:
                figure, _state = make_master_true_reconstruction_figure(
                    source, cache, step, show_conditions, show_helpers, bool(candidate_view), selected_candidate,
                    bool(show_current_highlight),
                )
            if saved_camera:
                figure.update_layout(scene_camera=saved_camera)
            return figure
        if mode == "ground_truth":
            patch, _state = patch_reconstruction_figure(source, step, show_conditions, show_helpers)
        else:
            patch, _state = patch_true_reconstruction_figure(
                source, cache, step, show_conditions, show_helpers, bool(candidate_view), selected_candidate,
                bool(show_current_highlight),
            )
        if saved_camera:
            patch["layout"]["scene"]["camera"] = saved_camera
        return patch

    @app.callback(
        Output("event-title", "children"), Output("event-detail", "children"),
        Output("frame-counter", "children"), Output("step-slider", "max"),
        Output("right-panel-title", "children"), Output("right-panel-subtitle", "children"),
        Output("candidate-controls", "style"), Output("candidate-button", "children"), Output("candidate-hint", "children"),
        Output("highlight-button", "children"),
        Input("viewer-mode", "value"), Input("sample-picker", "value"), Input("generation-step", "data"),
        Input("candidate-view", "data"), Input("show-current-highlight", "data"),
    )
    def render_event(mode: str, sample: str, step: int, candidate_view: bool, show_current_highlight: bool):
        cache = load_cache(sample) if mode == "ground_truth" else load_true_reconstruction(sample)
        step = max(0, min(int(step or 0), len(cache["events"]) - 1))
        event = cache["events"][step]
        title = "Causal ground truth" if mode == "ground_truth" else "Topology-driven reconstruction"
        subtitle = (
            "Original STEP geometry, shown as the reference sequence"
            if mode == "ground_truth" else
            "Calculated only from vertices, supporting surfaces, and topology"
        )
        controls_style = {"display": "none"}
        button_label, hint = "Show candidates", ""
        if mode == "reconstruction" and event["kind"] == "ambiguity":
            loop = cache["loop_results"][event["loop_index"]]
            controls_style = {"display": "flex"}
            button_label = "Exit candidates and continue" if candidate_view else f"Show {len(loop['solutions'])} candidates"
            hint = "Playback is paused because topology and endpoints do not identify a unique curve segment."
        return (
            event["title"], event["detail"],
            f"Frame {step + 1} / {len(cache['events'])} · {event['phase'].upper()}", len(cache["events"]) - 1,
            title, subtitle, controls_style, button_label, hint,
            "Hide highlighted intersections" if show_current_highlight else "Show highlighted intersections",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    make_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
