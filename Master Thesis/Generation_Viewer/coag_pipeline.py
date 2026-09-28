"""Build a single-sample CoAG/CoSENT web cache from MFinstSeg data.

The implementation is intentionally self-contained inside Generation_Viewer.  It
uses OpenCascade for exact STEP topology, then stores only JSON-compatible meshes,
curves, graph attributes, subgraphs, and generation order for the Dash app.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
WEB_CACHE_DIR = ROOT_DIR / "web_cache"

STOCK_LABEL_ID = 24
IGNORED_FEATURE_LABEL_IDS = {0, 23}  # chamfer and round/fillet

FACE_CATEGORIES = [
    "chamfer", "through_hole", "triangular_passage", "rectangular_passage",
    "6sides_passage", "triangular_through_slot", "rectangular_through_slot",
    "circular_through_slot", "rectangular_through_step", "2sides_through_step",
    "slanted_through_step", "Oring", "blind_hole", "triangular_pocket",
    "rectangular_pocket", "6sides_pocket", "circular_end_pocket",
    "rectangular_blind_slot", "v_circular_end_blind_slot",
    "h_circular_end_blind_slot", "triangular_blind_step",
    "circular_blind_step", "rectangular_blind_step", "round", "stock",
]

# Thesis definition: relations are evaluated for an ordered coedge pair.
RELATION_ORDER = [
    "mate", "next", "previous", "cocurve", "cosurface", "coface",
    "share_start", "share_end",
]

# Same class palette used by CoAG_Viewer/app_coag.py.
FEATURE_COLORS = {
    0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3",
    4: "#ff7f00", 5: "#ffff33", 6: "#a65628", 7: "#f781bf",
    8: "#999999", 9: "#66c2a5", 10: "#fc8d62", 11: "#8da0cb",
    12: "#e78ac3", 13: "#a6d854", 14: "#ffd92f", 15: "#e5c494",
    16: "#b3b3b3", 17: "#1b9e77", 18: "#d95f02", 19: "#7570b3",
    20: "#e7298a", 21: "#66a61e", 22: "#e6ab02", 23: "#a6761d",
    24: "#c9c9c9",
}

PATH_COLORS = [
    "#e11d48", "#2563eb", "#16a34a", "#9333ea", "#ea580c",
    "#0891b2", "#be123c", "#4f46e5", "#65a30d", "#c026d3",
    "#0f766e", "#b45309",
]


def feature_name(label_id: int | None) -> str:
    if label_id is None:
        return "unknown"
    return FACE_CATEGORIES[label_id] if 0 <= label_id < len(FACE_CATEGORIES) else f"unknown_{label_id}"


def feature_color(label_id: int | None) -> str:
    return FEATURE_COLORS.get(int(label_id) if label_id is not None else -1, "#94a3b8")


def _round(value: float, digits: int = 7) -> float:
    return round(float(value), digits)


def _xyz(point: Any) -> list[float]:
    return [_round(point.X()), _round(point.Y()), _round(point.Z())]


def resolve_sample_paths(sample_name: str, data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Resolve the three one-to-one files belonging to a sample."""
    sample = Path(sample_name).stem
    # Path.stem only removes .step from x.step; also accept x.json or x.coag.json.
    if sample.endswith(".coag"):
        sample = sample[:-5]
    step_candidates = [data_dir / "steps" / f"{sample}{ext}" for ext in (".step", ".stp", ".STEP", ".STP")]
    step_path = next((p for p in step_candidates if p.exists()), None)
    paths = {
        "step": step_path,
        "label": data_dir / "labels" / f"{sample}.json",
        "graph": data_dir / "graphs" / f"{sample}.json",
    }
    missing = [f"{kind}: {path}" for kind, path in paths.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError("Sample is incomplete:\n" + "\n".join(missing))
    return {key: Path(value) for key, value in paths.items()}


def load_mfinstseg_labels(path: Path) -> dict[str, Any]:
    """Read MFinstSeg labels and normalize its NxN instance adjacency matrix.

    In this dataset ``inst[i][j] == 1`` means faces i and j belong to the same
    feature instance; the columns are not standalone instance IDs.  Components
    are assigned a stable ID equal to their smallest face ID.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        part_id, data = raw[0][0], raw[0][1]
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
        part_id, data = path.stem, raw[0]
    elif isinstance(raw, dict):
        part_id, data = raw.get("part_id", path.stem), raw
    else:
        raise ValueError(f"Unsupported label format: {path}")

    seg = {int(k): int(v) for k, v in data.get("seg", {}).items()}
    bottom = {int(k): int(v) for k, v in data.get("bottom", {}).items()}
    rows = data.get("inst", [])
    graph = nx.Graph()
    graph.add_nodes_from(range(max(len(rows), len(seg))))
    for i, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for j, value in enumerate(row):
            if int(value) == 1:
                graph.add_edge(i, j)

    face_instance_id: dict[int, int | None] = {face_id: None for face_id in seg}
    instance_faces: dict[int, list[int]] = {}
    for component in nx.connected_components(graph):
        members = sorted(i for i in component if i < len(rows) and any(int(v) for v in rows[i]))
        if not members:
            continue
        instance_id = min(members)
        instance_faces[instance_id] = members
        for face_id in members:
            face_instance_id[face_id] = instance_id

    return {
        "part_id": str(part_id), "seg": seg, "bottom": bottom,
        "face_instance_id": face_instance_id, "instance_faces": instance_faces,
    }


def _occ_imports() -> dict[str, Any]:
    """Import OCC lazily so JSON-only app startup has a clearer failure mode."""
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopAbs import (
            TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE,
            TopAbs_FORWARD, TopAbs_REVERSED,
        )
        from OCC.Core.TopExp import TopExp_Explorer, topexp
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
        from OCC.Core.BRepTools import BRepTools_WireExplorer, breptools
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.GeomAbs import (
            GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
            GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
            GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion,
            GeomAbs_OffsetSurface, GeomAbs_OtherSurface, GeomAbs_Line,
            GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
            GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve,
            GeomAbs_OffsetCurve, GeomAbs_OtherCurve,
        )
    except ImportError as exc:
        raise RuntimeError(
            "pythonocc-core is required. Run with the aagnet-mac environment: "
            "/opt/anaconda3/envs/aagnet-mac/bin/python"
        ) from exc
    return locals()


def _map_shapes(shape: Any, kind: Any, occ: dict[str, Any]) -> tuple[list[Any], Any]:
    indexed = occ["TopTools_IndexedMapOfShape"]()
    occ["topexp"].MapShapes(shape, kind, indexed)
    size = int(indexed.Size() if hasattr(indexed, "Size") else indexed.Extent())
    return [indexed.FindKey(i) for i in range(1, size + 1)], indexed


def _read_step(path: Path, occ: dict[str, Any]) -> Any:
    reader = occ["STEPControl_Reader"]()
    if reader.ReadFile(str(path)) != occ["IFSelect_RetDone"]:
        raise RuntimeError(f"Cannot read STEP: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP contains no transferable shape: {path}")
    return shape


def _face_type(face: Any, occ: dict[str, Any]) -> str:
    names = {
        occ["GeomAbs_Plane"]: "plane", occ["GeomAbs_Cylinder"]: "cylinder",
        occ["GeomAbs_Cone"]: "cone", occ["GeomAbs_Sphere"]: "sphere",
        occ["GeomAbs_Torus"]: "torus", occ["GeomAbs_BezierSurface"]: "bezier_surface",
        occ["GeomAbs_BSplineSurface"]: "bspline_surface",
        occ["GeomAbs_SurfaceOfRevolution"]: "surface_of_revolution",
        occ["GeomAbs_SurfaceOfExtrusion"]: "surface_of_extrusion",
        occ["GeomAbs_OffsetSurface"]: "offset_surface",
        occ["GeomAbs_OtherSurface"]: "other_surface",
    }
    try:
        return names.get(occ["BRepAdaptor_Surface"](face, True).GetType(), "unknown")
    except Exception:
        return "unknown"


def _edge_type(edge: Any, occ: dict[str, Any]) -> str:
    names = {
        occ["GeomAbs_Line"]: "line", occ["GeomAbs_Circle"]: "circle",
        occ["GeomAbs_Ellipse"]: "ellipse", occ["GeomAbs_Hyperbola"]: "hyperbola",
        occ["GeomAbs_Parabola"]: "parabola", occ["GeomAbs_BezierCurve"]: "bezier_curve",
        occ["GeomAbs_BSplineCurve"]: "bspline_curve",
        occ["GeomAbs_OffsetCurve"]: "offset_curve",
        occ["GeomAbs_OtherCurve"]: "other_curve",
    }
    try:
        return names.get(occ["BRepAdaptor_Curve"](edge).GetType(), "unknown")
    except Exception:
        return "unknown"


def _direction(direction: Any) -> tuple[float, float, float]:
    return (_round(direction.X(), 6), _round(direction.Y(), 6), _round(direction.Z(), 6))


def _point_tuple(point: Any) -> tuple[float, float, float]:
    return (_round(point.X(), 6), _round(point.Y(), 6), _round(point.Z(), 6))


def _surface_signature(face: Any, occ: dict[str, Any]) -> tuple[Any, ...]:
    try:
        adaptor = occ["BRepAdaptor_Surface"](face, True)
        kind = adaptor.GetType()
        name = _face_type(face, occ)
        if kind == occ["GeomAbs_Plane"]:
            item = adaptor.Plane()
            return (name, _point_tuple(item.Location()), _direction(item.Axis().Direction()))
        if kind == occ["GeomAbs_Cylinder"]:
            item = adaptor.Cylinder()
            return (name, _round(item.Radius(), 6), _point_tuple(item.Location()), _direction(item.Axis().Direction()))
        if kind == occ["GeomAbs_Cone"]:
            item = adaptor.Cone()
            return (name, _round(item.RefRadius(), 6), _round(item.SemiAngle(), 6), _point_tuple(item.Location()), _direction(item.Axis().Direction()))
        if kind == occ["GeomAbs_Sphere"]:
            item = adaptor.Sphere()
            return (name, _round(item.Radius(), 6), _point_tuple(item.Location()))
        if kind == occ["GeomAbs_Torus"]:
            item = adaptor.Torus()
            return (name, _round(item.MajorRadius(), 6), _round(item.MinorRadius(), 6), _point_tuple(item.Location()), _direction(item.Axis().Direction()))
    except Exception:
        pass
    return (_face_type(face, occ), id(face))


def _curve_signature(edge: Any, edge_id: int, occ: dict[str, Any]) -> tuple[Any, ...]:
    try:
        adaptor = occ["BRepAdaptor_Curve"](edge)
        kind = adaptor.GetType()
        name = _edge_type(edge, occ)
        if kind == occ["GeomAbs_Line"]:
            item = adaptor.Line()
            return (name, _point_tuple(item.Location()), _direction(item.Direction()))
        if kind == occ["GeomAbs_Circle"]:
            item = adaptor.Circle()
            return (name, _round(item.Radius(), 6), _point_tuple(item.Location()), _direction(item.Axis().Direction()))
        if kind == occ["GeomAbs_Ellipse"]:
            item = adaptor.Ellipse()
            return (name, _round(item.MajorRadius(), 6), _round(item.MinorRadius(), 6), _point_tuple(item.Location()), _direction(item.Axis().Direction()))
    except Exception:
        pass
    return (_edge_type(edge, occ), edge_id)


def _triangulate_face(face: Any, occ: dict[str, Any]) -> dict[str, list]:
    location = occ["TopLoc_Location"]()
    triangulation = occ["BRep_Tool"].Triangulation(face, location)
    if triangulation is None:
        return {"vertices": [], "triangles": []}
    transform = location.Transformation()
    vertices = []
    for index in range(1, triangulation.NbNodes() + 1):
        point = triangulation.Node(index).Transformed(transform)
        vertices.append(_xyz(point))
    triangles = []
    for index in range(1, triangulation.NbTriangles() + 1):
        a, b, c = triangulation.Triangle(index).Get()
        # Reverse winding for reversed faces so lighting remains visually coherent.
        if face.Orientation() == occ["TopAbs_REVERSED"]:
            b, c = c, b
        triangles.append([int(a) - 1, int(b) - 1, int(c) - 1])
    return {"vertices": vertices, "triangles": triangles}


def _sample_edge(edge: Any, occ: dict[str, Any], samples: int = 48) -> list[list[float]]:
    adaptor = occ["BRepAdaptor_Curve"](edge)
    first, last = float(adaptor.FirstParameter()), float(adaptor.LastParameter())
    if not math.isfinite(first) or not math.isfinite(last) or abs(last - first) > 1e9:
        first, last = 0.0, 1.0
    if abs(last - first) < 1e-12:
        return [_xyz(adaptor.Value(first))]
    count = max(2, samples)
    return [_xyz(adaptor.Value(first + (last - first) * i / (count - 1))) for i in range(count)]


def _bbox_from_points(point_groups: Iterable[Iterable[Iterable[float]]]) -> list[float]:
    points = [p for group in point_groups for p in group]
    if not points:
        return [0, 0, 0, 1, 1, 1]
    return [min(p[a] for p in points) for a in range(3)] + [max(p[a] for p in points) for a in range(3)]


def _finite_mid(a: float, b: float) -> float:
    return (a + b) / 2 if math.isfinite(a) and math.isfinite(b) and abs(a) < 1e9 and abs(b) < 1e9 else 0.0


def _support_surface_mesh(face: Any, bbox: list[float], occ: dict[str, Any], resolution: int = 24) -> dict[str, Any]:
    """Sample a small local patch of the face's underlying analytic surface.

    ``bbox`` is the bounding box of one machining-feature instance, not the
    complete part.  The native UV bounds of the source face are expanded only
    by a modest local margin.  This keeps prerequisite surfaces close to the
    feature while still making their underlying support geometry visible.
    """
    adaptor = occ["BRepAdaptor_Surface"](face, True)
    u1, u2, v1, v2 = [float(x) for x in occ["breptools"].UVBounds(face)]
    local_diag = math.sqrt(sum((bbox[i + 3] - bbox[i]) ** 2 for i in range(3))) or 1.0
    u_mid, v_mid = _finite_mid(u1, u2), _finite_mid(v1, v2)
    kind = adaptor.GetType()

    finite_uv = all(math.isfinite(x) and abs(x) < 1e9 for x in (u1, u2, v1, v2))
    if not finite_uv:
        # Analytic plane/cylinder parameters are metric in their unbounded
        # directions, so a feature-local fallback remains meaningful.
        u1, u2 = u_mid - local_diag / 2, u_mid + local_diag / 2
        v1, v2 = v_mid - local_diag / 2, v_mid + local_diag / 2

    u_span = max(abs(u2 - u1), 1e-6)
    v_span = max(abs(v2 - v1), 1e-6)
    metric_pad = max(local_diag * 0.10, 1e-4)
    if kind == occ["GeomAbs_Plane"]:
        # Plane UV coordinates are metric. Extend the trimmed feature face by
        # roughly 18%, with a minimum margin based on the feature instance size.
        u_pad = max(u_span * 0.18, metric_pad)
        v_pad = max(v_span * 0.18, metric_pad)
        u1, u2, v1, v2 = u1 - u_pad, u2 + u_pad, v1 - v_pad, v2 + v_pad
    elif kind in (occ["GeomAbs_Cylinder"], occ["GeomAbs_Cone"]):
        # Preserve a full revolution when the feature face already wraps the
        # axis; otherwise show only a slightly enlarged angular patch.
        if u_span >= 1.8 * math.pi:
            u1, u2 = 0.0, 2 * math.pi
        else:
            u_pad = min(max(u_span * 0.12, 0.08), 0.35)
            u1, u2 = u1 - u_pad, u2 + u_pad
        v_pad = max(v_span * 0.18, metric_pad)
        v1, v2 = v1 - v_pad, v2 + v_pad
    elif kind == occ["GeomAbs_Sphere"]:
        u_pad = min(max(u_span * 0.12, 0.08), 0.35)
        v_pad = min(max(v_span * 0.12, 0.08), 0.25)
        u1, u2 = u1 - u_pad, u2 + u_pad
        v1, v2 = max(-math.pi / 2, v1 - v_pad), min(math.pi / 2, v2 + v_pad)
    elif kind == occ["GeomAbs_Torus"]:
        u_pad = min(max(u_span * 0.10, 0.06), 0.30)
        v_pad = min(max(v_span * 0.10, 0.06), 0.30)
        u1, u2, v1, v2 = u1 - u_pad, u2 + u_pad, v1 - v_pad, v2 + v_pad
    else:
        # Do not extrapolate spline/other bounded surfaces beyond their domain.
        pass

    nu = max(8, resolution)
    nv = max(8, resolution)
    vertices = []
    for iv in range(nv):
        v = v1 + (v2 - v1) * iv / (nv - 1)
        for iu in range(nu):
            u = u1 + (u2 - u1) * iu / (nu - 1)
            try:
                vertices.append(_xyz(adaptor.Value(u, v)))
            except Exception:
                vertices.append([0.0, 0.0, 0.0])
    triangles = []
    for iv in range(nv - 1):
        for iu in range(nu - 1):
            a = iv * nu + iu
            triangles.extend(([a, a + 1, a + nu + 1], [a, a + nu + 1, a + nu]))
    return {"vertices": vertices, "triangles": triangles, "surface_type": _face_type(face, occ)}


def _extract_geometry_and_coag(step_path: Path, labels: dict[str, Any]) -> dict[str, Any]:
    occ = _occ_imports()
    shape = _read_step(step_path, occ)
    faces, face_map = _map_shapes(shape, occ["TopAbs_FACE"], occ)
    brep_edges, edge_map = _map_shapes(shape, occ["TopAbs_EDGE"], occ)
    _, vertex_map = _map_shapes(shape, occ["TopAbs_VERTEX"], occ)
    occ["BRepMesh_IncrementalMesh"](shape, 0.08, False, 0.45, True)

    face_meshes = []
    all_points: list[list[list[float]]] = []
    for face_id, face in enumerate(faces):
        mesh = _triangulate_face(face, occ)
        all_points.append(mesh["vertices"])
        label_id = labels["seg"].get(face_id)
        instance_id = labels["face_instance_id"].get(face_id)
        face_meshes.append({
            "face_id": face_id, "face_label": label_id,
            "face_label_name": feature_name(label_id), "feature_instance_id": instance_id,
            "bottom": labels["bottom"].get(face_id, 0),
            "face_type": _face_type(face, occ), **mesh,
        })

    edge_geometry = []
    for edge_id, edge in enumerate(brep_edges):
        points = _sample_edge(edge, occ)
        all_points.append(points)
        edge_geometry.append({"edge_id": edge_id, "edge_type": _edge_type(edge, occ), "points": points})
    bbox = _bbox_from_points(all_points)
    margin = max(math.sqrt(sum((bbox[i + 3] - bbox[i]) ** 2 for i in range(3))) * 0.12, 1e-3)
    view_bbox = [bbox[i] - margin for i in range(3)] + [bbox[i + 3] + margin for i in range(3)]

    nodes: list[dict[str, Any]] = []
    runtime = {
        "next": {}, "previous": {}, "edge_to_nodes": defaultdict(list),
        "curve_sig": {}, "surface_sig": {},
    }
    node_id = 0
    for face_id, face in enumerate(faces):
        label_id = labels["seg"].get(face_id)
        instance_id = labels["face_instance_id"].get(face_id)
        face_type = _face_type(face, occ)
        outer_wire = occ["breptools"].OuterWire(face)
        wire_explorer = occ["TopExp_Explorer"](face, occ["TopAbs_WIRE"])
        loop_id = 0
        while wire_explorer.More():
            wire = occ["topods"].Wire(wire_explorer.Current())
            is_outer = not outer_wire.IsNull() and wire.IsSame(outer_wire)
            ordered_edges = []
            iterator = occ["BRepTools_WireExplorer"](wire, face)
            while iterator.More():
                ordered_edges.append(occ["topods"].Edge(iterator.Current()))
                iterator.Next()
            if not ordered_edges:  # defensive fallback for uncommon wire representations
                iterator2 = occ["TopExp_Explorer"](wire, occ["TopAbs_EDGE"])
                while iterator2.More():
                    ordered_edges.append(occ["topods"].Edge(iterator2.Current()))
                    iterator2.Next()

            loop_nodes = []
            for local_edge_id, edge_occ in enumerate(ordered_edges):
                mapped = int(edge_map.FindIndex(edge_occ))
                edge_id = mapped - 1 if mapped > 0 else None
                start_vertex = occ["topexp"].FirstVertex(edge_occ, True)
                end_vertex = occ["topexp"].LastVertex(edge_occ, True)
                start_id = int(vertex_map.FindIndex(start_vertex)) - 1 if not start_vertex.IsNull() else None
                end_id = int(vertex_map.FindIndex(end_vertex)) - 1 if not end_vertex.IsNull() else None
                node = {
                    "id": node_id, "face_id": face_id, "face_label": label_id,
                    "face_label_name": feature_name(label_id),
                    "feature_instance_id": instance_id,
                    "is_machining_feature": label_id not in (None, STOCK_LABEL_ID) and label_id not in IGNORED_FEATURE_LABEL_IDS,
                    "edge_id": edge_id, "edge_type": _edge_type(edge_occ, occ),
                    "face_type": face_type, "edge_convexity": "unknown",
                    "dihedral_type": "unknown", "orientation": edge_occ.Orientation() != occ["TopAbs_REVERSED"],
                    "loop_id": loop_id, "loop_type": "outer" if is_outer else "inner",
                    "is_outer_loop": is_outer, "local_edge_id_in_loop": local_edge_id,
                    "start_vertex_id": start_id, "end_vertex_id": end_id,
                    "bottom": labels["bottom"].get(face_id, 0),
                }
                nodes.append(node)
                runtime["edge_to_nodes"][edge_id].append(node_id)
                runtime["curve_sig"][node_id] = _curve_signature(edge_occ, edge_id if edge_id is not None else -1, occ)
                runtime["surface_sig"][node_id] = _surface_signature(face, occ)
                loop_nodes.append(node_id)
                node_id += 1
            if len(loop_nodes) > 1:
                for index, current in enumerate(loop_nodes):
                    runtime["next"][current] = loop_nodes[(index + 1) % len(loop_nodes)]
                    runtime["previous"][current] = loop_nodes[(index - 1) % len(loop_nodes)]
            loop_id += 1
            wire_explorer.Next()

    graph_edges = []
    for left in range(len(nodes)):
        a = nodes[left]
        for right in range(left + 1, len(nodes)):
            b = nodes[right]
            same_edge = a["edge_id"] is not None and a["edge_id"] == b["edge_id"]
            relations = {
                "mate": bool(same_edge),
                "next": runtime["next"].get(left) == right,
                "previous": runtime["previous"].get(left) == right,
                "cocurve": bool(same_edge or runtime["curve_sig"][left] == runtime["curve_sig"][right]),
                "cosurface": bool(a["face_id"] == b["face_id"] or runtime["surface_sig"][left] == runtime["surface_sig"][right]),
                "coface": a["face_id"] == b["face_id"],
                "share_start": a["start_vertex_id"] is not None and a["start_vertex_id"] == b["start_vertex_id"],
                "share_end": a["end_vertex_id"] is not None and a["end_vertex_id"] == b["end_vertex_id"],
            }
            vector = [int(relations[name]) for name in RELATION_ORDER]
            if any(vector):
                graph_edges.append({"u": left, "v": right, "relations": relations, "relation_vector": vector})

    # Build a local bounding box for every machining-feature instance. Support
    # surfaces use this local scale rather than the complete model extent.
    instance_points: dict[int, list[list[float]]] = defaultdict(list)
    for face_mesh in face_meshes:
        label_id = face_mesh.get("face_label")
        instance_id = face_mesh.get("feature_instance_id")
        if (
            instance_id is not None
            and label_id not in (None, STOCK_LABEL_ID)
            and label_id not in IGNORED_FEATURE_LABEL_IDS
        ):
            instance_points[int(instance_id)].extend(face_mesh.get("vertices") or [])
    instance_bboxes = {
        instance_id: _bbox_from_points([points])
        for instance_id, points in instance_points.items()
        if points
    }

    # Local analytic support surfaces for real machining-feature faces only.
    support_surfaces = []
    seen_supports: set[tuple[Any, ...]] = set()
    for face_id, face in enumerate(faces):
        label_id = labels["seg"].get(face_id)
        instance_id = labels["face_instance_id"].get(face_id)
        if label_id in (None, STOCK_LABEL_ID) or label_id in IGNORED_FEATURE_LABEL_IDS or instance_id is None:
            continue
        signature = _surface_signature(face, occ)
        key = (instance_id, signature)
        if key in seen_supports:
            continue
        seen_supports.add(key)
        local_bbox = instance_bboxes.get(int(instance_id), bbox)
        mesh = _support_surface_mesh(face, local_bbox, occ)
        support_surfaces.append({
            "source_face_id": face_id, "feature_instance_id": instance_id,
            "face_label": label_id, "face_label_name": feature_name(label_id), **mesh,
            "local_bbox": local_bbox,
        })

    occurrence_counts = Counter(n["edge_id"] for n in nodes if n["edge_id"] is not None)
    return {
        "n_faces": len(faces), "n_brep_edges": len(brep_edges),
        "n_coedges": len(nodes), "n_graph_edges": len(graph_edges),
        "bbox": bbox, "view_bbox": view_bbox, "face_meshes": face_meshes,
        "brep_edges": edge_geometry, "support_surfaces": support_surfaces,
        "nodes": nodes, "edges": graph_edges,
        "edge_occurrence_histogram": dict(Counter(occurrence_counts.values())),
    }


def _induced_edges(edge_list: list[dict[str, Any]], node_ids: set[int]) -> list[int]:
    return [i for i, edge in enumerate(edge_list) if edge["u"] in node_ids and edge["v"] in node_ids]


def split_coag(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split CoAG into feature-instance graphs plus one remainder graph.

    Every subgraph contains its core nodes and the opposite endpoint of every
    crossing edge.  Thus boundary nodes are intentionally repeated and no graph
    edge disappears merely because of the split.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        adjacency[edge["u"]].add(edge["v"])
        adjacency[edge["v"]].add(edge["u"])

    feature_groups: dict[int, set[int]] = defaultdict(set)
    for node in nodes:
        if node["is_machining_feature"] and node["feature_instance_id"] is not None:
            feature_groups[int(node["feature_instance_id"])].add(int(node["id"]))

    subgraphs = []
    feature_core_union = set().union(*feature_groups.values()) if feature_groups else set()
    for instance_id, core in sorted(feature_groups.items()):
        boundary = {neighbor for node_id in core for neighbor in adjacency[node_id] if neighbor not in core}
        all_nodes = core | boundary
        label_ids = sorted({int(nodes[n]["face_label"]) for n in core if nodes[n]["face_label"] is not None})
        label_id = label_ids[0] if label_ids else None
        subgraphs.append({
            "id": f"feature_{instance_id}", "kind": "feature",
            "title": f"Feature {instance_id}: {feature_name(label_id)}",
            "feature_instance_id": instance_id, "face_label": label_id,
            "core_nodes": sorted(core), "boundary_nodes": sorted(boundary),
            "node_ids": sorted(all_nodes), "edge_indices": _induced_edges(edges, all_nodes),
        })

    remainder_core = set(range(len(nodes))) - feature_core_union
    remainder_boundary = {neighbor for node_id in remainder_core for neighbor in adjacency[node_id] if neighbor not in remainder_core}
    remainder_nodes = remainder_core | remainder_boundary
    subgraphs.append({
        "id": "remainder", "kind": "remainder", "title": "Non-feature / completion graph",
        "feature_instance_id": None, "face_label": STOCK_LABEL_ID,
        "core_nodes": sorted(remainder_core), "boundary_nodes": sorted(remainder_boundary),
        "node_ids": sorted(remainder_nodes), "edge_indices": _induced_edges(edges, remainder_nodes),
    })
    return subgraphs


def _oriented_vector(edge: dict[str, Any], source: int, target: int) -> list[int]:
    relations = dict(edge["relations"])
    if source == edge["v"] and target == edge["u"]:
        relations["next"], relations["previous"] = relations["previous"], relations["next"]
    return [int(relations[name]) for name in RELATION_ORDER]


def build_cosent(
    subgraph: dict[str, Any], all_edges: list[dict[str, Any]],
    seed: int, preferred_start: int | None = None,
) -> dict[str, Any]:
    """FAG_SENT_Viewer-compatible causal Hamiltonian neighbor-trail sampler."""
    rng = random.Random(seed)
    graph = nx.Graph()
    graph.add_nodes_from(subgraph["node_ids"])
    for edge_index in subgraph["edge_indices"]:
        edge = all_edges[edge_index]
        graph.add_edge(edge["u"], edge["v"], edge_index=edge_index)
    unvisited = set(subgraph["node_ids"])
    if not unvisited:
        return {"steps": [], "trails": [], "node_order": []}

    if preferred_start in unvisited:
        current_node = int(preferred_start)
    else:
        candidates = sorted(set(subgraph["core_nodes"]) & unvisited) or sorted(unvisited)
        current_node = rng.choice(candidates)
    unvisited.remove(current_node)
    visited = {current_node}
    trails: list[list[int]] = [[current_node]]
    steps = []

    def make_step(node_id: int, previous: int | None, trail_id: int) -> dict[str, Any]:
        prior_neighbors = sorted(set(graph.neighbors(node_id)) & (visited - {node_id}))
        induced = []
        for neighbor in prior_neighbors:
            edge_index = int(graph.edges[node_id, neighbor]["edge_index"])
            induced.append({
                "neighbor": neighbor, "edge_index": edge_index,
                "relation_vector": _oriented_vector(all_edges[edge_index], node_id, neighbor),
            })
        return {
            "index": len(steps), "trail_id": trail_id, "node_id": node_id,
            "prev_node_id": previous, "prior_neighbors": induced,
            "is_boundary_context": node_id in set(subgraph["boundary_nodes"]),
        }

    steps.append(make_step(current_node, None, 0))
    while unvisited:
        candidates = sorted(set(graph.neighbors(current_node)) & unvisited)
        if candidates:
            next_node = rng.choice(candidates)
            previous = current_node
            current_node = next_node
            unvisited.remove(current_node)
            visited.add(current_node)
            trails[-1].append(current_node)
            steps.append(make_step(current_node, previous, len(trails) - 1))
            continue
        # Restart at an unvisited node. Prefer one touching the visited prefix.
        connected = sorted(n for n in unvisited if set(graph.neighbors(n)) & visited)
        current_node = rng.choice(connected or sorted(unvisited))
        unvisited.remove(current_node)
        visited.add(current_node)
        trails.append([current_node])
        steps.append(make_step(current_node, None, len(trails) - 1))
    return {"steps": steps, "trails": trails, "node_order": [step["node_id"] for step in steps]}


def _add_cosent_and_generation(subgraphs: list[dict[str, Any]], nodes: list[dict[str, Any]], edges: list[dict[str, Any]], seed: int) -> list[int]:
    feature_boundary_nodes = set()
    for index, subgraph in enumerate(subgraphs):
        if subgraph["kind"] == "feature":
            feature_boundary_nodes.update(subgraph["boundary_nodes"])
            preferred = min(subgraph["core_nodes"]) if subgraph["core_nodes"] else None
        else:
            # Requirement: the completion CoSENT begins at any feature/remainder boundary node.
            candidates = sorted(set(subgraph["boundary_nodes"]) & feature_boundary_nodes)
            preferred = candidates[0] if candidates else (subgraph["boundary_nodes"][0] if subgraph["boundary_nodes"] else None)
        subgraph["path_color"] = PATH_COLORS[index % len(PATH_COLORS)]
        subgraph["cosent"] = build_cosent(subgraph, edges, seed + index * 1009, preferred)

    # Boundary context nodes occur in more than one CoSENT, but a CoAG node is
    # generated only once. Feature sequences precede the completion sequence.
    order = []
    seen = set()
    for subgraph in subgraphs:
        for node_id in subgraph["cosent"]["node_order"]:
            if node_id not in seen:
                seen.add(node_id)
                order.append(node_id)
    # Defensive guarantee for isolated nodes.
    order.extend(node_id for node_id in range(len(nodes)) if node_id not in seen)
    return order


def _graph_layout(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], seed: int) -> dict[str, list[float]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from((edge["u"], edge["v"]) for edge in edges)
    positions = nx.spring_layout(graph, seed=seed, iterations=160, k=1.7 / math.sqrt(max(len(nodes), 1)))
    return {str(node_id): [_round(x, 8), _round(y, 8)] for node_id, (x, y) in positions.items()}


def build_sample_cache(sample_name: str, seed: int = 42, force: bool = False) -> Path:
    paths = resolve_sample_paths(sample_name)
    sample = paths["step"].stem
    output = WEB_CACHE_DIR / f"{sample}.webcache.json"
    if output.exists() and not force:
        return output

    labels = load_mfinstseg_labels(paths["label"])
    payload = _extract_geometry_and_coag(paths["step"], labels)
    subgraphs = split_coag(payload["nodes"], payload["edges"])
    generation_order = _add_cosent_and_generation(subgraphs, payload["nodes"], payload["edges"], seed)
    payload.update({
        "schema_version": 1, "sample_name": sample, "part_id": labels["part_id"],
        "source_files": {key: str(path.resolve()) for key, path in paths.items()},
        "face_categories": FACE_CATEGORIES, "feature_colors": {str(k): v for k, v in FEATURE_COLORS.items()},
        "ignored_feature_labels": sorted(IGNORED_FEATURE_LABEL_IDS),
        "relation_order": RELATION_ORDER, "subgraphs": subgraphs,
        "generation_order": generation_order,
        "graph_layout": _graph_layout(payload["nodes"], payload["edges"], seed),
    })
    WEB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return output


def load_cache(sample_name: str) -> dict[str, Any]:
    sample = Path(sample_name).stem
    path = WEB_CACHE_DIR / f"{sample}.webcache.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing web cache: {path}. Run prepare_sample.py first.")
    return json.loads(path.read_text(encoding="utf-8"))
