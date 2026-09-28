"""Extract a surface-first CoAG reconstruction sequence from MFInstSeg STEP files.

The sequence deliberately contains exact vertex coordinates and exact supporting
surface parameters, but no generated edge geometry.  Edge curves in the cache are
post-processing evidence: an untrimmed candidate is recovered after both incident
supporting surfaces are known, then STEP topology supplies the ground-truth branch,
end vertices, loop membership, and final trim used by this research viewer.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CACHE_DIR = APP_DIR / "web_cache"
GENERATION_PIPELINE = APP_DIR.parent / "Generation_Viewer" / "coag_pipeline.py"
FEATURE_SPACE_SEEDS = APP_DIR.parent / "Seed_Extractor" / "data" / "feature_space_seeds.json"


def _load_generation_pipeline():
    spec = importlib.util.spec_from_file_location("thesis_generation_pipeline", GENERATION_PIPELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared extraction code: {GENERATION_PIPELINE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gp = _load_generation_pipeline()


def _r(value: float, digits: int = 7) -> float:
    return round(float(value), digits)


def _point(value: Any) -> list[float]:
    return [_r(value.X()), _r(value.Y()), _r(value.Z())]


def _edge_bbox(edge: Any, model_bbox: list[float]) -> dict[str, Any]:
    """Return the exact axis-aligned edge box and its intrinsic display dimension."""
    box = Bnd_Box()
    box.SetGap(0.0)
    try:
        brepbndlib.AddOptimal(edge, box, True, False)
    except Exception:
        brepbndlib.Add(edge, box, True)
    xmin, ymin, zmin, xmax, ymax, zmax = [float(value) for value in box.Get()]
    minimum = [_r(xmin), _r(ymin), _r(zmin)]
    maximum = [_r(xmax), _r(ymax), _r(zmax)]
    model_diagonal = math.sqrt(sum((model_bbox[index + 3] - model_bbox[index]) ** 2 for index in range(3))) or 1.0
    flat_tolerance = max(model_diagonal * 1e-7, 1e-8)
    flat_axes = [index for index in range(3) if maximum[index] - minimum[index] <= flat_tolerance]
    return {
        "min_corner": minimum,
        "max_corner": maximum,
        "flat_axes": flat_axes,
        "dimension": 3 - len(flat_axes),
    }


def _direction(value: Any) -> list[float]:
    return [_r(value.X()), _r(value.Y()), _r(value.Z())]


def _frame(position: Any) -> dict[str, list[float]]:
    return {
        "origin": _point(position.Location()),
        "axis_x": _direction(position.XDirection()),
        "axis_y": _direction(position.YDirection()),
        "axis_z": _direction(position.Direction()),
    }


def _surface_parameters(face: Any, occ: dict[str, Any]) -> dict[str, Any]:
    """Return exact OpenCascade analytic parameters or explicit B-spline poles."""
    adaptor = occ["BRepAdaptor_Surface"](face, True)
    kind = adaptor.GetType()
    name = gp._face_type(face, occ)
    result: dict[str, Any] = {"type": name}
    try:
        if kind == occ["GeomAbs_Plane"]:
            result.update(_frame(adaptor.Plane().Position()))
        elif kind == occ["GeomAbs_Cylinder"]:
            item = adaptor.Cylinder()
            result.update(_frame(item.Position()))
            result["radius"] = _r(item.Radius())
        elif kind == occ["GeomAbs_Cone"]:
            item = adaptor.Cone()
            result.update(_frame(item.Position()))
            result["reference_radius"] = _r(item.RefRadius())
            result["semi_angle"] = _r(item.SemiAngle())
        elif kind == occ["GeomAbs_Sphere"]:
            item = adaptor.Sphere()
            result.update(_frame(item.Position()))
            result["radius"] = _r(item.Radius())
        elif kind == occ["GeomAbs_Torus"]:
            item = adaptor.Torus()
            result.update(_frame(item.Position()))
            result["major_radius"] = _r(item.MajorRadius())
            result["minor_radius"] = _r(item.MinorRadius())
        elif name in {"bspline_surface", "bezier_surface"}:
            spline = adaptor.BSpline()
            result.update({
                "u_degree": int(spline.UDegree()),
                "v_degree": int(spline.VDegree()),
                "n_u_poles": int(spline.NbUPoles()),
                "n_v_poles": int(spline.NbVPoles()),
                "poles": [
                    _point(spline.Pole(i, j))
                    for i in range(1, spline.NbUPoles() + 1)
                    for j in range(1, spline.NbVPoles() + 1)
                ],
            })
        else:
            u1, u2, v1, v2 = [float(x) for x in occ["breptools"].UVBounds(face)]
            u = gp._finite_mid(u1, u2)
            v = gp._finite_mid(v1, v2)
            result["origin"] = _point(adaptor.Value(u, v))
    except Exception as exc:
        result["extraction_note"] = f"partial parameter record: {type(exc).__name__}"
    return result


def _canonical_direction(values: list[float]) -> tuple[list[float], int]:
    """Choose one sign for an unoriented axis so equal supports share a key."""
    sign = 1
    for value in values:
        if abs(value) > 1e-8:
            sign = 1 if value > 0 else -1
            break
    return ([_r(sign * value, 6) for value in values], sign)


def _surface_key(face: Any, face_id: int, occ: dict[str, Any]) -> tuple[Any, ...]:
    """Canonicalize analytic supports independently of a face's chosen origin."""
    adaptor = occ["BRepAdaptor_Surface"](face, True)
    kind = adaptor.GetType()
    name = gp._face_type(face, occ)
    try:
        if kind == occ["GeomAbs_Plane"]:
            plane = adaptor.Plane()
            normal, sign = _canonical_direction(_direction(plane.Axis().Direction()))
            location = _point(plane.Location())
            offset = sign * sum(location[i] * _direction(plane.Axis().Direction())[i] for i in range(3))
            return (name, tuple(normal), _r(offset, 6))
        if kind == occ["GeomAbs_Cylinder"]:
            cylinder = adaptor.Cylinder()
            axis, _ = _canonical_direction(_direction(cylinder.Axis().Direction()))
            location = _point(cylinder.Location())
            along = sum(location[i] * axis[i] for i in range(3))
            closest = [_r(location[i] - along * axis[i], 6) for i in range(3)]
            return (name, tuple(axis), tuple(closest), _r(cylinder.Radius(), 6))
        if kind == occ["GeomAbs_Cone"]:
            cone = adaptor.Cone()
            raw_axis = _direction(cone.Axis().Direction())
            axis, sign = _canonical_direction(raw_axis)
            location = _point(cone.Location())
            angle = float(cone.SemiAngle())
            apex_shift = float(cone.RefRadius()) / math.tan(angle) if abs(math.tan(angle)) > 1e-9 else 0.0
            apex = [_r(location[i] - apex_shift * raw_axis[i], 6) for i in range(3)]
            return (name, tuple(axis), tuple(apex), _r(sign * angle, 6))
        if kind == occ["GeomAbs_Sphere"]:
            sphere = adaptor.Sphere()
            return (name, tuple(_point(sphere.Location())), _r(sphere.Radius(), 6))
        if kind == occ["GeomAbs_Torus"]:
            torus = adaptor.Torus()
            axis, _ = _canonical_direction(_direction(torus.Axis().Direction()))
            return (name, tuple(_point(torus.Location())), tuple(axis), _r(torus.MajorRadius(), 6), _r(torus.MinorRadius(), 6))
    except Exception:
        pass
    return (name, face_id)


def _direct_quantize(parameters: dict[str, Any], bbox: list[float]) -> dict[str, Any]:
    """A transparent first baseline: fixed scalar bins, without a learned VAE."""
    diagonal = math.sqrt(sum((bbox[i + 3] - bbox[i]) ** 2 for i in range(3))) or 1.0

    def q_coord(vector: list[float]) -> list[int]:
        out = []
        for axis, value in enumerate(vector):
            lo, hi = bbox[axis], bbox[axis + 3]
            span = hi - lo or 1.0
            out.append(max(0, min(1023, round((value - lo) / span * 1023))))
        return out

    def q_direction(vector: list[float]) -> list[int]:
        return [max(0, min(1023, round((value + 1.0) * 511.5))) for value in vector]

    quantized: dict[str, Any] = {"scheme": "fixed_scalar_v1"}
    for key in ("origin",):
        if key in parameters:
            quantized[key] = q_coord(parameters[key])
    for key in ("axis_x", "axis_y", "axis_z"):
        if key in parameters:
            quantized[key] = q_direction(parameters[key])
    for key in ("radius", "reference_radius", "major_radius", "minor_radius"):
        if key in parameters:
            quantized[key] = max(0, min(4095, round(parameters[key] / diagonal * 4095)))
    if "semi_angle" in parameters:
        quantized["semi_angle"] = max(0, min(4095, round((parameters["semi_angle"] + math.pi) / (2 * math.pi) * 4095)))
    if "poles" in parameters:
        quantized["poles"] = [q_coord(pole) for pole in parameters["poles"]]
    return quantized


def _extended_curve(edge: Any, bbox: list[float], occ: dict[str, Any]) -> tuple[list[list[float]], str]:
    """Visualize the selected untrimmed intersection branch inside the model box.

    Analytic lines and conics are expanded to their underlying curve.  General
    curves retain the STEP-selected branch and are labelled as a fallback.  They
    are never treated as generated tokens.
    """
    adaptor = occ["BRepAdaptor_Curve"](edge)
    kind = adaptor.GetType()
    center = [(bbox[i] + bbox[i + 3]) / 2 for i in range(3)]
    diagonal = math.sqrt(sum((bbox[i + 3] - bbox[i]) ** 2 for i in range(3))) or 1.0
    if kind == occ["GeomAbs_Line"]:
        line = adaptor.Line()
        origin = _point(line.Location())
        direction = _direction(line.Direction())
        projection = sum((center[i] - origin[i]) * direction[i] for i in range(3))
        extent = diagonal * 0.8
        points = [
            [origin[a] + (projection + sign * extent) * direction[a] for a in range(3)]
            for sign in (-1, 1)
        ]
        return [[_r(x) for x in point] for point in points], "analytic line from surface pair"
    if kind in {occ["GeomAbs_Circle"], occ["GeomAbs_Ellipse"]}:
        return [
            _point(adaptor.Value(2 * math.pi * i / 127))
            for i in range(128)
        ], "complete analytic conic from surface pair"
    return gp._sample_edge(edge, occ, samples=96), "STEP-selected intersection branch fallback"


def _surface_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    feature_ids = record["feature_instance_ids"]
    return (0 if feature_ids else 1, min(feature_ids) if feature_ids else 10**9, record["source_face_ids"][0])


def _topological_plane(face: Any, occ: dict[str, Any]) -> dict[str, Any] | None:
    adaptor = occ["BRepAdaptor_Surface"](face, True)
    if adaptor.GetType() != occ["GeomAbs_Plane"]:
        return None
    plane = adaptor.Plane()
    normal = np.asarray(_direction(plane.Axis().Direction()), dtype=float)
    if face.Orientation() == occ["TopAbs_REVERSED"]:
        normal *= -1.0
    normal /= np.linalg.norm(normal)
    location = np.asarray(_point(plane.Location()), dtype=float)
    return {"normal": normal, "offset": float(normal @ location)}


def _polygon_from_oriented_planes(
    planes: list[dict[str, Any]], origin: np.ndarray, scale: float,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    normals = np.stack([plane["normal"] for plane in planes])
    _, singular_values, vh = np.linalg.svd(normals, full_matrices=True)
    if len(singular_values) < 2 or singular_values[1] < 1e-7:
        return None
    axis = vh[-1]
    axis /= np.linalg.norm(axis)
    helper = np.asarray([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.asarray([0.0, 1.0, 0.0])
    basis_u = np.cross(axis, helper)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    constraints = []
    for plane in planes:
        normal, offset = plane["normal"], plane["offset"]
        constraints.append((float(normal @ basis_u), float(normal @ basis_v), float(offset - normal @ origin)))
    tolerance = max(scale * 1e-7, 1e-7)
    vertices_2d: list[np.ndarray] = []
    for left, right in itertools.combinations(range(len(constraints)), 2):
        a1, b1, c1 = constraints[left]
        a2, b2, c2 = constraints[right]
        determinant = a1 * b2 - a2 * b1
        if abs(determinant) < 1e-10:
            continue
        point = np.asarray([(c1 * b2 - c2 * b1) / determinant, (a1 * c2 - a2 * c1) / determinant])
        if all(a * point[0] + b * point[1] >= c - tolerance for a, b, c in constraints):
            if not any(np.linalg.norm(point - existing) <= tolerance * 10 for existing in vertices_2d):
                vertices_2d.append(point)
    if len(vertices_2d) != len(planes):
        return None
    center = sum(vertices_2d) / len(vertices_2d)
    vertices_2d.sort(key=lambda point: math.atan2(point[1] - center[1], point[0] - center[0]))
    vertices_3d = [origin + point[0] * basis_u + point[1] * basis_v for point in vertices_2d]
    return vertices_3d, axis


def _prismatic_feature_space(
    instance_id: int, category_id: int, required_sides: int, face_ids: list[int],
    faces: list[Any], face_meshes: list[dict[str, Any]], model_bbox: list[float], occ: dict[str, Any],
) -> dict[str, Any] | None:
    model_diagonal = math.sqrt(sum((model_bbox[i + 3] - model_bbox[i]) ** 2 for i in range(3))) or 1.0
    feature_points = [
        np.asarray(point, dtype=float)
        for face_id in face_ids for point in (face_meshes[face_id].get("vertices") or [])
    ]
    if not feature_points:
        return None
    origin = sum(feature_points) / len(feature_points)
    unique_planes: list[dict[str, Any]] = []
    plane_keys: set[tuple[Any, ...]] = set()
    for face_id in face_ids:
        plane = _topological_plane(faces[face_id], occ)
        if plane is None:
            continue
        plane["face_id"] = face_id
        canonical_normal, sign = _canonical_direction(plane["normal"].tolist())
        plane_key = (tuple(canonical_normal), _r(sign * plane["offset"], 5))
        if plane_key in plane_keys:
            continue
        plane_keys.add(plane_key)
        unique_planes.append(plane)
    candidates = []
    for selection in itertools.combinations(unique_planes, required_sides):
        result = _polygon_from_oriented_planes(list(selection), origin, model_diagonal)
        if result is None:
            continue
        polygon, axis = result
        area_vector = np.zeros(3)
        for index, point in enumerate(polygon):
            area_vector += np.cross(point - origin, polygon[(index + 1) % len(polygon)] - origin)
        area = abs(float(area_vector @ axis)) * 0.5
        if area > 1e-10:
            candidates.append((area, list(selection), polygon, axis))
    if not candidates:
        return None
    _, selected_planes, polygon, axis = min(candidates, key=lambda item: item[0])
    projections = [float((point - origin) @ axis) for point in feature_points]
    t_min, t_max = min(projections), max(projections)
    span = max(t_max - t_min, model_diagonal * 0.04)
    extension = max(span * 0.14, model_diagonal * 0.025)
    t_min -= extension
    t_max += extension

    # Pockets have one bottom plane parallel to the extrusion axis.  Keep the
    # exact bottom boundary and extend only toward the open side.
    bottom_plane = None
    selected_face_ids = {int(plane["face_id"]) for plane in selected_planes}
    for plane in unique_planes:
        if int(plane["face_id"]) in selected_face_ids:
            continue
        alignment = float(plane["normal"] @ axis)
        if abs(alignment) > 0.98:
            bottom_plane = plane
            bottom_t = float((plane["offset"] - plane["normal"] @ origin) / alignment)
            if alignment > 0:
                t_min = max(t_min, bottom_t)
            else:
                t_max = min(t_max, bottom_t)
            break

    lower = [point + t_min * axis for point in polygon]
    upper = [point + t_max * axis for point in polygon]
    vertices = [[_r(value) for value in point] for point in lower + upper]
    n = len(polygon)
    triangles = []
    for index in range(n):
        nxt = (index + 1) % n
        triangles.extend(([index, nxt, n + nxt], [index, n + nxt, n + index]))
    if bottom_plane is not None:
        cap_start = 0 if abs(t_min - bottom_t) < abs(t_max - bottom_t) else n
        for index in range(1, n - 1):
            triangles.append([cap_start, cap_start + index, cap_start + index + 1])
    return {
        "id": f"feature-{instance_id}", "feature_instance_id": instance_id,
        "category_id": category_id, "category_name": gp.feature_name(category_id),
        "geometry_kind": "closed_oriented_prismatic_section",
        "source_face_ids": sorted(selected_face_ids | ({int(bottom_plane['face_id'])} if bottom_plane else set())),
        "axis": [_r(value) for value in axis], "axial_range": [_r(t_min), _r(t_max)],
        "cross_section_vertices": [[_r(value) for value in point] for point in polygon],
        "mesh": {"vertices": vertices, "triangles": triangles},
    }


def _general_planar_feature_space(
    instance_id: int, category_id: int, face_ids: list[int], faces: list[Any],
    face_meshes: list[dict[str, Any]], model_bbox: list[float], occ: dict[str, Any],
) -> dict[str, Any] | None:
    """Clip an oriented planar feature cell by a finite visualization envelope.

    Unlike the previous independent support patches, every green polygon here is
    clipped simultaneously by all other active feature halfspaces.  The envelope
    only caps genuinely unbounded recession directions for display.
    """
    feature_points = [
        np.asarray(point, dtype=float)
        for face_id in face_ids for point in (face_meshes[face_id].get("vertices") or [])
    ]
    if not feature_points:
        return None
    plane_keys: set[tuple[Any, ...]] = set()
    planes: list[dict[str, Any]] = []
    for face_id in face_ids:
        plane = _topological_plane(faces[face_id], occ)
        if plane is None:
            return None
        canonical_normal, sign = _canonical_direction(plane["normal"].tolist())
        key = (tuple(canonical_normal), _r(sign * plane["offset"], 5))
        if key in plane_keys:
            continue
        plane_keys.add(key)
        plane["face_id"] = face_id
        planes.append(plane)
    if not planes:
        return None
    points_array = np.stack(feature_points)
    local_min, local_max = points_array.min(axis=0), points_array.max(axis=0)
    model_diagonal = math.sqrt(sum((model_bbox[i + 3] - model_bbox[i]) ** 2 for i in range(3))) or 1.0
    local_span = np.linalg.norm(local_max - local_min)
    padding = max(local_span * 0.18, model_diagonal * 0.035)
    local_min -= padding
    local_max += padding
    constraints = [(plane["normal"], float(plane["offset"]), True) for plane in planes]
    for axis_index in range(3):
        positive = np.zeros(3); positive[axis_index] = 1.0
        negative = -positive
        constraints.append((positive, float(local_min[axis_index]), False))
        constraints.append((negative, float(-local_max[axis_index]), False))
    tolerance = max(model_diagonal * 1e-7, 1e-7)
    vertices: list[np.ndarray] = []
    for selection in itertools.combinations(range(len(constraints)), 3):
        matrix = np.stack([constraints[index][0] for index in selection])
        if abs(float(np.linalg.det(matrix))) < 1e-10:
            continue
        rhs = np.asarray([constraints[index][1] for index in selection])
        point = np.linalg.solve(matrix, rhs)
        if all(float(normal @ point) >= offset - tolerance for normal, offset, _active in constraints):
            if not any(np.linalg.norm(point - existing) <= tolerance * 10 for existing in vertices):
                vertices.append(point)
    if len(vertices) < 4:
        return None
    mesh_vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    active_face_ids = []
    for plane in planes:
        polygon = [point for point in vertices if abs(float(plane["normal"] @ point) - plane["offset"]) <= tolerance * 20]
        if len(polygon) < 3:
            continue
        center = sum(polygon) / len(polygon)
        helper = np.asarray([1.0, 0.0, 0.0]) if abs(plane["normal"][0]) < 0.8 else np.asarray([0.0, 1.0, 0.0])
        basis_u = np.cross(plane["normal"], helper); basis_u /= np.linalg.norm(basis_u)
        basis_v = np.cross(plane["normal"], basis_u)
        polygon.sort(key=lambda point: math.atan2(float((point - center) @ basis_v), float((point - center) @ basis_u)))
        start = len(mesh_vertices)
        mesh_vertices.extend([[_r(value) for value in point] for point in polygon])
        for index in range(1, len(polygon) - 1):
            triangles.append([start, start + index, start + index + 1])
        active_face_ids.append(int(plane["face_id"]))
    if not triangles:
        return None
    return {
        "id": f"feature-{instance_id}", "feature_instance_id": instance_id,
        "category_id": category_id, "category_name": gp.feature_name(category_id),
        "geometry_kind": "oriented_halfspace_cell_clipped",
        "source_face_ids": sorted(active_face_ids),
        "visualization_envelope": [*[_r(value) for value in local_min], *[_r(value) for value in local_max]],
        "mesh": {"vertices": mesh_vertices, "triangles": triangles},
    }


def _build_feature_spaces(
    labels: dict[str, Any], faces: list[Any], face_meshes: list[dict[str, Any]],
    surfaces: list[dict[str, Any]], bbox: list[float], occ: dict[str, Any],
) -> list[dict[str, Any]]:
    seed_data = json.loads(FEATURE_SPACE_SEEDS.read_text(encoding="utf-8"))
    specs = {int(feature["category_id"]): feature for feature in seed_data["features"]}
    result = []
    for instance_id, face_ids in sorted(labels["instance_faces"].items()):
        category_ids = [labels["seg"].get(face_id) for face_id in face_ids]
        category_ids = [int(value) for value in category_ids if value is not None and value not in gp.IGNORED_FEATURE_LABEL_IDS and value != gp.STOCK_LABEL_ID]
        if not category_ids:
            continue
        category_id = Counter(category_ids).most_common(1)[0][0]
        spec = specs.get(category_id, {})
        prismatic = next((constraint for constraint in spec.get("group_constraints", []) if constraint.get("type") == "closed_oriented_prismatic_section"), None)
        if prismatic:
            feature_space = _prismatic_feature_space(
                int(instance_id), category_id, len(prismatic["faces"]), list(face_ids),
                faces, face_meshes, bbox, occ,
            )
            if feature_space is not None:
                result.append(feature_space)
                continue
        if all(gp._face_type(faces[face_id], occ) == "plane" for face_id in face_ids):
            feature_space = _general_planar_feature_space(
                int(instance_id), category_id, list(face_ids), faces, face_meshes, bbox, occ,
            )
            if feature_space is not None:
                result.append(feature_space)
                continue
        # Trimmed-cylinder and mixed analytic cells preserve the source patches'
        # angular/topological domains, as required by the seed policy.  Merge the
        # mutually trimmed feature boundary instead of drawing independent local
        # support patches that extend through one another.
        merged_vertices: list[list[float]] = []
        merged_triangles: list[list[int]] = []
        for face_id in face_ids:
            mesh = face_meshes[face_id]
            vertices = mesh.get("vertices") or []
            triangles = mesh.get("triangles") or []
            start = len(merged_vertices)
            merged_vertices.extend(vertices)
            merged_triangles.extend([[start + int(a), start + int(b), start + int(c)] for a, b, c in triangles])
        if merged_triangles:
            result.append({
                "id": f"feature-{instance_id}", "feature_instance_id": int(instance_id),
                "category_id": category_id, "category_name": gp.feature_name(category_id),
                "geometry_kind": spec.get("space_model", "trimmed_piecewise_boundary_cell"),
                "source_face_ids": sorted(face_ids),
                "mesh": {"vertices": merged_vertices, "triangles": merged_triangles},
            })
    return result


def _build_events(
    surfaces: list[dict[str, Any]], nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
    vertices: list[dict[str, Any]], loops: list[dict[str, Any]], surface_order: list[int],
) -> tuple[list[dict[str, Any]], list[int]]:
    events: list[dict[str, Any]] = [{
        "phase": "condition", "kind": "condition_surfaces",
        "title": "Feature-space conditions",
        "detail": "Supporting surfaces are visible in light green; they are conditions, not generated topology.",
    }]
    face_to_surface = {
        face_id: surface["id"] for surface in surfaces for face_id in surface["source_face_ids"]
    }
    generated_vertices: set[int] = set()
    generated_edges: set[int] = set()
    edge_occurrence_count: dict[int, int] = defaultdict(int)
    edge_by_id = {edge["edge_id"]: edge for edge in edges}
    coedge_order: list[int] = []

    for surface_id in surface_order:
        surface = surfaces[surface_id]
        surface_nodes = [node for node in nodes if face_to_surface.get(node["face_id"]) == surface_id]
        surface_nodes.sort(key=lambda node: (not node["is_outer_loop"], node["face_id"], node["loop_id"], node["local_edge_id_in_loop"]))
        new_nodes = [node for node in surface_nodes if node["edge_id"] not in generated_edges]
        mate_nodes = [node for node in surface_nodes if node["edge_id"] in generated_edges]
        ordered_nodes = new_nodes + mate_nodes
        for node in ordered_nodes:
            coedge_order.append(node["id"])
            for vertex_id in (node.get("start_vertex_id"), node.get("end_vertex_id")):
                if vertex_id is None or vertex_id in generated_vertices:
                    continue
                generated_vertices.add(vertex_id)
                xyz = vertices[vertex_id]["xyz"]
                events.append({
                    "phase": "geometry", "kind": "vertex", "vertex_id": vertex_id,
                    "surface_id": surface_id, "title": f"NEW_VERTEX v{vertex_id}",
                    "detail": f"Exact XYZ = {xyz}; later coedges use VERTEX_REF.",
                })
            if node.get("edge_id") is not None:
                edge_id = int(node["edge_id"])
                if edge_occurrence_count[edge_id] == 1:
                    bbox = edge_by_id[edge_id]["bbox"]
                    for corner_index, corner_name in enumerate(("min_corner", "max_corner")):
                        corner = bbox[corner_name]
                        events.append({
                            "phase": "geometry", "kind": "edge_bbox_corner",
                            "edge_id": edge_id, "corner_index": corner_index,
                            "surface_id": surface_id, "point": corner,
                            "title": f"MATE_EDGE e{edge_id}: bbox corner {corner_index + 1}",
                            "detail": (
                                f"Second coedge occurrence reveals the edge AABB {corner_name} = {corner}; "
                                "coordinates are relative to the model origin."
                            ),
                        })
                edge_occurrence_count[edge_id] += 1
                generated_edges.add(edge_id)
        parameters = surface["parameters"]
        for kind, label in (("origin", "origin"), ("axis_z", "z axis"), ("axis_x", "x axis"), ("axis_y", "y axis")):
            if kind in parameters:
                events.append({
                    "phase": "geometry", "kind": kind, "surface_id": surface_id,
                    "title": f"Surface s{surface_id}: {label}",
                    "detail": f"Exact value = {parameters[kind]}; direct code = {surface['quantized'].get(kind)}.",
                })
        for parameter_name in ("radius", "reference_radius", "semi_angle", "major_radius", "minor_radius", "u_degree", "v_degree", "n_u_poles", "n_v_poles"):
            if parameter_name not in parameters:
                continue
            events.append({
                "phase": "geometry", "kind": "surface_parameter", "surface_id": surface_id,
                "parameter_name": parameter_name,
                "title": f"Surface s{surface_id}: {parameter_name}",
                "detail": f"Exact value = {parameters[parameter_name]}; direct code = {surface['quantized'].get(parameter_name, 'fixed grammar / integer')}.",
            })
        for pole_index, pole in enumerate(parameters.get("poles", [])):
            events.append({
                "phase": "geometry", "kind": "control_point", "surface_id": surface_id,
                "pole_index": pole_index, "point": pole,
                "title": f"Surface s{surface_id}: control point P{pole_index}",
                "detail": f"Exact XYZ = {pole}; direct code = {surface['quantized'].get('poles', [])[pole_index]}.",
            })
        events.append({
            "phase": "geometry", "kind": "surface", "surface_id": surface_id,
            "title": f"Surface s{surface_id}: {parameters['type']}",
            "detail": "The complete supporting surface is now known; its face trims are not yet applied.",
        })

    edge_rank = {edge_id: rank for rank, edge_id in enumerate(dict.fromkeys(nodes[n]["edge_id"] for n in coedge_order if nodes[n]["edge_id"] is not None))}
    for edge in sorted(edges, key=lambda item: edge_rank.get(item["edge_id"], 10**9)):
        events.append({
            "phase": "intersection", "kind": "intersection", "edge_id": edge["edge_id"],
            "title": f"Surface intersection candidate e{edge['edge_id']}",
            "detail": f"Incident supporting surfaces {edge['surface_ids']}; {edge['derivation_mode']}.",
        })

    events.append({
        "phase": "trim", "kind": "hide_helpers", "title": "Enter loop reconstruction",
        "detail": "Surface patches, origins, and local frames are hidden by default; display controls remain available.",
    })
    surface_rank = {surface_id: rank for rank, surface_id in enumerate(surface_order)}
    ordered_loops = sorted(loops, key=lambda loop: (loop["loop_type"] != "outer", surface_rank.get(loop["surface_id"], 10**9), loop["face_id"], loop["loop_id"]))
    for loop in ordered_loops:
        events.append({
            "phase": "trim", "kind": "loop_candidates", "loop_key": loop["key"],
            "title": f"Refer {loop['loop_type']} loop candidates",
            "detail": f"Highlight untrimmed edges {loop['edge_ids']} before enumerating trim branches.",
        })
        events.append({
            "phase": "trim", "kind": "loop_trimmed", "loop_key": loop["key"],
            "title": f"Commit {loop['loop_type']} loop",
            "detail": "Endpoint references and cyclic coedge order select the closed trimmed result.",
        })
    events.extend([
        {
            "phase": "final", "kind": "trimmed_edges", "title": "All loops reconstructed",
            "detail": "Untrimmed intersection candidates are hidden; only vertex-bounded trimmed edges remain.",
        },
        {
            "phase": "final", "kind": "trimmed_faces", "title": "Reconstructed B-Rep",
            "detail": "Loop-to-surface ownership reveals the final trimmed faces: vertices + edges + faces.",
        },
    ])
    return events, coedge_order


def extract_sample(sample: str, force: bool = False) -> Path:
    step_path = DATA_DIR / "steps" / f"{sample}.step"
    label_path = DATA_DIR / "labels" / f"{sample}.json"
    output = CACHE_DIR / f"{sample}.geometry.json"
    if output.exists() and not force:
        return output
    labels = gp.load_mfinstseg_labels(label_path)
    base = gp._extract_geometry_and_coag(step_path, labels)
    occ = gp._occ_imports()
    shape = gp._read_step(step_path, occ)
    faces, _ = gp._map_shapes(shape, occ["TopAbs_FACE"], occ)
    brep_edges, _ = gp._map_shapes(shape, occ["TopAbs_EDGE"], occ)
    vertices_occ, _ = gp._map_shapes(shape, occ["TopAbs_VERTEX"], occ)

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    face_to_signature: dict[int, tuple[Any, ...]] = {}
    for face_id, face in enumerate(faces):
        signature = _surface_key(face, face_id, occ)
        face_to_signature[face_id] = signature
        label_id = labels["seg"].get(face_id)
        instance_id = labels["face_instance_id"].get(face_id)
        is_feature = label_id not in (None, gp.STOCK_LABEL_ID) and label_id not in gp.IGNORED_FEATURE_LABEL_IDS and instance_id is not None
        record = grouped.setdefault(signature, {
            "source_face_ids": [], "feature_instance_ids": set(), "feature_label_ids": set(), "representative_face_id": face_id,
        })
        record["source_face_ids"].append(face_id)
        if is_feature:
            record["feature_instance_ids"].add(int(instance_id))
            record["feature_label_ids"].add(int(label_id))

    raw_surfaces = sorted(grouped.values(), key=lambda item: (0 if item["feature_instance_ids"] else 1, min(item["feature_instance_ids"]) if item["feature_instance_ids"] else 10**9, item["source_face_ids"][0]))
    surfaces: list[dict[str, Any]] = []
    face_to_surface: dict[int, int] = {}
    for surface_id, record in enumerate(raw_surfaces):
        face_id = record["representative_face_id"]
        face = faces[face_id]
        parameters = _surface_parameters(face, occ)
        mesh = gp._support_surface_mesh(face, base["bbox"], occ, resolution=26)
        surface = {
            "id": surface_id,
            "source_face_ids": sorted(record["source_face_ids"]),
            "feature_instance_ids": sorted(record["feature_instance_ids"]),
            "feature_label_ids": sorted(record["feature_label_ids"]),
            "is_feature_space": bool(record["feature_instance_ids"]),
            "parameters": parameters,
            "quantized": _direct_quantize(parameters, base["bbox"]),
            "mesh": mesh,
        }
        surfaces.append(surface)
        for source_face_id in surface["source_face_ids"]:
            face_to_surface[source_face_id] = surface_id

    vertices = []
    for vertex_id, vertex in enumerate(vertices_occ):
        xyz = _point(occ["BRep_Tool"].Pnt(vertex))
        vertices.append({"id": vertex_id, "xyz": xyz, "quantized": _direct_quantize({"origin": xyz}, base["bbox"])["origin"]})

    edge_faces: dict[int, set[int]] = defaultdict(set)
    for node in base["nodes"]:
        if node.get("edge_id") is not None:
            edge_faces[int(node["edge_id"])].add(int(node["face_id"]))
    edges = []
    trimmed_by_id = {int(edge["edge_id"]): edge for edge in base["brep_edges"]}
    for edge_id, edge_occ in enumerate(brep_edges):
        untrimmed, mode = _extended_curve(edge_occ, base["bbox"], occ)
        face_ids = sorted(edge_faces.get(edge_id, set()))
        surface_ids = sorted({face_to_surface[face_id] for face_id in face_ids})
        edges.append({
            "edge_id": edge_id, "edge_type": gp._edge_type(edge_occ, occ),
            "face_ids": face_ids, "surface_ids": surface_ids,
            "bbox": _edge_bbox(edge_occ, base["bbox"]),
            "untrimmed_points": untrimmed,
            "trimmed_points": trimmed_by_id[edge_id]["points"],
            "derivation_mode": mode,
        })

    loop_nodes: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for node in base["nodes"]:
        loop_nodes[(node["face_id"], node["loop_id"])].append(node)
    loops = []
    for (face_id, loop_id), members in loop_nodes.items():
        members.sort(key=lambda node: node["local_edge_id_in_loop"])
        loops.append({
            "key": f"f{face_id}-l{loop_id}", "face_id": face_id, "surface_id": face_to_surface[face_id],
            "loop_id": loop_id, "loop_type": members[0]["loop_type"],
            "edge_ids": [node["edge_id"] for node in members if node.get("edge_id") is not None],
            "coedge_ids": [node["id"] for node in members],
        })

    feature_spaces = _build_feature_spaces(labels, faces, base["face_meshes"], surfaces, base["bbox"], occ)
    surface_order = [surface["id"] for surface in surfaces]
    events, coedge_order = _build_events(surfaces, base["nodes"], edges, vertices, loops, surface_order)
    payload = {
        "schema_version": 1, "sample_name": sample,
        "method_contract": {
            "generated_geometry": ["vertex XYZ", "supporting-surface exact parameters", "mate-edge AABB corner pair"],
            "not_generated": ["edge curve geometry"],
            "edge_postprocess": "surface-pair candidate + STEP-selected branch for ground-truth demonstration + vertex/loop trim",
            "geometry_tokenizer": "direct fixed scalar quantization; no VAE",
        },
        "bbox": base["bbox"], "view_bbox": base["view_bbox"],
        "face_meshes": base["face_meshes"], "surfaces": surfaces, "feature_spaces": feature_spaces,
        "vertices": vertices, "edges": edges, "loops": loops,
        "coag_nodes": base["nodes"], "coag_edges": base["edges"],
        "surface_order": surface_order, "coedge_order": coedge_order, "events": events,
        "feature_categories": gp.FACE_CATEGORIES,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="*", help="STEP basenames; default extracts every copied sample")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    samples = args.samples or sorted(path.stem for path in (DATA_DIR / "steps").glob("*.step"))
    for sample in samples:
        path = extract_sample(sample, force=args.force)
        print(path)


if __name__ == "__main__":
    main()
