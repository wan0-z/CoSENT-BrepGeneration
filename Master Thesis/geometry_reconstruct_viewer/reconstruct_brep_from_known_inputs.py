"""Reconstruct B-Rep geometry from vertices, surfaces, and complete topology only.

Allowed input fields are exact vertex XYZ, exact supporting-surface parameters,
and CoAG/edge/loop incidence.  Original STEP edge samples and face meshes are
never used to compute intersections, trims, candidate selection, or rebuilt
faces.  They may remain in the source cache for the separate ground-truth mode.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeWire,
)
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Geom import (
    Geom_ConicalSurface, Geom_CylindricalSurface, Geom_Line,
    Geom_Plane, Geom_SphericalSurface, Geom_ToroidalSurface,
)
from OCC.Core.GeomAPI import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Ax1, gp_Ax3, gp_Dir, gp_Pnt
from scipy.spatial import Delaunay

import extract_geometry_sequence as source_extractor


APP_DIR = Path(__file__).resolve().parent
SOURCE_CACHE_DIR = APP_DIR / "web_cache"
OUTPUT_DIR = APP_DIR / "reconstruction_cache"


def _r(value: float, digits: int = 7) -> float:
    return round(float(value), digits)


def _xyz(point: Any) -> list[float]:
    return [_r(point.X()), _r(point.Y()), _r(point.Z())]


def _distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def _curve_length(points: list[list[float]]) -> float:
    return sum(_distance(points[index - 1], points[index]) for index in range(1, len(points)))


def _make_surface(parameters: dict[str, Any]) -> Any:
    if not all(key in parameters for key in ("origin", "axis_z", "axis_x")):
        raise ValueError(f"Surface lacks an exact frame: {parameters.get('type')}")
    axis = gp_Ax3(
        gp_Pnt(*parameters["origin"]), gp_Dir(*parameters["axis_z"]), gp_Dir(*parameters["axis_x"]),
    )
    kind = parameters["type"]
    if kind == "plane":
        return Geom_Plane(axis)
    if kind == "cylinder":
        return Geom_CylindricalSurface(axis, float(parameters["radius"]))
    if kind == "cone":
        return Geom_ConicalSurface(axis, float(parameters["semi_angle"]), float(parameters["reference_radius"]))
    if kind == "sphere":
        return Geom_SphericalSurface(axis, float(parameters["radius"]))
    if kind == "torus":
        return Geom_ToroidalSurface(axis, float(parameters["major_radius"]), float(parameters["minor_radius"]))
    raise ValueError(f"Unsupported exact reconstruction surface: {kind}")


def _bbox_corners(bbox: list[float]) -> list[list[float]]:
    return [
        [bbox[3 if x else 0], bbox[4 if y else 1], bbox[5 if z else 2]]
        for x, y, z in itertools.product((0, 1), repeat=3)
    ]


def _project_parameter(curve: Any, point: list[float]) -> tuple[float, float] | None:
    projector = GeomAPI_ProjectPointOnCurve(gp_Pnt(*point), curve)
    if projector.NbPoints() <= 0:
        return None
    return float(projector.LowerDistanceParameter()), float(projector.LowerDistance())


def _curve_domain(curve: Any, bbox: list[float]) -> dict[str, Any]:
    adaptor = GeomAdaptor_Curve(curve)
    first, last = float(adaptor.FirstParameter()), float(adaptor.LastParameter())
    periodic = bool(adaptor.IsPeriodic())
    period = float(adaptor.Period()) if periodic else None
    if adaptor.GetType() == GeomAbs_Line or not math.isfinite(first) or not math.isfinite(last) or abs(first) > 1e50 or abs(last) > 1e50:
        parameters = [projection[0] for corner in _bbox_corners(bbox) if (projection := _project_parameter(curve, corner))]
        if not parameters:
            parameters = [-1.0, 1.0]
        span = max(parameters) - min(parameters)
        padding = max(span * 0.12, 1.0)
        first, last = min(parameters) - padding, max(parameters) + padding
        periodic, period = False, None
    return {"first": first, "last": last, "periodic": periodic, "period": period}


def _value(curve: Any, parameter: float, domain: dict[str, Any]) -> list[float]:
    if domain["periodic"]:
        first, period = domain["first"], domain["period"]
        parameter = first + ((parameter - first) % period)
    return _xyz(curve.Value(parameter))


def _sample_interval(curve: Any, start: float, end: float, domain: dict[str, Any], count: int = 96) -> list[list[float]]:
    count = max(2, count)
    return [_value(curve, start + (end - start) * index / (count - 1), domain) for index in range(count)]


def _tangent_plane_cylinder_line(
    first_parameters: dict[str, Any], second_parameters: dict[str, Any], witness: list[float],
) -> Any | None:
    cylinder = first_parameters if first_parameters["type"] == "cylinder" else second_parameters if second_parameters["type"] == "cylinder" else None
    plane = first_parameters if first_parameters["type"] == "plane" else second_parameters if second_parameters["type"] == "plane" else None
    if cylinder is None or plane is None:
        return None
    axis = cylinder["axis_z"]
    plane_normal = plane["axis_z"]
    if abs(sum(axis[index] * plane_normal[index] for index in range(3))) > 1e-6:
        return None
    return Geom_Line(gp_Ax1(gp_Pnt(*witness), gp_Dir(*axis)))


def _intersect_pair(
    pair: tuple[int, int], surfaces: dict[int, Any], surface_parameters: dict[int, dict[str, Any]],
    bbox: list[float], witness_points: list[list[float]], tolerance: float,
) -> tuple[list[dict[str, Any]], list[Any], str]:
    solver = GeomAPI_IntSS(surfaces[pair[0]], surfaces[pair[1]], tolerance)
    curves = [solver.Line(index) for index in range(1, solver.NbLines() + 1)] if solver.IsDone() else []
    method = "GeomAPI_IntSS"
    if not curves and witness_points:
        tangent = _tangent_plane_cylinder_line(
            surface_parameters[pair[0]], surface_parameters[pair[1]], witness_points[0],
        )
        if tangent is not None:
            curves = [tangent]
            method = "analytic tangent plane-cylinder fallback"
    branches = []
    for branch_id, curve in enumerate(curves):
        domain = _curve_domain(curve, bbox)
        points = _sample_interval(curve, domain["first"], domain["last"], domain, 128)
        branches.append({
            "id": branch_id, "points": points, "periodic": domain["periodic"],
            "parameter_domain": [_r(domain["first"]), _r(domain["last"])],
            "period": _r(domain["period"]) if domain["period"] is not None else None,
        })
    return branches, curves, method


def _edge_vertex_ids(nodes: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    if not nodes:
        return None, None
    first = nodes[0]
    return first.get("start_vertex_id"), first.get("end_vertex_id")


def _candidate_segments(
    curves: list[Any], branches: list[dict[str, Any]], start_point: list[float], end_point: list[float],
    tolerance: float,
) -> list[dict[str, Any]]:
    candidates = []
    same_vertex = _distance(start_point, end_point) <= tolerance
    for branch, curve in zip(branches, curves):
        domain = {
            "first": branch["parameter_domain"][0], "last": branch["parameter_domain"][1],
            "periodic": branch["periodic"], "period": branch["period"],
        }
        start_projection = _project_parameter(curve, start_point)
        end_projection = _project_parameter(curve, end_point)
        if start_projection is None or end_projection is None:
            continue
        start_parameter, start_distance = start_projection
        end_parameter, end_distance = end_projection
        if start_distance > tolerance or end_distance > tolerance:
            continue
        intervals = []
        if domain["periodic"]:
            period = float(domain["period"])
            if same_vertex:
                intervals.append((start_parameter, start_parameter + period, "full_period"))
            else:
                forward = (end_parameter - start_parameter) % period
                if forward <= tolerance:
                    forward = period
                intervals.append((start_parameter, start_parameter + forward, "periodic_forward"))
                intervals.append((start_parameter, start_parameter - (period - forward), "periodic_complement"))
        elif not same_vertex:
            intervals.append((start_parameter, end_parameter, "open_interval"))
        for interval_start, interval_end, interval_kind in intervals:
            points = _sample_interval(curve, interval_start, interval_end, domain, 80)
            length = _curve_length(points)
            candidate = {
                "branch_id": branch["id"], "interval": [_r(interval_start), _r(interval_end)],
                "interval_kind": interval_kind, "points": points, "length": _r(length),
                "endpoint_error": _r(max(start_distance, end_distance), 9),
            }
            if not any(
                abs(existing["length"] - candidate["length"]) < tolerance
                and _distance(existing["points"][len(existing["points"]) // 2], candidate["points"][len(candidate["points"]) // 2]) < tolerance
                for existing in candidates
            ):
                candidates.append(candidate)
    candidates.sort(key=lambda item: (item["length"], item["branch_id"], item["interval_kind"]))
    for candidate_id, candidate in enumerate(candidates):
        candidate["id"] = candidate_id
    return candidates


def _reverse_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    result["points"] = list(reversed(candidate["points"]))
    result["interval"] = list(reversed(candidate["interval"]))
    return result


def _loop_solutions(loop: dict[str, Any], occurrence_candidates: list[dict[str, Any]], maximum: int = 64) -> tuple[list[dict[str, Any]], bool]:
    topologically_closed = all(
        item.get("end_vertex_id") == occurrence_candidates[(index + 1) % len(occurrence_candidates)].get("start_vertex_id")
        for index, item in enumerate(occurrence_candidates)
    ) if occurrence_candidates else False
    if not topologically_closed:
        return [], False
    option_counts = [max(1, len(item["candidates"])) for item in occurrence_candidates]
    total = math.prod(option_counts)
    solutions = []
    for choice in itertools.islice(itertools.product(*(range(count) for count in option_counts)), maximum):
        segments = []
        total_length = 0.0
        valid = True
        for item, candidate_index in zip(occurrence_candidates, choice):
            if not item["candidates"]:
                valid = False
                break
            candidate = item["candidates"][candidate_index]
            segments.append({"edge_id": item["edge_id"], "candidate_id": candidate["id"], "points": candidate["points"]})
            total_length += candidate["length"]
        if valid:
            solutions.append({
                "id": len(solutions), "choice": list(choice), "segments": segments,
                "total_length": _r(total_length), "closed_by_topology": True,
            })
    solutions.sort(key=lambda item: item["total_length"])
    for index, solution in enumerate(solutions):
        solution["id"] = index
    return solutions, total > maximum


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous; x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _project_loop_to_uv(surface: Any, segments: list[dict[str, Any]], periodic_u: bool) -> list[tuple[float, float]]:
    points_3d = []
    for segment in segments:
        points = segment.get("points") or []
        sampled = points[::4] if len(points) > 8 else points
        if sampled and sampled[-1] != points[-1]:
            sampled = [*sampled, points[-1]]
        for point in sampled:
            if not points_3d or _distance(points_3d[-1], point) > 1e-7:
                points_3d.append(point)
    result: list[tuple[float, float]] = []
    for point in points_3d:
        projector = GeomAPI_ProjectPointOnSurf(gp_Pnt(*point), surface)
        if projector.NbPoints() <= 0:
            continue
        u, v = [float(value) for value in projector.LowerDistanceParameters()]
        if periodic_u and result:
            previous_u = result[-1][0]
            while u - previous_u > math.pi:
                u -= 2 * math.pi
            while u - previous_u < -math.pi:
                u += 2 * math.pi
        if not result or math.hypot(u - result[-1][0], v - result[-1][1]) > 1e-8:
            result.append((u, v))
    if len(result) > 2 and math.hypot(result[0][0] - result[-1][0], result[0][1] - result[-1][1]) < 1e-8:
        result.pop()
    return result


def _uv_trimmed_mesh(surface: Any, parameters: dict[str, Any], face_loops: list[dict[str, Any]]) -> dict[str, Any] | None:
    outer_loop = next((loop for loop in face_loops if loop["loop_type"] == "outer" and loop["solutions"]), None)
    if outer_loop is None:
        return None
    periodic_u = parameters["type"] in {"cylinder", "cone", "sphere", "torus"}
    outer = _project_loop_to_uv(surface, outer_loop["solutions"][0]["segments"], periodic_u)
    if len(outer) < 3:
        return None
    holes = []
    outer_mean_u = sum(point[0] for point in outer) / len(outer)
    for loop in face_loops:
        if loop["loop_type"] != "inner" or not loop["solutions"]:
            continue
        hole = _project_loop_to_uv(surface, loop["solutions"][0]["segments"], periodic_u)
        if len(hole) < 3:
            continue
        if periodic_u:
            hole_mean_u = sum(point[0] for point in hole) / len(hole)
            shift = round((outer_mean_u - hole_mean_u) / (2 * math.pi)) * 2 * math.pi
            hole = [(u + shift, v) for u, v in hole]
        holes.append(hole)
    minimum_u, maximum_u = min(u for u, _ in outer), max(u for u, _ in outer)
    minimum_v, maximum_v = min(v for _, v in outer), max(v for _, v in outer)
    if maximum_u - minimum_u < 1e-10 or maximum_v - minimum_v < 1e-10:
        return None
    uv_points = list(outer)
    for hole in holes:
        uv_points.extend(hole)
    resolution = 18
    for iv in range(1, resolution):
        v = minimum_v + (maximum_v - minimum_v) * iv / resolution
        for iu in range(1, resolution):
            u = minimum_u + (maximum_u - minimum_u) * iu / resolution
            if _point_in_polygon((u, v), outer) and not any(_point_in_polygon((u, v), hole) for hole in holes):
                uv_points.append((u, v))
    # Deduplicate UV samples before Delaunay.
    unique_uv = []
    for point in uv_points:
        if not any(math.hypot(point[0] - existing[0], point[1] - existing[1]) < 1e-8 for existing in unique_uv):
            unique_uv.append(point)
    if len(unique_uv) < 3:
        return None
    triangulation = Delaunay(unique_uv)
    triangles = []
    for triangle in triangulation.simplices:
        centroid = (
            sum(unique_uv[int(index)][0] for index in triangle) / 3,
            sum(unique_uv[int(index)][1] for index in triangle) / 3,
        )
        if _point_in_polygon(centroid, outer) and not any(_point_in_polygon(centroid, hole) for hole in holes):
            triangles.append([int(index) for index in triangle])
    if not triangles:
        return None
    vertices = []
    for u, v in unique_uv:
        if periodic_u:
            u = u % (2 * math.pi)
        vertices.append(_xyz(surface.Value(u, v)))
    return {"vertices": vertices, "triangles": triangles}


def _build_reconstructed_faces(
    source: dict[str, Any], surface_objects: dict[int, Any], edge_runtime: dict[int, dict[str, Any]],
    loop_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    occ = source_extractor.gp._occ_imports()
    vertex_shapes = {
        vertex["id"]: BRepBuilderAPI_MakeVertex(gp_Pnt(*vertex["xyz"])).Vertex()
        for vertex in source["vertices"]
    }
    edge_shapes = {}
    for edge_id, runtime in edge_runtime.items():
        if runtime["status"] != "reconstructed" or not runtime["candidates"]:
            continue
        selected = runtime["candidates"][0]
        curve = runtime["curves"][selected["branch_id"]]
        start_id, end_id = runtime["canonical_vertices"]
        try:
            parameter_start, parameter_end = float(selected["interval"][0]), float(selected["interval"][1])
            shape_matches_canonical = parameter_start <= parameter_end
            edge_builder = BRepBuilderAPI_MakeEdge(
                curve, min(parameter_start, parameter_end), max(parameter_start, parameter_end),
            )
            if edge_builder.IsDone():
                edge_shapes[edge_id] = edge_builder.Edge()
                runtime["shape_matches_canonical"] = shape_matches_canonical
        except Exception:
            continue
    loops_by_face: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for loop in source["loops"]:
        loops_by_face[int(loop["face_id"])].append(loop)
    node_map = {node["id"]: node for node in source["coag_nodes"]}
    face_meshes = []
    stats = Counter()
    for face_id, loops in sorted(loops_by_face.items()):
        wire_records = []
        for loop in sorted(loops, key=lambda item: item["loop_type"] != "outer"):
            wire_builder = BRepBuilderAPI_MakeWire()
            complete = True
            for coedge_id in loop["coedge_ids"]:
                node = node_map[coedge_id]
                edge_id = node.get("edge_id")
                if edge_id not in edge_shapes:
                    complete = False
                    break
                runtime = edge_runtime[edge_id]
                canonical_start, canonical_end = runtime["canonical_vertices"]
                same_orientation = node.get("start_vertex_id") == canonical_start and node.get("end_vertex_id") == canonical_end
                use_shape_forward = same_orientation == runtime.get("shape_matches_canonical", True)
                edge_shape = edge_shapes[edge_id] if use_shape_forward else topods.Edge(edge_shapes[edge_id].Reversed())
                wire_builder.Add(edge_shape)
            if not complete or not wire_builder.IsDone():
                continue
            wire_records.append((loop["loop_type"], wire_builder.Wire()))
        outer = next((wire for loop_type, wire in wire_records if loop_type == "outer"), None)
        if outer is None:
            stats["faces_skipped_missing_outer_wire"] += 1
            continue
        surface_id = next(surface["id"] for surface in source["surfaces"] if face_id in surface["source_face_ids"])
        try:
            face_builder = BRepBuilderAPI_MakeFace(surface_objects[surface_id], outer, True)
            for loop_type, wire in wire_records:
                if loop_type != "outer":
                    face_builder.Add(wire)
            if not face_builder.IsDone():
                stats["faces_builder_failed"] += 1
                continue
            face = face_builder.Face()
            BRepMesh_IncrementalMesh(face, 0.08, False, 0.45, True)
            mesh = source_extractor.gp._triangulate_face(face, occ)
            if not mesh["triangles"]:
                stats["faces_empty_mesh"] += 1
                continue
            face_meshes.append({"face_id": face_id, "surface_id": surface_id, **mesh})
            stats["faces_reconstructed"] += 1
            stats["faces_valid"] += int(BRepCheck_Analyzer(face).IsValid())
        except Exception:
            stats["faces_exception"] += 1
    stats["edges_reconstructed"] = len(edge_shapes)
    stats["edges_requested"] = len(source["edges"])
    reconstructed_face_ids = {face["face_id"] for face in face_meshes}
    results_by_face: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for loop in loop_results:
        results_by_face[int(loop["face_id"])].append(loop)
    face_to_surface = {
        face_id: surface["id"] for surface in source["surfaces"] for face_id in surface["source_face_ids"]
    }
    for face_id, face_loops in sorted(results_by_face.items()):
        if face_id in reconstructed_face_ids:
            continue
        surface_id = face_to_surface[face_id]
        mesh = _uv_trimmed_mesh(
            surface_objects[surface_id], source["surfaces"][surface_id]["parameters"], face_loops,
        ) if surface_id in surface_objects else None
        if mesh is None:
            stats["faces_uv_fallback_failed"] += 1
            continue
        face_meshes.append({"face_id": face_id, "surface_id": surface_id, "reconstruction_method": "surface_uv_from_calculated_loops", **mesh})
        stats["faces_uv_reconstructed"] += 1
    stats["faces_total_visualized"] = len(face_meshes)
    return face_meshes, dict(stats)


def reconstruct_sample(sample: str, force: bool = False) -> Path:
    source_path = SOURCE_CACHE_DIR / f"{sample}.geometry.json"
    output_path = OUTPUT_DIR / f"{sample}.reconstruction.json"
    if output_path.exists() and not force:
        return output_path
    source = json.loads(source_path.read_text(encoding="utf-8"))
    bbox = source["view_bbox"]
    diagonal = math.sqrt(sum((bbox[index + 3] - bbox[index]) ** 2 for index in range(3))) or 1.0
    tolerance = max(diagonal * 1e-6, 1e-6)
    vertices = {vertex["id"]: vertex["xyz"] for vertex in source["vertices"]}
    surface_parameters = {surface["id"]: surface["parameters"] for surface in source["surfaces"]}
    surface_objects = {}
    unsupported_surfaces = {}
    for surface_id, parameters in surface_parameters.items():
        try:
            surface_objects[surface_id] = _make_surface(parameters)
        except Exception as exc:
            unsupported_surfaces[surface_id] = str(exc)

    nodes_by_edge: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in source["coag_nodes"]:
        if node.get("edge_id") is not None:
            nodes_by_edge[int(node["edge_id"])].append(node)
    edges_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    edge_records = {}
    for edge in source["edges"]:
        edge_id = int(edge["edge_id"])
        pair = tuple(int(value) for value in edge["surface_ids"])
        start_id, end_id = _edge_vertex_ids(nodes_by_edge[edge_id])
        record = {
            "edge_id": edge_id, "surface_ids": list(pair),
            "canonical_vertices": [start_id, end_id], "bbox": edge.get("bbox"),
            "status": "pending", "candidates": [],
        }
        edge_records[edge_id] = record
        if len(pair) == 2:
            edges_by_pair[pair].append(edge_id)
        else:
            record["status"] = "seam_or_cosurface_skipped"

    intersection_groups = []
    pair_runtime = {}
    for pair in sorted(edges_by_pair):
        witness_points = []
        for edge_id in edges_by_pair[pair]:
            start_id, end_id = edge_records[edge_id]["canonical_vertices"]
            for vertex_id in (start_id, end_id):
                if vertex_id is not None:
                    witness_points.append(vertices[vertex_id])
        if pair[0] not in surface_objects or pair[1] not in surface_objects:
            branches, curves, method = [], [], "unsupported surface type"
        else:
            branches, curves, method = _intersect_pair(
                pair, surface_objects, surface_parameters, bbox, witness_points, tolerance,
            )
        group_id = len(intersection_groups)
        pair_runtime[pair] = {"curves": curves, "branches": branches, "group_id": group_id}
        intersection_groups.append({
            "id": group_id, "surface_ids": list(pair), "edge_ids": edges_by_pair[pair],
            "method": method, "branches": branches,
        })

    edge_runtime = {}
    for edge_id, record in edge_records.items():
        pair = tuple(record["surface_ids"])
        runtime_record = dict(record)
        runtime_record["curves"] = []
        if len(pair) != 2:
            edge_runtime[edge_id] = runtime_record
            continue
        start_id, end_id = record["canonical_vertices"]
        if start_id is None or end_id is None:
            runtime_record["status"] = "missing_topological_vertex"
            edge_runtime[edge_id] = runtime_record
            continue
        pair_data = pair_runtime[pair]
        candidates = _candidate_segments(
            pair_data["curves"], pair_data["branches"], vertices[start_id], vertices[end_id], tolerance,
        )
        runtime_record.update({
            "intersection_group_id": pair_data["group_id"], "candidates": candidates,
            "status": "reconstructed" if candidates else "no_candidate",
            "curves": pair_data["curves"],
        })
        edge_runtime[edge_id] = runtime_record
        record.update({key: value for key, value in runtime_record.items() if key != "curves"})

    node_map = {node["id"]: node for node in source["coag_nodes"]}
    loop_results = []
    for loop in sorted(source["loops"], key=lambda item: (item["loop_type"] != "outer", item["face_id"], item["loop_id"])):
        occurrence_candidates = []
        for coedge_id in loop["coedge_ids"]:
            node = node_map[coedge_id]
            edge_id = int(node["edge_id"])
            edge_data = edge_records[edge_id]
            candidates = edge_data["candidates"]
            canonical_start, canonical_end = edge_data["canonical_vertices"]
            same_orientation = node.get("start_vertex_id") == canonical_start and node.get("end_vertex_id") == canonical_end
            oriented_candidates = candidates if same_orientation else [_reverse_candidate(candidate) for candidate in candidates]
            if edge_data["status"] == "seam_or_cosurface_skipped":
                oriented_candidates = [{
                    "id": -1, "branch_id": -1, "interval": [], "interval_kind": "skipped_seam",
                    "points": [], "length": 0.0, "endpoint_error": 0.0,
                }]
            occurrence_candidates.append({
                "coedge_id": coedge_id, "edge_id": edge_id,
                "start_vertex_id": node.get("start_vertex_id"), "end_vertex_id": node.get("end_vertex_id"),
                "status": edge_data["status"], "candidates": oriented_candidates,
            })
        solutions, truncated = _loop_solutions(loop, occurrence_candidates)
        ambiguous = len(solutions) > 1
        selected_solution_id = solutions[0]["id"] if solutions else None
        loop_results.append({
            "key": loop["key"], "face_id": loop["face_id"], "surface_id": loop["surface_id"],
            "loop_id": loop["loop_id"], "loop_type": loop["loop_type"],
            "occurrences": occurrence_candidates, "solutions": solutions,
            "selected_solution_id": selected_solution_id,
            "candidate_count": math.prod(max(1, len(item["candidates"])) for item in occurrence_candidates),
            "candidates_truncated": truncated, "ambiguous": ambiguous,
            "status": "ambiguous" if ambiguous else "unique" if len(solutions) == 1 else "incomplete",
        })

    rebuilt_faces, reconstruction_stats = _build_reconstructed_faces(source, surface_objects, edge_runtime, loop_results)
    events = [
        dict(event) for event in source["events"]
        if event.get("phase") in {"condition", "geometry"}
    ]
    if not events:
        events = [{
            "phase": "geometry", "kind": "reconstruction_start", "title": "Known reconstruction inputs",
            "detail": "Vertices, supporting surfaces, topology, and mate-edge AABB corners are available.",
        }]
    for group in intersection_groups:
        events.append({
            "phase": "intersection", "kind": "intersection_group", "intersection_group_id": group["id"],
            "title": f"Intersect surfaces {group['surface_ids']}",
            "detail": f"{len(group['branches'])} complete model-space branch(es), computed by {group['method']}.",
        })
    events.append({
        "phase": "trim", "kind": "hide_surface_helpers", "title": "Group calculated curves by loop",
        "detail": "Supporting surfaces and frames are hidden; outer loops are processed before inner loops.",
    })
    for loop_index, loop in enumerate(loop_results):
        events.append({
            "phase": "trim", "kind": "loop_candidates", "loop_index": loop_index,
            "title": f"Trim {loop['loop_type']} loop {loop['key']}",
            "detail": f"Topology supplies oriented endpoints for {len(loop['occurrences'])} coedges.",
        })
        if loop["ambiguous"]:
            events.append({
                "phase": "ambiguity", "kind": "ambiguity", "loop_index": loop_index,
                "title": f"Ambiguous trim: {loop['key']}",
                "detail": f"{loop['candidate_count']} topology-consistent candidate loop(s) remain. Playback is paused.",
            })
        events.append({
            "phase": "trim", "kind": "loop_commit", "loop_index": loop_index,
            "title": f"Commit default hypothesis for {loop['key']}",
            "detail": "The shortest topology-consistent candidate is retained only as a deterministic hypothesis.",
        })
    events.append({
        "phase": "final", "kind": "reconstruction_final", "title": "Surface/topology reconstruction result",
        "detail": (
            f"Reconstructed {reconstruction_stats.get('edges_reconstructed', 0)} non-seam edges and visualized "
            f"{reconstruction_stats.get('faces_total_visualized', 0)} trimmed faces "
            f"({reconstruction_stats.get('faces_reconstructed', 0)} OCC wire builds + "
            f"{reconstruction_stats.get('faces_uv_reconstructed', 0)} surface-UV fallbacks)."
        ),
    })

    serializable_edges = []
    for edge_id in sorted(edge_records):
        serializable_edges.append(edge_records[edge_id])
    payload = {
        "schema_version": 1, "sample_name": sample,
        "input_contract": {
            "allowed": [
                "exact vertex XYZ", "exact supporting-surface parameters",
                "complete edge/coedge/loop topology", "mate-edge AABB corner pair",
            ],
            "forbidden": ["original STEP edge curve", "original trimmed edge samples", "original face mesh for reconstruction"],
        },
        "bbox": source["bbox"], "view_bbox": source["view_bbox"],
        "vertices": source["vertices"], "surfaces": source["surfaces"],
        "topology": {"coag_nodes": source["coag_nodes"], "coag_edges": source["coag_edges"], "loops": source["loops"]},
        "unsupported_surfaces": unsupported_surfaces,
        "intersection_groups": intersection_groups, "edges": serializable_edges,
        "loop_results": loop_results, "reconstructed_face_meshes": rebuilt_faces,
        "reconstruction_stats": reconstruction_stats, "events": events,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="*", help="Sample basenames; default reconstructs every source cache")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    samples = args.samples or sorted(path.name.removesuffix(".geometry.json") for path in SOURCE_CACHE_DIR.glob("*.geometry.json"))
    for sample in samples:
        print(reconstruct_sample(sample, force=args.force))


if __name__ == "__main__":
    main()
