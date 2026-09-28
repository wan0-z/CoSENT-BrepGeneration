from __future__ import annotations

import argparse
import itertools
import json
import math
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec
from OCC.Core.TopAbs import TopAbs_REVERSED

import extract_one_step as legacy
from evaluation import calculate_evaluation


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE = "20221123_142528_0"
INPUT_ROOT = ROOT / "data" / "single"
OUTPUT_ROOT = ROOT / "output_space" / "single"
SEED_PATH = ROOT / "data" / "feature_space_seeds.json"
ANGLE_TOLERANCE_DEGREES = 1.0
LENGTH_TOLERANCE = 1.0e-5
RADIUS_TOLERANCE = 1.0e-5
VOLUME_RELATIVE_TOLERANCE = 1.0e-9
VOLUME_MINIMUM_TOLERANCE = 1.0e-8
MAX_MATCHES_PER_FEATURE = 5000
ACTIVE_PATCH_SAMPLE_FRACTION = 0.70


def normalized(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        raise ValueError("zero-length vector")
    return vector / length


def parallel(first: Sequence[float], second: Sequence[float]) -> bool:
    return abs(float(np.dot(normalized(first), normalized(second)))) >= math.cos(math.radians(ANGLE_TOLERANCE_DEGREES))


def perpendicular(first: Sequence[float], second: Sequence[float]) -> bool:
    return abs(float(np.dot(normalized(first), normalized(second)))) <= math.sin(math.radians(ANGLE_TOLERANCE_DEGREES))


def face_area(face: Any) -> float:
    properties = GProp_GProps()
    brepgprop.SurfaceProperties(face, properties)
    return float(properties.Mass())


def cylinder_face_orientation(face: Any) -> Tuple[int, Dict[str, Any]]:
    adaptor = BRepAdaptor_Surface(face, True)
    cylinder = adaptor.Cylinder()
    u0, u1 = float(adaptor.FirstUParameter()), float(adaptor.LastUParameter())
    v0, v1 = float(adaptor.FirstVParameter()), float(adaptor.LastVParameter())
    u = 0.5 * (u0 + u1)
    v = 0.5 * (v0 + v1)
    point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    adaptor.D1(u, v, point, du, dv)
    normal = du.Crossed(dv)
    if face.Orientation() == TopAbs_REVERSED:
        normal.Reverse()
    normal.Normalize()
    axis = cylinder.Axis()
    direction = axis.Direction()
    origin = axis.Location()
    q = gp_Vec(origin, point)
    direction_vector = gp_Vec(direction)
    radial = q.Subtracted(direction_vector.Multiplied(q.Dot(direction_vector)))
    radial.Normalize()
    sigma = 1 if radial.Dot(normal) >= 0.0 else -1
    canonical_axis = legacy.canonical_direction(direction.X(), direction.Y(), direction.Z())
    axis_location = legacy.canonical_axis_location(origin, canonical_axis)
    return sigma, {
        "axis": list(canonical_axis),
        "axis_location": list(axis_location),
        "radius": float(cylinder.Radius()),
        "angular_span": min(2.0 * math.pi, abs(u1 - u0)),
    }


def plane_face_orientation(face: Any) -> Tuple[int, Dict[str, Any]]:
    halfspace = legacy.get_oriented_plane_halfspace(face)
    if halfspace is None:
        raise ValueError("not a plane")
    support = legacy.get_surface_support_key(face)
    canonical = normalized(support[1])
    representative = normalized(halfspace["normal"])
    sign = 1 if float(np.dot(canonical, representative)) >= 0.0 else -1
    return sign, {"normal": representative.tolist(), "offset": float(halfspace["offset"])}


def build_oriented_surface_graph(faces: Sequence[Any], available_faces: Set[int]) -> nx.Graph:
    groups: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for face_id in sorted(available_faces):
        face = faces[face_id]
        adaptor = BRepAdaptor_Surface(face, True)
        surface_type = adaptor.GetType()
        if surface_type not in {GeomAbs_Plane, GeomAbs_Cylinder}:
            continue
        support_key = legacy.get_surface_support_key(face)
        support_string = json.dumps(support_key, sort_keys=True)
        if surface_type == GeomAbs_Plane:
            orientation, geometry = plane_face_orientation(face)
            kind = "plane"
        else:
            orientation, geometry = cylinder_face_orientation(face)
            kind = "cylinder"
        key = (support_string, orientation)
        if key not in groups:
            groups[key] = {
                "surface_type": kind,
                "support_key": list(support_key),
                "support_string": support_string,
                "orientation_sign": orientation,
                "representative_direction": geometry.get("normal") or ([value * orientation for value in geometry["axis"]]),
                "geometry": geometry,
                "face_ids": [],
                "face_areas": {},
                "angular_span": 0.0,
            }
        groups[key]["face_ids"].append(face_id)
        groups[key]["face_areas"][str(face_id)] = face_area(face)
        groups[key]["angular_span"] = min(2.0 * math.pi, groups[key]["angular_span"] + float(geometry.get("angular_span", 0.0)))

    graph = nx.Graph()
    for surface_id, attributes in enumerate(groups.values()):
        graph.add_node(surface_id, surface_id=surface_id, **attributes)
    for first, second in itertools.combinations(graph.nodes, 2):
        a, b = graph.nodes[first], graph.nodes[second]
        if a["support_string"] == b["support_string"]:
            continue
        graph.add_edge(first, second, code=edge_code(a, b))
    return graph


def edge_code(first: Dict[str, Any], second: Dict[str, Any]) -> str:
    if first["surface_type"] == second["surface_type"] == "plane":
        n0, n1 = first["geometry"]["normal"], second["geometry"]["normal"]
        is_parallel = parallel(n0, n1)
        opposite = is_parallel and float(np.dot(normalized(n0), normalized(n1))) < 0.0
        return "00" + ("1" if is_parallel else "0") + ("1" if opposite else "0") + ("1" if perpendicular(n0, n1) else "0")
    if {first["surface_type"], second["surface_type"]} == {"plane", "cylinder"}:
        plane_node = first if first["surface_type"] == "plane" else second
        cylinder_node = second if first["surface_type"] == "plane" else first
        normal = normalized(plane_node["geometry"]["normal"])
        axis = normalized(cylinder_node["geometry"]["axis"])
        axis_perpendicular = perpendicular(axis, normal)
        axis_parallel = parallel(axis, normal)
        location = np.asarray(cylinder_node["geometry"]["axis_location"], dtype=float)
        distance = abs(float(np.dot(normal, location) - plane_node["geometry"]["offset"]))
        tangent = axis_perpendicular and abs(distance - float(cylinder_node["geometry"]["radius"])) <= max(LENGTH_TOLERANCE, RADIUS_TOLERANCE)
        return "01" + ("1" if axis_perpendicular else "0") + ("1" if tangent else "0") + ("1" if axis_parallel else "0")
    axis0 = normalized(first["geometry"]["axis"])
    axis1 = normalized(second["geometry"]["axis"])
    axes_parallel = parallel(axis0, axis1)
    location0 = np.asarray(first["geometry"]["axis_location"], dtype=float)
    location1 = np.asarray(second["geometry"]["axis_location"], dtype=float)
    coaxial = axes_parallel and float(np.linalg.norm(np.cross(location1 - location0, axis0))) <= LENGTH_TOLERANCE
    equal_radius = abs(float(first["geometry"]["radius"]) - float(second["geometry"]["radius"])) <= RADIUS_TOLERANCE
    return "11" + ("1" if axes_parallel else "0") + ("1" if coaxial else "0") + ("1" if equal_radius else "0")


def load_seeds(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        features = json.load(file)["features"]
    for feature in features:
        feature["cylinder_count"] = sum(node["surface_type"] == "cylinder" for node in feature["nodes"])
        feature["edge_one_bit_sum"] = sum(edge["code"].count("1") for edge in feature.get("edges", []))
        feature["node_count"] = len(feature["nodes"])
    return sorted(features, key=lambda f: (-f["node_count"], -f["cylinder_count"], -f["edge_one_bit_sum"], f["category_id"]))


def candidate_nodes(graph: nx.Graph, seed_node: Dict[str, Any]) -> List[int]:
    result = []
    for node_id, attributes in graph.nodes(data=True):
        if attributes["surface_type"] != seed_node["surface_type"]:
            continue
        if seed_node.get("trimmed") and attributes["surface_type"] == "cylinder" and attributes["angular_span"] > math.pi + math.radians(2.0):
            continue
        if not seed_node.get("trimmed") and attributes["surface_type"] == "cylinder" and attributes["angular_span"] <= math.pi + math.radians(2.0):
            continue
        result.append(int(node_id))
    return result


def find_seed_matches(graph: nx.Graph, feature: Dict[str, Any]) -> List[Dict[int, int]]:
    seed_nodes = {int(node["id"]): node for node in feature["nodes"]}
    seed_edges = {frozenset(int(value) for value in edge["nodes"]): edge["code"] for edge in feature.get("edges", [])}
    candidates = {node_id: candidate_nodes(graph, node) for node_id, node in seed_nodes.items()}
    order = sorted(seed_nodes, key=lambda node_id: len(candidates[node_id]))
    matches: List[Dict[int, int]] = []
    mapping: Dict[int, int] = {}
    used: Set[int] = set()

    def visit(index: int) -> None:
        if len(matches) >= MAX_MATCHES_PER_FEATURE:
            return
        if index == len(order):
            if surface_constraints_satisfied(graph, feature, mapping):
                matches.append(dict(mapping))
            return
        seed_id = order[index]
        for model_id in candidates[seed_id]:
            if model_id in used:
                continue
            if any(graph.nodes[model_id]["support_string"] == graph.nodes[other]["support_string"] for other in used):
                continue
            valid = True
            for mapped_seed, mapped_model in mapping.items():
                required = seed_edges.get(frozenset((seed_id, mapped_seed)))
                if required is None:
                    continue
                if not graph.has_edge(model_id, mapped_model) or graph.edges[model_id, mapped_model]["code"] != required:
                    valid = False
                    break
            if not valid:
                continue
            mapping[seed_id] = model_id
            used.add(model_id)
            visit(index + 1)
            used.remove(model_id)
            del mapping[seed_id]
    visit(0)
    unique, seen = [], set()
    for mapping in matches:
        key = tuple(sorted(mapping.values()))
        if key not in seen:
            seen.add(key)
            unique.append(mapping)
    return unique


def closed_prismatic_section(graph: nx.Graph, mapped_nodes: Sequence[int]) -> Tuple[bool, Optional[np.ndarray]]:
    normals = np.asarray([graph.nodes[node]["geometry"]["normal"] for node in mapped_nodes], dtype=float)
    offsets = np.asarray([graph.nodes[node]["geometry"]["offset"] for node in mapped_nodes], dtype=float)
    if np.linalg.matrix_rank(normals, tol=1.0e-7) != 2:
        return False, None
    _, _, vh = np.linalg.svd(normals)
    axis = normalized(vh[-1])
    result = linprog(np.zeros(len(normals)), A_eq=np.vstack([normals.T, np.ones(len(normals))]), b_eq=np.array([0.0, 0.0, 0.0, 1.0]), bounds=[(1.0e-5, None)] * len(normals), method="highs")
    if not result.success:
        return False, None
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    e1 = normalized(np.cross(axis, helper))
    e2 = np.cross(axis, e1)
    matrix = np.column_stack([normals @ e1, normals @ e2])
    vertices: List[np.ndarray] = []
    active = [False] * len(normals)
    for first, second in itertools.combinations(range(len(normals)), 2):
        local = matrix[[first, second], :]
        if abs(float(np.linalg.det(local))) <= 1.0e-9:
            continue
        point = np.linalg.solve(local, offsets[[first, second]])
        if np.all(matrix @ point - offsets >= -1.0e-5):
            vertices.append(point)
            active[first] = True
            active[second] = True
    if len(vertices) < 3 or not all(active):
        return False, None
    if max(float(np.linalg.norm(point)) for point in vertices) > 1.0e12:
        return False, None
    return True, axis


def surface_constraints_satisfied(graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int]) -> bool:
    axes: Dict[Tuple[int, ...], np.ndarray] = {}
    for constraint in feature.get("surface_constraints", {}).get("group_constraints", []):
        kind = constraint["type"]
        if kind == "closed_oriented_prismatic_section":
            seed_nodes = tuple(int(value) for value in constraint["nodes"])
            valid, axis = closed_prismatic_section(graph, [mapping[value] for value in seed_nodes])
            if not valid or axis is None:
                return False
            axes[seed_nodes] = axis
        elif kind == "bottom_normal_parallel_group_axis":
            side_nodes = tuple(int(value) for value in constraint["side_nodes"])
            axis = axes.get(side_nodes)
            if axis is None:
                valid, axis = closed_prismatic_section(graph, [mapping[value] for value in side_nodes])
                if not valid or axis is None:
                    return False
            bottom_normal = graph.nodes[mapping[int(constraint["bottom"])]] ["geometry"]["normal"]
            if not parallel(axis, bottom_normal):
                return False
        elif kind == "radius_order":
            larger = graph.nodes[mapping[int(constraint["larger"])]] ["geometry"]["radius"]
            smaller = graph.nodes[mapping[int(constraint["smaller"])]] ["geometry"]["radius"]
            if float(larger) <= float(smaller) + RADIUS_TOLERANCE:
                return False
        elif kind == "trimmed_piecewise_closed_section":
            if any(not graph.nodes[mapping[int(value)]]["face_ids"] for value in constraint["boundary_nodes"]):
                return False
    return True


def face_bounds(graph: nx.MultiGraph, face_ids: Set[int]) -> Optional[Tuple[float, float, float, float, float, float]]:
    return legacy.feature_faces_sample_bounding_box(graph, face_ids)


def expanded_bounds(bounds: Sequence[float], ratio: float = 0.1) -> Tuple[float, float, float, float, float, float]:
    minimum = np.asarray(bounds[:3], dtype=float)
    maximum = np.asarray(bounds[3:], dtype=float)
    margin = np.maximum((maximum - minimum) * ratio, 1.0e-5)
    return tuple((minimum - margin).tolist() + (maximum + margin).tolist())


def shape_volume(shape: Optional[Any]) -> float:
    if shape is None or shape.IsNull():
        return 0.0
    properties = GProp_GProps()
    brepgprop.VolumeProperties(shape, properties)
    return max(0.0, float(properties.Mass()))


def common(first: Any, second: Any) -> Optional[Any]:
    operation = BRepAlgoAPI_Common(first, second)
    operation.Build()
    if not operation.IsDone() or operation.Shape().IsNull():
        return None
    return operation.Shape()


def cylinder_collision_shape(surface: Dict[str, Any], bounds: Sequence[float]) -> Optional[Any]:
    geometry = surface["geometry"]
    axis = normalized(geometry["axis"])
    origin = np.asarray(geometry["axis_location"], dtype=float)
    points = [np.asarray([x, y, z], dtype=float) for x in (bounds[0], bounds[3]) for y in (bounds[1], bounds[4]) for z in (bounds[2], bounds[5])]
    parameters = [float(np.dot(point - origin, axis)) for point in points]
    low, high = min(parameters), max(parameters)
    signed_offset = -1.0e-5 if int(surface["orientation_sign"]) < 0 else 1.0e-5
    radius = max(1.0e-6, float(geometry["radius"]) + signed_offset)
    base = origin + low * axis
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*base), gp_Dir(*axis)), radius, high - low).Shape()


def trimmed_patch_prism_shape(
    cylinder_surface: Dict[str, Any],
    fag: nx.MultiGraph,
    face_ids: Set[int],
    extension_ratio: float,
) -> Optional[Any]:
    axis = normalized(cylinder_surface["geometry"]["axis"])
    origin = np.asarray(cylinder_surface["geometry"]["axis_location"], dtype=float)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    first_basis = normalized(np.cross(axis, helper))
    second_basis = np.cross(axis, first_basis)
    points = [np.asarray(point, dtype=float) for face_id in face_ids for point in legacy.candidate_face_sample_points(fag, face_id)]
    if len(points) < 4:
        return None
    axial = np.asarray([float(np.dot(point - origin, axis)) for point in points])
    section = np.asarray([[float(np.dot(point - origin, first_basis)), float(np.dot(point - origin, second_basis))] for point in points])
    try:
        hull = ConvexHull(section)
    except Exception:
        return None
    polygon = section[hull.vertices]
    center = np.mean(polygon, axis=0)
    radial_scale = max(1.0, float(np.max(np.linalg.norm(polygon - center, axis=1))))
    polygon = center + (polygon - center) * max(0.0, 1.0 - 1.0e-5 / radial_scale)
    low, high = float(np.min(axial)), float(np.max(axial))
    margin = max((high - low) * extension_ratio, 1.0e-4)
    low -= margin
    high += margin
    builder = BRepBuilderAPI_MakePolygon()
    for first_value, second_value in polygon:
        point = origin + low * axis + float(first_value) * first_basis + float(second_value) * second_basis
        builder.Add(gp_Pnt(*point))
    builder.Close()
    face = BRepBuilderAPI_MakeFace(builder.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(*(axis * (high - low)))).Shape()


def collision_mesh(cell: Any) -> Dict[str, Any]:
    """Tessellate the exact Boolean input solid, including its artificial caps."""
    cell_faces = legacy.unique_shapes(legacy.explore_shapes(cell, legacy.TopAbs_FACE))
    meshes = legacy.triangulate_faces(cell, cell_faces)
    vertices, triangles = [], []
    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(mesh['vertices'])
        triangles.extend([[offset + index for index in triangle] for triangle in mesh['triangles']])
    return {'vertices': vertices, 'triangles': triangles}


def collision_free(part_shape: Any, fag: nx.MultiGraph, surface_graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int], face_ids: Set[int], capture_mesh: bool = False) -> Tuple[bool, Dict[str, Any]]:
    ratio = float(feature["face_constraints"]["recession_volume"].get("extension_ratio", 0.1))
    mapped = [surface_graph.nodes[mapping[int(node["id"])]] for node in feature["nodes"]]
    bounds = face_bounds(fag, face_ids)
    if bounds is None:
        return False, {"status": "missing_face_bounds"}
    bounds = expanded_bounds(bounds, ratio)
    if feature["space_model"] == "trimmed_boundary_cell" and feature["name"] != "circular_through_slot":
        trimmed_cylinders = [node for node in mapped if node["surface_type"] == "cylinder"]
        if trimmed_cylinders:
            trimmed_cell = trimmed_patch_prism_shape(trimmed_cylinders[0], fag, face_ids, ratio)
            if trimmed_cell is not None:
                collision_volume = shape_volume(common(trimmed_cell, part_shape))
                part_volume = shape_volume(part_shape)
                tolerance = max(VOLUME_MINIMUM_TOLERANCE, part_volume * VOLUME_RELATIVE_TOLERANCE)
                return collision_volume <= tolerance, {"status": "collision_free" if collision_volume <= tolerance else "collision_detected", "method": "trimmed_section_convex_hull_prism_boolean", "collision_volume": collision_volume, "volume_tolerance": tolerance, "feature_bounding_box": list(bounds), **({'mesh': collision_mesh(trimmed_cell)} if capture_mesh else {})}
    plane_halfspaces = [(node["geometry"]["normal"], node["geometry"]["offset"]) for node in mapped if node["surface_type"] == "plane"]
    if plane_halfspaces:
        cell, record = legacy.build_clipped_planar_feature_space(plane_halfspaces, bounds)
    else:
        cell = BRepPrimAPI_MakeBox(gp_Pnt(bounds[0], bounds[1], bounds[2]), gp_Pnt(bounds[3], bounds[4], bounds[5])).Shape()
        record = {"feature_bounding_box": list(bounds)}
    if cell is None:
        return False, {"status": "cell_build_failed", **record}
    cylinder_count = 0
    for node in mapped:
        if node["surface_type"] != "cylinder":
            continue
        cylinder_count += 1
        cylinder = cylinder_collision_shape(node, bounds)
        if cylinder is None:
            return False, {"status": "cylinder_cell_build_failed"}
        if int(node["orientation_sign"]) < 0:
            cell = common(cell, cylinder)
        else:
            operation = BRepAlgoAPI_Cut(cell, cylinder)
            operation.Build()
            cell = operation.Shape() if operation.IsDone() else None
        if cell is None or cell.IsNull():
            return False, {"status": "empty_or_failed_oriented_cell"}
    collision_volume = shape_volume(common(cell, part_shape))
    part_volume = shape_volume(part_shape)
    tolerance = max(VOLUME_MINIMUM_TOLERANCE, part_volume * VOLUME_RELATIVE_TOLERANCE)
    method = "exact_oriented_plane_cylinder_boolean"
    if feature["space_model"] == "trimmed_boundary_cell":
        method = "full_support_boolean_with_trimmed_patch_membership"
    return collision_volume <= tolerance, {"status": "collision_free" if collision_volume <= tolerance else "collision_detected", "method": method, "cylinder_count": cylinder_count, "collision_volume": collision_volume, "volume_tolerance": tolerance, **record, **({'mesh': collision_mesh(cell)} if capture_mesh else {})}


def angular_constraint_satisfied(feature: Dict[str, Any], surface_graph: nx.Graph, mapping: Dict[int, int]) -> bool:
    constraint = feature.get("face_constraints", {}).get("cylindrical_angular_span")
    if not constraint:
        return True
    role_to_id = {node["role"]: int(node["id"]) for node in feature["nodes"]}
    span = float(surface_graph.nodes[mapping[role_to_id[constraint["role"]]]]["angular_span"])
    threshold = float(constraint["radians"])
    return span > threshold if constraint["operator"] == ">" else span <= threshold + math.radians(2.0)


def oriented_support_value(surface: Dict[str, Any], point: Sequence[float]) -> float:
    position = np.asarray(point, dtype=float)
    if surface["surface_type"] == "plane":
        return float(np.dot(np.asarray(surface["geometry"]["normal"], dtype=float), position) - float(surface["geometry"]["offset"]))
    axis = normalized(surface["geometry"]["axis"])
    origin = np.asarray(surface["geometry"]["axis_location"], dtype=float)
    delta = position - origin
    radial = delta - float(np.dot(delta, axis)) * axis
    phi = float(np.dot(radial, radial)) - float(surface["geometry"]["radius"]) ** 2
    return float(surface["orientation_sign"]) * phi


def selected_patches_lie_on_active_cell(
    fag: nx.MultiGraph,
    surface_graph: nx.Graph,
    mapping: Dict[int, int],
    role_faces: Dict[int, Set[int]],
) -> bool:
    surfaces = {seed_id: surface_graph.nodes[model_id] for seed_id, model_id in mapping.items()}
    for seed_id, face_ids in role_faces.items():
        for face_id in face_ids:
            points = legacy.candidate_face_sample_points(fag, face_id)
            if not points:
                return False
            scale = max(1.0, max(abs(float(value)) for point in points for value in point))
            tolerance = 1.0e-5 * scale
            active_points = 0
            for point in points:
                if all(other_seed == seed_id or oriented_support_value(surface, point) >= -tolerance for other_seed, surface in surfaces.items()):
                    active_points += 1
            # A topological patch may extend beyond the final cell. It is a
            # valid boundary member when a nontrivial sampled portion lies on
            # the active cell boundary; requiring the complete patch would
            # reject labelled generalized faces that overhang an opening.
            if active_points < max(2, int(math.ceil(ACTIVE_PATCH_SAMPLE_FRACTION * len(points)))):
                return False
    return True


def choose_faces(part_shape: Any, fag: nx.MultiGraph, surface_graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int]) -> Tuple[Optional[Set[int]], Dict[int, Set[int]], Dict[str, Any]]:
    role_candidates = {seed_id: sorted(surface_graph.nodes[model_id]["face_ids"], key=lambda face_id: -surface_graph.nodes[model_id]["face_areas"].get(str(face_id), 0.0)) for seed_id, model_id in mapping.items()}
    if any(not values for values in role_candidates.values()):
        return None, {}, {"status": "empty_role"}
    seed_order = sorted(role_candidates)
    best: Optional[Tuple[float, Set[int], Dict[str, Any]]] = None
    combination_count = 0
    active_rejected = 0
    collision_rejected = 0
    last_collision = None
    for combination in itertools.product(*(role_candidates[seed_id] for seed_id in seed_order)):
        if len(set(combination)) != len(combination):
            continue
        combination_count += 1
        if combination_count > MAX_MATCHES_PER_FEATURE:
            break
        trial = set(combination)
        trial_roles = {seed_id: {face_id} for seed_id, face_id in zip(seed_order, combination)}
        if not selected_patches_lie_on_active_cell(fag, surface_graph, mapping, trial_roles):
            active_rejected += 1
            continue
        valid, trial_record = collision_free(part_shape, fag, surface_graph, feature, mapping, trial)
        if not valid:
            collision_rejected += 1
            last_collision = trial_record
            continue
        bounds = face_bounds(fag, trial)
        if bounds is None:
            continue
        diagonal = math.sqrt(sum((bounds[index + 3] - bounds[index]) ** 2 for index in range(3)))
        if best is None or diagonal < best[0]:
            best = (diagonal, trial, trial_record)
    if best is None:
        return None, {}, {"status": "no_spatially_compatible_basic_face_combination", "tested_combinations": combination_count, "active_boundary_rejections": active_rejected, "collision_rejections": collision_rejected, "last_collision": last_collision}
    _, selected, record = best
    for face_id in [value for values in role_candidates.values() for value in values if value not in selected]:
        trial = set(selected)
        trial.add(face_id)
        trial_roles = {seed_id: {candidate for candidate in trial if candidate in candidates} for seed_id, candidates in role_candidates.items()}
        if not selected_patches_lie_on_active_cell(fag, surface_graph, mapping, trial_roles):
            continue
        accepted, trial_record = collision_free(part_shape, fag, surface_graph, feature, mapping, trial)
        if accepted:
            selected = trial
            record = trial_record
    role_groups = {seed_id: {face_id for face_id in selected if face_id in candidates} for seed_id, candidates in role_candidates.items()}
    if any(not group for group in role_groups.values()):
        return None, {}, {"status": "required_role_lost"}
    return selected, role_groups, record


def serialize_sag(graph: nx.Graph) -> Dict[str, Any]:
    return {
        "nodes": [{key: value for key, value in attributes.items() if key != "face_areas"} for _, attributes in graph.nodes(data=True)],
        "edges": [{"source": int(first), "target": int(second), "code": attributes["code"]} for first, second, attributes in graph.edges(data=True)],
    }


def extract(part_shape: Any, faces: Sequence[Any], fag: nx.MultiGraph, features: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, Optional[int]], Dict[str, Any]]:
    available = set(range(len(faces)))
    predictions = {face_id: legacy.STOCK_CATEGORY_ID for face_id in available}
    instance_ids: Dict[int, Optional[int]] = {face_id: None for face_id in available}
    instances: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []
    next_instance = 0
    for priority, feature in enumerate(features, 1):
        while True:
            surface_graph = build_oriented_surface_graph(faces, available)
            if not snapshots:
                snapshots.append(serialize_sag(surface_graph))
            matches = find_seed_matches(surface_graph, feature)
            accepted_one = False
            for mapping in matches:
                if not angular_constraint_satisfied(feature, surface_graph, mapping):
                    continue
                selected, role_groups, collision = choose_faces(part_shape, fag, surface_graph, feature, mapping)
                if not selected or not selected.issubset(available):
                    continue
                role_names = {int(node["id"]): node["role"] for node in feature["nodes"]}
                _, collision = collision_free(part_shape, fag, surface_graph, feature, mapping, selected, capture_mesh=True)
                instance = {
                    "instance_id": next_instance,
                    "category_id": int(feature["category_id"]),
                    "category_name": feature["name"],
                    "priority_index": priority,
                    "variant": "space",
                    "basic_face_ids": sorted(selected),
                    "face_ids": sorted(selected),
                    "surface_mapping": {str(seed): int(model) for seed, model in mapping.items()},
                    "role_groups": {role_names[seed]: sorted(values) for seed, values in role_groups.items()},
                    "space_model": feature["space_model"],
                    "collision": collision,
                    "sag_complexity": {"cylinder_count": feature["cylinder_count"], "edge_one_bit_sum": feature["edge_one_bit_sum"], "node_count": feature["node_count"]},
                }
                instances.append(instance)
                for face_id in selected:
                    predictions[face_id] = int(feature["category_id"])
                    instance_ids[face_id] = next_instance
                available.difference_update(selected)
                next_instance += 1
                accepted_one = True
                break
            if not accepted_one:
                break
    final_sag = build_oriented_surface_graph(faces, available)
    return instances, predictions, instance_ids, {"initial": snapshots[0] if snapshots else serialize_sag(final_sag), "remaining": serialize_sag(final_sag)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one STEP file with oriented surface-space seeds.")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--step", type=Path)
    parser.add_argument("--label", type=Path)
    parser.add_argument("--seeds", type=Path, default=SEED_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    sample = str(args.sample)
    step_path = (args.step or INPUT_ROOT / sample / f"{sample}.step").resolve()
    label_path = (args.label or INPUT_ROOT / sample / f"{sample}.json").resolve()
    output = (args.output or OUTPUT_ROOT / sample).resolve()
    output.mkdir(parents=True, exist_ok=True)
    legacy.STEP_PATH = step_path
    legacy.GROUND_TRUTH_JSON_PATH = label_path
    legacy.FEATURE_SEED_JSON_PATH = args.seeds.resolve()
    shape = legacy.load_step_shape()
    ground_truth = legacy.load_ground_truth()
    features = load_seeds(args.seeds.resolve())
    fag, faces, fag_statistics = legacy.build_attributed_fag(shape)
    instances, predicted_seg, predicted_instance_ids, sag = extract(shape, faces, fag, features)
    cache = legacy.build_cache(shape, faces, fag, fag_statistics, ground_truth, instances, predicted_seg, predicted_instance_ids)
    cache["extraction_mode"] = "space"
    cache["surface_attributed_graph"] = sag
    cache["feature_seed_json_path"] = str(args.seeds.resolve())
    cache["evaluation"] = calculate_evaluation(faces=cache["faces"], instances=instances, category_names=legacy.FACE_CATEGORIES, target_category_ids=[int(feature["category_id"]) for feature in features] + [legacy.STOCK_CATEGORY_ID])
    with (output / "cache.json").open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)
    with (output / "surface_attributed_graph.json").open("w", encoding="utf-8") as file:
        json.dump(sag, file, ensure_ascii=False, indent=2)
    with (output / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(cache["evaluation"], file, ensure_ascii=False, indent=2)
    print(f"[DONE] sample={sample} instances={len(instances)}")
    print(f"[DONE] cache={output / 'cache.json'}")
    print(f"[DONE] face_accuracy={cache['evaluation']['face_accuracy']:.6f}")
    print(f"[DONE] feature_face_accuracy={cache['evaluation']['machining_feature_face_accuracy']:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        print(f"[FAILED] {exception}")
        traceback.print_exc()
        raise
