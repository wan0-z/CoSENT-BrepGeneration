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

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeHalfSpace
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Transform
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.TopAbs import TopAbs_REVERSED, TopAbs_EDGE

import extract_one_step as legacy
from evaluation import calculate_evaluation
from copy_data import copy_sample


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE = "20221123_142528_1059"
DATASET_ROOT = (
    ROOT.parent
    / "MF_Explorer"
    / "data"
    / "mfinstseg"
)
INPUT_ROOT = ROOT / "output_space" / "single"
OUTPUT_ROOT = ROOT / "output_space" / "single"
SEED_PATH = ROOT / "data" / "feature_space_seeds.json"
ANGLE_TOLERANCE_DEGREES = 1.0
LENGTH_TOLERANCE = 1.0e-5
RADIUS_TOLERANCE = 1.0e-5
VOLUME_RELATIVE_TOLERANCE = 1.0e-9
VOLUME_MINIMUM_TOLERANCE = 1.0e-8
MAX_MATCHES_PER_FEATURE = 5000
ACTIVE_PATCH_SAMPLE_FRACTION = 0.5


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
    axis_vector = normalized(canonical_axis)
    basis_x = normalized(np.eye(3)[np.argmin(np.abs(axis_vector))] - axis_vector * axis_vector[np.argmin(np.abs(axis_vector))])
    basis_y = np.cross(axis_vector, basis_x)
    start_point = adaptor.Value(u0, v)
    start_radial = np.array(start_point.Coord()) - np.asarray(axis_location)
    start_angle = math.atan2(float(start_radial @ basis_y), float(start_radial @ basis_x))
    angular_direction = 1 if float(np.dot(np.array(du.Coord()), np.cross(axis_vector, np.array(radial.Coord())))) >= 0 else -1
    span = min(2.0 * math.pi, abs(u1 - u0))
    interval_start = start_angle if angular_direction > 0 else start_angle - span
    return sigma, {
        "axis": list(canonical_axis),
        "axis_location": list(axis_location),
        "radius": float(cylinder.Radius()),
        "angular_span": min(2.0 * math.pi, abs(u1 - u0)),
        "angular_interval": {"start": interval_start % (2.0 * math.pi), "length": span},
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
                "face_angular_intervals": {},
            }
        groups[key]["face_ids"].append(face_id)
        groups[key]["face_areas"][str(face_id)] = face_area(face)
        if kind == "cylinder":
            groups[key]["face_angular_intervals"][str(face_id)] = geometry["angular_interval"]

    graph = nx.Graph()
    for surface_id, attributes in enumerate(groups.values()):
        if attributes["surface_type"] == "cylinder":
            attributes["angular_span"] = minimum_covering_arc(list(attributes["face_angular_intervals"].values()))
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
        definition = json.load(file)
        features = definition["features"]
    for feature in features:
        feature["_matching_policy"] = definition.get("matching_policy", {})
        feature["cylinder_count"] = sum(node["surface_type"] == "cylinder" for node in feature["nodes"])
        feature["edge_one_bit_sum"] = sum(edge["code"].count("1") for edge in feature.get("edges", []))
        feature["node_count"] = len(feature["nodes"])
    return sorted(features, key=lambda f: (-f["node_count"], -f["cylinder_count"], -f["edge_one_bit_sum"], f["category_id"]))


def candidate_nodes(graph: nx.Graph, seed_node: Dict[str, Any]) -> List[int]:
    result = []
    for node_id, attributes in graph.nodes(data=True):
        if attributes["surface_type"] != seed_node["surface_type"]:
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


def shape_volume(shape: Optional[Any]) -> float:
    if shape is None or shape.IsNull():
        return 0.0
    properties = GProp_GProps()
    brepgprop.VolumeProperties(shape, properties)
    return max(0.0, float(properties.Mass()))


def copy_shape(shape: Any) -> Any:
    copied = BRepBuilderAPI_Copy(shape, True, False).Shape()
    if copied.IsNull():
        raise RuntimeError("OpenCascade shape copy failed")
    return copied


def boolean_shape(operation_type: Any, first: Any, second: Any) -> Any:
    operation = operation_type(first, second)
    operation.SetNonDestructive(True)
    operation.Build()
    if not operation.IsDone() or operation.Shape().IsNull():
        raise RuntimeError("OpenCascade Boolean operation failed")
    return operation.Shape()


def common(first: Any, second: Any) -> Any:
    return boolean_shape(BRepAlgoAPI_Common, first, second)


def prepare_face_geometry(part_shape: Any, faces: Sequence[Any], fag: nx.MultiGraph) -> None:
    if "_space_faces" in fag.graph:
        return
    pristine_part_shape = copy_shape(part_shape)
    fag.graph["_space_faces"] = faces
    fag.graph["_space_pristine_part_shape"] = pristine_part_shape
    fag.graph["_space_part_volume"] = shape_volume(pristine_part_shape)
    meshes = legacy.triangulate_faces(part_shape, faces)
    fag.graph["_space_meshes"] = meshes
    all_vertices = np.array([point for mesh in meshes for point in mesh["vertices"]])
    fag.graph["_space_length_scale"] = float(np.linalg.norm(np.ptp(all_vertices, axis=0))) if len(all_vertices) else 1.
    for face_id, mesh in enumerate(meshes):
        vertices = np.asarray(mesh["vertices"], dtype=float)
        triangles = np.asarray(mesh["triangles"], dtype=int)
        if not len(triangles):
            fag.nodes[face_id]["_space_area_samples"] = []
            continue
        triplets = vertices[triangles]
        areas = np.linalg.norm(np.cross(triplets[:, 1] - triplets[:, 0], triplets[:, 2] - triplets[:, 0]), axis=1) / 2
        if areas.sum() <= 0:
            fag.nodes[face_id]["_space_area_samples"] = []
            continue
        # Deterministic area-stratified sampling, not edge/UV-grid point counts.
        count = 256
        indexes = np.searchsorted(np.cumsum(areas) / areas.sum(), (np.arange(count) + .5) / count)
        a = np.sqrt(((np.arange(count) + .5) * .6180339887498949) % 1)
        b = ((np.arange(count) + .5) * .4142135623730951) % 1
        weights = np.column_stack([1 - a, a * (1 - b), a * b])
        points = np.einsum("ni,nij->nj", weights, triplets[indexes])
        adaptor = BRepAdaptor_Surface(faces[face_id], True)
        if adaptor.GetType() == GeomAbs_Cylinder:
            _, geometry = cylinder_face_orientation(faces[face_id])
            axis, origin = normalized(geometry["axis"]), np.asarray(geometry["axis_location"])
            delta = points - origin
            axial = np.outer(delta @ axis, axis)
            radial = delta - axial
            points = origin + axial + radial / np.linalg.norm(radial, axis=1)[:, None] * geometry["radius"]
        fag.nodes[face_id]["_space_area_samples"] = points.tolist()


def feature_local_frame(surface_graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int]) -> np.ndarray:
    recipe = feature["face_constraints"]["feature_space"]["local_crs"]
    if recipe["origin"] != "model_origin":
        raise ValueError("Only the model-origin CRS is supported")
    def direction(spec):
        kind = spec["type"]
        if kind == "plane_normal":
            return normalized(surface_graph.nodes[mapping[int(spec["node"])]]["geometry"]["normal"])
        if kind == "cylinder_axis":
            axis = normalized(surface_graph.nodes[mapping[int(spec["node"])]]["geometry"]["axis"])
        elif kind == "intersection_axis":
            normals = [surface_graph.nodes[mapping[int(i)]]["geometry"]["normal"] for i in spec["nodes"]]
            axis = normalized(np.cross(*normals))
        elif kind == "normal_bisector":
            return normalized(sum(np.asarray(surface_graph.nodes[mapping[int(i)]]["geometry"]["normal"]) for i in spec["nodes"]))
        else:
            raise ValueError("Unknown CRS direction: " + kind)
        axis *= 1 if axis[np.argmax(np.abs(axis))] >= 0 else -1
        if "align_with_normal" in spec:
            normal = surface_graph.nodes[mapping[int(spec["align_with_normal"])]]["geometry"]["normal"]
            if float(axis @ normal) < 0:
                axis = -axis
        return axis
    z = direction(recipe["z"])
    x = np.eye(3)[np.argmin(np.abs(z))] if recipe["x"]["type"] == "stable_perpendicular" else direction(recipe["x"])
    x = normalized(x - (x @ z) * z)
    return np.column_stack([x, normalized(np.cross(z, x)), z])


def feature_space_contexts(surface_graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int]) -> List[Dict[str, Any]]:
    frame = feature_local_frame(surface_graph, feature, mapping)
    spec = feature["face_constraints"]["feature_space"]
    context = {"frame": frame, "opening": None}
    if spec["construction"] != "minkowski_extrusion":
        return [context]
    section = spec["section"]
    ray = section["opening_ray"]
    if ray is None:
        return [context]
    if ray["type"] == "plane_normal":
        context["opening"] = normalized(surface_graph.nodes[mapping[int(ray["node"])]]["geometry"]["normal"])
        return [context]
    # The two fixed U-opening branches are determined by matched supports.
    # Never rotate the opening when the selected topological patches change.
    opening = frame[:, 1]
    return [{"frame": frame, "opening": opening}, {"frame": frame, "opening": -opening}]


def prism_shape(polygon: Sequence[Sequence[float]], vector: Sequence[float]) -> Any:
    builder = BRepBuilderAPI_MakePolygon()
    for point in polygon:
        builder.Add(gp_Pnt(*map(float, point)))
    builder.Close()
    return BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(builder.Wire()).Face(), gp_Vec(*map(float, vector))).Shape()


def frame_transform(frame: np.ndarray) -> Any:
    transform = gp_Trsf()
    transform.SetValues(*[float(value) for row in frame for value in [*row, 0.]])
    return transform


def local_face_bounds(fag: nx.MultiGraph, face_ids: Set[int], frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    key = (tuple(sorted(face_ids)), tuple(frame.round(12).flat))
    cache = fag.graph.setdefault("_space_bbox_cache", {})
    if key not in cache:
        bounds = Bnd_Box()
        transform = frame_transform(frame.T)
        for face_id in face_ids:
            local = BRepBuilderAPI_Transform(fag.graph["_space_faces"][face_id], transform, True).Shape()
            brepbndlib.AddOptimal(local, bounds, False, False)
        if bounds.IsVoid():
            raise ValueError("missing_face_bounds")
        values = bounds.Get()
        cache[key] = (np.array(values[:3]), np.array(values[3:]))
    return cache[key]


def oriented_box(frame: np.ndarray, low: np.ndarray, high: np.ndarray) -> Any:
    polygon = [frame @ np.array([x, y, low[2]]) for x, y in
               [(low[0], low[1]), (high[0], low[1]), (high[0], high[1]), (low[0], high[1])]]
    return prism_shape(polygon, frame[:, 2] * (high[2] - low[2]))


def clip_halfspace(shape: Any, normal: Sequence[float], offset: float) -> Any:
    normal = normalized(normal)
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(*map(float, normal * offset)), gp_Dir(*map(float, normal)))).Face()
    half = BRepPrimAPI_MakeHalfSpace(face, gp_Pnt(*map(float, normal * (offset + 1.)))).Solid()
    return common(shape, half)


def cylinder_collision_shape(surface: Dict[str, Any], corners: np.ndarray, epsilon: float = 0.) -> Any:
    geometry = surface["geometry"]
    axis = normalized(geometry["axis"])
    origin = np.asarray(geometry["axis_location"], dtype=float)
    parameters = (corners - origin) @ axis
    padding = max(LENGTH_TOLERANCE, float(np.ptp(parameters)))
    low, high = float(parameters.min() - padding), float(parameters.max() + padding)
    radius = float(geometry["radius"]) + int(surface["orientation_sign"]) * epsilon
    if radius <= LENGTH_TOLERANCE:
        raise ValueError("Cylinder disappears under inward offset")
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*map(float, origin + low * axis)), gp_Dir(*map(float, axis))), radius, high - low).Shape()


def section_parameters(surface_graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int], context: Dict[str, Any]) -> Dict[str, Any]:
    section = feature["face_constraints"]["feature_space"]["section"]
    frame = context["frame"]
    cylinders = [surface_graph.nodes[mapping[int(i)]] for i in section["generator"]["cylinder_nodes"]]
    if any(int(node["orientation_sign"]) >= 0 for node in cylinders):
        raise ValueError("Minkowski disk boundary requires inward cylinder normals")
    centers = [np.asarray(node["geometry"]["axis_location"]) @ frame for node in cylinders]
    bottom = surface_graph.nodes[mapping[int(section["bottom_node"])]]
    normal = normalized(bottom["geometry"]["normal"])
    if not parallel(normal, frame[:, 2]) or normal @ frame[:, 2] < 0:
        raise ValueError("Minkowski extrusion frame must point along bottom normal")
    return {"a": centers[0][:2], "b": centers[-1][:2],
            "radius": float(surface_graph.nodes[mapping[int(section["disk_radius_from_node"])]]["geometry"]["radius"]),
            "bottom": float(bottom["geometry"]["offset"]),
            "opening": None if context["opening"] is None else (context["opening"] @ frame)[:2]}


def capsule_prism(a: np.ndarray, b: np.ndarray, radius: float, low: float, high: float) -> Any:
    def disk(point):
        return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(float(point[0]), float(point[1]), low), gp_Dir(0, 0, 1)), radius, high - low).Shape()
    result = disk(a)
    if np.linalg.norm(b - a) < LENGTH_TOLERANCE:
        return result
    tangent = normalized(b - a)
    normal = np.array([-tangent[1], tangent[0]]) * radius
    rectangle = prism_shape([[*p, low] for p in (a + normal, b + normal, b - normal, a - normal)], [0, 0, high - low])
    result = boolean_shape(BRepAlgoAPI_Fuse, result, rectangle)
    return boolean_shape(BRepAlgoAPI_Fuse, result, disk(b))


def trimmed_patch_prism_shape(
    surface_graph: nx.Graph,
    feature: Dict[str, Any],
    mapping: Dict[int, int],
    context: Dict[str, Any],
    low: np.ndarray,
    high: np.ndarray,
    epsilon: float = 0.,
) -> Any:
    """Exact filled Minkowski section from supports; no selected-patch hull."""
    parameters = section_parameters(surface_graph, feature, mapping, context)
    a, b, radius = parameters["a"], parameters["b"], parameters["radius"] - epsilon
    if radius <= LENGTH_TOLERANCE:
        raise ValueError("Minkowski section disappears under inward offset")
    z0, z1 = float(low[2] - 1), float(high[2] + 1)
    ray = parameters["opening"]
    if ray is None:
        cell = capsule_prism(a, b, radius, z0, z1)
    else:
        reach = 4 * (np.linalg.norm(high - low) + np.linalg.norm(a - low[:2]) + np.linalg.norm(b - high[:2]) + radius + 1)
        v = normalized(ray) * reach
        if np.linalg.norm(a - b) < LENGTH_TOLERANCE:
            cell = capsule_prism(a, a + v, radius, z0, z1)
        else:
            corners = [a, b, b + v, a + v]
            cell = prism_shape([[*p, z0] for p in corners], [0, 0, z1 - z0])
            for p, q in zip(corners, corners[1:] + corners[:1]):
                cell = boolean_shape(BRepAlgoAPI_Fuse, cell, capsule_prism(p, q, radius, z0, z1))
    cell = clip_halfspace(cell, [0, 0, 1], parameters["bottom"] + epsilon)
    return BRepBuilderAPI_Transform(cell, frame_transform(context["frame"]), True).Shape()


def finite_collision_domain(
    fag: nx.MultiGraph,
    surface_graph: nx.Graph,
    feature: Dict[str, Any],
    mapping: Dict[int, int],
    face_ids: Set[int],
    context: Dict[str, Any],
    epsilon: float = 0.,
    extension_ratio: Optional[float] = None,
) -> Tuple[Any, Dict[str, Any]]:
    frame = context["frame"]
    low, high = local_face_bounds(fag, face_ids, frame)
    extent = high - low
    if np.any(extent <= LENGTH_TOLERANCE):
        raise ValueError("degenerate_envelope")
    ratio = float(feature.get("_matching_policy", {}).get("collision_policy", {}).get("extension_ratio", .1)) if extension_ratio is None else float(extension_ratio)
    elo, ehi = low - ratio * extent, high + ratio * extent
    cell = oriented_box(frame, elo, ehi)
    spec = feature["face_constraints"]["feature_space"]
    if spec["construction"] == "minkowski_extrusion":
        cell = common(cell, trimmed_patch_prism_shape(surface_graph, feature, mapping, context, elo, ehi, epsilon))
    elif spec["construction"] == "implicit_intersection":
        corners = np.array([frame @ np.array(p) for p in itertools.product(*zip(elo, ehi))])
        for role in spec.get("boundary_nodes", mapping):
            surface = surface_graph.nodes[mapping[int(role)]]
            if surface["surface_type"] == "plane":
                cell = clip_halfspace(cell, surface["geometry"]["normal"], surface["geometry"]["offset"] + epsilon)
            else:
                cylinder = cylinder_collision_shape(surface, corners, epsilon)
                operation = BRepAlgoAPI_Common if int(surface["orientation_sign"]) < 0 else BRepAlgoAPI_Cut
                cell = boolean_shape(operation, cell, cylinder)
    else:
        raise ValueError("Unknown feature-space construction")
    return cell, {"method": spec["construction"], "local_crs": {"origin": [0., 0., 0.], "axes_columns": frame.tolist()},
                  "local_bbox": [*low.tolist(), *high.tolist()], "expanded_local_bbox": [*elo.tolist(), *ehi.tolist()],
                  "extension_ratio": ratio, "epsilon": epsilon,
                  "opening_direction": None if context["opening"] is None else context["opening"].tolist()}


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


def collision_free(part_shape: Any, fag: nx.MultiGraph, surface_graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int], face_ids: Set[int], capture_mesh: bool = False, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
    context = context or feature_space_contexts(surface_graph, feature, mapping)[0]
    try:
        low, high = local_face_bounds(fag, face_ids, context["frame"])
        epsilon = max(LENGTH_TOLERANCE, float(np.linalg.norm(high - low)) * 1.0e-6)
        cell, record = finite_collision_domain(fag, surface_graph, feature, mapping, face_ids, context, epsilon)
        cell_volume = shape_volume(cell)
        if cell_volume <= VOLUME_MINIMUM_TOLERANCE:
            return False, {"status": "empty_feature_cell", **record}
        collision_part_shape = copy_shape(fag.graph["_space_pristine_part_shape"])
        common_collision_volume = shape_volume(common(copy_shape(cell), collision_part_shape))
        remaining_cell = boolean_shape(BRepAlgoAPI_Cut, copy_shape(cell), copy_shape(fag.graph["_space_pristine_part_shape"]))
        difference_collision_volume = min(cell_volume, max(0.0, cell_volume - shape_volume(remaining_cell)))
        collision_volume = max(common_collision_volume, difference_collision_volume)
        part_volume = fag.graph["_space_part_volume"]
        tolerance = max(VOLUME_MINIMUM_TOLERANCE, part_volume * VOLUME_RELATIVE_TOLERANCE)
        record.update(status="collision_free" if collision_volume <= tolerance else "collision_detected",
                      collision_volume=collision_volume, common_collision_volume=common_collision_volume,
                      difference_collision_volume=difference_collision_volume, volume_tolerance=tolerance)
        if capture_mesh:
            # Display the geometric domain, not the numerically eroded solid.
            exact, _ = finite_collision_domain(fag, surface_graph, feature, mapping, face_ids, context)
            record["mesh"] = collision_mesh(exact)
            record["mesh_domain"] = "unoffset_finite_collision_domain"
        return collision_volume <= tolerance, record
    except (RuntimeError, ValueError) as error:
        return False, {"status": "cell_build_failed", "error": str(error)}


def minimum_covering_arc(intervals: Sequence[Dict[str, float]]) -> float:
    period = 2 * math.pi
    pieces = []
    for interval in intervals:
        length = min(period, float(interval["length"]))
        if length >= period - 1.0e-9:
            return period
        if length <= 0:
            continue
        start = float(interval["start"]) % period
        end = start + length
        pieces.append((start, min(end, period)))
        if end > period:
            pieces.append((0., end - period))
    if not pieces:
        return 0.
    merged = []
    for start, end in sorted(pieces):
        if merged and start <= merged[-1][1] + 1.0e-10:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    gaps.append(merged[0][0] + period - merged[-1][1])
    return period - max(gaps)


def trim_constraints_record(feature: Dict[str, Any], surface_graph: nx.Graph, mapping: Dict[int, int], role_faces: Dict[int, Set[int]]) -> Dict[str, Any]:
    checks = []
    for constraint in feature.get("face_constraints", {}).get("trim_constraints", []):
        role = int(constraint["node"])
        surface = surface_graph.nodes[mapping[role]]
        if constraint["quantity"] != "cylindrical_minimum_covering_arc":
            raise ValueError("Unsupported trim quantity: " + constraint["quantity"])
        intervals = [surface["face_angular_intervals"][str(face)] for face in sorted(role_faces.get(role, set()))]
        span = minimum_covering_arc(intervals)
        limit, operator = float(constraint["radians"]), constraint["operator"]
        tolerance = 1.0e-9
        accepted = span > tolerance and {"<": span < limit - tolerance, "<=": span <= limit + tolerance,
                                         ">": span > limit + tolerance, ">=": span >= limit - tolerance}[operator]
        checks.append({"role": role, "face_ids": sorted(role_faces.get(role, set())), "intervals": intervals,
                       "span_radians": span, "span_degrees": math.degrees(span), "operator": operator,
                       "limit_radians": limit, "passed": bool(accepted)})
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def angular_constraint_satisfied(feature: Dict[str, Any], surface_graph: nx.Graph, mapping: Dict[int, int], role_faces: Optional[Dict[int, Set[int]]] = None) -> bool:
    if role_faces is None:
        raise ValueError("Trim must be checked on selected topological faces, not whole surface nodes")
    return trim_constraints_record(feature, surface_graph, mapping, role_faces)["passed"]


def surface_intersection(first: Any, second: Any, tolerance: float) -> Any:
    operation = BRepAlgoAPI_Common(copy_shape(first), copy_shape(second))
    operation.SetNonDestructive(True)
    operation.SetFuzzyValue(tolerance)
    operation.Build()
    if not operation.IsDone() or operation.Shape().IsNull():
        raise RuntimeError("Effective-boundary surface intersection failed")
    return operation.Shape()


def effective_boundary_faces(fag: nx.MultiGraph, surface_graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int], face_ids: Set[int], context: Dict[str, Any]) -> List[Tuple[Any, Any]]:
    """E(T) = boundary(K) intersect the original, unexpanded local bbox.

    Build K in a strictly larger guard box. Its artificial caps are outside
    B(T), so intersecting each boundary face with B(T) removes them without
    confusing a genuine seed face coincident with a side of B(T) with a cap.
    """
    frame = context["frame"]
    low, high = local_face_bounds(fag, face_ids, frame)
    tolerance = max(LENGTH_TOLERANCE, fag.graph["_space_length_scale"] * 1.0e-7)
    extent = high - low
    if np.any(extent <= LENGTH_TOLERANCE):
        raise ValueError("degenerate_envelope")
    opening = None if context["opening"] is None else tuple(context["opening"].round(12))
    key = (feature["name"], tuple(sorted(mapping.items())), tuple(sorted(face_ids)), opening)
    cache = surface_graph.graph.setdefault("_space_effective_boundaries", {})
    if key not in cache:
        # Keep caps farther away than the Boolean tolerance, including thin boxes.
        guard_ratio = max(.1, 100 * tolerance / float(extent.min()))
        guard_cell, _ = finite_collision_domain(fag, surface_graph, feature, mapping, face_ids, context, extension_ratio=guard_ratio)
        original_box = oriented_box(frame, low, high)
        boundaries = []
        for face in legacy.unique_shapes(legacy.explore_shapes(guard_cell, legacy.TopAbs_FACE)):
            clipped = surface_intersection(face, original_box, tolerance)
            if face_area(clipped) > tolerance ** 2:
                boundaries.append((clipped, face))
        cache[key] = boundaries
    return cache[key]


def boundary_orientation_matches(surface: Dict[str, Any], boundary_face: Any) -> bool:
    # The outward normal of the feature-space solid points opposite to the
    # topological solid normal, which points into the feature space.
    adaptor = BRepAdaptor_Surface(boundary_face, True)
    if surface["surface_type"] == "plane" and adaptor.GetType() == GeomAbs_Plane:
        _, geometry = plane_face_orientation(boundary_face)
        return float(np.dot(normalized(surface["geometry"]["normal"]), normalized(geometry["normal"]))) <= -math.cos(math.radians(ANGLE_TOLERANCE_DEGREES))
    if surface["surface_type"] == "cylinder" and adaptor.GetType() == GeomAbs_Cylinder:
        sign, _ = cylinder_face_orientation(boundary_face)
        return sign == -int(surface["orientation_sign"])
    return False


def active_boundary_record(fag: nx.MultiGraph, surface_graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int], role_faces: Dict[int, Set[int]], context: Dict[str, Any]) -> Dict[str, Any]:
    policy = feature.get("_matching_policy", {}).get("face_active_boundary", {})
    threshold = float(policy.get("minimum_area_fraction", policy.get("sampled_minimum_fraction", ACTIVE_PATCH_SAMPLE_FRACTION)))
    face_ids = set().union(*role_faces.values())
    tolerance = max(LENGTH_TOLERANCE, fag.graph["_space_length_scale"] * 1.0e-7)
    low, high = local_face_bounds(fag, face_ids, context["frame"])
    boundaries = effective_boundary_faces(fag, surface_graph, feature, mapping, face_ids, context)
    checks = []
    for role, ids in role_faces.items():
        surface = surface_graph.nodes[mapping[role]]
        role_boundaries = [shape for shape, source_face in boundaries if boundary_orientation_matches(surface, source_face)]
        for face_id in sorted(ids):
            face = fag.graph["_space_faces"][face_id]
            total_area = face_area(face)
            # Boolean split boundary faces have disjoint interiors. Summation
            # therefore measures f intersect E(T), without counting bbox caps.
            overlap_area = sum(face_area(surface_intersection(face, boundary, tolerance)) for boundary in role_boundaries)
            fraction = min(1., max(0., overlap_area / total_area)) if total_area > tolerance ** 2 else 0.
            check = {"role": role, "face_id": face_id, "face_area": total_area,
                     "active_area": min(total_area, overlap_area), "active_fraction": fraction,
                     "required_fraction": threshold, "active_percentage": 100. * fraction,
                     "active_percentage_text": f"{100. * fraction:.6f}%",
                     "passed": bool(total_area > tolerance ** 2 and overlap_area > tolerance ** 2 and fraction >= threshold)}
            properties = GProp_GProps()
            brepgprop.SurfaceProperties(face, properties)
            check["center"] = list(properties.CentreOfMass().Coord())
            checks.append(check)
    return {"passed": all(check["passed"] for check in checks), "faces": checks,
            "initial_local_bbox": [*low.tolist(), *high.tolist()],
            "effective_boundary": "boundary(K) intersect unexpanded B(T); artificial box caps excluded",
            "metric": "boolean_surface_overlap_area_ratio"}


def filter_face_combination(part_shape: Any, fag: nx.MultiGraph, surface_graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int], role_faces: Dict[int, Set[int]], context: Dict[str, Any]) -> Tuple[Optional[Set[int]], Dict[int, Set[int]], Dict[str, Any]]:
    trial = set().union(*role_faces.values())
    valid, collision = collision_free(part_shape, fag, surface_graph, feature, mapping, trial, context=context)
    if not valid:
        record_candidate_stage(fag, feature, trial, collision["status"])
        return None, {}, collision
    try:
        boundary = active_boundary_record(fag, surface_graph, feature, mapping, role_faces, context)
    except (RuntimeError, ValueError) as error:
        return None, {}, {"status": "active_boundary_build_failed", "error": str(error), "initial_collision": collision}
    removed = {check["face_id"] for check in boundary["faces"] if not check["passed"]}
    retained = {role: set(ids) - removed for role, ids in role_faces.items()}
    history = {"initial_face_ids": sorted(trial), "removed_face_ids": sorted(removed),
               "initial_collision": collision, "boundary_check": boundary}
    if any(not retained.get(role) for role in mapping):
        record_candidate_stage(fag, feature, trial, "active_boundary_required_role_lost")
        return None, retained, {"status": "active_boundary_required_role_lost", "boundary_filter": history}
    selected = trial - removed
    if removed:
        record_candidate_stage(fag, feature, trial, "active_boundary_faces_removed")
        # K and its frame remain fixed; only the final local envelope changes.
        valid, collision = collision_free(part_shape, fag, surface_graph, feature, mapping, selected, context=context)
        if not valid:
            return None, retained, {**collision, "boundary_filter": history}
    if not angular_constraint_satisfied(feature, surface_graph, mapping, retained):
        record_candidate_stage(fag, feature, selected, "trim_rejected_after_boundary_filter")
        return None, retained, {"status": "trim_rejected", "boundary_filter": history}
    record_candidate_stage(fag, feature, selected, "checks_passed")
    return selected, retained, {**collision, "boundary_filter": history}


def selected_patches_lie_on_active_cell(
    fag: nx.MultiGraph,
    surface_graph: nx.Graph,
    mapping: Dict[int, int],
    role_faces: Dict[int, Set[int]],
    feature: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    context = context or feature_space_contexts(surface_graph, feature, mapping)[0]
    return active_boundary_record(fag, surface_graph, feature, mapping, role_faces, context)["passed"]


def record_candidate_stage(fag: nx.MultiGraph, feature: Dict[str, Any], faces: Set[int], stage: str) -> None:
    key = tuple(sorted(faces))
    # Ground truth controls logging only. It never controls a recognition branch.
    if key in fag.graph.get("_space_gt_face_sets", set()):
        counts = fag.graph.setdefault("_space_gt_attempts", {}).setdefault(key, {})
        label = feature["name"] + ":" + stage
        counts[label] = counts.get(label, 0) + 1


def choose_faces(part_shape: Any, fag: nx.MultiGraph, surface_graph: nx.Graph, feature: Dict[str, Any], mapping: Dict[int, int], context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Set[int]], Dict[int, Set[int]], Dict[str, Any]]:
    context = context or feature_space_contexts(surface_graph, feature, mapping)[0]
    role_candidates = {seed_id: sorted(surface_graph.nodes[model_id]["face_ids"], key=lambda face_id: -surface_graph.nodes[model_id]["face_areas"].get(str(face_id), 0.0)) for seed_id, model_id in mapping.items()}
    if any(not values for values in role_candidates.values()):
        return None, {}, {"status": "empty_role"}
    seed_order = sorted(role_candidates)
    role_options = {}
    for role, candidates in role_candidates.items():
        lower = next((rule for rule in feature["face_constraints"].get("trim_constraints", []) if int(rule["node"]) == role and rule["operator"] in (">", ">=")), None)
        if lower is None:
            role_options[role] = [{face} for face in candidates]
        else:
            # A through/blind hole may need several fragments to exceed pi.
            # Do not reject its individual fragments before forming the union.
            options = []
            surface = surface_graph.nodes[mapping[role]]
            examined = 0
            for size in range(1, len(candidates) + 1):
                for values in itertools.combinations(candidates, size):
                    examined += 1
                    span = minimum_covering_arc([surface["face_angular_intervals"][str(face)] for face in values])
                    if span > float(lower["radians"]) + 1.0e-9 or (lower["operator"] == ">=" and span >= float(lower["radians"]) - 1.0e-9):
                        if not any(option.issubset(values) for option in options):
                            options.append(set(values))
                    if examined >= MAX_MATCHES_PER_FEATURE:
                        break
                if examined >= MAX_MATCHES_PER_FEATURE:
                    break
            role_options[role] = options
    best: Optional[Tuple[float, Set[int], Dict[str, Any]]] = None
    combination_count = 0
    active_rejected = 0
    collision_rejected = 0
    last_collision = None
    for combination in itertools.product(*(role_options[seed_id] for seed_id in seed_order)):
        if sum(len(group) for group in combination) != len(set().union(*combination)):
            continue
        combination_count += 1
        if combination_count > MAX_MATCHES_PER_FEATURE:
            break
        trial = set().union(*combination)
        trial_roles = {seed_id: group for seed_id, group in zip(seed_order, combination)}
        filtered, _, trial_record = filter_face_combination(part_shape, fag, surface_graph, feature, mapping, trial_roles, context)
        if filtered is None:
            if trial_record["status"].startswith("active_boundary"):
                active_rejected += 1
            elif trial_record["status"] == "collision_detected":
                collision_rejected += 1
                last_collision = trial_record
            continue
        bounds = face_bounds(fag, filtered)
        if bounds is None:
            continue
        diagonal = math.sqrt(sum((bounds[index + 3] - bounds[index]) ** 2 for index in range(3)))
        if best is None or diagonal < best[0]:
            best = (diagonal, filtered, trial_record)
    if best is None:
        return None, {}, {"status": "no_spatially_compatible_basic_face_combination", "tested_combinations": combination_count, "active_boundary_rejections": active_rejected, "collision_rejections": collision_rejected, "last_collision": last_collision}
    _, selected, record = best
    for face_id in [value for values in role_candidates.values() for value in values if value not in selected]:
        trial = set(selected)
        trial.add(face_id)
        trial_roles = {seed_id: {candidate for candidate in trial if candidate in candidates} for seed_id, candidates in role_candidates.items()}
        filtered, _, trial_record = filter_face_combination(part_shape, fag, surface_graph, feature, mapping, trial_roles, context)
        if filtered is not None:
            selected = filtered
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
    prepare_face_geometry(part_shape, faces, fag)
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
            surface_matches = find_seed_matches(surface_graph, feature)
            matches = []
            for mapping in surface_matches:
                try:
                    contexts = feature_space_contexts(surface_graph, feature, mapping)
                    for context in contexts:
                        if feature["face_constraints"]["feature_space"]["construction"] == "minkowski_extrusion":
                            section_parameters(surface_graph, feature, mapping, context)
                        matches.append((mapping, context))
                except ValueError:
                    continue
            accepted_one = False
            for mapping, context in matches:
                selected, role_groups, collision = choose_faces(part_shape, fag, surface_graph, feature, mapping, context)
                if not selected or not selected.issubset(available):
                    continue
                role_names = {int(node["id"]): node["role"] for node in feature["nodes"]}
                boundary_filter = collision.get("boundary_filter")
                _, collision = collision_free(part_shape, fag, surface_graph, feature, mapping, selected, capture_mesh=True, context=context)
                if boundary_filter is not None:
                    collision["boundary_filter"] = boundary_filter
                record_candidate_stage(fag, feature, selected, "accepted")
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
                    "active_boundary": active_boundary_record(fag, surface_graph, feature, mapping, role_groups, context),
                    "trim": trim_constraints_record(feature, surface_graph, mapping, role_groups),
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
        print(f"[SPACE] {feature['name']}: total_instances={len(instances)} remaining_faces={len(available)}", flush=True)
    final_sag = build_oriented_surface_graph(faces, available)
    return instances, predictions, instance_ids, {"initial": snapshots[0] if snapshots else serialize_sag(final_sag), "remaining": serialize_sag(final_sag)}


def ground_truth_groups(ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups = {}
    for face_id, category in ground_truth["seg"].items():
        instance_id = ground_truth["instance_ids"].get(face_id)
        if instance_id is None or category == legacy.STOCK_CATEGORY_ID:
            continue
        category = 8 if category in (10, 20) else int(category)
        group = groups.setdefault((int(instance_id), category), {"gt_instance_id": int(instance_id), "category_id": category, "face_ids": []})
        group["face_ids"].append(int(face_id))
    return [dict(group, face_ids=sorted(group["face_ids"])) for group in groups.values()]


def diagnose_ground_truth(part_shape: Any, faces: Sequence[Any], fag: nx.MultiGraph, features: List[Dict[str, Any]], ground_truth: Dict[str, Any], instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """GT-conditioned replay is diagnostic only, after label-blind extraction."""
    graph = build_oriented_surface_graph(faces, set(range(len(faces))))
    by_category = {int(feature["category_id"]): feature for feature in features}
    reports = []
    for group in ground_truth_groups(ground_truth):
        ids = set(group["face_ids"])
        category = group["category_id"]
        feature = by_category.get(category)
        report = {**group, "category_name": legacy.FACE_CATEGORIES[category],
                  "finite_collision_domain": None, "active_boundary": None,
                  "actual_exact_face_set_attempts": fag.graph.get("_space_gt_attempts", {}).get(tuple(sorted(ids)), {}),
                  "diagnostic_basis": "GT-only replay on initial SAG; never used to accept or reject recognition candidates"}
        recovered = next((instance for instance in instances if int(instance["category_id"]) == category and set(instance["face_ids"]) == ids), None)
        if recovered:
            report.update(stage="extracted", predicted_instance_id=recovered["instance_id"],
                          finite_collision_domain=recovered["collision"], active_boundary=recovered["active_boundary"],
                          trim=recovered["trim"], domain_available=True)
            reports.append(report)
            continue
        if feature is None:
            report.update(stage="excluded_category", domain_available=False, domain_unavailable_reason="No seed for this GT category")
            reports.append(report)
            continue
        model_nodes = [node for node, attrs in graph.nodes(data=True) if ids.intersection(attrs["face_ids"])]
        roles = [int(node["id"]) for node in feature["nodes"]]
        role_types = {int(node["id"]): node["surface_type"] for node in feature["nodes"]}
        unsupported = ids - set().union(*(set(graph.nodes[node]["face_ids"]) for node in model_nodes)) if model_nodes else ids
        if len(model_nodes) != len(roles) or unsupported:
            report.update(stage="sag_matching", domain_available=False,
                          domain_unavailable_reason="GT faces do not provide exactly the required distinct oriented supports",
                          required_support_count=len(roles), actual_support_count=len(model_nodes), unsupported_faces=sorted(unsupported))
            reports.append(report)
            continue
        best = None
        best_mismatches = None
        group_passed = False
        for permutation in itertools.permutations(model_nodes):
            mapping = dict(zip(roles, permutation))
            if any(graph.nodes[mapping[role]]["surface_type"] != role_types[role] for role in roles):
                continue
            mismatches = []
            for edge in feature.get("edges", []):
                a, b = [mapping[int(value)] for value in edge["nodes"]]
                actual = graph.edges[a, b]["code"] if graph.has_edge(a, b) else None
                if actual != edge["code"]:
                    mismatches.append({"roles": edge["nodes"], "expected": edge["code"], "actual": actual})
            if best_mismatches is None or len(mismatches) < len(best_mismatches):
                best_mismatches = mismatches
            if mismatches:
                continue
            if not surface_constraints_satisfied(graph, feature, mapping):
                if best is None:
                    best = (0, "surface_constraints", mapping, None, None)
                continue
            group_passed = True
            role_faces = {role: ids.intersection(graph.nodes[node]["face_ids"]) for role, node in mapping.items()}
            trim = trim_constraints_record(feature, graph, mapping, role_faces)
            try:
                contexts = feature_space_contexts(graph, feature, mapping)
            except ValueError as error:
                if best is None or best[0] < 1:
                    best = (1, "feature_space_construction", mapping, None, {"error": str(error)})
                continue
            for context in contexts:
                try:
                    passed, collision = collision_free(part_shape, fag, graph, feature, mapping, ids, context=context)
                    removed = []
                    lost_roles = []
                    if not passed:
                        score, stage = 2, "collision" if collision["status"] == "collision_detected" else "feature_space_construction"
                    else:
                        active = active_boundary_record(fag, graph, feature, mapping, role_faces, context)
                        removed = [check["face_id"] for check in active["faces"] if not check["passed"]]
                        retained = {role: members - set(removed) for role, members in role_faces.items()}
                        lost_roles = [role for role, members in retained.items() if not members]
                        trim = trim_constraints_record(feature, graph, mapping, retained)
                        if removed:
                            score, stage = 3, "active_boundary"
                        elif not trim["passed"]:
                            score, stage = 4, "trim"
                        else:
                            score, stage = 5, "passes_gt_replay"
                    if best is None or score > best[0]:
                        best = (score, stage, mapping, context, {"trim": trim, "collision": collision,
                                "removed_face_ids": removed, "lost_surface_roles": lost_roles})
                except (RuntimeError, ValueError) as error:
                    if best is None or best[0] < 1:
                        best = (1, "feature_space_construction", mapping, context, {"error": str(error)})
                if best and best[0] == 5:
                    break
            if best and best[0] == 5:
                break
        if best is None:
            report.update(stage="sag_matching", edge_mismatches=best_mismatches,
                          domain_available=False, domain_unavailable_reason="No valid SAG role mapping; no feature cell can be uniquely defined")
            reports.append(report)
            continue
        _, stage, mapping, context, details = best
        report.update(stage=stage, surface_mapping={str(role): node for role, node in mapping.items()},
                      surface_constraints_passed=group_passed)
        if details:
            report.update(details)
        if context is not None:
            role_faces = {role: ids.intersection(graph.nodes[node]["face_ids"]) for role, node in mapping.items()}
            try:
                # Replay after candidate selection; never influences extraction or ranking.
                report["active_boundary"] = active_boundary_record(fag, graph, feature, mapping, role_faces, context)
                report["active_boundary"].update(diagnostic_only=True, used_for_recognition=False,
                    evaluated_after_collision_failure=(report.get("collision") or {}).get("status") == "collision_detected")
                cell, record = finite_collision_domain(fag, graph, feature, mapping, ids, context)
                record.update(mesh=collision_mesh(cell), mesh_domain="unoffset_finite_collision_domain",
                              diagnostic_only=True, status=(report.get("collision") or {}).get("status", "not_reached_collision_stage"))
                if report.get("collision"):
                    record.update({key: value for key, value in report["collision"].items() if key not in ("mesh", "epsilon")})
                report["finite_collision_domain"] = record
                report["domain_available"] = True
            except (RuntimeError, ValueError) as error:
                report.update(domain_available=False, domain_unavailable_reason=str(error))
        else:
            report.update(domain_available=False, domain_unavailable_reason="No valid feature-space frame")
        if stage == "passes_gt_replay":
            occupying = [instance["instance_id"] for instance in instances if ids.intersection(instance["face_ids"])]
            report["stage"] = "ownership_or_different_face_set" if occupying else "face_combination_search"
            report["overlapping_predicted_instances"] = occupying
            report["explanation"] = "The full GT set passes replay; recognition selected a different set or did not reach this combination. Not a collision/boundary rejection."
        reports.append(report)
        print(f"[GT] instance={group['gt_instance_id']} {feature['name']} stage={report['stage']}", flush=True)
    for report in reports:
        boundary = report.get("active_boundary")
        if boundary is None:
            boundary = {"passed": None, "faces": [], "diagnostic_only": True,
                        "unavailable_reason": report.get("domain_unavailable_reason", "No valid effective boundary")}
            report["active_boundary"] = boundary
        indexed = {int(check["face_id"]): check for check in boundary["faces"]}
        for face_id in report["face_ids"]:
            if face_id not in indexed:
                properties = GProp_GProps()
                brepgprop.SurfaceProperties(faces[face_id], properties)
                boundary["faces"].append({"face_id": face_id, "center": list(properties.CentreOfMass().Coord()),
                    "active_fraction": None, "active_percentage": None, "active_percentage_text": "N/A",
                    "required_fraction": ACTIVE_PATCH_SAMPLE_FRACTION, "passed": None,
                    "unavailable_reason": boundary.get("unavailable_reason", "No valid effective boundary")})
    return reports


def topological_edge_mesh(part_shape: Any, length_scale: float) -> List[Dict[str, Any]]:
    """Cache actual BRep edges, including seams, not triangle diagonals."""
    result = []
    for edge_id, edge in enumerate(legacy.unique_shapes(legacy.explore_shapes(part_shape, TopAbs_EDGE))):
        curve = BRepAdaptor_Curve(edge)
        first, last = curve.FirstParameter(), curve.LastParameter()
        if not math.isfinite(first) or not math.isfinite(last) or last <= first:
            continue
        sampler = GCPnts_QuasiUniformDeflection(curve, max(length_scale * 1.e-4, 1.e-5))
        points = ([list(sampler.Value(i).Coord()) for i in range(1, sampler.NbPoints() + 1)]
                  if sampler.IsDone() and sampler.NbPoints() >= 2 else
                  [list(curve.Value(float(t)).Coord()) for t in np.linspace(first, last, 129)])
        result.append({"edge_id": edge_id, "points": points})
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one STEP file with oriented surface-space seeds.")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--step", type=Path)
    parser.add_argument("--label", type=Path)
    parser.add_argument("--seeds", type=Path, default=SEED_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_space_cache(step_path: Path, label_path: Path, seed_path: Path) -> Dict[str, Any]:
    """Shared single/batch pipeline, including post-extraction GT diagnostics."""
    legacy.STEP_PATH = step_path
    legacy.GROUND_TRUTH_JSON_PATH = label_path
    legacy.FEATURE_SEED_JSON_PATH = seed_path.resolve()
    shape = legacy.load_step_shape()
    ground_truth = legacy.load_ground_truth()
    features = load_seeds(seed_path.resolve())
    fag, faces, fag_statistics = legacy.build_attributed_fag(shape)
    fag.graph["_space_gt_face_sets"] = {tuple(group["face_ids"]) for group in ground_truth_groups(ground_truth)}
    instances, predicted_seg, predicted_instance_ids, sag = extract(shape, faces, fag, features)
    diagnostics = diagnose_ground_truth(shape, faces, fag, features, ground_truth, instances)
    cache = legacy.build_cache(shape, faces, fag, fag_statistics, ground_truth, instances, predicted_seg, predicted_instance_ids)
    cache["extraction_mode"] = "space"
    cache["surface_attributed_graph"] = sag
    cache["feature_seed_json_path"] = str(seed_path.resolve())
    cache["topological_edges"] = topological_edge_mesh(shape, fag.graph["_space_length_scale"])
    cache["space_diagnostics"] = {"schema_version": 2, "ground_truth_used_for_recognition": False, "gt_instances": diagnostics}
    cache["evaluation"] = calculate_evaluation(faces=cache["faces"], instances=instances, category_names=legacy.FACE_CATEGORIES, target_category_ids=[int(feature["category_id"]) for feature in features] + [legacy.STOCK_CATEGORY_ID])
    return cache


def main() -> None:
    args = arguments()
    sample = str(args.sample)
    step_path = (args.step or INPUT_ROOT / sample / f"{sample}.step").resolve()
    label_path = (args.label or INPUT_ROOT / sample / f"{sample}.json").resolve()
    output = (args.output or OUTPUT_ROOT / sample).resolve()
    if args.step is None and args.label is None and (not step_path.exists() or not label_path.exists()):
        input_directory = (INPUT_ROOT / sample).resolve()
        if input_directory.exists():
            if any(input_directory.iterdir()):
                raise FileNotFoundError(f"Sample folder exists but is missing its STEP or label file: {input_directory}")
            input_directory.rmdir()
        destination = copy_sample(
            dataset_root=DATASET_ROOT,
            sample_name=sample,
            destination_root=INPUT_ROOT,
        )
        print(f"Sample copied to: {destination}")
    output.mkdir(parents=True, exist_ok=True)
    cache = build_space_cache(step_path, label_path, args.seeds)
    sag = cache["surface_attributed_graph"]
    with (output / "cache.json").open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)
    with (output / "surface_attributed_graph.json").open("w", encoding="utf-8") as file:
        json.dump(sag, file, ensure_ascii=False, indent=2)
    with (output / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(cache["evaluation"], file, ensure_ascii=False, indent=2)
    with (output / "diagnostic_report.json").open("w", encoding="utf-8") as file:
        json.dump(cache["space_diagnostics"], file, ensure_ascii=False, indent=2)
    print(f"[DONE] sample={sample} instances={len(cache['instances'])}")
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
