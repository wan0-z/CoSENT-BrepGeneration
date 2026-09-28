# build_coag_cache.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict


# ============================================================
# 默认路径
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = (APP_DIR.parent / "Generation_Viewer" / "data").resolve()
STEPS_DIR = ROOT_DIR / "steps"
LABELS_DIR = ROOT_DIR / "labels"
OUT_DIR = APP_DIR / "coag_cache"

MESH_LINEAR_DEFLECTION = 0.08
MESH_ANGULAR_DEFLECTION = 0.5

STOCK_LABEL_ID = 24

FACE_CATEGORIES = [
    'chamfer',                      # 0
    'through_hole',                 # 1
    'triangular_passage',           # 2
    'rectangular_passage',          # 3
    '6sides_passage',               # 4
    'triangular_through_slot',      # 5
    'rectangular_through_slot',     # 6
    'circular_through_slot',        # 7
    'rectangular_through_step',     # 8
    '2sides_through_step',          # 9
    'slanted_through_step',         # 10
    'Oring',                        # 11
    'blind_hole',                   # 12
    'triangular_pocket',            # 13
    'rectangular_pocket',           # 14
    '6sides_pocket',                # 15
    'circular_end_pocket',          # 16
    'rectangular_blind_slot',       # 17
    'v_circular_end_blind_slot',    # 18
    'h_circular_end_blind_slot',    # 19
    'triangular_blind_step',        # 20
    'circular_blind_step',          # 21
    'rectangular_blind_step',       # 22
    'round',                        # 23
    'stock'                         # 24
]

RELATION_ORDER = [
    "mate",
    "next",
    "previous",
    "cocurve",
    "cosurface",
    "coface",
]


# ============================================================
# OCC imports
# ============================================================

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopAbs import (
        TopAbs_FACE,
        TopAbs_WIRE,
        TopAbs_EDGE,
        TopAbs_FORWARD,
        TopAbs_REVERSED,
    )
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepTools import breptools
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane,
        GeomAbs_Cylinder,
        GeomAbs_Cone,
        GeomAbs_Sphere,
        GeomAbs_Torus,
        GeomAbs_BezierSurface,
        GeomAbs_BSplineSurface,
        GeomAbs_SurfaceOfRevolution,
        GeomAbs_SurfaceOfExtrusion,
        GeomAbs_OffsetSurface,
        GeomAbs_OtherSurface,
        GeomAbs_Line,
        GeomAbs_Circle,
        GeomAbs_Ellipse,
        GeomAbs_Hyperbola,
        GeomAbs_Parabola,
        GeomAbs_BezierCurve,
        GeomAbs_BSplineCurve,
        GeomAbs_OffsetCurve,
        GeomAbs_OtherCurve,
    )
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.gp import gp_Pnt, gp_Dir
except Exception as e:
    raise ImportError(
        "没有成功导入 pythonOCC。请确认你的环境里可以 import OCC.Core。"
    ) from e


# ============================================================
# 基础工具
# ============================================================

def safe_name(label_id: Optional[int]) -> Optional[str]:
    if label_id is None:
        return None
    if 0 <= label_id < len(FACE_CATEGORIES):
        return FACE_CATEGORIES[label_id]
    return f"unknown_{label_id}"


def shape_key(shape: Any) -> int:
    """
    用 HashCode 做单次运行内的拓扑实体 key。
    """
    try:
        return int(shape.HashCode(2_147_483_647))
    except Exception:
        return id(shape)


def round_float(x: float, ndigits: int = 6) -> float:
    try:
        return round(float(x), ndigits)
    except Exception:
        return 0.0


def pnt_tuple(p: Any, ndigits: int = 6) -> Tuple[float, float, float]:
    return (
        round_float(p.X(), ndigits),
        round_float(p.Y(), ndigits),
        round_float(p.Z(), ndigits),
    )


def dir_tuple(d: Any, ndigits: int = 6) -> Tuple[float, float, float]:
    return (
        round_float(d.X(), ndigits),
        round_float(d.Y(), ndigits),
        round_float(d.Z(), ndigits),
    )


def read_step_shape(step_path: Path) -> Any:
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP 读取失败: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP 读取结果为空: {step_path}")
    return shape


def indexed_map_size(indexed: Any) -> int:
    """
    兼容不同 pythonOCC / OCP 版本：
    有些版本叫 Extent()
    有些版本叫 Size()
    """
    if hasattr(indexed, "Extent"):
        return int(indexed.Extent())
    if hasattr(indexed, "Size"):
        return int(indexed.Size())
    if hasattr(indexed, "Length"):
        return int(indexed.Length())
    raise AttributeError(
        "TopTools_IndexedMapOfShape has no Extent(), Size(), or Length(). "
        f"Available attributes: {dir(indexed)}"
    )


def map_shapes(shape: Any, kind: Any) -> Tuple[List[Any], Any]:
    indexed = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, kind, indexed)

    shapes = []

    n = indexed_map_size(indexed)

    for i in range(1, n + 1):
        s = indexed.FindKey(i)
        shapes.append(s)

    return shapes, indexed


def load_label_file(label_path: Path) -> Dict[str, Any]:
    with label_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        if len(raw) == 0:
            raise ValueError(f"空 label 文件: {label_path}")
        # 常见格式: [[part_id, {"seg":..., "inst":..., "bottom":...}]]
        if isinstance(raw[0], list) and len(raw[0]) >= 2 and isinstance(raw[0][1], dict):
            part_id = raw[0][0]
            data = raw[0][1]
        elif isinstance(raw[0], dict):
            part_id = label_path.stem
            data = raw[0]
        else:
            raise ValueError(f"不认识的 label list 格式: {label_path}")
    elif isinstance(raw, dict):
        part_id = raw.get("part_id", label_path.stem)
        data = raw
    else:
        raise ValueError(f"不认识的 label 格式: {label_path}")

    seg_raw = data.get("seg", {})
    inst_raw = data.get("inst", [])
    bottom_raw = data.get("bottom", {})

    seg: Dict[int, int] = {}
    for k, v in seg_raw.items():
        seg[int(k)] = int(v)

    bottom: Dict[int, int] = {}
    for k, v in bottom_raw.items():
        bottom[int(k)] = int(v)

    face_instance_ids: Dict[int, List[int]] = {}
    if isinstance(inst_raw, list):
        for face_id, row in enumerate(inst_raw):
            ids = []
            if isinstance(row, list):
                for j, val in enumerate(row):
                    if int(val) == 1:
                        ids.append(j)
            face_instance_ids[face_id] = ids

    return {
        "part_id": str(part_id),
        "seg": seg,
        "inst": inst_raw,
        "bottom": bottom,
        "face_instance_ids": face_instance_ids,
    }


# ============================================================
# 几何类型
# ============================================================

def get_face_type(face: Any) -> str:
    try:
        surf = BRepAdaptor_Surface(face, True)
        t = surf.GetType()
        if t == GeomAbs_Plane:
            return "plane"
        if t == GeomAbs_Cylinder:
            return "cylinder"
        if t == GeomAbs_Cone:
            return "cone"
        if t == GeomAbs_Sphere:
            return "sphere"
        if t == GeomAbs_Torus:
            return "torus"
        if t == GeomAbs_BezierSurface:
            return "bezier_surface"
        if t == GeomAbs_BSplineSurface:
            return "bspline_surface"
        if t == GeomAbs_SurfaceOfRevolution:
            return "surface_of_revolution"
        if t == GeomAbs_SurfaceOfExtrusion:
            return "surface_of_extrusion"
        if t == GeomAbs_OffsetSurface:
            return "offset_surface"
        if t == GeomAbs_OtherSurface:
            return "other_surface"
        return str(t)
    except Exception:
        return "unknown"


def get_edge_type(edge: Any) -> str:
    try:
        curve = BRepAdaptor_Curve(edge)
        t = curve.GetType()
        if t == GeomAbs_Line:
            return "line"
        if t == GeomAbs_Circle:
            return "circle"
        if t == GeomAbs_Ellipse:
            return "ellipse"
        if t == GeomAbs_Hyperbola:
            return "hyperbola"
        if t == GeomAbs_Parabola:
            return "parabola"
        if t == GeomAbs_BezierCurve:
            return "bezier_curve"
        if t == GeomAbs_BSplineCurve:
            return "bspline_curve"
        if t == GeomAbs_OffsetCurve:
            return "offset_curve"
        if t == GeomAbs_OtherCurve:
            return "other_curve"
        return str(t)
    except Exception:
        return "unknown"


def surface_signature(face: Any) -> Tuple[Any, ...]:
    """
    近似判断 cosurface 用的 surface signature。
    不是几何建模级别的严格等价，只是为了 graph relation 可用。
    """
    try:
        surf = BRepAdaptor_Surface(face, True)
        t = surf.GetType()
        ft = get_face_type(face)

        if t == GeomAbs_Plane:
            pln = surf.Plane()
            return (
                ft,
                pnt_tuple(pln.Location()),
                dir_tuple(pln.Axis().Direction()),
            )

        if t == GeomAbs_Cylinder:
            cyl = surf.Cylinder()
            return (
                ft,
                round_float(cyl.Radius()),
                pnt_tuple(cyl.Location()),
                dir_tuple(cyl.Axis().Direction()),
            )

        if t == GeomAbs_Cone:
            con = surf.Cone()
            return (
                ft,
                round_float(con.RefRadius()),
                round_float(con.SemiAngle()),
                pnt_tuple(con.Location()),
                dir_tuple(con.Axis().Direction()),
            )

        if t == GeomAbs_Sphere:
            sph = surf.Sphere()
            return (
                ft,
                round_float(sph.Radius()),
                pnt_tuple(sph.Location()),
            )

        if t == GeomAbs_Torus:
            tor = surf.Torus()
            return (
                ft,
                round_float(tor.MajorRadius()),
                round_float(tor.MinorRadius()),
                pnt_tuple(tor.Location()),
                dir_tuple(tor.Axis().Direction()),
            )

        return (ft, shape_key(face))
    except Exception:
        return ("unknown", shape_key(face))


def curve_signature(edge: Any) -> Tuple[Any, ...]:
    """
    近似判断 cocurve 用的 curve signature。
    mate/coedge 同一条 edge 肯定会 cocurve。
    不同 edge 是否 cocurve 用解析几何参数近似。
    """
    try:
        curve = BRepAdaptor_Curve(edge)
        t = curve.GetType()
        et = get_edge_type(edge)

        if t == GeomAbs_Line:
            line = curve.Line()
            return (
                et,
                pnt_tuple(line.Location()),
                dir_tuple(line.Direction()),
            )

        if t == GeomAbs_Circle:
            circle = curve.Circle()
            return (
                et,
                round_float(circle.Radius()),
                pnt_tuple(circle.Location()),
                dir_tuple(circle.Axis().Direction()),
            )

        if t == GeomAbs_Ellipse:
            ellipse = curve.Ellipse()
            return (
                et,
                round_float(ellipse.MajorRadius()),
                round_float(ellipse.MinorRadius()),
                pnt_tuple(ellipse.Location()),
                dir_tuple(ellipse.Axis().Direction()),
            )

        return (et, shape_key(edge))
    except Exception:
        return ("unknown", shape_key(edge))


# ============================================================
# mesh 提取，用于 Dash 3D viewer
# ============================================================

def triangulate_face_mesh(face: Any) -> Dict[str, Any]:
    """
    返回单个 face 的三角网格。
    """
    vertices: List[List[float]] = []
    triangles: List[List[int]] = []

    try:
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            return {"vertices": vertices, "triangles": triangles}

        trsf = loc.Transformation()

        nb_nodes = tri.NbNodes()
        for i in range(1, nb_nodes + 1):
            p = tri.Node(i)
            p.Transform(trsf)
            vertices.append([float(p.X()), float(p.Y()), float(p.Z())])

        nb_triangles = tri.NbTriangles()
        for i in range(1, nb_triangles + 1):
            t = tri.Triangle(i)
            a, b, c = t.Get()
            # Plotly 是 0-based
            triangles.append([int(a) - 1, int(b) - 1, int(c) - 1])

    except Exception:
        pass

    return {"vertices": vertices, "triangles": triangles}


# ============================================================
# coedge 提取
# ============================================================

def edge_vertices_in_oriented_coedge(edge: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    返回 edge occurrence 的 start/end vertex key。
    注意这里使用 coedge 的 orientation 修正方向。
    """
    try:
        v_first = topexp.FirstVertex(edge)
        v_last = topexp.LastVertex(edge)
    except Exception:
        try:
            v_first, v_last = topods.Vertex(), topods.Vertex()
            topexp.Vertices(edge, v_first, v_last)
        except Exception:
            return None, None

    start_key = shape_key(v_first)
    end_key = shape_key(v_last)

    ori = edge.Orientation()
    if ori == TopAbs_REVERSED:
        start_key, end_key = end_key, start_key

    return start_key, end_key


def edge_orientation_bool(edge: Any) -> bool:
    """
    True 表示 coedge 和 parent edge 同向。
    """
    try:
        return edge.Orientation() != TopAbs_REVERSED
    except Exception:
        return True


def is_outer_wire(face: Any, wire: Any) -> bool:
    try:
        outer = breptools.OuterWire(face)
        return not outer.IsNull() and outer.IsSame(wire)
    except Exception:
        return False


def extract_coedges(
    shape: Any,
    faces: List[Any],
    face_indexed_map: Any,
    edge_indexed_map: Any,
    label_data: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    遍历 Face -> Wire -> Edge occurrence。
    每一个 edge occurrence 就是一个 coedge node。
    """
    seg = label_data["seg"]
    bottom = label_data["bottom"]
    face_instance_ids = label_data["face_instance_ids"]

    coedges: List[Dict[str, Any]] = []
    runtime: Dict[str, Any] = {
        "coedge_edge_shape": {},
        "coedge_face_shape": {},
        "coedge_curve_sig": {},
        "coedge_surface_sig": {},
        "edge_to_coedges": defaultdict(list),
        "face_to_coedges": defaultdict(list),
        "next": {},
        "previous": {},
    }

    coedge_id = 0

    for face_id, face in enumerate(faces):
        face_label = seg.get(face_id, None)
        face_label_name = safe_name(face_label)
        face_instances = face_instance_ids.get(face_id, [])
        face_type = get_face_type(face)
        surf_sig = surface_signature(face)

        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        wire_id = 0

        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            loop_type = "outer" if is_outer_wire(face, wire) else "inner"

            wire_coedge_ids: List[int] = []

            edge_exp = TopExp_Explorer(wire, TopAbs_EDGE)
            local_edge_id = 0

            while edge_exp.More():
                edge_occ = topods.Edge(edge_exp.Current())

                # IndexedMap.FindIndex uses OpenCascade topological identity and
                # is stable across oriented edge occurrences. HashCode is not
                # exposed by current pythonOCC builds.
                mapped_index = int(edge_indexed_map.FindIndex(edge_occ))
                edge_id = mapped_index - 1 if mapped_index > 0 else -1

                start_v, end_v = edge_vertices_in_oriented_coedge(edge_occ)

                node = {
                    "id": coedge_id,

                    "face_id": face_id,
                    "face_label": face_label,
                    "face_label_name": face_label_name,
                    "face_instance_ids": face_instances,
                    "face_instance_id": face_instances[0] if len(face_instances) > 0 else None,
                    "is_stock": face_label == STOCK_LABEL_ID,
                    "bottom": int(bottom.get(face_id, 0)),

                    "loop_id": wire_id,
                    "loop_type": loop_type,
                    "is_outer_loop": loop_type == "outer",
                    "local_edge_id_in_loop": local_edge_id,

                    "edge_id": edge_id,
                    "edge_type": get_edge_type(edge_occ),
                    "edge_convexity": "unknown",
                    "dihedral_type": "unknown",

                    "start_vertex_key": start_v,
                    "end_vertex_key": end_v,
                    "orientation": edge_orientation_bool(edge_occ),

                    "face_type": face_type,

                    "is_feature_internal": False,
                    "internal_feature_label": None,
                    "internal_feature_name": None,
                    "internal_feature_instance_ids": [],
                }

                coedges.append(node)

                runtime["coedge_edge_shape"][coedge_id] = edge_occ
                runtime["coedge_face_shape"][coedge_id] = face
                runtime["coedge_curve_sig"][coedge_id] = curve_signature(edge_occ)
                runtime["coedge_surface_sig"][coedge_id] = surf_sig
                runtime["edge_to_coedges"][edge_id].append(coedge_id)
                runtime["face_to_coedges"][face_id].append(coedge_id)

                wire_coedge_ids.append(coedge_id)

                coedge_id += 1
                local_edge_id += 1
                edge_exp.Next()

            # next / previous
            if len(wire_coedge_ids) >= 2:
                n = len(wire_coedge_ids)
                for i, cid in enumerate(wire_coedge_ids):
                    nxt = wire_coedge_ids[(i + 1) % n]
                    prv = wire_coedge_ids[(i - 1) % n]
                    runtime["next"][cid] = nxt
                    runtime["previous"][cid] = prv

            wire_id += 1
            wire_exp.Next()

    return coedges, runtime


# ============================================================
# feature internal 判断
# ============================================================

def instance_intersects(a: List[int], b: List[int]) -> bool:
    if not a or not b:
        return False
    return len(set(a).intersection(set(b))) > 0


def is_same_feature_instance(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    只有 mate 两侧属于同一个非 stock feature，才认为是 feature internal。
    """
    la = a.get("face_label")
    lb = b.get("face_label")

    if la is None or lb is None:
        return False

    if la == STOCK_LABEL_ID or lb == STOCK_LABEL_ID:
        return False

    if la != lb:
        return False

    ia = a.get("face_instance_ids", []) or []
    ib = b.get("face_instance_ids", []) or []

    if instance_intersects(ia, ib):
        return True

    # 如果 label 文件没有 instance 信息，给一个保守 fallback：
    # 两边都是同一非 stock label 且两边都没有 instance，则认为同 feature。
    if len(ia) == 0 and len(ib) == 0:
        return True

    return False


def mark_feature_internal_nodes(coedges: List[Dict[str, Any]], runtime: Dict[str, Any]) -> None:
    edge_to_coedges = runtime["edge_to_coedges"]

    for edge_id, cids in edge_to_coedges.items():
        if edge_id == -1:
            continue
        if len(cids) < 2:
            continue

        # 通常 manifold B-Rep 一条 edge 两侧各一个 coedge。
        # 如果有多个 occurrence，就两两检查。
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                ci = cids[i]
                cj = cids[j]
                ni = coedges[ci]
                nj = coedges[cj]

                if is_same_feature_instance(ni, nj):
                    label = ni["face_label"]
                    instances = sorted(
                        list(
                            set(ni.get("face_instance_ids", []))
                            .intersection(set(nj.get("face_instance_ids", [])))
                        )
                    )

                    if len(instances) == 0:
                        instances = ni.get("face_instance_ids", [])

                    for n in (ni, nj):
                        n["is_feature_internal"] = True
                        n["internal_feature_label"] = label
                        n["internal_feature_name"] = safe_name(label)
                        n["internal_feature_instance_ids"] = instances


# ============================================================
# CoAG edge relation
# ============================================================

def compute_relation_vector(
    i: int,
    j: int,
    coedges: List[Dict[str, Any]],
    runtime: Dict[str, Any],
) -> Tuple[Dict[str, bool], List[int]]:
    ni = coedges[i]
    nj = coedges[j]

    same_edge = (
        ni.get("edge_id") is not None
        and ni.get("edge_id") != -1
        and ni.get("edge_id") == nj.get("edge_id")
    )

    # mate：同一条 parent edge，方向相反。
    # 如果方向信息不可靠，同一条 edge 的不同 coedge occurrence 也会被 mate 捕获。
    mate = bool(
        same_edge
        and i != j
        and (
            ni.get("orientation") != nj.get("orientation")
            or ni.get("face_id") != nj.get("face_id")
        )
    )

    next_rel = runtime["next"].get(i) == j
    previous_rel = runtime["previous"].get(i) == j

    cocurve = bool(
        same_edge
        or runtime["coedge_curve_sig"].get(i) == runtime["coedge_curve_sig"].get(j)
    )

    cosurface = bool(
        ni.get("face_id") == nj.get("face_id")
        or runtime["coedge_surface_sig"].get(i) == runtime["coedge_surface_sig"].get(j)
    )

    coface = ni.get("face_id") == nj.get("face_id")

    relations = {
        "mate": mate,
        "next": bool(next_rel),
        "previous": bool(previous_rel),
        "cocurve": bool(cocurve),
        "cosurface": bool(cosurface),
        "coface": bool(coface),
    }

    vector = [1 if relations[k] else 0 for k in RELATION_ORDER]
    return relations, vector


def build_coag_edges(
    coedges: List[Dict[str, Any]],
    runtime: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    只保留前 6 个关系。
    只要 6 维中至少一个是 1，就添加一条 graph edge。
    """
    n = len(coedges)
    edges: List[Dict[str, Any]] = []

    for i in range(n):
        for j in range(i + 1, n):
            relations, vector = compute_relation_vector(i, j, coedges, runtime)
            if any(vector):
                edges.append({
                    "u": i,
                    "v": j,
                    "relations": relations,
                    "relation_vector": vector,
                })

    return edges


# ============================================================
# face mesh
# ============================================================

def build_face_meshes(
    shape: Any,
    faces: List[Any],
    label_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    BRepMesh_IncrementalMesh(
        shape,
        MESH_LINEAR_DEFLECTION,
        False,
        MESH_ANGULAR_DEFLECTION,
        True,
    )

    seg = label_data["seg"]
    bottom = label_data["bottom"]
    face_instance_ids = label_data["face_instance_ids"]

    face_meshes: List[Dict[str, Any]] = []

    for face_id, face in enumerate(faces):
        label_id = seg.get(face_id, None)
        mesh = triangulate_face_mesh(face)

        face_meshes.append({
            "face_id": face_id,
            "face_label": label_id,
            "face_label_name": safe_name(label_id),
            "face_instance_ids": face_instance_ids.get(face_id, []),
            "bottom": int(bottom.get(face_id, 0)),
            "face_type": get_face_type(face),
            "vertices": mesh["vertices"],
            "triangles": mesh["triangles"],
        })

    return face_meshes


# ============================================================
# 单文件处理
# ============================================================

def extract_one_part(step_path: Path, label_path: Path, out_json_path: Path) -> Dict[str, Any]:
    label_data = load_label_file(label_path)
    shape = read_step_shape(step_path)

    faces, face_indexed_map = map_shapes(shape, TopAbs_FACE)
    edges, edge_indexed_map = map_shapes(shape, TopAbs_EDGE)

    coedges, runtime = extract_coedges(
        shape=shape,
        faces=faces,
        face_indexed_map=face_indexed_map,
        edge_indexed_map=edge_indexed_map,
        label_data=label_data,
    )

    mark_feature_internal_nodes(coedges, runtime)
    graph_edges = build_coag_edges(coedges, runtime)
    face_meshes = build_face_meshes(shape, faces, label_data)

    result = {
        "part_id": label_data["part_id"],
        "step_path": str(step_path),
        "label_path": str(label_path),

        "n_faces": len(faces),
        "n_brep_edges": len(edges),
        "n_coedges": len(coedges),
        "n_graph_edges": len(graph_edges),

        "face_categories": FACE_CATEGORIES,
        "stock_label_id": STOCK_LABEL_ID,

        "relation_order": RELATION_ORDER,
        "relation_note": "Only 6 dimensions are used: mate, next, previous, cocurve, cosurface, coface. share_start/share_end are removed.",

        "nodes": coedges,
        "edges": graph_edges,
        "face_meshes": face_meshes,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ============================================================
# 批处理
# ============================================================

def find_step_files(steps_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.step", "*.stp", "*.STEP", "*.STP"):
        files.extend(steps_dir.glob(ext))
    return sorted(files)


def find_label_for_step(step_path: Path, labels_dir: Path) -> Optional[Path]:
    candidates = [
        labels_dir / f"{step_path.stem}.json",
        labels_dir / f"{step_path.name}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    step_files = find_step_files(STEPS_DIR)
    if not step_files:
        print(f"[ERROR] 没有在 steps 文件夹找到 STEP/STP 文件: {STEPS_DIR}")
        return

    print(f"[INFO] ROOT_DIR  = {ROOT_DIR}")
    print(f"[INFO] STEPS_DIR = {STEPS_DIR}")
    print(f"[INFO] LABELS_DIR= {LABELS_DIR}")
    print(f"[INFO] OUT_DIR   = {OUT_DIR}")
    print(f"[INFO] 找到 STEP 文件数量: {len(step_files)}")

    index_items: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []

    for k, step_path in enumerate(step_files, start=1):
        label_path = find_label_for_step(step_path, LABELS_DIR)
        if label_path is None:
            msg = f"找不到对应 label json: {step_path.name}"
            print(f"[WARN] {msg}")
            failed_items.append({
                "step_path": str(step_path),
                "error": msg,
            })
            continue

        out_json_path = OUT_DIR / f"{step_path.stem}.coag.json"

        print(f"\n[INFO] ({k}/{len(step_files)}) 处理: {step_path.name}")

        try:
            result = extract_one_part(step_path, label_path, out_json_path)
            item = {
                "part_id": result["part_id"],
                "step_path": str(step_path),
                "label_path": str(label_path),
                "coag_json": str(out_json_path),
                "n_faces": result["n_faces"],
                "n_brep_edges": result["n_brep_edges"],
                "n_coedges": result["n_coedges"],
                "n_graph_edges": result["n_graph_edges"],
            }
            index_items.append(item)

            print(
                f"[OK] faces={result['n_faces']} "
                f"brep_edges={result['n_brep_edges']} "
                f"coedges={result['n_coedges']} "
                f"graph_edges={result['n_graph_edges']}"
            )

        except Exception as e:
            print(f"[FAILED] {step_path.name}: {e}")
            traceback.print_exc()
            failed_items.append({
                "step_path": str(step_path),
                "label_path": str(label_path),
                "error": repr(e),
            })

    index = {
        "root_dir": str(ROOT_DIR),
        "steps_dir": str(STEPS_DIR),
        "labels_dir": str(LABELS_DIR),
        "out_dir": str(OUT_DIR),
        "n_success": len(index_items),
        "n_failed": len(failed_items),
        "items": index_items,
        "failed": failed_items,
    }

    index_path = OUT_DIR / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print(f"[INFO] 成功: {len(index_items)}")
    print(f"[INFO] 失败: {len(failed_items)}")
    print(f"[INFO] index: {index_path}")


if __name__ == "__main__":
    main()
