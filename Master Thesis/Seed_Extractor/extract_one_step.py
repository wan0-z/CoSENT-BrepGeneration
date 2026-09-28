# extract_one_step.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import json
import math
import traceback

from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple

import networkx as nx

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve2d
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeHalfSpace
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.GeomAbs import GeomAbs_BezierCurve
from OCC.Core.GeomAbs import GeomAbs_BezierSurface
from OCC.Core.GeomAbs import GeomAbs_BSplineCurve
from OCC.Core.GeomAbs import GeomAbs_BSplineSurface
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Core.GeomAbs import GeomAbs_Cone
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.GeomAbs import GeomAbs_Ellipse
from OCC.Core.GeomAbs import GeomAbs_Hyperbola
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Core.GeomAbs import GeomAbs_OffsetCurve
from OCC.Core.GeomAbs import GeomAbs_OffsetSurface
from OCC.Core.GeomAbs import GeomAbs_OtherCurve
from OCC.Core.GeomAbs import GeomAbs_OtherSurface
from OCC.Core.GeomAbs import GeomAbs_Parabola
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.GeomAbs import GeomAbs_Sphere
from OCC.Core.GeomAbs import GeomAbs_SurfaceOfExtrusion
from OCC.Core.GeomAbs import GeomAbs_SurfaceOfRevolution
from OCC.Core.GeomAbs import GeomAbs_Torus
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.gp import gp_Pnt
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.gp import gp_Dir
from OCC.Core.gp import gp_Pln
from OCC.Core.gp import gp_Vec
from OCC.Core.gp import gp_Vec2d
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Core.GProp import GProp_GProps

from evaluation import calculate_evaluation


# ============================================================
# 手动配置
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

DEFAULT_SINGLE_SAMPLE_NAME = "20221123_142528_0"

SINGLE_DATA_ROOT = SCRIPT_DIRECTORY / "data" / "single"

SINGLE_OUTPUT_ROOT = SCRIPT_DIRECTORY / "output" / "single"

STEP_PATH = (
    SINGLE_DATA_ROOT
    / DEFAULT_SINGLE_SAMPLE_NAME
    / f"{DEFAULT_SINGLE_SAMPLE_NAME}.step"
)

GROUND_TRUTH_JSON_PATH = (
    SINGLE_DATA_ROOT
    / DEFAULT_SINGLE_SAMPLE_NAME
    / f"{DEFAULT_SINGLE_SAMPLE_NAME}.json"
)

FEATURE_SEED_JSON_PATH = SCRIPT_DIRECTORY / "data" / "feature_seeds.json"

OUTPUT_CACHE_PATH = (
    SINGLE_OUTPUT_ROOT
    / DEFAULT_SINGLE_SAMPLE_NAME
    / "cache.json"
)

OUTPUT_FAG_PATH = (
    SINGLE_OUTPUT_ROOT
    / DEFAULT_SINGLE_SAMPLE_NAME
    / "attributed_fag.json"
)

OUTPUT_EVALUATION_PATH = (
    SINGLE_OUTPUT_ROOT
    / DEFAULT_SINGLE_SAMPLE_NAME
    / "evaluation.json"
)

LINEAR_DEFLECTION = 0.1

ANGULAR_DEFLECTION = 0.5

PARALLEL_MESHING = True

GEOMETRIC_TOLERANCE = 1.0e-6

EQUATION_TOLERANCE = 1.0e-5

SUPPORT_KEY_DIGITS = 6

RIGHT_ANGLE_TOLERANCE_DEGREES = 10.0

G1_NORMAL_TOLERANCE_DEGREES = 0.1

FEATURE_SPACE_HALFSPACE_TOLERANCE = 1.0e-5

GEOMETRIC_RELATION_ANGLE_TOLERANCE_DEGREES = 1.0

GEOMETRIC_RELATION_LENGTH_TOLERANCE = 1.0e-5

GEOMETRIC_RELATION_RADIUS_TOLERANCE = 1.0e-5

FEATURE_SPACE_COLLISION_OFFSET_RELATIVE = 1.0e-6

FEATURE_SPACE_COLLISION_OFFSET_MINIMUM = 1.0e-5

FEATURE_SPACE_COLLISION_VOLUME_RELATIVE_TOLERANCE = 1.0e-9

FEATURE_SPACE_COLLISION_VOLUME_MINIMUM_TOLERANCE = 1.0e-8

EDGE_SAMPLE_RATIOS = [
    0.50,
    0.35,
    0.65,
    0.20,
    0.80,
]

REQUIRE_EXACT_EDGE_TYPE = True

REQUIRE_EXACT_CONVEXITY = True

REQUIRE_EXACT_DIHEDRAL_TYPE = True

ALLOW_UNKNOWN_EDGE_ATTRIBUTES = False

REQUIRE_BASIC_EXTERNAL_BOUNDARY_RULE = True

REQUIRE_EXPANDED_FACE_EXTERNAL_BOUNDARY_RULE = True

RETRY_FAILED_EXPANSION_CANDIDATES = True

REVERSE_CONVEXITY_SIGN = False

MAX_BASIC_MATCHES_PER_FEATURE = 10000

PRINT_MATCH_DETAILS = True


# ============================================================
# 类别
# ============================================================

FACE_CATEGORIES = [
    "chamfer",
    "through_hole",
    "triangular_passage",
    "rectangular_passage",
    "6sides_passage",
    "triangular_through_slot",
    "rectangular_through_slot",
    "circular_through_slot",
    "rectangular_through_step",
    "2sides_through_step",
    "slanted_through_step",
    "Oring",
    "blind_hole",
    "triangular_pocket",
    "rectangular_pocket",
    "6sides_pocket",
    "circular_end_pocket",
    "rectangular_blind_slot",
    "v_circular_end_blind_slot",
    "h_circular_end_blind_slot",
    "triangular_blind_step",
    "circular_blind_step",
    "rectangular_blind_step",
    "round",
    "stock",
]

EXCLUDED_CATEGORY_IDS = {
    0,
    10,
    20,
    23,
}

STOCK_CATEGORY_ID = 24


# ============================================================
# 颜色
# ============================================================

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
# 通用辅助函数
# ============================================================

def round_value(
    value: float,
) -> float:
    return round(
        float(value),
        SUPPORT_KEY_DIGITS,
    )


def same_shape(
    first_shape: Any,
    second_shape: Any,
) -> bool:
    try:
        return bool(
            first_shape.IsSame(
                second_shape
            )
        )

    except Exception:
        return bool(
            first_shape.IsEqual(
                second_shape
            )
        )


def unique_shapes(
    shapes: Iterable[Any],
) -> List[Any]:
    result: List[Any] = []

    for shape in shapes:
        exists = False

        for existing_shape in result:
            if same_shape(
                shape,
                existing_shape,
            ):
                exists = True
                break

        if not exists:
            result.append(
                shape
            )

    return result


def explore_shapes(
    parent_shape: Any,
    shape_type: Any,
) -> List[Any]:
    result: List[Any] = []

    explorer = TopExp_Explorer(
        parent_shape,
        shape_type,
    )

    while explorer.More():
        current = explorer.Current()

        if shape_type == TopAbs_FACE:
            result.append(
                topods.Face(
                    current
                )
            )

        elif shape_type == TopAbs_EDGE:
            result.append(
                topods.Edge(
                    current
                )
            )

        else:
            result.append(
                current
            )

        explorer.Next()

    return result


def canonical_direction(
    x: float,
    y: float,
    z: float,
) -> Tuple[float, float, float]:
    length = math.sqrt(
        x * x
        + y * y
        + z * z
    )

    if length <= GEOMETRIC_TOLERANCE:
        return (
            0.0,
            0.0,
            0.0,
        )

    x /= length
    y /= length
    z /= length

    for component in (
        x,
        y,
        z,
    ):
        if abs(component) <= GEOMETRIC_TOLERANCE:
            continue

        if component < 0.0:
            x = -x
            y = -y
            z = -z

        break

    return (
        round_value(x),
        round_value(y),
        round_value(z),
    )


def canonical_axis_location(
    location: Any,
    direction: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    location_vector = gp_Vec(
        float(location.X()),
        float(location.Y()),
        float(location.Z()),
    )

    direction_vector = gp_Vec(
        direction[0],
        direction[1],
        direction[2],
    )

    projection_length = location_vector.Dot(
        direction_vector
    )

    perpendicular = location_vector.Subtracted(
        direction_vector.Multiplied(
            projection_length
        )
    )

    return (
        round_value(
            perpendicular.X()
        ),
        round_value(
            perpendicular.Y()
        ),
        round_value(
            perpendicular.Z()
        ),
    )


# ============================================================
# STEP和标签读取
# ============================================================

def load_step_shape() -> Any:
    reader = STEPControl_Reader()

    status = reader.ReadFile(
        str(STEP_PATH)
    )

    if status != IFSelect_RetDone:
        raise RuntimeError(
            f"STEP读取失败: {STEP_PATH}"
        )

    transferred = reader.TransferRoots()

    if transferred <= 0:
        raise RuntimeError(
            "STEP文件没有可转换的root"
        )

    shape = reader.OneShape()

    if shape.IsNull():
        raise RuntimeError(
            "STEP转换结果为空"
        )

    return shape


def load_feature_seeds() -> List[Dict[str, Any]]:
    with FEATURE_SEED_JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    features = []

    for feature in data["features"]:
        category_id = int(
            feature["category_id"]
        )

        if category_id in EXCLUDED_CATEGORY_IDS:
            continue

        feature_copy = dict(
            feature
        )

        feature_copy["node_count"] = len(
            feature["nodes"]
        )

        feature_copy["edge_count"] = len(
            feature["edges"]
        )

        features.append(
            feature_copy
        )

    features.sort(
        key=lambda item: (
            -int(item["node_count"]),
            -int(item["edge_count"]),
            int(item["category_id"]),
        )
    )

    return features


def load_ground_truth() -> Dict[str, Any]:
    with GROUND_TRUTH_JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_data = json.load(
            file
        )

    if (
        isinstance(raw_data, list)
        and len(raw_data) > 0
        and isinstance(raw_data[0], list)
        and len(raw_data[0]) == 2
    ):
        part_name = str(
            raw_data[0][0]
        )

        label_data = raw_data[0][1]

    elif isinstance(raw_data, dict):
        part_name = STEP_PATH.stem
        label_data = raw_data

    else:
        raise ValueError(
            "无法识别真实标签JSON格式"
        )

    seg_raw = label_data.get(
        "seg",
        {}
    )

    bottom_raw = label_data.get(
        "bottom",
        {}
    )

    inst_matrix = label_data.get(
        "inst",
        []
    )

    seg = {
        int(face_id): int(category_id)
        for face_id, category_id in seg_raw.items()
    }

    bottom = {
        int(face_id): int(value)
        for face_id, value in bottom_raw.items()
    }

    instance_ids = recover_instance_ids_from_matrix(
        inst_matrix
    )

    return {
        "part_name": part_name,
        "seg": seg,
        "bottom": bottom,
        "instance_ids": instance_ids,
        "inst_matrix": inst_matrix,
    }


def recover_instance_ids_from_matrix(
    inst_matrix: Sequence[Sequence[int]],
) -> Dict[int, Optional[int]]:
    face_count = len(
        inst_matrix
    )

    graph = nx.Graph()

    graph.add_nodes_from(
        range(face_count)
    )

    for first_face_id in range(
        face_count
    ):
        row = inst_matrix[
            first_face_id
        ]

        for second_face_id in range(
            min(
                len(row),
                face_count,
            )
        ):
            if first_face_id == second_face_id:
                continue

            if int(
                row[second_face_id]
            ) == 1:
                graph.add_edge(
                    first_face_id,
                    second_face_id,
                )

    result: Dict[int, Optional[int]] = {
        face_id: None
        for face_id in range(face_count)
    }

    instance_id = 0

    for component in nx.connected_components(
        graph
    ):
        component_faces = sorted(
            int(face_id)
            for face_id in component
        )

        if len(component_faces) <= 1:
            face_id = component_faces[0]

            has_self_instance = False

            if (
                face_id < len(inst_matrix)
                and face_id < len(
                    inst_matrix[face_id]
                )
            ):
                has_self_instance = bool(
                    inst_matrix[
                        face_id
                    ][
                        face_id
                    ]
                )

            if not has_self_instance:
                continue

        for face_id in component_faces:
            result[
                face_id
            ] = instance_id

        instance_id += 1

    return result


# ============================================================
# 几何类型
# ============================================================

def get_face_type(
    face: Any,
) -> str:
    adaptor = BRepAdaptor_Surface(
        face,
        True,
    )

    surface_type = adaptor.GetType()

    if surface_type == GeomAbs_Plane:
        return "plane"

    if surface_type == GeomAbs_Cylinder:
        return "cylinder"

    if surface_type == GeomAbs_Cone:
        return "cone"

    if surface_type == GeomAbs_Sphere:
        return "sphere"

    if surface_type == GeomAbs_Torus:
        return "torus"

    if surface_type == GeomAbs_BezierSurface:
        return "bezier"

    if surface_type == GeomAbs_BSplineSurface:
        return "bspline"

    if surface_type == GeomAbs_SurfaceOfRevolution:
        return "surface_of_revolution"

    if surface_type == GeomAbs_SurfaceOfExtrusion:
        return "surface_of_extrusion"

    if surface_type == GeomAbs_OffsetSurface:
        return "offset_surface"

    if surface_type == GeomAbs_OtherSurface:
        return "other"

    return "other"


def get_edge_type(
    edge: Any,
) -> str:
    adaptor = BRepAdaptor_Curve(
        edge
    )

    curve_type = adaptor.GetType()

    if curve_type == GeomAbs_Line:
        return "line"

    if curve_type == GeomAbs_Circle:
        return "circle"

    if curve_type == GeomAbs_Ellipse:
        return "ellipse"

    if curve_type == GeomAbs_Hyperbola:
        return "hyperbola"

    if curve_type == GeomAbs_Parabola:
        return "parabola"

    if curve_type == GeomAbs_BezierCurve:
        return "bezier"

    if curve_type == GeomAbs_BSplineCurve:
        return "bspline"

    if curve_type == GeomAbs_OffsetCurve:
        return "offset_curve"

    if curve_type == GeomAbs_OtherCurve:
        return "other"

    return "other"


def get_oriented_plane_halfspace(
    face: Any,
) -> Optional[Dict[str, Any]]:
    adaptor = BRepAdaptor_Surface(face, True)
    if adaptor.GetType() != GeomAbs_Plane:
        return None

    plane = adaptor.Plane()
    direction = plane.Axis().Direction()
    normal = [
        float(direction.X()),
        float(direction.Y()),
        float(direction.Z()),
    ]
    if face.Orientation() == TopAbs_REVERSED:
        normal = [-value for value in normal]

    location = plane.Location()
    offset = (
        normal[0] * float(location.X())
        + normal[1] * float(location.Y())
        + normal[2] * float(location.Z())
    )
    return {
        "normal": normal,
        "offset": float(offset),
    }


# ============================================================
# 严格cosurface支持方程
# ============================================================

def get_surface_support_key(
    face: Any,
) -> Tuple[Any, ...]:
    adaptor = BRepAdaptor_Surface(
        face,
        True,
    )

    surface_type = adaptor.GetType()

    if surface_type == GeomAbs_Plane:
        plane = adaptor.Plane()

        direction = canonical_direction(
            plane.Axis().Direction().X(),
            plane.Axis().Direction().Y(),
            plane.Axis().Direction().Z(),
        )

        location = plane.Location()

        distance = (
            direction[0] * location.X()
            + direction[1] * location.Y()
            + direction[2] * location.Z()
        )

        return (
            "plane",
            direction,
            round_value(distance),
        )

    if surface_type == GeomAbs_Cylinder:
        cylinder = adaptor.Cylinder()

        direction = canonical_direction(
            cylinder.Axis().Direction().X(),
            cylinder.Axis().Direction().Y(),
            cylinder.Axis().Direction().Z(),
        )

        axis_location = canonical_axis_location(
            cylinder.Axis().Location(),
            direction,
        )

        return (
            "cylinder",
            direction,
            axis_location,
            round_value(
                cylinder.Radius()
            ),
        )

    if surface_type == GeomAbs_Cone:
        cone = adaptor.Cone()

        direction = canonical_direction(
            cone.Axis().Direction().X(),
            cone.Axis().Direction().Y(),
            cone.Axis().Direction().Z(),
        )

        apex = cone.Apex()

        return (
            "cone",
            direction,
            (
                round_value(
                    apex.X()
                ),
                round_value(
                    apex.Y()
                ),
                round_value(
                    apex.Z()
                ),
            ),
            round_value(
                abs(
                    cone.SemiAngle()
                )
            ),
        )

    if surface_type == GeomAbs_Sphere:
        sphere = adaptor.Sphere()

        center = sphere.Location()

        return (
            "sphere",
            (
                round_value(
                    center.X()
                ),
                round_value(
                    center.Y()
                ),
                round_value(
                    center.Z()
                ),
            ),
            round_value(
                sphere.Radius()
            ),
        )

    if surface_type == GeomAbs_Torus:
        torus = adaptor.Torus()

        direction = canonical_direction(
            torus.Axis().Direction().X(),
            torus.Axis().Direction().Y(),
            torus.Axis().Direction().Z(),
        )

        center = torus.Location()

        return (
            "torus",
            direction,
            (
                round_value(
                    center.X()
                ),
                round_value(
                    center.Y()
                ),
                round_value(
                    center.Z()
                ),
            ),
            round_value(
                torus.MajorRadius()
            ),
            round_value(
                torus.MinorRadius()
            ),
        )

    return (
        get_face_type(face),
        str(
            BRep_Tool.Surface(
                face
            )
        ),
    )


# ============================================================
# 严格cocurve支持方程
# ============================================================

def get_curve_support_key(
    edge: Any,
) -> Tuple[Any, ...]:
    adaptor = BRepAdaptor_Curve(
        edge
    )

    curve_type = adaptor.GetType()

    if curve_type == GeomAbs_Line:
        line = adaptor.Line()

        direction = canonical_direction(
            line.Direction().X(),
            line.Direction().Y(),
            line.Direction().Z(),
        )

        location = canonical_axis_location(
            line.Location(),
            direction,
        )

        return (
            "line",
            direction,
            location,
        )

    if curve_type == GeomAbs_Circle:
        circle = adaptor.Circle()

        direction = canonical_direction(
            circle.Axis().Direction().X(),
            circle.Axis().Direction().Y(),
            circle.Axis().Direction().Z(),
        )

        center = circle.Location()

        return (
            "circle",
            direction,
            (
                round_value(
                    center.X()
                ),
                round_value(
                    center.Y()
                ),
                round_value(
                    center.Z()
                ),
            ),
            round_value(
                circle.Radius()
            ),
        )

    if curve_type == GeomAbs_Ellipse:
        ellipse = adaptor.Ellipse()

        direction = canonical_direction(
            ellipse.Axis().Direction().X(),
            ellipse.Axis().Direction().Y(),
            ellipse.Axis().Direction().Z(),
        )

        center = ellipse.Location()

        major_direction = canonical_direction(
            ellipse.XAxis().Direction().X(),
            ellipse.XAxis().Direction().Y(),
            ellipse.XAxis().Direction().Z(),
        )

        return (
            "ellipse",
            direction,
            major_direction,
            (
                round_value(
                    center.X()
                ),
                round_value(
                    center.Y()
                ),
                round_value(
                    center.Z()
                ),
            ),
            round_value(
                ellipse.MajorRadius()
            ),
            round_value(
                ellipse.MinorRadius()
            ),
        )

    return (
        get_edge_type(edge),
        str(
            adaptor.Curve()
        ),
    )


# ============================================================
# seam检测
# ============================================================

def is_seam_edge_on_face(
    edge: Any,
    face: Any,
) -> bool:
    try:
        return bool(
            BRep_Tool.IsClosed(
                edge,
                face,
            )
        )

    except Exception:
        occurrence_count = 0

        explorer = TopExp_Explorer(
            face,
            TopAbs_EDGE,
        )

        while explorer.More():
            candidate = topods.Edge(
                explorer.Current()
            )

            if same_shape(
                candidate,
                edge,
            ):
                occurrence_count += 1

            explorer.Next()

        return occurrence_count >= 2


# ============================================================
# 稳健二面角和凹凸分析
# ============================================================

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def find_oriented_edge_in_face(
    target_edge: Any,
    face: Any,
) -> Any:
    explorer = TopExp_Explorer(
        face,
        TopAbs_EDGE,
    )

    while explorer.More():
        oriented_edge = topods.Edge(
            explorer.Current()
        )

        if same_shape(
            oriented_edge,
            target_edge,
        ):
            return oriented_edge

        explorer.Next()

    raise RuntimeError(
        "没有在face中找到目标edge occurrence"
    )


def valid_parameter_range(
    first_parameter: float,
    last_parameter: float,
) -> bool:
    return (
        math.isfinite(first_parameter)
        and math.isfinite(last_parameter)
        and abs(last_parameter - first_parameter)
        > GEOMETRIC_TOLERANCE
    )


def sample_parameters(
    first_parameter: float,
    last_parameter: float,
) -> List[float]:
    if not valid_parameter_range(
        first_parameter,
        last_parameter,
    ):
        raise RuntimeError(
            (
                "无效edge参数区间: "
                f"{first_parameter}, {last_parameter}"
            )
        )

    return [
        float(
            first_parameter
            + ratio
            * (
                last_parameter
                - first_parameter
            )
        )
        for ratio in EDGE_SAMPLE_RATIOS
    ]


def evaluate_face_normal_from_pcurve(
    edge: Any,
    face: Any,
    parameter: float,
) -> Tuple[
    gp_Pnt,
    gp_Vec,
    Dict[str, float],
]:
    oriented_edge = find_oriented_edge_in_face(
        edge,
        face,
    )

    curve_2d = BRepAdaptor_Curve2d(
        oriented_edge,
        face,
    )

    uv_point = gp_Pnt2d()
    uv_derivative = gp_Vec2d()

    curve_2d.D1(
        parameter,
        uv_point,
        uv_derivative,
    )

    u_parameter = float(
        uv_point.X()
    )

    v_parameter = float(
        uv_point.Y()
    )

    surface = BRepAdaptor_Surface(
        face,
        True,
    )

    surface_point = gp_Pnt()
    derivative_u = gp_Vec()
    derivative_v = gp_Vec()

    surface.D1(
        u_parameter,
        v_parameter,
        surface_point,
        derivative_u,
        derivative_v,
    )

    if (
        derivative_u.Magnitude()
        <= GEOMETRIC_TOLERANCE
    ):
        raise RuntimeError(
            "曲面U方向导数接近零"
        )

    if (
        derivative_v.Magnitude()
        <= GEOMETRIC_TOLERANCE
    ):
        raise RuntimeError(
            "曲面V方向导数接近零"
        )

    normal = derivative_u.Crossed(
        derivative_v
    )

    if normal.Magnitude() <= GEOMETRIC_TOLERANCE:
        raise RuntimeError(
            "曲面法向叉积接近零"
        )

    normal.Normalize()

    if face.Orientation() == TopAbs_REVERSED:
        normal.Reverse()

    return (
        surface_point,
        normal,
        {
            "u": u_parameter,
            "v": v_parameter,
        },
    )


def evaluate_oriented_edge_tangent(
    edge: Any,
    reference_face: Any,
    parameter: float,
) -> Tuple[gp_Pnt, gp_Vec]:
    oriented_edge = find_oriented_edge_in_face(
        edge,
        reference_face,
    )

    curve = BRepAdaptor_Curve(
        edge
    )

    point = gp_Pnt()
    tangent = gp_Vec()

    curve.D1(
        parameter,
        point,
        tangent,
    )

    if tangent.Magnitude() <= GEOMETRIC_TOLERANCE:
        raise RuntimeError(
            "edge tangent长度接近零"
        )

    tangent.Normalize()

    if oriented_edge.Orientation() == TopAbs_REVERSED:
        tangent.Reverse()

    return (
        point,
        tangent,
    )


def classify_dihedral_type(
    normal_angle_degrees: float,
    interior_dihedral_degrees: float,
) -> str:
    if (
        normal_angle_degrees
        <= G1_NORMAL_TOLERANCE_DEGREES
    ):
        return "tangent"

    if (
        normal_angle_degrees
        >= 180.0
        - G1_NORMAL_TOLERANCE_DEGREES
    ):
        return "flat"

    if abs(
        interior_dihedral_degrees
        - 90.0
    ) <= RIGHT_ANGLE_TOLERANCE_DEGREES:
        return "right"

    if interior_dihedral_degrees < 90.0:
        return "acute"

    return "obtuse"


def classify_non_smooth_dihedral_type(
    interior_dihedral_degrees: float,
) -> str:
    if abs(
        interior_dihedral_degrees - 90.0
    ) <= RIGHT_ANGLE_TOLERANCE_DEGREES:
        return "right"
    if interior_dihedral_degrees < 90.0:
        return "acute"
    return "obtuse"


def samples_are_g1_continuous(
    successful_samples: Sequence[Dict[str, Any]],
    first_face: Any,
    second_face: Any,
) -> bool:
    if not successful_samples:
        return False

    if (
        get_face_type(first_face) == "plane"
        and get_face_type(second_face) == "plane"
        and get_surface_support_key(first_face)
        != get_surface_support_key(second_face)
    ):
        return False

    return all(
        min(
            float(sample["normal_angle_degrees"]),
            180.0 - float(sample["normal_angle_degrees"]),
        ) <= G1_NORMAL_TOLERANCE_DEGREES
        for sample in successful_samples
    )


def calculate_relation_at_parameter(
    edge: Any,
    first_face: Any,
    second_face: Any,
    parameter: float,
) -> Dict[str, Any]:
    (
        edge_point,
        tangent,
    ) = evaluate_oriented_edge_tangent(
        edge,
        first_face,
        parameter,
    )

    (
        first_surface_point,
        first_normal,
        first_uv,
    ) = evaluate_face_normal_from_pcurve(
        edge,
        first_face,
        parameter,
    )

    (
        second_surface_point,
        second_normal,
        second_uv,
    ) = evaluate_face_normal_from_pcurve(
        edge,
        second_face,
        parameter,
    )

    dot_value = clamp(
        first_normal.Dot(
            second_normal
        ),
        -1.0,
        1.0,
    )

    normal_angle_degrees = math.degrees(
        math.acos(
            dot_value
        )
    )

    interior_dihedral_degrees = (
        180.0
        - normal_angle_degrees
    )

    dihedral_type = classify_dihedral_type(
        normal_angle_degrees,
        interior_dihedral_degrees,
    )

    signed_value = tangent.Dot(
        first_normal.Crossed(
            second_normal
        )
    )

    if dihedral_type in {
        "tangent",
        "flat",
    }:
        convexity = "smooth"

    else:
        is_concave = signed_value < 0.0

        if REVERSE_CONVEXITY_SIGN:
            is_concave = not is_concave

        convexity = (
            "concave"
            if is_concave
            else "convex"
        )

    return {
        "parameter": float(parameter),
        "edge_point": [
            float(edge_point.X()),
            float(edge_point.Y()),
            float(edge_point.Z()),
        ],
        "first_surface_point": [
            float(first_surface_point.X()),
            float(first_surface_point.Y()),
            float(first_surface_point.Z()),
        ],
        "second_surface_point": [
            float(second_surface_point.X()),
            float(second_surface_point.Y()),
            float(second_surface_point.Z()),
        ],
        "first_uv": first_uv,
        "second_uv": second_uv,
        "tangent": [
            float(tangent.X()),
            float(tangent.Y()),
            float(tangent.Z()),
        ],
        "first_outward_normal": [
            float(first_normal.X()),
            float(first_normal.Y()),
            float(first_normal.Z()),
        ],
        "second_outward_normal": [
            float(second_normal.X()),
            float(second_normal.Y()),
            float(second_normal.Z()),
        ],
        "normal_angle_degrees": float(
            normal_angle_degrees
        ),
        "interior_dihedral_degrees": float(
            interior_dihedral_degrees
        ),
        "dihedral_angle_degrees": float(
            interior_dihedral_degrees
        ),
        "dihedral_type": dihedral_type,
        "signed_value": float(
            signed_value
        ),
        "convexity": convexity,
    }


def calculate_edge_attributes(
    edge: Any,
    first_face: Any,
    second_face: Any,
) -> Dict[str, Any]:
    curve = BRepAdaptor_Curve(
        edge
    )

    parameters = sample_parameters(
        float(
            curve.FirstParameter()
        ),
        float(
            curve.LastParameter()
        ),
    )

    successful_samples: List[
        Dict[str, Any]
    ] = []

    failed_samples: List[
        Dict[str, Any]
    ] = []

    for parameter in parameters:
        try:
            successful_samples.append(
                calculate_relation_at_parameter(
                    edge,
                    first_face,
                    second_face,
                    parameter,
                )
            )

        except Exception as exception:
            failed_samples.append(
                {
                    "parameter": float(
                        parameter
                    ),
                    "error_type": type(
                        exception
                    ).__name__,
                    "error_message": str(
                        exception
                    ),
                }
            )

    if not successful_samples:
        raise RuntimeError(
            (
                "所有edge采样点均计算失败: "
                f"{failed_samples}"
            )
        )

    selected_sample = successful_samples[
        len(successful_samples) // 2
    ]

    g1_continuous = samples_are_g1_continuous(
        successful_samples,
        first_face,
        second_face,
    )

    convexity_counts: Dict[str, int] = defaultdict(int)
    dihedral_counts: Dict[str, int] = defaultdict(int)

    for sample in successful_samples:
        if g1_continuous:
            convexity_counts["smooth"] += 1
            dihedral_counts[sample["dihedral_type"]] += 1
            continue

        dihedral_counts[
            classify_non_smooth_dihedral_type(
                float(sample["interior_dihedral_degrees"])
            )
        ] += 1
        is_concave = float(sample["signed_value"]) < 0.0
        if REVERSE_CONVEXITY_SIGN:
            is_concave = not is_concave
        convexity_counts[
            "concave" if is_concave else "convex"
        ] += 1

    result = dict(
        selected_sample
    )

    result.update(
        {
            "edge_type": get_edge_type(
                edge
            ),
            "curve_type": get_edge_type(
                edge
            ),
            "curve_support_key": list(
                get_curve_support_key(
                    edge
                )
            ),
            "convexity": max(
                convexity_counts,
                key=convexity_counts.get,
            ),
            "dihedral_type": max(
                dihedral_counts,
                key=dihedral_counts.get,
            ),
            "successful_sample_count": len(
                successful_samples
            ),
            "failed_sample_count": len(
                failed_samples
            ),
            "failed_samples": failed_samples,
            "edge_sample_points": [
                list(sample["edge_point"])
                for sample in successful_samples
            ],
            "status": "success",
            "g1_continuous": g1_continuous,
        }
    )

    return result


# ============================================================
# 建立包含多重边和seam self-loop的MultiGraph FAG
# ============================================================

def build_attributed_fag(
    shape: Any,
) -> Tuple[
    nx.MultiGraph,
    List[Any],
    Dict[str, int],
]:
    faces = unique_shapes(
        explore_shapes(
            shape,
            TopAbs_FACE,
        )
    )

    graph = nx.MultiGraph()

    face_meshes = triangulate_faces(
        shape,
        faces,
    )

    edge_records: List[Dict[str, Any]] = []

    for face_id, face in enumerate(
        faces
    ):
        oriented_plane_halfspace = get_oriented_plane_halfspace(face)
        face_edges = unique_shapes(
            explore_shapes(
                face,
                TopAbs_EDGE,
            )
        )

        seam_count = 0

        for edge in face_edges:
            if is_seam_edge_on_face(
                edge,
                face,
            ):
                seam_count += 1

            matched_record = None

            for record in edge_records:
                if same_shape(
                    record["edge"],
                    edge,
                ):
                    matched_record = record
                    break

            if matched_record is None:
                edge_records.append(
                    {
                        "edge": edge,
                        "incident_faces": [
                            face_id,
                        ],
                    }
                )

            elif (
                face_id
                not in matched_record[
                    "incident_faces"
                ]
            ):
                matched_record[
                    "incident_faces"
                ].append(
                    face_id
                )

        graph.add_node(
            face_id,
            face_id=face_id,
            face_type=get_face_type(
                face
            ),
            surface_type=get_face_type(
                face
            ),
            surface_support_key=list(
                get_surface_support_key(
                    face
                )
            ),
            seam_edge_count=seam_count,
            topological_edge_count=len(
                face_edges
            ),
            feature_space_normal=(
                oriented_plane_halfspace["normal"]
                if oriented_plane_halfspace is not None
                else None
            ),
            feature_space_offset=(
                oriented_plane_halfspace["offset"]
                if oriented_plane_halfspace is not None
                else None
            ),
            _face_sample_points=[
                list(point)
                for point in face_meshes[face_id].get("vertices", [])
            ],
        )

    successful_two_face_edge_count = 0
    failed_two_face_edge_count = 0
    seam_edge_count = 0
    boundary_edge_count = 0

    for edge_id, record in enumerate(
        edge_records
    ):
        edge = record["edge"]
        incident_faces = record[
            "incident_faces"
        ]

        edge_type = get_edge_type(
            edge
        )

        curve_support_key = list(
            get_curve_support_key(
                edge
            )
        )

        if len(incident_faces) == 1:
            face_id = int(
                incident_faces[0]
            )

            if is_seam_edge_on_face(
                edge,
                faces[face_id],
            ):
                graph.add_edge(
                    face_id,
                    face_id,
                    key=edge_id,
                    topological_edge_id=edge_id,
                    edge_type=edge_type,
                    curve_type=edge_type,
                    curve_support_key=(
                        curve_support_key
                    ),
                    normal_angle_degrees=0.0,
                    interior_dihedral_degrees=180.0,
                    dihedral_angle_degrees=180.0,
                    dihedral_type="tangent",
                    convexity="smooth",
                    signed_value=0.0,
                    is_seam=True,
                    is_boundary=False,
                    status="success",
                )

                seam_edge_count += 1

            else:
                boundary_edge_count += 1

            continue

        if len(incident_faces) < 2:
            continue

        for first_index in range(
            len(incident_faces)
        ):
            for second_index in range(
                first_index + 1,
                len(incident_faces),
            ):
                first_face_id = int(
                    incident_faces[
                        first_index
                    ]
                )

                second_face_id = int(
                    incident_faces[
                        second_index
                    ]
                )

                try:
                    attributes = calculate_edge_attributes(
                        edge,
                        faces[first_face_id],
                        faces[second_face_id],
                    )

                    attributes.update(
                        {
                            "topological_edge_id": edge_id,
                            "edge_type": edge_type,
                            "curve_type": edge_type,
                            "curve_support_key": (
                                curve_support_key
                            ),
                            "is_seam": False,
                            "is_boundary": False,
                            "status": "success",
                        }
                    )

                    graph.add_edge(
                        first_face_id,
                        second_face_id,
                        key=edge_id,
                        **attributes,
                    )

                    successful_two_face_edge_count += 1

                except Exception as exception:
                    failed_two_face_edge_count += 1

                    graph.add_edge(
                        first_face_id,
                        second_face_id,
                        key=edge_id,
                        topological_edge_id=edge_id,
                        edge_type=edge_type,
                        curve_type=edge_type,
                        curve_support_key=(
                            curve_support_key
                        ),
                        normal_angle_degrees=None,
                        interior_dihedral_degrees=None,
                        dihedral_angle_degrees=None,
                        dihedral_type="unknown",
                        convexity="unknown",
                        signed_value=None,
                        is_seam=False,
                        is_boundary=False,
                        status="failed",
                        attribute_error=(
                            f"{type(exception).__name__}: "
                            f"{exception}"
                        ),
                    )

                    print(
                        (
                            f"[FAILED EDGE] E{edge_id}: "
                            f"F{first_face_id} <-> "
                            f"F{second_face_id} | "
                            f"{type(exception).__name__}: "
                            f"{exception}"
                        )
                    )

    statistics = {
        "face_count": len(faces),
        "unique_topological_edge_count": len(
            edge_records
        ),
        "successful_two_face_edge_count": (
            successful_two_face_edge_count
        ),
        "failed_two_face_edge_count": (
            failed_two_face_edge_count
        ),
        "seam_edge_count": seam_edge_count,
        "boundary_edge_count": boundary_edge_count,
    }

    return (
        graph,
        faces,
        statistics,
    )


# ============================================================
# Seed图
# ============================================================

def build_seed_graph(
    feature: Dict[str, Any],
) -> nx.MultiGraph:
    graph = nx.MultiGraph()

    graph.graph["dihedral_sum_degrees"] = feature.get(
        "dihedral_sum_degrees"
    )
    graph.graph["dihedral_sum_tolerance_degrees"] = float(
        feature.get("dihedral_sum_tolerance_degrees", 1.0)
    )
    graph.graph["geometric_constraints"] = list(
        feature.get("geometric_constraints", [])
    )

    for node in feature["nodes"]:
        node_id = int(
            node["id"]
        )

        graph.add_node(
            node_id,
            seed_node_id=node_id,
            role=node.get(
                "role",
                f"role_{node_id}",
            ),
            face_type=node.get(
                "surface_type",
                node.get(
                    "face_type",
                    "unknown",
                ),
            ),
        )

    for edge_index, edge in enumerate(
        feature["edges"]
    ):
        source = int(
            edge["source"]
        )

        target = int(
            edge["target"]
        )

        graph.add_edge(
            source,
            target,
            key=edge_index,
            seed_edge_id=edge_index,
            edge_type=edge.get(
                "edge_type",
                edge.get(
                    "curve_type",
                    "unknown",
                ),
            ),
            curve_type=edge.get(
                "curve_type",
                edge.get(
                    "edge_type",
                    "unknown",
                ),
            ),
            convexity=edge.get(
                "convexity",
                "unknown",
            ),
            dihedral_type=edge.get(
                "dihedral_type",
                "unknown",
            ),
            self_loop=bool(
                edge.get(
                    "self_loop",
                    source == target,
                )
            ),
        )

    return graph


# ============================================================
# Edge bundle匹配
# ============================================================

def one_model_edge_matches_seed_edge(
    model_edge: Dict[str, Any],
    seed_edge: Dict[str, Any],
) -> bool:
    model_edge_type = model_edge.get(
        "edge_type",
        model_edge.get(
            "curve_type",
            "unknown",
        ),
    )

    seed_edge_type = seed_edge.get(
        "edge_type",
        seed_edge.get(
            "curve_type",
            "unknown",
        ),
    )

    if REQUIRE_EXACT_EDGE_TYPE:
        if model_edge_type != seed_edge_type:
            return False

    if REQUIRE_EXACT_CONVEXITY:
        model_convexity = model_edge.get(
            "convexity",
            "unknown",
        )

        seed_convexity = seed_edge.get(
            "convexity",
            "unknown",
        )

        if (
            model_convexity == "unknown"
            and ALLOW_UNKNOWN_EDGE_ATTRIBUTES
        ):
            pass

        elif model_convexity != seed_convexity:
            return False

    if REQUIRE_EXACT_DIHEDRAL_TYPE:
        model_dihedral_type = model_edge.get(
            "dihedral_type",
            "unknown",
        )

        seed_dihedral_type = seed_edge.get(
            "dihedral_type",
            "unknown",
        )

        if isinstance(seed_dihedral_type, (list, tuple, set)):
            accepted_seed_dihedral_types = {
                str(value)
                for value in seed_dihedral_type
            }
        else:
            accepted_seed_dihedral_types = {
                str(seed_dihedral_type)
            }

        if (
            model_dihedral_type == "unknown"
            and ALLOW_UNKNOWN_EDGE_ATTRIBUTES
        ):
            pass

        elif (
            "any" not in accepted_seed_dihedral_types
            and model_dihedral_type not in accepted_seed_dihedral_types
        ):
            return False

    if bool(
        seed_edge.get(
            "self_loop",
            False,
        )
    ):
        if not bool(
            model_edge.get(
                "is_seam",
                False,
            )
        ):
            return False

    return True


def edge_bundle_can_cover_seed_bundle(
    model_edges: List[Dict[str, Any]],
    seed_edges: List[Dict[str, Any]],
) -> bool:
    if len(model_edges) < len(seed_edges):
        return False

    used_model_indices: Set[int] = set()

    seed_edges_sorted = sorted(
        seed_edges,
        key=lambda edge: (
            bool(
                edge.get(
                    "self_loop",
                    False,
                )
            ),
            edge.get(
                "edge_type",
                edge.get(
                    "curve_type",
                    "",
                ),
            ),
            edge.get(
                "convexity",
                "",
            ),
        ),
        reverse=True,
    )

    def backtrack(
        seed_index: int,
    ) -> bool:
        if seed_index >= len(
            seed_edges_sorted
        ):
            return True

        seed_edge = seed_edges_sorted[
            seed_index
        ]

        for model_index, model_edge in enumerate(
            model_edges
        ):
            if model_index in used_model_indices:
                continue

            if not one_model_edge_matches_seed_edge(
                model_edge,
                seed_edge,
            ):
                continue

            used_model_indices.add(
                model_index
            )

            if backtrack(
                seed_index + 1
            ):
                return True

            used_model_indices.remove(
                model_index
            )

        return False

    return backtrack(
        0
    )


def get_multigraph_edge_bundle(
    graph: nx.MultiGraph,
    first_node: int,
    second_node: int,
) -> List[Dict[str, Any]]:
    if not graph.has_edge(
        first_node,
        second_node,
    ):
        return []

    edge_dictionary = graph.get_edge_data(
        first_node,
        second_node,
    )

    return [
        dict(attributes)
        for attributes in edge_dictionary.values()
    ]


def basic_mapping_satisfies_seed_constraints(
    model_graph: nx.MultiGraph,
    seed_graph: nx.MultiGraph,
    seed_to_model: Dict[int, int],
) -> bool:
    target_sum = seed_graph.graph.get("dihedral_sum_degrees")
    if target_sum is None:
        return True

    angle_options: List[List[float]] = []
    for source, target, attributes in seed_graph.edges(data=True):
        model_edges = get_multigraph_edge_bundle(
            model_graph,
            int(seed_to_model[int(source)]),
            int(seed_to_model[int(target)]),
        )
        compatible_angles = [
            float(model_edge["dihedral_angle_degrees"])
            for model_edge in model_edges
            if model_edge.get("dihedral_angle_degrees") is not None
            and one_model_edge_matches_seed_edge(model_edge, attributes)
        ]
        if not compatible_angles:
            return False
        angle_options.append(compatible_angles)

    tolerance = float(
        seed_graph.graph.get("dihedral_sum_tolerance_degrees", 1.0)
    )
    return any(
        abs(sum(angles) - float(target_sum)) <= tolerance
        for angles in itertools.product(*angle_options)
    )


def vector_dot(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return sum(float(a) * float(b) for a, b in zip(first, second))


def vector_cross(
    first: Sequence[float],
    second: Sequence[float],
) -> List[float]:
    return [
        float(first[1]) * float(second[2]) - float(first[2]) * float(second[1]),
        float(first[2]) * float(second[0]) - float(first[0]) * float(second[2]),
        float(first[0]) * float(second[1]) - float(first[1]) * float(second[0]),
    ]


def normalized_vector(values: Sequence[float]) -> Optional[List[float]]:
    length = math.sqrt(vector_dot(values, values))
    if length <= GEOMETRIC_TOLERANCE:
        return None
    return [float(value) / length for value in values]


def mapped_plane_normal(
    model_graph: nx.MultiGraph,
    seed_to_model: Dict[int, int],
    seed_face_id: int,
) -> Optional[List[float]]:
    attributes = model_graph.nodes[int(seed_to_model[int(seed_face_id)])]
    if attributes.get("face_type") != "plane":
        return None
    normal = attributes.get("feature_space_normal")
    if normal is None:
        return None
    return normalized_vector(normal)


def mapped_cylinder_parameters(
    model_graph: nx.MultiGraph,
    seed_to_model: Dict[int, int],
    seed_face_id: int,
) -> Optional[Tuple[List[float], List[float], float]]:
    attributes = model_graph.nodes[int(seed_to_model[int(seed_face_id)])]
    support_key = attributes.get("surface_support_key")
    if not isinstance(support_key, (list, tuple)) or len(support_key) < 4:
        return None
    if support_key[0] != "cylinder":
        return None
    axis = normalized_vector(support_key[1])
    if axis is None:
        return None
    location = [float(value) for value in support_key[2]]
    return axis, location, float(support_key[3])


def plane_normals_form_strict_convex_hull(
    normals: Sequence[Sequence[float]],
) -> bool:
    """Check rank(N)=2 and existence of strictly positive lambda with sum(lambda_i n_i)=0."""
    unit_normals = [normalized_vector(normal) for normal in normals]
    if any(normal is None for normal in unit_normals):
        return False
    vectors = [normal for normal in unit_normals if normal is not None]
    if len(vectors) < 3:
        return False

    angular_tolerance = math.radians(
        GEOMETRIC_RELATION_ANGLE_TOLERANCE_DEGREES
    )
    first = vectors[0]
    second = next(
        (
            vector
            for vector in vectors[1:]
            if math.sqrt(vector_dot(vector_cross(first, vector), vector_cross(first, vector)))
            > math.sin(angular_tolerance)
        ),
        None,
    )
    if second is None:
        return False

    rank_plane_normal = normalized_vector(vector_cross(first, second))
    if rank_plane_normal is None:
        return False
    if any(
        abs(vector_dot(vector, rank_plane_normal)) > math.sin(angular_tolerance)
        for vector in vectors
    ):
        return False

    first_basis = first
    second_basis = normalized_vector(vector_cross(rank_plane_normal, first_basis))
    if second_basis is None:
        return False
    angles = sorted(
        math.atan2(
            vector_dot(vector, second_basis),
            vector_dot(vector, first_basis),
        )
        % (2.0 * math.pi)
        for vector in vectors
    )
    gaps = [
        angles[index + 1] - angles[index]
        for index in range(len(angles) - 1)
    ]
    gaps.append(angles[0] + 2.0 * math.pi - angles[-1])

    # For unit vectors in one two-dimensional subspace, zero belongs to the
    # relative interior of their convex hull exactly when no closed semicircle
    # contains every direction. This is equivalent to strictly positive lambdas.
    return max(gaps) < math.pi - 1.0e-8


def basic_mapping_satisfies_geometric_constraints(
    model_graph: nx.MultiGraph,
    seed_graph: nx.MultiGraph,
    seed_to_model: Dict[int, int],
) -> bool:
    constraints = seed_graph.graph.get("geometric_constraints", [])
    angular_tolerance = math.radians(
        GEOMETRIC_RELATION_ANGLE_TOLERANCE_DEGREES
    )

    for constraint in constraints:
        constraint_type = str(constraint.get("type", ""))
        seed_faces = [int(face_id) for face_id in constraint.get("faces", [])]

        if constraint_type in {"parallel", "perpendicular"}:
            if len(seed_faces) != 2:
                return False
            first = mapped_plane_normal(model_graph, seed_to_model, seed_faces[0])
            second = mapped_plane_normal(model_graph, seed_to_model, seed_faces[1])
            if first is None or second is None:
                return False
            cosine = abs(vector_dot(first, second))
            if constraint_type == "parallel" and cosine < math.cos(angular_tolerance):
                return False
            if constraint_type == "perpendicular" and cosine > math.sin(angular_tolerance):
                return False
            continue

        if constraint_type == "normal_convex_hull":
            normals = [
                mapped_plane_normal(model_graph, seed_to_model, seed_face_id)
                for seed_face_id in seed_faces
            ]
            if any(normal is None for normal in normals):
                return False
            if not plane_normals_form_strict_convex_hull(
                [normal for normal in normals if normal is not None]
            ):
                return False
            continue

        if constraint_type in {"coaxial", "equal_radius", "different_radius"}:
            if len(seed_faces) != 2:
                return False
            first = mapped_cylinder_parameters(
                model_graph, seed_to_model, seed_faces[0]
            )
            second = mapped_cylinder_parameters(
                model_graph, seed_to_model, seed_faces[1]
            )
            if first is None or second is None:
                return False
            first_axis, first_location, first_radius = first
            second_axis, second_location, second_radius = second

            if constraint_type == "coaxial":
                if abs(vector_dot(first_axis, second_axis)) < math.cos(angular_tolerance):
                    return False
                location_delta = [
                    first_location[index] - second_location[index]
                    for index in range(3)
                ]
                axis_distance = math.sqrt(
                    vector_dot(
                        vector_cross(location_delta, first_axis),
                        vector_cross(location_delta, first_axis),
                    )
                )
                if axis_distance > GEOMETRIC_RELATION_LENGTH_TOLERANCE:
                    return False
            elif constraint_type == "equal_radius":
                if abs(first_radius - second_radius) > GEOMETRIC_RELATION_RADIUS_TOLERANCE:
                    return False
            elif abs(first_radius - second_radius) <= GEOMETRIC_RELATION_RADIUS_TOLERANCE:
                return False
            continue

        raise ValueError(f"Unsupported geometric constraint: {constraint_type}")

    return True


# ============================================================
# 自定义基本型匹配
# ============================================================

def find_basic_matches(
    model_graph: nx.MultiGraph,
    seed_graph: nx.MultiGraph,
    occupied_faces: Set[int],
) -> List[Dict[int, int]]:
    seed_nodes = list(
        seed_graph.nodes
    )

    seed_nodes.sort(
        key=lambda node_id: (
            -seed_graph.degree(
                node_id
            ),
            node_id,
        )
    )

    candidate_faces_by_seed_node: Dict[
        int,
        List[int],
    ] = {}

    for seed_node_id in seed_nodes:
        required_face_type = seed_graph.nodes[
            seed_node_id
        ][
            "face_type"
        ]

        candidates = []

        for model_face_id, attributes in model_graph.nodes(
            data=True
        ):
            if model_face_id in occupied_faces:
                continue

            if attributes.get(
                "face_type"
            ) != required_face_type:
                continue

            seed_self_edges = get_multigraph_edge_bundle(
                seed_graph,
                seed_node_id,
                seed_node_id,
            )

            if seed_self_edges:
                model_self_edges = get_multigraph_edge_bundle(
                    model_graph,
                    model_face_id,
                    model_face_id,
                )

                if not edge_bundle_can_cover_seed_bundle(
                    model_self_edges,
                    seed_self_edges,
                ):
                    continue

            candidates.append(
                int(model_face_id)
            )

        candidate_faces_by_seed_node[
            seed_node_id
        ] = candidates

    results: List[
        Dict[int, int]
    ] = []

    mapping: Dict[int, int] = {}

    used_model_faces: Set[int] = set()

    def partial_mapping_is_valid(
        current_seed_node: int,
        current_model_face: int,
    ) -> bool:
        for other_seed_node, other_model_face in mapping.items():
            seed_edges = get_multigraph_edge_bundle(
                seed_graph,
                current_seed_node,
                other_seed_node,
            )

            if not seed_edges:
                continue

            model_edges = get_multigraph_edge_bundle(
                model_graph,
                current_model_face,
                other_model_face,
            )

            if not edge_bundle_can_cover_seed_bundle(
                model_edges,
                seed_edges,
            ):
                return False

        return True

    def backtrack(
        index: int,
    ) -> None:
        if len(results) >= MAX_BASIC_MATCHES_PER_FEATURE:
            return

        if index >= len(seed_nodes):
            if basic_mapping_satisfies_seed_constraints(
                model_graph=model_graph,
                seed_graph=seed_graph,
                seed_to_model=mapping,
            ):
                results.append(
                    dict(mapping)
                )

            return

        seed_node_id = seed_nodes[
            index
        ]

        for model_face_id in candidate_faces_by_seed_node[
            seed_node_id
        ]:
            if model_face_id in used_model_faces:
                continue

            if not partial_mapping_is_valid(
                seed_node_id,
                model_face_id,
            ):
                continue

            mapping[
                seed_node_id
            ] = model_face_id

            used_model_faces.add(
                model_face_id
            )

            backtrack(
                index + 1
            )

            used_model_faces.remove(
                model_face_id
            )

            del mapping[
                seed_node_id
            ]

    backtrack(
        0
    )

    return results


# ============================================================
# Basic source surface过滤
# ============================================================

def basic_source_surfaces_are_distinct(
    model_graph: nx.MultiGraph,
    seed_to_model: Dict[int, int],
) -> bool:
    """Reject a basic mapping when two seed roles share one support surface."""
    support_keys: Set[str] = set()

    for model_face_id in seed_to_model.values():
        key = support_key_string(
            model_graph.nodes[int(model_face_id)].get(
                "surface_support_key"
            )
        )

        if key in support_keys:
            return False

        support_keys.add(key)

    return True


def find_cosurface_expansion_candidates(
    model_graph: nx.MultiGraph,
    seed_to_model: Dict[int, int],
    occupied_faces: Set[int],
) -> Dict[int, Set[int]]:
    """Collect every unoccupied face strictly cosurface with each basic role."""
    support_to_faces: Dict[str, Set[int]] = defaultdict(set)

    for model_face_id, attributes in model_graph.nodes(data=True):
        support_to_faces[
            support_key_string(attributes.get("surface_support_key"))
        ].add(int(model_face_id))

    basic_faces = {
        int(model_face_id)
        for model_face_id in seed_to_model.values()
    }
    candidates_by_seed_node: Dict[int, Set[int]] = {}

    for seed_node_id, source_model_face_id in seed_to_model.items():
        source_support = support_key_string(
            model_graph.nodes[int(source_model_face_id)].get(
                "surface_support_key"
            )
        )
        candidates = set(support_to_faces.get(source_support, set()))
        candidates.difference_update(basic_faces)
        candidates.difference_update(occupied_faces)
        candidates_by_seed_node[int(seed_node_id)] = candidates

    return candidates_by_seed_node


# ============================================================
# 外部边检查
# ============================================================

def edge_satisfies_external_boundary_rule(
    attributes: Dict[str, Any],
) -> bool:
    convexity = attributes.get("convexity")
    dihedral_type = attributes.get("dihedral_type")
    return (
        convexity == "convex"
        or (
            convexity == "smooth"
            and dihedral_type != "acute"
        )
    )


def planar_feature_halfspaces(
    graph: nx.MultiGraph,
    feature_faces: Set[int],
) -> Optional[List[Tuple[List[float], float]]]:
    if not feature_faces:
        return None

    halfspaces: List[Tuple[List[float], float]] = []
    for face_id in feature_faces:
        attributes = graph.nodes[int(face_id)]
        if attributes.get("face_type") != "plane":
            return None
        normal = attributes.get("feature_space_normal")
        offset = attributes.get("feature_space_offset")
        if normal is None or offset is None:
            return None
        halfspaces.append(
            ([float(value) for value in normal], float(offset))
        )
    return halfspaces


def point_is_in_planar_feature_space(
    point: Sequence[float],
    halfspaces: Sequence[Tuple[Sequence[float], float]],
) -> bool:
    return all(
        sum(
            float(normal[index]) * float(point[index])
            for index in range(3)
        ) - float(offset)
        >= -FEATURE_SPACE_HALFSPACE_TOLERANCE
        for normal, offset in halfspaces
    )


def edge_is_on_active_feature_space_boundary(
    graph: nx.MultiGraph,
    feature_faces: Set[int],
    attributes: Dict[str, Any],
) -> bool:
    halfspaces = planar_feature_halfspaces(graph, feature_faces)
    if halfspaces is None:
        return True

    sample_points = attributes.get("edge_sample_points") or []
    if not sample_points and attributes.get("edge_point") is not None:
        sample_points = [attributes["edge_point"]]
    if not sample_points:
        return True

    return any(
        point_is_in_planar_feature_space(point, halfspaces)
        for point in sample_points
    )


def all_external_edges_satisfy_boundary_rule(
    graph: nx.MultiGraph,
    feature_faces: Set[int],
    ignored_external_faces: Optional[Set[int]] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    ignored_external_faces = ignored_external_faces or set()
    invalid_edges: List[
        Dict[str, Any]
    ] = []

    for feature_face_id in feature_faces:
        for first_face_id, second_face_id, edge_key, attributes in graph.edges(
            feature_face_id,
            keys=True,
            data=True,
        ):
            other_face_id = (
                second_face_id
                if first_face_id == feature_face_id
                else first_face_id
            )

            if other_face_id == feature_face_id:
                continue

            if other_face_id in feature_faces:
                continue

            if other_face_id in ignored_external_faces:
                continue

            if not edge_is_on_active_feature_space_boundary(
                graph,
                feature_faces,
                attributes,
            ):
                continue

            if edge_satisfies_external_boundary_rule(
                attributes
            ):
                continue

            invalid_edges.append(
                {
                    "feature_face_id": int(
                        feature_face_id
                    ),
                    "external_face_id": int(
                        other_face_id
                    ),
                    "edge_key": int(
                        edge_key
                    ),
                    "edge_type": attributes.get(
                        "edge_type"
                    ),
                    "convexity": attributes.get(
                        "convexity"
                    ),
                    "dihedral_type": attributes.get(
                        "dihedral_type"
                    ),
                    "dihedral_angle_degrees": attributes.get(
                        "dihedral_angle_degrees"
                    ),
                }
            )

    return (
        len(invalid_edges) == 0,
        invalid_edges,
    )


def candidate_face_external_edges_satisfy_boundary_rule(
    graph: nx.MultiGraph,
    candidate_face_id: int,
    temporary_feature_faces: Set[int],
    ignored_external_faces: Optional[Set[int]] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    ignored_external_faces = ignored_external_faces or set()
    invalid_edges: List[
        Dict[str, Any]
    ] = []

    for first_face_id, second_face_id, edge_key, attributes in graph.edges(
        candidate_face_id,
        keys=True,
        data=True,
    ):
        other_face_id = (
            second_face_id
            if first_face_id == candidate_face_id
            else first_face_id
        )

        if other_face_id == candidate_face_id:
            continue

        if other_face_id in temporary_feature_faces:
            continue

        if other_face_id in ignored_external_faces:
            continue

        if not edge_is_on_active_feature_space_boundary(
            graph,
            temporary_feature_faces,
            attributes,
        ):
            continue

        if edge_satisfies_external_boundary_rule(
            attributes
        ):
            continue

        invalid_edges.append(
            {
                "candidate_face_id": int(
                    candidate_face_id
                ),
                "external_face_id": int(
                    other_face_id
                ),
                "edge_key": int(
                    edge_key
                ),
                "edge_type": attributes.get(
                    "edge_type"
                ),
                "convexity": attributes.get(
                    "convexity"
                ),
                "dihedral_type": attributes.get(
                    "dihedral_type"
                ),
                "dihedral_angle_degrees": attributes.get(
                    "dihedral_angle_degrees"
                ),
            }
        )

    return (
        len(invalid_edges) == 0,
        invalid_edges,
    )


# ============================================================
# Cocurve连接验证
# ============================================================

def support_key_string(
    support_key: Any,
) -> str:
    return json.dumps(
        support_key,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def find_basic_edge_support_keys(
    model_graph: nx.MultiGraph,
    seed_graph: nx.MultiGraph,
    seed_to_model: Dict[int, int],
) -> Dict[Tuple[int, int], Set[str]]:
    result: Dict[
        Tuple[int, int],
        Set[str],
    ] = defaultdict(
        set
    )

    for first_seed_node, second_seed_node in seed_graph.edges():
        first_model_face = seed_to_model[
            first_seed_node
        ]

        second_model_face = seed_to_model[
            second_seed_node
        ]

        seed_edges = get_multigraph_edge_bundle(
            seed_graph,
            first_seed_node,
            second_seed_node,
        )

        model_edges = get_multigraph_edge_bundle(
            model_graph,
            first_model_face,
            second_model_face,
        )

        for seed_edge in seed_edges:
            for model_edge in model_edges:
                if one_model_edge_matches_seed_edge(
                    model_edge,
                    seed_edge,
                ):
                    pair_key = tuple(
                        sorted(
                            (
                                int(first_seed_node),
                                int(second_seed_node),
                            )
                        )
                    )

                    result[
                        pair_key
                    ].add(
                        support_key_string(
                            model_edge.get(
                                "curve_support_key"
                            )
                        )
                    )

    return result


def candidate_connects_to_neighbor_group_by_cocurve(
    model_graph: nx.MultiGraph,
    seed_graph: nx.MultiGraph,
    candidate_face_id: int,
    source_seed_node_id: int,
    role_groups: Dict[int, Set[int]],
    basic_curve_supports: Dict[Tuple[int, int], Set[str]],
) -> bool:
    neighbor_seed_nodes = set(
        seed_graph.neighbors(
            source_seed_node_id
        )
    )

    neighbor_seed_nodes.discard(
        source_seed_node_id
    )

    for neighbor_seed_node_id in neighbor_seed_nodes:
        pair_key = tuple(
            sorted(
                (
                    int(source_seed_node_id),
                    int(neighbor_seed_node_id),
                )
            )
        )

        allowed_curve_supports = basic_curve_supports.get(
            pair_key,
            set(),
        )

        if not allowed_curve_supports:
            return False

        neighbor_group = role_groups[
            neighbor_seed_node_id
        ]

        found_cocurve_connection = False

        for neighbor_face_id in neighbor_group:
            model_edges = get_multigraph_edge_bundle(
                model_graph,
                candidate_face_id,
                neighbor_face_id,
            )

            for model_edge in model_edges:
                curve_support_key = support_key_string(
                    model_edge.get(
                        "curve_support_key"
                    )
                )

                if curve_support_key not in allowed_curve_supports:
                    continue

                seed_edges = get_multigraph_edge_bundle(
                    seed_graph,
                    source_seed_node_id,
                    neighbor_seed_node_id,
                )

                for seed_edge in seed_edges:
                    if one_model_edge_matches_seed_edge(
                        model_edge,
                        seed_edge,
                    ):
                        found_cocurve_connection = True
                        break

                if found_cocurve_connection:
                    break

            if found_cocurve_connection:
                break

        if not found_cocurve_connection:
            return False

    return True


def candidate_is_topologically_disconnected_from_feature(
    model_graph: nx.MultiGraph,
    candidate_face_id: int,
    feature_faces: Set[int],
) -> bool:
    return not any(
        model_graph.has_edge(int(candidate_face_id), int(feature_face_id))
        for feature_face_id in feature_faces
    )


def candidate_has_same_oriented_normal_as_source(
    model_graph: nx.MultiGraph,
    candidate_face_id: int,
    source_face_id: int,
) -> bool:
    candidate_normal = normalized_vector(
        model_graph.nodes[int(candidate_face_id)].get("feature_space_normal") or []
    )
    source_normal = normalized_vector(
        model_graph.nodes[int(source_face_id)].get("feature_space_normal") or []
    )
    if candidate_normal is None or source_normal is None:
        return False
    return vector_dot(candidate_normal, source_normal) >= math.cos(
        math.radians(GEOMETRIC_RELATION_ANGLE_TOLERANCE_DEGREES)
    )


def candidate_face_sample_points(
    model_graph: nx.MultiGraph,
    candidate_face_id: int,
) -> List[List[float]]:
    points = [
        [float(value) for value in point]
        for point in model_graph.nodes[int(candidate_face_id)].get(
            "_face_sample_points", []
        )
    ]
    for _, _, attributes in model_graph.edges(
        int(candidate_face_id),
        data=True,
    ):
        edge_points = attributes.get("edge_sample_points") or []
        if not edge_points and attributes.get("edge_point") is not None:
            edge_points = [attributes["edge_point"]]
        points.extend(
            [float(value) for value in point]
            for point in edge_points
        )
    return points


def candidate_lies_inside_other_feature_halfspaces(
    model_graph: nx.MultiGraph,
    candidate_face_id: int,
    source_seed_node_id: int,
    role_groups: Dict[int, Set[int]],
) -> bool:
    other_feature_faces = {
        int(face_id)
        for seed_node_id, face_ids in role_groups.items()
        if int(seed_node_id) != int(source_seed_node_id)
        for face_id in face_ids
    }
    halfspaces = planar_feature_halfspaces(
        model_graph,
        other_feature_faces,
    )
    if halfspaces is None:
        return False
    sample_points = candidate_face_sample_points(
        model_graph,
        candidate_face_id,
    )
    if not sample_points:
        return False
    return all(
        point_is_in_planar_feature_space(point, halfspaces)
        for point in sample_points
    )


def disconnected_candidate_edges_satisfy_boundary_rule(
    model_graph: nx.MultiGraph,
    candidate_face_id: int,
) -> Tuple[bool, List[Dict[str, Any]]]:
    invalid_edges: List[Dict[str, Any]] = []
    for first_face_id, second_face_id, edge_key, attributes in model_graph.edges(
        int(candidate_face_id),
        keys=True,
        data=True,
    ):
        other_face_id = (
            int(second_face_id)
            if int(first_face_id) == int(candidate_face_id)
            else int(first_face_id)
        )
        if other_face_id == int(candidate_face_id):
            continue
        is_valid = attributes.get("convexity") == "convex" or (
            attributes.get("convexity") == "smooth"
            and bool(attributes.get("g1_continuous", False))
        )
        if is_valid:
            continue
        invalid_edges.append(
            {
                "candidate_face_id": int(candidate_face_id),
                "external_face_id": other_face_id,
                "edge_key": int(edge_key),
                "edge_type": attributes.get("edge_type"),
                "convexity": attributes.get("convexity"),
                "dihedral_type": attributes.get("dihedral_type"),
                "g1_continuous": bool(attributes.get("g1_continuous", False)),
            }
        )
    return len(invalid_edges) == 0, invalid_edges


def disconnected_cosurface_candidate_is_valid(
    model_graph: nx.MultiGraph,
    candidate_face_id: int,
    source_seed_node_id: int,
    seed_to_model: Dict[int, int],
    role_groups: Dict[int, Set[int]],
    feature_faces: Set[int],
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    source_face_id = int(seed_to_model[int(source_seed_node_id)])
    if not candidate_has_same_oriented_normal_as_source(
        model_graph,
        candidate_face_id,
        source_face_id,
    ):
        return False, "opposite_or_inconsistent_oriented_normal", []
    if not candidate_is_topologically_disconnected_from_feature(
        model_graph,
        candidate_face_id,
        feature_faces,
    ):
        return False, "connected_to_current_feature_without_valid_cocurve", []
    if not candidate_lies_inside_other_feature_halfspaces(
        model_graph,
        candidate_face_id,
        source_seed_node_id,
        role_groups,
    ):
        return False, "face_samples_outside_feature_space", []
    edges_valid, invalid_edges = (
        disconnected_candidate_edges_satisfy_boundary_rule(
            model_graph,
            candidate_face_id,
        )
    )
    if not edges_valid:
        return False, "disconnected_face_edge_violates_boundary_rule", invalid_edges
    return True, "accepted_disconnected_cosurface", []


def shape_volume(shape: Any) -> float:
    if shape is None or shape.IsNull():
        return 0.0
    properties = GProp_GProps()
    brepgprop.VolumeProperties(shape, properties)
    return abs(float(properties.Mass()))


def shape_bounding_box(
    shape: Any,
) -> Tuple[float, float, float, float, float, float]:
    bounding_box = Bnd_Box()
    brepbndlib.Add(shape, bounding_box)
    return tuple(float(value) for value in bounding_box.Get())


def common_shape(first_shape: Any, second_shape: Any) -> Optional[Any]:
    operation = BRepAlgoAPI_Common(first_shape, second_shape)
    operation.Build()
    if not operation.IsDone():
        return None
    result = operation.Shape()
    if result.IsNull():
        return None
    return result


def feature_faces_sample_bounding_box(
    model_graph: nx.MultiGraph,
    feature_faces: Set[int],
) -> Optional[Tuple[float, float, float, float, float, float]]:
    points = [
        point
        for face_id in feature_faces
        for point in candidate_face_sample_points(model_graph, int(face_id))
    ]
    if not points:
        return None
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        min(point[2] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
        max(point[2] for point in points),
    )


def build_clipped_planar_feature_space(
    halfspaces: Sequence[Tuple[Sequence[float], float]],
    feature_bounds: Tuple[float, float, float, float, float, float],
) -> Tuple[Optional[Any], Dict[str, Any]]:
    minimum_x, minimum_y, minimum_z, maximum_x, maximum_y, maximum_z = (
        feature_bounds
    )
    diagonal = math.sqrt(
        (maximum_x - minimum_x) ** 2
        + (maximum_y - minimum_y) ** 2
        + (maximum_z - minimum_z) ** 2
    )
    epsilon = max(
        FEATURE_SPACE_COLLISION_OFFSET_MINIMUM,
        diagonal * FEATURE_SPACE_COLLISION_OFFSET_RELATIVE,
    )
    numerical_padding = 2.0 * epsilon
    clipped_shape = BRepPrimAPI_MakeBox(
        gp_Pnt(
            minimum_x - numerical_padding,
            minimum_y - numerical_padding,
            minimum_z - numerical_padding,
        ),
        gp_Pnt(
            maximum_x + numerical_padding,
            maximum_y + numerical_padding,
            maximum_z + numerical_padding,
        ),
    ).Shape()

    reference_distance = max(1.0, diagonal + 4.0 * numerical_padding)
    for normal_values, offset in halfspaces:
        normal = normalized_vector(normal_values)
        if normal is None:
            return None, {
                "epsilon": epsilon,
                "feature_bounding_box_diagonal": diagonal,
            }
        shifted_offset = float(offset) + epsilon
        plane_point = [shifted_offset * component for component in normal]
        plane = gp_Pln(
            gp_Pnt(*plane_point),
            gp_Dir(*normal),
        )
        plane_face = BRepBuilderAPI_MakeFace(plane).Face()
        inside_point = gp_Pnt(
            *(plane_point[index] + normal[index] * reference_distance for index in range(3))
        )
        halfspace_shape = BRepPrimAPI_MakeHalfSpace(
            plane_face,
            inside_point,
        ).Solid()
        clipped_shape = common_shape(clipped_shape, halfspace_shape)
        if clipped_shape is None:
            return None, {
                "epsilon": epsilon,
                "feature_bounding_box_diagonal": diagonal,
            }

    return clipped_shape, {
        "epsilon": epsilon,
        "feature_bounding_box_diagonal": diagonal,
        "feature_bounding_box": [
            minimum_x,
            minimum_y,
            minimum_z,
            maximum_x,
            maximum_y,
            maximum_z,
        ],
        "numerical_padding": numerical_padding,
    }


def planar_feature_space_is_collision_free(
    part_shape: Optional[Any],
    model_graph: nx.MultiGraph,
    feature_faces: Set[int],
) -> Tuple[bool, Dict[str, Any]]:
    if part_shape is None:
        return True, {"status": "not_checked_no_part_shape"}
    halfspaces = planar_feature_halfspaces(model_graph, feature_faces)
    if halfspaces is None:
        return True, {"status": "not_applicable_nonplanar_feature"}

    feature_bounds = feature_faces_sample_bounding_box(
        model_graph,
        feature_faces,
    )
    if feature_bounds is None:
        return False, {"status": "missing_feature_face_samples"}

    clipped_space, build_record = build_clipped_planar_feature_space(
        halfspaces,
        feature_bounds,
    )
    if clipped_space is None:
        return False, {
            "status": "feature_space_boolean_build_failed",
            **build_record,
        }
    collision_shape = common_shape(clipped_space, part_shape)
    collision_volume = shape_volume(collision_shape)
    part_volume = shape_volume(part_shape)
    volume_tolerance = max(
        FEATURE_SPACE_COLLISION_VOLUME_MINIMUM_TOLERANCE,
        part_volume * FEATURE_SPACE_COLLISION_VOLUME_RELATIVE_TOLERANCE,
    )
    return collision_volume <= volume_tolerance, {
        "status": (
            "collision_free"
            if collision_volume <= volume_tolerance
            else "collision_detected"
        ),
        "collision_volume": collision_volume,
        "volume_tolerance": volume_tolerance,
        "part_volume": part_volume,
        **build_record,
    }


# ============================================================
# 逐面扩展
# ============================================================

def expand_basic_feature(
    model_graph: nx.MultiGraph,
    seed_graph: nx.MultiGraph,
    seed_to_model: Dict[int, int],
    occupied_faces: Set[int],
    cosurface_candidates_by_seed_node: Optional[
        Dict[int, Set[int]]
    ] = None,
) -> Tuple[
    Dict[int, Set[int]],
    List[Dict[str, Any]],
]:
    role_groups: Dict[
        int,
        Set[int],
    ] = {
        seed_node_id: {
            model_face_id,
        }
        for seed_node_id, model_face_id in seed_to_model.items()
    }

    feature_faces: Set[int] = set(
        seed_to_model.values()
    )

    basic_curve_supports = find_basic_edge_support_keys(
        model_graph,
        seed_graph,
        seed_to_model,
    )

    if cosurface_candidates_by_seed_node is None:
        cosurface_candidates_by_seed_node = (
            find_cosurface_expansion_candidates(
                model_graph=model_graph,
                seed_to_model=seed_to_model,
                occupied_faces=occupied_faces,
            )
        )

    pending_candidates: Dict[int, Set[int]] = {
        int(seed_node_id): set(face_ids)
        for seed_node_id, face_ids
        in cosurface_candidates_by_seed_node.items()
    }
    all_cosurface_candidates = {
        int(face_id)
        for face_ids in pending_candidates.values()
        for face_id in face_ids
    }

    expansion_records: List[
        Dict[str, Any]
    ] = []

    changed = True

    while changed:
        changed = False

        for seed_node_id in sorted(
            pending_candidates
        ):
            current_candidates = sorted(
                pending_candidates[
                    seed_node_id
                ]
            )

            accepted_this_round: Set[int] = set()

            permanently_rejected: Set[int] = set()

            for candidate_face_id in current_candidates:
                if candidate_face_id in occupied_faces:
                    permanently_rejected.add(
                        candidate_face_id
                    )

                    continue

                if candidate_face_id in feature_faces:
                    permanently_rejected.add(
                        candidate_face_id
                    )

                    continue

                cocurve_valid = (
                    candidate_connects_to_neighbor_group_by_cocurve(
                        model_graph=model_graph,
                        seed_graph=seed_graph,
                        candidate_face_id=candidate_face_id,
                        source_seed_node_id=seed_node_id,
                        role_groups=role_groups,
                        basic_curve_supports=basic_curve_supports,
                    )
                )

                expansion_mode = "connected_cocurve"

                if not cocurve_valid:
                    (
                        disconnected_valid,
                        disconnected_reason,
                        disconnected_invalid_edges,
                    ) = disconnected_cosurface_candidate_is_valid(
                        model_graph=model_graph,
                        candidate_face_id=candidate_face_id,
                        source_seed_node_id=seed_node_id,
                        seed_to_model=seed_to_model,
                        role_groups=role_groups,
                        feature_faces=feature_faces,
                    )

                    if not disconnected_valid:
                        expansion_records.append(
                            {
                                "candidate_face_id": candidate_face_id,
                                "seed_node_id": seed_node_id,
                                "accepted": False,
                                "reason": disconnected_reason,
                                "invalid_external_edges": (
                                    disconnected_invalid_edges
                                ),
                            }
                        )

                        if not RETRY_FAILED_EXPANSION_CANDIDATES:
                            permanently_rejected.add(
                                candidate_face_id
                            )

                        continue

                    expansion_mode = "disconnected_cosurface"

                temporary_feature_faces = set(
                    feature_faces
                )

                temporary_feature_faces.add(
                    candidate_face_id
                )

                external_valid = True
                invalid_external_edges = []

                if (
                    expansion_mode == "connected_cocurve"
                    and REQUIRE_EXPANDED_FACE_EXTERNAL_BOUNDARY_RULE
                ):
                    (
                        external_valid,
                        invalid_external_edges,
                    ) = candidate_face_external_edges_satisfy_boundary_rule(
                        graph=model_graph,
                        candidate_face_id=candidate_face_id,
                        temporary_feature_faces=temporary_feature_faces,
                        ignored_external_faces=(
                            all_cosurface_candidates
                            - temporary_feature_faces
                        ),
                    )

                if not external_valid:
                    expansion_records.append(
                        {
                            "candidate_face_id": candidate_face_id,
                            "seed_node_id": seed_node_id,
                            "accepted": False,
                            "reason": (
                                "candidate_external_edge_violates_boundary_rule"
                            ),
                            "invalid_external_edges": (
                                invalid_external_edges
                            ),
                        }
                    )

                    if not RETRY_FAILED_EXPANSION_CANDIDATES:
                        permanently_rejected.add(
                            candidate_face_id
                        )

                    continue

                role_groups[
                    seed_node_id
                ].add(
                    candidate_face_id
                )

                feature_faces.add(
                    candidate_face_id
                )

                accepted_this_round.add(
                    candidate_face_id
                )

                expansion_records.append(
                    {
                        "candidate_face_id": candidate_face_id,
                        "seed_node_id": seed_node_id,
                        "accepted": True,
                        "reason": (
                            "accepted_disconnected_cosurface"
                            if expansion_mode == "disconnected_cosurface"
                            else "accepted_connected_cocurve"
                        ),
                        "expansion_mode": expansion_mode,
                    }
                )

                changed = True

            pending_candidates[
                seed_node_id
            ].difference_update(
                accepted_this_round
            )

            pending_candidates[
                seed_node_id
            ].difference_update(
                permanently_rejected
            )

        if not changed:
            break

    return (
        role_groups,
        expansion_records,
    )


# ============================================================
# 单面特征
# ============================================================

def identify_single_face_feature(
    model_graph: nx.MultiGraph,
    feature: Dict[str, Any],
    occupied_faces: Set[int],
) -> List[Dict[str, Any]]:
    category_id = int(
        feature["category_id"]
    )

    category_name = str(
        feature["name"]
    )

    seed_graph = build_seed_graph(
        feature
    )

    if seed_graph.number_of_nodes() != 1:
        return []

    seed_node_id = next(
        iter(
            seed_graph.nodes
        )
    )

    required_face_type = seed_graph.nodes[
        seed_node_id
    ][
        "face_type"
    ]

    seed_self_edges = get_multigraph_edge_bundle(
        seed_graph,
        seed_node_id,
        seed_node_id,
    )

    results = []

    for face_id, attributes in model_graph.nodes(
        data=True
    ):
        face_id = int(
            face_id
        )

        if face_id in occupied_faces:
            continue

        if attributes.get(
            "face_type"
        ) != required_face_type:
            continue

        if seed_self_edges:
            model_self_edges = get_multigraph_edge_bundle(
                model_graph,
                face_id,
                face_id,
            )

            if not edge_bundle_can_cover_seed_bundle(
                model_self_edges,
                seed_self_edges,
            ):
                continue

        basic_faces = {
            face_id,
        }

        external_valid, invalid_external_edges = (
            all_external_edges_satisfy_boundary_rule(
                model_graph,
                basic_faces,
            )
        )

        if (
            REQUIRE_BASIC_EXTERNAL_BOUNDARY_RULE
            and not external_valid
        ):
            continue

        results.append(
            {
                "category_id": category_id,
                "category_name": category_name,
                "basic_face_ids": [
                    face_id,
                ],
                "face_ids": [
                    face_id,
                ],
                "role_groups": {
                    feature["nodes"][0].get(
                        "role",
                        "single_face",
                    ): [
                        face_id,
                    ],
                },
                "variant": "single_face",
                "invalid_basic_external_edges": (
                    invalid_external_edges
                ),
                "expansion_records": [],
            }
        )

    return results


# ============================================================
# 顺序特征提取
# ============================================================

def extract_features_in_priority_order(
    model_graph: nx.MultiGraph,
    features: List[Dict[str, Any]],
    part_shape: Optional[Any] = None,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, int],
    Dict[int, Optional[int]],
]:
    occupied_faces: Set[int] = set()

    predicted_seg: Dict[int, int] = {
        int(face_id): STOCK_CATEGORY_ID
        for face_id in model_graph.nodes
    }

    predicted_instance_ids: Dict[
        int,
        Optional[int],
    ] = {
        int(face_id): None
        for face_id in model_graph.nodes
    }

    instances: List[
        Dict[str, Any]
    ] = []

    next_instance_id = 0

    for feature_index, feature in enumerate(
        features,
        start=1,
    ):
        category_id = int(
            feature["category_id"]
        )

        category_name = str(
            feature["name"]
        )

        seed_graph = build_seed_graph(
            feature
        )

        print(
            (
                f"[FEATURE {feature_index}/{len(features)}] "
                f"{category_name} | "
                f"nodes={seed_graph.number_of_nodes()} | "
                f"edges={seed_graph.number_of_edges()}"
            )
        )

        if seed_graph.number_of_nodes() == 1:
            candidates = identify_single_face_feature(
                model_graph=model_graph,
                feature=feature,
                occupied_faces=occupied_faces,
            )

            for candidate in candidates:
                face_ids = set(
                    candidate["face_ids"]
                )

                if face_ids.intersection(
                    occupied_faces
                ):
                    continue

                candidate[
                    "instance_id"
                ] = next_instance_id

                candidate[
                    "priority_index"
                ] = feature_index

                instances.append(
                    candidate
                )

                for face_id in face_ids:
                    occupied_faces.add(
                        face_id
                    )

                    predicted_seg[
                        face_id
                    ] = category_id

                    predicted_instance_ids[
                        face_id
                    ] = next_instance_id

                next_instance_id += 1

            print(
                (
                    f"    accepted single-face: "
                    f"{len(candidates)}"
                )
            )

            continue

        basic_matches = find_basic_matches(
            model_graph=model_graph,
            seed_graph=seed_graph,
            occupied_faces=occupied_faces,
        )

        print(
            f"    raw basic matches: {len(basic_matches)}"
        )

        accepted_count = 0

        for seed_to_model in basic_matches:
            basic_faces = {
                int(face_id)
                for face_id in seed_to_model.values()
            }

            if basic_faces.intersection(
                occupied_faces
            ):
                continue

            if not basic_source_surfaces_are_distinct(
                model_graph=model_graph,
                seed_to_model=seed_to_model,
            ):
                continue

            if not basic_mapping_satisfies_geometric_constraints(
                model_graph=model_graph,
                seed_graph=seed_graph,
                seed_to_model=seed_to_model,
            ):
                continue

            cosurface_candidates_by_seed_node = (
                find_cosurface_expansion_candidates(
                    model_graph=model_graph,
                    seed_to_model=seed_to_model,
                    occupied_faces=occupied_faces,
                )
            )
            potential_expansion_faces = {
                int(face_id)
                for face_ids in cosurface_candidates_by_seed_node.values()
                for face_id in face_ids
            }

            basic_external_valid = True
            invalid_basic_external_edges = []

            if REQUIRE_BASIC_EXTERNAL_BOUNDARY_RULE:
                (
                    basic_external_valid,
                    invalid_basic_external_edges,
                ) = all_external_edges_satisfy_boundary_rule(
                    graph=model_graph,
                    feature_faces=basic_faces,
                    ignored_external_faces=potential_expansion_faces,
                )

            if not basic_external_valid:
                continue

            role_groups, expansion_records = (
                expand_basic_feature(
                    model_graph=model_graph,
                    seed_graph=seed_graph,
                    seed_to_model=seed_to_model,
                    occupied_faces=occupied_faces,
                    cosurface_candidates_by_seed_node=(
                        cosurface_candidates_by_seed_node
                    ),
                )
            )

            final_faces = {
                int(face_id)
                for group in role_groups.values()
                for face_id in group
            }

            if final_faces.intersection(
                occupied_faces
            ):
                continue

            final_external_valid, invalid_final_external_edges = (
                all_external_edges_satisfy_boundary_rule(
                    graph=model_graph,
                    feature_faces=final_faces,
                )
            )

            if not final_external_valid:
                continue

            (
                feature_space_collision_free,
                feature_space_collision_record,
            ) = planar_feature_space_is_collision_free(
                part_shape=part_shape,
                model_graph=model_graph,
                feature_faces=final_faces,
            )

            if not feature_space_collision_free:
                continue

            role_name_lookup = {
                int(node["id"]): node.get(
                    "role",
                    f"role_{node['id']}",
                )
                for node in feature["nodes"]
            }

            serialized_role_groups = {
                role_name_lookup[
                    seed_node_id
                ]: sorted(
                    face_ids
                )
                for seed_node_id, face_ids in role_groups.items()
            }

            instance = {
                "instance_id": next_instance_id,
                "category_id": category_id,
                "category_name": category_name,
                "priority_index": feature_index,
                "basic_face_ids": sorted(
                    basic_faces
                ),
                "face_ids": sorted(
                    final_faces
                ),
                "variant": (
                    "basic"
                    if final_faces == basic_faces
                    else "generalized"
                ),
                "seed_to_model": {
                    str(seed_node_id): int(model_face_id)
                    for seed_node_id, model_face_id in seed_to_model.items()
                },
                "role_groups": serialized_role_groups,
                "invalid_basic_external_edges": (
                    invalid_basic_external_edges
                ),
                "invalid_final_external_edges": (
                    invalid_final_external_edges
                ),
                "feature_space_collision": feature_space_collision_record,
                "expansion_records": expansion_records,
            }

            instances.append(
                instance
            )

            for face_id in final_faces:
                occupied_faces.add(
                    face_id
                )

                predicted_seg[
                    face_id
                ] = category_id

                predicted_instance_ids[
                    face_id
                ] = next_instance_id

            next_instance_id += 1

            accepted_count += 1

            if PRINT_MATCH_DETAILS:
                print(
                    (
                        f"    accepted instance "
                        f"{instance['instance_id']}: "
                        f"basic={instance['basic_face_ids']} "
                        f"final={instance['face_ids']}"
                    )
                )

        print(
            f"    accepted: {accepted_count}"
        )

    return (
        instances,
        predicted_seg,
        predicted_instance_ids,
    )


# ============================================================
# Mesh
# ============================================================

def triangulation_is_null(
    triangulation: Any,
) -> bool:
    if triangulation is None:
        return True

    try:
        return bool(
            triangulation.IsNull()
        )

    except Exception:
        return False


def get_triangulation_node_count(
    triangulation: Any,
) -> int:
    if hasattr(
        triangulation,
        "NbNodes",
    ):
        return int(
            triangulation.NbNodes()
        )

    return int(
        triangulation.Nodes().Length()
    )


def get_triangulation_triangle_count(
    triangulation: Any,
) -> int:
    if hasattr(
        triangulation,
        "NbTriangles",
    ):
        return int(
            triangulation.NbTriangles()
        )

    return int(
        triangulation.Triangles().Length()
    )


def get_triangulation_node(
    triangulation: Any,
    node_index: int,
) -> Any:
    if hasattr(
        triangulation,
        "Node",
    ):
        return triangulation.Node(
            node_index
        )

    return triangulation.Nodes().Value(
        node_index
    )


def get_triangulation_triangle(
    triangulation: Any,
    triangle_index: int,
) -> Any:
    if hasattr(
        triangulation,
        "Triangle",
    ):
        return triangulation.Triangle(
            triangle_index
        )

    return triangulation.Triangles().Value(
        triangle_index
    )


def triangulate_faces(
    shape: Any,
    faces: Sequence[Any],
) -> List[Dict[str, Any]]:
    mesher = BRepMesh_IncrementalMesh(
        shape,
        LINEAR_DEFLECTION,
        False,
        ANGULAR_DEFLECTION,
        PARALLEL_MESHING,
    )

    mesher.Perform()

    result = []

    for face_id, face in enumerate(
        faces
    ):
        location = TopLoc_Location()

        triangulation = BRep_Tool.Triangulation(
            face,
            location,
        )

        if triangulation_is_null(
            triangulation
        ):
            result.append(
                {
                    "face_id": face_id,
                    "vertices": [],
                    "triangles": [],
                }
            )

            continue

        transformation = location.Transformation()

        vertices = []

        for node_index in range(
            1,
            get_triangulation_node_count(
                triangulation
            )
            + 1,
        ):
            point = get_triangulation_node(
                triangulation,
                node_index,
            )

            point.Transform(
                transformation
            )

            vertices.append(
                [
                    float(
                        point.X()
                    ),
                    float(
                        point.Y()
                    ),
                    float(
                        point.Z()
                    ),
                ]
            )

        triangles = []

        for triangle_index in range(
            1,
            get_triangulation_triangle_count(
                triangulation
            )
            + 1,
        ):
            triangle = get_triangulation_triangle(
                triangulation,
                triangle_index,
            )

            first_node, second_node, third_node = (
                triangle.Get()
            )

            first_node -= 1
            second_node -= 1
            third_node -= 1

            if face.Orientation() == TopAbs_REVERSED:
                second_node, third_node = (
                    third_node,
                    second_node,
                )

            triangles.append(
                [
                    int(first_node),
                    int(second_node),
                    int(third_node),
                ]
            )

        result.append(
            {
                "face_id": face_id,
                "vertices": vertices,
                "triangles": triangles,
            }
        )

    return result


# ============================================================
# 序列化
# ============================================================

def serialize_fag(
    graph: nx.MultiGraph,
) -> Dict[str, Any]:
    nodes = []

    for face_id, attributes in graph.nodes(
        data=True
    ):
        record = {
            key: value
            for key, value in attributes.items()
            if not str(key).startswith("_")
        }

        record[
            "face_id"
        ] = int(face_id)

        nodes.append(
            record
        )

    edges = []

    for (
        source,
        target,
        edge_key,
        attributes,
    ) in graph.edges(
        keys=True,
        data=True,
    ):
        record = dict(
            attributes
        )

        record.update(
            {
                "source": int(source),
                "target": int(target),
                "edge_key": int(edge_key),
            }
        )

        edges.append(
            record
        )

    return {
        "graph_type": "MultiGraph",
        "nodes": nodes,
        "edges": edges,
    }


def build_cache(
    shape: Any,
    faces: List[Any],
    graph: nx.MultiGraph,
    fag_statistics: Dict[str, int],
    ground_truth: Dict[str, Any],
    instances: List[Dict[str, Any]],
    predicted_seg: Dict[int, int],
    predicted_instance_ids: Dict[int, Optional[int]],
) -> Dict[str, Any]:
    meshes = triangulate_faces(
        shape,
        faces,
    )

    face_records = []

    for face_id in range(
        len(faces)
    ):
        graph_attributes = graph.nodes[
            face_id
        ]

        true_category_id = int(
            ground_truth["seg"].get(
                face_id,
                STOCK_CATEGORY_ID,
            )
        )

        predicted_category_id = int(
            predicted_seg.get(
                face_id,
                STOCK_CATEGORY_ID,
            )
        )

        face_edges = []

        for (
            first_face_id,
            second_face_id,
            edge_key,
            edge_attributes,
        ) in graph.edges(
            face_id,
            keys=True,
            data=True,
        ):
            other_face_id = (
                second_face_id
                if first_face_id == face_id
                else first_face_id
            )

            face_edges.append(
                {
                    "edge_key": int(edge_key),
                    "other_face_id": int(
                        other_face_id
                    ),
                    "edge_type": edge_attributes.get(
                        "edge_type"
                    ),
                    "convexity": edge_attributes.get(
                        "convexity"
                    ),
                    "dihedral_type": edge_attributes.get(
                        "dihedral_type"
                    ),
                    "dihedral_angle_degrees": (
                        edge_attributes.get(
                            "dihedral_angle_degrees"
                        )
                    ),
                    "is_seam": bool(
                        edge_attributes.get(
                            "is_seam",
                            False,
                        )
                    ),
                }
            )

        mesh = meshes[
            face_id
        ]

        face_records.append(
            {
                "face_id": face_id,
                "face_type": graph_attributes.get(
                    "face_type"
                ),
                "surface_support_key": graph_attributes.get(
                    "surface_support_key"
                ),
                "seam_edge_count": graph_attributes.get(
                    "seam_edge_count"
                ),
                "ground_truth_category_id": true_category_id,
                "ground_truth_category_name": (
                    FACE_CATEGORIES[
                        true_category_id
                    ]
                    if 0 <= true_category_id < len(
                        FACE_CATEGORIES
                    )
                    else str(
                        true_category_id
                    )
                ),
                "ground_truth_instance_id": (
                    ground_truth[
                        "instance_ids"
                    ].get(
                        face_id
                    )
                ),
                "ground_truth_bottom": int(
                    ground_truth["bottom"].get(
                        face_id,
                        0,
                    )
                ),
                "predicted_category_id": predicted_category_id,
                "predicted_category_name": (
                    FACE_CATEGORIES[
                        predicted_category_id
                    ]
                    if 0 <= predicted_category_id < len(
                        FACE_CATEGORIES
                    )
                    else str(
                        predicted_category_id
                    )
                ),
                "predicted_instance_id": (
                    predicted_instance_ids.get(
                        face_id
                    )
                ),
                "ground_truth_color": CATEGORY_TO_COLOR.get(
                    true_category_id,
                    CATEGORY_TO_COLOR[
                        STOCK_CATEGORY_ID
                    ],
                ),
                "predicted_color": CATEGORY_TO_COLOR.get(
                    predicted_category_id,
                    CATEGORY_TO_COLOR[
                        STOCK_CATEGORY_ID
                    ],
                ),
                "incident_edges": face_edges,
                "vertices": mesh[
                    "vertices"
                ],
                "triangles": mesh[
                    "triangles"
                ],
            }
        )

    cache = {
        "step_path": str(
            STEP_PATH.resolve()
        ),
        "ground_truth_json_path": str(
            GROUND_TRUTH_JSON_PATH.resolve()
        ),
        "feature_seed_json_path": str(
            FEATURE_SEED_JSON_PATH.resolve()
        ),
        "part_name": ground_truth[
            "part_name"
        ],
        "face_count": len(
            faces
        ),
        "feature_instance_count": len(
            instances
        ),
        "fag_statistics": fag_statistics,
        "instances": instances,
        "fag": serialize_fag(
            graph
        ),
        "faces": face_records,
        "category_names": FACE_CATEGORIES,
        "category_colors": {
            str(category_id): color
            for category_id, color in CATEGORY_TO_COLOR.items()
        },
    }

    cache["evaluation"] = calculate_evaluation(
        faces=face_records,
        instances=instances,
        category_names=FACE_CATEGORIES,
        target_category_ids=[
            int(feature["category_id"])
            for feature in load_feature_seeds()
        ] + [STOCK_CATEGORY_ID],
    )

    return cache


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract feature-seed instances from one STEP file.",
    )
    parser.add_argument(
        "--sample",
        default=DEFAULT_SINGLE_SAMPLE_NAME,
        help="Folder and file stem under data/single.",
    )
    parser.add_argument("--step", type=Path, default=None)
    parser.add_argument("--label", type=Path, default=None)
    parser.add_argument("--seeds", type=Path, default=FEATURE_SEED_JSON_PATH)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--fag-output", type=Path, default=None)
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    global STEP_PATH
    global GROUND_TRUTH_JSON_PATH
    global FEATURE_SEED_JSON_PATH
    global OUTPUT_CACHE_PATH
    global OUTPUT_FAG_PATH
    global OUTPUT_EVALUATION_PATH

    arguments = parse_arguments()
    sample_name = str(arguments.sample)
    sample_data_directory = SINGLE_DATA_ROOT / sample_name
    sample_output_directory = SINGLE_OUTPUT_ROOT / sample_name

    STEP_PATH = (
        arguments.step
        or sample_data_directory / f"{sample_name}.step"
    ).resolve()
    GROUND_TRUTH_JSON_PATH = (
        arguments.label
        or sample_data_directory / f"{sample_name}.json"
    ).resolve()
    FEATURE_SEED_JSON_PATH = arguments.seeds.resolve()
    OUTPUT_CACHE_PATH = (
        arguments.cache
        or sample_output_directory / "cache.json"
    ).resolve()
    OUTPUT_FAG_PATH = (
        arguments.fag_output
        or sample_output_directory / "attributed_fag.json"
    ).resolve()
    OUTPUT_EVALUATION_PATH = (
        arguments.evaluation_output
        or sample_output_directory / "evaluation.json"
    ).resolve()

    if not STEP_PATH.exists():
        raise FileNotFoundError(
            f"STEP_PATH不存在: {STEP_PATH}"
        )

    if not GROUND_TRUTH_JSON_PATH.exists():
        raise FileNotFoundError(
            (
                "GROUND_TRUTH_JSON_PATH不存在: "
                f"{GROUND_TRUTH_JSON_PATH}"
            )
        )

    if not FEATURE_SEED_JSON_PATH.exists():
        raise FileNotFoundError(
            (
                "FEATURE_SEED_JSON_PATH不存在: "
                f"{FEATURE_SEED_JSON_PATH}"
            )
        )

    OUTPUT_CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] STEP: {STEP_PATH}"
    )

    print(
        f"[INFO] Label: {GROUND_TRUTH_JSON_PATH}"
    )

    print(
        f"[INFO] Seed: {FEATURE_SEED_JSON_PATH}"
    )

    shape = load_step_shape()

    ground_truth = load_ground_truth()

    features = load_feature_seeds()

    print(
        "[INFO] Feature matching order:"
    )

    for index, feature in enumerate(
        features,
        start=1,
    ):
        print(
            (
                f"    {index:02d}. "
                f"{feature['name']} | "
                f"nodes={feature['node_count']} | "
                f"edges={feature['edge_count']}"
            )
        )

    (
        graph,
        faces,
        fag_statistics,
    ) = build_attributed_fag(
        shape
    )

    print(
        (
            f"[INFO] FAG nodes={graph.number_of_nodes()}, "
            f"topological edges={graph.number_of_edges()}"
        )
    )

    print(
        (
            "[INFO] FAG attributes: "
            f"success="
            f"{fag_statistics['successful_two_face_edge_count']}, "
            f"failed="
            f"{fag_statistics['failed_two_face_edge_count']}, "
            f"seam="
            f"{fag_statistics['seam_edge_count']}, "
            f"boundary="
            f"{fag_statistics['boundary_edge_count']}"
        )
    )

    if (
        fag_statistics[
            "failed_two_face_edge_count"
        ]
        > 0
    ):
        raise RuntimeError(
            (
                "FAG中存在属性计算失败的edge，"
                "停止feature matching。"
            )
        )

    if len(faces) != len(
        ground_truth["seg"]
    ):
        print(
            (
                "[WARNING] STEP面数量与seg标签数量不同: "
                f"STEP={len(faces)}, "
                f"seg={len(ground_truth['seg'])}"
            )
        )

    (
        instances,
        predicted_seg,
        predicted_instance_ids,
    ) = extract_features_in_priority_order(
        model_graph=graph,
        features=features,
        part_shape=shape,
    )

    cache = build_cache(
        shape=shape,
        faces=faces,
        graph=graph,
        fag_statistics=fag_statistics,
        ground_truth=ground_truth,
        instances=instances,
        predicted_seg=predicted_seg,
        predicted_instance_ids=predicted_instance_ids,
    )

    with OUTPUT_CACHE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cache,
            file,
            ensure_ascii=False,
            indent=2,
        )

    attributed_fag_output = {
        "step_path": str(STEP_PATH),
        "statistics": fag_statistics,
        "fag": cache["fag"],
    }

    with OUTPUT_FAG_PATH.open("w", encoding="utf-8") as file:
        json.dump(attributed_fag_output, file, ensure_ascii=False, indent=2)

    with OUTPUT_EVALUATION_PATH.open("w", encoding="utf-8") as file:
        json.dump(cache["evaluation"], file, ensure_ascii=False, indent=2)

    print(
        (
            f"[DONE] cache saved: "
            f"{OUTPUT_CACHE_PATH}"
        )
    )

    print(
        (
            f"[DONE] recognized instances: "
            f"{len(instances)}"
        )
    )

    print(f"[DONE] attributed FAG saved: {OUTPUT_FAG_PATH}")
    print(f"[DONE] evaluation saved: {OUTPUT_EVALUATION_PATH}")
    print(
        "[DONE] face accuracy: "
        f"{cache['evaluation']['face_accuracy']:.4f}"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exception:
        print(
            f"[FAILED] {exception}"
        )

        traceback.print_exc()
