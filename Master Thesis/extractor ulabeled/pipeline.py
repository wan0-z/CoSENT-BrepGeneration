"""Unlabelled STEP extraction built on Seed_Extractor's surface-space logic."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SEED_EXTRACTOR_ROOT = PROJECT_ROOT / "Seed_Extractor"
if str(SEED_EXTRACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SEED_EXTRACTOR_ROOT))

import extract_one_step as legacy  # noqa: E402
import extractor_one_step_space as space_core  # noqa: E402

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # noqa: E402
from OCC.Core.BRepCheck import BRepCheck_Analyzer  # noqa: E402
from OCC.Core.BRepGProp import brepgprop  # noqa: E402
from OCC.Core.GProp import GProp_GProps  # noqa: E402
from OCC.Core.GeomAbs import GeomAbs_Cylinder  # noqa: E402
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain  # noqa: E402
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE  # noqa: E402


SEED_PATH = SEED_EXTRACTOR_ROOT / "data" / "feature_space_seeds.json"
STOCK_CATEGORY_ID = legacy.STOCK_CATEGORY_ID
SCHEMA_VERSION = 1
MAX_RELATIVE_VOLUME_CHANGE = 1.0e-5


def shape_volume(shape: Any) -> float:
    properties = GProp_GProps()
    brepgprop.VolumeProperties(shape, properties)
    return float(properties.Mass())


def cylindrical_faces(shape: Any) -> List[Any]:
    return [
        face
        for face in legacy.unique_shapes(legacy.explore_shapes(shape, TopAbs_FACE))
        if BRepAdaptor_Surface(face, True).GetType() == GeomAbs_Cylinder
    ]


def cylinder_support(face: Any) -> Tuple[Any, ...]:
    """Use the Seed Extractor's canonical cylinder key for merge decisions."""
    return tuple(legacy.get_surface_support_key(face))


def cylinder_seam_count(faces: Sequence[Any]) -> int:
    return sum(
        legacy.is_seam_edge_on_face(edge, face)
        for face in faces
        for edge in legacy.unique_shapes(legacy.explore_shapes(face, TopAbs_EDGE))
    )


def edge_incidence(shape: Any) -> List[Tuple[Any, List[Any]]]:
    faces = legacy.unique_shapes(legacy.explore_shapes(shape, TopAbs_FACE))
    records: List[Tuple[Any, List[Any]]] = []
    for face in faces:
        for edge in legacy.unique_shapes(legacy.explore_shapes(face, TopAbs_EDGE)):
            record = next((item for item in records if legacy.same_shape(item[0], edge)), None)
            if record is None:
                records.append((edge, [face]))
            elif not any(legacy.same_shape(existing, face) for existing in record[1]):
                record[1].append(face)
    return records


def preprocess_cylinders(shape: Any) -> Tuple[Any, Dict[str, Any]]:
    """Merge adjacent pieces of the same cylindrical support and keep all other splits.

    ``ShapeUpgrade_UnifySameDomain`` normally merges planar faces as well.  Every
    edge is therefore protected except an edge shared by two faces on the same
    canonical cylindrical support.  A merged periodic cylinder is represented by
    OpenCascade as one cylindrical face with its normal seam edge.
    """
    before_faces = cylindrical_faces(shape)
    candidates: List[Any] = []
    protected: List[Any] = []
    support_candidates: Dict[str, int] = defaultdict(int)
    for edge, incident_faces in edge_incidence(shape):
        mergeable = False
        if len(incident_faces) == 2:
            first, second = incident_faces
            first_surface = BRepAdaptor_Surface(first, True)
            second_surface = BRepAdaptor_Surface(second, True)
            if (
                first_surface.GetType() == GeomAbs_Cylinder
                and second_surface.GetType() == GeomAbs_Cylinder
                and cylinder_support(first) == cylinder_support(second)
            ):
                mergeable = True
                support_candidates[json.dumps(cylinder_support(first))] += 1
        (candidates if mergeable else protected).append(edge)

    if not candidates:
        return shape, {
            "algorithm": "ShapeUpgrade_UnifySameDomain_selective_cylinders",
            "candidate_internal_edges": 0,
            "cylindrical_faces_before": len(before_faces),
            "cylindrical_faces_after": len(before_faces),
            "merged_cylindrical_face_count": 0,
            "seam_edges_before": cylinder_seam_count(before_faces),
            "seam_edges_after": cylinder_seam_count(before_faces),
            "changed": False,
            "valid": bool(BRepCheck_Analyzer(shape).IsValid()),
            "relative_volume_change": 0.0,
        }

    original_volume = shape_volume(shape)
    unifier = ShapeUpgrade_UnifySameDomain(shape, False, True, False)
    for edge in protected:
        unifier.KeepShape(edge)
    unifier.SetSafeInputMode(True)
    unifier.Build()
    result = unifier.Shape()
    if result.IsNull():
        raise RuntimeError("Cylinder unification produced a null shape")
    valid = bool(BRepCheck_Analyzer(result).IsValid())
    if not valid:
        raise RuntimeError("Cylinder unification produced an invalid B-Rep")
    result_volume = shape_volume(result)
    relative_volume_change = abs(result_volume - original_volume) / max(abs(original_volume), 1.0)
    # STEP import and same-domain reconstruction can move toleranced boundary
    # representations by a few parts per million without changing topology.
    if relative_volume_change > MAX_RELATIVE_VOLUME_CHANGE:
        raise RuntimeError(
            f"Cylinder unification changed volume by {relative_volume_change:.3e}"
        )
    after_faces = cylindrical_faces(result)
    return result, {
        "algorithm": "ShapeUpgrade_UnifySameDomain_selective_cylinders",
        "candidate_internal_edges": len(candidates),
        "candidate_support_count": len(support_candidates),
        "cylindrical_faces_before": len(before_faces),
        "cylindrical_faces_after": len(after_faces),
        "merged_cylindrical_face_count": max(0, len(before_faces) - len(after_faces)),
        "seam_edges_before": cylinder_seam_count(before_faces),
        "seam_edges_after": cylinder_seam_count(after_faces),
        "changed": len(before_faces) != len(after_faces),
        "valid": valid,
        "relative_volume_change": relative_volume_change,
    }


def build_unlabelled_cache(step_path: Path, seed_path: Path = SEED_PATH) -> Dict[str, Any]:
    """Run the existing space extractor without constructing fake labels."""
    step_path = step_path.resolve()
    seed_path = seed_path.resolve()
    legacy.STEP_PATH = step_path
    legacy.FEATURE_SEED_JSON_PATH = seed_path
    legacy.PRINT_MATCH_DETAILS = False

    source_shape = legacy.load_step_shape()
    shape, preprocessing = preprocess_cylinders(source_shape)
    features = space_core.load_seeds(seed_path)
    fag, faces, fag_statistics = legacy.build_attributed_fag(shape)
    instances, predicted_seg, predicted_instance_ids, sag = space_core.extract(
        shape, faces, fag, features
    )
    meshes = legacy.triangulate_faces(shape, faces)
    face_records = []
    for face_id, mesh in enumerate(meshes):
        graph_attributes = fag.nodes[face_id]
        category_id = int(predicted_seg.get(face_id, STOCK_CATEGORY_ID))
        face_records.append(
            {
                "face_id": face_id,
                "face_type": graph_attributes.get("face_type"),
                "surface_support_key": graph_attributes.get("surface_support_key"),
                "seam_edge_count": int(graph_attributes.get("seam_edge_count", 0)),
                "predicted_category_id": category_id,
                "predicted_category_name": legacy.FACE_CATEGORIES[category_id],
                "predicted_instance_id": predicted_instance_ids.get(face_id),
                "predicted_color": legacy.CATEGORY_TO_COLOR[category_id],
                "vertices": mesh["vertices"],
                "triangles": mesh["triangles"],
            }
        )

    category_ids = sorted({int(instance["category_id"]) for instance in instances})
    return {
        "schema_version": SCHEMA_VERSION,
        "extraction_mode": "space_unlabelled",
        "step_path": str(step_path),
        "feature_seed_json_path": str(seed_path),
        "part_name": step_path.stem,
        "face_count": len(faces),
        "feature_instance_count": len(instances),
        "present_category_ids": category_ids,
        "present_category_names": [legacy.FACE_CATEGORIES[value] for value in category_ids],
        "preprocessing": preprocessing,
        "fag_statistics": fag_statistics,
        "instances": instances,
        "fag": legacy.serialize_fag(fag),
        "surface_attributed_graph": sag,
        "topological_edges": space_core.topological_edge_mesh(
            shape, fag.graph["_space_length_scale"]
        ),
        "faces": face_records,
        "category_names": legacy.FACE_CATEGORIES,
        "category_colors": {
            str(category_id): color
            for category_id, color in legacy.CATEGORY_TO_COLOR.items()
        },
    }


def pipeline_signature(seed_path: Path = SEED_PATH) -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(space_core.__file__), Path(legacy.__file__), seed_path):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_reusable_cache(path: Path, signature: str) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("pipeline_signature") != signature
        or payload.get("extraction_mode") != "space_unlabelled"
    ):
        return None
    return payload
