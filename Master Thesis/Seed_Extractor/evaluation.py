# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


STOCK_CATEGORY_ID = 24

CATEGORY_NORMALIZATION = {
    10: 8,  # slanted_through_step -> rectangular_through_step
    20: 8,  # triangular_blind_step -> rectangular_through_step
}

IGNORED_GROUND_TRUTH_CATEGORY_IDS = {
    0,   # chamfer
    23,  # round / fillet
}

def normalize_category_id(category_id: int) -> int:
    category_id = int(category_id)
    return int(CATEGORY_NORMALIZATION.get(category_id, category_id))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def category_name(category_id: int, category_names: Sequence[str]) -> str:
    if 0 <= category_id < len(category_names):
        return str(category_names[category_id])
    return str(category_id)


def seed_surface_node_counts_from_features(
    features: Sequence[Dict[str, Any]],
) -> Dict[int, int]:
    """Return the supporting-surface count required by each seed graph."""
    result: Dict[int, int] = {}
    for feature in features:
        category_id = normalize_category_id(int(feature["category_id"]))
        surface_graph = feature.get("surface_graph", {})
        nodes = surface_graph.get("nodes", feature.get("nodes", []))
        result[category_id] = max(result.get(category_id, 0), len(nodes))
    return result


def find_invalid_ground_truth_instances(
    faces: Sequence[Dict[str, Any]],
    seed_surface_node_counts: Dict[int, int],
    category_names: Sequence[str] = (),
    collision_diagnostics: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Find GT instances that must be excluded before metric calculation.

    An instance is invalid when its unique supporting-surface count differs
    from the seed graph node count, or when GT replay fails collision checking
    and the collision domain contains points sampled from a chamfer/round face
    connected to the instance.
    """
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for face in faces:
        raw_category_id = int(face["ground_truth_category_id"])
        instance_id = face.get("ground_truth_instance_id")
        if (
            instance_id is None
            or raw_category_id == STOCK_CATEGORY_ID
            or raw_category_id in IGNORED_GROUND_TRUTH_CATEGORY_IDS
        ):
            continue
        grouped[(int(instance_id), raw_category_id)].append(face)

    invalid_instances: List[Dict[str, Any]] = []
    collision_by_instance: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for diagnostic in collision_diagnostics:
        collision = diagnostic.get("collision") or diagnostic.get(
            "finite_collision_domain"
        ) or {}
        contacts = diagnostic.get("collision_modifier_face_contacts", [])
        if collision.get("status") != "collision_detected" or not contacts:
            continue
        key = (
            int(diagnostic["gt_instance_id"]),
            normalize_category_id(int(diagnostic["category_id"])),
        )
        collision_by_instance[key] = {
            "collision_volume": collision.get("collision_volume"),
            "volume_tolerance": collision.get("volume_tolerance"),
            "modifier_face_contacts": contacts,
        }

    for (instance_id, raw_category_id), instance_faces in grouped.items():
        normalized_id = normalize_category_id(raw_category_id)
        face_ids = sorted(int(face["face_id"]) for face in instance_faces)
        seed_surface_node_count = int(
            seed_surface_node_counts.get(normalized_id, 0)
        )
        supporting_surface_keys = set()
        for face in instance_faces:
            support_key = face.get("surface_support_key")
            if support_key is None:
                support_key = ["unknown_face", int(face["face_id"])]
            supporting_surface_keys.add(
                json.dumps(
                    support_key,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        supporting_surface_count = len(supporting_surface_keys)
        reasons: List[str] = []
        collision_evidence = collision_by_instance.get(
            (instance_id, normalized_id)
        )

        if (
            seed_surface_node_count
            and supporting_surface_count < seed_surface_node_count
        ):
            reasons.append("below_seed_supporting_surface_count")
        if (
            seed_surface_node_count
            and supporting_surface_count > seed_surface_node_count
        ):
            reasons.append("above_seed_supporting_surface_count")

        if collision_evidence:
            reasons.append("collision_contains_connected_chamfer_or_round")

        if reasons:
            invalid_instances.append(
                {
                    "ground_truth_instance_id": instance_id,
                    "category_id": raw_category_id,
                    "normalized_category_id": normalized_id,
                    "category_name": category_name(
                        raw_category_id, category_names
                    ),
                    "face_ids": face_ids,
                    "face_count": len(face_ids),
                    "seed_surface_node_count": seed_surface_node_count,
                    "supporting_surface_count": supporting_surface_count,
                    "reasons": reasons,
                    "collision_evidence": collision_evidence,
                }
            )

    return sorted(
        invalid_instances,
        key=lambda item: (
            int(item["category_id"]),
            int(item["ground_truth_instance_id"]),
        ),
    )


def annotate_invalid_label_faces(
    faces: Sequence[Dict[str, Any]],
    invalid_instances: Sequence[Dict[str, Any]],
) -> None:
    """Attach invalid-label metadata to face records in place."""
    reason_by_face: Dict[int, Set[str]] = defaultdict(set)
    for instance in invalid_instances:
        for face_id in instance.get("face_ids", []):
            reason_by_face[int(face_id)].update(instance.get("reasons", []))
    for face in faces:
        reasons = sorted(reason_by_face.get(int(face["face_id"]), set()))
        face["invalid_label"] = bool(reasons)
        face["invalid_label_reasons"] = reasons


def build_ground_truth_instances(
    faces: Sequence[Dict[str, Any]],
    excluded_face_ids: Optional[Set[int]] = None,
) -> Set[Tuple[int, Tuple[int, ...]]]:
    grouped_faces: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    excluded_face_ids = excluded_face_ids or set()

    for face in faces:
        if int(face["face_id"]) in excluded_face_ids:
            continue
        true_category_id = int(face["ground_truth_category_id"])
        if true_category_id in IGNORED_GROUND_TRUTH_CATEGORY_IDS:
            continue

        instance_id: Optional[int] = face.get("ground_truth_instance_id")
        if instance_id is None or true_category_id == STOCK_CATEGORY_ID:
            continue

        normalized_id = normalize_category_id(true_category_id)
        grouped_faces[(int(instance_id), normalized_id)].add(int(face["face_id"]))

    return {
        (normalized_id, tuple(sorted(face_ids)))
        for (_, normalized_id), face_ids in grouped_faces.items()
        if face_ids
    }


def build_predicted_instances(
    instances: Sequence[Dict[str, Any]],
    excluded_face_ids: Optional[Set[int]] = None,
) -> Set[Tuple[int, Tuple[int, ...]]]:
    excluded_face_ids = excluded_face_ids or set()
    return {
        (
            normalize_category_id(int(instance["category_id"])),
            tuple(sorted(int(face_id) for face_id in instance.get("face_ids", []))),
        )
        for instance in instances
        if instance.get("face_ids")
        and not excluded_face_ids.intersection(
            int(face_id) for face_id in instance.get("face_ids", [])
        )
    }


def calculate_evaluation(
    faces: Sequence[Dict[str, Any]],
    instances: Sequence[Dict[str, Any]],
    category_names: Sequence[str],
    target_category_ids: Optional[Iterable[int]] = None,
    seed_surface_node_counts: Optional[Dict[int, int]] = None,
    invalid_instances: Optional[Sequence[Dict[str, Any]]] = None,
    collision_diagnostics: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    if invalid_instances is None:
        invalid_instances = find_invalid_ground_truth_instances(
            faces,
            seed_surface_node_counts or {},
            category_names,
            collision_diagnostics,
        )
    invalid_face_ids = {
        int(face_id)
        for instance in invalid_instances
        for face_id in instance.get("face_ids", [])
    }
    if target_category_ids is None:
        target_ids = {
            normalize_category_id(int(face["ground_truth_category_id"]))
            for face in faces
            if int(face["ground_truth_category_id"])
            not in IGNORED_GROUND_TRUTH_CATEGORY_IDS
        }
        target_ids.update(
            normalize_category_id(int(face["predicted_category_id"]))
            for face in faces
        )
    else:
        target_ids = {
            normalize_category_id(int(category_id))
            for category_id in target_category_ids
        }

    target_ids.add(STOCK_CATEGORY_ID)
    label_ids = sorted(target_ids)
    label_to_index = {
        category_id: index
        for index, category_id in enumerate(label_ids)
    }
    matrix = [[0 for _ in label_ids] for _ in label_ids]
    evaluated_pairs: List[Tuple[int, int]] = []
    ignored_face_ids: List[int] = []

    for face in faces:
        true_raw = int(face["ground_truth_category_id"])
        if (
            true_raw in IGNORED_GROUND_TRUTH_CATEGORY_IDS
            or int(face["face_id"]) in invalid_face_ids
        ):
            ignored_face_ids.append(int(face["face_id"]))
            continue

        true_id = normalize_category_id(true_raw)
        predicted_id = normalize_category_id(int(face["predicted_category_id"]))

        if true_id not in label_to_index:
            continue
        if predicted_id not in label_to_index:
            label_to_index[predicted_id] = len(label_ids)
            label_ids.append(predicted_id)
            for row in matrix:
                row.append(0)
            matrix.append([0 for _ in label_ids])

        matrix[label_to_index[true_id]][label_to_index[predicted_id]] += 1
        evaluated_pairs.append((true_id, predicted_id))

    correct_count = sum(true_id == predicted_id for true_id, predicted_id in evaluated_pairs)
    feature_pairs = [pair for pair in evaluated_pairs if pair[0] != STOCK_CATEGORY_ID]
    true_feature_count = len(feature_pairs)
    predicted_feature_count = sum(
        predicted_id != STOCK_CATEGORY_ID for _, predicted_id in evaluated_pairs
    )
    detected_feature_count = sum(
        true_id != STOCK_CATEGORY_ID and predicted_id != STOCK_CATEGORY_ID
        for true_id, predicted_id in evaluated_pairs
    )
    detection_precision = safe_divide(detected_feature_count, predicted_feature_count)
    detection_recall = safe_divide(detected_feature_count, true_feature_count)
    detection_f1 = safe_divide(
        2.0 * detection_precision * detection_recall,
        detection_precision + detection_recall,
    )

    per_class: List[Dict[str, Any]] = []
    for category_id in label_ids:
        index = label_to_index[category_id]
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = safe_divide(true_positive, predicted_count)
        recall = safe_divide(true_positive, support)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        per_class.append(
            {
                "category_id": category_id,
                "category_name": category_name(category_id, category_names),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted_count": predicted_count,
            }
        )

    ground_truth_instances = build_ground_truth_instances(
        faces, invalid_face_ids
    )
    predicted_instances = build_predicted_instances(
        instances, invalid_face_ids
    )
    exact_instances = ground_truth_instances.intersection(predicted_instances)
    instance_precision = safe_divide(len(exact_instances), len(predicted_instances))
    instance_recall = safe_divide(len(exact_instances), len(ground_truth_instances))

    return {
        "evaluation_scope": (
            "target machining classes plus stock; chamfer, round, and invalid "
            "ground-truth instances ignored"
        ),
        "category_normalization": {
            str(source): target for source, target in CATEGORY_NORMALIZATION.items()
        },
        "ignored_ground_truth_category_ids": sorted(
            IGNORED_GROUND_TRUTH_CATEGORY_IDS
        ),
        "ignored_face_ids": sorted(ignored_face_ids),
        "invalid_label_face_ids": sorted(invalid_face_ids),
        "invalid_label_instance_count": len(invalid_instances),
        "invalid_label_instances": list(invalid_instances),
        "evaluated_face_count": len(evaluated_pairs),
        "correct_face_count": correct_count,
        "face_accuracy": safe_divide(correct_count, len(evaluated_pairs)),
        "machining_feature_face_accuracy": safe_divide(
            sum(true_id == predicted_id for true_id, predicted_id in feature_pairs),
            len(feature_pairs),
        ),
        "feature_detection": {
            "precision": detection_precision,
            "recall": detection_recall,
            "f1": detection_f1,
            "true_feature_face_count": true_feature_count,
            "predicted_feature_face_count": predicted_feature_count,
            "detected_feature_face_count": detected_feature_count,
        },
        "exact_instance": {
            "precision": instance_precision,
            "recall": instance_recall,
            "f1": safe_divide(
                2.0 * instance_precision * instance_recall,
                instance_precision + instance_recall,
            ),
            "ground_truth_count": len(ground_truth_instances),
            "predicted_count": len(predicted_instances),
            "correct_count": len(exact_instances),
        },
        "confusion_matrix": {
            "label_ids": label_ids,
            "label_names": [
                category_name(category_id, category_names)
                for category_id in label_ids
            ],
            "rows_are_ground_truth": True,
            "columns_are_predictions": True,
            "values": matrix,
        },
        "per_class": per_class,
    }


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Recalculate face and exact-instance metrics from a cache.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=(
            script_directory
            / "output"
            / "single"
            / "20221123_142528_0"
            / "cache.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            script_directory
            / "output"
            / "single"
            / "20221123_142528_0"
            / "evaluation.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    with arguments.cache.open("r", encoding="utf-8") as file:
        cache = json.load(file)

    report = calculate_evaluation(
        faces=cache.get("faces", []),
        instances=cache.get("instances", []),
        category_names=cache.get("category_names", []),
        target_category_ids=(
            cache.get("evaluation", {})
            .get("confusion_matrix", {})
            .get("label_ids")
        ),
        invalid_instances=cache.get("invalid_label_instances"),
        collision_diagnostics=cache.get(
            "invalid_label_collision_diagnostics", []
        ) or cache.get("space_diagnostics", {}).get("gt_instances", []),
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"Face accuracy: {report['face_accuracy']:.4f}")
    print(f"Machining-feature face accuracy: {report['machining_feature_face_accuracy']:.4f}")
    print(f"Evaluation saved: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
