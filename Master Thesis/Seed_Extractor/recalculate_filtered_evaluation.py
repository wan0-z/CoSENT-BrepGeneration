# -*- coding: utf-8 -*-
"""Recalculate cached metrics after excluding invalid GT feature instances.

This script does not run feature extraction or load STEP geometry.  It uses the
face labels, predictions, instance records, and FAG adjacency already stored in
each cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from evaluation import (
    STOCK_CATEGORY_ID,
    annotate_invalid_label_faces,
    calculate_evaluation,
    find_invalid_ground_truth_instances,
    seed_surface_node_counts_from_features,
    safe_divide,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOTS = (
    SCRIPT_DIRECTORY / "output" / "batch" / "web_cache",
    SCRIPT_DIRECTORY / "output_space" / "batch" / "web_cache",
    SCRIPT_DIRECTORY / "output" / "single",
    SCRIPT_DIRECTORY / "output_space" / "single",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def features_from_seed(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path)
    return list(payload.get("features", payload))


def seed_path_for_cache(cache: Dict[str, Any], cache_path: Path) -> Path:
    recorded = Path(cache.get("feature_seed_json_path", ""))
    if recorded.is_file():
        return recorded
    if "output_space" in cache_path.parts or cache.get("extraction_mode") == "space":
        return SCRIPT_DIRECTORY / "data" / "feature_space_seeds.json"
    return SCRIPT_DIRECTORY / "data" / "feature_seeds.json"


def space_pipeline_signature(seed_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        SCRIPT_DIRECTORY / "extractor_one_step_space.py",
        SCRIPT_DIRECTORY / "extract_one_step.py",
        SCRIPT_DIRECTORY / "evaluation.py",
        seed_path,
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compact_collision_diagnostics(
    diagnostics: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    result = []
    for diagnostic in diagnostics:
        contacts = diagnostic.get("collision_modifier_face_contacts", [])
        collision = diagnostic.get("collision") or diagnostic.get(
            "finite_collision_domain"
        ) or {}
        if collision.get("status") != "collision_detected" or not contacts:
            continue
        result.append(
            {
                "gt_instance_id": int(diagnostic["gt_instance_id"]),
                "category_id": int(diagnostic["category_id"]),
                "collision": {
                    "status": "collision_detected",
                    "collision_volume": collision.get("collision_volume"),
                    "volume_tolerance": collision.get("volume_tolerance"),
                },
                "collision_modifier_face_contacts": contacts,
            }
        )
    return result


def collision_diagnostics_for_cache(
    cache: Dict[str, Any], cache_path: Path
) -> List[Dict[str, Any]]:
    diagnostics = cache.get("invalid_label_collision_diagnostics", [])
    if diagnostics:
        return list(diagnostics)
    diagnostics = cache.get("space_diagnostics", {}).get("gt_instances", [])
    if diagnostics:
        return compact_collision_diagnostics(diagnostics)
    companion = None
    normal_batch_root = SCRIPT_DIRECTORY / "output" / "batch" / "web_cache"
    normal_single_root = SCRIPT_DIRECTORY / "output" / "single"
    if normal_batch_root in cache_path.parents:
        companion = (
            SCRIPT_DIRECTORY
            / "output_space"
            / "batch"
            / "web_cache"
            / cache_path.name
        )
    elif normal_single_root in cache_path.parents:
        companion = (
            SCRIPT_DIRECTORY
            / "output_space"
            / "single"
            / cache_path.parent.name
            / "cache.json"
        )
    if companion is None or not companion.is_file():
        return []
    companion_cache = read_json(companion)
    return compact_collision_diagnostics(
        companion_cache.get("space_diagnostics", {}).get("gt_instances", [])
    )


def recalculate_cache(cache_path: Path, dry_run: bool) -> Dict[str, Any]:
    cache = read_json(cache_path)
    seed_path = seed_path_for_cache(cache, cache_path)
    features = features_from_seed(seed_path)
    seed_surface_node_counts = seed_surface_node_counts_from_features(features)
    collision_diagnostics = collision_diagnostics_for_cache(cache, cache_path)
    invalid_instances = find_invalid_ground_truth_instances(
        cache.get("faces", []),
        seed_surface_node_counts,
        cache.get("category_names", []),
        collision_diagnostics,
    )
    annotate_invalid_label_faces(cache.get("faces", []), invalid_instances)
    cache["invalid_label_instances"] = invalid_instances
    cache["invalid_label_collision_diagnostics"] = collision_diagnostics
    cache["evaluation"] = calculate_evaluation(
        faces=cache.get("faces", []),
        instances=cache.get("instances", []),
        category_names=cache.get("category_names", []),
        target_category_ids=(
            cache.get("evaluation", {})
            .get("confusion_matrix", {})
            .get("label_ids")
        ),
        seed_surface_node_counts=seed_surface_node_counts,
        invalid_instances=invalid_instances,
        collision_diagnostics=collision_diagnostics,
    )
    if "space_pipeline_signature" in cache:
        cache["space_pipeline_signature"] = space_pipeline_signature(seed_path)
    if not dry_run:
        write_json(cache_path, cache)
        evaluation_path = cache_path.parent / "evaluation.json"
        if cache_path.name == "cache.json" and evaluation_path.exists():
            write_json(evaluation_path, cache["evaluation"])
    return cache


def aggregate_evaluations(
    evaluations: Sequence[Dict[str, Any]],
    category_names: Sequence[str],
) -> Dict[str, Any]:
    if not evaluations:
        return {}
    label_ids = sorted(
        {
            int(category_id)
            for evaluation in evaluations
            for category_id in evaluation["confusion_matrix"]["label_ids"]
        }
    )
    label_index = {category_id: index for index, category_id in enumerate(label_ids)}
    matrix = [[0 for _ in label_ids] for _ in label_ids]
    totals = {
        "evaluated_face_count": 0,
        "correct_face_count": 0,
        "true_feature_face_count": 0,
        "predicted_feature_face_count": 0,
        "detected_feature_face_count": 0,
        "ground_truth_instance_count": 0,
        "predicted_instance_count": 0,
        "correct_instance_count": 0,
        "invalid_label_instance_count": 0,
    }
    for evaluation in evaluations:
        source = evaluation["confusion_matrix"]
        source_index = {
            int(category_id): index
            for index, category_id in enumerate(source["label_ids"])
        }
        for true_id in source_index:
            for predicted_id in source_index:
                matrix[label_index[true_id]][label_index[predicted_id]] += int(
                    source["values"][source_index[true_id]][source_index[predicted_id]]
                )
        totals["evaluated_face_count"] += int(evaluation["evaluated_face_count"])
        totals["correct_face_count"] += int(evaluation["correct_face_count"])
        detection = evaluation["feature_detection"]
        for key in (
            "true_feature_face_count",
            "predicted_feature_face_count",
            "detected_feature_face_count",
        ):
            totals[key] += int(detection[key])
        exact = evaluation["exact_instance"]
        totals["ground_truth_instance_count"] += int(exact["ground_truth_count"])
        totals["predicted_instance_count"] += int(exact["predicted_count"])
        totals["correct_instance_count"] += int(exact["correct_count"])
        totals["invalid_label_instance_count"] += int(
            evaluation.get("invalid_label_instance_count", 0)
        )

    per_class = []
    for category_id in label_ids:
        index = label_index[category_id]
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = safe_divide(true_positive, predicted_count)
        recall = safe_divide(true_positive, support)
        per_class.append(
            {
                "category_id": category_id,
                "category_name": (
                    category_names[category_id]
                    if 0 <= category_id < len(category_names)
                    else str(category_id)
                ),
                "precision": precision,
                "recall": recall,
                "f1": safe_divide(2 * precision * recall, precision + recall),
                "support": support,
                "predicted_count": predicted_count,
            }
        )

    feature_precision = safe_divide(
        totals["detected_feature_face_count"],
        totals["predicted_feature_face_count"],
    )
    feature_recall = safe_divide(
        totals["detected_feature_face_count"],
        totals["true_feature_face_count"],
    )
    instance_precision = safe_divide(
        totals["correct_instance_count"], totals["predicted_instance_count"]
    )
    instance_recall = safe_divide(
        totals["correct_instance_count"], totals["ground_truth_instance_count"]
    )
    stock_index = label_index[STOCK_CATEGORY_ID]
    correct_feature_faces = sum(
        matrix[index][index]
        for index in range(len(label_ids))
        if index != stock_index
    )
    return {
        "sample_count": len(evaluations),
        "invalid_label_instance_count": totals["invalid_label_instance_count"],
        "evaluated_face_count": totals["evaluated_face_count"],
        "correct_face_count": totals["correct_face_count"],
        "face_accuracy": safe_divide(
            totals["correct_face_count"], totals["evaluated_face_count"]
        ),
        "machining_feature_face_accuracy": safe_divide(
            correct_feature_faces, totals["true_feature_face_count"]
        ),
        "feature_detection": {
            "precision": feature_precision,
            "recall": feature_recall,
            "f1": safe_divide(
                2 * feature_precision * feature_recall,
                feature_precision + feature_recall,
            ),
            "true_feature_face_count": totals["true_feature_face_count"],
            "predicted_feature_face_count": totals["predicted_feature_face_count"],
            "detected_feature_face_count": totals["detected_feature_face_count"],
        },
        "exact_instance": {
            "precision": instance_precision,
            "recall": instance_recall,
            "f1": safe_divide(
                2 * instance_precision * instance_recall,
                instance_precision + instance_recall,
            ),
            "ground_truth_count": totals["ground_truth_instance_count"],
            "predicted_count": totals["predicted_instance_count"],
            "correct_count": totals["correct_instance_count"],
        },
        "confusion_matrix": {
            "label_ids": label_ids,
            "label_names": [
                category_names[item] if item < len(category_names) else str(item)
                for item in label_ids
            ],
            "rows_are_ground_truth": True,
            "columns_are_predictions": True,
            "values": matrix,
        },
        "per_class": per_class,
    }


def write_per_class_csv(path: Path, evaluation: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "category_id",
                "category_name",
                "precision",
                "recall",
                "f1",
                "support",
                "predicted_count",
            ),
        )
        writer.writeheader()
        writer.writerows(evaluation.get("per_class", []))


def update_batch_summary(
    cache_root: Path,
    caches: Sequence[Dict[str, Any]],
    cache_paths: Sequence[Path],
    dry_run: bool,
) -> None:
    summary_path = cache_root.parent / "summary.json"
    if not summary_path.exists() or not caches:
        return
    summary = read_json(summary_path)
    cache_by_name = {
        path.stem: cache for path, cache in zip(cache_paths, caches)
    }
    for result in summary.get("results", []):
        cache = cache_by_name.get(str(result.get("sample_name")))
        if cache is None:
            continue
        evaluation = cache["evaluation"]
        result["face_accuracy"] = evaluation["face_accuracy"]
        result["exact_instance_precision"] = evaluation["exact_instance"]["precision"]
        result["exact_instance_recall"] = evaluation["exact_instance"]["recall"]
    evaluation = aggregate_evaluations(
        [cache["evaluation"] for cache in caches],
        caches[0].get("category_names", []),
    )
    summary["evaluation"] = evaluation
    summary["evaluation_filter"] = "invalid_ground_truth_instances_excluded"
    if not dry_run:
        write_json(summary_path, summary)
        write_per_class_csv(cache_root.parent / "per_class_metrics.csv", evaluation)


def cache_paths_under(root: Path) -> List[Path]:
    if not root.exists():
        return []
    if root.name == "web_cache":
        return sorted(root.glob("*.json"))
    return sorted(root.rglob("cache.json"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate cached accuracy after excluding invalid GT labels; "
            "feature extraction is not rerun."
        )
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        action="append",
        dest="cache_roots",
        help="Cache root to update; may be supplied more than once.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    roots: Iterable[Path] = arguments.cache_roots or DEFAULT_CACHE_ROOTS
    total_caches = 0
    total_invalid = 0
    for root in roots:
        root = root.resolve()
        paths = cache_paths_under(root)
        caches = []
        for cache_path in paths:
            cache = recalculate_cache(cache_path, arguments.dry_run)
            caches.append(cache)
            total_caches += 1
            total_invalid += len(cache.get("invalid_label_instances", []))
        update_batch_summary(root, caches, paths, arguments.dry_run)
        if paths:
            print(f"{root}: updated {len(paths)} caches")
    action = "would update" if arguments.dry_run else "updated"
    print(
        f"{action} {total_caches} caches; "
        f"found {total_invalid} invalid-label instances"
    )


if __name__ == "__main__":
    main()
