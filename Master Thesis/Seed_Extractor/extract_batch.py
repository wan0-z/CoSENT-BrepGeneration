# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
import traceback

from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import extract_one_step as core

from evaluation import STOCK_CATEGORY_ID, safe_divide


SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# ======================== User configuration ========================
# DATASET_ROOT must contain the "steps" and "labels" subdirectories.
DATASET_ROOT = (
    SCRIPT_DIRECTORY.parent
    / "MF_Explorer"
    / "data"
    / "mfinstseg"
)

# Change this value to control how many naturally-sorted samples are processed.
SAMPLE_LIMIT = 200

FEATURE_SEED_JSON_PATH = SCRIPT_DIRECTORY / "data" / "feature_seeds.json"
BATCH_OUTPUT_ROOT = SCRIPT_DIRECTORY / "output" / "batch"
OVERWRITE_EXISTING = True
# ====================================================================

STEP_EXTENSIONS = {".step", ".stp"}


def natural_sort_key(path: Path) -> List[Any]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def list_sample_pairs(
    dataset_root: Path,
    sample_limit: Optional[int],
) -> List[Tuple[Path, Path]]:
    step_directory = dataset_root / "steps"
    label_directory = dataset_root / "labels"

    if not step_directory.is_dir():
        raise FileNotFoundError(f"STEP directory does not exist: {step_directory}")
    if not label_directory.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_directory}")

    step_paths = sorted(
        (
            path
            for path in step_directory.iterdir()
            if path.is_file() and path.suffix.lower() in STEP_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if sample_limit is not None:
        step_paths = step_paths[: max(0, int(sample_limit))]

    return [
        (step_path, label_directory / f"{step_path.stem}.json")
        for step_path in step_paths
    ]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def configure_core_paths(step_path: Path, label_path: Path) -> None:
    core.STEP_PATH = step_path.resolve()
    core.GROUND_TRUTH_JSON_PATH = label_path.resolve()
    core.FEATURE_SEED_JSON_PATH = FEATURE_SEED_JSON_PATH.resolve()


def process_one_sample(
    step_path: Path,
    label_path: Path,
    features: List[Dict[str, Any]],
    cache_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label JSON: {label_path}")

    if cache_path.exists() and not OVERWRITE_EXISTING:
        with cache_path.open("r", encoding="utf-8") as file:
            cache = json.load(file)
        return cache, {
            "sample_name": step_path.stem,
            "status": "reused",
            "cache_path": str(cache_path.resolve()),
        }

    configure_core_paths(step_path, label_path)
    shape = core.load_step_shape()
    ground_truth = core.load_ground_truth()
    graph, faces, fag_statistics = core.build_attributed_fag(shape)

    if fag_statistics["failed_two_face_edge_count"] > 0:
        raise RuntimeError(
            "Attributed FAG contains failed edge attributes; matching was stopped."
        )

    instances, predicted_seg, predicted_instance_ids = (
        core.extract_features_in_priority_order(
            model_graph=graph,
            features=features,
            part_shape=shape,
        )
    )
    cache = core.build_cache(
        shape=shape,
        faces=faces,
        graph=graph,
        fag_statistics=fag_statistics,
        ground_truth=ground_truth,
        instances=instances,
        predicted_seg=predicted_seg,
        predicted_instance_ids=predicted_instance_ids,
    )
    write_json(cache_path, cache)

    evaluation = cache["evaluation"]
    return cache, {
        "sample_name": step_path.stem,
        "status": "success",
        "step_path": str(step_path.resolve()),
        "label_path": str(label_path.resolve()),
        "cache_path": str(cache_path.resolve()),
        "face_count": len(faces),
        "predicted_instance_count": len(instances),
        "face_accuracy": evaluation["face_accuracy"],
        "exact_instance_precision": evaluation["exact_instance"]["precision"],
        "exact_instance_recall": evaluation["exact_instance"]["recall"],
    }


def aggregate_evaluations(
    evaluations: Sequence[Dict[str, Any]],
    category_names: Sequence[str],
) -> Dict[str, Any]:
    if not evaluations:
        return {}

    label_ids = list(evaluations[0]["confusion_matrix"]["label_ids"])
    label_index = {category_id: index for index, category_id in enumerate(label_ids)}
    matrix = [[0 for _ in label_ids] for _ in label_ids]

    evaluated_face_count = 0
    correct_face_count = 0
    true_feature_face_count = 0
    predicted_feature_face_count = 0
    detected_feature_face_count = 0
    ground_truth_instance_count = 0
    predicted_instance_count = 0
    correct_instance_count = 0

    for evaluation in evaluations:
        source_matrix = evaluation["confusion_matrix"]
        source_ids = source_matrix["label_ids"]
        source_index = {
            category_id: index
            for index, category_id in enumerate(source_ids)
        }

        for true_id in label_ids:
            for predicted_id in label_ids:
                matrix[label_index[true_id]][label_index[predicted_id]] += int(
                    source_matrix["values"][source_index[true_id]][source_index[predicted_id]]
                )

        evaluated_face_count += int(evaluation["evaluated_face_count"])
        correct_face_count += int(evaluation["correct_face_count"])
        detection = evaluation["feature_detection"]
        true_feature_face_count += int(detection["true_feature_face_count"])
        predicted_feature_face_count += int(detection["predicted_feature_face_count"])
        detected_feature_face_count += int(detection["detected_feature_face_count"])
        exact_instance = evaluation["exact_instance"]
        ground_truth_instance_count += int(exact_instance["ground_truth_count"])
        predicted_instance_count += int(exact_instance["predicted_count"])
        correct_instance_count += int(exact_instance["correct_count"])

    stock_index = label_index[STOCK_CATEGORY_ID]
    correct_feature_face_count = sum(
        matrix[index][index]
        for index in range(len(label_ids))
        if index != stock_index
    )
    feature_precision = safe_divide(
        detected_feature_face_count,
        predicted_feature_face_count,
    )
    feature_recall = safe_divide(
        detected_feature_face_count,
        true_feature_face_count,
    )
    instance_precision = safe_divide(
        correct_instance_count,
        predicted_instance_count,
    )
    instance_recall = safe_divide(
        correct_instance_count,
        ground_truth_instance_count,
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
                "category_name": category_names[category_id],
                "precision": precision,
                "recall": recall,
                "f1": safe_divide(2.0 * precision * recall, precision + recall),
                "support": support,
                "predicted_count": predicted_count,
            }
        )

    return {
        "sample_count": len(evaluations),
        "evaluated_face_count": evaluated_face_count,
        "correct_face_count": correct_face_count,
        "face_accuracy": safe_divide(correct_face_count, evaluated_face_count),
        "machining_feature_face_accuracy": safe_divide(
            correct_feature_face_count,
            true_feature_face_count,
        ),
        "feature_detection": {
            "precision": feature_precision,
            "recall": feature_recall,
            "f1": safe_divide(
                2.0 * feature_precision * feature_recall,
                feature_precision + feature_recall,
            ),
            "true_feature_face_count": true_feature_face_count,
            "predicted_feature_face_count": predicted_feature_face_count,
            "detected_feature_face_count": detected_feature_face_count,
        },
        "exact_instance": {
            "precision": instance_precision,
            "recall": instance_recall,
            "f1": safe_divide(
                2.0 * instance_precision * instance_recall,
                instance_precision + instance_recall,
            ),
            "ground_truth_count": ground_truth_instance_count,
            "predicted_count": predicted_instance_count,
            "correct_count": correct_instance_count,
        },
        "confusion_matrix": {
            "label_ids": label_ids,
            "label_names": [category_names[category_id] for category_id in label_ids],
            "rows_are_ground_truth": True,
            "columns_are_predictions": True,
            "values": matrix,
        },
        "per_class": per_class,
    }


def write_per_class_csv(path: Path, evaluation: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "category_id",
                "category_name",
                "precision",
                "recall",
                "f1",
                "support",
                "predicted_count",
            ],
        )
        writer.writeheader()
        writer.writerows(evaluation.get("per_class", []))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current feature-seed extraction logic on a dataset batch."
    )
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--limit", type=int, default=SAMPLE_LIMIT)
    parser.add_argument("--output-root", type=Path, default=BATCH_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dataset_root = arguments.data_root.resolve()
    output_root = arguments.output_root.resolve()
    web_cache_root = output_root / "web_cache"
    summary_path = output_root / "summary.json"
    per_class_csv_path = output_root / "per_class_metrics.csv"

    if not FEATURE_SEED_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Feature seed JSON does not exist: {FEATURE_SEED_JSON_PATH}"
        )

    sample_pairs = list_sample_pairs(dataset_root, arguments.limit)
    if not sample_pairs:
        raise RuntimeError(f"No STEP samples found under {dataset_root}")

    core.FEATURE_SEED_JSON_PATH = FEATURE_SEED_JSON_PATH.resolve()
    core.PRINT_MATCH_DETAILS = False
    features = core.load_feature_seeds()
    web_cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {dataset_root}")
    print(f"Selected samples: {len(sample_pairs)}")
    print(f"Batch output: {output_root}")

    started_at = time.perf_counter()
    results: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []

    for index, (step_path, label_path) in enumerate(sample_pairs, start=1):
        sample_started_at = time.perf_counter()
        cache_path = web_cache_root / f"{step_path.stem}.json"
        captured_output = io.StringIO()

        try:
            with redirect_stdout(captured_output):
                cache, result = process_one_sample(
                    step_path=step_path,
                    label_path=label_path,
                    features=features,
                    cache_path=cache_path,
                )
            evaluations.append(cache["evaluation"])
            result["elapsed_seconds"] = time.perf_counter() - sample_started_at
            results.append(result)
            print(
                f"[{index:03d}/{len(sample_pairs):03d}] {step_path.stem} "
                f"OK faces={result.get('face_count', cache.get('face_count'))} "
                f"instances={cache.get('feature_instance_count')} "
                f"accuracy={cache['evaluation']['face_accuracy']:.4f}"
            )
        except Exception as exception:
            diagnostic_lines = captured_output.getvalue().strip().splitlines()
            result = {
                "sample_name": step_path.stem,
                "status": "failed",
                "step_path": str(step_path.resolve()),
                "label_path": str(label_path.resolve()),
                "error_type": type(exception).__name__,
                "error_message": str(exception),
                "diagnostic_tail": diagnostic_lines[-20:],
                "elapsed_seconds": time.perf_counter() - sample_started_at,
            }
            results.append(result)
            print(
                f"[{index:03d}/{len(sample_pairs):03d}] {step_path.stem} "
                f"FAILED {type(exception).__name__}: {exception}"
            )

    aggregate = aggregate_evaluations(evaluations, core.FACE_CATEGORIES)
    summary = {
        "dataset_root": str(dataset_root),
        "step_directory": str((dataset_root / "steps").resolve()),
        "label_directory": str((dataset_root / "labels").resolve()),
        "feature_seed_json_path": str(FEATURE_SEED_JSON_PATH.resolve()),
        "requested_limit": arguments.limit,
        "selected_sample_count": len(sample_pairs),
        "successful_sample_count": sum(
            result["status"] in {"success", "reused"}
            for result in results
        ),
        "failed_sample_count": sum(
            result["status"] == "failed"
            for result in results
        ),
        "elapsed_seconds": time.perf_counter() - started_at,
        "evaluation": aggregate,
        "results": results,
    }
    write_json(summary_path, summary)
    write_per_class_csv(per_class_csv_path, aggregate)

    print(f"Summary saved: {summary_path}")
    if aggregate:
        print(f"Batch face accuracy: {aggregate['face_accuracy']:.4f}")
        print(
            "Batch machining-feature face accuracy: "
            f"{aggregate['machining_feature_face_accuracy']:.4f}"
        )
        print(
            "Batch exact-instance precision/recall: "
            f"{aggregate['exact_instance']['precision']:.4f} / "
            f"{aggregate['exact_instance']['recall']:.4f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        print(f"Batch extraction failed: {exception}")
        traceback.print_exc()
        raise
