# -*- coding: utf-8 -*-
"""Batch runner for the oriented surface-space feature extractor."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import extract_batch as batch_common
import extract_one_step as legacy
import extractor_one_step_space as space_core


SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# ======================== User configuration ========================
# DATASET_ROOT must contain the "steps" and "labels" subdirectories.
DATASET_ROOT = (
    SCRIPT_DIRECTORY.parent
    / "MF_Explorer"
    / "data"
    / "mfinstseg"
)

# Change this value to control how many naturally sorted samples are processed.
SAMPLE_LIMIT = 1500

FEATURE_SEED_JSON_PATH = SCRIPT_DIRECTORY / "data" / "feature_space_seeds.json"
BATCH_OUTPUT_ROOT = SCRIPT_DIRECTORY / "output_space" / "batch"
WORKER_COUNT = min(4, max(1, os.cpu_count() or 1))
OVERWRITE_EXISTING = False
# ====================================================================


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    batch_common.write_json(path, payload)


def pipeline_signature(seed_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (Path(space_core.__file__), Path(legacy.__file__),
                 SCRIPT_DIRECTORY / "evaluation.py", seed_path):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def process_one_sample(
    step_path_string: str,
    label_path_string: str,
    seed_path_string: str,
    cache_path_string: str,
    overwrite_existing: bool,
) -> Dict[str, Any]:
    """Process one sample in an isolated worker and write its Dash cache."""
    step_path = Path(step_path_string)
    label_path = Path(label_path_string)
    seed_path = Path(seed_path_string)
    cache_path = Path(cache_path_string)
    sample_started_at = time.perf_counter()
    captured_output = io.StringIO()

    try:
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label JSON: {label_path}")

        signature = pipeline_signature(seed_path)
        cache = None
        if cache_path.exists() and not overwrite_existing:
            with cache_path.open("r", encoding="utf-8") as file:
                cache = json.load(file)
        if (cache is not None and cache.get("space_pipeline_signature") == signature
                and cache.get("space_diagnostics", {}).get("schema_version") == 2
                and "topological_edges" in cache):
            return {
                "sample_name": step_path.stem,
                "status": "reused",
                "step_path": str(step_path.resolve()),
                "label_path": str(label_path.resolve()),
                "cache_path": str(cache_path.resolve()),
                "face_count": int(cache.get("face_count", len(cache.get("faces", [])))),
                "predicted_instance_count": int(cache.get("feature_instance_count", len(cache.get("instances", [])))),
                "evaluation": cache["evaluation"],
                "elapsed_seconds": time.perf_counter() - sample_started_at,
            }

        with redirect_stdout(captured_output):
            legacy.PRINT_MATCH_DETAILS = False
            cache = space_core.build_space_cache(step_path.resolve(), label_path.resolve(), seed_path.resolve())
            cache["space_pipeline_signature"] = signature
            details_root = cache_path.parent.parent / "sample_details" / step_path.stem
            for filename, key in (("surface_attributed_graph.json", "surface_attributed_graph"),
                                  ("diagnostic_report.json", "space_diagnostics"),
                                  ("evaluation.json", "evaluation")):
                write_json(details_root / filename, cache[key])
            write_json(cache_path, cache)

        evaluation = cache["evaluation"]
        return {
            "sample_name": step_path.stem,
            "status": "success",
            "step_path": str(step_path.resolve()),
            "label_path": str(label_path.resolve()),
            "cache_path": str(cache_path.resolve()),
            "face_count": int(cache["face_count"]),
            "predicted_instance_count": len(cache["instances"]),
            "face_accuracy": evaluation["face_accuracy"],
            "exact_instance_precision": evaluation["exact_instance"]["precision"],
            "exact_instance_recall": evaluation["exact_instance"]["recall"],
            "evaluation": evaluation,
            "elapsed_seconds": time.perf_counter() - sample_started_at,
        }
    except Exception as exception:
        return {
            "sample_name": step_path.stem,
            "status": "failed",
            "step_path": str(step_path.resolve()),
            "label_path": str(label_path.resolve()),
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "diagnostic_tail": captured_output.getvalue().strip().splitlines()[-20:],
            "traceback_tail": traceback.format_exc().strip().splitlines()[-20:],
            "elapsed_seconds": time.perf_counter() - sample_started_at,
        }


def summary_payload(
    dataset_root: Path,
    requested_limit: Optional[int],
    selected_sample_count: int,
    results: List[Dict[str, Any]],
    started_at: float,
) -> Dict[str, Any]:
    evaluations = [
        result["evaluation"]
        for result in results
        if result.get("status") in {"success", "reused"}
        and isinstance(result.get("evaluation"), dict)
    ]
    aggregate = batch_common.aggregate_evaluations(
        evaluations,
        legacy.FACE_CATEGORIES,
    )
    public_results = [
        {key: value for key, value in result.items() if key != "evaluation"}
        for result in results
    ]
    return {
        "extraction_mode": "space",
        "dataset_root": str(dataset_root),
        "step_directory": str((dataset_root / "steps").resolve()),
        "label_directory": str((dataset_root / "labels").resolve()),
        "feature_seed_json_path": str(FEATURE_SEED_JSON_PATH.resolve()),
        "requested_limit": requested_limit,
        "selected_sample_count": selected_sample_count,
        "completed_sample_count": len(results),
        "successful_sample_count": sum(
            result.get("status") in {"success", "reused"}
            for result in results
        ),
        "failed_sample_count": sum(
            result.get("status") == "failed"
            for result in results
        ),
        "elapsed_seconds": time.perf_counter() - started_at,
        "evaluation": aggregate,
        "results": public_results,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run oriented surface-space feature extraction on a dataset batch."
    )
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--limit", type=int, default=SAMPLE_LIMIT)
    parser.add_argument("--output-root", type=Path, default=BATCH_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=WORKER_COUNT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_EXISTING,
        help="Recompute samples whose web cache already exists.",
    )
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
    sample_pairs = batch_common.list_sample_pairs(dataset_root, arguments.limit)
    if not sample_pairs:
        raise RuntimeError(f"No STEP samples found under {dataset_root}")

    web_cache_root.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(arguments.workers))
    print(f"Dataset root: {dataset_root}")
    print(f"Selected samples: {len(sample_pairs)}")
    print(f"Batch output: {output_root}")
    print(f"Worker processes: {workers}")

    started_at = time.perf_counter()
    results: List[Dict[str, Any]] = []
    futures = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        for step_path, label_path in sample_pairs:
            cache_path = web_cache_root / f"{step_path.stem}.json"
            future = executor.submit(
                process_one_sample,
                str(step_path.resolve()),
                str(label_path.resolve()),
                str(FEATURE_SEED_JSON_PATH.resolve()),
                str(cache_path.resolve()),
                bool(arguments.overwrite),
            )
            futures[future] = step_path.stem

        for completed_index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            status = result.get("status", "failed").upper()
            if status in {"SUCCESS", "REUSED"}:
                evaluation = result["evaluation"]
                print(
                    f"[{completed_index:03d}/{len(sample_pairs):03d}] "
                    f"{result['sample_name']} {status} "
                    f"faces={result.get('face_count', 0)} "
                    f"instances={result.get('predicted_instance_count', 0)} "
                    f"accuracy={evaluation['face_accuracy']:.4f}"
                )
            else:
                print(
                    f"[{completed_index:03d}/{len(sample_pairs):03d}] "
                    f"{result['sample_name']} FAILED "
                    f"{result.get('error_type')}: {result.get('error_message')}"
                )

            # Write a resumable checkpoint after every completed sample.
            summary = summary_payload(
                dataset_root,
                arguments.limit,
                len(sample_pairs),
                results,
                started_at,
            )
            write_json(summary_path, summary)
            if summary["evaluation"]:
                batch_common.write_per_class_csv(
                    per_class_csv_path,
                    summary["evaluation"],
                )

    results.sort(key=lambda result: batch_common.natural_sort_key(Path(result["sample_name"])))
    summary = summary_payload(
        dataset_root,
        arguments.limit,
        len(sample_pairs),
        results,
        started_at,
    )
    write_json(summary_path, summary)
    aggregate = summary["evaluation"]
    if aggregate:
        batch_common.write_per_class_csv(per_class_csv_path, aggregate)

    print(f"Summary saved: {summary_path}")
    if aggregate:
        print(f"Batch face accuracy: {aggregate['face_accuracy']:.4f}")
        print(
            "Batch machining-feature face accuracy: "
            f"{aggregate['machining_feature_face_accuracy']:.4f}"
        )
        print(
            "Batch feature detection precision/recall: "
            f"{aggregate['feature_detection']['precision']:.4f} / "
            f"{aggregate['feature_detection']['recall']:.4f}"
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
        print(f"Space batch extraction failed: {exception}")
        traceback.print_exc()
        raise
