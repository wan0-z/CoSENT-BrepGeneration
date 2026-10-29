"""Batch entry point for unlabelled STEP files."""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List

from pipeline import SEED_PATH, build_unlabelled_cache, load_reusable_cache, pipeline_signature


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
STEP_ROOT = PROJECT_ROOT / "datasets" / "eccv2026-cad-challenge-data" / "train" / "target_step"
OUTPUT_ROOT = HERE / "output"
SAMPLE_NUMBER = 50
REDO_EXIST = False
WORKERS = min(4, max(1, os.cpu_count() or 1))


def natural_key(path: Path) -> List[Any]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_steps(root: Path) -> List[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in {".step", ".stp"}),
        key=natural_key,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_one(step_string: str, cache_string: str, seed_string: str, signature: str) -> Dict[str, Any]:
    step_path, cache_path, seed_path = Path(step_string), Path(cache_string), Path(seed_string)
    started = time.perf_counter()
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            cache = build_unlabelled_cache(step_path, seed_path)
        cache["pipeline_signature"] = signature
        write_json(cache_path, cache)
        return {
            "sample_name": step_path.stem,
            "status": "success",
            "step_path": str(step_path.resolve()),
            "cache_path": str(cache_path.resolve()),
            "face_count": cache["face_count"],
            "feature_instance_count": cache["feature_instance_count"],
            "present_category_ids": cache["present_category_ids"],
            "preprocessing": cache["preprocessing"],
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "sample_name": step_path.stem,
            "status": "failed",
            "step_path": str(step_path.resolve()),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "diagnostic_tail": captured.getvalue().splitlines()[-20:],
            "traceback_tail": traceback.format_exc().splitlines()[-20:],
            "elapsed_seconds": time.perf_counter() - started,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract space features from unlabelled STEP files.")
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--sample-number", type=int, default=SAMPLE_NUMBER)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seeds", type=Path, default=SEED_PATH)
    parser.set_defaults(redo_exist=REDO_EXIST)
    parser.add_argument(
        "--redo-exist",
        action="store_true",
        dest="redo_exist",
        help="Recompute every sample in the requested prefix, including existing caches.",
    )
    parser.add_argument(
        "--no-redo-exist",
        action="store_false",
        dest="redo_exist",
        help="Keep valid caches and only process the missing samples needed to reach sample-number.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_number < 0:
        raise ValueError("sample-number must be non-negative")
    step_root = args.step_root.resolve()
    output_root = args.output_root.resolve()
    cache_root = output_root / "web_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    all_steps = list_steps(step_root)
    selected_steps = all_steps[: args.sample_number]
    signature = pipeline_signature(args.seeds.resolve())

    reused_results: List[Dict[str, Any]] = []
    pending: List[Path] = []
    for step_path in selected_steps:
        cache_path = cache_root / f"{step_path.stem}.json"
        cache = None if args.redo_exist else load_reusable_cache(cache_path, signature)
        if cache is None:
            pending.append(step_path)
        else:
            reused_results.append(
                {
                    "sample_name": step_path.stem,
                    "status": "reused",
                    "step_path": str(step_path.resolve()),
                    "cache_path": str(cache_path.resolve()),
                    "face_count": cache.get("face_count", 0),
                    "feature_instance_count": cache.get("feature_instance_count", 0),
                    "present_category_ids": cache.get("present_category_ids", []),
                    "preprocessing": cache.get("preprocessing", {}),
                    "elapsed_seconds": 0.0,
                }
            )

    print(f"STEP root: {step_root}")
    print(f"Requested total: {args.sample_number}")
    print(f"Selected total: {len(selected_steps)}")
    print(f"Reused: {len(reused_results)}")
    print(f"To process: {len(pending)}")
    print(f"redo_exist: {args.redo_exist}")

    started = time.perf_counter()
    results = list(reused_results)

    def save_summary() -> None:
        ordered = sorted(results, key=lambda item: natural_key(Path(item["sample_name"])))
        write_json(
            output_root / "summary.json",
            {
                "schema_version": 1,
                "step_root": str(step_root),
                "seed_path": str(args.seeds.resolve()),
                "sample_number": args.sample_number,
                "redo_exist": bool(args.redo_exist),
                "selected_count": len(selected_steps),
                "completed_count": len(ordered),
                "successful_count": sum(item["status"] in {"success", "reused"} for item in ordered),
                "failed_count": sum(item["status"] == "failed" for item in ordered),
                "elapsed_seconds": time.perf_counter() - started,
                "results": ordered,
            },
        )

    save_summary()
    futures = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=get_context("spawn")) as executor:
        for step_path in pending:
            cache_path = cache_root / f"{step_path.stem}.json"
            future = executor.submit(
                process_one,
                str(step_path.resolve()),
                str(cache_path.resolve()),
                str(args.seeds.resolve()),
                signature,
            )
            futures[future] = step_path.stem
        for completed_index, future in enumerate(as_completed(futures), len(reused_results) + 1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed_index}/{len(selected_steps)}] {result['sample_name']} "
                f"{result['status'].upper()} instances={result.get('feature_instance_count', 0)} "
                f"seconds={result['elapsed_seconds']:.2f}"
            )
            save_summary()
    save_summary()
    print(f"Summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()

