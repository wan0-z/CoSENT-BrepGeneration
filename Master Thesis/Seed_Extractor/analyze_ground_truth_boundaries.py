# -*- coding: utf-8 -*-
"""Analyze external-edge attributes of ground-truth machining-feature instances."""

from __future__ import annotations

import argparse
import csv
import json
import time

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

import extract_one_step as core
from extract_batch import DATASET_ROOT, SAMPLE_LIMIT, list_sample_pairs


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIRECTORY / "output" / "batch" / "ground_truth_boundary_analysis"

# Chamfer and round/fillet are outside the current seed-extraction target set.
# Slanted through step and triangular blind step remain in this analysis because
# evaluation maps them to rectangular through step.
IGNORED_CATEGORY_IDS = {0, 23, 24}


EdgeRule = Callable[[Dict[str, Any]], bool]


EDGE_RULES: Dict[str, EdgeRule] = {
    "convex_only": lambda edge: edge["convexity"] == "convex",
    "convex_or_smooth": lambda edge: edge["convexity"] in {"convex", "smooth"},
    "non_acute": lambda edge: edge["dihedral_type"] != "acute",
    "convex_any_dihedral_or_smooth_non_acute": lambda edge: (
        edge["convexity"] == "convex"
        or (
            edge["convexity"] == "smooth"
            and edge["dihedral_type"] != "acute"
        )
    ),
    "convex_and_non_acute": lambda edge: (
        edge["convexity"] == "convex" and edge["dihedral_type"] != "acute"
    ),
    "convex_or_smooth_and_non_acute": lambda edge: (
        edge["convexity"] in {"convex", "smooth"}
        and edge["dihedral_type"] != "acute"
    ),
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def category_name(category_id: int) -> str:
    if 0 <= category_id < len(core.FACE_CATEGORIES):
        return core.FACE_CATEGORIES[category_id]
    return str(category_id)


def configure_core(step_path: Path, label_path: Path) -> None:
    core.STEP_PATH = step_path.resolve()
    core.GROUND_TRUTH_JSON_PATH = label_path.resolve()


def build_ground_truth_groups(
    ground_truth: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    grouped_faces: Dict[int, Set[int]] = defaultdict(set)
    grouped_categories: Dict[int, Counter[int]] = defaultdict(Counter)

    for face_id, instance_id in ground_truth["instance_ids"].items():
        if instance_id is None:
            continue
        category_id = int(ground_truth["seg"].get(int(face_id), 24))
        if category_id in IGNORED_CATEGORY_IDS:
            continue
        grouped_faces[int(instance_id)].add(int(face_id))
        grouped_categories[int(instance_id)][category_id] += 1

    result: Dict[int, Dict[str, Any]] = {}
    for instance_id, face_ids in grouped_faces.items():
        category_counts = grouped_categories[instance_id]
        category_id = int(category_counts.most_common(1)[0][0])
        result[instance_id] = {
            "instance_id": instance_id,
            "category_id": category_id,
            "category_name": category_name(category_id),
            "category_ids_present": sorted(category_counts),
            "face_ids": sorted(face_ids),
        }
    return result


def collect_external_edges(
    graph: nx.MultiGraph,
    feature_faces: Set[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int, int]] = set()

    for feature_face_id in sorted(feature_faces):
        for first_face_id, second_face_id, edge_key, attributes in graph.edges(
            feature_face_id,
            keys=True,
            data=True,
        ):
            other_face_id = (
                int(second_face_id)
                if int(first_face_id) == feature_face_id
                else int(first_face_id)
            )
            if other_face_id == feature_face_id or other_face_id in feature_faces:
                continue

            identity = (
                min(feature_face_id, other_face_id),
                max(feature_face_id, other_face_id),
                int(edge_key),
            )
            if identity in seen:
                continue
            seen.add(identity)
            records.append({
                "feature_face_id": feature_face_id,
                "external_face_id": other_face_id,
                "edge_key": int(edge_key),
                "edge_type": str(attributes.get("edge_type", "unknown")),
                "convexity": str(attributes.get("convexity", "unknown")),
                "dihedral_type": str(attributes.get("dihedral_type", "unknown")),
                "dihedral_angle_degrees": attributes.get("dihedral_angle_degrees"),
                "is_seam": bool(attributes.get("is_seam", False)),
            })
    return records


def rule_results(external_edges: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    return {
        rule_name: bool(external_edges) and all(rule(edge) for edge in external_edges)
        for rule_name, rule in EDGE_RULES.items()
    }


def analyze_sample(step_path: Path, label_path: Path) -> Dict[str, Any]:
    configure_core(step_path, label_path)
    shape = core.load_step_shape()
    ground_truth = core.load_ground_truth()
    graph, _, fag_statistics = core.build_attributed_fag(shape)
    instances = []

    for instance in build_ground_truth_groups(ground_truth).values():
        external_edges = collect_external_edges(graph, set(instance["face_ids"]))
        instances.append({
            **instance,
            "external_edge_count": len(external_edges),
            "external_edges": external_edges,
            "rules": rule_results(external_edges),
        })

    return {
        "sample_name": step_path.stem,
        "step_path": str(step_path.resolve()),
        "label_path": str(label_path.resolve()),
        "fag_statistics": fag_statistics,
        "instance_count": len(instances),
        "instances": instances,
    }


def aggregate_results(sample_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    instances = [
        instance
        for sample in sample_results
        for instance in sample["instances"]
    ]
    boundary_instances = [
        instance for instance in instances if instance["external_edge_count"] > 0
    ]
    boundaryless_instances = len(instances) - len(boundary_instances)
    edges = [
        edge
        for instance in boundary_instances
        for edge in instance["external_edges"]
    ]

    combination_counts = Counter(
        (edge["convexity"], edge["dihedral_type"])
        for edge in edges
    )
    convexity_counts = Counter(edge["convexity"] for edge in edges)
    dihedral_counts = Counter(edge["dihedral_type"] for edge in edges)

    rule_rows = []
    for rule_name, rule in EDGE_RULES.items():
        accepted_edges = sum(rule(edge) for edge in edges)
        accepted_instances = sum(
            instance["rules"][rule_name] for instance in boundary_instances
        )
        rule_rows.append({
            "rule": rule_name,
            "accepted_edge_count": accepted_edges,
            "total_edge_count": len(edges),
            "edge_coverage": accepted_edges / len(edges) if edges else 0.0,
            "accepted_instance_count": accepted_instances,
            "total_instance_count": len(boundary_instances),
            "instance_coverage": (
                accepted_instances / len(boundary_instances)
                if boundary_instances
                else 0.0
            ),
        })

    category_instances: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for instance in boundary_instances:
        category_instances[int(instance["category_id"])].append(instance)

    per_category_rows = []
    for category_id in sorted(category_instances):
        category_records = category_instances[category_id]
        category_edges = [
            edge for instance in category_records for edge in instance["external_edges"]
        ]
        for rule_name, rule in EDGE_RULES.items():
            accepted_instances = sum(
                instance["rules"][rule_name] for instance in category_records
            )
            per_category_rows.append({
                "category_id": category_id,
                "category_name": category_name(category_id),
                "rule": rule_name,
                "instance_count": len(category_records),
                "accepted_instance_count": accepted_instances,
                "instance_coverage": accepted_instances / len(category_records),
                "edge_count": len(category_edges),
                "edge_coverage": (
                    sum(rule(edge) for edge in category_edges) / len(category_edges)
                    if category_edges
                    else 0.0
                ),
            })

    combination_rows = []
    for (convexity, dihedral_type), count in sorted(
        combination_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        combination_rows.append({
            "convexity": convexity,
            "dihedral_type": dihedral_type,
            "edge_count": count,
            "edge_fraction": count / len(edges) if edges else 0.0,
        })

    recommended_rule = max(
        rule_rows,
        key=lambda row: (row["instance_coverage"], row["edge_coverage"]),
    )

    return {
        "sample_count": len(sample_results),
        "instance_count": len(instances),
        "instances_with_external_edges": len(boundary_instances),
        "instances_without_external_edges": boundaryless_instances,
        "external_edge_count": len(edges),
        "convexity_counts": dict(sorted(convexity_counts.items())),
        "dihedral_type_counts": dict(sorted(dihedral_counts.items())),
        "edge_combinations": combination_rows,
        "rule_coverage": rule_rows,
        "recommended_rule_by_ground_truth_coverage": recommended_rule["rule"],
        "interpretation_note": (
            "The recommendation maximizes coverage of ground-truth instances only. "
            "It does not measure rejection of false candidate subgraphs; extraction "
            "precision must be evaluated separately."
        ),
        "per_category_rule_coverage": per_category_rows,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ground-truth feature-instance external-edge attributes."
    )
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--limit", type=int, default=SAMPLE_LIMIT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    sample_pairs = list_sample_pairs(arguments.data_root.resolve(), arguments.limit)
    output_root = arguments.output_root.resolve()
    started_at = time.perf_counter()
    sample_results = []
    failures = []

    print(f"Dataset root: {arguments.data_root.resolve()}")
    print(f"Selected samples: {len(sample_pairs)}")
    print(f"Output root: {output_root}")

    for index, (step_path, label_path) in enumerate(sample_pairs, start=1):
        try:
            result = analyze_sample(step_path, label_path)
            sample_results.append(result)
            print(
                f"[{index:03d}/{len(sample_pairs):03d}] {step_path.stem} "
                f"OK instances={result['instance_count']}"
            )
        except Exception as exception:
            failures.append({
                "sample_name": step_path.stem,
                "error_type": type(exception).__name__,
                "error_message": str(exception),
            })
            print(
                f"[{index:03d}/{len(sample_pairs):03d}] {step_path.stem} "
                f"FAILED {type(exception).__name__}: {exception}"
            )

    aggregate = aggregate_results(sample_results)
    summary = {
        "dataset_root": str(arguments.data_root.resolve()),
        "requested_limit": arguments.limit,
        "selected_sample_count": len(sample_pairs),
        "successful_sample_count": len(sample_results),
        "failed_sample_count": len(failures),
        "ignored_category_ids": sorted(IGNORED_CATEGORY_IDS),
        "ignored_category_names": [
            category_name(category_id) for category_id in sorted(IGNORED_CATEGORY_IDS)
        ],
        "elapsed_seconds": time.perf_counter() - started_at,
        "failures": failures,
        **aggregate,
    }

    write_json(output_root / "summary.json", summary)
    write_json(output_root / "sample_instance_records.json", sample_results)
    write_csv(output_root / "edge_combinations.csv", aggregate["edge_combinations"])
    write_csv(output_root / "rule_coverage.csv", aggregate["rule_coverage"])
    write_csv(
        output_root / "per_category_rule_coverage.csv",
        aggregate["per_category_rule_coverage"],
    )

    print(f"Successful samples: {len(sample_results)}/{len(sample_pairs)}")
    print(f"Ground-truth instances: {aggregate['instance_count']}")
    print(f"External edges: {aggregate['external_edge_count']}")
    for row in aggregate["rule_coverage"]:
        print(
            f"{row['rule']}: edge={row['edge_coverage']:.4f} "
            f"instance={row['instance_coverage']:.4f}"
        )
    print(f"Summary saved: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
