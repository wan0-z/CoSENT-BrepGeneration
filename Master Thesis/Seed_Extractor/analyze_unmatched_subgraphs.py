# -*- coding: utf-8 -*-
"""Classify missed ground-truth instances by seed-node and seed-edge sufficiency."""

from __future__ import annotations

import json

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

import extract_one_step as core
from evaluation import normalize_category_id


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_DIRECTORY / "output" / "batch" / "web_cache"
OUTPUT_PATH = (
    SCRIPT_DIRECTORY / "output" / "batch" / "unmatched_subgraph_analysis.json"
)


def build_graph(cache: Dict[str, Any]) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for node in cache.get("fag", {}).get("nodes", []):
        face_id = int(node["face_id"])
        graph.add_node(face_id, **node)
    for edge in cache.get("fag", {}).get("edges", []):
        graph.add_edge(
            int(edge["source"]),
            int(edge["target"]),
            key=int(edge.get("edge_key", edge.get("topological_edge_id", 0))),
            **{
                key: value
                for key, value in edge.items()
                if key not in {"source", "target", "edge_key"}
            },
        )
    return graph


def ground_truth_instances(
    cache: Dict[str, Any],
) -> Dict[Tuple[int, int], Set[int]]:
    groups: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for face in cache.get("faces", []):
        raw_category_id = int(face["ground_truth_category_id"])
        if raw_category_id in {0, 23, 24}:
            continue
        instance_id = face.get("ground_truth_instance_id")
        if instance_id is None:
            continue
        category_id = normalize_category_id(raw_category_id)
        groups[(int(instance_id), category_id)].add(int(face["face_id"]))
    return groups


def predicted_exact_instances(cache: Dict[str, Any]) -> Set[Tuple[int, Tuple[int, ...]]]:
    return {
        (
            normalize_category_id(int(instance["category_id"])),
            tuple(sorted(int(face_id) for face_id in instance.get("face_ids", []))),
        )
        for instance in cache.get("instances", [])
        if instance.get("face_ids")
    }


def required_and_available_face_types(
    seed_graph: nx.MultiGraph,
    model_graph: nx.MultiGraph,
) -> Tuple[Counter[str], Counter[str]]:
    required = Counter(
        str(attributes.get("face_type", "unknown"))
        for _, attributes in seed_graph.nodes(data=True)
    )
    available = Counter(
        str(attributes.get("face_type", "unknown"))
        for _, attributes in model_graph.nodes(data=True)
    )
    return required, available


def node_types_are_sufficient(
    seed_graph: nx.MultiGraph,
    model_graph: nx.MultiGraph,
) -> bool:
    required, available = required_and_available_face_types(seed_graph, model_graph)
    return all(available[face_type] >= count for face_type, count in required.items())


def topology_only_mapping_exists(
    seed_graph: nx.MultiGraph,
    model_graph: nx.MultiGraph,
) -> bool:
    seed_nodes = sorted(
        seed_graph.nodes,
        key=lambda node_id: (-seed_graph.degree(node_id), int(node_id)),
    )
    candidates = {
        int(seed_node_id): [
            int(model_face_id)
            for model_face_id, attributes in model_graph.nodes(data=True)
            if attributes.get("face_type")
            == seed_graph.nodes[seed_node_id].get("face_type")
        ]
        for seed_node_id in seed_nodes
    }
    mapping: Dict[int, int] = {}
    used: Set[int] = set()

    def edge_count(graph: nx.MultiGraph, first: int, second: int) -> int:
        if not graph.has_edge(first, second):
            return 0
        return len(graph.get_edge_data(first, second))

    def backtrack(index: int) -> bool:
        if index == len(seed_nodes):
            return True
        seed_node_id = int(seed_nodes[index])
        required_self_loops = edge_count(seed_graph, seed_node_id, seed_node_id)
        for model_face_id in candidates[seed_node_id]:
            if model_face_id in used:
                continue
            if edge_count(model_graph, model_face_id, model_face_id) < required_self_loops:
                continue
            valid = True
            for other_seed_node, other_model_face in mapping.items():
                if edge_count(
                    model_graph, model_face_id, other_model_face
                ) < edge_count(seed_graph, seed_node_id, other_seed_node):
                    valid = False
                    break
            if not valid:
                continue
            mapping[seed_node_id] = model_face_id
            used.add(model_face_id)
            if backtrack(index + 1):
                return True
            used.remove(model_face_id)
            del mapping[seed_node_id]
        return False

    return backtrack(0)


def category_name(category_id: int) -> str:
    if 0 <= category_id < len(core.FACE_CATEGORIES):
        return core.FACE_CATEGORIES[category_id]
    return str(category_id)


def analyze() -> Dict[str, Any]:
    features = core.load_feature_seeds()
    feature_by_category = {
        int(feature["category_id"]): feature for feature in features
    }
    details: List[Dict[str, Any]] = []
    total_ground_truth_instances = 0
    exact_instances = 0

    for cache_path in sorted(CACHE_ROOT.glob("*.json")):
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        full_graph = build_graph(cache)
        predicted = predicted_exact_instances(cache)
        for (ground_truth_instance_id, category_id), face_ids in (
            ground_truth_instances(cache).items()
        ):
            total_ground_truth_instances += 1
            exact_key = (category_id, tuple(sorted(face_ids)))
            if exact_key in predicted:
                exact_instances += 1
                continue

            feature = feature_by_category.get(category_id)
            induced_edge_count: Optional[int] = None
            if feature is None:
                classification = "missing_seed_definition"
                seed_node_count = None
                seed_edge_count = None
                required_types: Counter[str] = Counter()
                available_types: Counter[str] = Counter()
            else:
                seed_graph = core.build_seed_graph(feature)
                instance_graph = full_graph.subgraph(face_ids).copy()
                seed_node_count = seed_graph.number_of_nodes()
                seed_edge_count = seed_graph.number_of_edges()
                induced_edge_count = instance_graph.number_of_edges()
                required_types, available_types = required_and_available_face_types(
                    seed_graph, instance_graph
                )

                if (
                    instance_graph.number_of_nodes() < seed_node_count
                    or not node_types_are_sufficient(seed_graph, instance_graph)
                ):
                    classification = "missing_seed_nodes_or_required_face_types"
                elif not topology_only_mapping_exists(seed_graph, instance_graph):
                    classification = (
                        "missing_seed_edge_count"
                        if induced_edge_count < seed_edge_count
                        else "seed_edge_connectivity_or_multiplicity_incompatible"
                    )
                else:
                    attributed_mappings = core.find_basic_matches(
                        instance_graph,
                        seed_graph,
                        occupied_faces=set(),
                    )
                    if not attributed_mappings:
                        classification = "topology_present_but_edge_attributes_fail"
                    elif not any(
                        core.basic_source_surfaces_are_distinct(
                            instance_graph, mapping
                        )
                        and core.basic_mapping_satisfies_geometric_constraints(
                            instance_graph, seed_graph, mapping
                        )
                        for mapping in attributed_mappings
                    ):
                        classification = "attributed_subgraph_present_but_geometry_fails"
                    else:
                        classification = "valid_basic_subgraph_rejected_later"

            details.append(
                {
                    "sample_name": cache_path.stem,
                    "ground_truth_instance_id": ground_truth_instance_id,
                    "category_id": category_id,
                    "category_name": category_name(category_id),
                    "face_ids": sorted(face_ids),
                    "ground_truth_face_count": len(face_ids),
                    "seed_node_count": seed_node_count,
                    "seed_edge_count": seed_edge_count,
                    "ground_truth_induced_edge_count": induced_edge_count,
                    "required_face_types": dict(required_types),
                    "available_face_types": dict(available_types),
                    "classification": classification,
                }
            )

    counts = Counter(detail["classification"] for detail in details)
    by_category: Dict[str, Counter[str]] = defaultdict(Counter)
    for detail in details:
        by_category[detail["category_name"]][detail["classification"]] += 1

    missing_nodes = counts["missing_seed_nodes_or_required_face_types"]
    missing_edge_count = counts["missing_seed_edge_count"]
    incompatible_edge_structure = counts[
        "seed_edge_connectivity_or_multiplicity_incompatible"
    ]
    missing_edges = missing_edge_count + incompatible_edge_structure
    result = {
        "scope": {
            "cache_root": str(CACHE_ROOT.resolve()),
            "sample_count": len(list(CACHE_ROOT.glob("*.json"))),
            "ground_truth_instance_count": total_ground_truth_instances,
            "exact_instance_count": exact_instances,
            "unmatched_instance_count": len(details),
        },
        "structural_deficiency": {
            "missing_nodes_or_required_face_types": missing_nodes,
            "literal_induced_edge_count_shortage": missing_edge_count,
            "edge_count_sufficient_but_connectivity_or_multiplicity_incompatible": (
                incompatible_edge_structure
            ),
            "total_missing_edges_or_required_edge_relations": missing_edges,
            "total_missing_nodes_or_edges": missing_nodes + missing_edges,
            "fraction_of_unmatched": (
                (missing_nodes + missing_edges) / len(details) if details else 0.0
            ),
        },
        "classification_counts": dict(sorted(counts.items())),
        "classification_by_category": {
            name: dict(sorted(category_counts.items()))
            for name, category_counts in sorted(by_category.items())
        },
        "details": details,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    analysis = analyze()
    print(json.dumps({
        "scope": analysis["scope"],
        "structural_deficiency": analysis["structural_deficiency"],
        "classification_counts": analysis["classification_counts"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT_PATH}")
