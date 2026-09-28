# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx

import extract_one_step as extractor
from analyze_ground_truth_boundaries import collect_external_edges, rule_results
from copy_data import copy_sample
from evaluation import calculate_evaluation


def add_relation(
    graph: nx.MultiGraph,
    source: int,
    target: int,
    key: int,
    convexity: str,
    support: str,
    dihedral_type: str = "right",
) -> None:
    graph.add_edge(
        source,
        target,
        key=key,
        edge_type="line",
        curve_type="line",
        curve_support_key=["line", support],
        convexity=convexity,
        dihedral_type=dihedral_type,
        is_seam=False,
    )


class SeedExtractionLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed_graph = nx.MultiGraph()
        self.seed_graph.add_node(0, face_type="plane")
        self.seed_graph.add_node(1, face_type="plane")
        self.seed_graph.add_edge(
            0,
            1,
            key=0,
            edge_type="line",
            curve_type="line",
            convexity="concave",
            dihedral_type="right",
            self_loop=False,
        )

    def test_basic_mapping_rejects_shared_support_surface(self) -> None:
        graph = nx.MultiGraph()
        graph.add_node(10, face_type="plane", surface_support_key=["plane", "A"])
        graph.add_node(11, face_type="plane", surface_support_key=["plane", "A"])
        self.assertFalse(
            extractor.basic_source_surfaces_are_distinct(
                graph,
                {0: 10, 1: 11},
            )
        )

    def test_shallow_crease_is_not_classified_as_g1_tangent(self) -> None:
        self.assertEqual(
            "obtuse",
            extractor.classify_dihedral_type(4.9269, 175.0731),
        )
        self.assertEqual(
            "tangent",
            extractor.classify_dihedral_type(0.01, 179.99),
        )

    def test_seed_edge_accepts_multiple_dihedral_types(self) -> None:
        seed_edge = {
            "edge_type": "line",
            "convexity": "concave",
            "dihedral_type": ["obtuse", "acute"],
            "self_loop": False,
        }
        model_edge = {
            "edge_type": "line",
            "convexity": "concave",
            "dihedral_type": "acute",
            "is_seam": False,
        }
        self.assertTrue(
            extractor.one_model_edge_matches_seed_edge(model_edge, seed_edge)
        )
        model_edge["dihedral_type"] = "obtuse"
        self.assertTrue(
            extractor.one_model_edge_matches_seed_edge(model_edge, seed_edge)
        )
        model_edge["dihedral_type"] = "right"
        self.assertFalse(
            extractor.one_model_edge_matches_seed_edge(model_edge, seed_edge)
        )

    def test_basic_mapping_requires_configured_dihedral_sum(self) -> None:
        feature = {
            "dihedral_sum_degrees": 180.0,
            "dihedral_sum_tolerance_degrees": 1.0,
            "nodes": [
                {"id": 0, "surface_type": "plane"},
                {"id": 1, "surface_type": "plane"},
                {"id": 2, "surface_type": "plane"},
            ],
            "edges": [
                {"source": 0, "target": 1, "curve_type": "line", "convexity": "concave", "dihedral_type": "any"},
                {"source": 1, "target": 2, "curve_type": "line", "convexity": "concave", "dihedral_type": "any"},
                {"source": 2, "target": 0, "curve_type": "line", "convexity": "concave", "dihedral_type": "any"},
            ],
        }
        seed_graph = extractor.build_seed_graph(feature)
        model_graph = nx.MultiGraph()
        model_graph.add_nodes_from([10, 11, 12])
        for key, (source, target, angle, angle_type) in enumerate([
            (10, 11, 30.0, "acute"),
            (11, 12, 60.0, "acute"),
            (12, 10, 90.0, "right"),
        ]):
            add_relation(
                model_graph, source, target, key, "concave", str(key), angle_type
            )
            model_graph[source][target][key]["dihedral_angle_degrees"] = angle

        mapping = {0: 10, 1: 11, 2: 12}
        self.assertTrue(
            extractor.basic_mapping_satisfies_seed_constraints(
                model_graph, seed_graph, mapping
            )
        )
        model_graph[10][11][0]["dihedral_angle_degrees"] = 40.0
        self.assertFalse(
            extractor.basic_mapping_satisfies_seed_constraints(
                model_graph, seed_graph, mapping
            )
        )

    def test_normal_convex_hull_requires_rank_two_and_positive_span(self) -> None:
        valid_normals = [
            [1.0, 0.0, 0.0],
            [-0.5, 0.8660254038, 0.0],
            [-0.5, -0.8660254038, 0.0],
        ]
        invalid_normals = [
            [1.0, 0.0, 0.0],
            [0.8660254038, 0.5, 0.0],
            [-1.0, 0.0, 0.0],
        ]
        self.assertTrue(
            extractor.plane_normals_form_strict_convex_hull(valid_normals)
        )
        self.assertFalse(
            extractor.plane_normals_form_strict_convex_hull(invalid_normals)
        )

    def test_parallel_face_constraint_uses_mapped_plane_normals(self) -> None:
        feature = {
            "nodes": [
                {"id": 0, "surface_type": "plane"},
                {"id": 1, "surface_type": "plane"},
            ],
            "edges": [],
            "geometric_constraints": [
                {"type": "parallel", "faces": [0, 1]},
            ],
        }
        seed_graph = extractor.build_seed_graph(feature)
        model_graph = nx.MultiGraph()
        model_graph.add_node(
            10, face_type="plane", feature_space_normal=[1.0, 0.0, 0.0]
        )
        model_graph.add_node(
            11, face_type="plane", feature_space_normal=[-1.0, 0.0, 0.0]
        )
        mapping = {0: 10, 1: 11}
        self.assertTrue(
            extractor.basic_mapping_satisfies_geometric_constraints(
                model_graph, seed_graph, mapping
            )
        )
        model_graph.nodes[11]["feature_space_normal"] = [0.0, 1.0, 0.0]
        self.assertFalse(
            extractor.basic_mapping_satisfies_geometric_constraints(
                model_graph, seed_graph, mapping
            )
        )

    def test_cylinder_axis_and_radius_constraints(self) -> None:
        feature = {
            "nodes": [
                {"id": 0, "surface_type": "cylinder"},
                {"id": 1, "surface_type": "cylinder"},
            ],
            "edges": [],
            "geometric_constraints": [
                {"type": "coaxial", "faces": [0, 1]},
                {"type": "different_radius", "faces": [0, 1]},
            ],
        }
        seed_graph = extractor.build_seed_graph(feature)
        model_graph = nx.MultiGraph()
        model_graph.add_node(
            10,
            face_type="cylinder",
            surface_support_key=["cylinder", [0, 0, 1], [1, 2, 0], 5.0],
        )
        model_graph.add_node(
            11,
            face_type="cylinder",
            surface_support_key=["cylinder", [0, 0, -1], [1, 2, 0], 3.0],
        )
        self.assertTrue(
            extractor.basic_mapping_satisfies_geometric_constraints(
                model_graph, seed_graph, {0: 10, 1: 11}
            )
        )
        model_graph.nodes[11]["surface_support_key"][2] = [2, 2, 0]
        self.assertFalse(
            extractor.basic_mapping_satisfies_geometric_constraints(
                model_graph, seed_graph, {0: 10, 1: 11}
            )
        )

    def test_basic_boundary_accepts_convex_with_any_dihedral(self) -> None:
        graph = nx.MultiGraph()
        graph.add_nodes_from([0, 1, 2])
        add_relation(graph, 0, 1, 0, "concave", "internal")
        add_relation(graph, 0, 2, 1, "convex", "outside", dihedral_type="acute")
        valid, invalid = extractor.all_external_edges_satisfy_boundary_rule(
            graph, {0, 1}
        )
        self.assertTrue(valid)
        self.assertEqual([], invalid)

        graph[0][2][1]["dihedral_type"] = "right"
        graph[0][2][1]["convexity"] = "concave"
        valid, invalid = extractor.all_external_edges_satisfy_boundary_rule(
            graph, {0, 1}
        )
        self.assertFalse(valid)
        self.assertEqual(1, len(invalid))

    def test_basic_boundary_accepts_only_non_acute_smooth_edges(self) -> None:
        graph = nx.MultiGraph()
        graph.add_nodes_from([0, 1, 2])
        add_relation(graph, 0, 1, 0, "concave", "internal")
        add_relation(graph, 0, 2, 1, "smooth", "outside", dihedral_type="tangent")
        valid, _ = extractor.all_external_edges_satisfy_boundary_rule(graph, {0, 1})
        self.assertTrue(valid)
        graph[0][2][1]["dihedral_type"] = "acute"
        valid, _ = extractor.all_external_edges_satisfy_boundary_rule(graph, {0, 1})
        self.assertFalse(valid)

    def test_basic_boundary_defers_edges_to_cosurface_candidates(self) -> None:
        graph = nx.MultiGraph()
        graph.add_node(0, face_type="plane", surface_support_key=["plane", "A"])
        graph.add_node(1, face_type="plane", surface_support_key=["plane", "B"])
        graph.add_node(2, face_type="plane", surface_support_key=["plane", "A"])
        add_relation(graph, 0, 1, 0, "concave", "shared-curve")
        add_relation(
            graph, 2, 1, 1, "concave", "shared-curve", dihedral_type="acute"
        )

        candidates = extractor.find_cosurface_expansion_candidates(
            model_graph=graph,
            seed_to_model={0: 0, 1: 1},
            occupied_faces=set(),
        )
        potential_faces = set().union(*candidates.values())

        without_deferral, _ = extractor.all_external_edges_satisfy_boundary_rule(
            graph,
            {0, 1},
        )
        with_deferral, _ = extractor.all_external_edges_satisfy_boundary_rule(
            graph,
            {0, 1},
            ignored_external_faces=potential_faces,
        )

        self.assertFalse(without_deferral)
        self.assertTrue(with_deferral)
        self.assertEqual({2}, potential_faces)

    def test_planar_feature_space_ignores_external_edge_outside_active_halfspaces(self) -> None:
        graph = nx.MultiGraph()
        graph.add_node(
            0,
            face_type="plane",
            feature_space_normal=[1.0, 0.0, 0.0],
            feature_space_offset=0.0,
        )
        graph.add_node(
            1,
            face_type="plane",
            feature_space_normal=[0.0, 1.0, 0.0],
            feature_space_offset=0.0,
        )
        graph.add_node(2, face_type="plane")
        add_relation(graph, 0, 1, 0, "concave", "internal")
        add_relation(graph, 0, 2, 1, "concave", "external")
        graph[0][2][1]["edge_sample_points"] = [[0.0, -1.0, 0.0]]

        valid, invalid = extractor.all_external_edges_satisfy_boundary_rule(
            graph, {0, 1}
        )
        self.assertTrue(valid)
        self.assertEqual([], invalid)

        graph[0][2][1]["edge_sample_points"] = [[0.0, 1.0, 0.0]]
        valid, invalid = extractor.all_external_edges_satisfy_boundary_rule(
            graph, {0, 1}
        )
        self.assertFalse(valid)
        self.assertEqual(1, len(invalid))

    def test_expansion_accepts_and_rejects_each_candidate_independently(self) -> None:
        graph = nx.MultiGraph()
        graph.add_node(0, face_type="plane", surface_support_key=["plane", "A"])
        graph.add_node(1, face_type="plane", surface_support_key=["plane", "B"])
        graph.add_node(2, face_type="plane", surface_support_key=["plane", "A"])
        graph.add_node(3, face_type="plane", surface_support_key=["plane", "C"])
        graph.add_node(4, face_type="plane", surface_support_key=["plane", "A"])
        graph.add_node(5, face_type="plane", surface_support_key=["plane", "D"])

        add_relation(graph, 0, 1, 0, "concave", "shared-curve")
        add_relation(graph, 2, 1, 1, "concave", "shared-curve")
        add_relation(graph, 2, 3, 2, "convex", "candidate-boundary")
        add_relation(graph, 4, 1, 3, "concave", "shared-curve")
        add_relation(
            graph, 4, 5, 4, "concave", "bad-boundary", dihedral_type="acute"
        )

        role_groups, records = extractor.expand_basic_feature(
            model_graph=graph,
            seed_graph=self.seed_graph,
            seed_to_model={0: 0, 1: 1},
            occupied_faces=set(),
        )

        self.assertEqual({0, 2}, role_groups[0])
        self.assertEqual({1}, role_groups[1])
        accepted_faces = {
            record["candidate_face_id"]
            for record in records
            if record.get("accepted")
        }
        self.assertEqual({2}, accepted_faces)
        self.assertTrue(any(
            record.get("candidate_face_id") == 4
            and record.get("reason") == "candidate_external_edge_violates_boundary_rule"
            for record in records
        ))

    def test_disconnected_cosurface_candidate_is_accepted_inside_feature_space(self) -> None:
        graph = nx.MultiGraph()
        graph.add_node(
            0,
            face_type="plane",
            surface_support_key=["plane", "A"],
            feature_space_normal=[1.0, 0.0, 0.0],
            feature_space_offset=0.0,
            _face_sample_points=[[0.0, 0.0, 0.0]],
        )
        graph.add_node(
            1,
            face_type="plane",
            surface_support_key=["plane", "B"],
            feature_space_normal=[0.0, 1.0, 0.0],
            feature_space_offset=0.0,
            _face_sample_points=[[0.0, 0.0, 0.0]],
        )
        graph.add_node(
            2,
            face_type="plane",
            surface_support_key=["plane", "A"],
            feature_space_normal=[1.0, 0.0, 0.0],
            feature_space_offset=0.0,
            _face_sample_points=[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]],
        )
        graph.add_node(3, face_type="plane", surface_support_key=["plane", "C"])
        add_relation(graph, 0, 1, 0, "concave", "basic")
        add_relation(graph, 2, 3, 1, "convex", "candidate-boundary")
        graph[2][3][1]["g1_continuous"] = False

        role_groups, records = extractor.expand_basic_feature(
            model_graph=graph,
            seed_graph=self.seed_graph,
            seed_to_model={0: 0, 1: 1},
            occupied_faces=set(),
        )

        self.assertEqual({0, 2}, role_groups[0])
        self.assertTrue(any(
            record.get("candidate_face_id") == 2
            and record.get("reason") == "accepted_disconnected_cosurface"
            for record in records
        ))

    def test_feature_space_volume_collision_uses_inward_offset(self) -> None:
        part_shape = extractor.BRepPrimAPI_MakeBox(
            extractor.gp_Pnt(0.0, 0.0, 0.0),
            extractor.gp_Pnt(10.0, 10.0, 10.0),
        ).Shape()
        graph = nx.MultiGraph()
        graph.add_node(
            0,
            face_type="plane",
            feature_space_normal=[0.0, 0.0, 1.0],
            feature_space_offset=10.0,
            _face_sample_points=[
                [0.0, 0.0, 5.0],
                [10.0, 10.0, 10.0],
            ],
        )
        valid, record = extractor.planar_feature_space_is_collision_free(
            part_shape,
            graph,
            {0},
        )
        self.assertTrue(valid)
        self.assertEqual("collision_free", record["status"])

        graph.nodes[0]["feature_space_offset"] = 5.0
        valid, record = extractor.planar_feature_space_is_collision_free(
            part_shape,
            graph,
            {0},
        )
        self.assertFalse(valid)
        self.assertEqual("collision_detected", record["status"])


class GroundTruthBoundaryAnalysisTests(unittest.TestCase):
    def test_external_edges_and_rule_coverage_use_complete_instance_set(self) -> None:
        graph = nx.MultiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        add_relation(graph, 0, 1, 0, "concave", "internal")
        add_relation(graph, 0, 2, 1, "convex", "external-acute", "acute")
        add_relation(graph, 1, 3, 2, "smooth", "external-tangent", "tangent")

        external_edges = collect_external_edges(graph, {0, 1})
        self.assertEqual(2, len(external_edges))
        results = rule_results(external_edges)
        self.assertFalse(results["convex_only"])
        self.assertTrue(results["convex_or_smooth"])
        self.assertFalse(results["non_acute"])
        self.assertTrue(results["convex_any_dihedral_or_smooth_non_acute"])


class EvaluationTests(unittest.TestCase):
    def test_confusion_matrix_normalizes_step_labels_and_ignores_exclusions(self) -> None:
        faces = [
            {
                "face_id": 0,
                "ground_truth_category_id": 10,
                "predicted_category_id": 8,
                "ground_truth_instance_id": 1,
            },
            {
                "face_id": 1,
                "ground_truth_category_id": 0,
                "predicted_category_id": 24,
                "ground_truth_instance_id": 2,
            },
            {
                "face_id": 2,
                "ground_truth_category_id": 24,
                "predicted_category_id": 24,
                "ground_truth_instance_id": None,
            },
        ]
        instances = [{"category_id": 8, "face_ids": [0]}]
        names = [str(index) for index in range(25)]

        result = calculate_evaluation(
            faces=faces,
            instances=instances,
            category_names=names,
            target_category_ids=[8, 24],
        )

        self.assertEqual(2, result["evaluated_face_count"])
        self.assertEqual([1], result["ignored_face_ids"])
        self.assertEqual(1.0, result["face_accuracy"])
        self.assertEqual(1, result["exact_instance"]["correct_count"])


class CopyDataTests(unittest.TestCase):
    def test_copy_sample_adds_folder_without_overwriting(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            destination_root = root / "single"
            (dataset_root / "steps").mkdir(parents=True)
            (dataset_root / "labels").mkdir(parents=True)
            (dataset_root / "steps" / "sample.step").write_text(
                "step-data",
                encoding="utf-8",
            )
            (dataset_root / "labels" / "sample.json").write_text(
                "{}",
                encoding="utf-8",
            )

            destination = copy_sample(
                dataset_root=dataset_root,
                sample_name="sample",
                destination_root=destination_root,
            )

            self.assertEqual("step-data", (destination / "sample.step").read_text())
            self.assertTrue((destination / "sample.json").exists())
            with self.assertRaises(FileExistsError):
                copy_sample(
                    dataset_root=dataset_root,
                    sample_name="sample",
                    destination_root=destination_root,
                )


if __name__ == "__main__":
    unittest.main()
