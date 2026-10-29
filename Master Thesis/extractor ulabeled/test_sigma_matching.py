"""Regression tests for oriented cylindrical seed matching."""

from __future__ import annotations

import unittest

import networkx as nx

from pipeline import SEED_PATH, space_core


class SigmaMatchingTests(unittest.TestCase):
    def test_candidate_nodes_respect_requested_sigma(self) -> None:
        graph = nx.Graph()
        graph.add_node(0, surface_type="cylinder", orientation_sign=-1)
        graph.add_node(1, surface_type="cylinder", orientation_sign=1)
        graph.add_node(2, surface_type="plane", orientation_sign=-1)

        self.assertEqual(
            space_core.candidate_nodes(
                graph,
                {"surface_type": "cylinder", "orientation_sign": -1},
            ),
            [0],
        )
        self.assertEqual(
            space_core.candidate_nodes(
                graph,
                {"surface_type": "cylinder", "orientation_sign": 1},
            ),
            [1],
        )

    def test_all_cylindrical_seed_roles_define_sigma(self) -> None:
        features = space_core.load_seeds(SEED_PATH)
        cylindrical_nodes = [
            node
            for feature in features
            for node in feature["nodes"]
            if node["surface_type"] == "cylinder"
        ]
        self.assertTrue(cylindrical_nodes)
        self.assertTrue(all(node.get("orientation_sign") in {-1, 1} for node in cylindrical_nodes))

    def test_oring_uses_annular_orientation(self) -> None:
        feature = next(
            feature
            for feature in space_core.load_seeds(SEED_PATH)
            if feature["name"] == "Oring"
        )
        by_role = {
            node["role"]: node["orientation_sign"]
            for node in feature["nodes"]
            if node["surface_type"] == "cylinder"
        }
        self.assertEqual(by_role["outer_cylindrical_wall"], -1)
        self.assertEqual(by_role["inner_cylindrical_wall"], 1)


if __name__ == "__main__":
    unittest.main()
