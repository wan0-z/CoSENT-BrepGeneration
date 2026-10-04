import unittest

from evaluation import (
    annotate_invalid_label_faces,
    calculate_evaluation,
    find_invalid_ground_truth_instances,
)


class InvalidLabelEvaluationTests(unittest.TestCase):
    def test_too_few_supporting_surfaces_excludes_entire_gt_instance(self):
        faces = [
            {
                "face_id": face_id,
                "ground_truth_category_id": 6,
                "ground_truth_instance_id": 7,
                "ground_truth_bottom": 0,
                "predicted_category_id": 6,
                "incident_edges": [],
                "surface_support_key": (
                    ["plane", "shared"]
                    if face_id < 3
                    else ["plane", "other"]
                ),
            }
            for face_id in range(4)
        ]
        faces.append(
            {
                "face_id": 4,
                "ground_truth_category_id": 24,
                "ground_truth_instance_id": None,
                "ground_truth_bottom": 0,
                "predicted_category_id": 24,
                "incident_edges": [],
            }
        )
        invalid = find_invalid_ground_truth_instances(faces, {6: 3})
        annotate_invalid_label_faces(faces, invalid)
        report = calculate_evaluation(
            faces,
            [{"category_id": 6, "face_ids": [0]}],
            [str(index) for index in range(25)],
            target_category_ids=[6, 24],
            seed_surface_node_counts={6: 3},
            invalid_instances=invalid,
        )
        self.assertEqual([0, 1, 2, 3], report["invalid_label_face_ids"])
        self.assertEqual(
            ["below_seed_supporting_surface_count"], invalid[0]["reasons"]
        )
        self.assertEqual(4, invalid[0]["face_count"])
        self.assertEqual(2, invalid[0]["supporting_surface_count"])
        self.assertEqual(3, invalid[0]["seed_surface_node_count"])
        self.assertEqual(1, report["evaluated_face_count"])
        self.assertEqual(0, report["exact_instance"]["predicted_count"])
        self.assertTrue(faces[0]["invalid_label"])

    def test_adjacency_without_collision_evidence_is_not_invalid(self):
        faces = [
            {
                "face_id": face_id,
                "ground_truth_category_id": 13,
                "ground_truth_instance_id": 2,
                "ground_truth_bottom": int(face_id == 3),
                "predicted_category_id": 13,
                "incident_edges": [],
            }
            for face_id in range(4)
        ]
        faces.append(
            {
                "face_id": 4,
                "ground_truth_category_id": 23,
                "ground_truth_instance_id": 9,
                "ground_truth_bottom": 0,
                "predicted_category_id": 24,
                "incident_edges": [
                    {"other_face_id": 0},
                    {"other_face_id": 1},
                ],
            }
        )
        self.assertEqual(
            [], find_invalid_ground_truth_instances(faces, {13: 4})
        )

    def test_collision_contact_rule_applies_to_every_feature_category(self):
        faces = [
            {
                "face_id": face_id,
                "ground_truth_category_id": 6,
                "ground_truth_instance_id": 4,
                "predicted_category_id": 6,
                "incident_edges": [],
                "surface_support_key": ["plane", face_id],
            }
            for face_id in range(3)
        ]
        faces.append(
            {
                "face_id": 3,
                "ground_truth_category_id": 0,
                "ground_truth_instance_id": 8,
                "predicted_category_id": 24,
                "incident_edges": [
                    {"other_face_id": 0},
                    {"other_face_id": 2},
                ],
            }
        )
        diagnostics = [
            {
                "gt_instance_id": 4,
                "category_id": 6,
                "collision": {
                    "status": "collision_detected",
                    "collision_volume": 2.5,
                    "volume_tolerance": 1.0e-6,
                },
                "collision_modifier_face_contacts": [
                    {
                        "face_id": 3,
                        "category_id": 0,
                        "connected_instance_face_ids": [0, 2],
                        "sample_count": 256,
                        "collision_point_count": 12,
                    }
                ],
            }
        ]
        invalid = find_invalid_ground_truth_instances(
            faces, {6: 3}, collision_diagnostics=diagnostics
        )
        self.assertEqual(1, len(invalid))
        self.assertEqual(
            ["collision_contains_connected_chamfer_or_round"],
            invalid[0]["reasons"],
        )
        self.assertEqual(
            3,
            invalid[0]["collision_evidence"]["modifier_face_contacts"][0][
                "face_id"
            ],
        )

    def test_collision_free_status_does_not_trigger_contact_rule(self):
        faces = [
            {
                "face_id": face_id,
                "ground_truth_category_id": 6,
                "ground_truth_instance_id": 4,
                "predicted_category_id": 6,
                "incident_edges": [],
                "surface_support_key": ["plane", face_id],
            }
            for face_id in range(3)
        ]
        diagnostics = [
            {
                "gt_instance_id": 4,
                "category_id": 6,
                "collision": {"status": "collision_free"},
                "collision_modifier_face_contacts": [
                    {"face_id": 8, "collision_point_count": 1}
                ],
            }
        ]
        self.assertEqual(
            [],
            find_invalid_ground_truth_instances(
                faces, {6: 3}, collision_diagnostics=diagnostics
            ),
        )

    def test_supporting_surfaces_are_counted_after_deduplication(self):
        common = {
            "ground_truth_category_id": 7,
            "ground_truth_instance_id": 3,
            "ground_truth_bottom": 0,
            "predicted_category_id": 7,
            "incident_edges": [],
        }
        same_surface_faces = [
            {
                **common,
                "face_id": face_id,
                "surface_support_key": ["cylinder", [0, 0, 1], [1, 2, 0], 4],
            }
            for face_id in (0, 1)
        ]
        self.assertEqual(
            [], find_invalid_ground_truth_instances(same_surface_faces, {7: 1})
        )

        different_surface_faces = [dict(face) for face in same_surface_faces]
        different_surface_faces[1]["surface_support_key"] = [
            "cylinder",
            [0, 0, 1],
            [8, 2, 0],
            4,
        ]
        invalid = find_invalid_ground_truth_instances(
            different_surface_faces, {7: 1}
        )
        self.assertEqual(1, len(invalid))
        self.assertEqual(
            ["above_seed_supporting_surface_count"], invalid[0]["reasons"]
        )
        self.assertEqual(2, invalid[0]["supporting_surface_count"])
        self.assertEqual(1, invalid[0]["seed_surface_node_count"])


if __name__ == "__main__":
    unittest.main()
