"""Single/batch diagnostic-cache regression checks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import extract_batch_space as batch


class BatchSpacePipelineTests(unittest.TestCase):
    def test_partition_reuses_only_current_valid_outputs(self):
        signature = batch.pipeline_signature(batch.FEATURE_SEED_JSON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "web_cache"
            sample_pairs = []
            for name in ("sample_1", "sample_2", "sample_3"):
                label = root / f"{name}.json"
                batch.write_json(label, {})
                sample_pairs.append((root / f"{name}.step", label))

            valid_cache = {
                "space_pipeline_signature": signature,
                "space_diagnostics": {"schema_version": 3},
                "topological_edges": [],
                "evaluation": {"confusion_matrix": {}},
                "face_count": 1,
                "feature_instance_count": 0,
            }
            batch.write_json(cache_root / "sample_1.json", valid_cache)
            stale_cache = dict(valid_cache, space_pipeline_signature="stale")
            batch.write_json(cache_root / "sample_2.json", stale_cache)

            reused, pending = batch.partition_samples(
                sample_pairs,
                cache_root,
                signature,
            )

            self.assertEqual([result["sample_name"] for result in reused], ["sample_1"])
            self.assertEqual([step.stem for step, _ in pending], ["sample_2", "sample_3"])

    def test_shared_pipeline_and_cache_version(self):
        source = Path(__file__).parent / "output_space/single/20221123_142528_10/cache.json"
        reference = json.loads(source.read_text())
        reference["space_diagnostics"]["schema_version"] = 3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = root / "sample.json"
            cache_path = root / "web_cache/sample.json"
            batch.write_json(label, {})
            arguments = (str(root / "sample.step"), str(label),
                         str(batch.FEATURE_SEED_JSON_PATH), str(cache_path), False)
            with patch.object(batch.space_core, "build_space_cache", return_value=reference) as shared:
                first = batch.process_one_sample(*arguments)
                self.assertEqual(first["status"], "success")
                shared.assert_called_once()
                cached = json.loads(cache_path.read_text())
                for key in ("instances", "space_diagnostics", "topological_edges", "evaluation"):
                    self.assertEqual(cached[key], reference[key])
                self.assertTrue((root / "sample_details/sample/diagnostic_report.json").exists())
                second = batch.process_one_sample(*arguments)
                self.assertEqual(second["status"], "reused")
                self.assertEqual(shared.call_count, 1)
                cached.pop("space_pipeline_signature")
                batch.write_json(cache_path, cached)
                third = batch.process_one_sample(*arguments)
                self.assertEqual(third["status"], "success")
                self.assertEqual(shared.call_count, 2)


if __name__ == "__main__":
    unittest.main()
