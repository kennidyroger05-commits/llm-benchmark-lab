"""Tests for result exports without writing into the project results directory."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.exporter import export_results
from benchmark.results import BenchmarkResult


class BenchmarkExporterTests(unittest.TestCase):
    def make_results(self):
        return [
            BenchmarkResult(
                task_id=f"T{i}",
                phase="phase1",
                condition="raw",
                model_id="org/model-1b",
                provider="fake",
                prompt=f"prompt-{i}",
                response=f"response-{i}",
                input_tokens=i,
                output_tokens=2,
                max_new_tokens=20,
                do_sample=False,
            )
            for i in range(1, 11)
        ]

    def test_exports_all_fields_to_json_and_csv_creating_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "results"
            json_path, csv_path = export_results(
                self.make_results(), "org/model-1b", results_root=root, run_id="run-001"
            )

            self.assertEqual(json_path.parent, root / "raw" / "phase1" / "org-model-1b")
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            json_rows = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(json_rows), 10)
            self.assertEqual(json_rows[0]["task_id"], "T1")
            self.assertIn("tokens_per_second", json_rows[0])
            self.assertEqual(json_rows[0]["max_new_tokens"], 20)

            with csv_path.open(encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                csv_rows = list(reader)
            self.assertEqual(len(csv_rows), 10)
            self.assertEqual(csv_rows[0]["task_id"], "T1")
            self.assertIn("gpu_name", reader.fieldnames)
            self.assertIn("seed", reader.fieldnames)

    def test_export_does_not_overwrite_existing_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = (self.make_results(), "org/model-1b")
            kwargs = {"results_root": Path(temp_dir), "run_id": "fixed-run"}
            first_paths = export_results(*args, **kwargs)
            original_json = first_paths[0].read_text(encoding="utf-8")

            with self.assertRaises(FileExistsError):
                export_results(*args, **kwargs)

            self.assertEqual(first_paths[0].read_text(encoding="utf-8"), original_json)
            self.assertTrue(first_paths[1].is_file())

    def test_model_slug_is_sanitized_and_run_id_is_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, _ = export_results(
                self.make_results(), "org/model name", results_root=Path(temp_dir), run_id="safe"
            )
            self.assertEqual(json_path.parent.name, "org-model-name")
            with self.assertRaises(ValueError):
                export_results(self.make_results(), "org/model", results_root=Path(temp_dir), run_id="../unsafe")


if __name__ == "__main__":
    unittest.main()
