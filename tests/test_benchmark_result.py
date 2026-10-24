"""Tests for provider-independent benchmark result values."""

import unittest
import json
from datetime import datetime

from benchmark.results import BenchmarkResult


class BenchmarkResultTests(unittest.TestCase):
    def test_creation_derives_total_tokens_and_rate(self):
        result = BenchmarkResult(
            task_id="T1",
            input_tokens=12,
            output_tokens=8,
            inference_latency_seconds=2.0,
            max_new_tokens=32,
            temperature=0.7,
            do_sample=True,
            seed=41,
        )

        self.assertEqual(result.total_tokens, 20)
        self.assertEqual(result.tokens_per_second, 4.0)
        self.assertEqual(datetime.fromisoformat(result.timestamp).utcoffset().total_seconds(), 0)

    def test_to_dict_is_plain_serializable_data(self):
        result = BenchmarkResult(task_id="T2", status="error", error="offline")

        values = result.to_dict()
        self.assertIs(type(values), dict)
        self.assertEqual(values["task_id"], "T2")
        self.assertEqual(values["status"], "error")
        self.assertEqual(values["error"], "offline")
        self.assertEqual(values["timestamp"], result.timestamp)
        result = BenchmarkResult(max_new_tokens=64, temperature=0.8, do_sample=True, seed=123)
        values = result.to_dict()
        self.assertEqual(
            {name: values[name] for name in ("max_new_tokens", "temperature", "do_sample", "seed")},
            {"max_new_tokens": 64, "temperature": 0.8, "do_sample": True, "seed": 123},
        )
        json.dumps(values)

    def test_only_supported_statuses_are_accepted(self):
        with self.assertRaisesRegex(ValueError, "status"):
            BenchmarkResult(status="timeout")


if __name__ == "__main__":
    unittest.main()
