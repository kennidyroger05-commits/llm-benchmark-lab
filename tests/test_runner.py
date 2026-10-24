"""Runner tests using a fake provider and the local Benchmark V1 data."""

import json
import unittest
from pathlib import Path

from benchmark.results import BenchmarkResult
from benchmark.runner import BenchmarkRunner, DEFAULT_BENCHMARK_PATH


class FakeProvider:
    """Return simple results while recording call order and generation options."""

    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def generate(self, prompt, **options):
        call_number = len(self.calls)
        self.calls.append((prompt, options))
        if call_number == self.fail_at:
            raise RuntimeError("simulated inference failure")
        return BenchmarkResult(
            provider="fake",
            response=f"response-{call_number}",
            status="success",
            input_tokens=2,
            output_tokens=1,
            inference_latency_seconds=0.5,
        )


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        with DEFAULT_BENCHMARK_PATH.open(encoding="utf-8") as benchmark_file:
            self.tasks = json.load(benchmark_file)

    def test_runs_all_ten_tasks_sequentially_with_correct_prompts_and_context(self):
        provider = FakeProvider()
        runner = BenchmarkRunner(provider, "org/model-1b")

        results = runner.run(max_new_tokens=77)

        self.assertEqual(len(runner.tasks), 10)
        self.assertEqual(len(provider.calls), 10)
        self.assertEqual(len(results), 10)
        for index, (task, result, call) in enumerate(zip(self.tasks, results, provider.calls)):
            expected_prompt = task["raw_prompt"] if task["id"] in {"T5", "T6"} else task["prompt"]
            self.assertEqual(call[0], expected_prompt)
            self.assertEqual(result.prompt, expected_prompt)
            self.assertEqual(result.task_id, task["id"])
            self.assertEqual(result.phase, "phase1")
            self.assertEqual(result.condition, "raw")
            self.assertEqual(result.model_id, "org/model-1b")
            self.assertEqual(result.response, f"response-{index}")
            self.assertEqual(call[1], {
                "max_new_tokens": 77,
                "do_sample": False,
                "temperature": None,
                "seed": None,
            })

    def test_continues_after_a_task_failure_and_uses_configured_sampling(self):
        provider = FakeProvider(fail_at=3)
        runner = BenchmarkRunner(provider, "org/model-3b", phase="phase1", condition="raw")

        results = runner.run(
            max_new_tokens=32,
            do_sample=True,
            temperature=0.7,
            seed=123,
        )

        self.assertEqual(len(results), 10)
        self.assertEqual(results[3].status, "error")
        self.assertEqual(results[3].error, "simulated inference failure")
        self.assertEqual(results[2].status, "success")
        self.assertEqual(results[4].status, "success")
        self.assertEqual(results[3].temperature, 0.7)
        self.assertEqual(results[3].seed, 123)
        self.assertEqual(provider.calls[-1][1]["max_new_tokens"], 32)

    def test_rejects_non_result_provider_return_without_stopping_run(self):
        class InvalidProvider(FakeProvider):
            def generate(self, prompt, **options):
                self.calls.append((prompt, options))
                return "not a BenchmarkResult"

        results = BenchmarkRunner(InvalidProvider(), "org/model").run()

        self.assertEqual(len(results), 10)
        self.assertTrue(all(result.status == "error" for result in results))
        self.assertIn("must return BenchmarkResult", results[0].error)


if __name__ == "__main__":
    unittest.main()
