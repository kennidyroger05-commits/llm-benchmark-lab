"""Colab Phase 1 orchestration tests that never load a real model."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from benchmark.results import BenchmarkResult
from run_phase1_colab import (
    CONDITION,
    GENERATION_OPTIONS,
    MODEL_IDS,
    PHASE,
    WARMUP_PROMPTS,
    run_phase1_colab,
)


class FakeProvider:
    def __init__(self, events, device="cuda:0"):
        self.events = events
        self.hardware = {
            "device": device,
            "gpu_name": "Mock GPU" if device.startswith("cuda") else None,
            "vram_gb": 8.0 if device.startswith("cuda") else None,
            "dtype": "float16",
        }
        self.model_load_time_seconds = 1.5
        self.calls = []

    def load_tokenizer(self):
        self.events.append("tokenizer_loaded")

    def load_model(self):
        self.events.append("model_loaded")

    def generate(self, prompt, **options):
        index = len(self.calls)
        self.calls.append((prompt, options))
        self.events.append(f"generate_{index}")
        return BenchmarkResult(prompt=prompt, response="discardable", status="success")


class FakeRunner:
    def __init__(self, provider, events, **context):
        self.provider = provider
        self.events = events
        self.context = context
        self.events.append("runner_created")

    def run(self, **options):
        self.events.append("runner_started")
        self.options = options
        self.results = []
        for index in range(10):
            self.events.append(f"task_{index}")
            self.results.append(BenchmarkResult(task_id=f"T{index + 1}", status="success"))
        return self.results


class Phase1ColabTests(unittest.TestCase):
    def run_with_device(self, device, model="1b"):
        events = []
        provider = FakeProvider(events, device=device)

        def runner_factory(**context):
            return FakeRunner(events=events, **context)

        def exporter(results, **kwargs):
            events.append("exported")
            self.assertEqual(results, fake_runner.results)
            self.assertEqual(kwargs["model_id"], MODEL_IDS[model])
            self.assertEqual(kwargs["phase"], "phase1")
            self.assertEqual(kwargs["condition"], "raw")
            return Path("mock/results.json"), Path("mock/results.csv")

        # Capture the runner instance so export can prove only official results are saved.
        fake_runner = None

        def capturing_runner_factory(**context):
            nonlocal fake_runner
            fake_runner = runner_factory(**context)
            return fake_runner

        sync = Mock(side_effect=lambda selected_device: events.append(f"sync_{selected_device}"))
        torch_stub = SimpleNamespace(
            float16="float16",
            cuda=SimpleNamespace(is_available=lambda: device.startswith("cuda"), synchronize=sync),
        )
        dataset = Path(__file__).resolve().parents[1] / "prompts" / "benchmark_v1.json"
        dataset_before = dataset.read_bytes()

        result = run_phase1_colab(
            provider=provider,
            model=model,
            runner_factory=capturing_runner_factory,
            exporter=exporter,
            torch_module=torch_stub,
        )

        self.assertEqual(dataset.read_bytes(), dataset_before)
        return result, provider, events, sync, fake_runner

    def test_two_warmups_precede_sequential_runner_and_export(self):
        (results, json_path, csv_path), provider, events, sync, runner = self.run_with_device("cuda:0")

        self.assertEqual(len(WARMUP_PROMPTS), 2)
        self.assertEqual(MODEL_IDS["1b"], "meta-llama/Llama-3.2-1B-Instruct")
        self.assertEqual(GENERATION_OPTIONS, {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": None,
            "seed": None,
        })
        self.assertEqual(provider.calls[:2], [(prompt, GENERATION_OPTIONS) for prompt in WARMUP_PROMPTS])
        self.assertEqual(provider.calls[2:], [])
        self.assertEqual(events.count("generate_0"), 1)
        self.assertLess(events.index("model_loaded"), events.index("generate_0"))
        self.assertLess(events.index("generate_1"), events.index("sync_cuda:0"))
        self.assertLess(events.index("sync_cuda:0"), events.index("runner_created"))
        self.assertLess(events.index("runner_created"), events.index("runner_started"))
        self.assertLess(events.index("task_9"), events.index("exported"))
        self.assertEqual(sync.call_count, 1)
        self.assertEqual(runner.context, {
            "model_id": MODEL_IDS["1b"],
            "phase": PHASE,
            "condition": CONDITION,
        })
        self.assertEqual(runner.options, GENERATION_OPTIONS)
        self.assertEqual(len(results), 10)
        self.assertEqual([result.task_id for result in results], [f"T{i}" for i in range(1, 11)])
        self.assertEqual((json_path, csv_path), (Path("mock/results.json"), Path("mock/results.csv")))

    def test_cpu_warmup_does_not_synchronize_cuda(self):
        (_, _, _), _, events, sync, _ = self.run_with_device("cpu")

        sync.assert_not_called()
        self.assertLess(events.index("generate_1"), events.index("runner_created"))
        self.assertNotIn("sync_cpu", events)

    def test_all_official_aliases_use_the_same_protocol_and_model_id(self):
        for alias, expected_model_id in MODEL_IDS.items():
            with self.subTest(alias=alias):
                (results, _, _), provider, _, _, runner = self.run_with_device("cpu", model=alias)
                self.assertEqual(runner.context["model_id"], expected_model_id)
                self.assertEqual(runner.options, GENERATION_OPTIONS)
                self.assertEqual(provider.calls, [(prompt, GENERATION_OPTIONS) for prompt in WARMUP_PROMPTS])
                self.assertEqual(len(results), 10)

    def test_invalid_alias_is_rejected_before_loading_or_running(self):
        events = []
        provider = FakeProvider(events, device="cpu")

        with self.assertRaisesRegex(ValueError, "Alias de modelo inválido.*1b, 3b, 8b"):
            run_phase1_colab(model="70b", provider=provider, torch_module=SimpleNamespace())

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
