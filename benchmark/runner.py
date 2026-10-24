"""Sequential runner for the RAW phase of Benchmark V1."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

from benchmark.results import BenchmarkResult


DEFAULT_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "prompts" / "benchmark_v1.json"


class BenchmarkRunner:
    """Run the ten official tasks sequentially through one inference provider."""

    def __init__(
        self,
        provider,
        model_id: str,
        phase: str = "phase1",
        condition: str = "raw",
        benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    ):
        self.provider = provider
        self.model_id = model_id
        self.phase = phase
        self.condition = condition
        self.benchmark_path = Path(benchmark_path)
        with self.benchmark_path.open(encoding="utf-8") as benchmark_file:
            self.tasks = json.load(benchmark_file)

        if not isinstance(self.tasks, list) or len(self.tasks) != 10:
            raise ValueError("Benchmark V1 must contain exactly 10 tasks")

    def _select_prompt(self, task: dict) -> str:
        # RAW routing tasks contain simulated tool outputs only in raw_prompt.
        if self.condition == "raw" and task["id"] in {"T5", "T6"}:
            return task["raw_prompt"]
        return task["prompt"]

    def run(
        self,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> list[BenchmarkResult]:
        """Run each task in file order and preserve task-level failures as results."""
        results = []
        for task in self.tasks:
            prompt = self._select_prompt(task)
            try:
                result = self.provider.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    seed=seed,
                )
                if not isinstance(result, BenchmarkResult):
                    raise TypeError("provider.generate() must return BenchmarkResult")
            except Exception as error:
                result = BenchmarkResult(
                    prompt=prompt,
                    status="error",
                    error=str(error),
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    seed=seed if do_sample else None,
                )

            # Attach runner context while retaining provider measurements and output.
            results.append(replace(
                result,
                task_id=task["id"],
                phase=self.phase,
                condition=self.condition,
                model_id=self.model_id,
                prompt=prompt,
            ))
        return results
