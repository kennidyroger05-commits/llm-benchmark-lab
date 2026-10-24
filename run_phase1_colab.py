"""Prepare and run the official RAW Phase 1 benchmark in Google Colab."""

import argparse
from pathlib import Path

from benchmark.exporter import export_results
from benchmark.runner import BenchmarkRunner


MODEL_IDS = {
    "1b": "meta-llama/Llama-3.2-1B-Instruct",
    "3b": "meta-llama/Llama-3.2-3B-Instruct",
    "8b": "meta-llama/Llama-3.1-8B-Instruct",
}
PHASE = "phase1"
CONDITION = "raw"
GENERATION_OPTIONS = {
    "max_new_tokens": 512,
    "do_sample": False,
    "temperature": None,
    "seed": None,
}
WARMUP_PROMPTS = (
    "Responda apenas: pronto.",
    "Quanto é 2 + 2? Responda apenas com o número.",
)


def run_phase1_colab(
    model: str = "1b",
    provider=None,
    runner_factory=BenchmarkRunner,
    exporter=export_results,
    results_root=Path("results"),
    torch_module=None,
):
    """Load a selected official model, warm it up, run V1, and export results."""
    if model not in MODEL_IDS:
        raise ValueError(f"Alias de modelo inválido: {model}. Escolha um de: 1b, 3b, 8b.")
    model_id = MODEL_IDS[model]

    if torch_module is None:
        import torch as torch_module

    if provider is None:
        from benchmark.providers.transformers_provider import TransformersProvider

        # Keep dtype and device placement identical across all official model sizes.
        provider = TransformersProvider(
            model_id=model_id,
            torch_dtype=torch_module.float16,
            device_map="auto",
        )

    # Load tokenizer and model explicitly; neither is loaded by provider import.
    provider.load_tokenizer()
    provider.load_model()

    # Warm-ups use the exact generation options of the official run; discard outputs.
    for prompt in WARMUP_PROMPTS:
        result = provider.generate(prompt, **GENERATION_OPTIONS)
        if result.status != "success":
            raise RuntimeError(f"Warm-up falhou: {result.error}")

    device = provider.hardware.get("device")
    if device and str(device).startswith("cuda") and torch_module.cuda.is_available():
        # Finish pending GPU work before starting the official task sequence.
        torch_module.cuda.synchronize(device)

    # Create and start the official runner only after both warm-ups and synchronization.
    runner = runner_factory(
        provider=provider,
        model_id=model_id,
        phase=PHASE,
        condition=CONDITION,
    )
    results = runner.run(**GENERATION_OPTIONS)
    json_path, csv_path = exporter(
        results,
        model_id=model_id,
        phase=PHASE,
        condition=CONDITION,
        results_root=results_root,
    )

    success_count = sum(result.status == "success" for result in results)
    error_count = sum(result.status == "error" for result in results)
    print(f"Modelo: {model_id}")
    print(f"Dispositivo: {provider.hardware.get('device') or 'n/d'}")
    print(f"GPU: {provider.hardware.get('gpu_name') or 'n/d'}")
    vram = provider.hardware.get("vram_gb")
    print(f"VRAM (GB): {vram if vram is not None else 'n/d'}")
    print(f"dtype: {provider.hardware.get('dtype') or 'n/d'}")
    print(f"Tempo de carregamento (s): {provider.model_load_time_seconds}")
    print("Warm-up: concluído (2 inferências descartadas)")
    print(f"Tarefas: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Error: {error_count}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    return results, json_path, csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa a Fase 1 RAW do benchmark oficial.")
    parser.add_argument("--model", required=True, choices=tuple(MODEL_IDS), help="Alias oficial: 1b, 3b ou 8b")
    arguments = parser.parse_args()
    run_phase1_colab(model=arguments.model)
