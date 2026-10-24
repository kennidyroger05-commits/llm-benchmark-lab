"""Prepare and run the official RAW Phase 1 benchmark in Google Colab."""

from pathlib import Path

from benchmark.exporter import export_results
from benchmark.runner import BenchmarkRunner


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
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
    provider=None,
    runner_factory=BenchmarkRunner,
    exporter=export_results,
    results_root=Path("results"),
    torch_module=None,
):
    """Load the 1B model, discard two warm-ups, run V1, and export the results."""
    if torch_module is None:
        import torch as torch_module

    if provider is None:
        from benchmark.providers.transformers_provider import TransformersProvider

        # Construct the provider explicitly for the approved 1B Colab configuration.
        provider = TransformersProvider(
            model_id=MODEL_ID,
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
        model_id=MODEL_ID,
        phase=PHASE,
        condition=CONDITION,
    )
    results = runner.run(**GENERATION_OPTIONS)
    json_path, csv_path = exporter(
        results,
        model_id=MODEL_ID,
        phase=PHASE,
        condition=CONDITION,
        results_root=results_root,
    )

    success_count = sum(result.status == "success" for result in results)
    error_count = sum(result.status == "error" for result in results)
    print(f"Modelo: {MODEL_ID}")
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
    run_phase1_colab()
