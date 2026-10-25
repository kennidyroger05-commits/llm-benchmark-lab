"""Protocol V2 production wiring; V1 remains archived and separately testable."""
import json
from .phase2_runner import run_phase2 as shared_runner
from .phase2_supervisor import ProcessSupervisor
from .phase2_preflight_v2 import ROOT, FROZEN, run_preflight
from .revision_v2 import run_transformers_episode
from benchmark.providers.transformers_phase2 import TransformersEpisodeProvider
from benchmark.tools.restricted_python import RestrictedASTBackend
from benchmark.tools.domain_tools import DomainSession
from benchmark.evaluation.phase2_v2 import evaluate_episode, build_report


class V2Supervisor(ProcessSupervisor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, container_cleanup=False, **kwargs)


def load_provider(model_id, revision):
    """Shared explicit loading for collection and the non-benchmark smoke.

    Pin tokenizer and weights to the preflight revision. Loading itself never
    generates text, reads tasks, creates sessions or performs warm-ups.
    """
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(model_id, revision=revision, allow_patterns=[
        '*.json', '*.safetensors', 'tokenizer.model', '*.tiktoken'])
    provider = TransformersEpisodeProvider(model_id)
    provider._core.model_id = snapshot
    try:
        provider.load_tokenizer(); provider.load_model()
    finally:
        provider._core.model_id = model_id
    provider.validate_ready()
    return provider


def official_worker(settings, owner):
    provider = load_provider(settings['model_id'], settings['revision'])
    provider.warm_up()
    tasks = json.loads((ROOT / 'prompts/benchmark_v1.json').read_text())
    backend = RestrictedASTBackend()
    def handle(index):
        task = tasks[index]
        session = DomainSession.from_benchmark(task['id']) if task['id'] in {'T5', 'T6'} else None
        result = run_transformers_episode(task, provider, domain_session=session,
                                          sandbox_backend=backend, run_id=settings['run_id'])
        result.run_metadata.update(model_revision=settings['revision'], tokenizer_revision=settings['revision'],
                                   protocol_version='phase2_v2', sandbox_mode=backend.sandbox_mode)
        return result
    return handle


def run_phase2(model_id, output_root, **test_seams):
    options = dict(preflight_fn=run_preflight, supervisor_factory=V2Supervisor,
        worker=official_worker, evaluate_fn=evaluate_episode, report_builder=build_report,
        evaluation_backend=lambda owner: RestrictedASTBackend(), frozen_path=FROZEN,
        protocol_version='phase2_v2')
    options.update(test_seams)
    return shared_runner(model_id, output_root, **options)
