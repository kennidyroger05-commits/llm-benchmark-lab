"""Opt-in, one-generation provider diagnostic; never a benchmark episode/run.

    Preflight retains dataset/file hash checks, but no task JSON is loaded here.
    Only the existing provider owns template/tokenization/placement/generation/
    decoding/metrics. No episode loop, tools, evaluator or exporters are invoked.
"""
import json

from benchmark.providers.failure_diagnostics import failure_details, sanitized_message
from benchmark.providers.transformers_phase2 import INPUT_TOKEN_LIMIT, EpisodeProviderError
from .phase2_preflight_v2 import run_preflight
from .phase2_runner_v2 import load_provider
from .phase2_runtime import EpisodeDeadline


def synthetic_document():
    """Same textual-document transport, with entirely synthetic instructions.

    The provider wraps this document in one user message, applying the real chat
    template once. There is no separate system message or benchmark task ID.
    """
    return json.dumps({
        'protocol': 'Responda apenas com um objeto JSON completo.',
        'tool_catalog': [],
        'task': 'Retorne exatamente {"type":"final_answer","answer":"SMOKE_OK"}.',
        'history': [],
    }, ensure_ascii=False, separators=(',', ':'))


def run_provider_smoke(model_id, output_root='.', *, preflight_fn=None, provider_loader=None):
    """Normal V2 preflight, shared loader and at most one synthetic generation.

    No warm-ups/retries: those would exceed this diagnostic's generation budget.
    Non-empty text suffices; this is not protocol/content correctness evaluation.
    Results exist only in the returned diagnostic/CLI stdout, never official RAW.
    """
    preflight_fn = run_preflight if preflight_fn is None else preflight_fn
    provider_loader = load_provider if provider_loader is None else provider_loader
    common = {'mode': 'provider-smoke', 'experimental_result': False, 'model_id': model_id}
    stage = 'preflight'
    try:
        preflight = preflight_fn(model_id, output_root, inspect_model=True)
        if preflight['status'] != 'PASS':
            blocked = [check['requirement'] for check in preflight['checks'] if check['status'] != 'PASS']
            raise RuntimeError('preflight_failed: ' + ','.join(blocked))
        stage = 'model_load'
        provider = provider_loader(model_id, preflight['model_revision'])
        provider.begin_episode(deadline=EpisodeDeadline.start())
        document = synthetic_document()
        stage = 'tokenization'
        # Exercise the runtime's preflight + cached callback path. The callback
        # rechecks this limit itself; no estimate, truncation or second template.
        if provider.count_tokens(document) > INPUT_TOKEN_LIMIT:
            raise EpisodeProviderError('context_limit', 'preflight')
        stage = 'provider_generation'
        output = provider(document)
        if not isinstance(output, str) or not output.strip():
            raise ValueError('Model generated no non-whitespace text.')
        stage = 'result_serialization'
        metric = provider.turn_metrics[-1]
        result = {**common, 'status': 'PASS', 'stage': 'provider_generation',
                  'device': metric.device, 'dtype': metric.dtype,
                  'input_tokens': metric.input_tokens, 'output_tokens': metric.output_tokens,
                  'latency_seconds': metric.inference_latency_seconds,
                  'output_preview': sanitized_message(output)[:256]}
        # Fail safely rather than emitting non-JSON numbers/objects in the CLI.
        json.dumps(result, allow_nan=False)
        return result
    except Exception as error:
        diagnostic = failure_details(error, stage=stage)
        return {**common, 'status': 'FAIL', 'stage': diagnostic['stage'],
                'exception_type': diagnostic['exception_type'],
                'exception_message': diagnostic['exception_message'], 'secret_redacted': True}
