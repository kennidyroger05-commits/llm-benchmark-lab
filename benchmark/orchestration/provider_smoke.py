"""Opt-in, one-generation provider diagnostic; never a benchmark episode/run.

    Preflight retains dataset/file hash checks, but no task JSON is loaded here.
    Only the existing provider owns template/tokenization/placement/generation/
    decoding/metrics. No episode loop, tools, evaluator or exporters are invoked.
"""
import json

from benchmark.providers.failure_diagnostics import failure_details, sanitized_message
from benchmark.providers.transformers_phase2 import INPUT_TOKEN_LIMIT, EpisodeProviderError
from benchmark.providers.phase2_memory import execution_metadata
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
    return _run_smoke(model_id, output_root, preflight_fn, provider_loader)


def context_document(repetitions):
    return json.dumps({'protocol': 'Responda com uma palavra curta.', 'tool_catalog': [],
        'task': 'Texto sintético para diagnóstico de memória:' + ' x' * repetitions,
        'history': []}, ensure_ascii=False, separators=(',', ':'))


def size_synthetic_context(provider, target):
    """Find a nearby synthetic input using actual post-template token counts.

    No character-to-token estimate, padding tokens, truncation or task content.
    Only tokenizer calls occur here, never generation.
    """
    if type(target) is not int or not 512 <= target <= INPUT_TOKEN_LIMIT:
        raise ValueError('Context smoke target must be between 512 and 15872 input tokens.')
    low, high = 0, 1
    while provider.count_tokens(context_document(high)) < target and high < 32768:
        low, high = high, high * 2
    best = None
    while low <= high:
        middle = (low + high) // 2
        document = context_document(middle)
        count = provider.count_tokens(document)
        if count <= INPUT_TOKEN_LIMIT and (best is None or abs(count - target) < best[0]):
            best = (abs(count - target), document)
        if count == target:
            return document
        if count < target:
            low = middle + 1
        else:
            high = middle - 1
    if best is None or best[0] > 16:
        raise ValueError('Tokenizer cannot build a synthetic context within 16 tokens of the target.')
    return best[1]


def run_provider_context_smoke(model_id, output_root='.', *, input_tokens=8700,
                               preflight_fn=None, provider_loader=None):
    """Non-benchmark long prefill, eight output tokens, one generation maximum."""
    return _run_smoke(model_id, output_root, preflight_fn, provider_loader, input_tokens)


def _run_smoke(model_id, output_root, preflight_fn, provider_loader, context_target=None):
    preflight_fn = run_preflight if preflight_fn is None else preflight_fn
    provider_loader = load_provider if provider_loader is None else provider_loader
    long_context = context_target is not None
    common = {'mode': 'provider-context-smoke' if long_context else 'provider-smoke',
              'experimental_result': False, 'model_id': model_id}
    stage = 'preflight'
    provider = None
    try:
        if long_context and (type(context_target) is not int or not 512 <= context_target <= INPUT_TOKEN_LIMIT):
            raise ValueError('Context smoke target must be between 512 and 15872 input tokens.')
        preflight = preflight_fn(model_id, output_root, inspect_model=True)
        if preflight['status'] != 'PASS':
            blocked = [check['requirement'] for check in preflight['checks'] if check['status'] != 'PASS']
            raise RuntimeError('preflight_failed: ' + ','.join(blocked))
        stage = 'model_load'
        provider = provider_loader(model_id, preflight['model_revision'])
        deadline = EpisodeDeadline.start()
        provider.begin_episode(deadline=deadline)
        stage = 'tokenization'
        document = size_synthetic_context(provider, context_target) if long_context else synthetic_document()
        if long_context:
            # Sizing is not a generated turn or history; retain only the final input.
            provider.begin_episode(deadline=deadline)
        # Exercise the runtime's preflight + cached callback path. The callback
        # rechecks this limit itself; no estimate, truncation or second template.
        if provider.count_tokens(document) > INPUT_TOKEN_LIMIT:
            raise EpisodeProviderError('context_limit', 'preflight')
        stage = 'provider_generation'
        output = provider.generate_diagnostic(document) if long_context else provider(document)
        if not isinstance(output, str) or not output.strip():
            raise ValueError('Model generated no non-whitespace text.')
        stage = 'result_serialization'
        metric = provider.turn_metrics[-1]
        result = {**common, 'status': 'PASS', 'stage': 'provider_generation',
                  'device': metric.device, 'dtype': metric.dtype,
                  'input_tokens': metric.input_tokens, 'output_tokens': metric.output_tokens,
                  'latency_seconds': metric.inference_latency_seconds,
                  'output_preview': sanitized_message(output)[:256]}
        if long_context:
            result.update(input_token_target=context_target, max_new_tokens=8,
                cuda_memory=metric.cuda_memory, failure_details=None,
                forward_observation=metric.execution,
                **execution_metadata(provider._core.model))
        # Fail safely rather than emitting non-JSON numbers/objects in the CLI.
        json.dumps(result, allow_nan=False)
        return result
    except Exception as error:
        diagnostic = failure_details(error, stage=stage,
            turn_index=provider.turn_metrics[-1].turn_index if provider is not None and provider.turn_metrics else None)
        result = {**common, 'status': 'FAIL', 'stage': diagnostic['stage'],
                'exception_type': diagnostic['exception_type'],
                'exception_message': diagnostic['exception_message'], 'secret_redacted': True}
        if long_context:
            metric = provider.turn_metrics[-1] if provider is not None and provider.turn_metrics else None
            result.update(failure_details=diagnostic, input_token_target=context_target,
                input_tokens=getattr(metric, 'input_tokens', None),
                output_tokens=getattr(metric, 'output_tokens', None),
                latency_seconds=getattr(metric, 'inference_latency_seconds', None),
                cuda_memory=getattr(metric, 'cuda_memory', None), max_new_tokens=8,
                attention_implementation=None, cache_implementation=None,
                forward_observation=getattr(metric, 'execution', None))
            if provider is not None:
                result.update(execution_metadata(provider._core.model))
        return result
