"""Read-only sizes of the exact public document, never token estimates."""
from decimal import Decimal
import json
from benchmark.tools._sandbox_worker import json_bytes


def context_sizes(document):
    try:
        values = json.loads(document, parse_float=Decimal)
    except (ValueError, TypeError):
        return None
    if not isinstance(values, dict) or list(values) != ['protocol', 'tool_catalog', 'task', 'history']:
        return None  # Warm-up/plain text is not an episode document.
    parts = {key: {'characters': len(json_bytes(value).decode('utf-8')),
                   'utf8_bytes': len(json_bytes(value))} for key, value in values.items()}
    return {'document_characters': len(document), 'document_utf8_bytes': len(document.encode()),
            'components': parts, 'component_tokens': None,
            'token_note': 'Only the full post-template input token count is measured; component BPE counts are not additive.'}


def attention_memory_estimate(input_tokens, *, heads, kv_heads, head_dim, layers, output_tokens=512):
    """Theoretical bytes for batch=1, FP16 Q/K/V and FP32 attention softmax.

    These are separate allocations, not an estimate of complete process VRAM.
    Dimensions must come from a declared configuration, not a token heuristic.
    """
    return {'scores_fp16_bytes': heads * input_tokens**2 * 2,
            'softmax_fp32_bytes': heads * input_tokens**2 * 4,
            'expanded_key_value_fp16_bytes': 2 * heads * input_tokens * head_dim * 2,
            'kv_cache_fp16_max_bytes': 2 * layers * kv_heads * (input_tokens + output_tokens) * head_dim * 2}
