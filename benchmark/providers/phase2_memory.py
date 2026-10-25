"""Phase 2 SDPA execution policy, without changing tokens, masks or KV cache.

HF 4.57 enables native GQA when the causal mask is None. Torch 2.11's efficient
CUDA kernel does not support unequal Q/KV heads; Flash needs SM80+, unlike T4.
Expand K/V heads explicitly (the same operation as HF eager/repeat_kv), then
use ordinary SDPA. CUDA math fallback is forbidden, not silently retried.
"""
from contextlib import contextmanager, nullcontext

ATTENTION_NAME = 'phase2_sdpa_expanded_kv'


def expanded_kv_sdpa(module, query, key, value, attention_mask, dropout=0.0,
                     scaling=None, is_causal=None, **kwargs):
    """HF AttentionInterface signature; preserve its causal/padding semantics."""
    import torch
    if kwargs.get('output_attentions') or kwargs.get('head_mask') is not None:
        raise ValueError('Attention weights/head masks are unsupported by Phase 2 SDPA.')
    groups = query.shape[1] // key.shape[1]
    if query.shape[1] % key.shape[1] or key.shape[1] != value.shape[1]:
        raise ValueError('Invalid grouped attention head shapes.')
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, :key.shape[-2]]
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, 'is_causal', True)
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, is_causal=is_causal, enable_gqa=False)
    return output.transpose(1, 2).contiguous(), None


def configure_memory_attention(model):
    """Register a distinct HF alias; never replace global 'sdpa' or Phase 1."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS
    before = getattr(model.config, '_attn_implementation', None)
    AttentionInterface.register(ATTENTION_NAME, expanded_kv_sdpa)
    # Without this registration HF skips mask construction for unknown aliases.
    # Reuse its exact SDPA mask builder, including padding and cached decoding.
    AttentionMaskInterface.register(ATTENTION_NAME, ALL_MASK_ATTENTION_FUNCTIONS['sdpa'])
    model.set_attn_implementation(ATTENTION_NAME)
    if model.config._attn_implementation != ATTENTION_NAME:
        raise RuntimeError('Model refused memory-efficient attention configuration.')
    return before


def efficient_cuda_kernels(device):
    """Allow efficient/Flash kernels only on CUDA; never move tensors to CPU."""
    import torch
    if str(device).startswith('cuda') and torch.cuda.is_available():
        attention = torch.nn.attention
        return attention.sdpa_kernel([attention.SDPBackend.EFFICIENT_ATTENTION,
                                      attention.SDPBackend.FLASH_ATTENTION])
    return nullcontext()


@contextmanager
def cuda_memory_observation(device):
    """Per-call bytes, including failed attempts; no tensors retained or exported."""
    import torch
    metrics = None
    active = str(device).startswith('cuda') and torch.cuda.is_available()
    if active:
        free, total = torch.cuda.mem_get_info(device)
        metrics = {'device': str(device), 'free_before_bytes': free, 'total_bytes': total}
        torch.cuda.reset_peak_memory_stats(device)
    try:
        yield metrics
    finally:
        if active:
            try:
                free, _ = torch.cuda.mem_get_info(device)
                metrics.update(free_after_bytes=free,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            except Exception:
                metrics['measurement_error'] = 'cuda_memory_unavailable'


@contextmanager
def observe_forward(model, evidence):
    """First prefill shapes/cache class only; the hook never stores tensor refs."""
    register = getattr(model, 'register_forward_pre_hook', None)
    if not callable(register):
        yield
        return
    def capture(module, args, kwargs):
        if evidence:
            return
        for name in ('input_ids', 'attention_mask', 'cache_position'):
            tensor = kwargs.get(name)
            if tensor is not None:
                evidence[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
        cache = kwargs.get('past_key_values')
        evidence['actual_cache_class'] = type(cache).__name__ if cache is not None else None
        for name in ('use_cache', 'output_attentions', 'return_dict', 'logits_to_keep'):
            value = kwargs.get(name)
            if value is None or type(value) in (bool, int):
                evidence[name] = value
    handle = register(capture, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def execution_metadata(model):
    """Allowlisted configuration, not a claim to observe an individual CUDA kernel."""
    config = getattr(model, 'config', None)
    generation = getattr(model, 'generation_config', None)
    return {
        'attention_implementation': getattr(config, '_attn_implementation', None),
        'cuda_sdpa_backends_allowed': ['efficient', 'flash'],
        'native_gqa': False,
        'cache_implementation': getattr(generation, 'cache_implementation', None),
        'cache_policy': 'transformers_default' if getattr(generation, 'cache_implementation', None) is None else 'explicit',
        'use_cache': getattr(generation, 'use_cache', getattr(config, 'use_cache', None)),
        'output_attentions': getattr(generation, 'output_attentions', getattr(config, 'output_attentions', None)),
        'output_scores': getattr(generation, 'output_scores', None),
        'output_logits': getattr(generation, 'output_logits', None),
        'return_dict_in_generate': getattr(generation, 'return_dict_in_generate', None),
        'model_dimensions': {key: getattr(config, key, None) for key in (
            'num_attention_heads', 'num_key_value_heads', 'num_hidden_layers', 'hidden_size', 'head_dim')},
    }
