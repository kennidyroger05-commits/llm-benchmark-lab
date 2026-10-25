"""Memory-policy/context-smoke regressions. No official model or network calls."""
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_phase2 as cli
from benchmark.providers.phase2_memory import (ATTENTION_NAME, expanded_kv_sdpa,
    configure_memory_attention, efficient_cuda_kernels, cuda_memory_observation,
    execution_metadata, observe_forward)
from benchmark.providers.transformers_phase2 import GENERATION_OPTIONS
from benchmark.orchestration.provider_smoke import run_provider_context_smoke, size_synthetic_context
from benchmark.orchestration.context_diagnostics import context_sizes, attention_memory_estimate
from benchmark.orchestration.episode import _public_document, OPERATIONAL_PROTOCOL
from benchmark.tools.catalog_v2 import serialize_public_catalog, public_catalog_sha256
from test_phase2_transformers_provider import prepared, torch_stub, device
from test_phase2_provider_smoke import passing_preflight, MODEL

ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, heads, length=6):
        self.heads = heads
        self.shape = (1, len(heads), length, 8)
    def repeat_interleave(self, groups, dim):
        assert dim == 1
        return Tensor([h for h in self.heads for _ in range(groups)], self.shape[2])
    def transpose(self, *args): return self
    def contiguous(self): return self


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.torch = torch_stub(cuda=True)
        self.sdpa = Mock(return_value=Tensor(['result']))
        self.torch.nn.functional = SimpleNamespace(scaled_dot_product_attention=self.sdpa)
        self.patch = patch.dict('sys.modules', {'torch': self.torch})
        self.patch.start(); self.addCleanup(self.patch.stop)

    def test_expand_groups_preserves_order_without_native_gqa(self):
        q, k, v = Tensor(['q'] * 4), Tensor(['k0', 'k1']), Tensor(['v0', 'v1'])
        result, weights = expanded_kv_sdpa(SimpleNamespace(is_causal=True), q, k, v, None, scaling=.25)
        args, options = self.sdpa.call_args
        self.assertIs(args[0], q)
        self.assertEqual(args[1].heads, ['k0', 'k0', 'k1', 'k1'])
        self.assertEqual(args[2].heads, ['v0', 'v0', 'v1', 'v1'])
        self.assertFalse(options['enable_gqa']); self.assertTrue(options['is_causal'])
        self.assertEqual(options['scale'], .25); self.assertIsNone(weights)

    def test_equal_heads_are_not_copied(self):
        q, k = Tensor(['a', 'b']), Tensor(['c', 'd'])
        expanded_kv_sdpa(SimpleNamespace(is_causal=True), q, k, k, None)
        self.assertIs(self.sdpa.call_args.args[1], k)

    def test_cached_single_query_is_not_upper_left_causal(self):
        expanded_kv_sdpa(SimpleNamespace(is_causal=True), Tensor(['q'] * 4, 1), Tensor(['k'] * 2, 9), Tensor(['v'] * 2, 9), None)
        self.assertFalse(self.sdpa.call_args.kwargs['is_causal'])

    def test_padding_mask_preserved_with_same_hf_slice(self):
        mask = Mock(); mask.ndim = 4
        sliced = object(); mask.__getitem__ = Mock(return_value=sliced)
        expanded_kv_sdpa(SimpleNamespace(is_causal=True), Tensor(['q'] * 4), Tensor(['k'] * 2), Tensor(['v'] * 2), mask)
        self.assertIs(self.sdpa.call_args.kwargs['attn_mask'], sliced)
        self.assertFalse(self.sdpa.call_args.kwargs['is_causal'])

    def test_attention_weights_not_materialized(self):
        with self.assertRaises(ValueError):
            expanded_kv_sdpa(SimpleNamespace(), Tensor(['q']), Tensor(['k']), Tensor(['v']), None, output_attentions=True)
        self.sdpa.assert_not_called()

    def test_invalid_head_shapes_rejected(self):
        with self.assertRaises(ValueError):
            expanded_kv_sdpa(SimpleNamespace(), Tensor(['q'] * 3), Tensor(['k'] * 2), Tensor(['v'] * 2), None)

    def test_cuda_math_backend_never_allowed(self):
        with efficient_cuda_kernels(device('cuda:1')): pass
        self.torch.nn.attention.sdpa_kernel.assert_called_once_with(['efficient', 'flash'])

    def test_cpu_does_not_enter_cuda_kernel_context(self):
        with efficient_cuda_kernels(device('cpu')): pass
        self.torch.nn.attention.sdpa_kernel.assert_not_called()

    def test_no_usable_kernel_fails_without_retry(self):
        self.torch.nn.attention.sdpa_kernel.side_effect = RuntimeError('No available kernel')
        adapter, _, model = prepared(where='cuda:0')
        with self.assertRaises(RuntimeError): adapter('synthetic')
        self.assertEqual(model.options, [])
        self.assertEqual(adapter.last_failure.diagnostic['exception_type'], 'RuntimeError')

    def test_alias_uses_exact_original_mask_builder(self):
        attention = SimpleNamespace(register=Mock())
        masks = SimpleNamespace(register=Mock())
        original_mask = object()
        model = SimpleNamespace(config=SimpleNamespace(_attn_implementation='sdpa'))
        model.set_attn_implementation = lambda name: setattr(model.config, '_attn_implementation', name)
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(AttentionInterface=attention),
                'transformers.masking_utils': SimpleNamespace(AttentionMaskInterface=masks,
                    ALL_MASK_ATTENTION_FUNCTIONS={'sdpa': original_mask})}):
            self.assertEqual(configure_memory_attention(model), 'sdpa')
        attention.register.assert_called_once_with(ATTENTION_NAME, expanded_kv_sdpa)
        masks.register.assert_called_once_with(ATTENTION_NAME, original_mask)
        self.assertEqual(model.config._attn_implementation, ATTENTION_NAME)

    def test_memory_observation_on_success(self):
        with cuda_memory_observation(device('cuda:1')) as measured: pass
        self.assertEqual(measured['peak_allocated_bytes'], 5 * 1024**3)
        self.assertEqual(measured['peak_reserved_bytes'], 6 * 1024**3)
        self.assertEqual(measured['free_before_bytes'], 3 * 1024**3)
        self.assertEqual(measured['free_after_bytes'], 3 * 1024**3)
        self.torch.cuda.reset_peak_memory_stats.assert_called_once_with(device('cuda:1'))
        json.dumps(measured, allow_nan=False)

    def test_memory_observation_does_not_hide_oom(self):
        self.torch.cuda.mem_get_info.side_effect = [(3, 12), RuntimeError('device error')]
        with self.assertRaisesRegex(ValueError, 'original failure'):
            with cuda_memory_observation(device('cuda:0')) as measured:
                raise ValueError('original failure')
        self.assertEqual(measured['measurement_error'], 'cuda_memory_unavailable')

    def test_cpu_memory_no_cuda_calls(self):
        with cuda_memory_observation(device('cpu')) as measured: pass
        self.assertIsNone(measured); self.torch.cuda.mem_get_info.assert_not_called()

    def test_cache_metadata_not_changed_or_invented(self):
        config = SimpleNamespace(_attn_implementation=ATTENTION_NAME, use_cache=True, token='private')
        generation = SimpleNamespace(cache_implementation=None, use_cache=True)
        model = SimpleNamespace(config=config, generation_config=generation)
        metadata = execution_metadata(model)
        self.assertIsNone(metadata['cache_implementation'])
        self.assertEqual(metadata['cache_policy'], 'transformers_default')
        self.assertTrue(metadata['use_cache']); self.assertNotIn('private', str(metadata))
        self.assertIsNone(generation.cache_implementation)

    def test_forward_hook_stores_only_shapes_and_cache_class_then_removes(self):
        evidence = {}; remove = Mock()
        def register(callback, **kwargs):
            callback(None, (), {'input_ids': SimpleNamespace(shape=(1, 8631), dtype='int64'),
                'past_key_values': SimpleNamespace(secret='do not retain'), 'logits_to_keep': 1})
            return SimpleNamespace(remove=remove)
        with observe_forward(SimpleNamespace(register_forward_pre_hook=register), evidence): pass
        self.assertEqual(evidence['input_ids']['shape'], [1, 8631])
        self.assertEqual(evidence['actual_cache_class'], 'SimpleNamespace')
        self.assertEqual(evidence['logits_to_keep'], 1)
        self.assertNotIn('secret', str(evidence)); remove.assert_called_once()


class ContextSmokeTests(unittest.TestCase):
    def setUp(self):
        self.torch = torch_stub(cuda=True)
        self.patch = patch.dict('sys.modules', {'torch': self.torch})
        self.patch.start(); self.addCleanup(self.patch.stop)

    def adapter(self, output='synthetic response'):
        return prepared([output], input_count=lambda text: 70 + text.count(' x'), where='cuda:0')

    def smoke(self, adapter, directory='.'):
        return run_provider_context_smoke(MODEL, directory,
            preflight_fn=Mock(return_value=passing_preflight()), provider_loader=Mock(return_value=adapter))

    def test_real_counts_target_one_generation_eight_new_tokens(self):
        adapter, tokenizer, model = self.adapter()
        result = self.smoke(adapter)
        self.assertEqual(result['status'], 'PASS'); self.assertFalse(result['experimental_result'])
        self.assertEqual(result['input_tokens'], 8700)
        self.assertEqual(len(model.options), 1)
        self.assertEqual(model.options[0]['max_new_tokens'], 8)
        self.assertFalse(model.options[0]['do_sample'])
        self.assertEqual(len(adapter.turn_metrics), 1)
        self.assertEqual(len(adapter.context_checks), 1)
        self.assertEqual(result['cuda_memory']['peak_allocated_bytes'], 5 * 1024**3)
        json.dumps(result, allow_nan=False)

    def test_official_callback_still_512_same_visible_document(self):
        adapter, tokenizer, model = self.adapter()
        task = {'id': 'T1', 'prompt': 'Synthetic public task only.'}
        document = _public_document(task, None, serialize_public_catalog)
        from benchmark.tools._sandbox_worker import json_bytes
        text = json_bytes(document).decode()
        adapter(text)
        self.assertEqual(tokenizer.templates[0][0], [{'role': 'user', 'content': text}])
        self.assertEqual(document['protocol'], OPERATIONAL_PROTOCOL)
        self.assertEqual(document['tool_catalog'], json.loads(serialize_public_catalog()))
        self.assertEqual(model.options[0]['max_new_tokens'], 512)
        self.assertEqual(dict(GENERATION_OPTIONS), {'max_new_tokens': 512, 'do_sample': False, 'temperature': None, 'seed': None})
        self.assertFalse(tokenizer.tokenizations[0][1]['truncation'])

    def test_no_task_json_or_official_artifacts(self):
        adapter, _, _ = self.adapter()
        original = Path.read_text
        def deny_tasks(path, *args, **kwargs):
            if path.name == 'benchmark_v1.json': raise AssertionError('no tasks allowed')
            return original(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, 'read_text', deny_tasks):
            result = self.smoke(adapter, directory)
            self.assertEqual(list(Path(directory).rglob('*')), [])
        self.assertEqual(result['status'], 'PASS')

    def test_oom_preserves_details_memory_and_redacts_secret(self):
        class OutOfMemoryError(RuntimeError): pass
        adapter, _, model = self.adapter()
        model.error = OutOfMemoryError('Cannot allocate 8.85 GiB HF_TOKEN=synthetic-secret-value')
        with patch.dict(os.environ, {'HF_TOKEN': 'synthetic-secret-value'}):
            result = self.smoke(adapter)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['failure_details']['exception_type'], 'OutOfMemoryError')
        self.assertEqual(result['failure_details']['stage'], 'model_generate')
        self.assertEqual(result['input_tokens'], 8700)
        self.assertIsNotNone(result['cuda_memory'])
        self.assertNotIn('synthetic-secret-value', json.dumps(result))
        self.assertEqual(len(model.options), 1)
        self.assertIsNone(adapter._tokenizer.inputs)

    def test_context_policy_not_relaxed_by_eight_token_budget(self):
        adapter, _, model = prepared(['unused'], input_count=15873)
        with self.assertRaises(RuntimeError): adapter.generate_diagnostic('synthetic')
        self.assertEqual(adapter.context_checks[0].input_token_limit, 15872)
        self.assertEqual(model.options, [])

    def test_invalid_smoke_target_and_output_budget(self):
        adapter, _, model = self.adapter()
        for target in (511, 15873, True):
            with self.assertRaises(ValueError): size_synthetic_context(adapter, target)
        for budget in (0, 9, 512, True):
            with self.assertRaises(ValueError): adapter.generate_diagnostic('synthetic', max_new_tokens=budget)
        self.assertEqual(model.options, [])

    def test_constant_counter_fails_instead_of_estimating_characters(self):
        adapter, _, model = prepared(['unused'])
        result = self.smoke(adapter)
        self.assertEqual(result['status'], 'FAIL'); self.assertEqual(model.options, [])

    def test_cli_context_smoke_uses_non_experimental_path(self):
        adapter, _, model = self.adapter()
        out = io.StringIO()
        with patch('sys.argv', ['run_phase2.py', '--mode', 'provider-context-smoke', '--model', MODEL]), \
             patch('sys.stdout', out), \
             patch('benchmark.orchestration.provider_smoke.run_preflight', return_value=passing_preflight()), \
             patch('benchmark.orchestration.provider_smoke.load_provider', return_value=adapter), \
             patch.object(cli, 'run_phase2', side_effect=AssertionError('no collection')):
            self.assertEqual(cli.main(), 0)
        self.assertEqual(json.loads(out.getvalue())['mode'], 'provider-context-smoke')
        self.assertEqual(len(model.options), 1)

    def test_preflight_remains_blocking(self):
        loader = Mock()
        result = run_provider_context_smoke(MODEL, preflight_fn=Mock(return_value={'status': 'FAIL',
            'checks': [{'requirement': 'cuda_gpu', 'status': 'FAIL'}]}), provider_loader=loader)
        self.assertEqual(result['status'], 'FAIL'); loader.assert_not_called()

    def test_invalid_target_rejected_before_loading_weights(self):
        loader = Mock()
        result = run_provider_context_smoke(MODEL, input_tokens=15873,
            preflight_fn=Mock(return_value=passing_preflight()), provider_loader=loader)
        self.assertEqual(result['status'], 'FAIL'); loader.assert_not_called()


class ContextEvidenceTests(unittest.TestCase):
    def test_component_sizes_exact_and_not_token_estimates(self):
        document = '{"protocol":"á","tool_catalog":[],"task":"sintético","history":[]}'
        evidence = context_sizes(document)
        self.assertEqual(evidence['document_utf8_bytes'], len(document.encode()))
        self.assertEqual(evidence['components']['tool_catalog']['utf8_bytes'], 2)
        self.assertIsNone(evidence['component_tokens'])
        self.assertIsNone(context_sizes('warm-up text'))

    def test_theoretical_quadratic_allocation_matches_reported_oom(self):
        estimates = attention_memory_estimate(8493, heads=32, kv_heads=8, head_dim=64, layers=16)
        self.assertAlmostEqual(estimates['softmax_fp32_bytes'] / 1024**3, 8.60, places=2)
        self.assertLess(estimates['kv_cache_fp16_max_bytes'] / 1024**3, .3)

    def test_protected_sources_bytes_and_hash_preserved(self):
        names = ['prompts/benchmark_v1.json', 'prompts/benchmark_v1.md', 'benchmark/tools/catalog_v2.py',
            'benchmark/orchestration/episode.py', 'benchmark/providers/transformers_provider.py',
            'benchmark/results.py', 'run_phase1_colab.py']
        for prefix in ('results/raw', 'results/processed/phase1', 'reports/phase1'):
            names += subprocess.check_output(['git', 'ls-files', prefix], cwd=ROOT, text=True).splitlines()
        for name in names:
            self.assertEqual((ROOT/name).read_bytes(), subprocess.check_output(['git', 'show', 'afda967:'+name], cwd=ROOT), name)
        self.assertEqual(public_catalog_sha256(), 'f802970524fa3daadb8e30801aa40ed81995b1b8d0f1fd64053e34b178ddc112')


if __name__ == '__main__':
    unittest.main()
