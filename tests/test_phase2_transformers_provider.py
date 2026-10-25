"""Transformers adapter tests using only in-memory tokenizer/model/torch doubles."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmark.providers.transformers_provider import TransformersProvider
from benchmark.providers.transformers_phase2 import (
    TransformersEpisodeProvider, EpisodeProviderError, OFFICIAL_MODELS,
    GENERATION_OPTIONS, WARMUP_PROMPTS,
)


@dataclass(frozen=True)
class Device:
    type: str = 'cpu'
    index: int | None = None
    def __str__(self):
        return self.type + (':' + str(self.index) if self.index is not None else '')


def device(value):
    if isinstance(value, Device):
        return value
    name, _, index = str(value).partition(':')
    return Device(name, int(index) if index else None)


class Tensor(list):
    @property
    def shape(self): return (1, len(self))
    def to(self, destination):
        self.destination = destination
        return self


class Tokenizer:
    chat_template = 'fake single user template'
    model_max_length = 32768
    init_kwargs = {'_commit_hash': 'fake-tokenizer-revision'}
    def __init__(self, outputs, input_count=7):
        self.outputs = iter(outputs)
        self.input_count = input_count
        self.templates, self.tokenizations, self.decodes = [], [], []
    def apply_chat_template(self, messages, **options):
        self.templates.append((messages, options))
        return '<user>' + messages[0]['content'] + '</user><assistant>'
    def __call__(self, text, **options):
        self.tokenizations.append((text, options))
        count = self.input_count(text) if callable(self.input_count) else self.input_count
        return {'input_ids': Tensor([10] * count), 'attention_mask': Tensor([1] * count)}
    def decode(self, tokens, **options):
        self.decodes.append((list(tokens), options))
        return next(self.outputs)


class Model:
    def __init__(self, where='cpu', capacity=32768, new_tokens=2):
        self.embedding = SimpleNamespace(weight=SimpleNamespace(device=device(where), dtype='float16'))
        self.config = SimpleNamespace(max_position_embeddings=capacity, _commit_hash='fake-model-revision')
        self.hf_device_map = {'': where}
        self.options = []
        self.new_tokens = new_tokens
        self.error = None
        self.after_generate = None
        self.eval = Mock()
    def get_input_embeddings(self): return self.embedding
    def generate(self, **options):
        self.options.append(options)
        if self.error:
            raise self.error
        if self.after_generate:
            self.after_generate()
        return [Tensor(list(options['input_ids']) + [99] * self.new_tokens)]


def torch_stub(cuda=False):
    return SimpleNamespace(
        float16='float16', device=device, no_grad=nullcontext, manual_seed=Mock(),
        nn=SimpleNamespace(attention=SimpleNamespace(
            SDPBackend=SimpleNamespace(EFFICIENT_ATTENTION='efficient', FLASH_ATTENTION='flash'),
            sdpa_kernel=Mock(side_effect=lambda backends: nullcontext()))),
        cuda=SimpleNamespace(is_available=lambda: cuda, synchronize=Mock(),
            mem_get_info=Mock(return_value=(3 * 1024**3, 12 * 1024**3)),
            reset_peak_memory_stats=Mock(), max_memory_allocated=Mock(return_value=5 * 1024**3),
            max_memory_reserved=Mock(return_value=6 * 1024**3),
            current_device=lambda: 0, get_device_name=Mock(side_effect=lambda i: 'Fake GPU ' + str(i)),
            get_device_properties=Mock(return_value=SimpleNamespace(total_memory=12 * 1024**3))))


def prepared(outputs=('answer',), *, input_count=7, where='cpu', capacity=32768, new_tokens=2):
    core = TransformersProvider(OFFICIAL_MODELS[0], 'float16', 'auto')
    core.tokenizer = Tokenizer(outputs, input_count)
    core.model = Model(where, capacity, new_tokens)
    core.model_load_time_seconds = 2.5
    adapter = TransformersEpisodeProvider(core.model_id, provider=core)
    return adapter, core.tokenizer, core.model


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.torch = torch_stub()
        self.patcher = patch.dict('sys.modules', {'torch': self.torch})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_construction_unloaded_all_official_models(self):
        for model_id in OFFICIAL_MODELS:
            adapter = TransformersEpisodeProvider(model_id)
            self.assertIsNone(adapter._core.model)
            self.assertIsNone(adapter._core.tokenizer)
        from run_phase1_colab import MODEL_IDS
        self.assertEqual(set(OFFICIAL_MODELS), set(MODEL_IDS.values()))

    def test_unsupported_id_rejected_without_loading(self):
        with self.assertRaises(ValueError):
            TransformersEpisodeProvider('unapproved-model')

    def test_configuration_rejects_dtype_or_device_override(self):
        for dtype, placement in [('float32', 'auto'), ('float16', 'cpu')]:
            core = TransformersProvider(OFFICIAL_MODELS[0], dtype, placement)
            with self.assertRaises(ValueError):
                TransformersEpisodeProvider(core.model_id, provider=core)

    def test_reuses_explicit_phase1_loading_and_standard_hf_auth(self):
        tokenizer, model = Tokenizer(['answer']), Model()
        tokenizer_loader = Mock(return_value=tokenizer)
        model_loader = Mock(return_value=model)
        hf = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
                             AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader))
        adapter = TransformersEpisodeProvider(OFFICIAL_MODELS[0])
        with patch.dict('sys.modules', {'transformers': hf}), \
             patch('benchmark.providers.transformers_phase2.configure_memory_attention'):
            self.assertIs(adapter.load_tokenizer(), tokenizer)
            self.assertIs(adapter.load_model(), model)
        tokenizer_loader.assert_called_once_with(OFFICIAL_MODELS[0])
        model_loader.assert_called_once_with(OFFICIAL_MODELS[0], dtype='float16', device_map='auto')
        model.eval.assert_called_once()
        self.assertGreaterEqual(adapter._core.model_load_time_seconds, 0)

    def test_loading_exception_sanitized(self):
        adapter = TransformersEpisodeProvider(OFFICIAL_MODELS[0])
        with patch.object(adapter._core, 'load_tokenizer', side_effect=ValueError('HF_PRIVATE_SECRET')):
            with self.assertRaises(EpisodeProviderError) as caught:
                adapter.load_tokenizer()
        self.assertNotIn('HF_PRIVATE_SECRET', str(caught.exception))

    def test_unloaded_preflight(self):
        with self.assertRaises(EpisodeProviderError):
            TransformersEpisodeProvider(OFFICIAL_MODELS[0]).count_tokens('context')

    def test_callback_template_once_and_exact_tokenizer_text(self):
        adapter, tokenizer, model = prepared(['  raw answer \n'])
        document = '{"protocol":"p","tool_catalog":[],"task":"x","history":[]}'
        self.assertEqual(adapter.count_tokens(document), 7)
        self.assertEqual(adapter(document), '  raw answer \n')
        self.assertEqual(len(tokenizer.templates), 1)
        self.assertEqual(len(tokenizer.tokenizations), 1)
        self.assertEqual(len(model.options), 1)
        self.assertEqual(tokenizer.templates[0], ([{'role': 'user', 'content': document}],
                                                 {'tokenize': False, 'add_generation_prompt': True}))
        check = adapter.context_checks[0]
        self.assertEqual(check.document_text, document)
        self.assertEqual(check.tokenizer_text, tokenizer.tokenizations[0][0])
        self.assertEqual(check.tokenizer_text, '<user>' + document + '</user><assistant>')

    def test_same_document_new_turn_has_new_tokenization(self):
        adapter, tokenizer, _ = prepared(['one', 'two'])
        adapter('same')
        adapter('same')
        self.assertEqual(len(tokenizer.templates), 2)
        self.assertEqual(len(tokenizer.tokenizations), 2)
        self.assertEqual([t.turn_index for t in adapter.turn_metrics], [1, 2])

    def test_reuses_phase1_generate_with_official_options(self):
        from run_phase1_colab import GENERATION_OPTIONS as phase1_options
        adapter, _, model = prepared()
        with patch.object(adapter._core, 'generate', wraps=adapter._core.generate) as generate:
            adapter('context')
        self.assertEqual(dict(GENERATION_OPTIONS), phase1_options)
        generate.assert_called_once_with('context', max_new_tokens=512, do_sample=False,
                                         temperature=None, seed=None)
        self.assertEqual(set(model.options[0]), {'input_ids', 'attention_mask', 'max_new_tokens', 'do_sample'})
        self.assertFalse(model.options[0]['do_sample'])
        self.assertEqual(model.options[0]['max_new_tokens'], 512)
        self.torch.manual_seed.assert_not_called()

    def test_decode_only_new_tokens_and_no_cleanup(self):
        adapter, tokenizer, _ = prepared([' \n raw output!  '], new_tokens=3)
        self.assertEqual(adapter('context'), ' \n raw output!  ')
        self.assertEqual(tokenizer.decodes, [([99, 99, 99], {'skip_special_tokens': True,
                                                          'clean_up_tokenization_spaces': False})])

    def test_tokenization_never_truncates_or_adds_template_tokens_again(self):
        adapter, tokenizer, _ = prepared()
        adapter('context')
        self.assertEqual(tokenizer.tokenizations[0][1], {'return_tensors': 'pt',
                        'add_special_tokens': False, 'truncation': False})

    def test_exact_input_limit_accepted(self):
        adapter, _, model = prepared(input_count=15872)
        self.assertEqual(adapter('context'), 'answer')
        self.assertEqual(adapter.turn_metrics[0].input_tokens, 15872)
        self.assertFalse(adapter.context_checks[0].context_limit)
        self.assertEqual(len(model.options), 1)

    def test_over_limit_rejected_even_direct_callback(self):
        adapter, _, model = prepared(input_count=15873)
        with self.assertRaises(EpisodeProviderError) as caught:
            adapter('context')
        self.assertEqual(caught.exception.code, 'context_limit')
        self.assertEqual(model.options, [])
        self.assertTrue(adapter.context_checks[0].context_limit)
        self.assertEqual(adapter.context_checks[0].input_token_limit, 15872)
        self.assertEqual(adapter.turn_metrics, [])

    def test_declared_model_capacity_too_small_blocks_preflight(self):
        adapter, _, model = prepared(capacity=8192)
        with self.assertRaises(EpisodeProviderError) as caught:
            adapter('context')
        self.assertEqual(caught.exception.code, 'context_capacity_insufficient')
        self.assertEqual(model.options, [])

    def test_tokenizer_capacity_also_limits_context(self):
        adapter, tokenizer, _ = prepared()
        tokenizer.model_max_length = 8192
        with self.assertRaises(EpisodeProviderError):
            adapter.validate_ready()

    def test_huge_tokenizer_sentinel_is_not_real_capacity(self):
        adapter, tokenizer, _ = prepared(capacity=32768)
        tokenizer.model_max_length = 10**30
        self.assertEqual(adapter.validate_ready(), 32768)

    def test_unknown_capacity_blocks_preflight(self):
        adapter, tokenizer, model = prepared()
        model.config.max_position_embeddings = None
        tokenizer.model_max_length = 10**30
        with self.assertRaises(EpisodeProviderError) as caught:
            adapter.validate_ready()
        self.assertEqual(caught.exception.code, 'context_capacity_unverified')

    def test_missing_chat_template_is_not_plain_text_fallback(self):
        adapter, tokenizer, _ = prepared()
        tokenizer.chat_template = None
        with self.assertRaises(EpisodeProviderError):
            adapter('context')

    def test_effective_dtype_mismatch_no_fallback(self):
        adapter, _, model = prepared()
        model.embedding.weight.dtype = 'float32'
        with self.assertRaises(EpisodeProviderError) as caught:
            adapter('context')
        self.assertEqual(caught.exception.code, 'dtype_mismatch')
        self.assertEqual(model.options, [])

    def test_turn_counts_latency_and_run_totals(self):
        adapter, _, _ = prepared(['a', 'b'], new_tokens=3)
        with patch('benchmark.providers.transformers_provider.time.perf_counter', side_effect=[1, 1.25, 2, 2.5]):
            adapter('first')
            adapter('second')
        one, two = adapter.turn_metrics
        self.assertEqual(one.input_tokens, 7)
        self.assertEqual(one.output_tokens, 3)
        self.assertEqual(one.inference_latency_seconds, .25)
        self.assertEqual(two.inference_latency_seconds, .5)
        self.assertFalse(one.hit_max_new_tokens)
        meta = adapter.run_metadata()
        self.assertEqual(meta['total_input_tokens'], 14)
        self.assertEqual(meta['total_output_tokens'], 6)
        self.assertEqual(meta['total_generation_latency_seconds'], .75)
        self.assertEqual(meta['model_load_time_seconds'], 2.5)

    def test_hit_max_new_tokens(self):
        adapter, _, _ = prepared(new_tokens=512)
        adapter('context')
        self.assertTrue(adapter.turn_metrics[0].hit_max_new_tokens)
        self.assertEqual(adapter.turn_metrics[0].output_tokens, 512)

    def test_generation_error_not_a_fabricated_response(self):
        adapter, _, model = prepared()
        model.error = ValueError('PRIVATE_HF_TOKEN /path')
        with self.assertRaises(EpisodeProviderError) as caught:
            adapter('context')
        self.assertEqual(caught.exception.code, 'generation_failure')
        self.assertNotIn('PRIVATE_HF_TOKEN', str(caught.exception))
        self.assertEqual(adapter.turn_metrics[0].status, 'error')
        self.assertIsNone(adapter.turn_metrics[0].raw_output)
        self.assertEqual(adapter.run_metadata()['failed_turns'], 1)

    def test_cpu_never_synchronizes_cuda(self):
        adapter, _, _ = prepared()
        adapter('context')
        self.torch.cuda.synchronize.assert_not_called()
        self.assertEqual(adapter.run_metadata()['device'], 'cpu')
        self.assertFalse(adapter.run_metadata()['offload_observed'])

    def test_cuda_uses_explicit_actual_device_for_sync(self):
        self.torch.cuda.is_available = lambda: True
        adapter, _, model = prepared(where='cuda:1')
        adapter('context')
        self.assertEqual([c.args for c in self.torch.cuda.synchronize.call_args_list],
                         [(device('cuda:1'),), (device('cuda:1'),)])
        self.assertEqual(model.options[0]['input_ids'].destination, device('cuda:1'))
        metadata = adapter.run_metadata()
        self.assertEqual(metadata['gpu_name'], 'Fake GPU 1')
        self.assertEqual(metadata['vram_gb'], 12)
        self.assertTrue(metadata['cuda_available'])

    def test_accelerate_hook_device_and_offloaded_modules(self):
        self.torch.cuda.is_available = lambda: True
        adapter, _, model = prepared(where='meta')
        model.embedding._hf_hook = SimpleNamespace(execution_device='cuda:1')
        model.hf_device_map = {'embedding': 'cpu', 'blocks': 1, 'head': 'disk'}
        adapter('context')
        metadata = adapter.run_metadata()
        self.assertEqual(model.options[0]['input_ids'].destination, device('cuda:1'))
        self.assertEqual(metadata['device_map'], {'blocks': 'cuda:1', 'embedding': 'cpu', 'head': 'disk'})
        self.assertEqual(metadata['offloaded_modules'], ['embedding', 'head'])
        self.assertTrue(metadata['offload_observed'])
        self.assertIs(adapter._core.model, model)

    def test_sequential_accelerate_hooks_resolve_actual_input_device(self):
        self.torch.cuda.is_available = lambda: True
        adapter, _, model = prepared(where='meta')
        model.embedding._hf_hook = SimpleNamespace(hooks=[SimpleNamespace(),
            SimpleNamespace(execution_device='cuda:1')])
        adapter('context')
        self.assertEqual(model.options[0]['input_ids'].destination, device('cuda:1'))

    def test_meta_device_without_execution_hook_is_not_cpu_fallback(self):
        adapter, _, model = prepared(where='meta')
        with self.assertRaises(EpisodeProviderError):
            adapter('context')
        self.assertEqual(model.options, [])
        self.assertIs(adapter._core.model, model)

    def test_revisions_versions_are_allowlisted(self):
        adapter, tokenizer, model = prepared()
        tokenizer.init_kwargs = {'token': 'PRIVATE_HF', '_commit_hash': 'tokenizer-rev'}
        model.config.token = 'PRIVATE_HF'
        with patch('benchmark.providers.transformers_phase2.version', return_value='fake-version'):
            meta = adapter.run_metadata()
        self.assertEqual(meta['model_revision'], 'fake-model-revision')
        self.assertEqual(meta['tokenizer_revision'], 'tokenizer-rev')
        self.assertEqual(meta['torch_version'], 'fake-version')
        self.assertEqual(meta['accelerate_version'], 'fake-version')
        self.assertNotIn('PRIVATE_HF', str(meta))

    def test_environment_secret_not_logged_or_returned(self):
        adapter, _, model = prepared()
        model.error = ValueError('FICTITIOUS_PRIVATE_TOKEN')
        output = io.StringIO()
        with patch.dict(os.environ, {'HF_TOKEN': 'FICTITIOUS_PRIVATE_TOKEN'}), patch('sys.stdout', output):
            with self.assertRaises(EpisodeProviderError):
                adapter('context')
        self.assertEqual(output.getvalue(), '')
        self.assertNotIn('FICTITIOUS_PRIVATE_TOKEN', str(adapter.run_metadata()))
        self.assertNotIn('FICTITIOUS_PRIVATE_TOKEN', str(adapter.turn_metrics))

    def test_structural_determinism_except_latency(self):
        results = []
        for _ in range(2):
            adapter, _, _ = prepared(['raw'])
            adapter('context')
            metrics = asdict(adapter.turn_metrics[0])
            metrics.pop('inference_latency_seconds')
            results.append((adapter.context_checks, metrics))
        self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
