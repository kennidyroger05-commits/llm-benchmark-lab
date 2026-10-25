"""Opt-in CLI/provider smoke tests, with in-memory weights and no HF downloads."""
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
from benchmark.orchestration.provider_smoke import run_provider_smoke, synthetic_document
from benchmark.orchestration.phase2_preflight_v2 import run_preflight
from benchmark.tools.catalog_v2 import public_catalog_sha256
from test_phase2_operations import probes
from test_phase2_transformers_provider import prepared, torch_stub, Model, Tokenizer, device
from benchmark.providers.transformers_phase2 import OFFICIAL_MODELS

MODEL = OFFICIAL_MODELS[0]
ROOT = Path(__file__).resolve().parents[1]
SECRET = 'synthetic-smoke-credential-918274'
REPLY = '{"type":"final_answer","answer":"SMOKE_OK"}'


def passing_preflight():
    return {'status': 'PASS', 'checks': [], 'model_revision': 'synthetic-revision'}


class ProviderSmokeTests(unittest.TestCase):
    def setUp(self):
        self.torch = torch_stub(cuda=True)
        self.patcher = patch.dict('sys.modules', {'torch': self.torch})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def smoke(self, adapter, directory='.'):
        return run_provider_smoke(MODEL, directory,
            preflight_fn=Mock(return_value=passing_preflight()),
            provider_loader=Mock(return_value=adapter))

    def test_success_serialization_with_real_provider_generation_path(self):
        adapter, tokenizer, model = prepared([REPLY], where='cuda:0', new_tokens=9)
        with patch.object(adapter._core, 'generate', wraps=adapter._core.generate) as generate:
            result = self.smoke(adapter)
        generate.assert_called_once_with(synthetic_document(), max_new_tokens=512,
                                        do_sample=False, temperature=None, seed=None)
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(result, {'mode': 'provider-smoke', 'experimental_result': False,
            'model_id': MODEL, 'status': 'PASS', 'stage': 'provider_generation',
            'device': 'cuda:0', 'dtype': 'float16', 'input_tokens': 7, 'output_tokens': 9,
            'latency_seconds': adapter.turn_metrics[0].inference_latency_seconds,
            'output_preview': REPLY})
        self.assertGreaterEqual(result['latency_seconds'], 0)
        self.assertEqual(len(tokenizer.templates), 1)
        self.assertEqual(len(tokenizer.tokenizations), 1)
        self.assertEqual(len(model.options), 1)
        self.assertEqual(set(model.options[0]), {'input_ids', 'attention_mask', 'max_new_tokens', 'do_sample'})
        self.assertEqual(model.options[0]['input_ids'].destination, device('cuda:0'))
        self.assertEqual(model.options[0]['attention_mask'].destination, device('cuda:0'))
        self.assertEqual(tokenizer.decodes, [([99] * 9, {'skip_special_tokens': True,
                                                       'clean_up_tokenization_spaces': False})])
        self.assertEqual([call.args for call in self.torch.cuda.synchronize.call_args_list],
                         [(device('cuda:0'),), (device('cuda:0'),)])

    def test_shared_production_loader_pins_revision_and_loads_explicitly(self):
        # No fake provider class: only the external HF/torch boundaries are fake.
        tokenizer, model = Tokenizer([REPLY]), Model(where='cuda:0')
        snapshot = Mock(return_value='/tmp/synthetic-local-snapshot')
        token_loader, model_loader = Mock(return_value=tokenizer), Mock(return_value=model)
        modules = {'huggingface_hub': SimpleNamespace(snapshot_download=snapshot),
                   'transformers': SimpleNamespace(
                       AutoTokenizer=SimpleNamespace(from_pretrained=token_loader),
                       AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader))}
        with patch.dict('sys.modules', modules):
            result = run_provider_smoke(MODEL, preflight_fn=Mock(return_value=passing_preflight()))
        self.assertEqual(result['status'], 'PASS')
        snapshot.assert_called_once_with(MODEL, revision='synthetic-revision', allow_patterns=[
            '*.json', '*.safetensors', 'tokenizer.model', '*.tiktoken'])
        token_loader.assert_called_once_with('/tmp/synthetic-local-snapshot')
        model_loader.assert_called_once_with('/tmp/synthetic-local-snapshot', dtype='float16', device_map='auto')
        model.eval.assert_called_once()
        self.assertEqual(len(model.options), 1)

    def test_exactly_one_generation_no_warmups_no_retry(self):
        adapter, _, model = prepared([REPLY])
        with patch.object(adapter, 'warm_up', side_effect=AssertionError('warm-up forbidden')):
            self.assertEqual(self.smoke(adapter)['status'], 'PASS')
        self.assertEqual(len(model.options), 1)
        self.assertEqual(len(adapter.turn_metrics), 1)
        self.assertFalse(adapter.warmup_completed)

    def test_no_task_loading_sessions_evaluator_or_official_runner(self):
        adapter, _, _ = prepared([REPLY])
        read_text = Path.read_text
        def deny_task_json(path, *args, **kwargs):
            if path.name == 'benchmark_v1.json':
                raise AssertionError('Smoke must never load benchmark task JSON')
            return read_text(path, *args, **kwargs)
        with patch.object(Path, 'read_text', deny_task_json), \
             patch('benchmark.orchestration.phase2_runner_v2.shared_runner', side_effect=AssertionError('no collection')), \
             patch('benchmark.tools.domain_tools.DomainSession.from_benchmark', side_effect=AssertionError('no fixtures')), \
             patch('benchmark.orchestration.phase2_runner_v2.evaluate_episode', side_effect=AssertionError('no evaluation')):
            result = self.smoke(adapter)
        self.assertEqual(result['status'], 'PASS')

    def test_normal_preflight_checks_hashes_without_loading_task_json(self):
        adapter, _, _ = prepared([REPLY])
        checks = probes(); checks['restricted_python_runtime'] = lambda: True
        checks.pop('benchmark_hash'); checks.pop('catalog_hash'); checks.pop('frozen_files')
        checks.pop('output_writable')
        preflight = Mock(side_effect=lambda *args, **kwargs: run_preflight(*args, **kwargs, probes=checks))
        read_text = Path.read_text
        def deny_task_json(path, *args, **kwargs):
            if path.name == 'benchmark_v1.json':
                raise AssertionError('Only byte integrity checks are permitted')
            return read_text(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, 'read_text', deny_task_json):
            result = run_provider_smoke(MODEL, directory, preflight_fn=preflight,
                                        provider_loader=Mock(return_value=adapter))
            self.assertEqual(list(Path(directory).rglob('*')), [])
        self.assertEqual(result['status'], 'PASS')
        preflight.assert_called_once_with(MODEL, directory, inspect_model=True)

    def test_no_official_result_or_artifact_created(self):
        adapter, _, _ = prepared([REPLY])
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.smoke(adapter, directory)['status'], 'PASS')
            self.assertEqual(list(Path(directory).rglob('*')), [])
            self.assertFalse((Path(directory) / 'results/raw/phase2').exists())

    def test_synthetic_document_has_no_catalog_tasks_or_private_payload(self):
        document = json.loads(synthetic_document())
        self.assertEqual(list(document), ['protocol', 'tool_catalog', 'task', 'history'])
        self.assertEqual(document['tool_catalog'], [])
        self.assertEqual(document['history'], [])
        self.assertIn('SMOKE_OK', document['task'])
        for private in ('benchmark_v1', 'tool_fixtures', 'expected_answer', 'raw_prompt', 'T1', 'T10'):
            self.assertNotIn(private, synthetic_document())

    def test_non_empty_text_suffices_without_semantic_or_json_grading(self):
        adapter, _, _ = prepared(['arbitrary non-JSON response'])
        self.assertEqual(self.smoke(adapter)['status'], 'PASS')

    def test_empty_generation_fails_without_retry(self):
        for text in ('', ' \n\t '):
            adapter, _, model = prepared([text])
            result = self.smoke(adapter)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['stage'], 'provider_generation')
            self.assertEqual(result['exception_type'], 'ValueError')
            self.assertEqual(len(model.options), 1)

    def test_provider_failure_serialization_preserves_sanitized_diagnostic(self):
        adapter, _, model = prepared([REPLY])
        model.error = TypeError('Unexpected generation argument HF_TOKEN=' + SECRET + ' /private/weights')
        with patch.dict(os.environ, {'HF_TOKEN': SECRET}):
            result = self.smoke(adapter)
        payload = json.dumps(result)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['stage'], 'model_generate')
        self.assertEqual(result['exception_type'], 'TypeError')
        self.assertIn('Unexpected generation argument', result['exception_message'])
        self.assertTrue(result['secret_redacted'])
        self.assertFalse(result['experimental_result'])
        self.assertNotIn(SECRET, payload)
        self.assertNotIn('HF_TOKEN', payload)
        self.assertNotIn('/private/weights', payload)
        self.assertNotIn('Traceback', payload)
        self.assertEqual(len(model.options), 1)

    def test_loader_exception_sanitized(self):
        with patch.dict(os.environ, {'HF_TOKEN': SECRET}):
            result = run_provider_smoke(MODEL, preflight_fn=Mock(return_value=passing_preflight()),
                provider_loader=Mock(side_effect=ValueError('Cannot load HF_TOKEN=' + SECRET)))
        self.assertEqual(result['stage'], 'model_load')
        self.assertEqual(result['exception_type'], 'ValueError')
        self.assertNotIn(SECRET, json.dumps(result))

    def test_tokenization_failure_sanitized(self):
        adapter, tokenizer, model = prepared([REPLY])
        tokenizer.apply_chat_template = Mock(side_effect=TypeError('Template unsupported'))
        result = self.smoke(adapter)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['stage'], 'tokenization')
        self.assertEqual(result['exception_type'], 'TypeError')
        self.assertEqual(model.options, [])

    def test_preview_redacts_secrets_and_authorization_headers(self):
        adapter, _, _ = prepared(['HF_TOKEN=' + SECRET + '\nAuthorization: Bearer ' + SECRET])
        with patch.dict(os.environ, {'HF_TOKEN': SECRET}):
            result = self.smoke(adapter)
        self.assertEqual(result['status'], 'PASS')
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertNotIn('HF_TOKEN', json.dumps(result))
        self.assertNotIn('Authorization:', json.dumps(result))

    def test_preview_length_bounded(self):
        adapter, _, _ = prepared(['synthetic ' * 100])
        self.assertLessEqual(len(self.smoke(adapter)['output_preview']), 256)

    def test_context_limit_blocks_generate(self):
        adapter, _, model = prepared([REPLY], input_count=15873)
        result = self.smoke(adapter)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['stage'], 'tokenization')
        self.assertEqual(model.options, [])

    def test_preflight_failure_blocks_loading_for_every_check(self):
        for name in ('hf_auth', 'cuda_gpu', 'benchmark_hash', 'restricted_python_runtime', 'no_conflicting_run'):
            loader = Mock(side_effect=AssertionError('blocked before any model loading'))
            result = run_provider_smoke(MODEL, preflight_fn=Mock(return_value={
                'status': 'FAIL', 'checks': [{'requirement': name, 'status': 'FAIL'}]}), provider_loader=loader)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['stage'], 'preflight')
            self.assertIn('preflight_failed', result['exception_message'])
            loader.assert_not_called()

    def test_preflight_exception_sanitized_without_loading(self):
        loader = Mock()
        with patch.dict(os.environ, {'HF_TOKEN': SECRET}):
            result = run_provider_smoke(MODEL, preflight_fn=Mock(side_effect=RuntimeError('Bearer ' + SECRET)), provider_loader=loader)
        self.assertEqual(result['stage'], 'preflight')
        self.assertNotIn(SECRET, str(result))
        loader.assert_not_called()

    def test_cli_mode_success_json_and_exit_code(self):
        adapter, _, model = prepared([REPLY])
        output = io.StringIO()
        with patch('sys.argv', ['run_phase2.py', '--mode', 'provider-smoke', '--model', MODEL]), \
             patch('sys.stdout', output), \
             patch('benchmark.orchestration.provider_smoke.run_preflight', return_value=passing_preflight()) as preflight, \
             patch('benchmark.orchestration.provider_smoke.load_provider', return_value=adapter), \
             patch.object(cli, 'run_phase2', side_effect=AssertionError('no benchmark')):
            code = cli.main()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'PASS')
        preflight.assert_called_once_with(MODEL, '.', inspect_model=True)
        self.assertEqual(len(model.options), 1)

    def test_cli_failure_json_and_exit_code_without_secret_or_traceback(self):
        adapter, _, model = prepared([REPLY])
        model.error = ValueError('HF_TOKEN=' + SECRET)
        output = io.StringIO()
        with patch('sys.argv', ['run_phase2.py', '--mode', 'provider-smoke', '--model', MODEL]), \
             patch('sys.stdout', output), patch.dict(os.environ, {'HF_TOKEN': SECRET}), \
             patch('benchmark.orchestration.provider_smoke.run_preflight', return_value=passing_preflight()), \
             patch('benchmark.orchestration.provider_smoke.load_provider', return_value=adapter):
            code = cli.main()
        result = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(result['exception_type'], 'ValueError')
        self.assertTrue(result['secret_redacted'])
        self.assertNotIn(SECRET, output.getvalue())
        self.assertNotIn('Traceback', output.getvalue())

    def test_default_cli_still_environment_never_loads_smoke_model(self):
        with patch('sys.argv', ['run_phase2.py', '--model', MODEL]), patch('sys.stdout', io.StringIO()), \
             patch.object(cli, 'run_preflight', return_value={'status': 'FAIL'}) as preflight, \
             patch.object(cli, 'run_provider_smoke') as smoke:
            self.assertEqual(cli.main(), 1)
        preflight.assert_called_once_with(MODEL, '.', inspect_model=False)
        smoke.assert_not_called()

    def test_benchmark_catalog_and_phase1_preserved(self):
        names = ['prompts/benchmark_v1.json', 'prompts/benchmark_v1.md',
                 'benchmark/tools/catalog_v2.py', 'benchmark/providers/transformers_provider.py',
                 'benchmark/results.py', 'run_phase1_colab.py']
        for prefix in ('results/raw', 'results/processed/phase1', 'reports/phase1'):
            names += subprocess.check_output(['git', 'ls-files', prefix], cwd=ROOT, text=True).splitlines()
        for name in names:
            self.assertEqual((ROOT / name).read_bytes(), subprocess.check_output(
                ['git', 'show', '282a8b2:' + name], cwd=ROOT), name)
        self.assertEqual(public_catalog_sha256(), 'f802970524fa3daadb8e30801aa40ed81995b1b8d0f1fd64053e34b178ddc112')


if __name__ == '__main__':
    unittest.main()
