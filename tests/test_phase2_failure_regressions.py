"""Synthetic Phase 2 generation/RAW/publication regressions; no model downloads."""
from copy import deepcopy
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark.evaluation.phase2_v2 import build_report, evaluate_episode
from benchmark.evaluation.phase2_export import report_bytes
from benchmark.orchestration.episode import EpisodeResult, run_episode
from benchmark.orchestration.phase2_runtime import run_transformers_episode
from benchmark.orchestration.phase2_runner_v2 import run_phase2, V2Supervisor
from benchmark.providers.failure_diagnostics import failure_details
from benchmark.providers.transformers_phase2 import EpisodeProviderError, OFFICIAL_MODELS
from benchmark.tools.catalog_v2 import public_catalog_sha256
from test_phase2_evaluation import simple_task
from test_phase2_runtime import task, final
from test_phase2_transformers_provider import prepared, torch_stub

MODEL = OFFICIAL_MODELS[0]
ROOT = Path(__file__).resolve().parents[1]
SECRET = 'synthetic-private-hf-value-928364'


def failed_worker(settings, owner):
    def handle(index):
        raise ValueError('Invalid generation argument: temperature=None; HF_TOKEN=' + os.environ['HF_TOKEN'])
    return handle


def failed_startup(settings, owner):
    raise ValueError('Loader rejected an argument; HF_TOKEN=' + os.environ['HF_TOKEN'])


def unpicklable_worker(settings, owner):
    class Result:
        def __reduce__(self):
            raise ValueError('Serialization failed; HF_TOKEN=' + os.environ['HF_TOKEN'])
    return lambda index: Result()


def successful_rows():
    ep = EpisodeResult(technical_success=True, protocol_correct=True,
        termination_reason='final_answer', final_answer=[{'sku': 'toy', 'available': 3}],
        direct_answer=True, metadata={'model_id': MODEL, 'run_id': 'synthetic-run'})
    row = evaluate_episode(simple_task('T4'), ep)
    return [{**deepcopy(row), 'task_id': f'T{i}'} for i in range(1, 11)]


def report(rows=None, **options):
    return build_report(successful_rows() if rows is None else rows, model_id=MODEL,
                        run_metadata={'run_id': 'synthetic-run'}, **options)


class GenerationFailureTests(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.dict('sys.modules', {'torch': torch_stub()})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_transformers457_signature_accepts_frozen_greedy_kwargs(self):
        """Match GenerationMixin.generate's 4.57 API; reject unplanned kwargs."""
        adapter, tokenizer, model = prepared([' raw\n'], new_tokens=3)
        original = model.generate
        model.generation_config = SimpleNamespace(temperature=.6, top_p=.9, eos_token_id=[2, 3], pad_token_id=None)
        before = deepcopy(vars(model.generation_config))
        def generate(inputs=None, generation_config=None, logits_processor=None,
                     stopping_criteria=None, use_model_defaults=None, **kwargs):
            self.assertEqual(set(kwargs), {'input_ids', 'attention_mask', 'max_new_tokens', 'do_sample'})
            self.assertFalse(kwargs['do_sample'])
            self.assertEqual(kwargs['max_new_tokens'], 512)
            self.assertEqual(len(kwargs['attention_mask']), len(kwargs['input_ids']))
            self.assertIsNone(generation_config)
            return original(**kwargs)
        model.generate = generate
        self.assertEqual(adapter('Synthetic task, no benchmark data.'), ' raw\n')
        self.assertEqual(vars(model.generation_config), before)
        self.assertEqual(len(tokenizer.templates), 1)
        self.assertEqual(adapter.turn_metrics[0].output_tokens, 3)

    def test_warmups_then_long_synthetic_generation_succeeds(self):
        adapter, _, model = prepared(['one', 'two', final()], input_count=lambda text: 8631 if 'protocol' in text else 7)
        adapter.warm_up()
        result = run_transformers_episode(task(), adapter)
        self.assertTrue(result.episode.technical_success)
        self.assertEqual(result.turn_metrics[0].input_tokens, 8631)
        self.assertEqual(result.run_metadata['successful_turns'], 1)
        self.assertEqual(len(model.options), 3)

    def test_real_generate_exception_type_message_stage_preserved(self):
        adapter, _, model = prepared()
        model.error = ValueError('Unsupported generation option: temperature=None')
        result = run_transformers_episode(task(), adapter)
        diagnostic = result.episode.failure_details
        self.assertEqual(diagnostic, {'exception_type': 'ValueError',
            'exception_message': 'Unsupported generation option: temperature=None',
            'stage': 'model_generate', 'task_id': 'T1', 'turn_index': 1})
        self.assertEqual(result.turn_metrics[0].failure, diagnostic)
        self.assertEqual(result.episode.termination_reason, 'model_failure')
        self.assertIsNone(result.episode.protocol_correct)

    def test_failure_after_valid_first_turn_has_index_two(self):
        adapter, tokenizer, model = prepared()
        outputs = iter(['{"type":"tool_call","tool":"calculator_basic","arguments":{"expression":"1+2"}}'])
        def decode(*args, **kwargs):
            model.error = TypeError('Synthetic API failure')
            return next(outputs)
        tokenizer.decode = decode
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.episode.failure_details['turn_index'], 2)
        self.assertTrue(result.episode.protocol_correct)
        self.assertEqual(result.episode.model_outputs_total, 1)

    def test_tokenization_error_preserved(self):
        adapter, tokenizer, _ = prepared()
        tokenizer.apply_chat_template = lambda *a, **k: (_ for _ in ()).throw(TypeError('Template argument not supported'))
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.episode.failure_details['exception_type'], 'TypeError')
        self.assertEqual(result.episode.failure_details['stage'], 'tokenization')
        self.assertEqual(result.episode.termination_reason, 'context_preflight_failure')

    def test_decode_error_preserved(self):
        adapter, tokenizer, _ = prepared()
        tokenizer.decode = lambda *a, **k: (_ for _ in ()).throw(TypeError('Decode API mismatch'))
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.episode.failure_details['stage'], 'decode')
        self.assertEqual(result.episode.failure_details['exception_type'], 'TypeError')

    def test_input_placement_error_preserved(self):
        adapter, _, _ = prepared()
        adapter.count_tokens('synthetic')
        tensor = adapter._tokenizer.inputs['input_ids']
        tensor.to = lambda device: (_ for _ in ()).throw(RuntimeError('Device placement failed'))
        with self.assertRaises(EpisodeProviderError) as caught:
            adapter('synthetic')
        self.assertEqual(caught.exception.diagnostic['stage'], 'input_placement')
        self.assertEqual(caught.exception.diagnostic['exception_type'], 'RuntimeError')

    def test_zero_successful_turns_serialize_safe_diagnostics(self):
        adapter, _, model = prepared(['warm one', 'warm two'])
        adapter.warm_up()
        model.error = RuntimeError('Synthetic failure HF_TOKEN=' + SECRET + ' at /private/model/path')
        out = io.StringIO()
        with patch.dict(os.environ, {'HF_TOKEN': SECRET}), patch('sys.stdout', out):
            results = [run_transformers_episode(task(f'T{i}'), adapter) for i in range(1, 11)]
        payload = json.dumps([asdict(result) for result in results])
        self.assertNotIn(SECRET, payload)
        self.assertNotIn('/private/model/path', payload)
        self.assertEqual(out.getvalue(), '')
        self.assertEqual(results[-1].run_metadata['successful_turns'], 0)
        self.assertEqual(results[-1].run_metadata['failed_turns'], 10)
        self.assertEqual(results[-1].run_metadata['total_output_tokens'], 0)
        for i, result in enumerate(results, 1):
            self.assertEqual(result.episode.failure_details['task_id'], f'T{i}')
            self.assertTrue(result.episode.infrastructure_failure)
            self.assertFalse(result.episode.technical_success)

    def test_invalid_json_still_successful_generation(self):
        adapter, _, _ = prepared(['invalid JSON'])
        result = run_transformers_episode(task(), adapter)
        self.assertTrue(result.episode.technical_success)
        self.assertFalse(result.episode.protocol_correct)
        self.assertIsNone(result.episode.failure_details)
        self.assertEqual(result.run_metadata['successful_turns'], 1)

    def test_generic_callback_error_preserved_without_history_leak(self):
        error = ValueError('Synthetic callback incompatibility')
        result = run_episode(task(), lambda _: (_ for _ in ()).throw(error))
        self.assertEqual(result.failure_details['stage'], 'model_callback')
        self.assertNotIn('Synthetic callback incompatibility', str(result.full_history))


class PublicationGuardTests(unittest.TestCase):
    def test_ten_successful_tasks_allow_publication(self):
        result = report(collection_status='complete')
        self.assertTrue(result['publication_ready'])
        self.assertTrue(result['experimental_valid'])

    def test_all_tasks_infrastructure_failure_block_publication(self):
        rows = successful_rows()
        for row in rows:
            row.update(technical_success=False, protocol_correct=None, infrastructure_failure=True,
                       termination_reason='model_failure', task_correct=False, strict_correct=False)
        result = report(rows, collection_status='complete')
        self.assertFalse(result['publication_ready'])
        self.assertFalse(result['experimental_valid'])
        self.assertEqual(result['collection_status'], 'complete')
        self.assertEqual(result['summary']['technical_success_count'], 0)

    def test_single_infrastructure_failure_blocks(self):
        rows = successful_rows(); rows[0]['infrastructure_failure'] = True
        self.assertFalse(report(rows)['publication_ready'])

    def test_technical_failure_blocks_even_without_infra_flag(self):
        rows = successful_rows(); rows[0]['technical_success'] = False
        self.assertFalse(report(rows)['publication_ready'])

    def test_incomplete_collection_blocks(self):
        self.assertFalse(report(collection_status='incomplete')['publication_ready'])

    def test_missing_task_blocks(self):
        self.assertFalse(report(successful_rows()[:-1], collection_status='complete')['publication_ready'])

    def test_pending_isolated_execution_blocks(self):
        rows = successful_rows(); rows[6]['evaluation_pending'] = True
        self.assertFalse(report(rows)['publication_ready'])
        self.assertFalse(report(rows)['evaluation_complete'])

    def test_pending_diagnostic_alone_blocks(self):
        rows = successful_rows(); rows[7]['evaluation_error'] = 'pending_isolated_code_evaluation'
        self.assertFalse(report(rows)['publication_ready'])
        self.assertFalse(report(rows)['evaluation_complete'])

    def test_collection_cleanup_failure_blocks(self):
        self.assertFalse(report(collection_status='complete', collection_failure='cleanup_failure')['publication_ready'])

    def test_source_raw_incomplete_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.json'; path.write_text('{"status":"incomplete"}')
            self.assertFalse(report(source_raw_path=path, collection_status='complete')['publication_ready'])

    def test_source_raw_infrastructure_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.json'
            path.write_text('{"status":"complete","tasks":[{"episode":{"infrastructure_failure":true}}]}')
            self.assertFalse(report(source_raw_path=path)['publication_ready'])

    def test_invalid_model_protocol_remains_valid_experimental_observation(self):
        rows = successful_rows()
        rows[0].update(protocol_correct=False, task_correct=False, strict_correct=False,
                       termination_reason='invalid_protocol')
        self.assertTrue(report(rows)['publication_ready'])

    def test_context_limit_is_not_misclassified_as_infrastructure(self):
        rows = successful_rows()
        rows[0].update(technical_success=False, protocol_correct=None, task_correct=False,
                       strict_correct=False, termination_reason='context_limit_exceeded')
        self.assertTrue(report(rows)['publication_ready'])

    def test_guard_output_deterministic(self):
        self.assertEqual(report_bytes(report()), report_bytes(report()))


class FailureBoundaryTests(unittest.TestCase):
    def test_supervisor_preserves_task_exception_in_real_child(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'HF_TOKEN': SECRET}):
            worker = V2Supervisor(failed_worker, {}, directory)
            event = worker.execute(0, timeout=5)
            worker.close()
        self.assertEqual(event['supervision'], 'infrastructure_failure')
        self.assertEqual(event['failure_details']['exception_type'], 'ValueError')
        self.assertEqual(event['failure_details']['stage'], 'worker_task')
        self.assertEqual(event['failure_details']['task_id'], 'T1')
        self.assertNotIn(SECRET, str(event))

    def test_supervisor_preserves_startup_exception(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'HF_TOKEN': SECRET}):
            with self.assertRaises(RuntimeError) as caught:
                V2Supervisor(failed_startup, {}, directory)
        self.assertEqual(caught.exception.diagnostic['stage'], 'worker_startup')
        self.assertEqual(caught.exception.diagnostic['exception_type'], 'ValueError')
        self.assertNotIn(SECRET, str(caught.exception.diagnostic))

    def test_supervisor_preserves_serialization_exception(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'HF_TOKEN': SECRET}):
            worker = V2Supervisor(unpicklable_worker, {}, directory)
            event = worker.execute(0, timeout=5)
            worker.close()
        self.assertEqual(event['failure_details']['stage'], 'worker_serialization')
        self.assertNotIn(SECRET, str(event))

    def test_redaction_removes_unconfigured_tokens_bearer_and_paths(self):
        record = failure_details(ValueError('hf_FAKE1234 bearer arbitrary-credential /home/user/model C:\\secrets\\key'), stage='model_generate')
        for private in ('hf_FAKE1234', 'arbitrary-credential', '/home/user', 'C:\\secrets'):
            self.assertNotIn(private, str(record))

    def test_redaction_removes_multiple_secret_environment_values(self):
        with patch.dict(os.environ, {'HF_TOKEN': SECRET, 'GROQ_API_KEY': 'groq-private-fake'}):
            record = failure_details(ValueError(SECRET + ' groq-private-fake'), stage='model_generate')
        self.assertNotIn(SECRET, str(record)); self.assertNotIn('groq-private-fake', str(record))

    def test_environment_dump_not_retained(self):
        record = failure_details(ValueError('os.environ: environment values must not appear'), stage='model_generate')
        self.assertEqual(record['exception_message'], '[redacted environment diagnostic]')

    def test_message_bounded_after_redaction(self):
        with patch.dict(os.environ, {'HF_TOKEN': SECRET}):
            record = failure_details(ValueError('x' * 500 + SECRET + 'y' * 1000), stage='model_generate')
        self.assertLessEqual(len(record['exception_message']), 512)
        self.assertNotIn(SECRET, str(record))

    def test_redaction_preserves_relevant_transformers_argument_names(self):
        message = 'Unused model_kwargs: token_type_ids; max_new_tokens=512; tokenizer mismatch'
        self.assertEqual(failure_details(ValueError(message), stage='model_generate')['exception_message'], message)

    def test_v2_runner_marks_ten_model_failures_invalid_and_keeps_raw(self):
        class Supervisor:
            def __init__(self, target, settings, directory):
                self.adapter, _, self.model = prepared(['warm one', 'warm two'])
                self.adapter.warm_up()
                self.model.error = TypeError('Generation API rejected argument HF_TOKEN=' + SECRET)
                self.settings = settings
            def execute(self, index, timeout):
                result = run_transformers_episode(task(f'T{index+1}'), self.adapter,
                    run_id=self.settings['run_id'])
                return {'supervision': 'completed', 'result': result}
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {'torch': torch_stub()}), patch.dict(os.environ, {'HF_TOKEN': SECRET}):
            paths = run_phase2(MODEL, directory, preflight_fn=lambda *a, **k: {
                'status': 'PASS', 'checks': [], 'model_revision': 'synthetic-revision'}, supervisor_factory=Supervisor)
            raw = json.loads(paths['raw'].read_text()); manifest = json.loads(paths['manifest'].read_text())
            evaluation = json.loads(paths['evaluation'].read_text())
            for key in ('raw', 'manifest', 'evaluation'):
                self.assertNotIn(SECRET, paths[key].read_text())
        self.assertEqual(raw['status'], 'complete')
        self.assertFalse(raw['experimental_valid'])
        self.assertFalse(manifest['experimental_valid'])
        self.assertFalse(evaluation['publication_ready'])
        self.assertEqual(manifest['runtime']['successful_turns'], 0)
        self.assertEqual(manifest['runtime']['failed_turns'], 10)
        self.assertEqual(len(raw['tasks']), 10)
        self.assertEqual(raw['tasks'][9]['episode']['failure_details']['task_id'], 'T10')
        self.assertEqual(raw['tasks'][0]['episode']['failure_details']['exception_type'], 'TypeError')

    def test_v2_incomplete_run_never_publishable(self):
        from test_phase2_operations import FakeSupervisor
        class Timeout(FakeSupervisor):
            mode = 'timeout'
        with tempfile.TemporaryDirectory() as directory:
            paths = run_phase2(MODEL, directory, preflight_fn=lambda *a, **k: {
                'status': 'PASS', 'checks': [], 'model_revision': 'synthetic-revision'}, supervisor_factory=Timeout)
            evaluation = json.loads(paths['evaluation'].read_text())
        self.assertFalse(evaluation['publication_ready'])
        self.assertFalse(evaluation['experimental_valid'])

    def test_protected_benchmark_phase1_and_catalog_bytes_unchanged(self):
        names = ['prompts/benchmark_v1.json', 'prompts/benchmark_v1.md',
                 'benchmark/providers/transformers_provider.py', 'benchmark/results.py',
                 'run_phase1_colab.py', 'benchmark/tools/catalog_v2.py']
        for prefix in ('results/raw', 'results/processed/phase1', 'reports/phase1'):
            names += subprocess.check_output(['git', 'ls-files', prefix], cwd=ROOT, text=True).splitlines()
        for name in names:
            self.assertEqual((ROOT / name).read_bytes(), subprocess.check_output(
                ['git', 'show', '282a8b2:' + name], cwd=ROOT), name)
        self.assertEqual(public_catalog_sha256(), 'f802970524fa3daadb8e30801aa40ed81995b1b8d0f1fd64053e34b178ddc112')


if __name__ == '__main__':
    unittest.main()
