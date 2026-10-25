"""Non-official operational smoke. All artifacts live in temporary directories."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from benchmark.orchestration.phase2_preflight import run_preflight, ROOT, frozen_integrity
from benchmark.orchestration.phase2_runner import run_phase2
from benchmark.orchestration.phase2_supervisor import ProcessSupervisor, ScopedDockerBackend
from benchmark.orchestration.phase2_runtime import TransformersEpisodeResult
from benchmark.orchestration.episode import run_episode
from benchmark.providers.transformers_phase2 import OFFICIAL_MODELS
from benchmark.tools.domain_tools import DomainSession
from benchmark.tools.sandbox import DockerSandboxBackend
from test_phase2_evaluation import domain_task

MODEL = OFFICIAL_MODELS[0]


def probes():
    names = ('official_model', 'python', 'torch', 'transformers', 'accelerate', 'cuda_gpu',
             'float16', 'hf_auth', 'dependency_versions', 'catalog_hash', 'benchmark_hash', 'frozen_files', 'registry',
             'safe_sandbox', 'no_unsafe_fallback', 'output_writable', 'no_conflicting_run')
    result = {name: lambda: True for name in names}
    result['model_checks'] = lambda: {name: True for name in (
        'model_access', 'tokenizer', 'model_config', 'chat_template', 'context_capacity')}
    return result


def fast_worker(settings, owner):
    return lambda command: command


def slow_worker(settings, owner):
    def handle(command):
        time.sleep(30)
    return handle


def docker_worker(settings, owner):
    def handle(command):
        ScopedDockerBackend(owner)._launch('while True: pass', timeout=30)
    return handle


class FakeSupervisor:
    """Runs real episode/dispatch/evaluation pipeline, only the model is scripted."""
    mode = 'complete'
    def __init__(self, target, settings, directory):
        self.settings = settings
        self.indices = []
        self.sessions = []
        self.warmups = ['discard one', 'discard two']
        self.closed = False
        self.tasks = json.loads((ROOT / 'prompts/benchmark_v1.json').read_text())
    def execute(self, index, timeout):
        self.indices.append(index)
        if self.mode != 'complete' and index == 1:
            return {'supervision': self.mode, 'technical_success': False}
        task = self.tasks[index]
        session = None
        if task['id'] in {'T5', 'T6'}:
            task = domain_task(task['id'])
            session = DomainSession(tool_catalog=task['tool_catalog'], tool_fixtures=task['tool_fixtures'])
            self.sessions.append(session)
            calls = task['expected_tool_sequence']
        else:
            calls = [{'tool': 'calculator_basic', 'arguments': {'expression': '1+2'}}] if index == 1 else []
        outputs = [json.dumps({'type': 'tool_call', **call}) for call in calls]
        outputs += ['invalid JSON'] if index == 3 else [json.dumps({'type': 'final_answer', 'answer': 'synthetic'})]
        script = iter(outputs)
        episode = run_episode(task, lambda context: next(script), domain_session=session,
                              metadata={'model_id': MODEL, 'run_id': self.settings['run_id']})
        return {'supervision': 'completed', 'result': TransformersEpisodeResult(episode, [], [], {})}
    def close(self):
        self.closed = True


class PreflightTests(unittest.TestCase):
    def test_pass_test_seams(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_preflight(MODEL, directory, inspect_model=True, probes=probes())
        self.assertEqual(result['status'], 'PASS')
    def test_each_blocking_requirement(self):
        for name in ('benchmark_hash', 'registry', 'cuda_gpu', 'hf_auth', 'output_writable',
                     'safe_sandbox', 'catalog_hash', 'frozen_files', 'no_conflicting_run'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                checks = probes()
                checks[name] = lambda: False
                self.assertEqual(run_preflight(MODEL, directory, inspect_model=True, probes=checks)['status'], 'FAIL')
    def test_small_context(self):
        checks = probes()
        checks['model_checks'] = lambda: {'context_capacity': False}
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run_preflight(MODEL, directory, inspect_model=True, probes=checks)['status'], 'FAIL')
    def test_deferred_not_real_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_preflight(MODEL, directory, probes=probes())
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(any(c['status'] == 'not_executed_until_colab' for c in result['checks']))
    def test_real_frozen_integrity(self):
        import subprocess, hashlib
        frozen = json.loads((ROOT / 'benchmark/phase2_frozen.json').read_text())
        for name, digest in frozen.items():
            old = subprocess.run(['git', 'show', '2da126c:' + name], check=True, capture_output=True).stdout
            self.assertEqual(hashlib.sha256(old).hexdigest(), digest, name)
    def test_sandbox_failure_blocks_worker(self):
        result = {'status': 'FAIL', 'checks': [{'status': 'FAIL', 'code': 'safe_sandbox_unavailable'}]}
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'safe_sandbox_unavailable'):
                run_phase2(MODEL, directory, preflight_fn=lambda *a, **k: result,
                           supervisor_factory=lambda *a: self.fail('must not load'))


class SupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DockerSandboxBackend().is_available():
            raise unittest.SkipTest('Historical V1 Docker supervisor unavailable; V2 has no Docker.')
    def test_fast_child(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = ProcessSupervisor(fast_worker, {}, directory)
            self.assertEqual(worker.execute(7, timeout=2)['result'], 7)
            worker.close()
            self.assertFalse(worker.process.is_alive())
    def test_timeout_child_removed_no_next_task(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = ProcessSupervisor(slow_worker, {}, directory)
            self.assertEqual(worker.execute(0, timeout=.1)['supervision'], 'timeout')
            self.assertFalse(worker.process.is_alive())
            with self.assertRaises(RuntimeError):
                worker.execute(1, timeout=1)
    def test_real_docker_child_cleanup(self):
        self.assertTrue(DockerSandboxBackend().is_available())
        with tempfile.TemporaryDirectory() as directory:
            worker = ProcessSupervisor(docker_worker, {}, directory)
            event = worker.execute(0, timeout=.8)
            self.assertEqual(event['supervision'], 'timeout')
            self.assertFalse(worker.process.is_alive())
            result = DockerSandboxBackend()._docker(['ps', '-aq', '--filter', 'label=phase2.owner=' + worker.owner])
            self.assertFalse(result.stdout.strip())
            time.sleep(1)
            late = DockerSandboxBackend()._docker(['ps', '-aq', '--filter', 'label=phase2.owner=' + worker.owner])
            self.assertFalse(late.stdout.strip(), 'late Docker create must not survive cleanup')


class SmokeTests(unittest.TestCase):
    def run_fake(self, directory, mode='complete'):
        captured = []
        class Factory(FakeSupervisor):
            def __init__(self, *args):
                super().__init__(*args)
                self.mode = mode
                captured.append(self)
        preflight = lambda *a, **k: {'status': 'PASS', 'checks': [], 'model_revision': 'fake-revision'}
        result = run_phase2(MODEL, directory, preflight_fn=preflight, supervisor_factory=Factory)
        return result, captured[0]
    def test_end_to_end_direct_tool_domain_invalid_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            result, worker = self.run_fake(directory)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(worker.indices, list(range(10)))
            self.assertEqual(len(worker.warmups), 2)
            self.assertEqual(len(worker.sessions), 2)
            self.assertIsNot(worker.sessions[0], worker.sessions[1])
            self.assertTrue(worker.closed)
            raw = json.loads(result['raw'].read_text())
            self.assertEqual(len(raw['tasks']), 10)
            self.assertEqual(raw['tasks'][0]['episode']['termination_reason'], 'final_answer')
            self.assertEqual(raw['tasks'][1]['episode']['tool_calls_dispatched'], 1)
            self.assertEqual(raw['tasks'][3]['episode']['termination_reason'], 'invalid_protocol')
            self.assertNotIn('discard one', result['raw'].read_text())
            for record in raw['tasks']:
                self.assertEqual(json.loads(record['episode']['model_contexts'][0])['history'], [])
            self.assertNotIn('tool_fixtures', result['raw'].read_text())
            self.assertTrue(result['manifest'].exists())
            self.assertTrue(result['evaluation'].exists())
    def test_timeout_preserves_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            result, worker = self.run_fake(directory, 'timeout')
            raw = json.loads(result['raw'].read_text())
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(raw['tasks'][1]['episode']['termination_reason'], 'timeout')
            self.assertEqual(raw['tasks'][2]['episode']['termination_reason'], 'not_executed')
            self.assertEqual(worker.indices, [0, 1])
            self.assertEqual(json.loads(result['evaluation'].read_text())['summary']['total_tasks'], 10)
    def test_infrastructure_failure_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            result, worker = self.run_fake(directory, 'infrastructure_failure')
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(json.loads(result['raw'].read_text())['tasks'][1]['episode']['termination_reason'], 'executor_error')
            self.assertEqual(json.loads(result['evaluation'].read_text())['collection_status'], 'incomplete')
            self.assertTrue(worker.closed)
    def test_manifest_hashes_and_no_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.run_fake(directory)
            manifest = json.loads(result['manifest'].read_text())
            from benchmark.orchestration.phase2_preflight import sha
            self.assertEqual(manifest['raw_sha256'], sha(result['raw']))
            for path in result.values():
                if isinstance(path, Path):
                    self.assertNotIn('HF_TOKEN', path.read_text())
