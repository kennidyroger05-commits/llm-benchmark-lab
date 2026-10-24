"""Byte-stable, allowlisted Phase 2 report projections in temporary directories."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.evaluation.phase2 import evaluate_episode
from benchmark.evaluation.phase2_export import build_report, export_report, report_bytes
from test_phase2_evaluation import HASH, domain_task, make_episode, simple_task
from test_phase2_aggregation import sample_rows


class ExportTests(unittest.TestCase):
    def report(self, rows=None, **kwargs):
        return build_report(sample_rows() if rows is None else rows, model_id='fake-model',
                            run_metadata={'run_id': 'test-run'}, **kwargs)

    def test_valid_versioned_json(self):
        report = self.report()
        loaded = json.loads(report_bytes(report))
        self.assertEqual(loaded['phase'], 'phase2')
        self.assertEqual(loaded['condition'], 'tools')
        self.assertTrue(loaded['evaluation_version'])
        self.assertTrue(loaded['efficiency_policy_version'])
        self.assertEqual(len(loaded['task_evaluations']), 2)

    def test_same_synthetic_episode_twice_byte_identical(self):
        task = domain_task('T6')
        reports = [self.report([evaluate_episode(task, make_episode(task, task['expected_tool_sequence']))])
                   for _ in range(2)]
        self.assertEqual(report_bytes(reports[0]), report_bytes(reports[1]))
        self.assertEqual(reports[0]['structural_hash'], reports[1]['structural_hash'])

    def test_input_order_irrelevant(self):
        self.assertEqual(report_bytes(self.report()), report_bytes(self.report(list(reversed(sample_rows())))))

    def test_hashes_present_and_verifiable(self):
        report = self.report()
        self.assertEqual(report['catalog_hash'], HASH)
        expected = hashlib.sha256(Path('prompts/benchmark_v1.json').read_bytes()).hexdigest()
        self.assertEqual(report['benchmark_hash'], expected)
        self.assertEqual(len(report['structural_hash']), 64)
        for name, digest in report['evaluator_hashes'].items():
            self.assertEqual(digest, hashlib.sha256(Path(name).read_bytes()).hexdigest())
        self.assertIn('benchmark/evaluation/phase2_efficiency.py', report['evaluator_hashes'])
        self.assertIn('benchmark/evaluation/code_evaluation.py', report['evaluator_hashes'])

    def test_no_task_secrets_or_context_projected(self):
        task = domain_task()
        evaluation = evaluate_episode(task, make_episode(task, task['expected_tool_sequence']))
        evaluation.update(tool_fixtures=task['tool_fixtures'], expected_answer='PRIVATE_ANSWER',
                          raw_prompt='PRIVATE_RAW', expected_tool_sequence=task['expected_tool_sequence'])
        report = self.report([evaluation])
        data = report_bytes(report).decode()
        for text in ('PRIVATE_', 'tool_fixtures', 'expected_tool_sequence', 'final_expected_answer',
                     'raw_prompt', 'full_history', 'test_cases', 'toy-5'):
            self.assertNotIn(text, data)
        self.assertNotIn('matrix', report)

    def test_metadata_allowlist_rejects_private_fields(self):
        for name in ('tool_fixtures', 'expected_answer', 'timestamp'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_report(sample_rows(), model_id='fake-model', run_metadata={name: 'secret'})
        with self.assertRaises(ValueError):
            self.report(hardware_metadata={'fixture': {}})

    def test_nested_private_metadata_rejected(self):
        with self.assertRaises(ValueError):
            self.report(hardware_metadata={'gpu_name': {'tool_fixtures': 'private'}})
        with self.assertRaises(ValueError):
            build_report(sample_rows(), model_id='fake-model',
                         run_metadata={'model_revision': {'expected_answer': 'private'}})

    def test_run_id_can_be_derived_from_evaluations(self):
        report = build_report(sample_rows(), model_id='fake-model')
        self.assertEqual(report['run_metadata']['run_id'], 'test-run')
        self.assertEqual(report['summary']['run_id'], 'test-run')

    def test_hardware_optional_and_preserved_when_supplied(self):
        self.assertEqual(self.report()['hardware_metadata'], {})
        hardware = {'device': 'cuda:0', 'gpu_name': 'synthetic', 'vram_gb': 12, 'dtype': 'float16',
                    'device_map': {'': 'cuda:0'}, 'offload': False}
        self.assertEqual(self.report(hardware_metadata=hardware)['hardware_metadata'], hardware)

    def test_source_path_hash_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.json'
            path.write_bytes(b'private raw source, never copied')
            report = self.report(source_raw_path=path)
            self.assertEqual(report['source_raw']['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(report['source_raw']['path'], str(path))
            self.assertNotIn('private raw source', report_bytes(report).decode())

    def test_source_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.json'
            path.write_bytes(b'raw')
            with self.assertRaises(ValueError):
                self.report(source_raw_path=path, source_raw_hash='0' * 64)

    def test_hash_only_provenance(self):
        report = self.report(source_raw_hash='a' * 64)
        self.assertEqual(report['source_raw'], {'path': None, 'sha256': 'a' * 64})
        with self.assertRaises(ValueError):
            self.report(source_raw_hash='not a hash')

    def test_creates_destination_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reports/phase2/fake-model/run.json'
            report = self.report()
            self.assertEqual(export_report(report, path), path)
            self.assertEqual(path.read_bytes(), report_bytes(report))

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            export_report(self.report(), path)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                export_report(self.report(), path)
            self.assertEqual(path.read_bytes(), before)

    def test_protected_project_destinations_refused(self):
        for path in ('reports/phase1/DO_NOT_CREATE.json', 'results/raw/DO_NOT_CREATE.json',
                     'prompts/benchmark_v1.json', 'results/processed/phase1/DO_NOT_CREATE.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                export_report(self.report(), path)

    def test_modified_report_detected(self):
        report = self.report()
        report['summary']['total_tasks'] = 999
        with self.assertRaises(ValueError):
            report_bytes(report)

    def test_metadata_mutation_does_not_change_report(self):
        hardware = {'device': 'cpu'}
        report = self.report(hardware_metadata=hardware)
        before = report_bytes(report)
        hardware['device'] = 'changed'
        self.assertEqual(report_bytes(report), before)

    def test_reports_include_summary_tool_metrics_categories(self):
        report = self.report()
        self.assertEqual(report['tool_metrics']['tool_calls_failed'], 1)
        self.assertEqual(report['summary']['categories']['data']['total_tasks'], 2)

    def test_phase1_artifacts_unchanged_by_evaluation_and_export(self):
        paths = [Path('prompts/benchmark_v1.json'), Path('reports/phase1/phase1_summary.md')]
        paths += sorted(Path('results/processed/phase1').glob('*'))
        paths += sorted(Path('results/raw').rglob('*.json'))
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}
        with tempfile.TemporaryDirectory() as directory:
            export_report(self.report(), Path(directory) / 'phase2.json')
        after = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in before}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
