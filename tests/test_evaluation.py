"""Evaluation-only regression tests; exclude this file from fine-tuning data."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from benchmark.evaluation.aggregation import MODELS, aggregate, discover_runs
from benchmark.evaluation.code_evaluation import evaluate_code
from benchmark.evaluation.parsing import parse_json
from benchmark.evaluation.task_evaluators import evaluate_task
from evaluate_phase1 import main

ROOT = Path(__file__).resolve().parents[1]
TASKS = json.loads((ROOT / 'prompts/benchmark_v1.json').read_text())


class EvaluationTests(unittest.TestCase):
    def evaluate(self, index, response, **kwargs):
        return evaluate_task(TASKS[index-1], {'response': response, 'status': 'success', **kwargs})

    def answer(self, index):
        return json.dumps(TASKS[index-1]['expected_answer'])

    def test_correct_json_all_contracts(self):
        for index in [1, 2, 3, 4, 5, 6, 9]:
            with self.subTest(index=index):
                self.assertTrue(self.evaluate(index, self.answer(index))['strict_correct'])

    def test_json_fence_is_only_content_credit(self):
        for tag in ['json', '']:
            verdict = self.evaluate(1, f'```{tag}\n{self.answer(1)}\n```')
            self.assertFalse(verdict['strict_correct'])
            self.assertFalse(verdict['format_correct'])
            self.assertTrue(verdict['content_correct'])
            self.assertIn('markdown_not_allowed', verdict['failure_reasons'])

    def test_no_extraction_or_repair(self):
        for response in ['prefix ' + self.answer(1), self.answer(1) + ' explanation',
                         'intro\n```json\n' + self.answer(1) + '\n```', self.answer(1)[:-1]]:
            self.assertFalse(self.evaluate(1, response)['content_correct'])

    def test_duplicate_keys_and_nonfinite_rejected(self):
        for text in ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}']:
            with self.assertRaises(ValueError):
                parse_json(text)

    def test_extra_fields(self):
        for index in [1, 2, 9]:
            value = copy.deepcopy(TASKS[index-1]['expected_answer'])
            value['extra'] = 0
            v = self.evaluate(index, json.dumps(value))
            self.assertFalse(v['format_correct'])
            self.assertIn('extra_fields', v['failure_reasons'])

    def test_nested_extra_fields(self):
        value = copy.deepcopy(TASKS[8]['expected_answer'])
        value['actions'][0]['extra'] = 0
        self.assertFalse(self.evaluate(9, json.dumps(value))['format_correct'])

    def test_missing_field(self):
        value = copy.deepcopy(TASKS[8]['expected_answer'])
        del value['actions'][0]['owner']
        self.assertIn('missing_field', self.evaluate(9, json.dumps(value))['failure_reasons'])

    def test_wrong_types_including_bool_and_numeric_strings(self):
        for index, key, value in [(1, 'saldo_final', '222.51'), (2, 'pacotes_c', False),
                                   (2, 'pacotes_a', 5.0)]:
            data = copy.deepcopy(TASKS[index-1]['expected_answer'])
            data[key] = value
            v = self.evaluate(index, json.dumps(data))
            self.assertIn('wrong_type', v['failure_reasons'])
            self.assertFalse(v['strict_correct'])

    def test_decimal_zero_tolerance(self):
        self.assertTrue(self.evaluate(1, '{"saldo_final":222.5100,"juros_totais":622.51}')['strict_correct'])
        self.assertFalse(self.evaluate(1, '{"saldo_final":222.510000000000001,"juros_totais":622.51}')['content_correct'])

    def test_wrong_order(self):
        value = list(reversed(TASKS[2]['expected_answer']))
        v = self.evaluate(3, json.dumps(value))
        self.assertFalse(v['strict_correct'])
        self.assertTrue(v['format_correct'])
        self.assertIn('wrong_order', v['failure_reasons'])

    def test_token_limit_is_evidence_not_semantic_verdict(self):
        v = self.evaluate(1, self.answer(1), output_tokens=512, max_new_tokens=512)
        self.assertTrue(v['hit_max_new_tokens'])
        self.assertIsNone(v['truncated'])
        self.assertTrue(v['strict_correct'])
        bad = self.evaluate(1, '{', output_tokens=512, max_new_tokens=512)
        self.assertFalse(bad['strict_correct'])
        self.assertIn('invalid_json', bad['failure_reasons'])

    def test_technical_failure_does_not_erase_content_diagnosis(self):
        v = evaluate_task(TASKS[0], {'response': self.answer(1), 'status': 'error'})
        self.assertFalse(v['technical_success'])
        self.assertFalse(v['strict_correct'])
        self.assertTrue(v['content_correct'])

    def test_tool_extra_calls_and_extra_call_fields(self):
        for index in [5, 6]:
            data = copy.deepcopy(TASKS[index-1]['expected_answer'])
            data.append(data[-1])
            v = self.evaluate(index, json.dumps(data))
            self.assertIn('extra_tool_call', v['failure_reasons'])
            self.assertFalse(v['strict_correct'])
            data = copy.deepcopy(TASKS[index-1]['expected_answer'])
            data[0]['result'] = {}
            self.assertFalse(self.evaluate(index, json.dumps(data))['format_correct'])

    def test_tool_missing_call(self):
        self.assertIn('missing_tool_call', self.evaluate(5, json.dumps(TASKS[4]['expected_answer'][:-1]))['failure_reasons'])

    def test_tool_wrong_arguments(self):
        for index in [5, 6]:
            data = copy.deepcopy(TASKS[index-1]['expected_answer'])
            first_key = next(iter(data[0]['arguments']))
            data[0]['arguments'][first_key] = 'wrong'
            v = self.evaluate(index, json.dumps(data))
            self.assertIn('wrong_tool_arguments', v['failure_reasons'])
            self.assertFalse(v['content_correct'])

    def test_tool_order(self):
        data = list(reversed(TASKS[5]['expected_answer']))
        v = self.evaluate(6, json.dumps(data))
        self.assertIn('wrong_order', v['failure_reasons'])
        self.assertTrue(v['format_correct'])

    def test_correct_reference_code_both_tasks(self):
        for index in [7, 8]:
            v = self.evaluate(index, TASKS[index-1]['expected_answer']['reference_code'])
            self.assertTrue(v['strict_correct'], v)

    def test_fenced_reference_code_both_tasks(self):
        for index in [7, 8]:
            code = TASKS[index-1]['expected_answer']['reference_code'].rstrip('\n')
            v = self.evaluate(index, '```python\n' + code + '\n```')
            self.assertTrue(v['content_correct'], v)
            self.assertFalse(v['format_correct'])
            self.assertFalse(v['strict_correct'])

    def test_invalid_python(self):
        self.assertIn('invalid_python', self.evaluate(7, 'def reconcile(:')['failure_reasons'])

    def test_wrong_function_and_signature(self):
        for code in ['def other(events): return {}', 'def reconcile(): return {}']:
            self.assertIn('wrong_function', self.evaluate(7, code)['failure_reasons'])

    def test_wrong_code_behavior(self):
        self.assertIn('wrong_code_behavior', self.evaluate(8, 'def expand_ranges(text): return []')['failure_reasons'])

    def test_input_mutation_rejected(self):
        code = TASKS[6]['expected_answer']['reference_code'].replace('    latest = {}', '    events.sort(key=lambda e: e["id"])\n    latest = {}')
        self.assertFalse(self.evaluate(7, code)['content_correct'])

    def test_code_timeout(self):
        _, passed, reasons = evaluate_code('def reconcile(events):\n    while True: pass', TASKS[6], timeout=0.15)
        self.assertFalse(passed)
        self.assertEqual(reasons, ['code_timeout'])

    def test_no_network_files_process_or_dynamic_execution(self):
        for code in ['import socket\ndef reconcile(events): return socket.socket()',
                     'def reconcile(events): return open("/etc/passwd").read()',
                     'import os\ndef reconcile(events): return os.system("true")',
                     'def reconcile(events): return eval("1")']:
            self.assertFalse(self.evaluate(7, code)['content_correct'])

    def test_sandbox_blocks_side_effect_even_when_remaining_code_is_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'must-not-exist'
            reference = TASKS[6]['expected_answer']['reference_code']
            code = reference.replace('    latest = {}',
                                     f'    open({str(marker)!r}, "w").write("bad")\n    latest = {{}}')
            self.assertFalse(self.evaluate(7, code)['content_correct'])
            self.assertFalse(marker.exists())

    def test_t10_exact_eight_lines(self):
        text = TASKS[9]['expected_answer']
        for response in [text, text + '\n', text.replace('\n', '\r\n')]:
            self.assertTrue(self.evaluate(10, response)['strict_correct'])

    def test_t10_no_prefixes(self):
        text = '\n'.join(line.split('=')[1] for line in TASKS[9]['expected_answer'].splitlines())
        self.assertFalse(self.evaluate(10, text)['format_correct'])

    def test_t10_whitespace_wrong_values_and_fence(self):
        text = TASKS[9]['expected_answer']
        for response in [text + '\n\n', ' ' + text, text.replace('ACTIVE=2', 'ACTIVE=3')]:
            self.assertFalse(self.evaluate(10, response)['strict_correct'])
        v = self.evaluate(10, '```text\n' + text + '\n```')
        self.assertTrue(v['content_correct'])
        self.assertFalse(v['format_correct'])


class DiscoveryAggregationTests(unittest.TestCase):
    def make_runs(self, root):
        for alias, model in MODELS.items():
            rows = []
            for task in TASKS:
                answer = task['expected_answer']
                response = (answer['reference_code'] if task['evaluation_type'] == 'code' else
                            answer if isinstance(answer, str) else json.dumps(answer))
                rows.append({'task_id': task['id'], 'model_id': model, 'phase': 'phase1',
                             'condition': 'raw', 'prompt': task.get('raw_prompt', task['prompt']),
                             'response': response, 'status': 'success', 'input_tokens': 10,
                             'output_tokens': 20, 'total_tokens': 30,
                             'inference_latency_seconds': 2, 'model_load_time_seconds': 5,
                             'device': 'cuda:0', 'gpu_name': 'NVIDIA L4' if alias == '8b' else 'Tesla T4',
                             'vram_gb': 22 if alias == '8b' else 14.56, 'dtype': 'float16'})
            (root / (alias + '.json')).write_text(json.dumps(rows))

    def test_discovery_identifies_by_model_id_not_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_runs(root)
            (root / '1b.json').rename(root / 'unrelated-name.json')
            self.assertEqual(len(discover_runs(root, TASKS)), 3)

    def test_discovery_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'encontrados 0'):
                discover_runs(directory, TASKS)

    def test_discovery_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_runs(root)
            (root / 'duplicate.json').write_bytes((root / '1b.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'encontrados 2'):
                discover_runs(root, TASKS)

    def test_discovery_rejects_changed_prompt_or_duplicate_task(self):
        for key, value in [('prompt', 'different'), ('task_id', 'T2'), ('condition', 'tools')]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_runs(root)
                rows = json.loads((root / '1b.json').read_text())
                rows[0][key] = value
                (root / '1b.json').write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    discover_runs(root, TASKS)

    def test_aggregate_metrics_and_single_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_runs(root)
            rows = discover_runs(root, TASKS)['1b'][1]
            verdicts = [evaluate_task(t, r) for t, r in zip(TASKS, rows)]
            summary = aggregate(rows, verdicts)
            self.assertEqual(summary['model_load_time_seconds'], 5)
            self.assertEqual(summary['strict_correct_count'], 10)
            self.assertEqual(summary['total_tokens'], 300)
            self.assertEqual(summary['output_tokens_total'], 200)
            self.assertEqual(summary['inference_latency_total_seconds'], 20)
            self.assertEqual(summary['inference_latency_mean_seconds'], 2)
            self.assertEqual(summary['inference_latency_median_seconds'], 2)
            self.assertEqual(summary['observed_throughput_total'], 10)
            rows[-1]['gpu_name'] = 'other'
            with self.assertRaises(ValueError):
                aggregate(rows, verdicts)

    def test_deterministic_outputs_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'results/raw/phase1'
            raw.mkdir(parents=True)
            self.make_runs(raw)
            (root / 'prompts').mkdir()
            (root / 'prompts/benchmark_v1.json').write_text(json.dumps(TASKS))
            hardware = root / 'reports/phase1/hardware'
            hardware.mkdir(parents=True)
            (hardware / '8b_t4_fp16_attempt.md').write_text('CPU offload; KeyboardInterrupt; sem resultado de acurácia.')
            protected = [*raw.glob('*.json'), root / 'prompts/benchmark_v1.json']
            hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in protected}
            with redirect_stdout(io.StringIO()):
                main(root)
                outputs = [root / 'results/processed/phase1/phase1_evaluation.json',
                           root / 'results/processed/phase1/phase1_task_results.csv',
                           root / 'reports/phase1/phase1_summary.md']
                first = [p.read_bytes() for p in outputs]
                main(root)
            self.assertEqual(first, [p.read_bytes() for p in outputs])
            self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).digest() for p in protected})
            import csv
            with outputs[1].open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 30)


if __name__ == '__main__':
    unittest.main()
