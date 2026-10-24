"""Run/model aggregation tests with synthetic evidence, never official outputs."""
from copy import deepcopy
import unittest

from benchmark.evaluation.phase2 import evaluate_episode
from benchmark.evaluation.phase2_aggregation import aggregate_run, aggregate_runs, compare_phase1
from test_phase2_evaluation import call, domain_task, make_episode, simple_task


def sample_rows():
    task = simple_task('T3')
    first = evaluate_episode(task, make_episode(task))
    task = simple_task('T4')
    second = evaluate_episode(task, make_episode(task, [call('unknown', {})], answer=[]))
    return [first, second]


class AggregationTests(unittest.TestCase):
    def test_main_counts_rates(self):
        summary = aggregate_run(sample_rows(), model_id='fake-model', run_id='test-run')
        self.assertEqual(summary['total_tasks'], 2)
        self.assertEqual(summary['task_correct_count'], 1)
        self.assertEqual(summary['task_correct_rate'], .5)
        self.assertEqual(summary['technical_success_rate'], 1)
        self.assertEqual(summary['protocol_correct_rate'], 1)
        self.assertEqual(summary['direct_answer_rate'], .5)

    def test_null_protocol_not_success_denominator_all_tasks(self):
        rows = sample_rows()
        rows[1]['protocol_correct'] = None
        summary = aggregate_run(rows, model_id='fake-model', run_id='test-run')
        self.assertEqual(summary['protocol_correct_rate'], .5)
        self.assertEqual(summary['protocol_correct_null_count'], 1)

    def test_tool_totals_mean_and_classes(self):
        summary = aggregate_run(sample_rows(), model_id='fake-model', run_id='test-run')
        self.assertEqual(summary['tool_calls_total'], 1)
        self.assertEqual(summary['tool_calls_mean'], .5)
        self.assertEqual(summary['tool_calls_failed'], 1)
        self.assertEqual(summary['invalid'], 1)
        self.assertEqual(summary['selection_class_counts']['invalid'], 1)
        self.assertEqual(summary['selection_class_counts']['unclassified'], 1)
        self.assertEqual(summary['selection_class_rates']['invalid'], .5)
        self.assertEqual(sum(summary['selection_class_rates'].values()), 1)

    def test_dataset_denominator_only_applicable_tasks(self):
        rows = sample_rows()
        for task_id in ('T5', 'T6'):
            task = domain_task(task_id)
            rows.append(evaluate_episode(task, make_episode(task, task['expected_tool_sequence'])))
        summary = aggregate_run(rows, model_id='fake-model', run_id='test-run')
        self.assertEqual(summary['dataset_tools_contract_correct_denominator'], 2)
        self.assertEqual(summary['dataset_tools_contract_correct_rate'], 1)
        self.assertEqual(summary['categories']['multi-tool/domain']['total_tasks'], 2)
        self.assertEqual(summary['categories']['data']['total_tasks'], 2)
        self.assertEqual(len(summary['categories']), 5)

    def test_null_minimum_total_not_invented(self):
        task = simple_task()
        rows = [evaluate_episode(task, make_episode(task))]
        summary = aggregate_run(rows, model_id='fake-model', run_id='test-run')
        self.assertIsNone(summary['calls_over_minimum_total'])
        self.assertEqual(summary['calls_over_minimum_defined_tasks'], 0)

    def test_defined_minimum_sums_only_defined(self):
        summary = aggregate_run(sample_rows(), model_id='fake-model', run_id='test-run')
        self.assertEqual(summary['calls_over_minimum_total'], 0)
        self.assertEqual(summary['calls_over_minimum_defined_tasks'], 1)

    def test_duplicate_tasks_rejected(self):
        rows = sample_rows()
        with self.assertRaises(ValueError):
            aggregate_run(rows + rows, model_id='fake-model', run_id='test-run')

    def test_mixed_model_or_run_rejected(self):
        for key in ('model_id', 'run_id'):
            rows = sample_rows()
            rows[1][key] = 'another'
            with self.subTest(key=key), self.assertRaises(ValueError):
                aggregate_run(rows, model_id='fake-model', run_id='test-run')

    def test_grouping_keeps_runs_separate_and_sorted(self):
        first = sample_rows()
        second = deepcopy(first)
        for row in second:
            row['run_id'] = 'another-run'
        third = deepcopy(first)
        for row in third:
            row['model_id'] = 'z-model'
        summaries = aggregate_runs(list(reversed(first + second + third)))
        self.assertEqual([(s['model_id'], s['run_id']) for s in summaries], [
            ('fake-model', 'another-run'), ('fake-model', 'test-run'), ('z-model', 'test-run')])
        self.assertEqual([s['total_tasks'] for s in summaries], [2, 2, 2])

    def test_empty_summary_rates_undefined(self):
        summary = aggregate_run([], model_id='fake-model')
        self.assertEqual(summary['total_tasks'], 0)
        self.assertIsNone(summary['task_correct_rate'])
        self.assertIsNone(summary['tool_calls_mean'])

    def test_raw_zero_relative_gain_undefined(self):
        rows = sample_rows()
        raw = [{'task_id': row['task_id'], 'strict_correct': False} for row in rows]
        comparison = compare_phase1(rows, raw)
        self.assertIsNone(comparison['strict_gain_relative_percent'])
        self.assertEqual(comparison['strict_gain_absolute_pp'], 50)
        self.assertEqual(comparison['paired_tasks'], 2)

    def test_nonzero_raw_relative_and_absolute(self):
        rows = sample_rows()
        rows[1]['strict_correct'] = True
        raw = [{'task_id': row['task_id'], 'strict_correct': i == 0} for i, row in enumerate(rows)]
        comparison = compare_phase1(rows, raw)
        self.assertEqual(comparison['strict_gain_absolute_pp'], 50)
        self.assertEqual(comparison['strict_gain_relative_percent'], 100)
        self.assertNotIn('causal', comparison)

    def test_domain_comparison_marks_contract_difference(self):
        task = domain_task()
        row = evaluate_episode(task, make_episode(task, task['expected_tool_sequence']))
        raw = [{'task_id': 'T5', 'strict_correct': True}]
        before = deepcopy(raw)
        comparison = compare_phase1([row], raw)
        self.assertTrue(comparison['tasks'][0]['contract_changed'])
        self.assertTrue(comparison['tasks'][0]['phase1_strict_correct'])
        self.assertEqual(raw, before)

    def test_missing_pairs_or_duplicate_raw_rejected(self):
        rows = sample_rows()
        for raw in ([], [{'task_id': 'T4', 'strict_correct': True}] * 2):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                compare_phase1(rows, raw)

    def test_mixed_model_comparison_rejected(self):
        rows = sample_rows()
        raw = [{'task_id': row['task_id'], 'strict_correct': True, 'model_id': 'wrong'} for row in rows]
        with self.assertRaises(ValueError):
            compare_phase1(rows, raw)

    def test_aggregation_deterministic_and_does_not_mutate(self):
        rows = sample_rows()
        before = deepcopy(rows)
        one = aggregate_run(rows, model_id='fake-model', run_id='test-run')
        two = aggregate_run(list(reversed(rows)), model_id='fake-model', run_id='test-run')
        self.assertEqual(one, two)
        self.assertEqual(rows, before)


if __name__ == '__main__':
    unittest.main()
