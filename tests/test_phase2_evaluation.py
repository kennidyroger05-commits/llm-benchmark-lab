"""Private synthetic evaluator contracts, without model execution or public answers."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import unittest

from benchmark.evaluation.phase2 import evaluate_episode
from benchmark.evaluation.phase2_efficiency import typed_key
from benchmark.evaluation.parsing import parse_json
from benchmark.evaluation.task_evaluators import evaluate_task
from benchmark.orchestration.episode import EpisodeLimits, run_episode
from benchmark.tools.catalog import get_public_catalog, public_catalog_sha256
from benchmark.tools._sandbox_worker import json_bytes
from benchmark.tools.domain_tools import DomainSession
from benchmark.tools.sandbox import DockerSandboxBackend, SandboxBackend, SandboxResult, SandboxError

HASH = 'cfe666d04c4870461bd3c18bfd5b78a393d5778cd6fb2aac6399d6701dade92b'
MISSING = object()


def simple_task(task_id='T4'):
    return {'id': task_id, 'category': 'Manipulação e Filtragem de Dados',
            'prompt': 'Synthetic public task.', 'evaluation_type': 'json_schema',
            'expected_answer': [{'sku': 'toy', 'available': 3}]}


def domain_task(task_id='T5'):
    if task_id == 'T5':
        calls = [
            ('lookup_order', {'order_id': 'toy-5'}, {'status': 'delivered', 'lines': ['a', 'b']}),
            ('check_return', {'line_id': 'a', 'reason': 'damaged'}, {'eligible': True}),
            ('check_return', {'line_id': 'b', 'reason': 'changed_mind'}, {'eligible': False}),
            ('create_return', {'order_id': 'toy-5', 'line_ids': ['a']}, {'ok': True}),
        ]
        blocked = ['cancel_order', 'refund_payment']
        answer = {'order_id': 'toy-5', 'line_ids': ['a'], 'status': 'return_created'}
    else:
        calls = [
            ('get_stock', {'sku': 'toy-sku'}, {'available': 6}),
            ('reserve_stock', {'order_id': 'toy-6', 'sku': 'toy-sku', 'quantity': 2},
             {'ok': True, 'reservation_id': 'toy-reservation'}),
            ('charge', {'order_id': 'toy-6', 'amount_centavos': 70},
             {'ok': False, 'error': 'payment_declined'}),
            ('release_stock', {'reservation_id': 'toy-reservation'}, {'ok': True}),
            ('log_failure', {'order_id': 'toy-6', 'reason': 'payment_declined'}, {'ok': True}),
        ]
        blocked = ['issue_invoice', 'send_receipt']
        answer = {'order_id': 'toy-6', 'status': 'payment_failed', 'reservation_released': True,
                  'failure_logged': True}
    names = {name for name, *_ in calls} | set(blocked)
    public = [tool.to_dict() for tool in get_public_catalog() if tool.name in names]
    fixtures = {
        'cases': [{'id': f'PRIVATE_CASE_{i}', 'tool': name, 'arguments': args,
                   'requires': [f'PRIVATE_CASE_{j}' for j in range(i)], 'result': result}
                  for i, (name, args, result) in enumerate(calls)],
        'blocked_tools': blocked, 'dispatch_rules': ['PRIVATE_RULE_SENTINEL'],
        'errors': {code: {'ok': False, 'error': code} for code in (
            'unknown_tool', 'invalid_arguments', 'policy_violation', 'duplicate_call', 'precondition_failed')},
    }
    return {'id': task_id, 'category': 'Intenção e Roteamento de Múltiplas Ferramentas',
            'tools_prompt': 'Synthetic public business task.', 'evaluation_type': 'json_schema',
            'tool_catalog': public, 'tool_fixtures': fixtures,
            'expected_tool_sequence': [{'tool': name, 'arguments': args} for name, args, _ in calls],
            'final_expected_answer': answer, 'evaluation_notes_tools': 'PRIVATE_NOTES_SENTINEL'}


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
    def __call__(self, context):
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


class FakeCodeBackend(SandboxBackend):
    def __init__(self, passed=True, error=None):
        self.passed, self.error, self.requests = passed, error, []
    def is_available(self): return True
    def validate_environment(self): return {'available': True}
    def cancel(self): pass
    def cleanup(self): pass
    def run_python(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        if kwargs['mode'] == 'executor':
            return SandboxResult({'value': kwargs['data']})
        return SandboxResult({'passed': self.passed, 'cases': []})


def call(name, arguments):
    return {'tool': name, 'arguments': arguments}


def make_episode(task, calls=(), answer=MISSING, *, limits=EpisodeLimits(), outputs=None,
                 backend=None, model_id='fake-model', run_id='test-run'):
    if outputs is None:
        answer = task.get('final_expected_answer', task.get('expected_answer')) if answer is MISSING else answer
        outputs = [json.dumps({'type': 'tool_call', **item}) for item in calls]
        outputs += [json.dumps({'type': 'final_answer', 'answer': answer}, default=str)]
    session = (DomainSession(tool_catalog=task['tool_catalog'], tool_fixtures=task['tool_fixtures'])
               if task['id'] in {'T5', 'T6'} else None)
    return run_episode(task, ScriptedModel(outputs), domain_session=session, sandbox_backend=backend,
                       limits=limits, metadata={'model_id': model_id, 'run_id': run_id})


class EvaluationCoreTests(unittest.TestCase):
    def test_direct_native_answer(self):
        task = simple_task()
        result = evaluate_episode(task, make_episode(task))
        for key in ('technical_success', 'protocol_correct', 'task_correct', 'strict_correct',
                    'content_correct', 'format_correct', 'direct_answer'):
            self.assertTrue(result[key], key)
        self.assertIsNone(result['dataset_tools_contract_correct'])
        self.assertIsNone(result['tool_selection_class'])

    def test_technical_false_is_not_functional_false(self):
        task = simple_task()
        episode = make_episode(task)
        episode.technical_success = False  # Recorded recovered infrastructure event.
        result = evaluate_episode(task, episode)
        self.assertTrue(result['task_correct'])
        self.assertFalse(result['strict_correct'])
        self.assertFalse(result['technical_success'])

    def test_invalid_protocol_keeps_technical_success(self):
        task = simple_task()
        result = evaluate_episode(task, make_episode(task, outputs=['not json']))
        self.assertTrue(result['technical_success'])
        self.assertFalse(result['protocol_correct'])
        self.assertFalse(result['task_correct'])
        self.assertEqual(result['tool_calls_total'], 0)

    def test_no_output_protocol_null(self):
        task = simple_task()
        result = evaluate_episode(task, make_episode(task, outputs=[RuntimeError('private')]))
        self.assertIsNone(result['protocol_correct'])
        self.assertFalse(result['technical_success'])
        self.assertFalse(result['content_correct'])
        self.assertFalse(result['format_correct'])

    def test_limit_without_final(self):
        task = simple_task()
        episode = make_episode(task, [call('calculator_basic', {'expression': '2+3'})],
                               limits=EpisodeLimits(max_model_outputs=1))
        result = evaluate_episode(task, episode)
        self.assertTrue(result['protocol_correct'])
        self.assertFalse(result['task_correct'])
        self.assertFalse(result['strict_correct'])
        self.assertFalse(result['content_correct'])

    def test_wrong_final_content_right_format(self):
        task = simple_task()
        result = evaluate_episode(task, make_episode(task, answer=[{'sku': 'wrong', 'available': 3}]))
        self.assertFalse(result['content_correct'])
        self.assertTrue(result['format_correct'])
        self.assertFalse(result['task_correct'])

    def test_double_encoded_answer_format_incorrect_content_diagnostic(self):
        task = simple_task()
        result = evaluate_episode(task, make_episode(task, answer=json.dumps(task['expected_answer'])))
        self.assertTrue(result['content_correct'])
        self.assertFalse(result['format_correct'])
        self.assertFalse(result['task_correct'])

    def test_wrong_native_type(self):
        task = simple_task()
        result = evaluate_episode(task, make_episode(task, answer=True))
        self.assertFalse(result['format_correct'])
        self.assertFalse(result['content_correct'])

    def test_evaluates_answer_not_outer_envelope(self):
        task = simple_task()
        result = evaluate_episode(task, make_episode(task))
        self.assertTrue(result['task_correct'])
        outer = {'type': 'final_answer', 'answer': task['expected_answer']}
        self.assertFalse(evaluate_episode(task, make_episode(task, answer=outer))['task_correct'])

    def test_no_mutation_of_evidence_or_task(self):
        task = simple_task()
        episode = make_episode(task)
        before = deepcopy((task, episode.to_dict()))
        evaluate_episode(task, episode)
        self.assertEqual((task, episode.to_dict()), before)

    def test_task_id_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_episode(simple_task('T3'), make_episode(simple_task()))

    def test_decimal_numeric_values(self):
        task = {'id': 'T1', 'category': 'Cálculo Financeiro / Lógica Matemática Exata',
                'evaluation_type': 'numeric', 'prompt': 'Synthetic exact numbers.',
                'expected_answer': {'saldo_final': Decimal('7.25'), 'juros_totais': Decimal('2.25')}}
        output = '{"type":"final_answer","answer":{"saldo_final":7.25,"juros_totais":2.25}}'
        self.assertTrue(evaluate_episode(task, make_episode(task, outputs=[output]))['strict_correct'])

    def test_no_fixture_or_answers_returned(self):
        task = domain_task()
        result = evaluate_episode(task, make_episode(task, task['expected_tool_sequence']))
        for key in ('tool_fixtures', 'expected_tool_sequence', 'final_expected_answer', 'full_history',
                    'model_messages', 'tool_trace', 'expected_answer'):
            self.assertNotIn(key, result)
        self.assertNotIn('PRIVATE_', json.dumps(result))


class DomainEvaluationTests(unittest.TestCase):
    def exact(self, task_id):
        task = domain_task(task_id)
        return task, make_episode(task, task['expected_tool_sequence'])

    def test_t5_exact(self):
        task, episode = self.exact('T5')
        result = evaluate_episode(task, episode)
        self.assertTrue(result['strict_correct'])
        self.assertTrue(result['task_correct'])
        self.assertTrue(result['sequence_correct'])
        self.assertTrue(result['dataset_tools_contract_correct'])
        self.assertEqual(result['tool_selection_class'], 'optimal')

    def test_t6_exact_compensation_and_negative_business_result(self):
        task, episode = self.exact('T6')
        result = evaluate_episode(task, episode)
        self.assertTrue(result['strict_correct'])
        self.assertEqual(result['tool_calls_successful'], 5)
        self.assertEqual(result['tool_calls_failed'], 0)
        self.assertTrue(result['domain_actions_correct'])
        self.assertEqual(result['tool_selection_class'], 'optimal')
        self.assertFalse(episode.tool_trace[2]['envelope']['result']['ok'])

    def test_auxiliary_query_allows_task_not_strict(self):
        for task_id, query in [('T5', call('order_workflow_query', {'order_id': 'toy-5', 'fields': ['status']})),
                               ('T6', call('inventory_query', {'skus': ['toy-sku']}))]:
            with self.subTest(task_id=task_id):
                task = domain_task(task_id)
                calls = task['expected_tool_sequence']
                result = evaluate_episode(task, make_episode(task, calls[:1] + [query] + calls[1:]))
                self.assertTrue(result['task_correct'])
                self.assertFalse(result['strict_correct'])
                self.assertFalse(result['sequence_correct'])
                self.assertFalse(result['dataset_tools_contract_correct'])
                self.assertEqual(result['tool_selection_class'], 'acceptable')
                self.assertEqual(result['calls_over_minimum'], 0)

    def test_direct_domain_answer_is_not_resolution(self):
        for task_id in ('T5', 'T6'):
            task = domain_task(task_id)
            result = evaluate_episode(task, make_episode(task))
            self.assertTrue(result['content_correct'])
            self.assertTrue(result['direct_answer'])
            self.assertFalse(result['task_correct'])
            self.assertFalse(result['strict_correct'])

    def test_missing_required_action(self):
        for task_id in ('T5', 'T6'):
            task = domain_task(task_id)
            result = evaluate_episode(task, make_episode(task, task['expected_tool_sequence'][:-1]))
            self.assertFalse(result['task_correct'])
            self.assertFalse(result['domain_actions_correct'])

    def test_wrong_arguments(self):
        task = domain_task()
        calls = deepcopy(task['expected_tool_sequence'])
        calls[-1]['arguments']['line_ids'] = ['b']
        result = evaluate_episode(task, make_episode(task, calls))
        self.assertFalse(result['task_correct'])
        self.assertEqual(result['invalid'], 1)

    def test_wrong_order_without_recovery(self):
        task = domain_task('T6')
        calls = deepcopy(task['expected_tool_sequence'])
        calls[3], calls[4] = calls[4], calls[3]
        result = evaluate_episode(task, make_episode(task, calls))
        self.assertFalse(result['task_correct'])
        self.assertFalse(result['sequence_correct'])

    def test_precondition_error_then_recovery_does_not_remove_task_credit(self):
        task = domain_task()
        calls = task['expected_tool_sequence']
        result = evaluate_episode(task, make_episode(task, [calls[-1]] + calls))
        self.assertTrue(result['task_correct'])
        self.assertFalse(result['strict_correct'])
        self.assertEqual(result['tool_selection_class'], 'invalid')
        self.assertEqual(result['unnecessary_tool_calls'], 0)

    def test_extra_duplicate_refusal_does_not_change_state(self):
        task = domain_task()
        calls = task['expected_tool_sequence']
        result = evaluate_episode(task, make_episode(task, calls + calls[-1:]))
        self.assertTrue(result['task_correct'])
        self.assertFalse(result['strict_correct'])
        self.assertEqual(result['duplicates'], 1)
        self.assertEqual(result['unnecessary_tool_calls'], 1)
        self.assertEqual(result['tool_selection_class'], 'inefficient')

    def test_policy_blocked_extra_does_not_change_state(self):
        task = domain_task()
        result = evaluate_episode(task, make_episode(task, task['expected_tool_sequence'] + [
            call('cancel_order', {'order_id': 'toy-5'})]))
        self.assertTrue(result['task_correct'])
        self.assertEqual(result['unnecessary_tool_calls'], 1)
        self.assertEqual(result['tool_selection_class'], 'invalid')

    def test_final_wrong_but_trace_correct(self):
        task, episode = self.exact('T6')
        episode.final_answer['reservation_released'] = False
        result = evaluate_episode(task, episode)
        self.assertTrue(result['sequence_correct'])
        self.assertFalse(result['content_correct'])
        self.assertFalse(result['task_correct'])

    def test_tampered_result_is_not_action_evidence(self):
        task, episode = self.exact('T6')
        episode.tool_trace[2]['envelope']['result'] = {'ok': True}
        result = evaluate_episode(task, episode)
        self.assertFalse(result['domain_actions_correct'])
        self.assertFalse(result['task_correct'])
        self.assertFalse(result['sequence_correct'])

    def test_technical_failure_preserves_task_not_strict(self):
        task, episode = self.exact('T5')
        episode.technical_success = False
        result = evaluate_episode(task, episode)
        self.assertTrue(result['task_correct'])
        self.assertFalse(result['strict_correct'])
        self.assertFalse(result['dataset_tools_contract_correct'])

    def test_strict_implies_task(self):
        for task_id in ('T5', 'T6'):
            task, episode = self.exact(task_id)
            for answer in (deepcopy(episode.final_answer), None, {}):
                candidate = deepcopy(episode)
                candidate.final_answer = answer
                result = evaluate_episode(task, candidate)
                self.assertFalse(result['strict_correct'] and not result['task_correct'])
                self.assertEqual(result['strict_correct'], result['dataset_tools_contract_correct'])


class EfficiencyTests(unittest.TestCase):
    def grade(self, calls, task_id='T4'):
        task = simple_task(task_id)
        return evaluate_episode(task, make_episode(task, calls))

    def test_optimal_specialized(self):
        result = self.grade([call('json_query', {'data': [{'x': 1}], 'root': [], 'pipeline': []})])
        self.assertTrue(result['task_correct'])
        self.assertEqual(result['tool_selection_class'], 'optimal')
        self.assertEqual(result['calls_over_minimum'], 0)

    def test_general_acceptable(self):
        result = self.grade([call('table_transform', {'rows': [], 'pipeline': []})])
        self.assertEqual(result['tool_selection_class'], 'acceptable')
        self.assertTrue(result['task_correct'])

    def test_successful_redundant_pure_call(self):
        query = call('json_query', {'data': [], 'root': [], 'pipeline': []})
        result = self.grade([query, query])
        self.assertEqual(result['tool_selection_class'], 'inefficient')
        self.assertEqual(result['unnecessary_tool_calls'], 1)
        self.assertEqual(result['duplicates'], 1)
        self.assertEqual(result['calls_over_minimum'], 1)
        self.assertTrue(result['task_correct'])
        self.assertTrue(result['strict_correct'])

    def test_incompatible_extra_inefficient_not_unnecessary(self):
        result = self.grade([call('json_query', {'data': [], 'root': [], 'pipeline': []}),
                             call('calculator_basic', {'expression': '1+1'})])
        self.assertEqual(result['tool_selection_class'], 'inefficient')
        self.assertEqual(result['unnecessary_tool_calls'], 0)
        self.assertTrue(result['task_correct'])

    def test_unknown_dispatched_is_invalid(self):
        result = self.grade([call('unknown', {})])
        self.assertEqual(result['tool_calls_total'], 1)
        self.assertEqual(result['tool_calls_dispatched'], 1)
        self.assertEqual(result['invalid'], 1)
        self.assertEqual(result['tool_selection_class'], 'invalid')
        self.assertFalse(result['selection_valid'])
        self.assertTrue(result['task_correct'])

    def test_invalid_argument_dispatch(self):
        result = self.grade([call('json_query', {'data': []})])
        self.assertEqual(result['invalid'], 1)
        self.assertEqual(result['tool_selection_class'], 'invalid')

    def test_switches_and_success_failed_balance(self):
        result = self.grade([call('json_query', {'data': [], 'root': [], 'pipeline': []}),
                             call('json_query', {'data': []}),
                             call('table_transform', {'rows': [], 'pipeline': []})])
        self.assertEqual(result['switches'], 1)
        self.assertEqual(result['tool_calls_successful'], 2)
        self.assertEqual(result['tool_calls_failed'], 1)
        self.assertEqual(result['tool_calls_dispatched'], 3)

    def test_ninth_blocked_request_counts_total_not_dispatched(self):
        task = simple_task()
        query = call('json_query', {'data': [], 'root': [], 'pipeline': []})
        episode = make_episode(task, [query] * 9)
        result = evaluate_episode(task, episode)
        self.assertEqual(result['tool_calls_total'], 9)
        self.assertEqual(result['tool_calls_dispatched'], 8)
        self.assertEqual(result['tool_calls_blocked'], 1)
        self.assertEqual(result['duplicates'], 8)

    def test_no_sequence_for_non_domain(self):
        self.assertIsNone(self.grade([])['sequence_correct'])

    def test_direct_has_no_assisted_minimum(self):
        result = self.grade([])
        self.assertIsNone(result['minimum_required'])
        self.assertIsNone(result['calls_over_minimum'])
        self.assertIsNone(result['tool_selection_class'])

    def test_canonical_duplicates_ignore_object_order_preserve_types(self):
        self.assertEqual(typed_key({'a': 1, 'b': 2}), typed_key({'b': 2, 'a': 1}))
        self.assertNotEqual(typed_key([1, 2]), typed_key([2, 1]))
        self.assertNotEqual(typed_key(True), typed_key(1))
        self.assertNotEqual(typed_key(Decimal('1')), typed_key(1))

    def test_semantic_tool_error_not_automatically_invalid(self):
        task = simple_task('T2')
        task['expected_answer'] = {'pacotes_a': 5, 'pacotes_b': 11, 'pacotes_c': 0, 'custo_centavos': 68060}
        # No answer credit needed to classify assistance independently.
        result = evaluate_episode(task, make_episode(task, [call('calculator_basic', {'expression': '1/0'})]))
        self.assertEqual(result['tool_selection_class'], 'inefficient')
        self.assertEqual(result['invalid'], 0)
        self.assertTrue(result['selection_valid'])

    def test_general_python_answer_independent_of_tool_type(self):
        task = simple_task()
        backend = FakeCodeBackend()
        result = evaluate_episode(task, make_episode(task, [call('python_executor', {
            'code': 'def main(data): return data', 'data': None})], backend=backend))
        self.assertEqual(result['tool_selection_class'], 'acceptable')
        self.assertTrue(result['task_correct'])

    def test_reject_inconsistent_dispatch_evidence(self):
        task = simple_task()
        episode = make_episode(task)
        episode.tool_calls_dispatched = 1
        with self.assertRaises(ValueError):
            evaluate_episode(task, episode)

    def test_catalog_hash_unchanged(self):
        self.assertEqual(public_catalog_sha256(), HASH)


class CodeAndPhase1ReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = parse_json(Path('prompts/benchmark_v1.json').read_text())

    def test_non_code_format_content_matches_frozen_phase1(self):
        for task in self.tasks:
            if task['id'] in {'T5', 'T6', 'T7', 'T8'}:
                continue
            answer = task['expected_answer']
            output = json_bytes(
                {'type': 'final_answer', 'answer': answer}).decode()
            episode = make_episode(task, outputs=[output])
            verdict = evaluate_episode(task, episode)
            response = answer if isinstance(answer, str) else json_bytes(answer).decode()
            old = evaluate_task(task, {'response': response, 'status': 'success'})
            for key in ('format_correct', 'content_correct', 'strict_correct'):
                self.assertEqual(verdict[key], old[key], (task['id'], key))

    def test_code_requires_safe_backend_no_local_fallback(self):
        task = self.tasks[6]
        episode = make_episode(task, answer=task['expected_answer']['reference_code'])
        result = evaluate_episode(task, episode)
        self.assertFalse(result['content_correct'])
        self.assertEqual(result['evaluation_error'], 'safe_backend_unavailable')
        self.assertTrue(result['technical_success'])

    def test_code_contract_cases_private_backend_only(self):
        task = self.tasks[6]
        backend = FakeCodeBackend()
        result = evaluate_episode(task, make_episode(task, answer=task['expected_answer']['reference_code']),
                                  sandbox_backend=backend)
        self.assertTrue(result['strict_correct'])
        self.assertGreater(len(backend.requests[0]['cases']), len(task['expected_answer']['test_cases']))
        self.assertNotIn('cases', result)

    def test_fenced_code_content_diagnostic_not_format(self):
        task = self.tasks[6]
        answer = chr(96)*3 + 'python\n' + task['expected_answer']['reference_code'] + '\n' + chr(96)*3
        result = evaluate_episode(task, make_episode(task, answer=answer), sandbox_backend=FakeCodeBackend())
        self.assertTrue(result['content_correct'])
        self.assertFalse(result['format_correct'])
        self.assertFalse(result['task_correct'])

    def test_code_syntax_and_function_signature(self):
        task = self.tasks[6]
        for code in ('broken code (', 'def wrong_name(data): return data',
                     'def reconcile(a, b): return a'):
            with self.subTest(code=code):
                result = evaluate_episode(task, make_episode(task, answer=code), sandbox_backend=FakeCodeBackend())
                self.assertFalse(result['format_correct'])
                self.assertFalse(result['content_correct'])

    def test_code_infrastructure_error_is_evaluation_diagnostic(self):
        task = self.tasks[6]
        backend = FakeCodeBackend(error=SandboxError('execution_error', 'backend_infrastructure'))
        result = evaluate_episode(task, make_episode(task, answer=task['expected_answer']['reference_code']),
                                  sandbox_backend=backend)
        self.assertEqual(result['evaluation_error'], 'evaluation_infrastructure_failure')
        self.assertTrue(result['technical_success'])
        self.assertFalse(result['task_correct'])

    def test_structured_text_wrong_outer_spaces(self):
        task = self.tasks[9]
        result = evaluate_episode(task, make_episode(task, answer=' ' + task['expected_answer']))
        self.assertFalse(result['format_correct'])
        self.assertFalse(result['task_correct'])

    def test_structured_text_content_is_answer_string(self):
        task = self.tasks[9]
        result = evaluate_episode(task, make_episode(task, answer=task['expected_answer'] + '\r\n'))
        self.assertTrue(result['strict_correct'])

    def test_reference_code_passes_real_docker_frozen_cases(self):
        # Real secure sandbox, no model/provider, only frozen reference functions.
        backend = DockerSandboxBackend()
        self.assertTrue(backend.is_available())
        for task in self.tasks[6:8]:
            with self.subTest(task_id=task['id']):
                result = evaluate_episode(task, make_episode(task, answer=task['expected_answer']['reference_code']),
                                          sandbox_backend=backend)
                self.assertTrue(result['strict_correct'], result)
        backend.cleanup()


if __name__ == '__main__':
    unittest.main()
