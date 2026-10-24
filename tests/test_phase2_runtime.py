"""Provider/episode bridge, warm-up and cooperative deadline tests without models."""
from dataclasses import asdict
import io
import json
import os
import unittest
from unittest.mock import patch

from benchmark.orchestration.phase2_runtime import (
    EpisodeDeadline, EPISODE_TIMEOUT_SECONDS, run_transformers_episode,
)
from benchmark.providers.transformers_phase2 import EpisodeProviderError, WARMUP_PROMPTS
from benchmark.tools.catalog import public_catalog_sha256
from test_phase2_transformers_provider import prepared, torch_stub, device
from test_phase2_episode import domain, FakeBackend


def task(task_id='T1'):
    return {'id': task_id, 'prompt': 'Synthetic public task.',
            'tools_prompt': 'Synthetic public domain task.',
            'raw_prompt': 'PRIVATE_RAW', 'tool_fixtures': {'private': 'PRIVATE_FIXTURE'},
            'expected_tool_sequence': ['PRIVATE_SEQUENCE'], 'final_expected_answer': 'PRIVATE_ANSWER'}


def final(answer='done'):
    return json.dumps({'type': 'final_answer', 'answer': answer})


def call():
    return json.dumps({'type': 'tool_call', 'tool': 'calculator_basic', 'arguments': {'expression': '2+3'}})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.torch = torch_stub()
        self.patcher = patch.dict('sys.modules', {'torch': self.torch})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_tool_dispatch_then_final_full_history(self):
        adapter, tokenizer, model = prepared([call(), final('5')])
        result = run_transformers_episode(task(), adapter, run_id='synthetic-run')
        episode = result.episode
        self.assertEqual(episode.termination_reason, 'final_answer')
        self.assertEqual(episode.final_answer, '5')
        self.assertEqual(episode.model_outputs_total, 2)
        self.assertEqual(episode.tool_calls_dispatched, 1)
        self.assertEqual(len(tokenizer.templates), 2)
        self.assertEqual(len(tokenizer.tokenizations), 2)
        document = json.loads(tokenizer.templates[1][0][0]['content'])
        self.assertEqual(list(document), ['protocol', 'tool_catalog', 'task', 'history'])
        self.assertEqual(document['history'], [json.loads(call()), {'ok': True, 'result': {'value': '5'}}])
        self.assertEqual([t.turn_index for t in result.turn_metrics], [1, 2])
        self.assertEqual(result.episode.context_token_counts, [7, 7])
        self.assertEqual(result.run_metadata['total_input_tokens'], 14)
        self.assertEqual(result.run_metadata['total_output_tokens'], 4)
        self.assertEqual(len(model.options), 2)

    def test_provider_does_not_modify_json_or_history(self):
        raw = ' \n' + final({'value': 1}) + '  \n'
        adapter, _, _ = prepared([raw])
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.turn_metrics[0].raw_output, raw)
        self.assertEqual(result.episode.model_messages, [raw])
        self.assertTrue(result.episode.direct_answer)

    def test_invalid_protocol_is_not_provider_failure(self):
        adapter, _, model = prepared(['not json'])
        result = run_transformers_episode(task(), adapter)
        self.assertTrue(result.episode.technical_success)
        self.assertFalse(result.episode.protocol_correct)
        self.assertFalse(result.episode.infrastructure_failure)
        self.assertEqual(result.episode.termination_reason, 'invalid_protocol')
        self.assertEqual(result.turn_metrics[0].status, 'success')
        self.assertEqual(len(model.options), 1)

    def test_generate_failure_reaches_episode_as_model_failure(self):
        adapter, _, model = prepared()
        model.error = RuntimeError('private token /path')
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.episode.termination_reason, 'model_failure')
        self.assertFalse(result.episode.technical_success)
        self.assertIsNone(result.episode.protocol_correct)
        self.assertEqual(result.turn_metrics[0].error_code, 'generation_failure')
        self.assertNotIn('private token', str(result))

    def test_real_counter_exact_limit_accepted(self):
        adapter, _, model = prepared([final()], input_count=15872)
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.episode.termination_reason, 'final_answer')
        self.assertEqual(result.episode.context_token_counts, [15872])
        self.assertTrue(result.episode.token_count_validated)
        self.assertEqual(len(model.options), 1)

    def test_real_counter_above_limit_no_generate(self):
        adapter, tokenizer, model = prepared([final()], input_count=15873)
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.episode.termination_reason, 'context_limit_exceeded')
        self.assertFalse(result.episode.infrastructure_failure)
        self.assertEqual(result.episode.model_outputs_total, 0)
        self.assertEqual(model.options, [])
        self.assertEqual(result.turn_metrics, [])
        self.assertEqual(len(result.context_checks), 1)
        self.assertTrue(result.context_checks[0].context_limit)
        self.assertEqual(len(tokenizer.templates), 1)

    def test_context_growth_checked_each_turn(self):
        count = lambda text: 15873 if '"history":[{' in text else 7
        adapter, _, model = prepared([call(), final()], input_count=count)
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.episode.termination_reason, 'context_limit_exceeded')
        self.assertEqual(result.episode.model_outputs_total, 1)
        self.assertEqual(result.episode.tool_calls_dispatched, 1)
        self.assertEqual(len(model.options), 1)
        self.assertEqual(result.episode.context_token_counts, [7, 15873])

    def test_capacity_preflight_blocks_before_episode(self):
        adapter, tokenizer, model = prepared([final()], capacity=8192)
        with self.assertRaises(EpisodeProviderError):
            run_transformers_episode(task(), adapter)
        self.assertEqual(model.options, [])
        self.assertEqual(tokenizer.templates, [])

    def test_nine_outputs_limit_owned_by_episode(self):
        adapter, tokenizer, model = prepared([call()] * 9 + [final('next episode')])
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.episode.termination_reason, 'tool_call_limit')
        self.assertEqual(result.episode.model_outputs_total, 9)
        self.assertEqual(result.episode.tool_calls_dispatched, 8)
        self.assertEqual(len(result.turn_metrics), 9)
        self.assertEqual(len(model.options), 9)
        next_result = run_transformers_episode(task(), adapter)
        self.assertEqual(next_result.episode.termination_reason, 'final_answer')
        self.assertEqual(next_result.turn_metrics[0].turn_index, 1)
        self.assertEqual(next_result.run_metadata['successful_turns'], 10)
        self.assertEqual(len(tokenizer.templates), 10)

    def test_eighth_tool_then_ninth_final_allowed(self):
        adapter, _, _ = prepared([call()] * 8 + [final()])
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.episode.termination_reason, 'final_answer')
        self.assertEqual(result.episode.tool_calls_dispatched, 8)
        self.assertEqual(len(result.turn_metrics), 9)

    def test_warmup_exactly_two_discarded_from_all_metrics(self):
        adapter, tokenizer, model = prepared(['discard one', 'discard two', final()])
        adapter.warm_up()
        self.assertTrue(adapter.warmup_completed)
        self.assertEqual(len(model.options), 2)
        self.assertEqual([messages[0]['content'] for messages, _ in tokenizer.templates], list(WARMUP_PROMPTS))
        self.assertEqual(adapter.turn_metrics, [])
        self.assertEqual(adapter.context_checks, [])
        self.assertEqual(adapter.run_metadata()['total_input_tokens'], 0)
        self.assertEqual(adapter.run_metadata()['total_output_tokens'], 0)
        result = run_transformers_episode(task(), adapter)
        self.assertEqual(result.episode.model_outputs_total, 1)
        self.assertEqual(len(result.turn_metrics), 1)
        self.assertEqual(result.run_metadata['successful_turns'], 1)
        self.assertEqual(json.loads(tokenizer.templates[2][0][0]['content'])['history'], [])
        self.assertTrue(all(option['max_new_tokens'] == 512 for option in model.options))

    def test_warmup_cuda_synchronizes_after_two_generations(self):
        self.torch.cuda.is_available = lambda: True
        adapter, _, _ = prepared(['one', 'two'], where='cuda:1')
        adapter.warm_up()
        self.assertEqual(self.torch.cuda.synchronize.call_count, 5)
        self.assertEqual(self.torch.cuda.synchronize.call_args.args, (device('cuda:1'),))

    def test_warmup_does_not_touch_external_domain_session(self):
        session, _ = domain('T5')
        before = session.state_hash()
        adapter, _, _ = prepared(['one', 'two'])
        adapter.warm_up()
        self.assertEqual(session.state_hash(), before)
        self.assertEqual(session.call_log, [])

    def test_warmup_cannot_repeat_or_erase_existing_official_metrics(self):
        adapter, _, _ = prepared(['one', 'two', final()])
        adapter.warm_up()
        with self.assertRaises(EpisodeProviderError):
            adapter.warm_up()
        run_transformers_episode(task(), adapter)
        with self.assertRaises(EpisodeProviderError):
            adapter.warm_up()
        self.assertEqual(adapter.run_metadata()['successful_turns'], 1)

    def test_warmup_failure_propagates_without_official_turn(self):
        adapter, _, model = prepared()
        model.error = RuntimeError('failed warmup')
        with self.assertRaises(EpisodeProviderError):
            adapter.warm_up()
        self.assertFalse(adapter.warmup_completed)
        self.assertEqual(adapter.turn_metrics, [])
        self.assertEqual(adapter.run_metadata()['failed_turns'], 0)

    def test_private_task_fields_and_hf_secret_absent(self):
        adapter, _, model = prepared([call(), final()])
        output = io.StringIO()
        with patch.dict(os.environ, {'HF_TOKEN': 'FAKE_SECRET_NEVER_IN_CONTEXT'}), patch('sys.stdout', output):
            result = run_transformers_episode(task(), adapter)
        self.assertEqual(output.getvalue(), '')
        text = str(result)
        for private in ('FAKE_SECRET_NEVER_IN_CONTEXT', 'PRIVATE_RAW', 'PRIVATE_FIXTURE',
                        'PRIVATE_SEQUENCE', 'PRIVATE_ANSWER'):
            self.assertNotIn(private, text)

    def test_domain_session_injected_without_private_snapshot_in_context(self):
        session, cases = domain('T5')
        name, args, *_ = cases[0]
        raw = json.dumps({'type': 'tool_call', 'tool': name, 'arguments': args})
        adapter, tokenizer, _ = prepared([raw, final()])
        result = run_transformers_episode(task('T5'), adapter, domain_session=session)
        self.assertEqual(result.episode.tool_results[0]['ok'], True)
        self.assertEqual(len(session.call_log), 1)
        self.assertNotIn('PRIVATE_CASE', str(tokenizer.templates))
        self.assertNotIn('tool_fixtures', str(tokenizer.templates))

    def test_sandbox_backend_is_forwarded_without_local_execution(self):
        raw = json.dumps({'type': 'tool_call', 'tool': 'python_executor',
                          'arguments': {'code': 'def main(data): return data', 'data': 7}})
        backend = FakeBackend()
        adapter, _, _ = prepared([raw, final()])
        result = run_transformers_episode(task(), adapter, sandbox_backend=backend)
        self.assertEqual(result.episode.tool_results, [{'ok': True, 'result': {'value': 7}}])
        self.assertEqual(len(backend.calls), 1)

    def test_budget_stop_between_turns_preserves_technical_success(self):
        clock = [0]
        def counter(text):
            if '\"history\":[{' in text:
                clock[0] = 601
            return 7
        adapter, _, model = prepared([call(), final()], input_count=counter)
        deadline = EpisodeDeadline.start(clock=lambda: clock[0])
        result = run_transformers_episode(task(), adapter, deadline=deadline)
        self.assertEqual(result.episode.termination_reason, 'timeout')
        self.assertTrue(result.episode.technical_success)
        self.assertTrue(result.episode.protocol_correct)
        self.assertFalse(result.episode.infrastructure_failure)
        self.assertEqual(result.episode.tool_calls_dispatched, 1)
        self.assertEqual(len(model.options), 1)

    def test_deadline_default_contract_is_600_seconds(self):
        deadline = EpisodeDeadline.start(clock=lambda: 10)
        self.assertEqual(deadline.expires_at, 610)
        self.assertEqual(EPISODE_TIMEOUT_SECONDS, 600)

    def test_expired_budget_before_generation_not_invalid_protocol(self):
        adapter, _, model = prepared([final()])
        deadline = EpisodeDeadline(0, clock=lambda: 1)
        result = run_transformers_episode(task(), adapter, deadline=deadline)
        self.assertEqual(result.episode.termination_reason, 'timeout')
        self.assertEqual(result.episode.error_code, 'episode_timeout')
        self.assertIsNone(result.episode.protocol_correct)
        self.assertFalse(result.episode.technical_success)
        self.assertFalse(result.episode.infrastructure_failure)
        self.assertEqual(model.options, [])

    def test_late_generation_output_never_dispatched(self):
        clock = [0]
        adapter, _, model = prepared([call()])
        model.after_generate = lambda: clock.__setitem__(0, 601)
        deadline = EpisodeDeadline.start(clock=lambda: clock[0])
        result = run_transformers_episode(task(), adapter, deadline=deadline)
        self.assertEqual(result.episode.termination_reason, 'timeout')
        self.assertFalse(result.episode.technical_success)
        self.assertEqual(result.episode.tool_calls_dispatched, 0)
        self.assertEqual(result.episode.tool_trace, [])
        self.assertIsNone(result.turn_metrics[0].raw_output)

    def test_cancellation_is_not_normal_model_text(self):
        adapter, _, model = prepared([call()])
        deadline = EpisodeDeadline.start()
        deadline.request_cancel()
        result = run_transformers_episode(task(), adapter, deadline=deadline)
        self.assertEqual(result.episode.termination_reason, 'timeout')
        self.assertEqual(result.episode.error_code, 'episode_cancelled')
        self.assertEqual(model.options, [])

    def test_runtime_results_structurally_deterministic_except_latency(self):
        results = []
        for _ in range(2):
            adapter, _, _ = prepared([call(), final()])
            with patch('benchmark.providers.transformers_provider.time.perf_counter', side_effect=[1, 2, 3, 4]):
                result = run_transformers_episode(task(), adapter)
            results.append(asdict(result))
        self.assertEqual(results[0], results[1])
        self.assertEqual(public_catalog_sha256(), 'cfe666d04c4870461bd3c18bfd5b78a393d5778cd6fb2aac6399d6701dade92b')


if __name__ == '__main__':
    unittest.main()
