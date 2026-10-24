"""Synthetic episode tests: callbacks return canned text, never load a model."""
from copy import deepcopy
from decimal import Decimal
import json
import unittest
from unittest.mock import patch

from benchmark.orchestration.episode import (
    EpisodeLimits, OPERATIONAL_PROTOCOL, run_episode,
)
from benchmark.orchestration.tool_trace import finite_json_bytes
from benchmark.tools.catalog import (
    get_public_catalog, public_catalog_sha256, serialize_public_catalog,
)
from benchmark.tools.domain_tools import DomainSession
from benchmark.tools.sandbox import SandboxBackend, SandboxError, SandboxResult


CATALOG_HASH = "cfe666d04c4870461bd3c18bfd5b78a393d5778cd6fb2aac6399d6701dade92b"


def final(answer="synthetic answer"):
    return json.dumps({"type": "final_answer", "answer": answer})


def call(tool="calculator_basic", arguments=None):
    return json.dumps({"type": "tool_call", "tool": tool,
                       "arguments": {"expression": "2+3"} if arguments is None else arguments})


class FakeModel:
    """Save full callback inputs; exhausted scripts fail instead of guessing output."""
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.contexts = []

    def __call__(self, context):
        self.contexts.append(context)
        item = self.outputs[len(self.contexts) - 1]
        if isinstance(item, Exception):
            raise item
        return item


class FakeBackend(SandboxBackend):
    """No code execution, filesystem access or nested tool calls in this double."""
    def __init__(self, error=None, available=True):
        self.error = error
        self.available = available
        self.calls = []

    def is_available(self): return self.available
    def validate_environment(self): return {"available": self.available}
    def cancel(self): pass
    def cleanup(self): pass

    def run_python(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.error:
            raise self.error
        if kwargs["mode"] == "test_runner":
            return SandboxResult({"passed": True, "cases": [
                {"index": 0, "passed": True, "actual": 3, "exception": None}]})
        return SandboxResult({"value": kwargs["data"]})


def domain(task_id):
    """Tiny private worlds, unrelated to any official task data or answer."""
    if task_id == "T5":
        cases = [
            ("lookup_order", {"order_id": "toy-order"}, [], {"status": "shipped"}),
            ("create_return", {"order_id": "toy-order", "line_ids": ["toy-line"]},
             ["private-case-0"], {"ok": True}),
        ]
    else:
        cases = [
            ("get_stock", {"sku": "toy-sku"}, [], {"available": 3}),
            ("reserve_stock", {"order_id": "toy-order", "sku": "toy-sku", "quantity": 1},
             ["private-case-0"], {"ok": True, "reservation_id": "toy-reservation"}),
        ]
    catalog = {tool.name: tool.to_dict() for tool in get_public_catalog()}
    fixtures = {
        "cases": [{"id": f"private-case-{i}", "tool": name, "arguments": args,
                   "requires": requires, "result": result}
                  for i, (name, args, requires, result) in enumerate(cases)],
        "blocked_tools": [], "dispatch_rules": ["PRIVATE_FIXTURE_SENTINEL"],
        "errors": {code: {"ok": False, "error": code} for code in (
            "unknown_tool", "invalid_arguments", "policy_violation",
            "duplicate_call", "precondition_failed")},
    }
    return DomainSession(tool_catalog=[catalog[name] for name, *_ in cases],
                         tool_fixtures=fixtures), cases


def task(task_id="T1"):
    return {"id": task_id, "prompt": "Synthetic public task.",
            "tools_prompt": "Synthetic public domain task.",
            "raw_prompt": "PRIVATE_RAW_SENTINEL", "expected_answer": "PRIVATE_ANSWER_SENTINEL",
            "tool_fixtures": {"hidden": "PRIVATE_FIXTURE_SENTINEL"},
            "expected_tool_sequence": ["PRIVATE_SEQUENCE_SENTINEL"],
            "final_expected_answer": "PRIVATE_FINAL_SENTINEL",
            "evaluation_notes": "PRIVATE_NOTES_SENTINEL",
            "evaluation_notes_tools": "PRIVATE_TOOLS_NOTES_SENTINEL"}


class EpisodeTests(unittest.TestCase):
    def run_fake(self, outputs, *, public_task=None, **kwargs):
        model = FakeModel(outputs)
        result = run_episode(task() if public_task is None else public_task, model,
                             public_catalog=serialize_public_catalog(), **kwargs)
        self.assertEqual(result.model_contexts, model.contexts)
        return result, model

    def test_direct_final_first_output(self):
        result, model = self.run_fake([final()])
        self.assertTrue(result.direct_answer)
        self.assertTrue(result.technical_success)
        self.assertTrue(result.protocol_correct)
        self.assertFalse(result.infrastructure_failure)
        self.assertEqual(result.termination_reason, "final_answer")
        self.assertEqual(result.model_outputs_total, 1)
        self.assertEqual(result.tool_calls_dispatched, 0)
        self.assertEqual(result.tool_results, [])
        self.assertEqual(result.tool_trace, [])
        self.assertEqual(len(model.contexts), 1)

    def test_final_answers_preserve_json_types(self):
        for answer in (None, False, 7, "", [], {"answer": [2, True]}):
            with self.subTest(answer=answer):
                result, _ = self.run_fake([final(answer)])
                self.assertEqual(result.final_answer, answer)
                self.assertIs(type(result.final_answer), type(answer))
                self.assertEqual(result.termination_reason, "final_answer")

    def test_one_tool_and_public_envelope_history(self):
        raw = call()
        result, model = self.run_fake([raw, final("5")])
        expected = {"ok": True, "result": {"value": "5"}}
        self.assertEqual(result.model_outputs_total, 2)
        self.assertEqual(result.tool_calls_dispatched, 1)
        self.assertFalse(result.direct_answer)
        self.assertEqual(result.tool_results, [expected])
        self.assertEqual(json.loads(model.contexts[1])["history"], [json.loads(raw), expected])
        self.assertEqual(result.full_history["history"][-1], json.loads(final("5")))

    def test_multiple_tools_context_and_trace_order(self):
        outputs = [call(arguments={"expression": "4+1"}),
                   call(arguments={"expression": "5*2"}), final("10")]
        result, model = self.run_fake(outputs)
        last = json.loads(model.contexts[-1])
        self.assertEqual(last["history"], [json.loads(outputs[0]), result.tool_results[0],
                                           json.loads(outputs[1]), result.tool_results[1]])
        self.assertEqual([entry["call_index"] for entry in result.tool_trace], [1, 2])
        self.assertEqual([entry["arguments"]["expression"] for entry in result.tool_trace],
                         ["4+1", "5*2"])

    def assert_invalid(self, raw):
        result, model = self.run_fake([raw, final()])
        self.assertEqual(result.termination_reason, "invalid_protocol")
        self.assertTrue(result.technical_success)
        self.assertFalse(result.protocol_correct)
        self.assertFalse(result.infrastructure_failure)
        self.assertEqual(result.model_outputs_total, 1)
        self.assertEqual(result.tool_calls_dispatched, 0)
        self.assertEqual(result.model_messages, [raw])
        self.assertEqual(result.full_history["history"], [raw])
        self.assertEqual(len(model.contexts), 1)
        return result

    def test_invalid_broken_json(self): self.assert_invalid('{"type":')
    def test_invalid_prose_and_json(self): self.assert_invalid("Answer: " + final())
    def test_invalid_fenced_json(self): self.assert_invalid(chr(96)*3 + "json\n" + final() + "\n" + chr(96)*3)
    def test_invalid_unknown_envelope(self): self.assert_invalid('{"type":"other"}')
    def test_invalid_extra_fields(self): self.assert_invalid('{"type":"final_answer","answer":1,"extra":2}')
    def test_invalid_argument_type(self): self.assert_invalid('{"type":"tool_call","tool":"x","arguments":[]}')
    def test_invalid_duplicate_keys(self): self.assert_invalid('{"type":"final_answer","answer":1,"answer":2}')
    def test_invalid_non_finite(self): self.assert_invalid('{"type":"final_answer","answer":NaN}')
    def test_invalid_trailing_document(self): self.assert_invalid(final() + final())

    def test_invalid_after_valid_output_preserves_count_and_trace(self):
        result, model = self.run_fake([call(), "broken", final()])
        self.assertEqual(result.model_outputs_total, 2)
        self.assertEqual(result.tool_calls_dispatched, 1)
        self.assertFalse(result.protocol_correct)
        self.assertTrue(result.technical_success)
        self.assertEqual(len(result.tool_trace), 1)
        self.assertEqual(len(model.contexts), 2)

    def test_unknown_tool_is_dispatched_and_can_recover(self):
        result, model = self.run_fake([call("missing", {}), final()])
        self.assertEqual(result.tool_calls_dispatched, 1)
        self.assertEqual(result.tool_results[0]["error"]["code"], "unknown_tool")
        self.assertEqual(json.loads(model.contexts[1])["history"][1], result.tool_results[0])
        self.assertTrue(result.technical_success)
        self.assertTrue(result.protocol_correct)
        self.assertEqual(result.termination_reason, "final_answer")

    def test_invalid_arguments_can_recover(self):
        result, _ = self.run_fake([call(arguments={"expression": True}), final()])
        self.assertEqual(result.tool_results[0]["error"]["code"], "invalid_arguments")
        self.assertTrue(result.technical_success)
        self.assertTrue(result.protocol_correct)
        self.assertFalse(result.infrastructure_failure)

    def test_infeasible_can_recover(self):
        args = {"variables": [{"name": "x", "min": 0, "max": 1}],
                "constraints": [{"coefficients": {"x": 1}, "relation": "ge", "rhs": 2}],
                "objectives": [{"coefficients": {"x": 1}, "direction": "min"}]}
        result, _ = self.run_fake([call("integer_optimizer", args), final()])
        self.assertEqual(result.tool_results[0]["error"]["code"], "infeasible")
        self.assertTrue(result.technical_success)
        self.assertEqual(result.termination_reason, "final_answer")

    def test_precondition_failure_can_recover(self):
        session, cases = domain("T5")
        name, args, *_ = cases[1]
        before = session.state_hash()
        result, _ = self.run_fake([call(name, args), final()], public_task=task("T5"),
                                  domain_session=session)
        self.assertEqual(result.tool_results[0]["error"]["code"], "precondition_failed")
        self.assertEqual(session.state_hash(), before)
        self.assertTrue(result.technical_success)
        self.assertEqual(result.termination_reason, "final_answer")

    def test_executor_uses_only_explicit_backend(self):
        backend = FakeBackend()
        args = {"code": "def main(data): return data", "data": {"toy": 3}}
        result, _ = self.run_fake([call("python_executor", args), final()], sandbox_backend=backend)
        self.assertEqual(result.tool_results[0], {"ok": True, "result": {"value": {"toy": 3}}})
        self.assertEqual(backend.calls[0]["mode"], "executor")
        self.assertEqual(backend.calls[0]["data"], {"toy": 3})
        self.assertTrue(result.technical_success)

    def test_test_runner_uses_only_explicit_backend(self):
        backend = FakeBackend()
        args = {"code": "def add_one(x): return x+1", "function": "add_one",
                "cases": [{"args": [2], "kwargs": {}, "expected": 3}]}
        result, _ = self.run_fake([call("python_test_runner", args), final()], sandbox_backend=backend)
        self.assertTrue(result.tool_results[0]["result"]["passed"])
        self.assertEqual(backend.calls[0]["mode"], "test_runner")
        self.assertEqual(backend.calls[0]["cases"], args["cases"])
        self.assertLessEqual(backend.calls[0]["timeout"], 2)
        self.assertTrue(result.technical_success)

    def test_submitted_code_timeout_can_recover(self):
        backend = FakeBackend(SandboxError("tool_timeout", "submitted_code"))
        result, _ = self.run_fake([call("python_executor", {"code": "def main(data): return data", "data": 2}),
                                  final()], sandbox_backend=backend)
        self.assertEqual(result.tool_results[0]["error"]["code"], "tool_timeout")
        self.assertTrue(result.technical_success)
        self.assertFalse(result.infrastructure_failure)
        self.assertEqual(result.termination_reason, "final_answer")
        self.assertEqual(len(backend.calls), 1)

    def test_submitted_code_exception_can_recover(self):
        backend = FakeBackend(SandboxError("execution_error", "submitted_code"))
        result, _ = self.run_fake([call("python_executor", {"code": "def main(data): return data", "data": 2}),
                                  final()], sandbox_backend=backend)
        self.assertEqual(result.tool_results[0]["error"]["code"], "execution_error")
        self.assertTrue(result.technical_success)
        self.assertFalse(result.infrastructure_failure)

    def test_callback_failure_before_any_output(self):
        result, _ = self.run_fake([RuntimeError("private-token /private/path")])
        self.assertFalse(result.technical_success)
        self.assertTrue(result.infrastructure_failure)
        self.assertIsNone(result.protocol_correct)
        self.assertEqual(result.termination_reason, "model_failure")
        self.assertEqual(result.model_outputs_total, 0)
        self.assertNotIn("private-token", str(result.to_dict()))
        self.assertNotIn("/private/path", str(result.to_dict()))

    def test_callback_failure_after_valid_message(self):
        result, _ = self.run_fake([call(), RuntimeError("private")])
        self.assertFalse(result.technical_success)
        self.assertTrue(result.protocol_correct)
        self.assertEqual(result.model_outputs_total, 1)
        self.assertEqual(result.tool_calls_dispatched, 1)
        self.assertEqual(result.termination_reason, "model_failure")

    def test_callback_contract_requires_raw_string(self):
        result, _ = self.run_fake([{"type": "final_answer", "answer": 1}])
        self.assertEqual(result.termination_reason, "model_failure")
        self.assertIsNone(result.protocol_correct)
        self.assertFalse(result.technical_success)

    def test_backend_failure_stops_episode(self):
        backend = FakeBackend(SandboxError("execution_error", "backend_infrastructure"))
        result, model = self.run_fake([call("python_executor", {"code": "def main(data): return data", "data": None}),
                                      final()], sandbox_backend=backend)
        self.assertFalse(result.technical_success)
        self.assertTrue(result.infrastructure_failure)
        self.assertTrue(result.protocol_correct)
        self.assertEqual(result.termination_reason, "tool_infrastructure_failure")
        self.assertTrue(result.tool_trace[0]["infrastructure_failure"])
        self.assertEqual(len(model.contexts), 1)
        self.assertEqual(result.full_history["history"][-1], result.tool_results[0])

    def test_missing_backend_never_falls_back(self):
        with patch("benchmark.tools.code_tools.DockerSandboxBackend") as fallback:
            result, _ = self.run_fake([call("python_executor", {"code": "def main(data): return data", "data": None})])
        fallback.assert_not_called()
        self.assertEqual(result.termination_reason, "tool_infrastructure_failure")
        self.assertTrue(result.infrastructure_failure)

    def test_unavailable_backend_stops_episode(self):
        backend = FakeBackend(available=False)
        result, _ = self.run_fake([call("python_executor", {"code": "def main(data): return data", "data": None})],
                                  sandbox_backend=backend)
        self.assertEqual(result.termination_reason, "tool_infrastructure_failure")
        self.assertFalse(backend.calls)

    def test_unexpected_dispatch_failure_sanitized(self):
        with patch("benchmark.orchestration.episode.dispatch_tool", side_effect=RuntimeError("private-secret")):
            result, _ = self.run_fake([call()])
        self.assertEqual(result.termination_reason, "tool_infrastructure_failure")
        self.assertFalse(result.technical_success)
        self.assertEqual(result.tool_calls_dispatched, 1)
        self.assertNotIn("private-secret", str(result))

    def test_exactly_eight_calls_then_ninth_output_final(self):
        result, model = self.run_fake([call()] * 8 + [final()])
        self.assertEqual(result.tool_calls_dispatched, 8)
        self.assertEqual(len(result.tool_trace), 8)
        self.assertEqual(result.model_outputs_total, 9)
        self.assertEqual(len(model.contexts), 9)
        self.assertEqual(result.termination_reason, "final_answer")
        self.assertTrue(result.technical_success)

    def test_ninth_call_not_dispatched(self):
        result, model = self.run_fake([call()] * 9 + [final()])
        self.assertEqual(result.termination_reason, "tool_call_limit")
        self.assertEqual(result.model_outputs_total, 9)
        self.assertEqual(result.tool_calls_dispatched, 8)
        self.assertEqual(len(result.tool_trace), 8)
        self.assertEqual(len(result.tool_results), 8)
        self.assertEqual(len(model.contexts), 9)
        self.assertTrue(result.protocol_correct)
        self.assertTrue(result.technical_success)
        self.assertFalse(result.direct_answer)

    def test_model_output_limit_no_extra_callback(self):
        result, model = self.run_fake([call()] * 3, limits=EpisodeLimits(max_model_outputs=2))
        self.assertEqual(result.termination_reason, "model_output_limit")
        self.assertEqual(result.model_outputs_total, 2)
        self.assertEqual(result.tool_calls_dispatched, 2)
        self.assertEqual(len(model.contexts), 2)
        self.assertEqual(len(result.tool_results), 2)
        self.assertTrue(result.protocol_correct)
        self.assertTrue(result.technical_success)

    def test_limit_precedence_invalid_ninth_output(self):
        result, _ = self.run_fake([call()] * 8 + ["not JSON"])
        self.assertEqual(result.termination_reason, "invalid_protocol")
        self.assertEqual(result.model_outputs_total, 9)
        self.assertFalse(result.protocol_correct)

    def test_unknown_calls_count_toward_limit(self):
        result, _ = self.run_fake([call("missing", {})] * 9)
        self.assertEqual(result.tool_calls_dispatched, 8)
        self.assertEqual(result.termination_reason, "tool_call_limit")
        self.assertTrue(all(entry["error_code"] == "unknown_tool" for entry in result.tool_trace))

    def test_official_limit_defaults(self):
        self.assertEqual(EpisodeLimits(), EpisodeLimits(8, 9, 16384, 512))
        self.assertEqual(EpisodeLimits().input_token_limit, 15872)

    def test_limits_cannot_exceed_official_call_output_budgets(self):
        for options in ({"max_tool_calls": 9}, {"max_model_outputs": 10},
                        {"max_tool_calls": True}, {"max_model_outputs": 0},
                        {"reserved_output_tokens": 16384}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EpisodeLimits(**options)

    def test_t5_actions_evolve_external_session(self):
        self.assert_domain("T5")

    def test_t6_actions_evolve_external_session(self):
        self.assert_domain("T6")

    def assert_domain(self, task_id):
        session, cases = domain(task_id)
        before = session.state_hash()
        outputs = [call(name, args) for name, args, *_ in cases] + [final("synthetic outcome")]
        result, model = self.run_fake(outputs, public_task=task(task_id), domain_session=session)
        self.assertNotEqual(before, session.state_hash())
        self.assertEqual(len(session.action_log), 1)
        self.assertEqual(result.tool_calls_dispatched, 2)
        self.assertTrue(result.technical_success)
        self.assertEqual(result.termination_reason, "final_answer")
        for context in model.contexts:
            for private in ("private-case-", "PRIVATE_", "tool_fixtures", "expected_tool_sequence",
                            "final_expected_answer", "evaluation_notes", "raw_prompt", "requires"):
                self.assertNotIn(private, context)

    def test_domain_final_without_actions_allowed_without_scoring(self):
        for task_id in ("T5", "T6"):
            with self.subTest(task_id=task_id):
                session, _ = domain(task_id)
                before = session.state_hash()
                result, _ = self.run_fake([final()], public_task=task(task_id), domain_session=session)
                self.assertTrue(result.direct_answer)
                self.assertTrue(result.technical_success)
                self.assertEqual(session.state_hash(), before)
                self.assertNotIn("task_correct", result.to_dict())

    def test_missing_domain_session_controlled_failure(self):
        result, _ = self.run_fake([call("lookup_order", {"order_id": "toy-order"})], public_task=task("T5"))
        self.assertEqual(result.termination_reason, "tool_infrastructure_failure")
        self.assertTrue(result.infrastructure_failure)

    def test_domain_session_is_not_reset_or_shared(self):
        first, _ = domain("T5")
        second, _ = domain("T5")
        output = call("lookup_order", {"order_id": "toy-order"})
        self.run_fake([output, final()], public_task=task("T5"), domain_session=first)
        result, _ = self.run_fake([output, final()], public_task=task("T5"), domain_session=first)
        self.assertEqual(result.tool_results[0]["error"]["code"], "duplicate_call")
        independent, _ = self.run_fake([output, final()], public_task=task("T5"), domain_session=second)
        self.assertTrue(independent.tool_results[0]["ok"])

    def test_all_task_prompt_selection(self):
        for index in range(1, 11):
            task_id = f"T{index}"
            with self.subTest(task_id=task_id):
                public = task(task_id)
                result, model = self.run_fake([final()], public_task=public)
                expected = public["tools_prompt" if index in (5, 6) else "prompt"]
                self.assertEqual(json.loads(model.contexts[0])["task"], expected)
                self.assertEqual(result.full_history["task"], expected)

    def test_private_task_fields_never_serialized(self):
        public = task()
        # Copying the complete task record would be wrong even before serialization.
        class Private:
            def __deepcopy__(self, memo): raise AssertionError("private object copied")
        public["tool_fixtures"] = Private()
        _, model = self.run_fake([call(), final()], public_task=public)
        for context in model.contexts:
            for text in ("PRIVATE_", "tool_fixtures", "expected_answer", "expected_tool_sequence",
                         "final_expected_answer", "evaluation_notes", "raw_prompt"):
                self.assertNotIn(text, context)

    def test_task_text_preserved_without_rewriting(self):
        public = task()
        public["prompt"] = "\n  Synthetic exact spacing.\r\n"
        result, _ = self.run_fake([final()], public_task=public)
        self.assertEqual(result.full_history["task"], public["prompt"])

    def test_catalog_exact_public_projection_lexical_and_hash(self):
        result, _ = self.run_fake([final()])
        catalog = result.full_history["tool_catalog"]
        self.assertEqual(catalog, json.loads(serialize_public_catalog()))
        self.assertEqual(len(catalog), 25)
        self.assertEqual([tool["name"] for tool in catalog], sorted(tool["name"] for tool in catalog))
        self.assertTrue(all(set(tool) == {"name", "description", "parameters", "returns"} for tool in catalog))
        self.assertEqual(public_catalog_sha256(), CATALOG_HASH)

    def test_rejects_modified_or_reordered_catalog(self):
        for catalog in (b"[]", json.dumps(list(reversed(json.loads(serialize_public_catalog())))),
                        serialize_public_catalog() + b" "):
            model = FakeModel([final()])
            with self.subTest(catalog=catalog[:30]), self.assertRaises(ValueError):
                run_episode(task(), model, public_catalog=catalog)
            self.assertEqual(model.contexts, [])

    def test_default_and_explicit_string_catalog_identical(self):
        first = run_episode(task(), FakeModel([final()]))
        second = run_episode(task(), FakeModel([final()]), public_catalog=serialize_public_catalog().decode())
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_fixed_operational_text_and_document_field_order(self):
        _, model = self.run_fake([final()])
        document = json.loads(model.contexts[0])
        self.assertEqual(list(document), ["protocol", "tool_catalog", "task", "history"])
        self.assertEqual(document["protocol"], OPERATIONAL_PROTOCOL)
        self.assertEqual(document["history"], [])

    def test_full_history_representable_without_summary_or_truncation(self):
        outputs = [call()] * 8 + [final("x" * 20000)]
        result, model = self.run_fake(outputs)
        self.assertEqual(len(result.full_history["history"]), 17)
        for i, context in enumerate(model.contexts):
            self.assertEqual(json.loads(context)["history"], result.full_history["history"][:2*i])
        self.assertEqual(result.final_answer, "x" * 20000)
        self.assertEqual(result.model_messages, outputs)

    def test_callback_inputs_immutable_and_snapshot_of_each_turn(self):
        result, model = self.run_fake([call(), final()])
        self.assertTrue(all(isinstance(context, str) for context in model.contexts))
        self.assertEqual(json.loads(model.contexts[0])["history"], [])
        snapshot = result.to_dict()
        snapshot["full_history"]["history"].clear()
        snapshot["tool_trace"].clear()
        self.assertEqual(len(result.full_history["history"]), 3)
        self.assertEqual(len(result.tool_trace), 1)

    def test_decimal_numbers_preserved_in_history_and_final(self):
        raw = '{"type":"tool_call","tool":"python_executor","arguments":{"code":"def main(data): return data","data":0.123456789012345678901}}'
        result, model = self.run_fake([raw, '{"type":"final_answer","answer":0.123456789012345678901}'],
                                      sandbox_backend=FakeBackend())
        self.assertEqual(result.final_answer, Decimal("0.123456789012345678901"))
        self.assertIn('"value":0.123456789012345678901', model.contexts[1])
        self.assertIn(b'0.123456789012345678901', finite_json_bytes(result.to_dict()))

    def test_episode_structural_determinism(self):
        results = []
        for _ in range(2):
            session, cases = domain("T6")
            result, _ = self.run_fake([call(name, args) for name, args, *_ in cases] + [final()],
                                      public_task=task("T6"), domain_session=session)
            results.append(result.to_dict())
        self.assertEqual(results[0], results[1])
        self.assertEqual(finite_json_bytes(results[0]), finite_json_bytes(results[1]))

    def test_preflight_below_limit(self):
        result, _ = self.run_fake([final()], token_counter=lambda _: 15871)
        self.assertTrue(result.token_count_validated)
        self.assertEqual(result.context_token_counts, [15871])
        self.assertEqual(result.termination_reason, "final_answer")

    def test_preflight_exactly_at_limit(self):
        result, _ = self.run_fake([final()], token_counter=lambda _: 15872)
        self.assertEqual(result.termination_reason, "final_answer")
        self.assertTrue(result.technical_success)

    def test_preflight_above_limit_no_callback(self):
        result, model = self.run_fake([final()], token_counter=lambda _: 15873)
        self.assertEqual(result.termination_reason, "context_limit_exceeded")
        self.assertFalse(result.technical_success)
        self.assertFalse(result.infrastructure_failure)
        self.assertIsNone(result.protocol_correct)
        self.assertTrue(result.token_count_validated)
        self.assertEqual(model.contexts, [])
        self.assertEqual(result.model_outputs_total, 0)

    def test_preflight_every_turn_no_truncation(self):
        contexts = []
        def counter(context):
            contexts.append(context)
            return 15872 if len(contexts) == 1 else 15873
        result, model = self.run_fake([call(), final()], token_counter=counter)
        self.assertEqual(result.termination_reason, "context_limit_exceeded")
        self.assertEqual(len(model.contexts), 1)
        self.assertEqual(len(contexts), 2)
        self.assertEqual(len(json.loads(contexts[1])["history"]), 2)
        self.assertTrue(result.technical_success)
        self.assertTrue(result.protocol_correct)
        self.assertFalse(result.infrastructure_failure)

    def test_preflight_counter_receives_exact_callback_input(self):
        contexts = []
        def counter(context):
            contexts.append(context)
            return 10
        result, model = self.run_fake([call(), final()], token_counter=counter)
        self.assertEqual(contexts, model.contexts)
        self.assertEqual(result.context_token_counts, [10, 10])

    def test_no_token_counter_no_estimation(self):
        result, _ = self.run_fake([final()])
        self.assertFalse(result.token_count_validated)
        self.assertEqual(result.context_token_counts, [])

    def test_counter_failure_classified_not_model_protocol(self):
        def counter(_): raise RuntimeError("private-token")
        result, model = self.run_fake([final()], token_counter=counter)
        self.assertEqual(result.termination_reason, "context_preflight_failure")
        self.assertTrue(result.infrastructure_failure)
        self.assertIsNone(result.protocol_correct)
        self.assertNotIn("private-token", str(result))
        self.assertEqual(model.contexts, [])

    def test_counter_must_return_nonnegative_integer(self):
        for value in (-1, 1.5, True, None):
            with self.subTest(value=value):
                result, model = self.run_fake([final()], token_counter=lambda _, v=value: v)
                self.assertEqual(result.termination_reason, "context_preflight_failure")
                self.assertTrue(result.infrastructure_failure)
                self.assertEqual(model.contexts, [])

    def test_metadata_is_private_and_has_no_correctness_fields(self):
        metadata = {"model_id": "fake-model", "run_id": "controller-run", "attempt_id": "controller-attempt"}
        result, model = self.run_fake([final()], metadata=metadata)
        self.assertEqual(result.metadata["phase"], "phase2")
        self.assertEqual(result.metadata["condition"], "tools")
        self.assertEqual(result.metadata["task_id"], "T1")
        for context in model.contexts:
            for text in metadata.values():
                self.assertNotIn(text, context)
        for forbidden in ("task_correct", "strict_correct", "content_correct", "format_correct", "efficiency"):
            self.assertNotIn(forbidden, result.to_dict())
        metadata["run_id"] = "changed"
        self.assertEqual(result.metadata["run_id"], "controller-run")

    def test_configuration_error_before_any_generation(self):
        model = FakeModel([final()])
        for public in ({"id": "T5", "prompt": "must not fall back"}, {"id": "T11", "prompt": "x"},
                       {"id": "T1", "prompt": None}, {"id": "T6", "tools_prompt": ""}):
            with self.subTest(public=public), self.assertRaises(ValueError):
                run_episode(public, model)
        with self.assertRaises(ValueError):
            run_episode(task(), model, metadata={"tool_fixtures": "private"})
        self.assertEqual(model.contexts, [])


if __name__ == "__main__":
    unittest.main()
