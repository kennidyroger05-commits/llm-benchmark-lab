"""Single-dispatch tests; synthetic inputs and Docker, with no model loop."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.orchestration import dispatcher as dispatch
from benchmark.orchestration.dispatcher import DispatchContext, dispatch_tool, serialize_envelope
from benchmark.orchestration.protocol import parse_model_message
from benchmark.tools import get_public_catalog, public_catalog_sha256, serialize_public_catalog
from benchmark.tools import domain_tools
from benchmark.tools.domain_tools import DomainSession
from benchmark.tools.math_tools import ToolExecutionError
from benchmark.tools.sandbox import SandboxBackend, SandboxError, SandboxResult, DockerSandboxBackend

# Reuse the synthetic argument examples already covering the frozen schemas.
from test_phase2_tool_catalog import VALID


class FakeBackend(SandboxBackend):
    """Test double: records requests and returns preset results; never executes code."""
    def __init__(self, *, available=True, error=None):
        self.available = available
        self.error = error
        self.requests = []
    def is_available(self):
        return self.available
    def validate_environment(self):
        return {"available": self.available}
    def run_python(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if self.error:
            raise self.error
        if kwargs["mode"] == "executor":
            return SandboxResult({"value": deepcopy(kwargs["data"])})
        return SandboxResult({"passed": True, "cases": [
            {"index": 0, "passed": True, "actual": None, "exception": None}]})
    def cancel(self): pass
    def cleanup(self): pass


def synthetic_domain():
    """All original operations allowed here; no task IDs or official gabaritos."""
    public = {t.name: t for t in get_public_catalog()}
    names = [n.removeprefix("run_") for n in vars(domain_tools) if n.startswith("run_")]
    originals = [n for n in names if n not in {"inventory_query", "order_workflow_query"}]
    catalog = [{"name": n, "parameters": public[n].parameters} for n in originals]
    fixture = {
        "cases": [
            {"id": f"private-{index}", "tool": name, "arguments": deepcopy(VALID[name]),
             "requires": [], "result": (
                {"status": "shipped", "lines": ["sample"]} if name == "lookup_order" else
                {"eligible": True} if name == "check_return" else
                {"available": 8} if name == "get_stock" else
                {"ok": True, "reservation_id": "synthetic-reservation"} if name == "reserve_stock" else
                {"ok": False, "error": "payment_declined"} if name == "charge" else {"ok": True})}
            for index, name in enumerate(originals)
        ],
        "blocked_tools": [],
        "errors": {code: {"ok": False, "error": code} for code in (
            "unknown_tool", "invalid_arguments", "policy_violation",
            "duplicate_call", "precondition_failed")},
        "dispatch_rules": ["private-rule-sentinel"],
    }
    return catalog, fixture


def domain_session():
    catalog, fixture = synthetic_domain()
    return DomainSession(tool_catalog=catalog, tool_fixtures=fixture)


def arguments(name):
    args = deepcopy(VALID[name])
    if name == "json_query":
        args.update(data={"rows": []}, root=["rows"])
    if name == "python_executor":
        args["code"] = "def main(data): return data"
    if name == "python_test_runner":
        args["code"] = "def func(*args, **kwargs): return None"
    return args


class DispatcherTests(unittest.TestCase):
    def context(self, **changes):
        return DispatchContext(**changes)

    def last(self, context):
        return context.trace.entries[-1]

    def test_registry_exactly_matches_public_catalog(self):
        names = {tool.name for tool in get_public_catalog()}
        self.assertEqual(set(dispatch.TOOL_REGISTRY), names)
        self.assertEqual(len(names), 25)
        with self.assertRaises(TypeError):
            dispatch.TOOL_REGISTRY["alias"] = lambda _: None

    def test_each_of_25_resolves_and_invokes(self):
        for tool in get_public_catalog():
            with self.subTest(tool=tool.name):
                context = self.context(domain_session=domain_session(), sandbox_backend=FakeBackend())
                envelope = dispatch_tool(tool.name, arguments(tool.name), context)
                self.assertEqual(set(envelope), {"ok", "result"}, envelope)
                self.assertTrue(envelope["ok"], envelope)
                self.assertEqual(len(context.trace.entries), 1)
                self.assertTrue(self.last(context).tool_known)
                self.assertTrue(self.last(context).tool_execution_completed)
                self.assertFalse(self.last(context).infrastructure_failure)

    def test_unknown_is_valid_protocol_but_error_in_dispatch(self):
        parsed = parse_model_message('{"type":"tool_call","tool":"not_in_catalog","arguments":{}}')
        context = self.context()
        envelope = dispatch_tool(parsed.tool, parsed.arguments, context)
        self.assertEqual(envelope, {"ok": False, "error": {
            "code": "unknown_tool", "message": "Unknown tool."}})
        entry = self.last(context)
        self.assertTrue(entry.dispatched)
        self.assertTrue(entry.dispatch_completed)
        self.assertFalse(entry.tool_known)
        self.assertFalse(entry.tool_execution_completed)
        self.assertFalse(entry.infrastructure_failure)

    def test_success_and_error_exact_shapes(self):
        context = self.context()
        success = dispatch_tool("calculator_basic", {"expression": "1+3"}, context)
        error = dispatch_tool("calculator_basic", {"expression": "1/0"}, context)
        self.assertEqual(success, {"ok": True, "result": {"value": "4"}})
        self.assertEqual(set(error), {"ok", "error"})
        self.assertEqual(set(error["error"]), {"code", "message"})
        self.assertNotIn("infrastructure_failure", serialize_envelope(error).decode())

    def test_invalid_arguments_are_expected_execution_not_infrastructure(self):
        context = self.context()
        envelope = dispatch_tool("calculator_basic", {"expression": True}, context)
        self.assertEqual(envelope["error"]["code"], "invalid_arguments")
        entry = self.last(context)
        self.assertTrue(entry.tool_known)
        self.assertTrue(entry.dispatch_completed)
        self.assertTrue(entry.tool_execution_completed)
        self.assertFalse(entry.infrastructure_failure)

    def test_tool_resource_error_precedence_preserved(self):
        context = self.context()
        args = {"rows": [], "pipeline": [{"op": "project", "fields": []}] * 33}
        result = dispatch_tool("table_transform", args, context)
        self.assertEqual(result["error"]["code"], "resource_limit")
        self.assertFalse(self.last(context).infrastructure_failure)

    def test_infeasible_is_expected(self):
        context = self.context()
        args = arguments("integer_optimizer")
        args["constraints"] = [{"coefficients": {"x": 1}, "relation": "ge", "rhs": 10}]
        self.assertEqual(dispatch_tool("integer_optimizer", args, context)["error"]["code"], "infeasible")
        self.assertTrue(self.last(context).tool_execution_completed)
        self.assertFalse(self.last(context).infrastructure_failure)

    def test_all_documented_error_codes_preserved_and_messages_sanitized(self):
        expected = {
            "invalid_arguments", "invalid_data", "invalid_expression", "infeasible",
            "resource_limit", "policy_violation", "duplicate_call", "precondition_failed",
            "tool_timeout", "execution_error", "unknown_tool",
        }
        for code in expected:
            context = self.context()
            def fail(_):
                raise ToolExecutionError(code, "Traceback secret /private/token fixture-C99")
            with self.subTest(code=code), patch.dict(dispatch._REGISTRY, {"calculator_basic": fail}):
                envelope = dispatch_tool("calculator_basic", arguments("calculator_basic"), context)
                self.assertEqual(envelope["error"]["code"], code)
                self.assertNotIn("secret", str(envelope))
                self.assertNotIn("private", str(envelope))
                self.assertNotIn("Traceback", str(envelope))
                self.assertEqual(self.last(context).infrastructure_failure, code == "execution_error")

    def test_unexpected_exception_is_sanitized_infrastructure(self):
        context = self.context()
        def fail(_): raise RuntimeError("secret /path")
        with patch.dict(dispatch._REGISTRY, {"calculator_basic": fail}):
            envelope = dispatch_tool("calculator_basic", arguments("calculator_basic"), context)
        self.assertEqual(envelope["error"]["code"], "execution_error")
        self.assertNotIn("secret", context.trace.serialize().decode())
        self.assertTrue(self.last(context).infrastructure_failure)
        self.assertFalse(self.last(context).tool_execution_completed)

    def test_unexpected_validation_failure_is_sanitized(self):
        context = self.context()
        with patch("benchmark.tools.math_tools.validate_tool_arguments", side_effect=RuntimeError("/private")):
            result = dispatch_tool("calculator_basic", {}, context)
        self.assertEqual(result["error"]["code"], "execution_error")
        self.assertTrue(self.last(context).infrastructure_failure)
        self.assertNotIn("/private", context.trace.serialize().decode())

    def test_unrecognized_or_malformed_error_code_is_internal_failure(self):
        for code in ("unpublished_error", ["unhashable"]):
            context = self.context()
            def fail(_): raise ToolExecutionError(code, "private")
            with patch.dict(dispatch._REGISTRY, {"calculator_basic": fail}):
                envelope = dispatch_tool("calculator_basic", arguments("calculator_basic"), context)
            self.assertEqual(envelope["error"]["code"], "execution_error")
            self.assertTrue(self.last(context).infrastructure_failure)

    def test_invalid_returns_rejected_without_serializing_python_objects(self):
        for result in (object(), {1: "nonstring-key"}, {"x": float("nan")},
                       {"x": float("inf")}, {"x": Decimal("Infinity")}, ("tuple",),
                       {"x": "a" * 9000}):
            context = self.context()
            with self.subTest(result_type=type(result)), patch.dict(
                    dispatch._REGISTRY, {"calculator_basic": lambda _: result}):
                envelope = dispatch_tool("calculator_basic", arguments("calculator_basic"), context)
                self.assertEqual(envelope["error"]["code"], "execution_error")
                self.assertTrue(self.last(context).infrastructure_failure)

    def test_exact_decimal_envelope_and_trace_serialization(self):
        context = self.context(sandbox_backend=FakeBackend())
        number = Decimal("0.1234567890123456789012345678901")
        envelope = dispatch_tool("python_executor", {"code": "def main(data): return data",
                                                   "data": number}, context)
        encoded = serialize_envelope(envelope)
        self.assertEqual(json.loads(encoded, parse_float=Decimal)["result"]["value"], number)
        trace = json.loads(context.trace.serialize(), parse_float=Decimal)
        self.assertEqual(trace[0]["envelope"]["result"]["value"], number)
        self.assertIsInstance(envelope["result"]["value"], Decimal)

    def test_bad_envelope_serializer_shapes(self):
        for envelope in ({}, {"ok": 1, "result": 0}, {"ok": True, "result": 0, "extra": 1},
                         {"ok": False, "error": {"code": 0, "message": "x"}},
                         {"ok": False, "error": {"code": "x", "message": "x", "private": 1}}):
            with self.assertRaises(ValueError): serialize_envelope(envelope)

    def test_domain_without_session_is_controlled(self):
        context = self.context()
        result = dispatch_tool("lookup_order", arguments("lookup_order"), context)
        self.assertEqual(result["error"]["code"], "execution_error")
        self.assertTrue(self.last(context).infrastructure_failure)
        self.assertFalse(self.last(context).tool_execution_completed)
        self.assertIsNone(context.domain_session)

    def test_domain_session_injected_and_business_negative_is_success(self):
        session = domain_session()
        context = self.context(domain_session=session)
        before = session.state_hash()
        result = dispatch_tool("charge", arguments("charge"), context)
        self.assertEqual(result, {"ok": True, "result": {"ok": False, "error": "payment_declined"}})
        self.assertNotEqual(session.state_hash(), before)
        self.assertTrue(self.last(context).successful)
        self.assertFalse(self.last(context).failed)
        self.assertFalse(self.last(context).infrastructure_failure)

    def test_two_contexts_and_sessions_are_independent(self):
        a, b = domain_session(), domain_session()
        ca, cb = self.context(domain_session=a), self.context(domain_session=b)
        initial = b.snapshot()
        dispatch_tool("create_return", arguments("create_return"), ca)
        self.assertEqual(b.snapshot(), initial)
        self.assertEqual(len(cb.trace.entries), 0)
        self.assertEqual(len(ca.trace.entries), 1)

    def test_duplicate_error_logged_without_corrupting_state(self):
        session = domain_session()
        context = self.context(domain_session=session)
        dispatch_tool("create_return", arguments("create_return"), context)
        state = session.snapshot()
        result = dispatch_tool("create_return", arguments("create_return"), context)
        self.assertEqual(result["error"]["code"], "duplicate_call")
        self.assertEqual(session.snapshot(), state)
        self.assertEqual(self.last(context).error_code, "duplicate_call")
        self.assertTrue(self.last(context).tool_execution_completed)
        self.assertFalse(self.last(context).infrastructure_failure)

    def test_domain_precondition_and_policy(self):
        for blocked in (False, True):
            catalog, fixtures = synthetic_domain()
            create = next(c for c in fixtures["cases"] if c["tool"] == "create_return")
            lookup = next(c for c in fixtures["cases"] if c["tool"] == "lookup_order")
            create["requires"] = [lookup["id"]]
            if blocked:
                fixtures["blocked_tools"] = ["create_return"]
            session = DomainSession(tool_catalog=catalog, tool_fixtures=fixtures)
            initial = session.snapshot()
            context = self.context(domain_session=session)
            result = dispatch_tool("create_return", arguments("create_return"), context)
            self.assertEqual(result["error"]["code"], "policy_violation" if blocked else "precondition_failed")
            self.assertEqual(session.snapshot(), initial)
            self.assertFalse(self.last(context).infrastructure_failure)

    def test_domain_bad_fixture_result_fails_before_commit(self):
        catalog, fixtures = synthetic_domain()
        create = next(c for c in fixtures["cases"] if c["tool"] == "create_return")
        create["result"] = {"ok": True, "bad": "x" * 9000}
        session = DomainSession(tool_catalog=catalog, tool_fixtures=fixtures)
        before = session.snapshot()
        context = self.context(domain_session=session)
        result = dispatch_tool("create_return", arguments("create_return"), context)
        self.assertEqual(result["error"]["code"], "execution_error")
        self.assertEqual(session.snapshot(), before)
        self.assertEqual(session.action_log, [])
        self.assertTrue(self.last(context).infrastructure_failure)

    def test_auxiliary_queries_do_not_consume_state(self):
        session = domain_session()
        before = session.snapshot()
        context = self.context(domain_session=session)
        for tool in ("order_workflow_query", "inventory_query"):
            self.assertTrue(dispatch_tool(tool, arguments(tool), context)["ok"])
        self.assertEqual(session.snapshot(), before)

    def test_explicit_backend_injection(self):
        backend = FakeBackend()
        context = self.context(sandbox_backend=backend)
        for tool in ("python_executor", "python_test_runner"):
            self.assertTrue(dispatch_tool(tool, arguments(tool), context)["ok"])
        self.assertEqual([r["mode"] for r in backend.requests], ["executor", "test_runner"])
        self.assertTrue(all(set(r) == {"code", "function", "mode", "data", "cases", "timeout"}
                            for r in backend.requests))

    def test_no_backend_or_unavailable_no_fallback(self):
        for backend in (None, FakeBackend(available=False), object()):
            context = self.context(sandbox_backend=backend)
            with patch("benchmark.tools.code_tools.DockerSandboxBackend", side_effect=AssertionError("fallback")):
                result = dispatch_tool("python_executor", arguments("python_executor"), context)
            self.assertEqual(result["error"]["code"], "execution_error")
            self.assertTrue(self.last(context).infrastructure_failure)
            self.assertFalse(self.last(context).tool_execution_completed)

    def test_backend_availability_exception_is_infrastructure(self):
        backend = FakeBackend()
        context = self.context(sandbox_backend=backend)
        with patch.object(backend, "is_available", side_effect=RuntimeError("private socket")):
            result = dispatch_tool("python_executor", arguments("python_executor"), context)
        self.assertEqual(result["error"]["code"], "execution_error")
        self.assertTrue(self.last(context).infrastructure_failure)
        self.assertNotIn("private socket", str(result))

    def test_timeout_with_confirmed_cancellation_is_not_infrastructure(self):
        context = self.context(sandbox_backend=FakeBackend(error=SandboxError(
            "tool_timeout", "submitted_code")))
        result = dispatch_tool("python_executor", arguments("python_executor"), context)
        self.assertEqual(result["error"]["code"], "tool_timeout")
        self.assertFalse(self.last(context).infrastructure_failure)
        self.assertTrue(self.last(context).tool_execution_completed)

    def test_code_exception_vs_infrastructure_same_public_code(self):
        for origin, infrastructure in [("submitted_code", False), ("backend_infrastructure", True)]:
            context = self.context(sandbox_backend=FakeBackend(error=SandboxError(
                "execution_error", origin)))
            result = dispatch_tool("python_executor", arguments("python_executor"), context)
            self.assertEqual(result["error"]["code"], "execution_error")
            self.assertEqual(self.last(context).infrastructure_failure, infrastructure)
            self.assertEqual(self.last(context).error_origin, origin)
            self.assertNotIn(origin, str(result))

    def test_policy_refusal_is_not_infrastructure(self):
        context = self.context(sandbox_backend=FakeBackend())
        result = dispatch_tool("python_executor", {"code": "import os\ndef main(data): return 1",
                                                 "data": None}, context)
        self.assertEqual(result["error"]["code"], "invalid_data")
        self.assertEqual(self.last(context).error_origin, "sandbox_policy")
        self.assertFalse(self.last(context).infrastructure_failure)

    def test_trace_indices_envelopes_flags_and_switch_evidence(self):
        context = self.context()
        calls = [("calculator_basic", {"expression": "1+2"}), ("unknown", {}),
                 ("calculator_basic", {"expression": "1/0"})]
        results = [dispatch_tool(name, args, context) for name, args in calls]
        entries = context.trace.entries
        self.assertEqual([e.call_index for e in entries], [1, 2, 3])
        self.assertEqual([e.tool for e in entries], [c[0] for c in calls])
        self.assertEqual([e.arguments for e in entries], [c[1] for c in calls])
        self.assertEqual([e.envelope for e in entries], results)
        self.assertEqual([e.successful for e in entries], [True, False, False])
        self.assertEqual([e.failed for e in entries], [False, True, True])
        self.assertTrue(all(e.dispatched and e.dispatch_completed for e in entries))

    def test_trace_is_detached_and_inputs_not_mutated(self):
        context = self.context()
        args = {"rows": [{"x": [1]}], "pipeline": []}
        before = deepcopy(args)
        result = dispatch_tool("table_transform", args, context)
        self.assertEqual(args, before)
        result["result"]["rows"][0]["x"].append(99)
        args["rows"][0]["x"].append(88)
        snapshot = context.trace.entries
        snapshot[0].envelope.clear()
        self.assertEqual(context.trace.entries[0].arguments, before)
        self.assertEqual(context.trace.entries[0].envelope["result"]["rows"], before["rows"])

    def test_bad_arguments_cannot_serialize_private_python_objects(self):
        context = self.context(domain_session=domain_session())
        result = dispatch_tool("lookup_order", context.domain_session, context)
        self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.assertIsNone(self.last(context).arguments)
        self.assertFalse(self.last(context).arguments_recorded)
        self.assertNotIn("DomainSession", context.trace.serialize().decode())

    def test_json_arguments_wrong_top_level_recorded_as_supplied(self):
        for args in ([], None, "wrong"):
            context = self.context(domain_session=domain_session())
            self.assertEqual(dispatch_tool("lookup_order", args, context)["error"]["code"], "invalid_arguments")
            self.assertEqual(self.last(context).arguments, args)

    def test_deterministic_structural_trace_with_fresh_sessions_and_backend(self):
        contexts = [self.context(domain_session=domain_session(), sandbox_backend=FakeBackend()) for _ in range(2)]
        for context in contexts:
            for name in ("calculator_basic", "lookup_order", "create_return", "create_return",
                         "python_executor", "not-known"):
                args = arguments(name) if name in VALID else {}
                dispatch_tool(name, args, context)
        self.assertEqual(contexts[0].trace.to_dicts(), contexts[1].trace.to_dicts())
        self.assertEqual(contexts[0].trace.serialize(), contexts[1].trace.serialize())
        self.assertNotIn("timestamp", contexts[0].trace.serialize().decode())

    def test_context_or_private_fields_never_forwarded(self):
        session = domain_session()
        backend = FakeBackend()
        context = self.context(domain_session=session, sandbox_backend=backend)
        with patch.dict(dispatch._REGISTRY, {"calculator_basic": lambda args: {"received": args}}):
            result = dispatch_tool("calculator_basic", {"expression": "8-3"}, context)
        self.assertEqual(result, {"ok": True, "result": {"received": {"expression": "8-3"}}})
        result = dispatch_tool("python_executor", arguments("python_executor"), context)
        self.assertTrue(result["ok"])
        request = backend.requests[-1]
        self.assertNotIn(session, request.values())
        self.assertNotIn(backend, request.values())
        text = context.trace.serialize().decode()
        for private in ("tool_fixtures", "expected_answer", "expected_tool_sequence",
                        "evaluation_notes", "private-rule-sentinel", "consumed_cases"):
            self.assertNotIn(private, text)

    def test_loader_private_sentinels_stay_out_of_public_trace(self):
        catalog, fixtures = synthetic_domain()
        task = {"id": "private-task", "tool_catalog": catalog, "tool_fixtures": fixtures,
                "expected_answer": "sentinel-answer", "expected_tool_sequence": "sentinel-plan",
                "evaluation_notes": "sentinel-notes"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private-fixtures.json"
            path.write_text(json.dumps([task]), encoding="utf-8")
            session = DomainSession.from_benchmark("private-task", path)
        context = self.context(domain_session=session)
        dispatch_tool("order_workflow_query", arguments("order_workflow_query"), context)
        text = context.trace.serialize().decode()
        for value in ("sentinel-answer", "sentinel-plan", "sentinel-notes", str(path),
                      "private-task", "private-rule-sentinel"):
            self.assertNotIn(value, text)

    def test_negative_validation_result_is_still_execution_success(self):
        for name, args in [
            ("python_syntax_check", {"code": "{"}),
            ("json_validator", {"text": "{", "schema": {"type": "string"}}),
            ("text_format_validator", arguments("text_format_validator")),
        ]:
            with self.subTest(name=name):
                context = self.context()
                result = dispatch_tool(name, args, context)
                self.assertTrue(result["ok"])
                self.assertFalse(result["result"]["valid"])
                self.assertTrue(self.last(context).tool_execution_completed)
                self.assertFalse(self.last(context).infrastructure_failure)

    def test_trace_serialization_ignores_object_key_order(self):
        a, b = self.context(), self.context()
        dispatch_tool("table_transform", {"rows": [{"x": 1, "y": 2}], "pipeline": []}, a)
        dispatch_tool("table_transform", {"pipeline": [], "rows": [{"y": 2, "x": 1}]}, b)
        self.assertEqual(a.trace.serialize(), b.trace.serialize())

    def test_correct_official_session_supplied_by_controller(self):
        path = Path(__file__).resolve().parents[1] / "prompts/benchmark_v1.json"
        tasks = json.loads(path.read_text(encoding="utf-8"))
        for task in (t for t in tasks if t["id"] in ("T5", "T6")):
            session = DomainSession.from_benchmark(task["id"])
            context = self.context(domain_session=session)
            for case in task["tool_fixtures"]["cases"]:
                envelope = dispatch_tool(case["tool"], case["arguments"], context)
                self.assertEqual(envelope, {"ok": True, "result": case["result"]})
                self.assertFalse(self.last(context).infrastructure_failure)
            for marker in ("expected_answer", "expected_tool_sequence", "final_expected_answer",
                           "requires", "consumed_cases", "tool_fixtures", "evaluation_notes"):
                self.assertNotIn(marker, context.trace.serialize().decode())

    def test_catalog_hash_unchanged(self):
        original = serialize_public_catalog()
        dispatch_tool("calculator_basic", {"expression": "3*4"}, self.context())
        self.assertEqual(serialize_public_catalog(), original)
        self.assertEqual(public_catalog_sha256(),
                         "cfe666d04c4870461bd3c18bfd5b78a393d5778cd6fb2aac6399d6701dade92b")


class DispatcherDockerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = DockerSandboxBackend()
        if not cls.backend.is_available():
            raise unittest.SkipTest("V1 Docker integration unavailable; not a V2 requirement.")

    def test_real_executor_and_test_runner(self):
        context = DispatchContext(sandbox_backend=self.backend)
        executor = dispatch_tool("python_executor", {"code": "def main(data): return data + 2", "data": 3}, context)
        self.assertEqual(executor, {"ok": True, "result": {"value": 5}})
        runner = dispatch_tool("python_test_runner", {
            "code": "def f(x): return x + 1", "function": "f",
            "cases": [{"args": [8], "kwargs": {}, "expected": 9}]}, context)
        self.assertTrue(runner["ok"])
        self.assertTrue(runner["result"]["passed"])
        self.assertTrue(all(not e.infrastructure_failure for e in context.trace.entries))

    def test_real_timeout_cancelled_and_cleaned(self):
        context = DispatchContext(sandbox_backend=self.backend)
        result = dispatch_tool("python_executor", {"code": "def main(data):\n while True: pass",
                                                 "data": None}, context)
        self.assertEqual(result["error"]["code"], "tool_timeout")
        self.assertFalse(context.trace.entries[-1].infrastructure_failure)

    def test_real_backend_unavailable_is_controlled(self):
        backend = DockerSandboxBackend()
        backend.docker = None
        context = DispatchContext(sandbox_backend=backend)
        result = dispatch_tool("python_executor", arguments("python_executor"), context)
        self.assertEqual(result["error"]["code"], "execution_error")
        self.assertTrue(context.trace.entries[-1].infrastructure_failure)


if __name__ == "__main__":
    unittest.main()
