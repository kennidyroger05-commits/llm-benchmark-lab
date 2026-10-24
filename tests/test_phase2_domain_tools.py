"""Private fixture equivalence and synthetic domain-session tests, without model context."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from benchmark.tools import domain_tools as domain
from benchmark.tools import get_public_catalog, public_catalog_sha256, serialize_public_catalog
from benchmark.tools.math_tools import ToolExecutionError

ROOT = Path(__file__).resolve().parents[1]
DATASET = json.loads((ROOT / "prompts/benchmark_v1.json").read_text(encoding="utf-8"))
TASKS = {t["id"]: t for t in DATASET if t["id"] in ("T5", "T6")}
ORIGINAL = {
    "lookup_order", "check_return", "create_return", "cancel_order", "refund_payment",
    "get_stock", "reserve_stock", "charge", "release_stock", "log_failure",
    "issue_invoice", "send_receipt",
}
AUXILIARY = {"order_workflow_query", "inventory_query"}
PRIVATE_KEYS = {
    "expected_answer", "expected_tool_sequence", "final_expected_answer",
    "tool_fixtures", "evaluation_notes", "evaluation_notes_tools",
}


def call(session, name, args):
    return getattr(domain, "run_" + name)(deepcopy(args), session)


def session(task_id):
    return domain.DomainSession.from_benchmark(task_id)


def cases(task_id):
    return TASKS[task_id]["tool_fixtures"]["cases"]


def run_cases(instance, task_id, stop=None):
    results = []
    for case in cases(task_id)[:stop]:
        results.append(call(instance, case["tool"], case["arguments"]))
    return results


def synthetic():
    """Independent toy scenario with private sentinels; never a public prompt."""
    schemas = {t.name: t.parameters for t in get_public_catalog()}
    catalog = [{"name": name, "parameters": schemas[name]} for name in ORIGINAL]
    fixture = {
        "cases": [
            {"id": "read-order", "tool": "lookup_order", "arguments": {"order_id": "toy"},
             "requires": [], "result": {"status": "shipped", "lines": ["toy-line"]}},
            {"id": "eligibility", "tool": "check_return",
             "arguments": {"line_id": "toy-line", "reason": "toy-reason"},
             "requires": ["read-order"], "result": {"eligible": True}},
            {"id": "return", "tool": "create_return",
             "arguments": {"order_id": "toy", "line_ids": ["toy-line"]},
             "requires": ["read-order", "eligibility"], "result": {"ok": True}},
            {"id": "stock-a", "tool": "get_stock", "arguments": {"sku": "toy-a"},
             "requires": [], "result": {"available": 12}},
            {"id": "stock-b", "tool": "get_stock", "arguments": {"sku": "toy-b"},
             "requires": [], "result": {"available": 2}},
        ],
        "blocked_tools": [],
        "errors": {code: {"ok": False, "error": code} for code in [
            "unknown_tool", "invalid_arguments", "policy_violation",
            "duplicate_call", "precondition_failed"]},
        "dispatch_rules": ["private-rule-sentinel"],
    }
    return catalog, fixture


class DomainTests(unittest.TestCase):
    def error(self, instance, name, arguments, code):
        before = instance.snapshot()
        with self.assertRaises(ToolExecutionError) as caught:
            call(instance, name, arguments)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(instance.snapshot(), before)
        self.assertNotIn("C1", caught.exception.message)
        self.assertNotIn(str(ROOT), caught.exception.message)
        return caught.exception

    def test_exact_fourteen_explicit_apis(self):
        names = {n[4:] for n, f in vars(domain).items() if n.startswith("run_") and callable(f)}
        self.assertEqual(names, ORIGINAL | AUXILIARY)
        for name in names:
            self.assertEqual(list(inspect.signature(getattr(domain, "run_" + name)).parameters),
                             ["arguments", "session"])

    def test_loader_retains_no_task_answers(self):
        instance = session("T5")
        self.assertEqual(repr(instance), "DomainSession(private_snapshot)")
        for name in ["fixtures", "task", "expected_answer", "expected_tool_sequence"]:
            self.assertFalse(hasattr(instance, name))
        text = json.dumps(instance.snapshot())
        self.assertFalse(PRIVATE_KEYS & set(instance.snapshot()))
        for task in TASKS.values():
            self.assertNotIn(json.dumps(task["final_expected_answer"]), text)

    def test_loader_errors_are_sanitized(self):
        for args in [("unknown",), ("T5", ROOT / "no-such-private-file")]:
            with self.assertRaises(ToolExecutionError) as caught:
                domain.DomainSession.from_benchmark(*args)
            self.assertEqual(caught.exception.code, "execution_error")
            self.assertNotIn(str(ROOT), caught.exception.message)

    def test_catalog_and_source_unchanged(self):
        before = serialize_public_catalog()
        for task_id in TASKS:
            run_cases(session(task_id), task_id)
        self.assertEqual(serialize_public_catalog(), before)
        self.assertEqual(public_catalog_sha256(),
            "cfe666d04c4870461bd3c18bfd5b78a393d5778cd6fb2aac6399d6701dade92b")

    def test_fresh_sessions_initial_hash(self):
        for task_id in TASKS:
            a, b = session(task_id), session(task_id)
            self.assertEqual(a.snapshot(), b.snapshot())
            self.assertEqual(a.state_hash(), b.state_hash())
            run_cases(a, task_id)
            self.assertNotEqual(a.state_hash(), b.state_hash())
            self.assertEqual(b.snapshot(), session(task_id).snapshot())
            self.assertEqual(b.action_log, [])
            self.assertEqual(b.call_log, [])

    def test_reset_returns_initial_snapshot(self):
        for task_id in TASKS:
            instance = session(task_id)
            initial = instance.snapshot(), instance.state_hash()
            run_cases(instance, task_id)
            instance.reset()
            self.assertEqual((instance.snapshot(), instance.state_hash()), initial)
            self.assertEqual(instance.action_log, [])
            self.assertEqual(instance.call_log, [])
            self.assertEqual(run_cases(instance, task_id), run_cases(session(task_id), task_id))

    def test_input_fixture_copies(self):
        catalog, fixture = synthetic()
        instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
        catalog.clear()
        fixture["cases"][0]["result"]["lines"].append("private-sentinel")
        self.assertEqual(domain.run_lookup_order({"order_id": "toy"}, instance),
                         {"status": "shipped", "lines": ["toy-line"]})

    def test_controller_evidence_copies(self):
        instance = session("T6")
        run_cases(instance, "T6")
        state, actions, log = instance.snapshot(), instance.action_log, instance.call_log
        state["transactions"].clear()
        actions.clear()
        log[0]["result"].clear()
        self.assertTrue(instance.snapshot()["transactions"])
        self.assertTrue(instance.action_log)
        self.assertTrue(instance.call_log[0]["result"])

    def test_atomic_oversized_result(self):
        catalog, fixture = synthetic()
        fixture["cases"][2]["result"]["payload"] = "x" * 9000
        instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
        domain.run_lookup_order({"order_id": "toy"}, instance)
        domain.run_check_return({"line_id": "toy-line", "reason": "toy-reason"}, instance)
        self.error(instance, "create_return", {"order_id": "toy", "line_ids": ["toy-line"]}, "execution_error")
        self.assertEqual(instance.action_log, [])

    def test_atomic_unserializable_result(self):
        for value in [float("nan"), object()]:
            catalog, fixture = synthetic()
            fixture["cases"][0]["result"]["bad"] = value
            instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
            self.error(instance, "lookup_order", {"order_id": "toy"}, "execution_error")

    def test_result_validation_precedes_commit(self):
        instance = session("T6")
        run_cases(instance, "T6", 1)
        with patch.object(domain, "_public_result", side_effect=RuntimeError("private diagnostic")):
            error = self.error(instance, "reserve_stock", cases("T6")[1]["arguments"], "execution_error")
        self.assertNotIn("private diagnostic", error.message)
        # Failed validation did not consume the case: retry succeeds normally.
        self.assertEqual(call(instance, "reserve_stock", cases("T6")[1]["arguments"]),
                         cases("T6")[1]["result"])

    def test_logs_append_and_hash_transitions(self):
        instance = session("T6")
        for case in cases("T6"):
            before_hash = instance.state_hash()
            old_actions = instance.action_log
            call(instance, case["tool"], case["arguments"])
            self.assertEqual(instance.call_log[-1]["state_before"], before_hash)
            self.assertEqual(instance.call_log[-1]["state_after"], instance.state_hash())
            self.assertEqual(instance.action_log[:len(old_actions)], old_actions)
        self.assertEqual([r["ordinal"] for r in instance.call_log], list(range(1, 6)))

    def test_errors_logged_without_state_change(self):
        instance = session("T5")
        self.error(instance, "create_return", cases("T5")[-1]["arguments"], "precondition_failed")
        row = instance.call_log[-1]
        self.assertEqual(row["state_before"], row["state_after"])
        self.assertEqual(row["error"], "precondition_failed")
        self.assertEqual(instance.action_log, [])

    def test_read_queries_repeat_while_original_reads_follow_fixtures(self):
        instance = session("T5")
        args = {"order_id": cases("T5")[0]["arguments"]["order_id"], "fields": ["status"]}
        self.assertEqual(domain.run_order_workflow_query(args, instance),
                         domain.run_order_workflow_query(args, instance))
        first = cases("T5")[0]
        call(instance, first["tool"], first["arguments"])
        self.error(instance, first["tool"], first["arguments"], "duplicate_call")

    def test_lookup_and_return_results_exact(self):
        results = run_cases(session("T5"), "T5")
        self.assertEqual(results, [c["result"] for c in cases("T5")])
        self.assertEqual({r["eligible"] for r in results if "eligible" in r}, {True, False})

    def test_create_return_requires(self):
        self.error(session("T5"), "create_return", cases("T5")[-1]["arguments"], "precondition_failed")

    def test_check_return_order_requires(self):
        instance = session("T5")
        run_cases(instance, "T5", 1)
        self.error(instance, "check_return", cases("T5")[2]["arguments"], "precondition_failed")
        run_cases_tail = cases("T5")[1:]
        for case in run_cases_tail:
            self.assertEqual(call(instance, case["tool"], case["arguments"]), case["result"])

    def test_transaction_duplicate_and_exact_line_order(self):
        instance = session("T5")
        run_cases(instance, "T5")
        self.error(instance, "create_return", cases("T5")[-1]["arguments"], "duplicate_call")
        # Valid schema but no exact fixture match remains invalid_arguments.
        self.error(instance, "create_return", {"order_id": "not-the-order", "line_ids": ["other"]}, "invalid_arguments")

    def test_t5_blocked_operations(self):
        for name in ("cancel_order", "refund_payment"):
            self.error(session("T5"), name, cases("T5")[0]["arguments"], "policy_violation")

    def test_schema_before_policy_before_case(self):
        instance = session("T5")
        self.error(instance, "cancel_order", {"order_id": None}, "invalid_arguments")
        self.error(instance, "cancel_order", {"order_id": "not-in-fixtures"}, "policy_violation")

    def test_unknown_operation_in_episode_catalog(self):
        self.error(session("T5"), "get_stock", {"sku": "sample"}, "unknown_tool")

    def test_order_query_projection_only(self):
        instance = session("T5")
        order = cases("T5")[0]
        result = domain.run_order_workflow_query(
            {"order_id": order["arguments"]["order_id"], "fields": ["lines"]}, instance)
        self.assertEqual(result, {"order": {"lines": order["result"]["lines"]}})
        self.assertNotIn("eligible", json.dumps(result))
        self.assertEqual(instance.snapshot()["consumed_cases"], [])
        self.assertEqual(instance.action_log, [])

    def test_order_query_not_precondition(self):
        instance = session("T5")
        before = instance.state_hash()
        domain.run_order_workflow_query({**cases("T5")[0]["arguments"], "fields": ["status"]}, instance)
        self.assertEqual(before, instance.state_hash())
        self.error(instance, "check_return", cases("T5")[1]["arguments"], "precondition_failed")

    def test_query_rejects_unknown_order_and_fields(self):
        instance = session("T5")
        self.error(instance, "order_workflow_query", {"order_id": "unknown", "fields": ["status"]}, "invalid_arguments")
        self.error(instance, "order_workflow_query", {**cases("T5")[0]["arguments"], "fields": ["eligible"]}, "invalid_arguments")

    def test_t6_compensation_and_business_failure(self):
        instance = session("T6")
        results = run_cases(instance, "T6")
        self.assertEqual(results, [case["result"] for case in cases("T6")])
        failed = [c for c in cases("T6") if c["tool"] == "charge"][0]
        self.assertIs(failed["result"]["ok"], False)
        self.assertIn(failed["id"], instance.snapshot()["consumed_cases"])
        self.assertIsNone(instance.call_log[2]["error"])
        self.assertEqual([r["tool"] for r in instance.action_log],
                         ["reserve_stock", "charge", "release_stock", "log_failure"])
        transactions = instance.snapshot()["transactions"]
        self.assertEqual(transactions["release_stock"][0]["arguments"]["reservation_id"],
                         transactions["reserve_stock"][0]["result"]["reservation_id"])
        self.assertEqual(transactions["log_failure"][0]["arguments"]["reason"],
                         transactions["charge"][0]["result"]["error"])

    def test_reserve_without_get_stock(self):
        self.error(session("T6"), "reserve_stock", cases("T6")[1]["arguments"], "precondition_failed")

    def test_charge_not_repeated(self):
        instance = session("T6")
        run_cases(instance, "T6", 3)
        self.error(instance, "charge", cases("T6")[2]["arguments"], "duplicate_call")

    def test_release_then_log_requires(self):
        instance = session("T6")
        run_cases(instance, "T6", 3)
        self.error(instance, "log_failure", cases("T6")[-1]["arguments"], "precondition_failed")
        for case in cases("T6")[3:]:
            call(instance, case["tool"], case["arguments"])

    def test_t6_blocked_invoice_receipt(self):
        for name in ("issue_invoice", "send_receipt"):
            self.error(session("T6"), name, {"order_id": cases("T6")[1]["arguments"]["order_id"]}, "policy_violation")

    def test_inventory_query_read_only_not_precondition(self):
        instance = session("T6")
        sku = cases("T6")[0]["arguments"]["sku"]
        before = instance.state_hash()
        expected = {"rows": [{"sku": sku, "available": cases("T6")[0]["result"]["available"]}]}
        self.assertEqual(domain.run_inventory_query({"skus": [sku]}, instance), expected)
        self.assertEqual(domain.run_inventory_query({"skus": [sku]}, instance), expected)
        self.assertEqual(instance.state_hash(), before)
        self.assertEqual(instance.action_log, [])
        self.error(instance, "reserve_stock", cases("T6")[1]["arguments"], "precondition_failed")

    def test_inventory_query_order_and_all_or_nothing(self):
        catalog, fixture = synthetic()
        instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
        self.assertEqual(domain.run_inventory_query({"skus": ["toy-b", "toy-a"]}, instance),
                         {"rows": [{"sku": "toy-b", "available": 2}, {"sku": "toy-a", "available": 12}]})
        self.error(instance, "inventory_query", {"skus": ["toy-a", "unknown"]}, "invalid_arguments")
        self.assertEqual(instance.snapshot()["consumed_cases"], [])

    def test_args_unchanged_and_results_detached(self):
        for task_id in TASKS:
            instance = session(task_id)
            first = deepcopy(cases(task_id)[0]["arguments"])
            saved = deepcopy(first)
            result = getattr(domain, "run_" + cases(task_id)[0]["tool"])(first, instance)
            result.clear()
            self.assertEqual(first, saved)
            self.assertEqual(instance.call_log[0]["result"], cases(task_id)[0]["result"])

    def test_typed_matching_no_coercion(self):
        instance = session("T6")
        args = cases("T6")[1]["arguments"]
        for value in (True, "3", 3.0):
            self.error(instance, "reserve_stock", {**args, "quantity": value}, "invalid_arguments")

    def test_case_order_and_requires_driven_by_fixture(self):
        catalog, fixture = synthetic()
        fixture["cases"][2]["requires"] = ["read-order"]  # No hardcoded eligibility requirement.
        instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
        domain.run_lookup_order({"order_id": "toy"}, instance)
        self.assertEqual(domain.run_create_return({"order_id": "toy", "line_ids": ["toy-line"]}, instance), {"ok": True})

    def test_fixture_blocking_driven_by_snapshot(self):
        catalog, fixture = synthetic()
        fixture["blocked_tools"] = ["lookup_order"]
        instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
        self.error(instance, "lookup_order", {"order_id": "toy"}, "policy_violation")

    def test_generic_allowed_actions_with_synthetic_cases(self):
        catalog, fixture = synthetic()
        # Blocked in the official scenarios, still real implementations when fixture permits.
        for i, name in enumerate(("cancel_order", "refund_payment", "issue_invoice", "send_receipt")):
            fixture["cases"].append({"id": f"extra-{i}", "tool": name,
                "arguments": {"order_id": "toy"}, "requires": [], "result": {"ok": True}})
        instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
        for name in ("cancel_order", "refund_payment", "issue_invoice", "send_receipt"):
            self.assertEqual(call(instance, name, {"order_id": "toy"}), {"ok": True})
        self.assertEqual(len(instance.action_log), 4)

    def test_ambiguous_fixture_does_not_execute(self):
        catalog, fixture = synthetic()
        duplicate = deepcopy(fixture["cases"][0])
        duplicate["id"] = "different-id"
        fixture["cases"].append(duplicate)
        instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
        self.error(instance, "lookup_order", {"order_id": "toy"}, "invalid_arguments")

    def test_argument_key_order_ignored_but_array_order_preserved(self):
        catalog, fixture = synthetic()
        fixture["cases"][2]["requires"] = []
        fixture["cases"][2]["arguments"]["line_ids"] = ["first", "second"]
        instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
        self.error(instance, "create_return",
                   {"line_ids": ["second", "first"], "order_id": "toy"}, "invalid_arguments")
        self.assertEqual(domain.run_create_return(
            {"line_ids": ["first", "second"], "order_id": "toy"}, instance), {"ok": True})

    def test_loader_discards_unrelated_private_record_fields(self):
        catalog, fixture = synthetic()
        record = {"id": "synthetic", "tool_catalog": catalog, "tool_fixtures": fixture}
        for key in PRIVATE_KEYS - {"tool_fixtures"}:
            record[key] = "private-answer-sentinel-" + key
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.json"
            path.write_text(json.dumps([record]), encoding="utf-8")
            instance = domain.DomainSession.from_benchmark("synthetic", path)
        output = domain.run_order_workflow_query({"order_id": "toy", "fields": ["status"]}, instance)
        self.assertNotIn("private-answer-sentinel", repr(instance))
        self.assertNotIn("private-answer-sentinel", json.dumps(output))
        self.assertNotIn("private-answer-sentinel", json.dumps(instance.snapshot()))
        self.assertNotIn("private-answer-sentinel", json.dumps(instance.call_log))

    def test_failed_query_does_not_partially_consume_or_log_action(self):
        catalog, fixture = synthetic()
        instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
        before = instance.state_hash()
        self.error(instance, "inventory_query", {"skus": ["toy-a", "missing"]}, "invalid_arguments")
        self.assertEqual(instance.state_hash(), before)
        self.assertEqual(instance.action_log, [])
        self.assertIsNone(instance.call_log[-1]["result"])

    def test_deterministic_results_logs_and_hashes(self):
        for task_id in TASKS:
            a, b = session(task_id), session(task_id)
            self.assertEqual(run_cases(a, task_id), run_cases(b, task_id))
            self.assertEqual(a.snapshot(), b.snapshot())
            self.assertEqual(a.call_log, b.call_log)
            self.assertEqual(a.action_log, b.action_log)
            self.assertEqual(a.state_hash(), b.state_hash())

    def test_private_sentinels_not_exposed_in_queries(self):
        catalog, fixture = synthetic()
        fixture["cases"][1]["result"]["private-sentinel"] = "unrequested-eligibility"
        instance = domain.DomainSession(tool_catalog=catalog, tool_fixtures=fixture)
        result = domain.run_order_workflow_query({"order_id": "toy", "fields": ["status"]}, instance)
        text = json.dumps(result)
        self.assertNotIn("private-sentinel", text)
        self.assertNotIn("unrequested-eligibility", text)
        self.assertNotIn("private-rule-sentinel", text)
        self.assertEqual(set(result["order"]), {"status"})


if __name__ == "__main__":
    unittest.main()
