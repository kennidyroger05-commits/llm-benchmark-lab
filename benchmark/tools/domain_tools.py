"""Local domain tools over private, isolated deterministic fixture snapshots.

Only result payloads leave these functions; an eventual dispatcher owns envelopes.
Original operations, including original read operations, follow fixture duplicate
rules. Auxiliary read-only queries are repeatable and never consume cases.
Controller snapshots/logs below are private execution evidence, not model context.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import RLock

from .math_tools import ToolExecutionError
from .schemas import StructuralValidationError, validate_structure, validate_tool_arguments

_BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "prompts" / "benchmark_v1.json"
_ORIGINAL = frozenset((
    "lookup_order", "check_return", "create_return", "cancel_order", "refund_payment",
    "get_stock", "reserve_stock", "charge", "release_stock", "log_failure",
    "issue_invoice", "send_receipt",
))
_READS = frozenset(("lookup_order", "check_return", "get_stock"))
_MESSAGES = {
    "unknown_tool": "Domain operation unavailable.",
    "invalid_arguments": "Invalid domain arguments.",
    "policy_violation": "Operation blocked by policy.",
    "duplicate_call": "Operation already executed.",
    "precondition_failed": "Operation preconditions not satisfied.",
    "execution_error": "Domain execution failed.",
}
_MAX_RESULT_BYTES = 8192


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _typed_equal(a, b):
    """Never confuse bool/int; object key order alone is irrelevant."""
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(_typed_equal(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(_typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def _public_result(result):
    """Validate a copied public result before committing any consumed case/action."""
    try:
        validate_structure(result, {"$ref": "#/$defs/JSON"})
        # Include the future envelope's overhead in the public return budget.
        if len(_bytes({"ok": True, "result": result})) > _MAX_RESULT_BYTES:
            raise ValueError("oversized result")
        _bytes(result)
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise ToolExecutionError("execution_error", _MESSAGES["execution_error"]) from None
    return result


class DomainSession:
    """Controller-owned snapshot, with no public fixture/task-record accessor.

    transaction_state is an execution ledger grouped by action name. Business
    failure results are executed events too, not dispatch errors. It records
    reservation/charge/release/failure evidence without inventing service rules.
    reset() starts a fresh episode with the same private source snapshot.
    """
    def __init__(self, *, tool_catalog, tool_fixtures):
        self.__lock = RLock()
        try:
            catalog = deepcopy(tool_catalog)
            fixtures = deepcopy(tool_fixtures)
            names = [tool["name"] for tool in catalog]
            ids = [case["id"] for case in fixtures["cases"]]
            if (len(names) != len(set(names)) or not set(names) <= _ORIGINAL or
                    len(ids) != len(set(ids)) or any(type(i) is not str for i in ids)):
                raise ValueError("invalid snapshot")
            for case in fixtures["cases"]:
                if case["tool"] not in names or not set(case["requires"]) <= set(ids):
                    raise ValueError("invalid snapshot")
            self.__catalog = {tool["name"]: tool["parameters"] for tool in catalog}
            self.__fixtures = fixtures
        except (KeyError, TypeError, ValueError):
            raise ToolExecutionError("execution_error", _MESSAGES["execution_error"]) from None
        self.reset()

    @classmethod
    def from_benchmark(cls, task_id, benchmark_path=_BENCHMARK_PATH):
        """Trusted controller loader. No expected answer, sequence or notes are stored."""
        try:
            with Path(benchmark_path).open(encoding="utf-8") as stream:
                matches = [task for task in json.load(stream) if task["id"] == task_id]
            if len(matches) != 1:
                raise ValueError("task unavailable")
            task = matches[0]
            return cls(tool_catalog=task["tool_catalog"], tool_fixtures=task["tool_fixtures"])
        except ToolExecutionError:
            raise
        except (OSError, ValueError, KeyError, TypeError):
            raise ToolExecutionError("execution_error", _MESSAGES["execution_error"]) from None

    def __repr__(self):
        return "DomainSession(private_snapshot)"

    def reset(self):
        """Discard episode state and logs; do not modify the source snapshot."""
        with self.__lock:
            self.__consumed = set()
            self.__transactions = {}
            self.__actions = []
            self.__calls = []

    def _snapshot(self):
        return {"consumed_cases": sorted(self.__consumed),
                "transactions": deepcopy(self.__transactions)}

    def snapshot(self):
        """Detached controller evidence; never contains unconsumed fixture results."""
        with self.__lock:
            return self._snapshot()

    def state_hash(self):
        with self.__lock:
            return hashlib.sha256(_bytes(self._snapshot())).hexdigest()

    @property
    def action_log(self):
        """Detached committed action log, append-only within one episode."""
        with self.__lock:
            return deepcopy(self.__actions)

    @property
    def call_log(self):
        """Detached audit of all calls, including queries and errors."""
        with self.__lock:
            return deepcopy(self.__calls)

    def _fixture_error(self, code):
        # Preserve error codes from the dataset; never expose internal case IDs
        # or return the whole private errors map.
        try:
            public_code = self.__fixtures["errors"][code]["error"]
            if public_code != code:
                raise ValueError("invalid fixture error")
        except (KeyError, TypeError, ValueError):
            raise ToolExecutionError("execution_error", _MESSAGES["execution_error"]) from None
        raise ToolExecutionError(public_code, _MESSAGES[code])

    def _record_call(self, tool, arguments, result, error, before, after):
        self.__calls.append({
            "ordinal": len(self.__calls) + 1, "tool": tool,
            "arguments": deepcopy(arguments), "result": deepcopy(result),
            "error": error, "state_before": before, "state_after": after,
        })

    def _execute_fixture(self, tool, arguments):
        """Domain-only matching, implementing frozen dispatch precedence; not a dispatcher."""
        with self.__lock:
            before = self.state_hash()
            try:
                # These branches deliberately follow the dataset dispatch_rules.
                if tool not in self.__catalog:
                    self._fixture_error("unknown_tool")
                try:
                    validate_structure(arguments, self.__catalog[tool])
                except StructuralValidationError:
                    self._fixture_error("invalid_arguments")
                if tool in self.__fixtures["blocked_tools"]:
                    self._fixture_error("policy_violation")
                matches = [case for case in self.__fixtures["cases"]
                           if case["tool"] == tool and _typed_equal(case["arguments"], arguments)]
                if len(matches) != 1:
                    self._fixture_error("invalid_arguments")
                case = matches[0]
                if case["id"] in self.__consumed:
                    self._fixture_error("duplicate_call")
                if not set(case["requires"]) <= self.__consumed:
                    self._fixture_error("precondition_failed")
                result = _public_result(deepcopy(case["result"]))
                # Prepare everything that can fail before committing session state.
                consumed = self.__consumed | {case["id"]}
                transactions = deepcopy(self.__transactions)
                event = {"tool": tool, "arguments": deepcopy(arguments),
                         "result": deepcopy(result)}
                if tool not in _READS:
                    transactions.setdefault(tool, []).append(event)
                after = hashlib.sha256(_bytes({
                    "consumed_cases": sorted(consumed), "transactions": transactions,
                })).hexdigest()
                action = {**event, "state_before": before, "state_after": after}
                # Returned errors cannot leave either cases or business events committed.
                calls = deepcopy(self.__calls)
                calls.append({"ordinal": len(calls) + 1, **deepcopy(event),
                              "error": None, "state_before": before, "state_after": after})
                actions = deepcopy(self.__actions)
                if tool not in _READS:
                    actions.append(action)
                self.__consumed, self.__transactions = consumed, transactions
                self.__calls, self.__actions = calls, actions
                return result
            except ToolExecutionError as error:
                self._record_call(tool, arguments, None, error.code, before, before)
                raise
            except Exception:
                self._record_call(tool, arguments, None, "execution_error", before, before)
                raise ToolExecutionError("execution_error", _MESSAGES["execution_error"]) from None

    def _query(self, tool, arguments):
        with self.__lock:
            before = self.state_hash()
            try:
                try:
                    validate_tool_arguments(tool, arguments)
                except StructuralValidationError:
                    raise ToolExecutionError("invalid_arguments", _MESSAGES["invalid_arguments"]) from None
                if tool == "order_workflow_query":
                    matches = [case for case in self.__fixtures["cases"]
                               if case["tool"] == "lookup_order" and
                               case["arguments"]["order_id"] == arguments["order_id"]]
                    if len(matches) != 1:
                        raise ToolExecutionError("invalid_arguments", _MESSAGES["invalid_arguments"])
                    read = matches[0]["result"]
                    result = {"order": {field: deepcopy(read[field]) for field in arguments["fields"]}}
                else:
                    rows = []
                    for sku in arguments["skus"]:
                        matches = [case for case in self.__fixtures["cases"]
                                   if case["tool"] == "get_stock" and case["arguments"]["sku"] == sku]
                        if len(matches) != 1:
                            raise ToolExecutionError("invalid_arguments", _MESSAGES["invalid_arguments"])
                        rows.append({"sku": sku, "available": deepcopy(matches[0]["result"]["available"])})
                    result = {"rows": rows}
                result = _public_result(result)
                self._record_call(tool, arguments, result, None, before, before)
                return result
            except ToolExecutionError as error:
                self._record_call(tool, arguments, None, error.code, before, before)
                raise
            except Exception:
                self._record_call(tool, arguments, None, "execution_error", before, before)
                raise ToolExecutionError("execution_error", _MESSAGES["execution_error"]) from None


# Explicit APIs: the public payload contains arguments only, not the private task.
def run_lookup_order(arguments, session):
    return session._execute_fixture("lookup_order", arguments)

def run_check_return(arguments, session):
    return session._execute_fixture("check_return", arguments)

def run_create_return(arguments, session):
    return session._execute_fixture("create_return", arguments)

def run_cancel_order(arguments, session):
    return session._execute_fixture("cancel_order", arguments)

def run_refund_payment(arguments, session):
    return session._execute_fixture("refund_payment", arguments)

def run_order_workflow_query(arguments, session):
    return session._query("order_workflow_query", arguments)

def run_get_stock(arguments, session):
    return session._execute_fixture("get_stock", arguments)

def run_reserve_stock(arguments, session):
    return session._execute_fixture("reserve_stock", arguments)

def run_charge(arguments, session):
    return session._execute_fixture("charge", arguments)

def run_release_stock(arguments, session):
    return session._execute_fixture("release_stock", arguments)

def run_log_failure(arguments, session):
    return session._execute_fixture("log_failure", arguments)

def run_issue_invoice(arguments, session):
    return session._execute_fixture("issue_invoice", arguments)

def run_send_receipt(arguments, session):
    return session._execute_fixture("send_receipt", arguments)

def run_inventory_query(arguments, session):
    return session._query("inventory_query", arguments)
