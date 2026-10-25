"""Resolve one tool call, with a public envelope and separate technical evidence.

No model loop, implicit domain session, default sandbox or task evaluation.
Every call here consumes a dispatch, including unknown tools/invalid arguments.
A future loop must block its ninth call before invoking this API.

tool_execution_completed means the implementation returned a valid result or
finished with an expected ToolExecutionError. Unknown names, non-JSON inputs and
missing dependencies do not execute it. Schema validation is delegated to the
existing implementations, preserving their resource checks and domain precedence.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType

from benchmark.tools import math_tools, data_tools, domain_tools, code_tools, validators
from benchmark.tools.catalog import get_public_catalog
from benchmark.tools.domain_tools import DomainSession
from benchmark.tools.math_tools import ToolExecutionError
from benchmark.tools.sandbox import SandboxBackend, SandboxError
from benchmark.tools._sandbox_worker import json_bytes, pack

from .tool_trace import ToolTrace, finite_json_bytes

# Immutable routing definitions, with no per-episode state. Derived only from
# explicit run_* APIs in the five implementation modules; no aliasing.
_REGISTRY = {
    name.removeprefix("run_"): function
    for module in (math_tools, data_tools, domain_tools, code_tools, validators)
    for name, function in vars(module).items()
    if name.startswith("run_") and callable(function)
}
if set(_REGISTRY) != {tool.name for tool in get_public_catalog()}:
    raise RuntimeError("Tool registry differs from the frozen public catalog.")
TOOL_REGISTRY = MappingProxyType(_REGISTRY)
_DOMAIN_NAMES = frozenset(name.removeprefix("run_") for name in vars(domain_tools)
                          if name.startswith("run_"))
_SANDBOX_NAMES = frozenset(("python_executor", "python_test_runner"))
_MESSAGES = MappingProxyType({
    "unknown_tool": "Unknown tool.",
    "invalid_arguments": "Invalid tool arguments.",
    "invalid_data": "Invalid tool data.",
    "invalid_expression": "Invalid arithmetic expression.",
    "infeasible": "No feasible solution.",
    "resource_limit": "Tool resource limit exceeded.",
    "policy_violation": "Operation blocked by policy.",
    "duplicate_call": "Operation already executed.",
    "precondition_failed": "Operation preconditions not satisfied.",
    "tool_timeout": "Tool execution timed out.",
    "execution_error": "Tool execution failed.",
})


@dataclass(frozen=True)
class DispatchContext:
    """Execution dependencies only; never forwarded wholesale to any tool."""
    domain_session: DomainSession | None = field(default=None, repr=False)
    sandbox_backend: SandboxBackend | None = field(default=None, repr=False)
    trace: ToolTrace = field(default_factory=ToolTrace, repr=False)


def serialize_envelope(envelope):
    """Serialize exact finite JSON as UTF-8, without float or string coercion of Decimal.

    The existing code tools support Decimal values. This dedicated serializer
    preserves those as JSON numeric tokens; stdlib json.dumps alone cannot encode
    Decimal. Other non-JSON objects, invalid shapes, NaN and Infinity are rejected.
    """
    if type(envelope) is not dict or type(envelope.get("ok")) is not bool:
        raise ValueError("Invalid public envelope.")
    if envelope["ok"]:
        if set(envelope) != {"ok", "result"}:
            raise ValueError("Invalid success envelope.")
    else:
        if set(envelope) != {"ok", "error"} or type(envelope["error"]) is not dict:
            raise ValueError("Invalid error envelope.")
        error = envelope["error"]
        if (set(error) != {"code", "message"} or type(error["code"]) is not str
                or type(error["message"]) is not str):
            raise ValueError("Invalid error fields.")
    # Typed wire validation preserves numeric types and rejects non-string keys, tuples
    # and other Python objects before encoding the public JSON document.
    return finite_json_bytes(envelope)


def _error(code):
    return {"ok": False, "error": {"code": code, "message": _MESSAGES[code]}}


def dispatch_tool(tool_name, arguments, context: DispatchContext, *, registry=None):
    """Return exactly one sanitized envelope; technical fields live in context.trace."""
    if not isinstance(context, DispatchContext) or not isinstance(context.trace, ToolTrace):
        raise TypeError("A DispatchContext with ToolTrace is required.")
    # Trusted revision-specific routing only; model arguments cannot select it.
    registry = TOOL_REGISTRY if registry is None else registry
    known = isinstance(tool_name, str) and tool_name in registry
    completed = False
    infrastructure_failure = False
    origin = None
    # Do not repr or deepcopy non-JSON objects, such as accidentally supplied
    # sessions/secrets containers. Invalid input is recorded as null, explicitly.
    try:
        pack(arguments)
        json_bytes(arguments)
        recorded_arguments = deepcopy(arguments)
        arguments_recorded = True
    except (ValueError, TypeError, RecursionError, UnicodeError, ArithmeticError):
        recorded_arguments = None
        arguments_recorded = False
    safe_name = None
    if isinstance(tool_name, str):
        try:
            json_bytes(tool_name)
            safe_name = tool_name
        except UnicodeError:
            pass

    if not known:
        envelope = _error("unknown_tool")
    elif not arguments_recorded:
        envelope = _error("invalid_arguments")
    else:
        try:
            function = registry[tool_name]
            # Each implementation gets only detached public arguments and its
            # own dependency. Pure tools cannot access the domain or backend.
            public_arguments = deepcopy(recorded_arguments)
            if tool_name in _DOMAIN_NAMES:
                if not isinstance(context.domain_session, DomainSession):
                    raise SandboxError("execution_error", "backend_infrastructure", "domain_session")
                result = function(public_arguments, context.domain_session)
            elif tool_name in _SANDBOX_NAMES:
                backend = context.sandbox_backend
                if not isinstance(backend, SandboxBackend) or not backend.is_available():
                    raise SandboxError("execution_error", "backend_infrastructure", "preflight")
                result = function(public_arguments, backend=backend)
            else:
                result = function(public_arguments)
            envelope = {"ok": True, "result": deepcopy(result)}
            # DomainSession has already checked the same outer-envelope
            # serialization/size before committing its own atomic state.
            if len(serialize_envelope(envelope)) > 8192:
                raise ValueError("Invalid tool output size.")
            completed = True
        except ToolExecutionError as error:
            recognized = type(error.code) is str and error.code in _MESSAGES
            code = error.code if recognized else "execution_error"
            if isinstance(error, SandboxError):
                origin = (error.origin if type(error.origin) is str and error.origin in {
                    "submitted_code", "sandbox_policy", "backend_infrastructure"
                } else "backend_infrastructure")
                infrastructure_failure = origin == "backend_infrastructure"
            else:
                infrastructure_failure = code == "execution_error"
                origin = "backend_infrastructure" if infrastructure_failure else None
            if not recognized:
                infrastructure_failure = True
                origin = "backend_infrastructure"
            completed = not infrastructure_failure
            # Never copy tool messages, tracebacks, fixture IDs or paths.
            envelope = _error(code)
        except Exception:
            infrastructure_failure = True
            origin = "backend_infrastructure"
            envelope = _error("execution_error")

    context.trace.append(
        tool=safe_name, arguments=recorded_arguments, envelope=envelope,
        dispatched=True, successful=envelope["ok"], failed=not envelope["ok"],
        error_code=None if envelope["ok"] else envelope["error"]["code"],
        dispatch_completed=True, tool_known=known, tool_execution_completed=completed,
        infrastructure_failure=infrastructure_failure, arguments_recorded=arguments_recorded,
        error_origin=origin,
    )
    return deepcopy(envelope)
