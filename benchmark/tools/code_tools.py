"""Syntax parsing locally; submitted code only via a mandatory safe backend."""
import ast
import time

from .math_tools import ToolExecutionError
from .schemas import StructuralValidationError, validate_tool_arguments
from .sandbox import DockerSandboxBackend, SandboxError
from ._sandbox_worker import PolicyError, validate_code


def _arguments(name, arguments):
    try:
        validate_tool_arguments(name, arguments)
    except StructuralValidationError:
        raise ToolExecutionError("invalid_arguments", "Invalid code tool arguments.") from None


def run_python_syntax_check(arguments):
    _arguments("python_syntax_check", arguments)
    try:
        tree = ast.parse(arguments["code"])
        compile(tree, "<submitted>", "exec")
    except SyntaxError as error:
        return {"valid": False, "errors": [{"code": "syntax_error",
                "line": error.lineno, "column": error.offset}]}
    except (ValueError, RecursionError):
        return {"valid": False, "errors": [{"code": "invalid_python", "line": None, "column": None}]}
    return {"valid": True, "errors": []}


def _execute(arguments, mode, backend, started):
    function = "main" if mode == "executor" else arguments["function"]
    try:
        validate_code(arguments["code"], function, mode)
    except PolicyError:
        raise SandboxError("invalid_data", "sandbox_policy") from None
    # Even dependency-injected backends must implement the explicit interface.
    from .sandbox import SandboxBackend
    backend = backend or DockerSandboxBackend()
    if not isinstance(backend, SandboxBackend):
        raise SandboxError("execution_error", "backend_infrastructure")
    remaining = 2 - (time.monotonic() - started)
    if remaining <= 0:
        raise SandboxError("tool_timeout", "submitted_code")
    return backend.run_python(
        code=arguments["code"], function=function, mode=mode,
        data=arguments.get("data"), cases=arguments.get("cases"), timeout=remaining,
    ).result


def run_python_executor(arguments, backend=None):
    started = time.monotonic()
    _arguments("python_executor", arguments)
    return _execute(arguments, "executor", backend, started)


def run_python_test_runner(arguments, backend=None):
    started = time.monotonic()
    _arguments("python_test_runner", arguments)
    return _execute(arguments, "test_runner", backend, started)
