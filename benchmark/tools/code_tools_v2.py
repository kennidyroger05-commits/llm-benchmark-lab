"""V2 tool entrypoints: schema validation, no dynamic Python execution."""
import ast
from .restricted_python import RestrictedASTBackend
from .schemas import validate_tool_arguments, StructuralValidationError
from .math_tools import ToolExecutionError
from .sandbox import SandboxError


def _run(name, arguments, backend):
    try:
        validate_tool_arguments(name, arguments)
    except StructuralValidationError:
        raise ToolExecutionError('invalid_arguments', 'Invalid tool arguments.') from None
    if type(backend) is not RestrictedASTBackend:
        raise SandboxError('execution_error', 'backend_infrastructure')
    executor = name == 'python_executor'
    return backend.run_python(code=arguments['code'], mode='executor' if executor else 'test_runner',
        function='main' if executor else arguments['function'], data=arguments.get('data'),
        cases=arguments.get('cases'), timeout=2).result


def run_python_executor(arguments, backend=None):
    return _run('python_executor', arguments, backend)


def run_python_test_runner(arguments, backend=None):
    return _run('python_test_runner', arguments, backend)


def run_python_syntax_check(arguments):
    try:
        validate_tool_arguments('python_syntax_check', arguments)
    except StructuralValidationError:
        raise ToolExecutionError('invalid_arguments', 'Invalid tool arguments.') from None
    try:
        ast.parse(arguments['code'])
    except SyntaxError as error:
        return {'valid': False, 'errors': [{'code': 'syntax_error', 'line': error.lineno, 'column': error.offset}]}
    except (ValueError, RecursionError, MemoryError):
        return {'valid': False, 'errors': [{'code': 'invalid_python', 'line': None, 'column': None}]}
    return {'valid': True, 'errors': []}
