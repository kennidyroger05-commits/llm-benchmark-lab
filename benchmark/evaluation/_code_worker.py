"""Internal isolated worker. Receives code and evaluation-only cases on stdin."""
import ast
import builtins
import copy
import importlib
import io
import json
import resource
import sys

# Import permitted computational modules before disabling filesystem access.
ALLOWED = {'re', 'collections', 'itertools', 'functools', 'math', 'decimal',
           'statistics', 'bisect', 'heapq', 'operator', 'string', 'typing', 'json', 'copy'}
MODULES = {name: importlib.import_module(name) for name in sorted(ALLOWED)}


def same(a, b):
    """Python's bool/int equality is not sufficient for benchmark types."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    payload = json.load(sys.stdin)
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    tree = ast.parse(payload['code'])
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            return {'passed': False, 'failure_reasons': ['code_policy_violation']}
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            return {'passed': False, 'failure_reasons': ['code_policy_violation']}
    # Top level may contain imports, one function and a module docstring only.
    if any(not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)) and
           not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and
                isinstance(n.value.value, str)) for n in tree.body):
        return {'passed': False, 'failure_reasons': ['code_policy_violation']}

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in MODULES:
            raise PermissionError('import denied')
        return MODULES[name]

    def audit(event, args):
        if (event == 'open' or event.startswith(('socket.', 'subprocess.', 'ctypes.',
                                                'os.', 'shutil.', 'winreg.'))):
            raise PermissionError('external access denied')

    safe = dict(vars(builtins))
    for key in ['open', 'input', 'print', 'eval', 'exec', 'compile', 'breakpoint',
                'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr', 'help',
                'exit', 'quit']:
        safe.pop(key, None)
    safe['__import__'] = restricted_import
    namespace = {'__builtins__': safe}
    sys.addaudithook(audit)
    captured = io.StringIO()
    sys.stdout = captured
    try:
        exec(compile(tree, '<response>', 'exec'), namespace)
        function = namespace[payload['function']]
        for case in payload['cases']:
            argument = copy.deepcopy(case['input'])
            before = copy.deepcopy(argument)
            try:
                actual = function(argument)
            except ValueError:
                if case.get('raises') != 'ValueError':
                    return {'passed': False, 'failure_reasons': ['wrong_code_behavior']}
            else:
                if 'raises' in case or not same(actual, case['expected']):
                    return {'passed': False, 'failure_reasons': ['wrong_code_behavior']}
            if not same(argument, before) or captured.getvalue():
                return {'passed': False, 'failure_reasons': ['wrong_code_behavior']}
        return {'passed': True, 'failure_reasons': []}
    except PermissionError:
        return {'passed': False, 'failure_reasons': ['code_policy_violation']}
    except BaseException:
        return {'passed': False, 'failure_reasons': ['wrong_code_behavior']}
    finally:
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
