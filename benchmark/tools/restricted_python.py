"""Protocol V2: bounded AST interpretation over data, never Python bytecode.

No model-provided callable/object/module reaches this evaluator. Attribute syntax
is accepted only as a call to the explicit data-method table. Every statement,
loop, call, arithmetic operation and allocation is budgeted before expensive
work; wall time is supplementary. This is a language subset, not CPython in a
sandbox. Unsupported programs are refused, never retried in another backend.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, localcontext
import operator
import time

from .sandbox import SandboxBackend, SandboxError, SandboxResult
from ._sandbox_worker import json_bytes, equal

BUILTINS = frozenset('len range min max sum sorted enumerate zip abs all any int str bool list dict tuple reversed'.split())
METHODS = frozenset('append extend pop get keys values items strip split join startswith endswith isdigit'.split())
EXCEPTIONS = frozenset('ValueError TypeError KeyError IndexError ZeroDivisionError'.split())
DENIED = frozenset('open eval exec compile __import__ globals locals vars getattr setattr delattr input help breakpoint os sys pathlib subprocess socket ctypes pickle threading multiprocessing importlib inspect random time'.split())


@dataclass(frozen=True)
class Limits:
    source_chars: int = 32768
    ast_nodes: int = 4096
    depth: int = 32
    statements: int = 20000
    iterations: int = 10000
    range_size: int = 4096
    calls: int = 256
    call_depth: int = 16
    container: int = 4096
    string: int = 8192
    result_bytes: int = 8192
    arithmetic: int = 10000
    integer_bits: int = 256
    allocated: int = 250000
    data_nodes: int = 20000


class _Return(Exception):
    def __init__(self, value): self.value = value
class _Break(Exception): pass
class _Continue(Exception): pass
class _Submitted(Exception):
    def __init__(self, name): self.name = name


def _policy():
    raise SandboxError('policy_violation', 'sandbox_policy')


def _resource():
    raise SandboxError('resource_limit', 'submitted_code')


_ALLOWED = tuple(getattr(ast, name) for name in '''Module FunctionDef arguments arg Return Assign AugAssign
If For Expr Break Continue Pass Raise Name Load Store Constant List Tuple Dict Subscript Slice
BinOp UnaryOp BoolOp Compare Call Attribute keyword Add Sub Mult Div FloorDiv Mod Pow UAdd USub Not
And Or Eq NotEq Lt LtE Gt GtE In NotIn Is IsNot'''.split())


class Interpreter:
    def __init__(self, code, *, limits=Limits(), timeout=2):
        self.limits = limits
        self.deadline = time.monotonic() + min(2, timeout)
        self.used = {key: 0 for key in ('statements', 'iterations', 'calls', 'arithmetic', 'allocated')}
        self.level = 0
        self.iterators = []
        if type(code) is not str or len(code) > limits.source_chars:
            _resource()
        try:
            self.tree = ast.parse(code)
        except (SyntaxError, ValueError):
            raise SandboxError('invalid_data', 'sandbox_policy') from None
        except (RecursionError, MemoryError):
            _resource()
        self.functions = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef)}
        if len(self.functions) != len(self.tree.body) or not self.functions or len(self.functions) > 8:
            _policy()
        pending = [(self.tree, 0, None, 0)]
        count = 0
        while pending:
            node, depth, parent, loops = pending.pop()
            count += 1
            if count > limits.ast_nodes or depth > limits.depth:
                _resource()
            if type(node) not in _ALLOWED:
                _policy()
            if isinstance(node, (ast.Name, ast.arg, ast.FunctionDef)):
                name = node.id if isinstance(node, ast.Name) else node.arg if isinstance(node, ast.arg) else node.name
                if '__' in name or name in DENIED:
                    _policy()
                if not isinstance(node, ast.Name) or isinstance(node.ctx, ast.Store):
                    if name in BUILTINS | EXCEPTIONS:
                        _policy()
            if isinstance(node, (ast.Break, ast.Continue)) and not loops: _policy()
            if (isinstance(node, ast.arg) or isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)):
                key = node.arg if isinstance(node, ast.arg) else node.id
                if key in self.functions: _policy()
            if isinstance(node, ast.FunctionDef):
                a = node.args
                if (node not in self.tree.body or node.decorator_list or node.returns or node.type_comment
                        or getattr(node, 'type_params', ()) or a.vararg or a.kwarg or a.kwonlyargs
                        or a.defaults or any(arg.annotation for arg in a.posonlyargs + a.args)):
                    _policy()
            if isinstance(node, ast.Constant):
                if type(node.value) not in (str, int, bool, type(None)):
                    _policy()
                self.guard(node.value)
            if isinstance(node, ast.Attribute):
                if not isinstance(parent, ast.Call) or parent.func is not node or node.attr not in METHODS:
                    _policy()
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in BUILTINS | EXCEPTIONS | self.functions.keys():
                        _policy()
                elif not isinstance(node.func, ast.Attribute):
                    _policy()
                if len({k.arg for k in node.keywords}) != len(node.keywords): _policy()
                if any(k.arg is None for k in node.keywords):
                    _policy()
            if isinstance(node, ast.Raise):
                exc = node.exc
                if (node.cause or not isinstance(exc, ast.Call) or not isinstance(exc.func, ast.Name)
                        or exc.func.id not in EXCEPTIONS or exc.keywords or len(exc.args) > 1
                        or any(not isinstance(a, ast.Constant) or type(a.value) is not str for a in exc.args)):
                    _policy()
            if isinstance(node, ast.Dict) and any(key is None for key in node.keys):
                _policy()
            pending.extend((child, depth + 1, node, loops + int(isinstance(node, ast.For) and child in node.body)) for child in ast.iter_child_nodes(node))

    def tick(self, key, amount=1):
        self.used[key] += amount
        if self.used[key] > getattr(self.limits, key): _resource()
        if time.monotonic() > self.deadline:
            raise SandboxError('tool_timeout', 'submitted_code')

    def guard(self, value):
        """Reject cycles/custom objects and bound the complete reachable data tree."""
        nodes = 0
        def visit(v, depth, active):
            nonlocal nodes
            nodes += 1
            if nodes > self.limits.data_nodes or depth > self.limits.depth: _resource()
            kind = type(v)
            if kind is int:
                if v.bit_length() > self.limits.integer_bits: _resource()
            elif kind is str:
                nodes += len(v)
                if nodes > self.limits.data_nodes: _resource()
                if len(v) > self.limits.string: _resource()
            elif kind is Decimal:
                if not v.is_finite() or len(v.as_tuple().digits) > 128 or abs(v.adjusted()) > 256: _resource()
            elif kind in (list, tuple, dict):
                if len(v) > self.limits.container or id(v) in active: _resource()
                active = active | {id(v)}
                children = list(v.items()) if kind is dict else v
                for child in children: visit(child, depth + 1, active)
            elif kind not in (bool, type(None)):
                if not any(v is iterator for iterator in self.iterators): _policy()
        visit(value, 0, set())
        return value

    def iterable(self, value):
        if type(value) not in (list, tuple, str, dict) and not any(value is v for v in self.iterators):
            raise _Submitted('TypeError')
        return value

    def sequence(self, value):
        values = []
        for child in self.iterable(value):
            self.tick('iterations')
            if len(values) >= self.limits.container: _resource()
            self.tick('allocated'); values.append(child)
        return values

    def iterator(self, value):
        self.iterators.append(value)
        return value

    def arithmetic(self, op, a, b):
        self.tick('arithmetic')
        if type(a) in (list, tuple, str) or type(b) in (list, tuple, str):
            if isinstance(op, ast.Add) and type(a) is type(b) and type(a) in (list, tuple, str):
                size = len(a) + len(b)
            elif isinstance(op, ast.Mult) and type(a) in (list, tuple, str) and type(b) in (int, bool):
                size = len(a) * max(0, b)
            elif isinstance(op, ast.Mult) and type(b) in (list, tuple, str) and type(a) in (int, bool):
                a, b = b, a
                size = len(a) * max(0, b)
            else:
                raise _Submitted('TypeError')
            if size > (self.limits.string if type(a) is str else self.limits.container): _resource()
            self.tick('allocated', size)
        else:
            if type(a) not in (int, bool, Decimal) or type(b) not in (int, bool, Decimal): raise _Submitted('TypeError')
            if isinstance(op, ast.Pow):
                if type(a) is not int or type(b) is not int or not 0 <= b <= 64: _resource()
                if a.bit_length() * b > self.limits.integer_bits: _resource()
        functions = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
                     ast.Div: lambda x, y: Decimal(x) / Decimal(y)}
        with localcontext() as ctx:
            ctx.prec, ctx.Emax, ctx.Emin = 64, 256, -256
            result = functions[type(op)](a, b)
        return self.guard(result)

    def call(self, name, args, kwargs):
        self.tick('calls')
        if name in self.functions:
            node = self.functions[name]
            positional = [a.arg for a in node.args.posonlyargs]
            names = positional + [a.arg for a in node.args.args]
            if (len(args) > len(names) or kwargs.keys() - set(names) or set(positional) & kwargs.keys()
                    or set(names[:len(args)]) & kwargs.keys()): raise _Submitted('TypeError')
            scope = dict(zip(names, args)); scope.update(kwargs)
            if scope.keys() != set(names): raise _Submitted('TypeError')
            self.level += 1
            if self.level > self.limits.call_depth: _resource()
            try:
                self.block(node.body, scope)
            except _Return as returned:
                return returned.value
            finally:
                self.level -= 1
            return None
        if name in EXCEPTIONS: _policy()  # exception constructors only belong to Raise
        if kwargs and (name != 'sorted' or set(kwargs) != {'reverse'} or type(kwargs['reverse']) is not bool): _policy()
        if name == 'len': result = len(args[0]) if len(args) == 1 else self.bad_type()
        elif name == 'range':
            if not 1 <= len(args) <= 3 or any(type(a) not in (int, bool) for a in args): raise _Submitted('TypeError')
            result = range(*args)
            if len(result) > self.limits.range_size: _resource()
            result = self.iterator(result)
        elif name == 'abs': result = abs(args[0]) if len(args) == 1 else self.bad_type()
        elif name in {'min', 'max'}:
            if not args: raise _Submitted('TypeError')
            values = self.sequence(args[0]) if len(args) == 1 else args
            result = (min if name == 'min' else max)(values)
        elif name == 'enumerate':
            if not 1 <= len(args) <= 2: raise _Submitted('TypeError')
            result = self.iterator(enumerate(self.iterable(args[0]), args[1] if len(args) == 2 else 0))
        elif name == 'reversed':
            if len(args) != 1: raise _Submitted('TypeError')
            value = self.iterable(args[0])
            result = self.iterator(reversed(value))
        elif name in {'list', 'tuple'} and not args:
            result = [] if name == 'list' else ()
        elif name in {'sum', 'all', 'any', 'sorted', 'reversed', 'list', 'tuple', 'enumerate'}:
            if not args or len(args) > (2 if name in {'sum', 'enumerate'} else 1): raise _Submitted('TypeError')
            values = self.sequence(args[0])
            if name == 'sum':
                result = args[1] if len(args) == 2 else 0
                for item in values: result = self.arithmetic(ast.Add(), result, item)
            elif name == 'all': result = all(values)
            elif name == 'any': result = any(values)
            elif name == 'sorted': result = sorted(values, **kwargs)
            elif name == 'reversed': result = list(reversed(values))
            elif name == 'tuple': result = tuple(values)
            elif name == 'enumerate': result = list(enumerate(values, args[1] if len(args) == 2 else 0))
            else: result = values
        elif name == 'zip':
            result = self.iterator(zip(*(self.iterable(a) for a in args)))
        elif name in {'int', 'str', 'bool'} and not args:
            result = 0 if name == 'int' else '' if name == 'str' else False
        elif name in {'int', 'str', 'bool'}:
            if len(args) != 1: raise _Submitted('TypeError')
            value = args[0]
            if name == 'int':
                if type(value) not in (str, int, bool, Decimal): raise _Submitted('TypeError')
                if type(value) is str and len(value) > 78: _resource()
                result = int(value)
            elif name == 'bool': result = bool(value)
            else:
                if type(value) not in (str, int, bool, type(None)): _policy()
                result = str(value); self.tick('allocated', len(result))
        elif name == 'dict':
            if len(args) > 1: raise _Submitted('TypeError')
            values = args[0] if args else {}
            if type(values) is not dict: values = self.sequence(values)
            self.tick('allocated', len(values)); result = dict(values)
        else: _policy()
        return self.guard(result)

    @staticmethod
    def bad_type(): raise _Submitted('TypeError')

    def method(self, obj, name, args, kwargs):
        self.tick('calls')
        if kwargs: _policy()
        if type(obj) is list and name in {'append', 'extend', 'pop'}:
            if name == 'append' and len(args) == 1:
                if len(obj) >= self.limits.container: _resource()
                self.tick('allocated'); obj.append(args[0]); result = None
            elif name == 'extend' and len(args) == 1:
                values = self.sequence(args[0])
                if len(obj) + len(values) > self.limits.container: _resource()
                obj.extend(values); result = None
            elif name == 'pop' and len(args) <= 1: result = obj.pop(*args)
            else: raise _Submitted('TypeError')
            self.guard(obj)
        elif type(obj) is dict and name in {'get', 'keys', 'values', 'items'}:
            if name == 'get' and 1 <= len(args) <= 2: result = obj.get(*args)
            elif not args and name != 'get':
                self.tick('allocated', len(obj) * 3)
                result = list(obj.keys()) if name == 'keys' else list(obj.values()) if name == 'values' else list(obj.items())
            else: raise _Submitted('TypeError')
        elif type(obj) is str:
            if name == 'strip' and len(args) <= 1: result = obj.strip(*args)
            elif name == 'split' and len(args) <= 1:
                result = obj.split(args[0] if args else None, self.limits.container)
            elif name == 'join' and len(args) == 1:
                values = self.sequence(args[0])
                if any(type(v) is not str for v in values): raise _Submitted('TypeError')
                size = sum(map(len, values)) + len(obj) * max(0, len(values)-1)
                if size > self.limits.string: _resource()
                self.tick('allocated', size); result = obj.join(values)
            elif name == 'startswith' and len(args) == 1: result = obj.startswith(args[0])
            elif name == 'endswith' and len(args) == 1: result = obj.endswith(args[0])
            elif name == 'isdigit' and not args: result = obj.isdigit()
            else: _policy()
            if type(result) in (str, list): self.tick('allocated', len(result))
        else: _policy()
        return self.guard(result)

    def expr(self, node, scope):
        self.tick('statements')
        if isinstance(node, ast.Constant):
            self.tick('allocated', len(node.value) if type(node.value) is str else 1)
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in scope: raise _Submitted('NameError')
            return scope[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            self.tick('allocated', len(node.elts))
            values = [self.expr(v, scope) for v in node.elts]
            return self.guard(tuple(values) if isinstance(node, ast.Tuple) else values)
        if isinstance(node, ast.Dict):
            self.tick('allocated', len(node.keys) * 2)
            return self.guard({self.expr(k, scope): self.expr(v, scope) for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.Subscript):
            obj = self.expr(node.value, scope)
            index = self.expr(node.slice, scope)
            if type(obj) not in (list, tuple, dict, str): raise _Submitted('TypeError')
            if isinstance(index, slice): self.tick('allocated', len(obj))
            return self.guard(obj[index])
        if isinstance(node, ast.Slice):
            values = [self.expr(v, scope) if v is not None else None for v in (node.lower, node.upper, node.step)]
            if any(type(v) not in (int, bool) for v in values if v is not None): raise _Submitted('TypeError')
            return slice(*values)
        if isinstance(node, ast.BinOp): return self.arithmetic(node.op, self.expr(node.left, scope), self.expr(node.right, scope))
        if isinstance(node, ast.UnaryOp):
            v = self.expr(node.operand, scope)
            if isinstance(node.op, ast.Not): return not v
            if type(v) not in (int, bool, Decimal): raise _Submitted('TypeError')
            self.tick('arithmetic'); return self.guard(-v if isinstance(node.op, ast.USub) else +v)
        if isinstance(node, ast.BoolOp):
            for v in node.values:
                result = self.expr(v, scope)
                if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result: break
            return result
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, scope)
            functions = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                ast.In: lambda a,b: a in b, ast.NotIn: lambda a,b: a not in b,
                ast.Is: operator.is_, ast.IsNot: operator.is_not}
            for op, other in zip(node.ops, node.comparators):
                right = self.expr(other, scope)
                if isinstance(op, (ast.Is, ast.IsNot)) and not (left is None or right is None or type(left) is bool or type(right) is bool): _policy()
                if not functions[type(op)](left, right): return False
                left = right
            return True
        if isinstance(node, ast.Call):
            receiver = self.expr(node.func.value, scope) if isinstance(node.func, ast.Attribute) else None
            args = [self.expr(a, scope) for a in node.args]
            kwargs = {k.arg: self.expr(k.value, scope) for k in node.keywords}
            if isinstance(node.func, ast.Name): return self.call(node.func.id, args, kwargs)
            return self.method(receiver, node.func.attr, args, kwargs)
        _policy()

    def assign(self, node, value, scope):
        self.guard(value)
        if isinstance(node, ast.Name): scope[node.id] = value
        elif isinstance(node, (ast.List, ast.Tuple)):
            if type(value) not in (list, tuple): value = self.sequence(value)
            if len(node.elts) != len(value): raise _Submitted('ValueError')
            for child, item in zip(node.elts, value): self.assign(child, item, scope)
        elif isinstance(node, ast.Subscript):
            obj, index = self.expr(node.value, scope), self.expr(node.slice, scope)
            if type(obj) not in (list, dict) or isinstance(index, slice): _policy()
            self.tick('allocated'); obj[index] = value; self.guard(obj)
        else: _policy()

    def block(self, body, scope):
        for node in body:
            self.tick('statements')
            if isinstance(node, ast.Return): raise _Return(self.expr(node.value, scope) if node.value else None)
            elif isinstance(node, ast.Assign):
                value = self.expr(node.value, scope)
                for target in node.targets: self.assign(target, value, scope)
            elif isinstance(node, ast.AugAssign):
                target = node.target
                if isinstance(target, ast.Subscript):
                    obj, index = self.expr(target.value, scope), self.expr(target.slice, scope)
                    old = obj[index]
                elif isinstance(target, ast.Name): old = self.expr(target, scope)
                else: _policy()
                right = self.expr(node.value, scope)
                if type(old) is list and isinstance(node.op, (ast.Add, ast.Mult)):
                    if isinstance(node.op, ast.Add):
                        items = self.sequence(right)
                        if len(old) + len(items) > self.limits.container: _resource()
                        old.extend(items)
                    else:
                        if type(right) not in (int, bool): raise _Submitted('TypeError')
                        size = len(old) * max(0, right)
                        if size > self.limits.container: _resource()
                        self.tick('allocated', size); old *= right
                    value = self.guard(old)
                else: value = self.arithmetic(node.op, old, right)
                if isinstance(target, ast.Subscript):
                    obj[index] = value; self.guard(obj)
                else: scope[target.id] = value
            elif isinstance(node, ast.If): self.block(node.body if self.expr(node.test, scope) else node.orelse, scope)
            elif isinstance(node, ast.For):
                for value in self.iterable(self.expr(node.iter, scope)):
                    self.tick('iterations'); self.assign(node.target, value, scope)
                    try: self.block(node.body, scope)
                    except _Continue: continue
                    except _Break: break
                else: self.block(node.orelse, scope)
            elif isinstance(node, ast.Break): raise _Break()
            elif isinstance(node, ast.Continue): raise _Continue()
            elif isinstance(node, ast.Expr): self.expr(node.value, scope)
            elif isinstance(node, ast.Raise): raise _Submitted(node.exc.func.id)
            elif not isinstance(node, ast.Pass): _policy()


class RestrictedASTBackend(SandboxBackend):
    sandbox_mode = 'restricted_ast_interpreter'
    def is_available(self): return True
    def cancel(self): pass  # synchronous bounded evaluation; no child/resource retained
    def cleanup(self): pass

    def validate_environment(self):
        if self.run_python(code='def main(data): return sum(data)', function='main', mode='executor', data=[2,3]).result != {'value': 5}:
            raise SandboxError('execution_error', 'backend_infrastructure')
        try:
            self.run_python(code='def main(data): return open("/etc/passwd")', function='main', mode='executor')
        except SandboxError as error:
            if error.code != 'policy_violation': raise
        else:
            raise SandboxError('execution_error', 'backend_infrastructure')
        return {'sandbox_mode': self.sandbox_mode, 'self_tests': True}

    def run_python(self, *, code, function, mode, data=None, cases=None, timeout=2):
        try:
            runtime = Interpreter(code, timeout=timeout)
            if function not in runtime.functions: _policy()
            if mode == 'executor':
                node = runtime.functions[function]
                if [a.arg for a in node.args.posonlyargs + node.args.args] != ['data']: _policy()
                runtime.guard(data)
                result = {'value': runtime.call(function, [deepcopy(data)], {})}
            elif mode == 'test_runner':
                if type(cases) is not list or not 1 <= len(cases) <= 100: _resource()
                runtime.guard(cases)
                rows = []
                for index, case in enumerate(cases):
                    args, kwargs = deepcopy(case['args']), deepcopy(case['kwargs'])
                    before = deepcopy((args, kwargs))
                    actual, exception = None, None
                    try: actual = runtime.call(function, args, kwargs)
                    except _Submitted as error: exception = error.name
                    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as error: exception = type(error).__name__
                    passed = (exception == case['raises'] if 'raises' in case else
                              exception is None and equal(actual, case['expected']))
                    passed = passed and equal((args, kwargs), before)
                    rows.append({'index': index, 'passed': bool(passed), 'actual': actual, 'exception': exception})
                result = {'passed': all(r['passed'] for r in rows), 'cases': rows}
            else: _policy()
            runtime.guard(result)
            # json_bytes rejects non-string object keys/tuples and non-finite values.
            payload = json_bytes({'ok': True, 'result': result})
            if len(payload) > runtime.limits.result_bytes: _resource()
            return SandboxResult(result)
        except SandboxError: raise
        except (RecursionError, MemoryError, OverflowError): _resource()
        except Exception:
            raise SandboxError('execution_error', 'submitted_code') from None
