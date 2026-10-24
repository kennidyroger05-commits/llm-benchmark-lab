"""Trusted wire/policy/worker code. Model code runs only in main() inside Docker."""
import ast
import builtins
from copy import deepcopy
from decimal import Decimal
import importlib
import io
import json
import math
import sys
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace

ALLOWED_IMPORTS = {
    "re": "compile escape findall finditer fullmatch match search split sub subn IGNORECASE MULTILINE DOTALL ASCII VERBOSE",
    "math": "ceil floor trunc factorial gcd lcm isqrt comb perm prod fsum sqrt exp log log2 log10 sin cos tan pi e inf nan isfinite isnan isinf fabs copysign",
    "decimal": "Decimal Context localcontext getcontext setcontext ROUND_HALF_UP ROUND_HALF_EVEN ROUND_DOWN ROUND_UP ROUND_FLOOR ROUND_CEILING",
    "collections": "Counter defaultdict deque OrderedDict",
    "itertools": "accumulate chain combinations combinations_with_replacement compress count cycle dropwhile filterfalse groupby islice pairwise permutations product repeat starmap takewhile tee zip_longest",
    "functools": "reduce partial lru_cache cache cmp_to_key",
    "operator": "add sub mul truediv floordiv mod neg pos eq ne lt le gt ge itemgetter",
    "statistics": "mean median median_low median_high mode multimode fmean",
    "bisect": "bisect bisect_left bisect_right insort insort_left insort_right",
    "heapq": "heapify heappop heappush heappushpop heapreplace merge nlargest nsmallest",
    "string": "ascii_letters ascii_lowercase ascii_uppercase digits hexdigits octdigits punctuation whitespace",
    "json": "dumps loads JSONDecodeError",
}
DENIED_NAMES = {
    "open", "input", "eval", "exec", "compile", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir", "type", "object", "super", "breakpoint",
    "help", "exit", "quit", "memoryview",
}
DENIED_ATTRIBUTES = {
    "gi_frame", "gi_code", "ag_frame", "ag_code", "cr_frame", "cr_code",
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "tb_frame", "tb_next",
    "mro", "format_map",
}


class PolicyError(ValueError):
    pass


def validate_code(code, function, mode):
    """Syntax/AST policy only; this function never executes submitted code."""
    try:
        tree = ast.parse(code)
        # compile checks e.g. return/await placement, without executing anything.
        compile(tree, "<submitted>", "exec")
    except (SyntaxError, ValueError, RecursionError):
        raise PolicyError("invalid_python") from None
    exported = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(exported) != 1 or exported[0].name != function:
        raise PolicyError("wrong_function")
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)) and not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            raise PolicyError("top_level_statement")
    fn = exported[0]
    if fn.decorator_list:
        raise PolicyError("decorator")
    if mode == "executor":
        if (len(fn.args.posonlyargs) + len(fn.args.args) != 1 or fn.args.vararg
                or fn.args.kwarg or fn.args.kwonlyargs or fn.args.defaults):
            raise PolicyError("wrong_signature")
        if (fn.args.posonlyargs + fn.args.args)[0].arg != "data":
            raise PolicyError("wrong_signature")
    aliases = {alias.asname or alias.name: alias.name
               for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases
                and node.attr not in ALLOWED_IMPORTS.get(aliases[node.value.id], "").split()):
            raise PolicyError("forbidden_import_member")
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise PolicyError("decorator")
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
            raise PolicyError("unsupported_statement")
        if isinstance(node, ast.Name) and (node.id in DENIED_NAMES or node.id.startswith("__")):
            raise PolicyError("forbidden_name")
        if isinstance(node, ast.Attribute) and (
                node.attr.startswith("_") or node.attr in DENIED_ATTRIBUTES):
            raise PolicyError("forbidden_attribute")
        if isinstance(node, ast.Import):
            if any(a.name not in ALLOWED_IMPORTS or (a.asname or "").startswith("__") for a in node.names):
                raise PolicyError("forbidden_import")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                raise PolicyError("forbidden_import")
            if any(a.name not in ALLOWED_IMPORTS[node.module].split() or
                   (a.asname or "").startswith("__") for a in node.names):
                raise PolicyError("forbidden_import_member")
    return tree


def pack(value):
    """Typed internal wire format; dict-shaped user data cannot collide with tags."""
    kind = type(value)
    if value is None:
        return ["null", None]
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int", str(value)]
    if kind in (float, Decimal):
        if not (value.is_finite() if kind is Decimal else math.isfinite(value)):
            raise ValueError("non_finite")
        return ["decimal" if kind is Decimal else "float", str(value)]
    if kind is str:
        return ["str", value]
    if kind is list:
        return ["list", [pack(x) for x in value]]
    if kind is dict and all(type(k) is str for k in value):
        return ["dict", [[k, pack(v)] for k, v in value.items()]]
    raise ValueError("not_json")


def unpack(value):
    kind, data = value
    if kind == "null": return None
    if kind == "bool": return data
    if kind == "int": return int(data)
    if kind == "float": return float(data)
    if kind == "decimal": return Decimal(data)
    if kind == "str": return data
    if kind == "list": return [unpack(x) for x in data]
    if kind == "dict": return {k: unpack(v) for k, v in data}
    raise ValueError("invalid_wire")


def json_bytes(value):
    """Public JSON budget encoding, preserving finite Decimal as numeric tokens."""
    if isinstance(value, Decimal):
        if not value.is_finite(): raise ValueError("non_finite")
        return str(value).encode("ascii")
    if type(value) is list:
        return b"[" + b",".join(json_bytes(x) for x in value) + b"]"
    if type(value) is dict:
        if not all(type(k) is str for k in value): raise ValueError("not_json")
        return b"{" + b",".join(json_bytes(k) + b":" + json_bytes(v) for k, v in value.items()) + b"}"
    if value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("not_json")
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def equal(a, b):
    if type(a) is not type(b): return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


class CappedOutput(io.StringIO):
    def write(self, value):
        if self.tell() + len(value) > 8192:
            raise ValueError("stdout_limit")
        return super().write(value)


def _environment_ready():
    # Trusted bootstrap before any submitted code or audit hook. Refuse weaker
    # runtimes rather than silently accepting ignored Docker resource flags.
    import os
    from pathlib import Path
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        root_readonly = any(
            fields[1] == "/" and "ro" in fields[3].split(",")
            for fields in (line.split() for line in Path("/proc/mounts").read_text().splitlines())
        )
        return (
            os.getuid() != 0 and root_readonly
            and Path("/sys/fs/cgroup/memory.max").read_text().strip() == "268435456"
            and Path("/sys/fs/cgroup/pids.max").read_text().strip() == "16"
            and quota != "max" and int(quota) <= int(period)
            and set(os.listdir("/sys/class/net")) == {"lo"}
            and not os.path.exists("/var/run/docker.sock")
            and not any(k in os.environ for k in ("HF_TOKEN", "GITHUB_TOKEN", "PHASE2_FAKE_SECRET"))
        )
    except (OSError, ValueError, IndexError):
        return False


def main():
    if not _environment_ready():
        return {"ok": False, "origin": "backend_infrastructure", "code": "execution_error"}
    request = unpack(json.load(sys.stdin))
    try:
        tree = validate_code(request["code"], request["function"], request["mode"])
    except PolicyError:
        return {"ok": False, "origin": "sandbox_policy", "code": "invalid_data"}

    requested_modules = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules = {name: SimpleNamespace(**{
        member: getattr(importlib.import_module(name), member)
        for member in ALLOWED_IMPORTS[name].split()
    }) for name in sorted(requested_modules)}

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in modules:
            raise PermissionError("import_denied")
        if any(item not in ALLOWED_IMPORTS[name].split() for item in (fromlist or ())):
            raise PermissionError("import_denied")
        return modules[name]

    names = ("abs all any ascii bin bool bytearray bytes callable chr dict divmod enumerate "
             "filter float format frozenset hash hex int isinstance issubclass iter len list map "
             "max min next oct ord pow print range repr reversed round set slice sorted str sum tuple zip "
             "Exception ValueError TypeError KeyError IndexError ZeroDivisionError ArithmeticError "
             "MemoryError StopIteration AssertionError RuntimeError NotImplementedError").split()
    safe = {name: getattr(builtins, name) for name in names}
    safe["__import__"] = restricted_import
    namespace = {"__builtins__": safe, "__name__": "submitted"}
    output, stderr = CappedOutput(), CappedOutput()
    sys.stdin = io.StringIO("")  # Payload already consumed; model has no stdin channel.

    def audit(event, args):
        if event == "open" or event.startswith((
                "socket.", "subprocess.", "os.", "ctypes.", "sys._getframe", "import")):
            raise PermissionError("external_access_denied")
    sys.addaudithook(audit)
    try:
        with redirect_stdout(output), redirect_stderr(stderr):
            exec(compile(tree, "<submitted>", "exec"), namespace)
            function = namespace[request["function"]]
            if request["mode"] == "executor":
                result = {"value": function(deepcopy(request["data"]))}
            else:
                rows = []
                for index, case in enumerate(request["cases"]):
                    args, kwargs = deepcopy(case["args"]), deepcopy(case["kwargs"])
                    before = deepcopy((args, kwargs))
                    actual, exception = None, None
                    try:
                        actual = function(*args, **kwargs)
                        passed = "expected" in case and equal(actual, case.get("expected"))
                    except Exception as error:
                        exception = type(error).__name__
                        passed = case.get("raises") == exception
                    mutation = not equal(args, before[0]) or not equal(kwargs, before[1])
                    rows.append({"index": index, "passed": bool(passed and not mutation),
                                 "actual": actual, "exception": exception})
                result = {"passed": all(row["passed"] for row in rows), "cases": rows}
        if output.getvalue() or stderr.getvalue():
            return {"ok": False, "origin": "submitted_code", "code": "invalid_data"}
        try:
            wire = pack(result)
            if len(json_bytes(result)) > 8192:
                return {"ok": False, "origin": "submitted_code", "code": "resource_limit"}
        except (ValueError, TypeError, RecursionError, UnicodeError):
            return {"ok": False, "origin": "submitted_code", "code": "invalid_data"}
        return {"ok": True, "result": wire, "stdout": "", "stderr": ""}
    except MemoryError:
        return {"ok": False, "origin": "submitted_code", "code": "resource_limit"}
    except PermissionError:
        return {"ok": False, "origin": "sandbox_policy", "code": "invalid_data"}
    except BaseException:
        return {"ok": False, "origin": "submitted_code", "code": "execution_error"}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=True, allow_nan=False))
