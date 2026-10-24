"""Deterministic CSV/JSON/table pipelines, without dynamic execution.

Decimal inputs remain exact internally and become canonical strings at output.
Structured JSON decimals should come from the protocol parser as Decimal;
Python floats are rejected rather than pretending to recover lost precision.
"""
from copy import deepcopy
import csv
from decimal import Context, Decimal, DecimalException, localcontext, MAX_EMAX, MIN_EMIN
from functools import cmp_to_key
import io
import operator
import re

from .math_tools import ToolExecutionError, _canonical_decimal, _result
from .schemas import StructuralValidationError, validate_tool_arguments

MAX_INPUT = 10000
MAX_INTERMEDIATE = 100000


def _error(message):
    raise ToolExecutionError("invalid_data", message)


def _validate(name, arguments):
    # Resource failures have a distinct code even when the schema also has a cap.
    if type(arguments) is dict:
        if type(arguments.get("pipeline")) is list and len(arguments["pipeline"]) > 32:
            raise ToolExecutionError("resource_limit", "Pipeline exceeds 32 stages.")
        if name == "table_transform" and type(arguments.get("rows")) is list:
            _limit(arguments["rows"], MAX_INPUT)
    try:
        validate_tool_arguments(name, arguments)
    except StructuralValidationError as error:
        raise ToolExecutionError("invalid_arguments", str(error)) from error


def _limit(rows, maximum=MAX_INTERMEDIATE):
    if len(rows) > maximum:
        raise ToolExecutionError("resource_limit", "Row limit exceeded.")


def _copy_json(value):
    if type(value) is float:
        _error("Supply JSON decimals as Decimal, not float.")
    if type(value) is dict:
        return {k: _copy_json(v) for k, v in value.items()}
    if type(value) is list:
        return [_copy_json(v) for v in value]
    return value


def resolve_path(value, path):
    """Resolve keys/indices strictly, including an empty path for the current value."""
    if type(path) is not list:
        _error("Path must be a list.")
    for key in path:
        if type(key) is str and type(value) is dict and key in value:
            value = value[key]
        elif type(key) is int and key >= 0 and type(value) is list and key < len(value):
            value = value[key]
        else:
            _error("Missing path or incompatible path component.")
    return value


def _rows(value, maximum=MAX_INPUT):
    if type(value) is not list or any(type(row) is not dict for row in value):
        _error("Expected a list of objects.")
    _limit(value, maximum)
    return value


def _key(value):
    """Type-tagged hashable identity, preserving array order and ignoring object order."""
    if type(value) is dict:
        return ("object", tuple((k, _key(v)) for k, v in sorted(value.items())))
    if type(value) is list:
        return ("array", tuple(_key(v) for v in value))
    return (type(value).__name__, value)


def _compatible(a, b):
    if type(a) is not type(b):
        _error("Incompatible comparison types.")


def _compare(a, b):
    _compatible(a, b)
    if type(a) not in (int, Decimal, str):
        _error("Values are not orderable.")
    return (a > b) - (a < b)


def _numeric(value):
    if type(value) not in (int, Decimal):
        _error("Expected integer or Decimal.")


def _arithmetic(a, b, op):
    _numeric(a)
    _numeric(b)
    _compatible(a, b)
    if type(a) is int:
        return op(a, b)
    # Derive sufficient precision for exact finite add/subtract/multiply, not
    # the ambient global Decimal context. No arbitrary monetary rounding here.
    if op is operator.mul:
        precision = len(a.as_tuple().digits) + len(b.as_tuple().digits)
    else:
        precision = max(a.adjusted(), b.adjusted()) - min(
            a.as_tuple().exponent, b.as_tuple().exponent) + 3
    try:
        with localcontext(Context(prec=max(1, precision), Emax=MAX_EMAX, Emin=MIN_EMIN)):
            return op(a, b)
    except (DecimalException, ValueError) as error:
        raise ToolExecutionError("invalid_data", "Invalid decimal arithmetic.") from error


def _expr(expression, row):
    if "field" in expression:
        return resolve_path(row, expression["field"])
    if "literal" in expression:
        return deepcopy(expression["literal"])
    a, b = [_expr(operand, row) for operand in expression["operands"]]
    return _arithmetic(a, b, {
        "add": operator.add, "subtract": operator.sub, "multiply": operator.mul,
    }[expression["operator"]])


def _condition(row, condition):
    a = resolve_path(row, condition["path"])
    b = condition["value"]
    op = condition["operator"]
    if op == "in":
        # Validate all operands, even if an earlier item matches.
        for candidate in b:
            _compatible(a, candidate)
        return any(_key(a) == _key(candidate) for candidate in b)
    _compatible(a, b)
    if op in ("eq", "ne"):
        equal = _key(a) == _key(b)
        return equal if op == "eq" else not equal
    comparison = _compare(a, b)
    return {"lt": comparison < 0, "le": comparison <= 0,
            "gt": comparison > 0, "ge": comparison >= 0}[op]


def _unique_names(names):
    if len(names) != len(set(names)):
        _error("Duplicate output names.")


def _pipeline(rows, pipeline, data=None):
    for stage in pipeline:
        op = stage["op"]
        if op == "latest_by":
            groups = {}
            for row in rows:
                key = tuple(_key(resolve_path(row, p)) for p in stage["keys"])
                version = resolve_path(row, stage["version"])
                _numeric(version)
                if key not in groups:
                    groups[key] = (row, version, False)
                else:
                    previous, old_version, tied = groups[key]
                    comparison = _compare(version, old_version)
                    if comparison > 0:
                        groups[key] = (row, version, False)
                    elif comparison == 0:
                        groups[key] = (previous, old_version, True)
            if any(tied for _, _, tied in groups.values()):
                _error("Tied greatest version.")
            rows = [row for row, _, _ in groups.values()]
        elif op == "filter":
            out = []
            for row in rows:
                matches = [_condition(row, c) for c in stage["conditions"]]
                if all(matches):
                    out.append(row)
            rows = out
        elif op == "derive":
            rows = [{**row, stage["name"]: _expr(stage["expression"], row)} for row in rows]
        elif op == "explode":
            expansions = [(row, resolve_path(row, stage["path"])) for row in rows]
            if any(type(values) is not list for _, values in expansions):
                _error("explode expects an array.")
            if sum(len(values) for _, values in expansions) > MAX_INTERMEDIATE:
                raise ToolExecutionError("resource_limit", "Intermediate row limit exceeded.")
            rows = [{**deepcopy(row), stage["as"]: deepcopy(value)}
                    for row, values in expansions for value in values]
        elif op == "join":
            right = stage["right"]
            if type(right) is dict:
                right = resolve_path(data, right["root"])
            _rows(right)
            index = {}
            for row in right:
                index.setdefault(_key(resolve_path(row, stage["right_key"])), []).append(row)
            out = []
            for left in rows:
                for right_row in index.get(_key(resolve_path(left, stage["left_key"])), []):
                    if len(out) >= MAX_INTERMEDIATE:
                        raise ToolExecutionError("resource_limit", "Intermediate row limit exceeded.")
                    out.append({**deepcopy(left), stage["as"]: deepcopy(right_row)})
            rows = out
        elif op == "group":
            names = [s["name"] for s in stage["keys"] + stage["aggregates"]]
            _unique_names(names)
            groups = {}
            for row in rows:
                values = [resolve_path(row, s["path"]) for s in stage["keys"]]
                key = tuple(_key(v) for v in values)
                if key not in groups:
                    groups[key] = dict(zip((s["name"] for s in stage["keys"]), values))
                result = groups[key]
                for aggregate in stage["aggregates"]:
                    name = aggregate["name"]
                    if aggregate["function"] == "count":
                        result[name] = result.get(name, 0) + 1
                    else:
                        value = resolve_path(row, aggregate["path"])
                        _numeric(value)
                        result[name] = (_arithmetic(result[name], value, operator.add)
                                        if name in result else value)
            rows = list(groups.values())
        elif op == "sort":
            # Preflight even singleton rows, for which sorted() does not compare.
            for spec in stage["by"]:
                values = [resolve_path(row, spec["path"]) for row in rows]
                for value in values:
                    _compare(value, value)
                    _compatible(value, values[0])
            def compare(left, right):
                for spec in stage["by"]:
                    c = _compare(resolve_path(left, spec["path"]), resolve_path(right, spec["path"]))
                    if c:
                        return c if spec["direction"] == "asc" else -c
                return 0
            rows = sorted(rows, key=cmp_to_key(compare))
        elif op == "project":
            _unique_names([s["name"] for s in stage["fields"]])
            rows = [{s["name"]: deepcopy(resolve_path(row, s["path"]))
                     for s in stage["fields"]} for row in rows]
        _limit(rows)
    return rows


def _serialize(value):
    if isinstance(value, Decimal):
        return _canonical_decimal(value)
    if type(value) is list:
        return [_serialize(v) for v in value]
    if type(value) is dict:
        return {k: _serialize(v) for k, v in value.items()}
    return value


def _finish(rows):
    return _result({"rows": _serialize(rows)})


def _parse_cell(text, kind):
    if kind == "string":
        return text
    if kind == "integer" and re.fullmatch(r"[+-]?[0-9]+", text):
        try:
            return int(text)
        except ValueError as error:
            raise ToolExecutionError("invalid_data", "Invalid CSV integer.") from error
    if kind == "decimal" and re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?", text):
        return Decimal(text)
    if kind == "boolean" and text in ("true", "false"):
        return text == "true"
    _error("Invalid typed CSV cell.")


def run_csv_query(arguments):
    _validate("csv_query", arguments)
    try:
        reader = csv.reader(io.StringIO(arguments["csv"], newline=""),
                            delimiter=arguments["delimiter"], strict=True)
        header = next(reader, None)
        if not header or len(header) != len(set(header)) or set(header) != set(arguments["columns"]):
            _error("CSV requires unique headers exactly matching columns.")
        rows = []
        for cells in reader:
            if len(cells) != len(header):
                _error("Inconsistent CSV row width.")
            if len(rows) >= MAX_INPUT:
                raise ToolExecutionError("resource_limit", "Input row limit exceeded.")
            rows.append({k: _parse_cell(v, arguments["columns"][k]) for k, v in zip(header, cells)})
    except (csv.Error, TypeError) as error:
        raise ToolExecutionError("invalid_data", "Malformed CSV.") from error
    pipeline = _copy_json(arguments["pipeline"])
    return _finish(_pipeline(rows, pipeline))


def run_json_query(arguments):
    _validate("json_query", arguments)
    data = _copy_json(arguments["data"])
    rows = _rows(resolve_path(data, arguments["root"]))
    return _finish(_pipeline(rows, _copy_json(arguments["pipeline"]), data))


def run_table_transform(arguments):
    _validate("table_transform", arguments)
    rows = _rows(_copy_json(arguments["rows"]))
    return _finish(_pipeline(rows, _copy_json(arguments["pipeline"])))
