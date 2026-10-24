"""Format-only validators; no expected values, task data or output repair."""
from decimal import Decimal, DecimalException
import json

from benchmark.orchestration.protocol import _unique_object, _reject_constant, _parse_decimal
from .math_tools import ToolExecutionError, _result
from .schemas import StructuralValidationError, validate_tool_arguments


def _arguments(name, arguments):
    try:
        validate_tool_arguments(name, arguments)
    except StructuralValidationError:
        raise ToolExecutionError("invalid_arguments", "Invalid validator arguments.") from None


def run_json_validator(arguments):
    _arguments("json_validator", arguments)
    schema = arguments["schema"]
    # These relationships are schema validity, never expected-answer checks.
    def check_schema(s):
        if s["type"] == "object":
            if (len(s["required"]) != len(set(s["required"])) or
                    not set(s["required"]) <= s["properties"].keys()):
                raise ToolExecutionError("invalid_arguments", "Invalid structural schema.")
            for child in s["properties"].values():
                check_schema(child)
        elif s["type"] == "array":
            if s.get("minItems", 0) > s.get("maxItems", float("inf")):
                raise ToolExecutionError("invalid_arguments", "Invalid structural schema.")
            check_schema(s["items"])
    check_schema(schema)
    try:
        value = json.loads(arguments["text"], object_pairs_hook=_unique_object,
                           parse_float=_parse_decimal, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError, DecimalException):
        return {"valid": False, "errors": [{"path": [], "code": "invalid_json"}]}
    errors = []
    def visit(v, s, path):
        kind = s["type"]
        types = {"object": dict, "array": list, "string": str, "integer": int,
                 "boolean": bool, "null": type(None)}
        correct = type(v) in (int, Decimal) if kind == "number" else type(v) is types[kind]
        if not correct:
            errors.append({"path": path, "code": "wrong_type"})
            return
        if kind == "object":
            for key in sorted(set(s["required"]) - v.keys()):
                errors.append({"path": path + [key], "code": "missing_field"})
            for key in sorted(v):
                if key in s["properties"]:
                    visit(v[key], s["properties"][key], path + [key])
                elif not s["additionalProperties"]:
                    errors.append({"path": path + [key], "code": "extra_field"})
        elif kind == "array":
            if len(v) < s.get("minItems", 0):
                errors.append({"path": path, "code": "min_items"})
            if len(v) > s.get("maxItems", len(v)):
                errors.append({"path": path, "code": "max_items"})
            for index, child in enumerate(v):
                visit(child, s["items"], path + [index])
    visit(value, schema, [])
    return _result({"valid": not errors, "errors": errors})


def run_text_format_validator(arguments):
    _arguments("text_format_validator", arguments)
    if len(arguments["prefixes"]) != arguments["line_count"]:
        raise ToolExecutionError("invalid_arguments", "prefixes must match line_count.")
    text = arguments["text"].replace("\r\n", "\n")
    if arguments["allow_final_newline"] and text.endswith("\n"):
        text = text[:-1]
    # Empty text has one empty line, consistent with splitting on LF.
    lines = text.split("\n")
    errors = []
    if len(lines) != arguments["line_count"]:
        errors.append({"line": None, "code": "line_count"})
    for index, line in enumerate(lines):
        if index < len(arguments["prefixes"]) and not line.startswith(arguments["prefixes"][index]):
            errors.append({"line": index + 1, "code": "prefix"})
        if not arguments["allow_empty_lines"] and not line:
            errors.append({"line": index + 1, "code": "empty_line"})
        if not arguments["allow_outer_spaces"] and line != line.strip():
            errors.append({"line": index + 1, "code": "outer_space"})
    return _result({"valid": not errors, "errors": errors})
