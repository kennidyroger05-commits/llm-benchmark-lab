"""Small structural validator; never evaluates expressions, code or pipelines.

Contracts use a documented JSON Schema subset plus x-finiteJSON, x-exprNodes,
x-positiveDecimal and x-schemaLimits for the shared types' structural limits.
Dynamic maps explicitly validate each value; fixed records reject extra keys.
"""
from copy import deepcopy
from decimal import Decimal
import math
import re

from ._schema_definitions import DEFINITIONS, PARAMETERS


class StructuralValidationError(ValueError):
    code = "invalid_arguments"


class UnknownToolError(ValueError):
    code = "unknown_tool"


def _fail(path, message):
    raise StructuralValidationError(f"{path}: {message}")


def _finite_json(value, path="$"):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, Decimal) and value.is_finite():
        return
    if type(value) is list:
        for i, child in enumerate(value):
            _finite_json(child, f"{path}[{i}]")
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for key, child in value.items():
            _finite_json(child, f"{path}.{key}")
        return
    _fail(path, "expected finite JSON value")


def _expr_nodes(value):
    if type(value) is not dict:
        return 1
    return 1 + sum(_expr_nodes(v) for v in value.get("operands", []))


def _schema_size(value, depth=1):
    children = list(value.get("properties", {}).values())
    if "items" in value:
        children.append(value["items"])
    sizes = [_schema_size(c, depth + 1) for c in children]
    return 1 + sum(n for n, _ in sizes), max([depth] + [d for _, d in sizes])


def _validate(value, schema, path):
    if "$ref" in schema:
        _validate(value, DEFINITIONS[schema["$ref"].split("/")[-1]], path)
        return
    if schema.get("x-finiteJSON"):
        _finite_json(value, path)
        return
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                _validate(value, branch, path)
                break
            except StructuralValidationError:
                pass
        else:
            _fail(path, "no matching structural alternative")
    else:
        kind = schema["type"]
        valid = {
            "object": type(value) is dict, "array": type(value) is list,
            "string": type(value) is str, "integer": type(value) is int,
            "number": type(value) in (int, float, Decimal),
            "boolean": type(value) is bool, "null": value is None,
        }[kind]
        if not valid:
            _fail(path, f"expected {kind}")
        if kind == "object":
            if any(type(k) is not str for k in value):
                _fail(path, "object keys must be strings")
            if not set(schema.get("required", [])) <= value.keys():
                _fail(path, "missing required field")
            if len(value) > schema.get("maxProperties", len(value)):
                _fail(path, "too many properties")
            properties = schema.get("properties", {})
            additional = schema.get("additionalProperties", False)
            for key, child in value.items():
                if "propertyNames" in schema:
                    _validate(key, schema["propertyNames"], path)
                if key in properties:
                    _validate(child, properties[key], f"{path}.{key}")
                elif isinstance(additional, dict):
                    _validate(child, additional, f"{path}.{key}")
                elif not additional:
                    _fail(path, "extra field")
        elif kind == "array":
            if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
                _fail(path, "array length out of bounds")
            for i, child in enumerate(value):
                _validate(child, schema["items"], f"{path}[{i}]")
            if schema.get("uniqueItems"):
                # Unique arrays in these contracts contain strings/identifiers.
                if any(x == y for i, x in enumerate(value) for y in value[:i]):
                    _fail(path, "duplicate item")
        elif kind == "string":
            if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", len(value)):
                _fail(path, "string length out of bounds")
            if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
                _fail(path, "invalid string pattern")
        elif kind in ("number", "integer"):
            _finite_json(value, path)
            if "minimum" in schema and value < schema["minimum"]:
                _fail(path, "below minimum")
            if "maximum" in schema and value > schema["maximum"]:
                _fail(path, "above maximum")
        if "enum" in schema and value not in schema["enum"]:
            _fail(path, "invalid enum")
    if schema.get("x-positiveDecimal") and Decimal(value) <= 0:
        _fail(path, "expected positive decimal")
    if "x-exprNodes" in schema and _expr_nodes(value) > schema["x-exprNodes"]:
        _fail(path, "too many expression nodes")
    if schema.get("x-schemaLimits"):
        count, depth = _schema_size(value)
        if count > 256 or depth > 16:
            _fail(path, "structural schema size exceeded")


def validate_structure(value, schema):
    """Raise StructuralValidationError; return None without mutating input."""
    try:
        _validate(value, schema, "$")
    except (RecursionError, ArithmeticError) as error:
        raise StructuralValidationError("Invalid or excessively nested structure") from error


def validate_tool_arguments(tool_name, arguments):
    """Validate a known tool's argument record, independently of the protocol parser."""
    if not isinstance(tool_name, str) or tool_name not in PARAMETERS:
        raise UnknownToolError("Unknown tool")
    validate_structure(arguments, PARAMETERS[tool_name])


def public_schema(schema):
    """Copy a contract with only its reachable shared definitions, including recursion."""
    definitions = {}
    def visit(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                if name not in definitions:
                    definitions[name] = deepcopy(DEFINITIONS[name])
                    visit(definitions[name])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(schema)
    return {**deepcopy(schema), **({"$defs": definitions} if definitions else {})}
