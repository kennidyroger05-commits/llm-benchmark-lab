"""Declarative contracts transcribed from docs/phase2_tool_catalog.md."""


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


def array(items, **limits):
    return {"type": "array", "items": items, **limits}


def string(**limits):
    return {"type": "string", **limits}


def integer(**limits):
    return {"type": "integer", **limits}


def enum(*values):
    return {"type": "string", "enum": list(values)}


def ref(name):
    return {"$ref": "#/$defs/" + name}


def mapping(values, keys=None, **limits):
    return {"type": "object", "additionalProperties": values,
            **({"propertyNames": keys} if keys else {}), **limits}


JSON = ref("JSON")
IDENTIFIER = string(pattern=r"[A-Za-z][A-Za-z0-9_]*")
DECIMAL = string(pattern=r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")
NONEMPTY = string(minLength=1)
PATH = array({"anyOf": [string(), integer(minimum=0)]})
ROW = mapping(JSON)
DIRECTION = enum("asc", "desc")
VARIABLE = obj({"name": IDENTIFIER, "min": integer(), "max": integer()})
COEFFICIENTS = mapping(integer(), IDENTIFIER)
CONSTRAINT = obj({"coefficients": COEFFICIENTS, "relation": enum("eq", "le", "ge"), "rhs": integer()})
OBJECTIVE = obj({"coefficients": COEFFICIENTS, "direction": enum("min", "max")})
ASSIGN = obj({"op": enum("assign"), "name": IDENTIFIER, "expression": NONEMPTY})
QUANTIZE = obj({"op": enum("quantize"), "name": IDENTIFIER,
                "expression": NONEMPTY, "quantum": {**DECIMAL, "x-positiveDecimal": True}})
SIMPLE_STEP = {"anyOf": [ASSIGN, QUANTIZE]}
STEP = {"anyOf": [ASSIGN, QUANTIZE, obj({
    "op": enum("repeat"), "count": integer(minimum=1, maximum=1000),
    "steps": array(SIMPLE_STEP)})]}
EXPR = {"anyOf": [
    obj({"field": PATH}), obj({"literal": JSON}),
    obj({"operator": enum("add", "subtract", "multiply"),
         "operands": array(ref("Expr"), minItems=2, maxItems=2)})],
    "x-exprNodes": 64}
KEY_FIELD = obj({"name": NONEMPTY, "path": PATH})
STAGE = {"anyOf": [
    obj({"op": enum("latest_by"), "keys": array(PATH), "version": PATH}),
    obj({"op": enum("filter"), "conditions": array({"anyOf": [obj({
        "path": PATH, "operator": enum("eq", "ne", "lt", "le", "gt", "ge"),
        "value": JSON}), obj({"path": PATH, "operator": enum("in"), "value": array(JSON)})]})}),
    obj({"op": enum("derive"), "name": NONEMPTY, "expression": ref("Expr")}),
    obj({"op": enum("explode"), "path": PATH, "as": NONEMPTY}),
    obj({"op": enum("join"), "right": {"anyOf": [
        array(ROW), obj({"root": PATH})]}, "left_key": PATH, "right_key": PATH, "as": NONEMPTY}),
    obj({"op": enum("group"), "keys": array(KEY_FIELD), "aggregates": array({"anyOf": [obj({
        "name": NONEMPTY, "function": enum("sum"), "path": PATH}),
        obj({"name": NONEMPTY, "function": enum("count"), "path": {"type": "null"}})]})}),
    obj({"op": enum("sort"), "by": array(obj({
        "path": PATH, "direction": DIRECTION}), minItems=1)}),
    obj({"op": enum("project"), "fields": array(KEY_FIELD)})]}
TEST_CASE = {"anyOf": [
    obj({"args": array(JSON), "kwargs": ROW, "expected": JSON}),
    obj({"args": array(JSON), "kwargs": ROW, "raises": enum(
        "ValueError", "TypeError", "KeyError", "IndexError", "ZeroDivisionError")})]}
SCALAR_SCHEMAS = [obj({"type": enum(t)}) for t in (
    "string", "integer", "number", "boolean", "null")]
STRUCTURAL_SCHEMA = {"anyOf": SCALAR_SCHEMAS + [
    obj({"type": enum("object"), "properties": mapping(ref("StructuralSchema")),
         "required": array(string()), "additionalProperties": {"type": "boolean"}}),
    obj({"type": enum("array"), "items": ref("StructuralSchema"),
         "minItems": integer(minimum=0), "maxItems": integer(minimum=0)},
        required=["type", "items"])], "x-schemaLimits": True}
DEFINITIONS = {
    "JSON": {"x-finiteJSON": True}, "Expr": EXPR, "StructuralSchema": STRUCTURAL_SCHEMA,
    "Identifier": IDENTIFIER, "Decimal": DECIMAL, "Path": PATH, "Variable": VARIABLE,
    "Constraint": CONSTRAINT, "Objective": OBJECTIVE, "Step": STEP, "Stage": STAGE,
    "TestCase": TEST_CASE,
}
PIPELINE = array(ref("Stage"), maxItems=32)
PARAMETERS = {
    "calculator_basic": obj({"expression": string(minLength=1, maxLength=4096)}),
    "decimal_calculator": obj({
        "precision": integer(minimum=28, maximum=100),
        "rounding": enum("ROUND_HALF_UP", "ROUND_HALF_EVEN", "ROUND_DOWN",
                         "ROUND_UP", "ROUND_FLOOR", "ROUND_CEILING"),
        "variables": mapping(DECIMAL, IDENTIFIER, maxProperties=100),
        "steps": array(ref("Step")), "outputs": array(IDENTIFIER, uniqueItems=True)}),
    "integer_optimizer": obj({"variables": array(VARIABLE), "constraints": array(CONSTRAINT),
                              "objectives": array(OBJECTIVE, minItems=1)}),
    "csv_query": obj({"csv": string(), "delimiter": string(minLength=1, maxLength=1),
                     "columns": mapping(enum("string", "integer", "decimal", "boolean")),
                     "pipeline": PIPELINE}),
    "json_query": obj({"data": JSON, "root": PATH, "pipeline": PIPELINE}),
    "table_transform": obj({"rows": array(ROW, maxItems=10000), "pipeline": PIPELINE}),
    "python_executor": obj({"code": NONEMPTY, "data": JSON}),
    "lookup_order": obj({"order_id": NONEMPTY}),
    "check_return": obj({"line_id": NONEMPTY, "reason": NONEMPTY}),
    "create_return": obj({"order_id": NONEMPTY, "line_ids": array(NONEMPTY, minItems=1, uniqueItems=True)}),
    "cancel_order": obj({"order_id": NONEMPTY}),
    "refund_payment": obj({"order_id": NONEMPTY}),
    "order_workflow_query": obj({"order_id": NONEMPTY, "fields": array(enum("status", "lines"), minItems=1, uniqueItems=True)}),
    "get_stock": obj({"sku": NONEMPTY}),
    "reserve_stock": obj({"order_id": NONEMPTY, "sku": NONEMPTY, "quantity": integer(minimum=1)}),
    "charge": obj({"order_id": NONEMPTY, "amount_centavos": integer(minimum=1)}),
    "release_stock": obj({"reservation_id": NONEMPTY}),
    "log_failure": obj({"order_id": NONEMPTY, "reason": NONEMPTY}),
    "issue_invoice": obj({"order_id": NONEMPTY}),
    "send_receipt": obj({"order_id": NONEMPTY}),
    "inventory_query": obj({"skus": array(NONEMPTY, minItems=1, uniqueItems=True)}),
    "python_syntax_check": obj({"code": string()}),
    "python_test_runner": obj({"code": NONEMPTY, "function": IDENTIFIER,
                               "cases": array(ref("TestCase"), minItems=1, maxItems=100)}),
    "json_validator": obj({"text": string(), "schema": ref("StructuralSchema")}),
    "text_format_validator": obj({
        "text": string(), "line_count": integer(minimum=0), "prefixes": array(string()),
        "allow_empty_lines": {"type": "boolean"}, "allow_outer_spaces": {"type": "boolean"},
        "allow_final_newline": {"type": "boolean"}}),
}
# These describe public results, never scenario-specific fixture values.
BOOL = {"type": "boolean"}
NULLABLE_INT = {"anyOf": [integer(), {"type": "null"}]}
RETURNS = {name: ROW for name in PARAMETERS}
RETURNS.update({
    "calculator_basic": obj({"value": DECIMAL}),
    "decimal_calculator": obj({"values": mapping(DECIMAL)}),
    "integer_optimizer": obj({"assignment": mapping(integer()), "objective_values": array(integer()),
                              "combinations_examined": integer()}),
    "python_executor": obj({"value": JSON}),
    "lookup_order": obj({"status": string(), "lines": array(string())}),
    "check_return": obj({"eligible": BOOL}),
    "get_stock": obj({"available": integer()}),
    "order_workflow_query": obj({"order": obj({"status": string(), "lines": array(string())}, required=[])}),
    "reserve_stock": obj({"ok": BOOL, "reservation_id": string()}, required=["ok"]),
    "charge": obj({"ok": BOOL, "error": string()}, required=["ok"]),
    "python_syntax_check": obj({"valid": BOOL, "errors": array(obj({
        "code": string(), "line": NULLABLE_INT, "column": NULLABLE_INT}))}),
    "python_test_runner": obj({"passed": BOOL, "cases": array(obj({
        "index": integer(), "passed": BOOL, "actual": JSON,
        "exception": {"anyOf": [string(), {"type": "null"}]}}))}),
    "json_validator": obj({"valid": BOOL, "errors": array(obj({"path": PATH, "code": string()}))}),
    "text_format_validator": obj({"valid": BOOL, "errors": array(obj({"line": NULLABLE_INT, "code": string()}))}),
    "inventory_query": obj({"rows": array(obj({"sku": string(), "available": integer()}))}),
})
for _name in ("csv_query", "json_query", "table_transform"):
    RETURNS[_name] = obj({"rows": array(ROW)})
for _name in ("create_return", "release_stock", "log_failure"):
    RETURNS[_name] = obj({"ok": BOOL})

# CSV/table join operands must contain their explicit right-hand rows.
from copy import deepcopy as _deepcopy
for _name in ("csv_query", "table_transform"):
    _stage = _deepcopy(STAGE)
    for _branch in _stage["anyOf"]:
        if _branch["properties"]["op"]["enum"] == ["join"]:
            _branch["properties"]["right"] = array(ROW)
    PARAMETERS[_name]["properties"]["pipeline"] = array(_stage, maxItems=32)
