"""Three deterministic math tools, returning result payloads without envelopes.

Expressions are interpreted from an allowlisted AST, never compiled/executed.
Decimal // truncates toward zero; Decimal % has the dividend's sign, unlike
floor/remainder operations on Python integers for negative operands.
"""
import ast
from decimal import Context, Decimal, DecimalException, localcontext
import itertools
import json
import keyword
import operator
import re

from .schemas import StructuralValidationError, validate_tool_arguments

MAX_NODES = 256
MAX_OPERATIONS = 10000
MAX_VARIABLES = 100
MAX_COMBINATIONS = 1000000
MAX_RESULT_BYTES = 8192
BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}
LITERAL = re.compile(r"(0|[1-9][0-9]*)(\.[0-9]+)?")
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class ToolExecutionError(ValueError):
    """Stable tool error; input errors never become infrastructure failures."""
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)


def _validate(name, arguments):
    try:
        validate_tool_arguments(name, arguments)
    except StructuralValidationError as error:
        raise ToolExecutionError("invalid_arguments", str(error)) from error


def _identifier(name):
    if not IDENTIFIER.fullmatch(name) or keyword.iskeyword(name):
        raise ToolExecutionError("invalid_arguments", "Invalid or reserved identifier.")


def _expression(text, variables):
    if len(text) > 4096:
        raise ToolExecutionError("resource_limit", "Expression exceeds 4096 characters.")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ToolExecutionError("invalid_expression", "Invalid expression syntax.") from error
    except RecursionError as error:
        raise ToolExecutionError("resource_limit", "Expression nesting exceeded.") from error
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ToolExecutionError("resource_limit", "Expression exceeds 256 AST nodes.")
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
               ast.UAdd, ast.USub, *BINARY.keys())
    if any(type(node) not in allowed for node in nodes):
        raise ToolExecutionError("invalid_expression", "Unsupported expression node.")
    # Validate all nodes before evaluating any of them.
    for node in nodes:
        if isinstance(node, ast.Constant):
            token = ast.get_source_segment(text, node)
            if type(node.value) not in (int, float) or not LITERAL.fullmatch(token):
                raise ToolExecutionError("invalid_expression", "Expected decimal literal.")
        elif isinstance(node, ast.Name):
            if node.id not in variables or not IDENTIFIER.fullmatch(node.id):
                raise ToolExecutionError("invalid_expression", "Unknown variable.")

    def interpret(node):
        if isinstance(node, ast.Constant):
            return Decimal(ast.get_source_segment(text, node))
        if isinstance(node, ast.Name):
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            value = interpret(node.operand)
            return +value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            return BINARY[type(node.op)](interpret(node.left), interpret(node.right))
        return interpret(node.body)

    try:
        result = interpret(tree)
        if not result.is_finite():
            raise ToolExecutionError("invalid_expression", "Result must be finite.")
        return result
    except DecimalException as error:
        raise ToolExecutionError("invalid_expression", "Invalid decimal operation.") from error


def _canonical_decimal(value):
    """Format exactly, without context-dependent normalize() or float conversion."""
    if not value.is_finite():
        raise ToolExecutionError("invalid_expression", "Result must be finite.")
    if value.is_zero():
        return "0"
    # Remove coefficient zeros exactly; normalize() could round in the caller context.
    sign, digits, exponent = value.as_tuple()
    digits = list(digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    value = Decimal((sign, tuple(digits), exponent))
    # Avoid allocating enormous plain-decimal representations.
    if max(value.adjusted() + 1, -value.as_tuple().exponent) > MAX_RESULT_BYTES:
        raise ToolExecutionError("resource_limit", "Decimal result exceeds output budget.")
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _result(payload):
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_RESULT_BYTES:
        raise ToolExecutionError("resource_limit", "Result exceeds 8192 bytes.")
    return payload


def run_calculator_basic(arguments):
    _validate("calculator_basic", arguments)
    with localcontext(Context(prec=28, rounding="ROUND_HALF_EVEN")):
        return _result({"value": _canonical_decimal(_expression(arguments["expression"], {}))})


def run_decimal_calculator(arguments):
    _validate("decimal_calculator", arguments)
    variables = {name: Decimal(value) for name, value in arguments["variables"].items()}
    for name in variables:
        _identifier(name)
    # Preflight expanded primitive steps; repeats themselves are not assignments.
    expanded = 0
    for step in arguments["steps"]:
        expanded += len(step["steps"]) * step["count"] if step["op"] == "repeat" else 1
    if expanded > MAX_OPERATIONS:
        raise ToolExecutionError("resource_limit", "More than 10000 expanded operations.")

    def perform(step):
        _identifier(step["name"])
        if step["name"] not in variables and len(variables) >= MAX_VARIABLES:
            raise ToolExecutionError("resource_limit", "More than 100 variables.")
        value = _expression(step["expression"], variables)
        if step["op"] == "quantize":
            try:
                # Decimal.quantize uses the quantum's exponent, not a multiple of it.
                value = value.quantize(Decimal(step["quantum"]))
            except DecimalException as error:
                raise ToolExecutionError("invalid_expression", "Invalid quantization.") from error
        variables[step["name"]] = value

    with localcontext(Context(prec=arguments["precision"], rounding=arguments["rounding"])):
        for step in arguments["steps"]:
            if step["op"] == "repeat":
                for _ in range(step["count"]):
                    for inner in step["steps"]:
                        perform(inner)
            else:
                perform(step)
        for name in arguments["outputs"]:
            if name not in variables:
                raise ToolExecutionError("invalid_arguments", "Unknown output variable.")
        return _result({"values": {name: _canonical_decimal(variables[name])
                                  for name in arguments["outputs"]}})


def run_integer_optimizer(arguments):
    _validate("integer_optimizer", arguments)
    variables = arguments["variables"]
    names = [v["name"] for v in variables]
    if len(names) != len(set(names)):
        raise ToolExecutionError("invalid_arguments", "Duplicate variable name.")
    count = 1
    for variable in variables:
        _identifier(variable["name"])
        if variable["min"] > variable["max"]:
            raise ToolExecutionError("invalid_arguments", "Variable min exceeds max.")
        count *= variable["max"] - variable["min"] + 1
    if count > MAX_COMBINATIONS:
        raise ToolExecutionError("resource_limit", "Domain exceeds 1000000 combinations.")
    for item in arguments["constraints"] + arguments["objectives"]:
        if not item["coefficients"].keys() <= set(names):
            raise ToolExecutionError("invalid_arguments", "Unknown coefficient variable.")
    # Index coefficients once; objective and enumeration order remain explicit.
    def coefficients(item):
        return [(names.index(name), value) for name, value in item["coefficients"].items()]
    constraints = [(coefficients(c), c["relation"], c["rhs"]) for c in arguments["constraints"]]
    objectives = [(coefficients(o), o["direction"]) for o in arguments["objectives"]]
    def dot(coeffs, values):
        return sum(values[index] * coefficient for index, coefficient in coeffs)
    relations = {"eq": operator.eq, "le": operator.le, "ge": operator.ge}
    best_key = None
    examined = 0
    for values in itertools.product(*(range(v["min"], v["max"] + 1) for v in variables)):
        examined += 1
        if not all(relations[relation](dot(coeffs, values), rhs)
                   for coeffs, relation, rhs in constraints):
            continue
        scores = [dot(coeffs, values) for coeffs, _ in objectives]
        key = tuple(score if direction == "min" else -score
                    for score, (_, direction) in zip(scores, objectives)) + values
        if best_key is None or key < best_key:
            best_key, best_values, best_scores = key, values, scores
    if best_key is None:
        raise ToolExecutionError("infeasible", "No feasible assignment.")
    return _result({"assignment": dict(zip(names, best_values)),
                    "objective_values": best_scores, "combinations_examined": examined})
