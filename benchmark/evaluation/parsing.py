"""Strict JSON parsing and a single, explicitly diagnostic fence removal."""
import json
import re
from decimal import Decimal


def unfence(text, language):
    """Remove only one complete fence, with no surrounding prose or whitespace."""
    match = re.fullmatch(r'```(?:' + language + r')?\r?\n([\s\S]*?)\r?\n```(?:\r?\n)?', text)
    return match.group(1) if match else None


def parse_json(text):
    """Reject duplicate keys, non-finite numbers and trailing content."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('non-finite number')

    return json.loads(text, parse_float=Decimal, parse_constant=invalid, object_pairs_hook=pairs)


def differences(actual, expected, numeric=False):
    """Compare types, keys, values and array order without coercion."""
    errors = set()
    number = lambda x: type(x) in (int, Decimal)
    if numeric and number(expected):
        if not number(actual):
            return {'wrong_type'}
        return set() if actual == expected else {'wrong_value'}
    if type(actual) is not type(expected):
        return {'wrong_type'}
    if isinstance(expected, dict):
        if actual.keys() - expected.keys():
            errors.add('extra_fields')
        if expected.keys() - actual.keys():
            errors.add('missing_field')
        for key in actual.keys() & expected.keys():
            errors.update(differences(actual[key], expected[key], numeric))
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            errors.add('wrong_value')
        for a, e in zip(actual, expected):
            errors.update(differences(a, e, numeric))
        if actual != expected and len(actual) == len(expected):
            # Detect permutations using structural comparison, including nested objects.
            remaining = list(expected)
            for value in actual:
                for i, candidate in enumerate(remaining):
                    if not differences(value, candidate, numeric):
                        remaining.pop(i)
                        break
                else:
                    break
            else:
                errors.add('wrong_order')
    elif actual != expected:
        errors.add('wrong_value')
    return errors


def shape_errors(actual, expected, numeric=False):
    """Validate the output schema separately from the expected values/order."""
    errors = differences(actual, expected, numeric) & {'wrong_type', 'extra_fields', 'missing_field'}
    if isinstance(actual, list) and isinstance(expected, list) and expected:
        # Extra elements still need schema validation (e.g. extra tool calls).
        for index, value in enumerate(actual):
            errors.update(shape_errors(value, expected[min(index, len(expected)-1)], numeric))
    if isinstance(actual, list) and expected == []:
        # Empty lists in the frozen JSON answers are lists of string IDs.
        if any(type(value) is not str for value in actual):
            errors.add('wrong_type')
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in actual.keys() & expected.keys():
            errors.update(shape_errors(actual[key], expected[key], numeric))
    return errors
