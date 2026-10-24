"""Deterministic evaluation of the ten frozen RAW task contracts."""
from collections import Counter
from decimal import Decimal
import re

from .code_evaluation import evaluate_code
from .parsing import differences, parse_json, shape_errors, unfence


def schema_errors(value, schema):
    """Minimal validation for the closed tool schemas in Benchmark V1."""
    types = {'object': dict, 'array': list, 'string': str, 'integer': int, 'boolean': bool}
    if type(value) is not types[schema['type']]:
        return {'wrong_type'}
    errors = set()
    if isinstance(value, dict):
        properties = schema['properties']
        if value.keys() - properties.keys():
            errors.add('extra_fields')
        if set(schema['required']) - value.keys():
            errors.add('missing_field')
        for key in value.keys() & properties.keys():
            errors.update(schema_errors(value[key], properties[key]))
    elif isinstance(value, list):
        for item in value:
            errors.update(schema_errors(item, schema['items']))
        if len(value) < schema.get('minItems', 0):
            errors.add('wrong_value')
        if schema.get('uniqueItems') and len(set(map(repr, value))) != len(value):
            errors.add('wrong_value')
    elif isinstance(value, str) and len(value) < schema.get('minLength', 0):
        errors.add('wrong_value')
    elif type(value) is int and value < schema.get('minimum', value):
        errors.add('wrong_value')
    return errors


def tool_shape(value, task):
    if not isinstance(value, list):
        return {'wrong_type'}
    catalog = {t['name']: t['parameters'] for t in task['tool_catalog']}
    errors = set()
    for call in value:
        if not isinstance(call, dict):
            errors.add('wrong_type')
            continue
        errors.update(shape_errors(call, {'tool': '', 'arguments': {}}) - {'extra_fields'})
        if call.keys() - {'tool', 'arguments'}:
            errors.add('extra_fields')
        if isinstance(call.get('tool'), str) and call['tool'] in catalog:
            errors.update(schema_errors(call.get('arguments'), catalog[call['tool']]))
        elif isinstance(call.get('tool'), str):
            errors.add('unknown_tool')
    return errors


def evaluate_task(task, result):
    """Keep technical, format, content and official strict correctness separate."""
    response = result.get('response') or ''
    technical = result.get('status') == 'success'
    hit = (type(result.get('max_new_tokens')) is int and result['max_new_tokens'] > 0
           and result.get('output_tokens') == result['max_new_tokens'])
    reasons = set() if technical else {'technical_error'}
    if hit:
        reasons.add('hit_max_new_tokens')
    kind = task['evaluation_type']
    if kind == 'code':
        stripped = unfence(response, 'python')
        fenced = stripped is not None
        format_ok, content_ok, code_reasons = evaluate_code(
            stripped if fenced else response, task)
        format_ok = format_ok and not fenced
        reasons.update(code_reasons)
        if fenced or '```' in response:
            reasons.add('markdown_not_allowed')
    elif kind == 'structured_text':
        # The frozen T10 notes permit CRLF normalization and one optional final LF.
        normalize = lambda text: text.replace('\r\n', '\n').removesuffix('\n')
        expected = normalize(task['expected_answer'])
        actual = normalize(response)
        lines = actual.split('\n')
        keys = [line.split('=')[0] for line in expected.split('\n')]
        patterns = ['(?:ALERT|OK)', r'[0-9]+', r'[0-9]+', r'[0-9]+', r'[0-9]+',
                    r'(?:none|(?:auth|billing|search|storage)(?:,(?:auth|billing|search|storage))*)',
                    '(?:auth|billing|search|storage|none)', '(?:ESCALATE|MONITOR|NONE)']
        format_ok = len(lines) == 8 and all(
            re.fullmatch(re.escape(key) + '=' + pattern, line)
            for key, pattern, line in zip(keys, patterns, lines))
        stripped = unfence(response, '(?:text)?')
        content_ok = normalize(stripped if stripped is not None else response) == expected
        if not format_ok:
            reasons.add('wrong_line_format')
        if stripped is not None:
            reasons.add('markdown_not_allowed')
        if not content_ok:
            reasons.add('wrong_value')
    else:
        numeric = task['id'] == 'T1'
        # Support programmatic callers using a regular JSON loader as well.
        expected = task['expected_answer']
        if numeric:
            expected = {key: Decimal(str(value)) for key, value in expected.items()}
        parsed = None
        try:
            parsed = parse_json(response)
            parsed_strict = True
        except (ValueError, TypeError):
            parsed_strict = False
            reasons.add('invalid_json')
            stripped = unfence(response, 'json')
            if stripped is not None:
                reasons.add('markdown_not_allowed')
                try:
                    parsed = parse_json(stripped)
                except (ValueError, TypeError):
                    pass
            elif '```' in response:
                reasons.add('markdown_not_allowed')
        delta = differences(parsed, expected, numeric)
        shape = (tool_shape(parsed, task) if task['id'] in {'T5', 'T6'}
                 else shape_errors(parsed, expected, numeric))
        format_ok = parsed_strict and not shape
        content_ok = not delta
        reasons.update(shape if task['id'] in {'T5', 'T6'} else delta | shape)
        if task['id'] == 'T2' and not shape:
            a, b, c = (parsed[k] for k in ('pacotes_a', 'pacotes_b', 'pacotes_c'))
            if not (0 <= a <= 8 and b >= 2 and 0 <= c <= 6 and
                    12*a + 7*b + c == 137 and parsed['custo_centavos'] == 5890*a + 3510*b + 620*c):
                content_ok = False
                reasons.add('wrong_value')
        if task['id'] in {'T5', 'T6'} and isinstance(parsed, list):
            names = [c.get('tool') if isinstance(c, dict) else None for c in parsed]
            wanted = [c['tool'] for c in expected]
            if all(isinstance(n, str) for n in names):
                if Counter(names) - Counter(wanted):
                    reasons.add('extra_tool_call')
                if Counter(wanted) - Counter(names):
                    reasons.add('missing_tool_call')
                if any(n not in wanted for n in names):
                    reasons.add('wrong_tool_selection')
                if 'wrong_order' in delta:
                    reasons.add('wrong_order')
            elif len(parsed) < len(expected):
                reasons.add('missing_tool_call')
            # Check arguments against the matching tool, not its shifted position.
            # A duplicated valid call is extra; it does not have invalid arguments.
            for call in parsed:
                if isinstance(call, dict):
                    targets = [c for c in expected if c['tool'] == call.get('tool')]
                    if targets and all(differences(call.get('arguments'), c['arguments']) for c in targets):
                        reasons.add('wrong_tool_arguments')
    return {
        'task_id': task['id'], 'model_id': result.get('model_id'),
        'technical_success': technical,
        'strict_correct': bool(technical and format_ok and content_ok),
        'format_correct': bool(format_ok), 'content_correct': bool(content_ok),
        'hit_max_new_tokens': bool(hit),
        # Hitting the cap alone cannot confirm semantic truncation.
        'truncated': None if hit else False,
        'truncation_evidence': 'token_limit_only' if hit else 'none',
        'failure_reasons': sorted(reasons),
    }
