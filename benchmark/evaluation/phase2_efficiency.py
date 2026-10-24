"""Private/offline policy v1, transcribed from protocol sections 9–10.

Never import this module from prompt construction. Sets describe assistance, not
correct answers. minimum_calls is a fixed assisted baseline, not a global optimum.
Unnecessary calls are only the lower bound specified by section 9: identical
successful pure repeats, policy-blocked actions and refused domain duplicates.
"""
from dataclasses import dataclass

from benchmark.orchestration.protocol import ProtocolError, ToolCall, parse_model_message
from benchmark.tools.catalog import get_public_catalog
from benchmark.tools.schemas import StructuralValidationError, UnknownToolError, validate_tool_arguments

POLICY_VERSION = 'phase2-efficiency-v1'
P, D, B, I = 'python_executor', 'decimal_calculator', 'calculator_basic', 'integer_optimizer'
C, J, T = 'csv_query', 'json_query', 'table_transform'
S, R, V, F = 'python_syntax_check', 'python_test_runner', 'json_validator', 'text_format_validator'
DOMAIN_TOOLS = frozenset(t.name for t in get_public_catalog()) - {
    P, D, B, I, C, J, T, S, R, V, F}
PURE_TOOLS = frozenset(t.name for t in get_public_catalog()) - DOMAIN_TOOLS


@dataclass(frozen=True)
class EfficiencyPolicy:
    capable: frozenset
    optimal: tuple[frozenset, ...]
    acceptable: tuple[frozenset, ...]
    minimum: int
    auxiliary_minimum: int | None = None


def _policy(capable, optimal, acceptable, minimum=1, auxiliary_minimum=None):
    return EfficiencyPolicy(frozenset(capable), (frozenset(optimal),),
                            tuple(frozenset(s) for s in acceptable), minimum, auxiliary_minimum)


# No fixture values, answers or inferred/model-dependent baselines are stored.
_A5 = {'lookup_order', 'check_return', 'create_return'}
_A6 = {'get_stock', 'reserve_stock', 'charge', 'release_stock', 'log_failure'}
_PRIVATE_POLICIES = {
    'T1': _policy({D, B, P, V}, {D}, [{P}, {B}, {D, V}]),
    'T2': _policy({I, B, P, V}, {I}, [{P}, {I, V}]),
    'T3': _policy({C, T, J, P, V}, {C}, [{T}, {J}, {P}, {C, V}]),
    'T4': _policy({J, T, P, V}, {J}, [{T}, {P}, {J, V}]),
    'T5': _policy(_A5 | {'order_workflow_query', P, V}, _A5,
                  [_A5 | {x} for x in ('order_workflow_query', P, V)], 4, 5),
    'T6': _policy(_A6 | {'inventory_query', P, B, V}, _A6,
                  [_A6 | {x} for x in ('inventory_query', P, B, V)], 5, 6),
    'T7': _policy({P, S, R}, {R}, [{S}, {P}, {S, R}]),
    'T8': _policy({P, S, R}, {R}, [{S}, {P}, {S, R}]),
    'T9': _policy({P, J, T, V}, {P}, [{J}, {T}, {V}, {P, V}]),
    'T10': _policy({P, J, T, F}, {P}, [{J}, {T}, {F}, {P, F}]),
}


def typed_key(value):
    """Canonical equality: preserve scalar types/array order, ignore object order."""
    if type(value) is dict:
        return ('object', tuple((k, typed_key(value[k])) for k in sorted(value)))
    if type(value) is list:
        return ('array', tuple(typed_key(v) for v in value))
    return (type(value).__name__, value)


def tool_requests(episode):
    """Invalid protocol is not a call; the ninth valid blocked call still is."""
    requests = []
    for raw in episode.model_messages:
        try:
            message = parse_model_message(raw)
        except ProtocolError:
            continue
        if isinstance(message, ToolCall):
            requests.append(message)
    return requests


def evaluate_tools(task_id, episode, *, sequence_correct=None, domain_actions_correct=False):
    """Counters are structural, selection labels do not depend on answer accuracy.

    duplicates/switches cover all requests, including the blocked ninth.
    successful/failed/invalid/unnecessary cover dispatches only. selection_valid
    means at least one dispatched capable tool with valid schema, not success.
    calls_over_minimum uses total requests (frozen protocol), not successful calls.
    Non-domain zero-tool episodes have null class/minimum regardless of accuracy.
    """
    policy = _PRIVATE_POLICIES[task_id]
    requests = tool_requests(episode)
    trace = episode.tool_trace
    if len(trace) != episode.tool_calls_dispatched or len(trace) > len(requests):
        raise ValueError('Inconsistent episode dispatch evidence.')
    for entry, request in zip(trace, requests):
        if (not entry['dispatched'] or entry['tool'] != request.tool
                or typed_key(entry['arguments']) != typed_key(request.arguments)):
            raise ValueError('Inconsistent episode dispatch evidence.')
    seen, duplicates = set(), 0
    for request in requests:
        key = (request.tool, typed_key(request.arguments))
        duplicates += key in seen
        seen.add(key)
    unnecessary, pure_returns = 0, set()
    selected = False
    for entry in trace:
        name, args, envelope = entry['tool'], entry['arguments'], entry['envelope']
        try:
            validate_tool_arguments(name, args)
            selected |= name in policy.capable
        except (StructuralValidationError, UnknownToolError):
            pass
        needless = entry['error_code'] in {'policy_violation', 'duplicate_call'}
        if name in PURE_TOOLS and envelope['ok']:
            key = (name, typed_key(args), typed_key(envelope['result']))
            needless |= key in pure_returns
            pure_returns.add(key)
        unnecessary += bool(needless)
    names = frozenset(entry['tool'] for entry in trace)
    domain = task_id in {'T5', 'T6'}
    minimum = (policy.auxiliary_minimum if domain and any(
        alternative <= names for alternative in policy.acceptable) else policy.minimum)
    errors = [entry['error_code'] for entry in trace if not entry['envelope']['ok']]
    if not requests and not domain:
        selection_class, minimum = None, None
    elif not selected or any(code in {'unknown_tool', 'invalid_arguments', 'policy_violation',
                                      'precondition_failed'} for code in errors):
        selection_class = 'invalid'
    elif (names in policy.optimal and len(trace) == len(requests) == minimum and not errors
          and not duplicates and not unnecessary and (not domain or domain_actions_correct)):
        selection_class = 'optimal'
    elif (any(alternative <= names for alternative in policy.optimal + policy.acceptable)
          and names <= policy.capable and not errors and not unnecessary
          and len(trace) == len(requests)):
        selection_class = 'acceptable'
    else:
        selection_class = 'inefficient'
    return {
        'tool_calls_total': len(requests), 'tool_calls_dispatched': len(trace),
        'tool_calls_successful': sum(e['envelope']['ok'] for e in trace),
        'tool_calls_failed': sum(not e['envelope']['ok'] for e in trace),
        'duplicates': duplicates,
        'invalid': sum(e['error_code'] in {'unknown_tool', 'invalid_arguments'} for e in trace),
        'switches': sum(a.tool != b.tool for a, b in zip(requests, requests[1:])),
        'sequence_correct': sequence_correct, 'unnecessary_tool_calls': unnecessary,
        'minimum_required': minimum,
        'calls_over_minimum': None if minimum is None else max(0, len(requests) - minimum),
        'calls_below_minimum': None if minimum is None else max(0, minimum - len(requests)),
        'selection_valid': bool(selected), 'tool_selection_class': selection_class,
        'tool_calls_blocked': len(requests) - len(trace),
    }
