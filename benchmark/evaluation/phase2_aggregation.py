"""Run/model summaries, category breakdowns and explicitly paired RAW comparisons."""
from collections import Counter, defaultdict

from .phase2 import CATEGORIES

_BOOLEAN_METRICS = (
    'technical_success', 'protocol_correct', 'task_correct', 'strict_correct',
    'content_correct', 'format_correct', 'direct_answer',
)
_TOOL_SUMS = (
    'tool_calls_total', 'tool_calls_dispatched', 'tool_calls_successful',
    'tool_calls_failed', 'invalid', 'duplicates', 'switches', 'unnecessary_tool_calls',
    'tool_calls_blocked',
)


def _summary(rows):
    """All tasks are main denominators; null is not success and is counted too."""
    total = len(rows)
    result = {'total_tasks': total}
    for key in _BOOLEAN_METRICS:
        count = sum(row[key] is True for row in rows)
        result[key + '_count'] = count
        result[key + '_rate'] = count / total if total else None
        result[key + '_null_count'] = sum(row[key] is None for row in rows)
    applicable = [r for r in rows if r['dataset_tools_contract_correct'] is not None]
    count = sum(r['dataset_tools_contract_correct'] is True for r in applicable)
    result['dataset_tools_contract_correct_count'] = count
    result['dataset_tools_contract_correct_denominator'] = len(applicable)
    result['dataset_tools_contract_correct_rate'] = count / len(applicable) if applicable else None
    for key in _TOOL_SUMS:
        result[key] = sum(row[key] for row in rows)
    result['tool_calls_mean'] = result['tool_calls_total'] / total if total else None
    defined = [r['calls_over_minimum'] for r in rows if r['calls_over_minimum'] is not None]
    result['calls_over_minimum_total'] = sum(defined) if defined else None
    result['calls_over_minimum_defined_tasks'] = len(defined)
    counts = Counter(r['tool_selection_class'] for r in rows)
    result['selection_class_counts'] = {key: counts[key] for key in (
        'optimal', 'acceptable', 'inefficient', 'invalid', None)}
    # JSON object keys must be strings; direct/no-assistance is explicitly unclassified.
    result['selection_class_counts']['unclassified'] = result['selection_class_counts'].pop(None)
    result['selection_class_rates'] = {
        key: count / total if total else None for key, count in result['selection_class_counts'].items()}
    return result


def aggregate_run(evaluations, *, model_id, run_id=None):
    """Refuse mixed/duplicate tasks instead of silently averaging separate runs."""
    rows = list(evaluations)
    if len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate task in run.')
    if any(r['model_id'] not in (None, model_id) or r['run_id'] != run_id for r in rows):
        raise ValueError('Mixed model/run evaluations.')
    return {
        'model_id': model_id, 'run_id': run_id, **_summary(rows),
        'categories': {category: _summary([r for r in rows if r['category'] == category])
                       for category in CATEGORIES.values()},
    }


def aggregate_runs(evaluations):
    """Separate each (model_id, run_id), with stable sorting and no pooling repeats."""
    groups = defaultdict(list)
    for row in evaluations:
        if not row['model_id']:
            raise ValueError('model_id required for grouped aggregation.')
        groups[(row['model_id'], row['run_id'])].append(row)
    return [aggregate_run(groups[key], model_id=key[0], run_id=key[1])
            for key in sorted(groups, key=lambda k: (k[0], k[1] or ''))]


def compare_phase1(evaluations, phase1_evaluations):
    """Paired strict comparison; RAW strict is never relabelled task_correct.

    T5/T6 compare RAW plans versus TOOLS execution; these are different observed
    contracts, so the aggregate difference is descriptive, not a causal estimate.
    All paired tasks must match. RAW=0 leaves relative percent gain undefined.
    """
    current, raw = list(evaluations), list(phase1_evaluations)
    if (len({r['task_id'] for r in current}) != len(current)
            or len({r['task_id'] for r in raw}) != len(raw)
            or {r['task_id'] for r in current} != {r['task_id'] for r in raw}):
        raise ValueError('Comparison requires identical unique paired task IDs.')
    if any(type(r['strict_correct']) is not bool for r in raw):
        raise ValueError('RAW strict correctness must be boolean.')
    models = {r.get('model_id') for r in current + raw} - {None}
    if len(models) > 1 or len({r['run_id'] for r in current}) > 1:
        raise ValueError('Comparison must refer to one model/run.')
    prior = {r['task_id']: r for r in raw}
    pairs = [{'task_id': r['task_id'], 'phase1_strict_correct': prior[r['task_id']]['strict_correct'],
              'phase2_strict_correct': r['strict_correct'], 'phase2_task_correct': r['task_correct'],
              'contract_changed': r['task_id'] in {'T5', 'T6'}}
             for r in sorted(current, key=lambda r: int(r['task_id'][1:]))]
    raw_count = sum(p['phase1_strict_correct'] for p in pairs)
    now_count = sum(p['phase2_strict_correct'] for p in pairs)
    return {'tasks': pairs, 'paired_tasks': len(pairs),
            'strict_gain_absolute_pp': 100 * (now_count - raw_count) / len(pairs) if pairs else None,
            'strict_gain_relative_percent': 100 * (now_count - raw_count) / raw_count if raw_count else None,
            'interpretation': 'Descriptive paired comparison; RAW plans and TOOLS actions differ for T5/T6.'}
