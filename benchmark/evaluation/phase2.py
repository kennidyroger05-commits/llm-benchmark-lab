"""Private deterministic Phase 2 scoring, independent from model execution.

Only final answer values are evaluated. No usable final means false format and
content. Technical success is copied from execution evidence, never from parsing.
Domain evidence is the trusted controller trace, not a plan printed by a model:
replay verifies committed results, dependencies and final state against a fresh
private session. Refused auxiliary/recovery calls can leave task_correct intact.
No prompt, fixture, expected value or private state is returned in evaluations.
"""
import ast

from benchmark.orchestration.dispatcher import DispatchContext, dispatch_tool
from benchmark.tools.domain_tools import DomainSession
from benchmark.tools.sandbox import SandboxBackend, SandboxError
from benchmark.tools._sandbox_worker import json_bytes
from .code_evaluation import contract_cases
from .parsing import differences, parse_json, shape_errors, unfence
from .task_evaluators import evaluate_task
from .phase2_efficiency import DOMAIN_TOOLS, evaluate_tools, typed_key

EVALUATION_VERSION = 'phase2-evaluation-v1'
CATEGORIES = {
    'Cálculo Financeiro / Lógica Matemática Exata': 'financial/math',
    'Manipulação e Filtragem de Dados': 'data',
    'Intenção e Roteamento de Múltiplas Ferramentas': 'multi-tool/domain',
    'Programação e Automação': 'programming',
    'Síntese e Formatação Estruturada Restrita': 'structured synthesis',
}


def _session(task):
    return DomainSession(tool_catalog=task['tool_catalog'], tool_fixtures=task['tool_fixtures'])


def domain_contract(task, trace):
    """Return functional-state and strict-trace checks, without leaking projections.

    Replay only domain tools; general tools cannot execute business actions.
    Compare every observed domain envelope, even refusals/auxiliary queries.
    Snapshot equality includes consumed mandatory queries and committed actions.
    """
    expected_context = DispatchContext(domain_session=_session(task))
    for call in task['expected_tool_sequence']:
        dispatch_tool(call['tool'], call['arguments'], expected_context)
    expected_trace = expected_context.trace.to_dicts()
    if any(not entry['envelope']['ok'] for entry in expected_trace):
        raise ValueError('Invalid private domain contract.')
    actual_context = DispatchContext(domain_session=_session(task))
    verified = True
    for entry in trace:
        if entry['tool'] in DOMAIN_TOOLS:
            envelope = dispatch_tool(entry['tool'], entry['arguments'], actual_context)
            verified &= typed_key(envelope) == typed_key(entry['envelope'])
    state_correct = verified and (typed_key(actual_context.domain_session.snapshot()) ==
                                 typed_key(expected_context.domain_session.snapshot()))
    # Only these public observations belong to the strict dataset trace.
    project = lambda entries: [
        (e['tool'], typed_key(e['arguments']), typed_key(e['envelope'])) for e in entries]
    exact = bool(state_correct and project(trace) == project(expected_trace))
    return bool(state_correct), exact


def _code_answer(answer, task, backend):
    """Frozen Phase 1 function/signature/cases, evaluated only in safe sandbox.

    Reuse contract_cases (including evaluation-only edge cases), strict types,
    exception and mutation checks. The Phase 1 local process is not a fallback.
    Sandbox policy refusal is an evaluation diagnostic, not a provider failure.
    """
    stripped = unfence(answer, 'python')
    code = stripped if stripped is not None else answer
    try:
        tree = ast.parse(code)
        compile(tree, '<response>', 'exec')
    except (SyntaxError, ValueError, RecursionError):
        return False, False, 'invalid_python'
    name = task['expected_answer']['function']
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != name:
        return False, False, 'wrong_function'
    fn = functions[0]
    if (len(fn.args.posonlyargs) + len(fn.args.args) != 1 or fn.args.vararg
            or fn.args.kwarg or fn.args.kwonlyargs or fn.decorator_list):
        return False, False, 'wrong_function'
    format_ok = stripped is None
    if not isinstance(backend, SandboxBackend):
        return format_ok, False, 'safe_backend_unavailable'
    cases = []
    for case in contract_cases(task):
        cases.append({'args': [case['input']], 'kwargs': {}, **{
            key: case[key] for key in ('expected', 'raises') if key in case}})
    try:
        if not backend.is_available():
            return format_ok, False, 'safe_backend_unavailable'
        result = backend.run_python(code=code, function=name, mode='test_runner',
                                    cases=cases, timeout=2).result
        return format_ok, result['passed'] is True, None
    except SandboxError as error:
        diagnostic = ('evaluation_infrastructure_failure' if error.origin == 'backend_infrastructure'
                      else 'code_' + (error.code if error.code in {
                          'invalid_data', 'resource_limit', 'tool_timeout', 'execution_error'
                      } else 'execution_error'))
        return format_ok, False, diagnostic
    except Exception:
        return format_ok, False, 'evaluation_infrastructure_failure'


def _answer_scores(task, answer, backend):
    if task['id'] in {'T5', 'T6'}:
        expected = task['final_expected_answer']
        value = answer
        if isinstance(answer, str):
            # Same secondary JSON/fence content diagnostic as Phase 1; the
            # native-JSON requirement still rejects the string as final format.
            try:
                stripped = unfence(answer, 'json')
                value = parse_json(stripped if stripped is not None else answer)
            except (ValueError, TypeError):
                pass
        return (not shape_errors(answer, expected), not differences(value, expected), None)
    kind = task['evaluation_type']
    if kind in {'code', 'structured_text'} and not isinstance(answer, str):
        return False, False, None
    if kind == 'code':
        return _code_answer(answer, task, backend)
    # Serialize only native JSON answers. Strings are kept as diagnostic text,
    # but never receive format credit for doubly-encoded JSON.
    response = answer if isinstance(answer, str) else json_bytes(answer).decode('utf-8')
    verdict = evaluate_task(task, {'response': response, 'status': 'success'})
    format_ok = verdict['format_correct']
    if kind != 'structured_text' and isinstance(answer, str):
        format_ok = False
    return format_ok, verdict['content_correct'], None


def evaluate_episode(task, episode, *, sandbox_backend=None):
    """Score a trusted EpisodeResult; efficiency never enters task correctness."""
    if episode.metadata.get('task_id', task['id']) != task['id']:
        raise ValueError('Task and episode IDs differ.')
    domain = task['id'] in {'T5', 'T6'}
    final = episode.termination_reason == 'final_answer'
    format_ok, content_ok, diagnostic = (
        _answer_scores(task, episode.final_answer, sandbox_backend) if final else (False, False, None))
    state_ok, sequence_ok = domain_contract(task, episode.tool_trace) if domain else (True, None)
    functional = bool(final and episode.protocol_correct is True and format_ok and content_ok and state_ok)
    strict = bool(functional and episode.technical_success and (not domain or sequence_ok))
    tools = evaluate_tools(task['id'], episode, sequence_correct=sequence_ok,
                           domain_actions_correct=state_ok)
    return {
        'evaluation_version': EVALUATION_VERSION, 'task_id': task['id'],
        'category': CATEGORIES[task['category']],
        'model_id': episode.metadata.get('model_id'), 'run_id': episode.metadata.get('run_id'),
        'technical_success': episode.technical_success,
        'protocol_correct': episode.protocol_correct,
        'task_correct': functional, 'strict_correct': strict,
        'content_correct': bool(content_ok), 'format_correct': bool(format_ok),
        'dataset_tools_contract_correct': strict if domain else None,
        'direct_answer': episode.direct_answer,
        'termination_reason': episode.termination_reason,
        'infrastructure_failure': episode.infrastructure_failure,
        'domain_actions_correct': state_ok if domain else None,
        'evaluation_error': diagnostic, **tools,
    }
