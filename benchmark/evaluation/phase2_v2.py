"""V2 evaluation adapter: retain frozen answer contracts, never grade refusals wrong.

Code outside the AST subset/budget is PENDING external isolated evaluation.
No host Python/Docker fallback; official accuracy publication is blocked while
any task is pending. This does not change the benchmark's accepted Python syntax.
"""
import ast
from copy import deepcopy
from .phase2 import evaluate_episode as v1_evaluate, domain_contract, CATEGORIES
from .phase2_efficiency import evaluate_tools
from .code_evaluation import contract_cases
from .parsing import unfence
from .phase2_export import build_report as v1_report
from benchmark.tools.restricted_python import RestrictedASTBackend
from benchmark.tools.sandbox import SandboxError
from benchmark.orchestration.tool_trace import finite_json_bytes
import hashlib
import json
from pathlib import Path

VERSION = 'phase2-evaluation-v2'


def collection_valid(evaluations, status):
    """Operational validity only; ordinary protocol/content/budget stops survive.

    Context refusals and correctly enforced timeouts are not infrastructure
    failures. Incomplete collection still blocks independently of their metrics.
    """
    rows = list(evaluations)
    return (status == 'complete' and
            {r['task_id'] for r in rows} == {f'T{i}' for i in range(1, 11)} and
            not any(r.get('infrastructure_failure') or
                    (r.get('technical_success') is False and
                     r.get('termination_reason') not in {'context_limit_exceeded', 'timeout'})
                    for r in rows))


def evaluate_episode(task, episode, *, sandbox_backend=None):
    if task['evaluation_type'] != 'code' or episode.termination_reason != 'final_answer':
        result = v1_evaluate(task, episode)
        result.update(evaluation_version=VERSION, evaluation_pending=False)
        return result
    # Reuse non-code execution/selection evidence without passing the program to
    # the legacy code evaluator (which uses CPython compilation for format).
    proxy = {**task, 'evaluation_type': 'json_schema', 'expected_answer': {}}
    sanitized = deepcopy(episode); sanitized.final_answer = {}
    result = v1_evaluate(proxy, sanitized)
    answer = episode.final_answer
    format_ok, content, diagnostic, pending = False, False, None, False
    if type(answer) is str:
        stripped = unfence(answer, 'python')
        code = stripped if stripped is not None else answer
        try:
            tree = ast.parse(code)
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            name = task['expected_answer']['function']
            signature = functions[0].args if len(functions) == 1 else None
            shape = (signature is not None and functions[0].name == name and
                len(signature.posonlyargs) + len(signature.args) == 1 and not signature.vararg
                and not signature.kwarg and not signature.kwonlyargs and not functions[0].decorator_list)
            format_ok = bool(shape and stripped is None)
            if shape:
                if any(isinstance(n, ast.Div) for n in ast.walk(tree)):
                    raise SandboxError('policy_violation', 'sandbox_policy')
                cases = [{'args': [case['input']], 'kwargs': {}, **{k: case[k] for k in ('expected', 'raises') if k in case}}
                         for case in contract_cases(task)]
                verdict = RestrictedASTBackend().run_python(code=code, function=name, mode='test_runner', cases=cases).result
                content = verdict['passed']
        except SandboxError:
            # Unsupported syntax, budgets or an unrepresentable test result
            # cannot disprove equivalence under the original full-Python contract.
            content, pending, diagnostic = None, True, 'pending_isolated_code_evaluation'
        except (SyntaxError, ValueError): diagnostic = 'invalid_python'
        except (RecursionError, MemoryError):
            content, pending, diagnostic = None, True, 'pending_isolated_code_evaluation'
    functional = (None if pending and episode.protocol_correct is True and format_ok else
                  bool(episode.protocol_correct is True and format_ok and content))
    result.update(evaluation_version=VERSION, evaluation_pending=pending,
                  format_correct=format_ok, content_correct=content,
                  task_correct=functional,
                  strict_correct=None if functional is None and episode.technical_success else
                                 bool(functional and episode.technical_success), evaluation_error=diagnostic)
    return result


def build_report(evaluations, *, collection_status=None, collection_failure=None, **options):
    """Completion means ten visits; publication also requires intact collection.

    Wrong answers/invalid model protocol remain legitimate observations. Missing
    episodes, provider/infra failures and pending isolated grading do not. This
    operational guard does not modify any functional correctness metric.
    """
    rows = list(evaluations)
    report = v1_report(rows, **options)
    from benchmark.tools.catalog_v2 import public_catalog_sha256
    complete_ids = {r['task_id'] for r in rows} == {f'T{i}' for i in range(1, 11)}
    source = options.get('source_raw_path')
    raw = json.loads(Path(source).read_text()) if source is not None else {}
    statuses = [s for s in (collection_status, raw.get('status'), raw.get('collection_status')) if s is not None]
    complete = complete_ids and all(s == 'complete' for s in statuses)
    pending = any(r.get('evaluation_pending') or
                  r.get('evaluation_error') == 'pending_isolated_code_evaluation' for r in rows)
    infrastructure_failed = not collection_valid(rows, 'complete')
    # Cross-check RAW when supplied rather than trusting just an evaluation flag.
    infrastructure_failed |= any(record.get('episode', {}).get('infrastructure_failure') or
        (record.get('episode', {}).get('technical_success') is False and
         record.get('episode', {}).get('termination_reason') not in {'context_limit_exceeded', 'timeout'})
        for record in raw.get('tasks', []))
    collection_failed = bool(collection_failure or raw.get('failure'))
    experimental_valid = complete and not infrastructure_failed and not collection_failed
    report.update(evaluation_version=VERSION, protocol_version='phase2_v2',
                  sandbox_mode='restricted_ast_interpreter', catalog_hash=public_catalog_sha256(),
                  collection_status=collection_status or raw.get('status') or ('complete' if complete_ids else 'incomplete'),
                  collection_failure=collection_failure or raw.get('failure'),
                  experimental_valid=experimental_valid,
                  evaluation_complete=not pending,
                  publication_ready=experimental_valid and not pending)
    for source, projected in zip(sorted(rows, key=lambda r: int(r['task_id'][1:])), report['task_evaluations']):
        projected['evaluation_pending'] = source.get('evaluation_pending', False)
    report.pop('structural_hash')
    report['structural_hash'] = hashlib.sha256(finite_json_bytes(report)).hexdigest()
    return report
