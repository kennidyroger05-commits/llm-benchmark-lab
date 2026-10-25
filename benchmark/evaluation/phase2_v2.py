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

VERSION = 'phase2-evaluation-v2'


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


def build_report(evaluations, **options):
    rows = list(evaluations)
    report = v1_report(rows, **options)
    from benchmark.tools.catalog_v2 import public_catalog_sha256
    report.update(evaluation_version=VERSION, protocol_version='phase2_v2',
                  sandbox_mode='restricted_ast_interpreter', catalog_hash=public_catalog_sha256(),
                  evaluation_complete=not any(r.get('evaluation_pending') for r in rows),
                  publication_ready=not any(r.get('evaluation_pending') for r in rows))
    for source, projected in zip(sorted(rows, key=lambda r: int(r['task_id'][1:])), report['task_evaluations']):
        projected['evaluation_pending'] = source.get('evaluation_pending', False)
    report.pop('structural_hash')
    report['structural_hash'] = hashlib.sha256(finite_json_bytes(report)).hexdigest()
    return report
