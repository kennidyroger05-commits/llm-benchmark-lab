"""Execute contract tests in a resource-limited, restricted Python subprocess."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile

# These evaluation-only cases must never be included in fine-tuning corpora.
def contract_cases(task):
    """Extend dataset cases with edge cases from the frozen task contracts."""
    cases = list(task['expected_answer']['test_cases'])
    if task['id'] == 'T7':
        cases.extend([
            {'input': [{'id': 'z', 'seq': 0, 'status': 'settled', 'amount_centavos': 0},
                       {'id': 'A', 'seq': 9, 'status': 'settled', 'amount_centavos': -5},
                       {'id': 'A', 'seq': 1, 'status': 'void', 'amount_centavos': 80}],
             'expected': {'net_centavos': -5, 'ids': ['A', 'z']}},
            {'input': [{'id': 'a', 'seq': 0, 'status': 'void', 'amount_centavos': -1}],
             'expected': {'net_centavos': 0, 'ids': []}},
        ])
    else:
        for text, expected in [('', []), ('\t\n ', []), ('001,-0,0,01..03', [1, 0, 2, 3]),
                               ('2..2', [2]), ('0..-2,-1..1', [0, -1, -2, 1]),
                               (' 1\t,\n2 ', [1, 2]), ('0..999', list(range(1000)))]:
            cases.append({'input': text, 'expected': expected})
        for text in ['1,', ',1', ',', '+1..2', '1..+2', '1..', '..2', '1.2',
                     '--2', '1\t..2', '1.. 2', '１２', '١', '1e2', '1...2']:
            cases.append({'input': text, 'raises': 'ValueError'})
    return cases


def evaluate_code(code, task, timeout=2):
    """Return compilation/contract status; no model code runs in this process."""
    try:
        tree = ast.parse(code)
        compile(tree, '<response>', 'exec')
    except (SyntaxError, ValueError):
        return False, False, ['invalid_python']
    name = task['expected_answer']['function']
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != name:
        return False, False, ['wrong_function']
    fn = functions[0]
    if (len(fn.args.posonlyargs) + len(fn.args.args) != 1 or fn.args.vararg or
            fn.args.kwarg or fn.args.kwonlyargs or fn.decorator_list):
        return False, False, ['wrong_function']
    payload = {'code': code, 'function': name, 'cases': contract_cases(task)}
    worker = Path(__file__).with_name('_code_worker.py')
    with tempfile.TemporaryDirectory(prefix='benchmark-code-') as directory:
        try:
            result = subprocess.run(
                [sys.executable, '-I', '-S', str(worker)], input=json.dumps(payload),
                text=True, capture_output=True, timeout=timeout, cwd=directory,
                env={'PYTHONHASHSEED': '0'},
            )
        except subprocess.TimeoutExpired:
            return True, False, ['code_timeout']
    if result.returncode:
        return True, False, ['code_resource_or_process_error']
    try:
        verdict = json.loads(result.stdout)
    except ValueError:
        return True, False, ['code_process_error']
    return True, verdict['passed'], verdict['failure_reasons']
