"""Functional, hostile and integration tests. No Docker or model downloads."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark.tools.restricted_python import RestrictedASTBackend, Interpreter, Limits
from benchmark.tools.sandbox import SandboxError
from benchmark.tools.math_tools import ToolExecutionError
from benchmark.tools.code_tools_v2 import run_python_executor, run_python_test_runner, run_python_syntax_check
from benchmark.tools.catalog import public_catalog_sha256 as v1_hash
from benchmark.tools.catalog_v2 import public_catalog_sha256, get_public_catalog
from benchmark.orchestration.revision_v2 import run_episode, REGISTRY
from benchmark.orchestration.phase2_preflight_v2 import run_preflight, compatible_python, frozen_integrity
from benchmark.orchestration.phase2_runner_v2 import run_phase2, V2Supervisor
from benchmark.evaluation.phase2_v2 import evaluate_episode, build_report
from benchmark.evaluation.phase2_export import report_bytes
from test_phase2_evaluation import simple_task, domain_task, make_episode
from test_phase2_operations import FakeSupervisor


def execute(code, data=None):
    return run_python_executor({'code': code, 'data': data}, RestrictedASTBackend())['value']


HOSTILE = {
    'import_os': 'import os\ndef main(data): return os.environ',
    'import_socket': 'import socket\ndef main(data): return 1',
    'from_import': 'from os import environ\ndef main(data): return 1',
    'open': 'def main(data): return open("/etc/passwd")',
    'dunder_import': 'def main(data): return __import__("os")',
    'globals': 'def main(data): return globals()',
    'locals': 'def main(data): return locals()',
    'vars': 'def main(data): return vars(data)',
    'getattr': 'def main(data): return getattr(data, "__class__")',
    'setattr': 'def main(data): return setattr(data, "x", 1)',
    'class_attr': 'def main(data): return data.__class__',
    'dict_attr': 'def main(data): return data.__dict__',
    'dunder_traversal': 'def main(data): return ().__class__.__bases__[0].__subclasses__()',
    'subprocess': 'def main(data): return subprocess.run(["ls"])',
    'socket': 'def main(data): return socket.socket()',
    'pathlib': 'def main(data): return pathlib.Path("/etc/passwd").read_text()',
    'env': 'def main(data): return os.environ["HF_TOKEN"]',
    'thread': 'def main(data): return threading.Thread()',
    'process': 'def main(data): return multiprocessing.Process()',
    'ctypes': 'def main(data): return ctypes.CDLL(None)',
    'pickle': 'def main(data): return pickle.loads(data)',
    'clock': 'def main(data): return time.time()',
    'random': 'def main(data): return random.random()',
    'eval': 'def main(data): return eval(data)',
    'exec': 'def main(data): exec(data)',
    'compile': 'def main(data): return compile(data,"x","exec")',
    'input': 'def main(data): return input()',
    'class': 'class X: pass\ndef main(data): return 1',
    'lambda': 'def main(data): return (lambda: 1)()',
    'yield': 'def main(data): yield 1',
    'async': 'async def main(data): return 1',
    'with': 'def main(data):\n with data: pass',
    'try': 'def main(data):\n try: return 1\n except: return 2',
    'raise_arbitrary': 'def main(data): raise RuntimeError("x")',
    'decorator': '@open\ndef main(data): return 1',
    'while': 'def main(data):\n while True: pass',
    'huge_range': 'def main(data):\n for x in range(1000000000): pass\n return 1',
    'explosive_list': 'def main(data): return [0]*1000000000',
    'explosive_string': 'def main(data): return "x"*1000000000',
    'huge_integer': 'def main(data): return 2**1000000000',
    'recursion': 'def main(data): return main(data)',
    'deep_ast': 'def main(data): return ' + '['*80 + '0' + ']'*80,
    'cycle': 'def main(data):\n x=[]\n x.append(x)\n return x',
}


class HostileTests(unittest.TestCase):
    def reject(self, code):
        with self.assertRaises(SandboxError) as caught:
            execute(code, {})
        self.assertIn(caught.exception.code, {'policy_violation', 'resource_limit', 'invalid_data'})
        self.assertNotIn('TEST_HF_SECRET', str(caught.exception))

# Separate tests give an explicit hostile-test count in the report.
for name, code in HOSTILE.items():
    setattr(HostileTests, 'test_' + name, lambda self, source=code: self.reject(source))


class FunctionalTests(unittest.TestCase):
    def test_arithmetic(self): self.assertEqual(execute('def main(data): return (7+3)*2-4//2'), 18)
    def test_json_data(self): self.assertEqual(execute('def main(data): return data', {'x': [1, True, None]}), {'x': [1, True, None]})
    def test_decimal_passthrough(self): self.assertEqual(execute('def main(data): return data', Decimal('0.123456789')), Decimal('0.123456789'))
    def test_division(self): self.assertEqual(execute('def main(data): return 7/2'), Decimal('3.5'))
    def test_loop_condition(self):
        self.assertEqual(execute('def main(data):\n out=[]\n for n in data:\n  if n%2==0:\n   out.append(n*2)\n return out', [1,2,3,4]), [4,8])
    def test_index_slice(self): self.assertEqual(execute('def main(data): return data[1:3]', [1,2,3,4]), [2,3])
    def test_dict_mutation_local(self): self.assertEqual(execute('def main(data):\n out={}\n out["x"]=data[0]\n return out', [3]), {'x':3})
    def test_helper(self): self.assertEqual(execute('def helper(x): return x+1\ndef main(data): return helper(data)', 3), 4)
    def test_break_continue(self):
        self.assertEqual(execute('def main(data):\n x=0\n for n in range(10):\n  if n==2: continue\n  if n==4: break\n  x+=n\n return x'), 4)
    def test_methods(self):
        code='def main(data):\n values=data.strip().split(",")\n out=[]\n for x in values:\n  out.append(int(x))\n return sorted(out, reverse=True)'
        self.assertEqual(execute(code, ' 1,3,2 '), [3,2,1])
    def test_live_list_iteration_matches_python(self):
        self.assertEqual(execute('def main(data):\n xs=[1,2]\n for n in xs:\n  xs.pop()\n return xs'), [1])
    def test_augmented_list_preserves_aliases(self):
        self.assertEqual(execute('def main(data):\n a=[1]\n b=a\n a+=[2]\n return [a,b]'), [[1,2],[1,2]])
    def test_iterator_remains_lazy(self):
        self.assertEqual(execute('def main(data):\n xs=[1]\n it=enumerate(xs)\n xs.append(2)\n out=[]\n for i,x in it:\n  out.append(x)\n return out'), [1,2])
    def test_iterator_not_coerced_to_json_list(self):
        self.assertEqual(execute('def main(data):\n a,b=range(2)\n return [a,b]'), [0,1])
        self.assertRaises(SandboxError, execute, 'def main(data): return range(3)')
    def test_function_shadowing_refused_before_interpretation(self):
        self.assertRaises(SandboxError, execute, 'def f(x): return x+1\ndef main(data):\n f=2\n return f(data)')
    def test_default_scalar_constructors(self):
        self.assertEqual(execute('def main(data): return [int(),str(),bool()]'), [0,'',False])
    def test_dictionary_from_lazy_zip(self):
        self.assertEqual(execute('def main(data): return dict(zip(["a","b"],[1,2]))'), {'a':1,'b':2})
    def test_min_without_arguments_is_typeerror(self):
        result=RestrictedASTBackend().run_python(code='def f(x): return min()',function='f',mode='test_runner',cases=[{'args':[1],'kwargs':{},'raises':'TypeError'}]).result
        self.assertTrue(result['passed'])
    def test_builtins(self): self.assertEqual(execute('def main(data): return [len(data), sum(data), min(data), max(data), abs(-4), all(data), any(data)]', [1,2,3]), [3,6,1,3,4,True,True])
    def test_unpack_zip_enumerate(self): self.assertEqual(execute('def main(data):\n out=[]\n for i,x in enumerate(data):\n  out.append(i+x)\n return out', [5,6]), [5,7])
    def test_t7_like_synthetic(self):
        code='def main(data):\n latest={}\n for row in data:\n  key=row["key"]\n  if key not in latest or row["v"]>latest[key]["v"]:\n   latest[key]=row\n total=0\n for key in latest:\n  if latest[key]["active"]:\n   total+=latest[key]["amount"]\n return total'
        rows=[{'key':'toy','v':2,'active':True,'amount':7}, {'key':'toy','v':1,'active':False,'amount':99}]
        self.assertEqual(execute(code, rows), 7)
    def test_t8_like_synthetic(self):
        code='def main(data):\n out=[]\n for token in data.split(","):\n  bounds=token.strip().split(":")\n  a=int(bounds[0])\n  b=int(bounds[1])\n  for x in range(a,b+1):\n   if x not in out:\n    out.append(x)\n return out'
        self.assertEqual(execute(code, '1:3,2:4'), [1,2,3,4])
    def test_empty_builtins(self): self.assertEqual(execute('def main(data): return list()'), [])
    def test_multi_case_runner(self):
        args={'code':'def f(x): return x+1','function':'f','cases':[{'args':[1],'kwargs':{},'expected':2},{'args':[4],'kwargs':{},'expected':5}]}
        result=run_python_test_runner(args, RestrictedASTBackend())
        self.assertTrue(result['passed']); self.assertEqual([r['index'] for r in result['cases']], [0,1])
    def test_expected_exception(self):
        result=RestrictedASTBackend().run_python(code='def f(x):\n if x<0: raise ValueError("invalid")\n return x',function='f',mode='test_runner',cases=[{'args':[-1],'kwargs':{},'raises':'ValueError'}]).result
        self.assertTrue(result['passed']); self.assertEqual(result['cases'][0]['exception'], 'ValueError')
    def test_mutation_detected_and_original_preserved(self):
        cases=[{'args':[[1]],'kwargs':{},'expected':None}]; before=deepcopy(cases)
        result=RestrictedASTBackend().run_python(code='def f(x): x.append(2)',function='f',mode='test_runner',cases=cases).result
        self.assertFalse(result['passed']); self.assertEqual(cases,before)
    def test_bool_integer_distinct(self):
        result=RestrictedASTBackend().run_python(code='def f(x): return True',function='f',mode='test_runner',cases=[{'args':[1],'kwargs':{},'expected':1}]).result
        self.assertFalse(result['passed'])
    def test_kwargs(self):
        result=RestrictedASTBackend().run_python(code='def f(x): return x',function='f',mode='test_runner',cases=[{'args':[],'kwargs':{'x':2},'expected':2}]).result
        self.assertTrue(result['passed'])
    def test_deterministic(self): self.assertEqual(execute('def main(data): return sorted(data)', [3,1,2]), execute('def main(data): return sorted(data)', [3,1,2]))
    def test_syntax_check_not_policy(self):
        self.assertTrue(run_python_syntax_check({'code':'import os'})['valid'])
        self.assertFalse(run_python_syntax_check({'code':'def ('})['valid'])
    def test_missing_backend_no_fallback(self):
        with self.assertRaises(SandboxError): run_python_executor({'code':'def main(data): return 1','data':None})
    def test_no_host_io_calls(self):
        with patch.dict(os.environ, {'HF_TOKEN':'TEST_HF_SECRET'}), patch('subprocess.Popen', side_effect=AssertionError('process forbidden')), patch('socket.socket', side_effect=AssertionError('network forbidden')):
            self.assertEqual(execute('def main(data): return sum(data)', [1,2]), 3)
            for key in ('open','env','socket','subprocess'): self.assertRaises(SandboxError, execute, HOSTILE[key])
    def test_bad_signature(self): self.assertRaises(SandboxError, execute, 'def main(x): return x')
    def test_non_json_result(self): self.assertRaises(SandboxError, execute, 'def main(data): return (1,2)')
    def test_nonfinite_input(self): self.assertRaises(ToolExecutionError, execute, 'def main(data): return data', Decimal('NaN'))
    def test_no_stdout(self): self.assertRaises(SandboxError, execute, 'def main(data): print("x")')
    def test_schema_101_cases(self):
        self.assertRaises(Exception, run_python_test_runner, {'code':'def f(x): return x','function':'f','cases':[{'args':[1],'kwargs':{},'expected':1}]*101}, RestrictedASTBackend())
    def test_limits_each_counter(self):
        examples={'statements': ('def main(data): return 1', None), 'calls': ('def main(data): return len(data)', []),
            'iterations': ('def main(data):\n for x in data: pass\n return 1', [1]),
            'allocated': ('def main(data): return [1,2]',None), 'arithmetic': ('def main(data): return 1+2',None)}
        for key,(code,data) in examples.items():
            with self.subTest(key=key), self.assertRaises(SandboxError):
                runtime=Interpreter(code, limits=replace(Limits(), **{key:0}))
                runtime.call('main',[data],{})
    def test_node_budget(self): self.assertRaises(SandboxError, Interpreter, 'def main(data): return 1', limits=replace(Limits(), ast_nodes=1))
    def test_deep_input(self):
        value=1
        for _ in range(80): value=[value]
        self.assertRaises(SandboxError, execute, 'def main(data): return data', value)
    def test_result_budget(self): self.assertRaises(SandboxError, execute, 'def main(data): return "x"*8192')


class V2IntegrationTests(unittest.TestCase):
    def test_catalog_revision_explicit(self):
        self.assertEqual(v1_hash(),'cfe666d04c4870461bd3c18bfd5b78a393d5778cd6fb2aac6399d6701dade92b')
        self.assertNotEqual(v1_hash(), public_catalog_sha256())
        self.assertEqual(len(get_public_catalog()),25); self.assertEqual(len(REGISTRY),25)
    def test_episode_restricted_tool_to_final(self):
        outputs=iter([json.dumps({'type':'tool_call','tool':'python_executor','arguments':{'code':'def main(data): return sum(data)','data':[2,3]}}), json.dumps({'type':'final_answer','answer':5})])
        result=run_episode(simple_task(),lambda context:next(outputs))
        self.assertTrue(result.technical_success); self.assertEqual(result.final_answer,5)
        self.assertEqual(result.tool_results,[{'ok':True,'result':{'value':5}}])
        self.assertIn('Interpretação AST',result.model_contexts[0])
    def test_policy_refusal_technical_success(self):
        outputs=iter([json.dumps({'type':'tool_call','tool':'python_executor','arguments':{'code':HOSTILE['open'],'data':None}}),json.dumps({'type':'final_answer','answer':None})])
        result=run_episode(simple_task(),lambda c:next(outputs))
        self.assertTrue(result.technical_success); self.assertEqual(result.tool_trace[0]['error_code'],'policy_violation')
    def test_python313_allowed(self):
        for minor in (10,11,12,13): self.assertTrue(compatible_python((3,minor)))
        self.assertFalse(compatible_python((3,9))); self.assertFalse(compatible_python((3,14)))
    def test_actual_frozen_v2(self): self.assertTrue(frozen_integrity())
    def test_preflight_without_docker(self):
        from test_phase2_operations import probes, MODEL
        p=probes(); p['restricted_python_runtime']=lambda:True
        with tempfile.TemporaryDirectory() as directory, patch('benchmark.tools.sandbox.DockerSandboxBackend', side_effect=AssertionError('no Docker')):
            result=run_preflight(MODEL,directory,inspect_model=True,probes=p)
        self.assertEqual(result['status'],'PASS'); self.assertEqual(result['sandbox_mode'],'restricted_ast_interpreter')
        self.assertNotIn('safe_sandbox',[c['requirement'] for c in result['checks']])
    def test_runtime_selftest_failure_blocks(self):
        from test_phase2_operations import probes, MODEL
        p=probes(); p['restricted_python_runtime']=lambda:False
        with tempfile.TemporaryDirectory() as directory:
            result=run_preflight(MODEL,directory,inspect_model=True,probes=p)
        self.assertEqual(result['status'],'FAIL')
    def test_pending_full_python_is_not_incorrect(self):
        task=simple_task('T7');task['evaluation_type']='code'
        task['expected_answer']={'function':'f','test_cases':[{'input':1,'expected':2}]}
        episode=make_episode(task,answer='import math\ndef f(x): return x+1')
        result=evaluate_episode(task,episode)
        self.assertTrue(result['evaluation_pending']); self.assertIsNone(result['content_correct'])
        self.assertIsNone(result['task_correct']); self.assertIsNone(result['strict_correct'])
    def test_pending_report_not_publishable(self):
        task=simple_task('T7');task['evaluation_type']='code';task['expected_answer']={'function':'f','test_cases':[]}
        row=evaluate_episode(task,make_episode(task,answer='import re\ndef f(x): return x'))
        report=build_report([row],model_id='fake-model',run_metadata={'run_id':'test-run'})
        self.assertFalse(report['publication_ready']); self.assertFalse(report['evaluation_complete'])
        self.assertEqual(report['catalog_hash'],public_catalog_sha256()); self.assertEqual(report_bytes(report),report_bytes(report))
    def test_runner_smoke_without_docker(self):
        from test_phase2_operations import MODEL
        with tempfile.TemporaryDirectory() as directory, patch('benchmark.tools.sandbox.DockerSandboxBackend',side_effect=AssertionError('no Docker')):
            result=run_phase2(MODEL,directory,preflight_fn=lambda *a,**k:{'status':'PASS','checks':[],'model_revision':'fake'},supervisor_factory=FakeSupervisor)
            raw=json.loads(result['raw'].read_text());report=json.loads(result['evaluation'].read_text())
        self.assertEqual(raw['protocol_version'],'phase2_v2');self.assertEqual(report['protocol_version'],'phase2_v2')
    def test_supervisor_has_no_docker_cleanup(self):
        from test_phase2_operations import fast_worker
        with tempfile.TemporaryDirectory() as directory, patch('benchmark.orchestration.phase2_supervisor.DockerSandboxBackend',side_effect=AssertionError('no Docker')):
            worker=V2Supervisor(fast_worker,{},directory)
            self.assertEqual(worker.execute(3,timeout=2)['result'],3);worker.close()
