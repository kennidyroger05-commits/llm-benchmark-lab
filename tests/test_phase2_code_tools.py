"""Code tool integration uses synthetic functions, never official tasks/cases."""
from decimal import Decimal
import unittest
from unittest.mock import patch

from benchmark.tools.code_tools import run_python_executor as execute, run_python_test_runner as test_code, run_python_syntax_check as syntax
from benchmark.tools.sandbox import DockerSandboxBackend, SandboxError
from benchmark.tools.math_tools import ToolExecutionError


class CodeToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = DockerSandboxBackend()
        # Historical V1 integration; V2 uses the separately tested AST runtime.
        if not cls.backend.is_available():
            raise unittest.SkipTest("V1 Docker integration unavailable; not a V2 requirement.")

    def run_value(self, code, data=None):
        return execute({"code":code,"data":data},self.backend)["value"]

    def test_executor_data_and_json_types(self):
        for value in [{"x":[1,True]},[3,2]," string ",7,Decimal("0.12345678901234567890"),True,None]:
            with self.subTest(value=value):
                result=self.run_value("def main(data): return data",value)
                self.assertEqual(result,value)
                self.assertIs(type(result),type(value))

    def test_executor_allowed_imports(self):
        self.assertEqual(self.run_value(
            "import re\nfrom decimal import Decimal\ndef main(data):\n return [re.findall('[0-9]', data), Decimal('0.1') + Decimal('0.2')]", "a2b3"),
            [["2","3"],Decimal("0.3")])

    def test_forbidden_policy_never_launches_backend(self):
        for code in [
            "import os\ndef main(data): return os.environ",
            "import pathlib\ndef main(data): return pathlib.Path('/')",
            "import socket\ndef main(data): return socket.socket()",
            "import subprocess\ndef main(data): return subprocess.run(['true'])",
            "import sys\ndef main(data): return sys.stdin.read()",
            "def main(data): return open('/etc/passwd').read()",
            "def main(data): return input()",
            "def main(data): return getattr(data, '__class__')",
            "def main(data): return eval('1')",
            "def main(data): return __import__('os')",
            "import json\ndef main(data): return json.codecs.sys.modules",
            "import operator\ndef main(data): return operator.attrgetter('__class__')(data)",
            "def main(data): return (x for x in []).gi_frame.f_globals",
            "from json import sys\ndef main(data): return sys",
        ] + ["import " + name + "\ndef main(data): return 1"
             for name in ("ctypes","pickle","random","time","importlib","inspect")]:
            with self.subTest(code=code), patch.object(self.backend,"_launch") as launch:
                with self.assertRaises(SandboxError) as caught: self.run_value(code)
                self.assertEqual(caught.exception.origin,"sandbox_policy")
                launch.assert_not_called()

    def test_wrong_function_signature(self):
        for code in ["def other(data): return data","def main(): return 1",
                     "def main(x): return x","def main(data, x=1): return data"]:
            with self.subTest(code=code), self.assertRaises(SandboxError):
                self.run_value(code)

    def test_stdout_rejected(self):
        with self.assertRaises(SandboxError) as caught:
            self.run_value("def main(data):\n print('unwanted')\n return 1")
        self.assertEqual(caught.exception.origin,"submitted_code")

    def test_non_json_and_nonfinite_returns(self):
        for expr in ["{1,2}","float('nan')","float('inf')","{1:'value'}"]:
            with self.subTest(expr=expr), self.assertRaises(SandboxError) as caught:
                self.run_value("def main(data): return "+expr)
            self.assertEqual(caught.exception.code,"invalid_data")

    def test_submitted_exception_sanitized(self):
        with self.assertRaises(SandboxError) as caught:
            self.run_value("def main(data): raise ValueError('/secret/private/path')")
        self.assertEqual(caught.exception.origin,"submitted_code")
        self.assertNotIn("secret",caught.exception.message)

    def test_executor_timeout(self):
        with self.assertRaises(SandboxError) as caught:
            self.run_value("def main(data):\n while True: pass")
        self.assertEqual(caught.exception.code,"tool_timeout")
        self.assertEqual(caught.exception.origin,"submitted_code")

    def test_executor_determinism_and_no_mutation(self):
        data={"x":[1]}
        first=self.run_value("def main(data):\n data['x'].append(2)\n return data",data)
        self.assertEqual(first,self.run_value("def main(data):\n data['x'].append(2)\n return data",data))
        self.assertEqual(data,{"x":[1]})

    def test_test_runner_expected_and_raises(self):
        result=test_code({"code":"def f(x):\n if x < 0: raise ValueError('private')\n return x*2",
            "function":"f","cases":[{"args":[2],"kwargs":{},"expected":4},
                                   {"args":[-1],"kwargs":{},"raises":"ValueError"}]},self.backend)
        self.assertTrue(result["passed"])
        self.assertEqual(result["cases"][1]["exception"],"ValueError")

    def test_test_runner_type_order_and_unexpected_exception(self):
        for code,expected in [("return True",1),("return [2,1]",[1,2]),("return 1.0",1),
                              ("raise KeyError('no')",None)]:
            with self.subTest(code=code):
                result=test_code({"code":"def f(): "+code,"function":"f",
                    "cases":[{"args":[],"kwargs":{},"expected":expected}]},self.backend)
                self.assertFalse(result["passed"])

    def test_test_runner_detects_mutation(self):
        cases=[{"args":[[1]],"kwargs":{},"expected":2}]
        result=test_code({"code":"def f(xs):\n xs.append(2)\n return 2",
                         "function":"f","cases":cases},self.backend)
        self.assertFalse(result["passed"])
        self.assertEqual(cases[0]["args"],[[1]])

    def test_test_runner_stdout_and_timeout(self):
        for body in ["print('bad')\n return 0","while True: pass"]:
            with self.subTest(body=body), self.assertRaises(SandboxError):
                test_code({"code":"def f():\n "+body,"function":"f",
                    "cases":[{"args":[],"kwargs":{},"expected":0}]},self.backend)

    def test_test_runner_case_limit(self):
        case={"args":[],"kwargs":{},"expected":1}
        args={"code":"def f(): return 1","function":"f","cases":[case]*100}
        self.assertTrue(test_code(args,self.backend)["passed"])
        with self.assertRaises(ToolExecutionError) as caught:
            test_code({**args,"cases":[case]*101},self.backend)
        self.assertEqual(caught.exception.code,"invalid_arguments")

    def test_test_runner_result_budget(self):
        with self.assertRaises(SandboxError) as caught:
            self.run_value("def main(data): return 'x'*9000")
        self.assertEqual(caught.exception.code,"resource_limit")

    def test_no_official_case_access(self):
        result=test_code({"code":"def f(x): return x","function":"f",
            "cases":[{"args":["private-synthetic-sentinel"],"kwargs":{},
                      "expected":"private-synthetic-sentinel"}]},self.backend)
        self.assertEqual(len(result["cases"]),1)
        self.assertTrue(result["passed"])

    def test_unavailable_no_fallback(self):
        with patch.object(self.backend,"docker",None):
            with self.assertRaises(SandboxError) as caught:
                self.run_value("def main(data): return 1")
        self.assertEqual(caught.exception.origin,"backend_infrastructure")


class SyntaxTests(unittest.TestCase):
    def test_valid_and_empty(self):
        self.assertEqual(syntax({"code":""}),{"valid":True,"errors":[]})
        self.assertTrue(syntax({"code":"def f(): return 1"})["valid"])

    def test_invalid_locations(self):
        result=syntax({"code":"def f(:\n pass"})
        self.assertFalse(result["valid"])
        self.assertEqual(result["errors"][0]["line"],1)
        self.assertIsInstance(result["errors"][0]["column"],int)
        self.assertEqual(result,syntax({"code":"def f(:\n pass"}))

    def test_compile_placement_and_no_execution(self):
        self.assertFalse(syntax({"code":"return 1"})["valid"])
        self.assertTrue(syntax({"code":"raise AssertionError('not executed')"})["valid"])
        self.assertTrue(syntax({"code":"import os\nos.remove('/never-run')"})["valid"])
