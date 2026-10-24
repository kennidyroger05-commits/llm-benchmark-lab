"""Structural contracts only; never execute any submitted code or tool."""
from copy import deepcopy
from dataclasses import fields
from decimal import Decimal
import hashlib
import json
import re
import unittest
from unittest.mock import patch

from benchmark.tools import (
    PublicToolDefinition, get_public_catalog, serialize_public_catalog,
    public_catalog_sha256, validate_tool_arguments, StructuralValidationError, UnknownToolError,
)
from benchmark.tools.schemas import validate_structure
from benchmark.tools._schema_definitions import DEFINITIONS
from benchmark.tools import _schema_definitions as definitions
from benchmark.orchestration.protocol import parse_model_message

VALID = {
    "calculator_basic": {"expression": "2+7"},
    "decimal_calculator": {"precision": 28, "rounding": "ROUND_HALF_UP",
        "variables": {"initial": "1.05"}, "steps": [
            {"op":"assign","name":"x","expression":"initial"},
            {"op":"quantize","name":"x","expression":"x","quantum":"0.1"},
            {"op":"repeat","count":2,"steps":[{"op":"assign","name":"x","expression":"x"}]}],
        "outputs":["x"]},
    "integer_optimizer": {"variables":[{"name":"x","min":0,"max":2}],
        "constraints":[{"coefficients":{"x":1},"relation":"le","rhs":2}],
        "objectives":[{"coefficients":{"x":1},"direction":"min"}]},
    "csv_query": {"csv":"x\n2","delimiter":",","columns":{"x":"integer"},"pipeline":[]},
    "json_query": {"data":None,"root":[],"pipeline":[]},
    "table_transform": {"rows":[{"x":1}],"pipeline":[]},
    "python_executor": {"code":"not executed","data":None},
    "lookup_order": {"order_id":"sample"},
    "check_return": {"line_id":"sample","reason":"sample"},
    "create_return": {"order_id":"sample","line_ids":["sample"]},
    "cancel_order": {"order_id":"sample"},
    "refund_payment": {"order_id":"sample"},
    "order_workflow_query": {"order_id":"sample","fields":["status","lines"]},
    "get_stock": {"sku":"sample"},
    "reserve_stock": {"order_id":"sample","sku":"sample","quantity":1},
    "charge": {"order_id":"sample","amount_centavos":1},
    "release_stock": {"reservation_id":"sample"},
    "log_failure": {"order_id":"sample","reason":"sample"},
    "issue_invoice": {"order_id":"sample"},
    "send_receipt": {"order_id":"sample"},
    "inventory_query": {"skus":["sample"]},
    "python_syntax_check": {"code":""},
    "python_test_runner": {"code":"not executed","function":"func","cases":[
        {"args":[1],"kwargs":{"x":None},"expected":None}]},
    "json_validator": {"text":"","schema":{"type":"string"}},
    "text_format_validator": {"text":"","line_count":0,"prefixes":[],
        "allow_empty_lines":True,"allow_outer_spaces":False,"allow_final_newline":False},
}
STAGES = [
    {"op":"latest_by","keys":[["id"]],"version":["v"]},
    {"op":"filter","conditions":[{"path":["x"],"operator":"in","value":[1]}]},
    {"op":"derive","name":"y","expression":{"operator":"add","operands":[{"field":["x"]},{"literal":1}]}},
    {"op":"explode","path":["x"],"as":"y"},
    {"op":"join","right":[],"left_key":[],"right_key":[],"as":"y"},
    {"op":"group","keys":[{"name":"k","path":[]}],"aggregates":[{"name":"n","function":"count","path":None}]},
    {"op":"sort","by":[{"path":[],"direction":"asc"}]},
    {"op":"project","fields":[{"name":"x","path":[]}]},
]


class ToolContractsTests(unittest.TestCase):
    def bad(self,name,args):
        with self.assertRaises(StructuralValidationError):
            validate_tool_arguments(name,args)

    def test_exact_names_unique_sorted(self):
        names=[t.name for t in get_public_catalog()]
        self.assertEqual(names,sorted(VALID))
        self.assertEqual(len(names),25)
        self.assertEqual(len(set(names)),25)

    def test_sort_independent_of_declaration_order(self):
        reordered=dict(reversed(list(definitions.PARAMETERS.items())))
        with patch.dict(definitions.PARAMETERS,reordered,clear=True):
            self.assertEqual([t.name for t in get_public_catalog()],sorted(VALID))

    def test_public_fields_and_no_private_content(self):
        self.assertEqual([f.name for f in fields(PublicToolDefinition)],
                         ["name","description","parameters","returns"])
        data=json.loads(serialize_public_catalog())
        for tool in data:
            self.assertEqual(set(tool),{"name","description","parameters","returns"})
            self.assertEqual(tool["parameters"]["additionalProperties"],False)
            self.assertIsInstance(tool["returns"],dict)
        text=serialize_public_catalog().decode()
        for forbidden in ["expected_", "fixture", "optimal", "acceptable", "minimum_calls",
                          "evaluation_notes", "task_id", "priority", "recommended"]:
            self.assertNotIn(forbidden,text)
        self.assertIsNone(re.search(r"\bT(?:10|[1-9])\b",text))

    def test_determinism_and_hash(self):
        data=serialize_public_catalog()
        self.assertEqual(data,serialize_public_catalog())
        self.assertEqual(public_catalog_sha256(),hashlib.sha256(data).hexdigest())
        self.assertEqual(public_catalog_sha256(),public_catalog_sha256())
        self.assertRegex(public_catalog_sha256(),r"^[a-f0-9]{64}$")
        self.assertIn("operações",data.decode())
        self.assertNotIn(b"\\u00",data)

    def test_copies_isolate_mutation(self):
        original=serialize_public_catalog()
        a=get_public_catalog()
        a[0].parameters["private"]="sentinel"
        a[0].to_dict()["returns"]["private"]="sentinel"
        self.assertEqual(serialize_public_catalog(),original)

    def test_unknown_tool_is_distinct_from_structure_and_protocol(self):
        with self.assertRaises(UnknownToolError):
            validate_tool_arguments("unknown",{})
        self.assertEqual(parse_model_message(
            '{"type":"tool_call","tool":"unknown","arguments":{}}').tool,"unknown")

    def test_all_missing_and_extra_fields(self):
        for name, args in VALID.items():
            for key in args:
                invalid=deepcopy(args); del invalid[key]
                with self.subTest(name=name,key=key): self.bad(name,invalid)
            self.bad(name,{**deepcopy(args),"extra":None})

    def test_integer_types_and_bounds(self):
        for value in [True, "3", 0, -1, 1.0, Decimal("1")]:
            self.bad("reserve_stock",{**VALID["reserve_stock"],"quantity":value})
        for value in [27,101,True,"28"]:
            self.bad("decimal_calculator",{**VALID["decimal_calculator"],"precision":value})

    def test_enums(self):
        self.bad("decimal_calculator",{**VALID["decimal_calculator"],"rounding":"other"})
        self.bad("csv_query",{**VALID["csv_query"],"columns":{"x":"float"}})
        self.bad("order_workflow_query",{"order_id":"x","fields":["unknown"]})

    def test_unique_nonempty_arrays_strings(self):
        for value in [[],[""],["x","x"]]:
            self.bad("inventory_query",{"skus":value})
        self.bad("get_stock",{"sku":""})
        self.bad("csv_query",{**VALID["csv_query"],"delimiter":".."})

    def test_identifiers_and_decimal_strings(self):
        for key in ["_x","9a","á","x-y"]:
            self.bad("decimal_calculator",{**VALID["decimal_calculator"],"variables":{key:"1"}})
        for value in ["1e2","+1","01","NaN","Infinity",1,True,"1.",".2"]:
            self.bad("decimal_calculator",{**VALID["decimal_calculator"],"variables":{"x":value}})
        self.bad("python_test_runner",{**VALID["python_test_runner"],"function":"a.b"})

    def test_paths(self):
        for path in [[-1],[True],[1.0],[None]]:
            self.bad("json_query",{**VALID["json_query"],"root":path})
        validate_tool_arguments("json_query",{**VALID["json_query"],"root":["",0,"a"]})

    def test_finite_generic_json(self):
        for value in [None,True,7,Decimal("1.234"),0.5,"x",[],{"a":[False]}]:
            validate_tool_arguments("python_executor",{"code":"x","data":value})
        for value in [float("nan"),float("inf"),float("-inf"),Decimal("NaN"),Decimal("Infinity"),{1:"x"},(1,)]:
            self.bad("python_executor",{"code":"x","data":{"nested":[value]}})
        cyclic=[]; cyclic.append(cyclic)
        self.bad("python_executor",{"code":"x","data":cyclic})

    def test_steps(self):
        for step in VALID["decimal_calculator"]["steps"]:
            validate_structure(step,DEFINITIONS["Step"])
        repeat={"op":"repeat","count":1,"steps":[VALID["decimal_calculator"]["steps"][-1]]}
        for step in [repeat,{"op":"unknown"},{"op":"quantize","name":"x","expression":"x","quantum":"0"}]:
            with self.assertRaises(StructuralValidationError): validate_structure(step,DEFINITIONS["Step"])

    def test_all_stage_unions(self):
        for stage in STAGES:
            with self.subTest(op=stage["op"]):
                validate_tool_arguments("table_transform",{"rows":[],"pipeline":[stage]})
                self.bad("table_transform",{"rows":[],"pipeline":[{**stage,"extra":0}]})
        self.bad("table_transform",{"rows":[],"pipeline":[{"op":"other"}]})
        self.bad("table_transform",{"rows":[],"pipeline":[STAGES[0]]*33})

    def test_join_root_only_for_json(self):
        stage={**STAGES[4],"right":{"root":["r"]}}
        validate_tool_arguments("json_query",{"data":{},"root":[],"pipeline":[stage]})
        self.bad("table_transform",{"rows":[],"pipeline":[stage]})
        self.bad("csv_query",{**VALID["csv_query"],"pipeline":[stage]})

    def test_expr(self):
        for expression in [{"field":[]},{"literal":{"a":None}},STAGES[2]["expression"]]:
            validate_structure(expression,DEFINITIONS["Expr"])
        for expression in [{"field":[],"literal":0},{"operator":"divide","operands":[]},
                           {"operator":"add","operands":[{"literal":1}]}]:
            with self.assertRaises(StructuralValidationError): validate_structure(expression,DEFINITIONS["Expr"])
        expression={"literal":0}
        for _ in range(32): expression={"operator":"add","operands":[expression,{"literal":0}]}
        with self.assertRaises(StructuralValidationError): validate_structure(expression,DEFINITIONS["Expr"])

    def test_testcase_exclusive_outcome(self):
        base={"args":[],"kwargs":{}}
        for outcome in [{"expected":None},{"raises":"ValueError"}]:
            validate_structure({**base,**outcome},DEFINITIONS["TestCase"])
        for outcome in [{},{"expected":0,"raises":"ValueError"},{"raises":"RuntimeError"}]:
            with self.assertRaises(StructuralValidationError): validate_structure({**base,**outcome},DEFINITIONS["TestCase"])

    def test_structural_schema(self):
        valid=[{"type":"string"},{"type":"object","properties":{},"required":[],"additionalProperties":False},
               {"type":"array","items":{"type":"integer"},"minItems":0,"maxItems":2}]
        for schema in valid: validate_tool_arguments("json_validator",{"text":"","schema":schema})
        for schema in [{"type":"other"},{"type":"array"},{"type":"object"},
                       {"type":"string","enum":["x"]},{"type":"string","minItems":1},
                       {"type":"array","items":{"type":"null"},"minItems":True},
                       {"type":"string","$ref":"external"}]:
            self.bad("json_validator",{"text":"","schema":schema})
        schema={"type":"string"}
        for _ in range(16): schema={"type":"array","items":schema}
        self.bad("json_validator",{"text":"","schema":schema})

    def test_no_execution_or_mutation(self):
        args={"code":"raise AssertionError('must never execute')","data":None}
        before=deepcopy(args)
        validate_tool_arguments("python_executor",args)
        self.assertEqual(args,before)


# Each of the 25 definitions gets independently reported valid/invalid tests.
def _valid_test(name):
    def test(self): validate_tool_arguments(name,deepcopy(VALID[name]))
    return test

def _invalid_test(name):
    def test(self): self.bad(name,{**deepcopy(VALID[name]),"unexpected":True})
    return test

for _name in VALID:
    setattr(ToolContractsTests,"test_valid_"+_name,_valid_test(_name))
    setattr(ToolContractsTests,"test_invalid_"+_name,_invalid_test(_name))

if __name__=="__main__":
    unittest.main()
