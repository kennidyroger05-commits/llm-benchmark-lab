"""Only user-provided schema/format rules; no benchmark answers."""
import json
import unittest
from copy import deepcopy

from benchmark.tools.validators import run_json_validator as jv, run_text_format_validator as tv
from benchmark.tools.math_tools import ToolExecutionError


class ValidatorTests(unittest.TestCase):
    def json(self, value, schema):
        return jv({"text": json.dumps(value), "schema": schema})

    def test_object_nested_required(self):
        schema = {"type":"object","properties":{"x":{"type":"array","items":{"type":"integer"}}},
                  "required":["x"],"additionalProperties":False}
        self.assertTrue(self.json({"x":[1,2]},schema)["valid"])
        self.assertEqual(self.json({},schema)["errors"],[{"path":["x"],"code":"missing_field"}])
        self.assertEqual(self.json({"x":[True]},schema)["errors"],[{"path":["x",0],"code":"wrong_type"}])

    def test_additional_properties(self):
        for allowed in [True, False]:
            s={"type":"object","properties":{},"required":[],"additionalProperties":allowed}
            self.assertEqual(self.json({"extra":2},s)["valid"],allowed)

    def test_array_bounds(self):
        schema={"type":"array","items":{"type":"string"},"minItems":1,"maxItems":2}
        self.assertFalse(self.json([],schema)["valid"])
        self.assertTrue(self.json(["x"],schema)["valid"])
        self.assertFalse(self.json(["x","y","z"],schema)["valid"])

    def test_scalar_types(self):
        for kind,value in [("string","x"),("integer",1),("number",1.2),("boolean",True),("null",None)]:
            self.assertTrue(self.json(value,{"type":kind})["valid"])
        self.assertFalse(self.json(True,{"type":"integer"})["valid"])
        self.assertFalse(self.json(True,{"type":"number"})["valid"])
        self.assertFalse(self.json(1.0,{"type":"integer"})["valid"])
        self.assertTrue(jv({"text":"1e10000","schema":{"type":"number"}})["valid"])

    def test_json_entire_document(self):
        fence=chr(96)*3
        for text in ["{",'{"a":1,"a":2}',fence+'json\n{}\n'+fence,"{} prose","{}{}",
                     "NaN","Infinity","-Infinity"]:
            self.assertEqual(jv({"text":text,"schema":{"type":"object","properties":{},
                "required":[],"additionalProperties":False}}),
                {"valid":False,"errors":[{"path":[],"code":"invalid_json"}]})

    def test_unsupported_schema_keywords(self):
        for key,value in [("const",1),("enum",[1]),("$ref","external"),("pattern",".*"),
                          ("minItems",1)]:
            with self.subTest(key=key), self.assertRaises(ToolExecutionError) as caught:
                jv({"text":"1","schema":{"type":"integer",key:value}})
            self.assertEqual(caught.exception.code,"invalid_arguments")

    def test_schema_depth_and_nodes(self):
        s={"type":"string"}
        for _ in range(16): s={"type":"array","items":s}
        with self.assertRaises(ToolExecutionError): jv({"text":"[]","schema":s})
        s={"type":"object","properties":{str(i):{"type":"string"} for i in range(256)},
           "required":[],"additionalProperties":False}
        with self.assertRaises(ToolExecutionError): jv({"text":"{}","schema":s})

    def test_schema_required_and_bounds_relationship(self):
        for schema in [
            {"type":"array","items":{"type":"null"},"minItems":2,"maxItems":1},
            {"type":"object","properties":{},"required":["x"],"additionalProperties":False}]:
            with self.assertRaises(ToolExecutionError):
                jv({"text":"null","schema":schema})

    def test_json_deterministic_errors_no_expected_values(self):
        args={"text":'{"b":true,"a":null}',"schema":{"type":"object",
            "properties":{"b":{"type":"integer"},"a":{"type":"string"}},
            "required":[],"additionalProperties":False}}
        original=deepcopy(args)
        result=jv(args)
        self.assertEqual(result,jv(args))
        self.assertEqual(args,original)
        self.assertEqual([e["path"] for e in result["errors"]],[["a"],["b"]])
        self.assertTrue(all(set(e)=={"path","code"} for e in result["errors"]))

    def text(self, text, **changes):
        args={"text":text,"line_count":2,"prefixes":["A=","B="],"allow_empty_lines":False,
              "allow_outer_spaces":False,"allow_final_newline":False}
        args.update(changes)
        return tv(args)

    def test_text_valid_prefixes_not_content(self):
        self.assertTrue(self.text("A=arbitrary\nB=anything")["valid"])

    def test_text_crlf_and_single_final_newline(self):
        self.assertTrue(self.text("A=x\r\nB=y\r\n",allow_final_newline=True)["valid"])
        self.assertFalse(self.text("A=x\nB=y\n")["valid"])
        self.assertFalse(self.text("A=x\nB=y\n\n",allow_final_newline=True)["valid"])

    def test_text_line_count(self):
        self.assertIn({"line":None,"code":"line_count"},self.text("A=x")["errors"])

    def test_text_empty_lines(self):
        self.assertIn({"line":2,"code":"empty_line"},self.text("A=x\n")["errors"])
        self.assertTrue(self.text("",line_count=1,prefixes=[""],allow_empty_lines=True)["valid"])

    def test_text_outer_spaces(self):
        self.assertIn({"line":2,"code":"outer_space"},self.text("A=x\nB=y ")["errors"])
        self.assertTrue(self.text("A=x\nB=y ",allow_outer_spaces=True)["valid"])

    def test_text_specific_line_multiple_errors(self):
        result=self.text("A=x\n wrong ")
        self.assertEqual(result, self.text("A=x\n wrong "))
        self.assertEqual(result["errors"],[{"line":2,"code":"prefix"},{"line":2,"code":"outer_space"}])

    def test_prefix_count_invalid_arguments(self):
        with self.assertRaises(ToolExecutionError) as caught:
            self.text("x",prefixes=[])
        self.assertEqual(caught.exception.code,"invalid_arguments")
