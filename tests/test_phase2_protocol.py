"""Envelope tests; no catalog or task-specific logic."""
from dataclasses import asdict
from decimal import Decimal
import json
import unittest
from benchmark.orchestration.protocol import (
    FinalAnswer, ProtocolError, ToolCall, parse_model_message,
)

class Phase2ProtocolTests(unittest.TestCase):
    def invalid(self, raw):
        with self.assertRaises(ProtocolError) as caught:
            parse_model_message(raw)
        self.assertEqual(caught.exception.code, "invalid_protocol")

    def test_empty_arguments(self):
        msg = parse_model_message('{"type":"tool_call","tool":"x","arguments":{}}')
        self.assertIsInstance(msg, ToolCall)
        self.assertEqual(asdict(msg), {"type":"tool_call","tool":"x","arguments":{}})

    def test_nested_arguments_preserve_types(self):
        msg = parse_model_message(
            '{"type":"tool_call","tool":"x","arguments":{"a":{"b":[1,2.10,1e-30,true,false,null,"ç"]}}}'
        )
        values = msg.arguments["a"]["b"]
        self.assertEqual(values, [1, Decimal("2.10"), Decimal("1e-30"), True, False, None, "ç"])
        self.assertIs(type(values[0]), int)
        self.assertIs(type(values[1]), Decimal)
        self.assertIs(type(values[3]), bool)

    def test_unknown_tool_is_valid(self):
        msg = parse_model_message('{"type":"tool_call","tool":"inexistente","arguments":{}}')
        self.assertEqual(msg.tool, "inexistente")

    def test_final_object(self):
        self.assertEqual(parse_model_message(
            '{"type":"final_answer","answer":{"x":[1,null]}}'
        ), FinalAnswer({"x":[1,None]}))

    def test_final_array(self):
        self.assertEqual(parse_model_message(
            '{"type":"final_answer","answer":[3,2,1]}'
        ).answer, [3,2,1])

    def test_final_string(self):
        self.assertEqual(parse_model_message(
            '{"type":"final_answer","answer":"  conteúdo\\n"}'
        ).answer, "  conteúdo\n")

    def test_final_integer(self):
        msg = parse_model_message('{"type":"final_answer","answer":1234567890123456789012345}')
        self.assertIs(type(msg.answer), int)
        self.assertEqual(msg.answer, 1234567890123456789012345)

    def test_final_decimal_exact(self):
        msg = parse_model_message('{"type":"final_answer","answer":0.12345678901234567890123456789}')
        self.assertIs(type(msg.answer), Decimal)
        self.assertEqual(msg.answer, Decimal("0.12345678901234567890123456789"))

    def test_finite_exponent_without_float_overflow(self):
        msg = parse_model_message('{"type":"final_answer","answer":1e10000}')
        self.assertIsInstance(msg.answer, Decimal)
        self.assertTrue(msg.answer.is_finite())
        self.assertEqual(msg.answer, Decimal("1e10000"))

    def test_final_true(self):
        self.assertIs(parse_model_message('{"type":"final_answer","answer":true}').answer, True)

    def test_final_false(self):
        self.assertIs(parse_model_message('{"type":"final_answer","answer":false}').answer, False)

    def test_final_null(self):
        self.assertIsNone(parse_model_message('{"type":"final_answer","answer":null}').answer)

    def test_whitespace_and_key_order(self):
        self.assertEqual(parse_model_message(
            ' \t\r\n {"answer":false,"type":"final_answer"} \n\t'
        ), FinalAnswer(False))

    def test_valid_message_unchanged(self):
        expected = {
            "type":"tool_call", "tool":" unknown ",
            "arguments":{"order":["z","a"],"nested":{"x":True},"text":"  unmodified\n"},
        }
        raw = json.dumps(expected, ensure_ascii=False)
        self.assertEqual(asdict(parse_model_message(raw)), expected)
        self.assertEqual(raw, json.dumps(expected, ensure_ascii=False))

    def test_malformed_or_external_content(self):
        fence = chr(96)*3
        for raw in [
            "", " \t\n", "{", 'prosa {"type":"final_answer","answer":0}',
            '{"type":"final_answer","answer":0} prosa',
            fence+'json\n{"type":"final_answer","answer":0}\n'+fence,
            '{"type":"final_answer","answer":0}{}',
            '{"type":"final_answer","answer":0} true',
            '{"type":"final_answer","answer":0,}', "\ufeff{}",
        ]:
            with self.subTest(raw=raw):
                self.invalid(raw)

    def test_top_level_nonobjects(self):
        for raw in ['[]','"x"','0','0.5','true','false','null']:
            with self.subTest(raw=raw):
                self.invalid(raw)

    def test_duplicate_keys_all_depths(self):
        for raw in [
            '{"type":"final_answer","type":"final_answer","answer":0}',
            '{"type":"final_answer","answer":0,"answer":1}',
            '{"type":"tool_call","tool":"x","arguments":{"a":1,"a":2}}',
            '{"type":"final_answer","answer":[{"a":true,"a":false}]}',
            '{"type":"final_answer","answer":{"a":1,"\\u0061":1}}',
        ]:
            with self.subTest(raw=raw):
                self.invalid(raw)

    def test_non_json_constants(self):
        for constant in ["NaN","Infinity","-Infinity"]:
            for value in [constant,"["+constant+"]",'{"x":'+constant+"}"]:
                with self.subTest(value=value):
                    self.invalid('{"type":"final_answer","answer":'+value+"}")

    def test_missing_unknown_wrong_type_discriminator(self):
        for obj in [
            {"answer":0},{"type":"other","answer":0},
            {"type":None,"answer":0},{"type":[],"answer":0},
            {"type":True,"answer":0},{"type":{},"answer":0},
        ]:
            with self.subTest(obj=obj):
                self.invalid(json.dumps(obj))

    def test_tool_call_exact_fields(self):
        base = {"type":"tool_call","tool":"x","arguments":{}}
        for key in base:
            changed = dict(base)
            del changed[key]
            with self.subTest(missing=key):
                self.invalid(json.dumps(changed))
        self.invalid(json.dumps({**base,"extra":None}))

    def test_final_exact_fields(self):
        self.invalid('{"type":"final_answer"}')
        self.invalid('{"type":"final_answer","answer":null,"extra":0}')

    def test_tool_nonempty_string(self):
        for value in ["",None,0,True,[],{}]:
            with self.subTest(value=value):
                self.invalid(json.dumps({"type":"tool_call","tool":value,"arguments":{}}))

    def test_arguments_object(self):
        for value in [[],"",None,0,True]:
            with self.subTest(value=value):
                self.invalid(json.dumps({"type":"tool_call","tool":"x","arguments":value}))

    def test_no_coercion_or_task_validation(self):
        value = {"quantity":True,"arbitrary":"0001"}
        msg = parse_model_message(json.dumps({"type":"tool_call","tool":"x","arguments":value}))
        self.assertIs(msg.arguments["quantity"], True)
        self.assertEqual(msg.arguments["arbitrary"], "0001")
        self.assertEqual(parse_model_message(
            '{"type":"final_answer","answer":"not a task-specific answer"}'
        ).answer, "not a task-specific answer")

    def test_nonstring_input(self):
        for raw in [None,{},[],3,b'{"type":"final_answer","answer":0}']:
            with self.subTest(raw=raw):
                self.invalid(raw)

if __name__ == "__main__":
    unittest.main()
