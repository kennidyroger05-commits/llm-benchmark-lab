"""Synthetic math tests; no official tasks, model execution or external solvers."""
from copy import deepcopy
from decimal import Decimal, getcontext, localcontext
import unittest
from benchmark.tools.math_tools import (
    ToolExecutionError, run_calculator_basic as basic,
    run_decimal_calculator as decimal_tool, run_integer_optimizer as optimize,
    _canonical_decimal,
)


def decargs(**changes):
    args = {"precision":28, "rounding":"ROUND_HALF_UP", "variables":{},
            "steps":[], "outputs":[]}
    args.update(changes)
    return args


def optargs(**changes):
    args = {"variables":[{"name":"x","min":0,"max":3}],
            "constraints":[], "objectives":[{"coefficients":{"x":1},"direction":"min"}]}
    args.update(changes)
    return args


class MathTests(unittest.TestCase):
    def error(self, function, arguments, code):
        with self.assertRaises(ToolExecutionError) as caught:
            function(arguments)
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(caught.exception.message)

    def test_canonical_values(self):
        for source, expected in [("0","0"),("-0.000","0"),("12.000","12"),
                                 ("2.5000","2.5"),("0.00000001","0.00000001"),
                                 ("1E30","1"+"0"*30),("1.00000000000000000000000001","1.00000000000000000000000001")]:
            with self.subTest(source=source):
                self.assertEqual(_canonical_decimal(Decimal(source)),expected)

    def test_canonical_independent_of_context(self):
        with localcontext() as ctx:
            ctx.prec=2
            self.assertEqual(_canonical_decimal(Decimal("12345.67000")),"12345.67")
            self.assertEqual(_canonical_decimal(Decimal("1."+"0"*9000)),"1")

    def test_canonical_nonfinite(self):
        for value in ["NaN","Infinity","-Infinity"]:
            self.error(_canonical_decimal,Decimal(value),"invalid_expression")

    def test_canonical_output_budget(self):
        self.error(_canonical_decimal,Decimal("1E9000"),"resource_limit")

    def test_arithmetic_operators(self):
        for expression, expected in [("1+2","3"),("5-7","-2"),("3*4","12"),
                                      ("1/4","0.25"),("7//2","3"),("7%2","1"),
                                      ("-(+2)","-2"),("(2+3)*4","20"),("2+3*4","14"),
                                      ("2.500+0.500","3")]:
            with self.subTest(expression=expression):
                self.assertEqual(basic({"expression":expression}),{"value":expected})

    def test_decimal_exact_literals(self):
        self.assertEqual(basic({"expression":"0.1+0.2"}),{"value":"0.3"})
        self.assertEqual(basic({"expression":"0.123456789012345678901234567890123"}),
                         {"value":"0.123456789012345678901234567890123"})
        self.assertEqual(basic({"expression":"1/3"})["value"],"0."+"3"*28)

    def test_negative_quotient_and_remainder(self):
        for expr, expected in [("-7//2","-3"),("-7%2","-1"),("7//-2","-3"),
                                ("7%-2","1"),("-7%-2","-1")]:
            self.assertEqual(basic({"expression":expr})["value"],expected)

    def test_zero_division(self):
        for expr in ["1/0","1//0","1%0","0/0"]:
            self.error(basic,{"expression":expr},"invalid_expression")

    def test_forbidden_syntax(self):
        for expr in ["2**3","x","f()","'x'","[1]","{}", "(1).real","x[0]",
                     "[x for x in []]","True","0xff","1_000","1e3","1;2","{"]:
            with self.subTest(expr=expr):
                self.error(basic,{"expression":expr},"invalid_expression")

    def test_basic_schema_validation(self):
        for args in [{"expression":""},{"expression":2},{"expression":"1","extra":0},{}]:
            self.error(basic,args,"invalid_arguments")

    def test_character_limit(self):
        self.error(basic,{"expression":"1"*4097},"invalid_arguments")

    def test_ast_limit(self):
        self.error(basic,{"expression":"+".join(["1"]*86)},"resource_limit")
        self.assertEqual(basic({"expression":"+".join(["1"]*85)}),{"value":"85"})

    def test_assignment_variables_overwrite_output_order(self):
        args=decargs(variables={"x":"2"},steps=[
            {"op":"assign","name":"y","expression":"x*3"},
            {"op":"assign","name":"x","expression":"y+1"}],outputs=["y","x"])
        result=decimal_tool(args)
        self.assertEqual(result,{"values":{"y":"6","x":"7"}})
        self.assertEqual(list(result["values"]),["y","x"])

    def test_rounding_modes(self):
        for rounding, expected in [("ROUND_HALF_UP","1.3"),("ROUND_HALF_EVEN","1.2"),
                                   ("ROUND_DOWN","1.2"),("ROUND_UP","1.3"),
                                   ("ROUND_FLOOR","1.2"),("ROUND_CEILING","1.3")]:
            args=decargs(rounding=rounding,steps=[
                {"op":"quantize","name":"x","expression":"1.25","quantum":"0.1"}],outputs=["x"])
            self.assertEqual(decimal_tool(args)["values"]["x"],expected)

    def test_quantum_uses_exponent(self):
        result=decimal_tool(decargs(steps=[
            {"op":"quantize","name":"x","expression":"1.234","quantum":"0.05"}],outputs=["x"]))
        self.assertEqual(result["values"]["x"],"1.23")

    def test_recurrence_rounds_each_iteration(self):
        args=decargs(variables={"balance":"1.03"},steps=[
            {"op":"repeat","count":3,"steps":[
                {"op":"quantize","name":"interest","expression":"balance*0.1","quantum":"0.01"},
                {"op":"assign","name":"balance","expression":"balance+interest"}]}],outputs=["balance"])
        self.assertEqual(decimal_tool(args),{"values":{"balance":"1.36"}})

    def test_unknown_variable_and_output(self):
        self.error(decimal_tool,decargs(steps=[
            {"op":"assign","name":"x","expression":"missing+1"}]),"invalid_expression")
        self.error(decimal_tool,decargs(outputs=["missing"]),"invalid_arguments")

    def test_duplicate_output_and_repeat_nested(self):
        self.error(decimal_tool,decargs(variables={"x":"1"},outputs=["x","x"]),"invalid_arguments")
        self.error(decimal_tool,decargs(steps=[{"op":"repeat","count":1,"steps":[
            {"op":"repeat","count":1,"steps":[]}]}]),"invalid_arguments")

    def test_operation_limit_boundary(self):
        step={"op":"assign","name":"x","expression":"x+1"}
        args=decargs(variables={"x":"0"},steps=[
            {"op":"repeat","count":1000,"steps":[step]*10}],outputs=["x"])
        self.assertEqual(decimal_tool(args)["values"]["x"],"10000")
        self.error(decimal_tool,{**args,"steps":args["steps"]+[step]},"resource_limit")

    def test_variable_limit(self):
        variables={f"v{i}":"0" for i in range(100)}
        self.error(decimal_tool,decargs(variables=variables,steps=[
            {"op":"assign","name":"new","expression":"1"}]),"resource_limit")
        self.assertEqual(decimal_tool(decargs(variables=variables))["values"],{})

    def test_precision_min_max_and_isolation(self):
        for precision in [28,100]:
            result=decimal_tool(decargs(precision=precision,steps=[
                {"op":"assign","name":"x","expression":"1/3"}],outputs=["x"]))
            self.assertEqual(result["values"]["x"],"0."+"3"*precision)
        with localcontext() as ctx:
            ctx.prec=3; ctx.rounding="ROUND_DOWN"
            before=ctx.copy()
            self.assertEqual(basic({"expression":"1/3"})["value"],"0."+"3"*28)
            self.assertEqual(getcontext().prec,before.prec)
            self.assertEqual(getcontext().rounding,before.rounding)

    def test_decimal_no_input_mutation_or_state(self):
        args=decargs(variables={"x":"2"},steps=[
            {"op":"assign","name":"x","expression":"x+1"}],outputs=["x"])
        before=deepcopy(args)
        self.assertEqual(decimal_tool(args),decimal_tool(args))
        self.assertEqual(args,before)

    def test_invalid_decimal_arguments(self):
        for change in [{"precision":27},{"precision":101},{"precision":True},{"rounding":"UNKNOWN"},
                       {"variables":{"x":"1e2"}},{"steps":[{"op":"repeat","count":0,"steps":[]}]}]:
            self.error(decimal_tool,decargs(**change),"invalid_arguments")

    def test_reserved_identifier(self):
        self.error(decimal_tool,decargs(variables={"class":"1"}),"invalid_arguments")

    def test_invalid_quantize(self):
        self.error(decimal_tool,decargs(steps=[
            {"op":"quantize","name":"x","expression":"123456789012345678901234567890","quantum":"0.01"}]),
            "invalid_expression")

    def test_unique_optimizer_solution(self):
        args=optargs(constraints=[{"coefficients":{"x":1},"relation":"eq","rhs":2}])
        self.assertEqual(optimize(args),{"assignment":{"x":2},"objective_values":[2],"combinations_examined":4})

    def test_optimizer_min_max(self):
        for direction, expected in [("min",0),("max",3)]:
            self.assertEqual(optimize(optargs(objectives=[
                {"coefficients":{"x":1},"direction":direction}]))["assignment"],{"x":expected})

    def test_lexicographic_multiple_objectives(self):
        args=optargs(variables=[{"name":"x","min":0,"max":2},{"name":"y","min":0,"max":2}],
            objectives=[{"coefficients":{"x":1},"direction":"max"},
                        {"coefficients":{"y":1},"direction":"min"}])
        self.assertEqual(optimize(args),{"assignment":{"x":2,"y":0},"objective_values":[2,0],"combinations_examined":9})

    def test_final_tie_uses_variable_list_order(self):
        args=optargs(variables=[{"name":"z","min":-2,"max":2},{"name":"a","min":-1,"max":1}],
                     objectives=[{"coefficients":{},"direction":"min"}])
        self.assertEqual(optimize(args)["assignment"],{"z":-2,"a":-1})

    def test_negative_coefficients_le_ge(self):
        args=optargs(variables=[{"name":"x","min":-3,"max":3}],constraints=[
            {"coefficients":{"x":-1},"relation":"le","rhs":1},
            {"coefficients":{"x":1},"relation":"ge","rhs":-2}])
        self.assertEqual(optimize(args)["assignment"],{"x":-1})
        self.assertEqual(optimize(args)["combinations_examined"],7)

    def test_inclusive_singleton_bounds(self):
        self.assertEqual(optimize(optargs(variables=[{"name":"x","min":4,"max":4}]))["combinations_examined"],1)

    def test_infeasible(self):
        self.error(optimize,optargs(constraints=[
            {"coefficients":{"x":1},"relation":"ge","rhs":4}]),"infeasible")

    def test_duplicate_optimizer_names(self):
        self.error(optimize,optargs(variables=[
            {"name":"x","min":0,"max":1},{"name":"x","min":0,"max":1}]),"invalid_arguments")

    def test_unknown_constraint_and_objective_names(self):
        self.error(optimize,optargs(constraints=[
            {"coefficients":{"missing":0},"relation":"eq","rhs":0}]),"invalid_arguments")
        self.error(optimize,optargs(objectives=[
            {"coefficients":{"missing":1},"direction":"min"}]),"invalid_arguments")

    def test_invalid_bounds_and_domain_limit(self):
        self.error(optimize,optargs(variables=[{"name":"x","min":2,"max":1}]),"invalid_arguments")
        self.error(optimize,optargs(variables=[{"name":"x","min":0,"max":1000000}]),"resource_limit")

    def test_bool_and_string_optimizer_integers_rejected(self):
        for value in [True,"1"]:
            self.error(optimize,optargs(variables=[{"name":"x","min":value,"max":3}]),"invalid_arguments")
            self.error(optimize,optargs(objectives=[
                {"coefficients":{"x":value},"direction":"min"}]),"invalid_arguments")

    def test_optimizer_no_mutation_and_determinism(self):
        args=optargs(); before=deepcopy(args)
        self.assertEqual(optimize(args),optimize(args))
        self.assertEqual(args,before)

    def test_optimizer_exact_maximum_domain(self):
        args=optargs(variables=[{"name":"x","min":0,"max":999999}],
                     objectives=[{"coefficients":{},"direction":"min"}])
        result=optimize(args)
        self.assertEqual(result["combinations_examined"],1000000)
        self.assertEqual(result["assignment"],{"x":0})

    def test_empty_variable_tuple(self):
        result=optimize(optargs(variables=[],objectives=[{"coefficients":{},"direction":"min"}]))
        self.assertEqual(result,{"assignment":{},"objective_values":[0],"combinations_examined":1})

    def test_all_tools_repeat_deterministically(self):
        for function, args in [(basic,{"expression":"7/8"}),(decimal_tool,decargs()),(optimize,optargs())]:
            self.assertEqual(function(args),function(args))


if __name__=="__main__":
    unittest.main()
