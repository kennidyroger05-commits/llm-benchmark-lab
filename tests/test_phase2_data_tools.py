"""Synthetic data pipelines, independent of official benchmark tasks."""
from copy import deepcopy
from decimal import Decimal, localcontext
import unittest
from benchmark.tools.data_tools import (
    run_csv_query as csv_query, run_json_query as json_query,
    run_table_transform as table, resolve_path,
)
from benchmark.tools.math_tools import ToolExecutionError
from benchmark.tools import public_catalog_sha256

def tab(rows, *pipeline):
    return table({"rows":rows,"pipeline":list(pipeline)})["rows"]

def filt(op,value,path=["x"]):
    return {"op":"filter","conditions":[{"path":path,"operator":op,"value":value}]}

class DataToolsTests(unittest.TestCase):
    def error(self,fn,args,code="invalid_data"):
        with self.assertRaises(ToolExecutionError) as caught:
            fn(args)
        self.assertEqual(caught.exception.code,code)

    def test_path_empty(self):
        obj={"x":[]}
        self.assertIs(resolve_path(obj,[]),obj)
    def test_path_key_index(self):
        self.assertEqual(resolve_path({"a":[{"b":3}]},["a",0,"b"]),3)
    def test_path_errors(self):
        for obj,path in [({},["x"]),([], [0]),({"a":1},[0]),([1],["x"]),
                         ([1],[True]),([1],[-1]),(1,["x"]),({},None)]:
            with self.subTest(path=path): self.error(lambda a:resolve_path(*a),(obj,path))

    def test_latest_versions_position(self):
        rows=[{"k":"b","v":1},{"k":"a","v":5},{"k":"b","v":3}]
        self.assertEqual(tab(rows,{"op":"latest_by","keys":[["k"]],"version":["v"]}),
                         [{"k":"b","v":3},{"k":"a","v":5}])
    def test_latest_multiple_keys(self):
        rows=[{"a":1,"b":"x","v":1},{"a":1,"b":"y","v":2}]
        self.assertEqual(tab(rows,{"op":"latest_by","keys":[["a"],["b"]],"version":["v"]}),rows)
    def test_latest_tie_max(self):
        self.error(lambda rows:tab(rows,{"op":"latest_by","keys":[],"version":["v"]}),
                   [{"v":2},{"v":2}])
    def test_latest_lower_duplicate_not_max(self):
        self.assertEqual(tab([{"v":1},{"v":1},{"v":3}],{"op":"latest_by","keys":[],"version":["v"]}),[{"v":3}])
    def test_latest_wrong_version_type(self):
        for version in [True,"2",None]:
            self.error(lambda rows:tab(rows,{"op":"latest_by","keys":[],"version":["v"]}),[{"v":version}])
    def test_latest_before_filter_differs(self):
        rows=[{"id":"x","v":1,"keep":True},{"id":"x","v":2,"keep":False}]
        latest={"op":"latest_by","keys":[["id"]],"version":["v"]}
        condition=filt("eq",True,["keep"])
        self.assertEqual(tab(rows,latest,condition),[])
        self.assertEqual(tab(rows,condition,latest),rows[:1])

    def test_filter_all_operators(self):
        expected={"eq":[2],"ne":[1,3],"lt":[1],"le":[1,2],"gt":[3],"ge":[2,3],"in":[1,3]}
        for op,values in expected.items():
            with self.subTest(op=op):
                self.assertEqual([r["x"] for r in tab([{"x":i} for i in (1,2,3)],filt(op,[1,3] if op=="in" else 2))],values)
    def test_filter_and(self):
        condition=filt("gt",1)
        condition["conditions"].append({"path":["x"],"operator":"lt","value":3})
        self.assertEqual(tab([{"x":1},{"x":2},{"x":3}],condition),[{"x":2}])
    def test_filter_incompatible_bool_integer(self):
        for value in [True,"1",Decimal("1")]:
            self.error(lambda rows:tab(rows,filt("eq",value)),[{"x":1}])
    def test_filter_validates_all_conditions(self):
        condition=filt("eq",9)
        condition["conditions"].append({"path":["missing"],"operator":"eq","value":0})
        self.error(lambda rows:tab(rows,condition),[{"x":1}])
    def test_filter_in_types(self):
        self.error(lambda rows:tab(rows,filt("in",[1,True])),[{"x":1}])
    def test_filter_in_requires_list(self):
        self.error(lambda rows:tab(rows,filt("in",1)),[{"x":1}],"invalid_arguments")

    def test_derive_operators(self):
        for op,value in [("add",7),("subtract",3),("multiply",10)]:
            stage={"op":"derive","name":"x","expression":{"operator":op,"operands":[{"field":["x"]},{"literal":2}]}}
            self.assertEqual(tab([{"x":5}],stage),[{"x":value}])
    def test_derive_nested_field(self):
        stage={"op":"derive","name":"z","expression":{"operator":"add","operands":[{"field":["x"]},{"field":["nested","y"]}]}}
        self.assertEqual(tab([{"x":2,"nested":{"y":3}}],stage)[0]["z"],5)
    def test_derive_decimal_exact_context_independent(self):
        stage={"op":"derive","name":"x","expression":{"operator":"add","operands":[{"field":["x"]},{"literal":Decimal("0.0001")}]}}
        with localcontext() as ctx:
            ctx.prec=2
            self.assertEqual(tab([{"x":Decimal("12345.6789")}],stage),[{"x":"12345.679"}])
    def test_derive_incompatible(self):
        for value in [True,"2",Decimal("2")]:
            stage={"op":"derive","name":"y","expression":{"operator":"add","operands":[{"field":["x"]},{"literal":value}]}}
            self.error(lambda rows:tab(rows,stage),[{"x":1}])
    def test_derive_invalid_expression(self):
        self.error(lambda rows:tab(rows,{"op":"derive","name":"x","expression":{"operator":"divide","operands":[]}}),[],"invalid_arguments")
    def test_derive_literal_deep_copy(self):
        self.assertEqual(tab([{}],{"op":"derive","name":"x","expression":{"literal":[1,{"a":True}]}}),[{"x":[1,{"a":True}]}])

    def test_explode_order_and_empty(self):
        rows=[{"items":["z","a"],"n":1},{"items":[],"n":2}]
        result=tab(rows,{"op":"explode","path":["items"],"as":"item"})
        self.assertEqual([r["item"] for r in result],["z","a"])
        self.assertEqual([r["n"] for r in result],[1,1])
    def test_explode_wrong_type_and_missing(self):
        for row in [{},{"items":3}]:
            self.error(lambda rows:tab(rows,{"op":"explode","path":["items"],"as":"i"}),[row])

    def test_join_order_multiple_none(self):
        rows=[{"k":2},{"k":1},{"k":3}]
        right=[{"k":1,"tag":"z"},{"k":2,"tag":"x"},{"k":1,"tag":"a"}]
        result=tab(rows,{"op":"join","right":right,"left_key":["k"],"right_key":["k"],"as":"r"})
        self.assertEqual([(r["k"],r["r"]["tag"]) for r in result],[(2,"x"),(1,"z"),(1,"a")])
    def test_join_typed_equality(self):
        stage={"op":"join","right":[{"k":True},{"k":Decimal("1")},{"k":1}],"left_key":["k"],"right_key":["k"],"as":"r"}
        result=tab([{"k":1}],stage)
        self.assertEqual(len(result),1)
        self.assertIs(type(result[0]["r"]["k"]),int)
    def test_join_json_root(self):
        args={"data":{"left":[{"k":"s"}],"right":[{"k":"s","v":9}]},"root":["left"],"pipeline":[
            {"op":"join","right":{"root":["right"]},"left_key":["k"],"right_key":["k"],"as":"r"}]}
        self.assertEqual(json_query(args)["rows"][0]["r"]["v"],9)
    def test_join_invalid_root(self):
        self.error(json_query,{"data":{"left":[{"k":"s"}]},"root":["left"],"pipeline":[
            {"op":"join","right":{"root":["missing"]},"left_key":["k"],"right_key":["k"],"as":"r"}]})
    def test_join_root_rejected_for_table(self):
        self.error(table,{"rows":[],"pipeline":[{"op":"join","right":{"root":[]},"left_key":[],"right_key":[],"as":"r"}]},"invalid_arguments")

    def test_group_count_sum_order(self):
        stage={"op":"group","keys":[{"name":"g","path":["g"]}],"aggregates":[
            {"name":"n","function":"count","path":None},{"name":"s","function":"sum","path":["v"]}]}
        self.assertEqual(tab([{"g":"z","v":1},{"g":"a","v":7},{"g":"z","v":3}],stage),
                         [{"g":"z","n":2,"s":4},{"g":"a","n":1,"s":7}])
    def test_group_decimal_multiple_keys(self):
        stage={"op":"group","keys":[{"name":"a","path":["a"]},{"name":"b","path":["b"]}],
               "aggregates":[{"name":"s","function":"sum","path":["v"]}]}
        self.assertEqual(tab([{"a":1,"b":True,"v":Decimal("0.1")},{"a":1,"b":True,"v":Decimal("0.2")}],stage),
                         [{"a":1,"b":True,"s":"0.3"}])
    def test_group_keys_typed(self):
        self.assertEqual(len(tab([{"x":True},{"x":1}],{"op":"group","keys":[{"name":"x","path":["x"]}],"aggregates":[]})),2)
    def test_group_invalid_sum(self):
        stage={"op":"group","keys":[],"aggregates":[{"name":"s","function":"sum","path":["v"]}]}
        for rows in [[{"v":"1"}],[{}],[{"v":True}],[{"v":1},{"v":Decimal("2")}]]:
            self.error(lambda r:tab(r,stage),rows)
    def test_group_duplicate_names(self):
        self.error(lambda rows:tab(rows,{"op":"group","keys":[{"name":"x","path":[]}],
            "aggregates":[{"name":"x","function":"count","path":None}]}),[])

    def test_sort_asc_desc_stability(self):
        rows=[{"x":2,"tag":"b"},{"x":1,"tag":"a"},{"x":2,"tag":"c"}]
        for direction, expected in [("asc",["a","b","c"]),("desc",["b","c","a"])]:
            result=tab(rows,{"op":"sort","by":[{"path":["x"],"direction":direction}]})
            self.assertEqual([r["tag"] for r in result],expected)
    def test_sort_multiple_criteria(self):
        rows=[{"a":1,"b":"a"},{"a":0,"b":"c"},{"a":1,"b":"z"}]
        self.assertEqual(tab(rows,{"op":"sort","by":[{"path":["a"],"direction":"asc"},{"path":["b"],"direction":"desc"}]}),
                         [rows[1],rows[2],rows[0]])
    def test_sort_incompatible_and_missing(self):
        stage={"op":"sort","by":[{"path":["x"],"direction":"asc"}]}
        for rows in [[{"x":1},{"x":"1"}],[{}],[{"x":True}],[{"x":None}]]:
            self.error(lambda r:tab(r,stage),rows)

    def test_project_order(self):
        result=tab([{"a":1,"b":2}],{"op":"project","fields":[{"name":"second","path":["b"]},{"name":"first","path":["a"]}]})
        self.assertEqual(list(result[0]),["second","first"])
        self.assertEqual(result,[{"second":2,"first":1}])
    def test_project_invalid(self):
        for fields in [[{"name":"x","path":["missing"]}],[{"name":"x","path":[]},{"name":"x","path":[]}]]:
            self.error(lambda rows:tab(rows,{"op":"project","fields":fields}),[{}])

    def test_csv_typed_and_quotes(self):
        args={"csv":'name;n;money;active\n"a;b";+007;-1.2500;true\nz;-2;0.00;false\n',"delimiter":";",
              "columns":{"name":"string","n":"integer","money":"decimal","active":"boolean"},"pipeline":[]}
        self.assertEqual(csv_query(args)["rows"],[{"name":"a;b","n":7,"money":"-1.25","active":True},{"name":"z","n":-2,"money":"0","active":False}])
    def test_csv_pipeline_after_conversion(self):
        result=csv_query({"csv":"n\n3\n7\n","delimiter":",","columns":{"n":"integer"},"pipeline":[filt("gt",4,["n"])]})
        self.assertEqual(result,{"rows":[{"n":7}]})
    def test_csv_decimal_arithmetic_internal(self):
        self.assertEqual(csv_query({"csv":"v\n0.1\n0.2","delimiter":",","columns":{"v":"decimal"},"pipeline":[
            {"op":"group","keys":[],"aggregates":[{"name":"total","function":"sum","path":["v"]}]}]}),{"rows":[{"total":"0.3"}]})
    def test_csv_invalid_headers(self):
        for text,columns in [("",{}),("a,a\n1,2",{"a":"integer"}),("a\n1",{}),("a\n1",{"a":"integer","b":"string"})]:
            self.error(csv_query,{"csv":text,"delimiter":",","columns":columns,"pipeline":[]})
    def test_csv_invalid_row_width_and_quotes(self):
        for text in ["a\n1,2","a,b\n1",'a\n"unclosed']:
            self.error(csv_query,{"csv":text,"delimiter":",","columns":{k:"string" for k in text.splitlines()[0].split(",")},"pipeline":[]})
    def test_csv_invalid_typed_values(self):
        for kind,values in [("integer",["1.0"," 2","١","1e2"]),("decimal",["+1","01","1e2","NaN"]),("boolean",["True","1"," false"])]:
            for value in values:
                with self.subTest(kind=kind,value=value):
                    self.error(csv_query,{"csv":"v\n"+value,"delimiter":",","columns":{"v":kind},"pipeline":[]})

    def test_json_root_and_empty_pipeline(self):
        self.assertEqual(json_query({"data":{"a":[{"x":2}]},"root":["a"],"pipeline":[]}),{"rows":[{"x":2}]})
    def test_json_nonlist_nonobject(self):
        for data in [{},1,[1],None]:
            self.error(json_query,{"data":data,"root":[],"pipeline":[]})
    def test_table_empty_pipeline(self):
        self.assertEqual(tab([{"x":"001","a":[True]}]),[{"x":"001","a":[True]}])
    def test_float_rejected_without_precision_recovery(self):
        self.error(table,{"rows":[{"x":0.1}],"pipeline":[]})
    def test_limits_pipeline_input_intermediate_output(self):
        self.error(table,{"rows":[],"pipeline":[filt("eq",1)]*33},"resource_limit")
        self.error(table,{"rows":[{}]*10001,"pipeline":[]},"resource_limit")
        self.error(json_query,{"data":[{}]*10001,"root":[],"pipeline":[]},"resource_limit")
        self.error(csv_query,{"csv":"x\n"+"a\n"*10001,"delimiter":",","columns":{"x":"string"},"pipeline":[]},"resource_limit")
        self.error(table,{"rows":[{"xs":[0]*100001}],"pipeline":[{"op":"explode","path":["xs"],"as":"x"}]},"resource_limit")
        self.error(table,{"rows":[{"x":"a"*8200}],"pipeline":[]},"resource_limit")
    def test_limit_boundaries_accepted_before_aggregation(self):
        group={"op":"group","keys":[],"aggregates":[{"name":"n","function":"count","path":None}]}
        self.assertEqual(tab([{}]*10000,group),[{"n":10000}])
        self.assertEqual(tab([{"x":0}]*1000,{"op":"join","right":[{"x":0}]*100,
            "left_key":["x"],"right_key":["x"],"as":"r"},group),[{"n":100000}])

    def test_join_intermediate_limit(self):
        self.error(table,{"rows":[{"x":0}]*1001,"pipeline":[{"op":"join","right":[{"x":0}]*100,
            "left_key":["x"],"right_key":["x"],"as":"r"}]},"resource_limit")

    def test_nonmutation_all_entrypoints_and_outputs(self):
        inputs=[
            (table,{"rows":[{"nested":{"a":[1,2]},"xs":[{"z":1}]}],"pipeline":[{"op":"explode","path":["xs"],"as":"x"}]}),
            (json_query,{"data":{"l":[{"k":1}],"r":[{"k":1}]},"root":["l"],"pipeline":[{"op":"join","right":{"root":["r"]},"left_key":["k"],"right_key":["k"],"as":"r"}]}),
            (csv_query,{"csv":"x\n1","delimiter":",","columns":{"x":"integer"},"pipeline":[]})]
        for fn,args in inputs:
            before=deepcopy(args)
            first=fn(args)
            self.assertEqual(first,fn(args))
            self.assertEqual(args,before)
            first["rows"][0]["mutated"]=True
            self.assertEqual(args,before)
    def test_catalog_hash_unchanged(self):
        self.assertEqual(public_catalog_sha256(),"cfe666d04c4870461bd3c18bfd5b78a393d5778cd6fb2aac6399d6701dade92b")

if __name__=="__main__":
    unittest.main()
