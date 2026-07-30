import random
import argparse
import time
import unittest
import json
import sys
import numpy as np
from pprint import pprint, pformat
from pathlib import Path

from zolotone import *
from zolotone.egglog.rules import check_rules, rewrite_rules
from examples.bf16x8_dot_fp32_optimized import bf16x8_dot_fp32_optimized
from examples.bf16x8_dot_fp32_conventional import bf16x8_dot_fp32_conventional
from examples.CSA import CSA_tree4
from examples.fp32_add import fp32_add
from examples.fp32_mult import fp32_mult
from examples.max_exponent import OPTIMIZED_MAX_EXP4

from infra.compile_cpp import jit_compile, nonjit_compile

DEFAULT_SEED = 0
DEFAULT_N_POINTS = 1000


def dot_product_spec(a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3):
        res = 0.0
        res += a_0.evaluate().to_val() * b_0.evaluate().to_val()
        res += a_1.evaluate().to_val() * b_1.evaluate().to_val()
        res += a_2.evaluate().to_val() * b_2.evaluate().to_val()
        res += a_3.evaluate().to_val() * b_3.evaluate().to_val()
        return float(np.float32(res))

def run_spec_with_metrics(design: Node):
    return design.check_spec()

def _trace_runtime_s(proof_trace: list[dict]) -> float:
    return sum(float(stage.get("runtime_s", 0.0)) for stage in proof_trace)


def _report_name(proof_traces: list[list[dict]]) -> str:
    for proof_trace in proof_traces:
        if proof_trace:
            return str(proof_trace[0]["name"])
    raise AssertionError("Expected at least one non-empty proof trace from design.check_spec()")


def merge_spec_reports(reports: list[dict]):
    runtime_s_by_design = {}
    proved_by_design = {}

    total_runtime_s = 0.0
    design_verdicts = []

    for check_result in reports:
        proved = check_result["proved"]
        proof_traces = check_result["proof_traces"]
        if not proof_traces:
            raise AssertionError("Expected proof traces from design.check_spec()")
        design_name = _report_name(proof_traces)
        runtime_s = sum(_trace_runtime_s(proof_trace) for proof_trace in proof_traces)

        runtime_s_by_design[design_name] = runtime_s
        proved_by_design[design_name] = proved
        design_verdicts.append(proved)
        total_runtime_s += runtime_s

    return {
        "proved": all(design_verdicts),
        "runtime_s_total": total_runtime_s,
        "runtime_s_by_design": runtime_s_by_design,
        "proved_by_design": proved_by_design,
    }


class TestFusedDotProduct(unittest.TestCase):
    SEED = DEFAULT_SEED
    N_POINTS = DEFAULT_N_POINTS
    SPEC_REPORT = None
    IMPL_REPORT = None
    
    def test_rules_with_z3(self):
        rules = rewrite_rules()
        results = check_rules(rules, z3_timeout_ms=10000)

        for name, report in results.items():
            self.assertTrue(
                report["z3_status"] == "unsat" or report["dreal_status"] == "unsat",
                pformat(results),
            )
    
    def test_run_spec_verification_and_timing(self):
        print("\nRunning test_run_spec_verification_and_timing:")
        print(
            "\tConstructing CSA_tree4, bf16x8_dot_fp32_conventional, "
            "and bf16x8_dot_fp32_optimized composites."
        )
        print("\tRunning run_spec() for each design and reporting verification runtime.\n")

        rnd = random.Random(self.SEED)
        args = [Var(f"arg_{i}", sign=QT(rnd.randint(1, 20), rnd.randint(1, 20))) for i in range(4)]
        csa_tree4 = CSA_tree4(*args)
        
        a = [
            Var(name="a_0", sign=BFloat16T()),
            Var(name="a_1", sign=BFloat16T()),
            Var(name="a_2", sign=BFloat16T()),
            Var(name="a_3", sign=BFloat16T()),
        ]
        
        b = [
            Var(name="b_0", sign=BFloat16T()),
            Var(name="b_1", sign=BFloat16T()),
            Var(name="b_2", sign=BFloat16T()),
            Var(name="b_3", sign=BFloat16T()),
        ]
        
        conventional = bf16x8_dot_fp32_conventional(*a, *b)
        optimized = bf16x8_dot_fp32_optimized(*a, *b)
        
        report1 = run_spec_with_metrics(csa_tree4)
        report2 = run_spec_with_metrics(conventional)
        report3 = run_spec_with_metrics(optimized)
        
        overall_report = merge_spec_reports([report1, report2, report3])
        TestFusedDotProduct.SPEC_REPORT = overall_report
        
        pprint(overall_report)
        self.assertTrue(overall_report["proved"], pformat(overall_report))
    
    def test_designs_difference_with_fp_spec(self):
        SEED = self.SEED
        N_POINTS = self.N_POINTS
        TOTAL_POINTS = 0
        conventional_runtime_s = 0.0
        optimized_runtime_s = 0.0
        
        # Compile designs
        a = [
            Var(name="a_0", sign=BFloat16T()),
            Var(name="a_1", sign=BFloat16T()),
            Var(name="a_2", sign=BFloat16T()),
            Var(name="a_3", sign=BFloat16T()),
        ]
        
        b = [
            Var(name="b_0", sign=BFloat16T()),
            Var(name="b_1", sign=BFloat16T()),
            Var(name="b_2", sign=BFloat16T()),
            Var(name="b_3", sign=BFloat16T()),
        ]
        
        conventional = bf16x8_dot_fp32_conventional(*a, *b)
        optimized = bf16x8_dot_fp32_optimized(*a, *b)
        
        N = 4
        
        for shared_bits in range(5, BFloat16.exponent_bits+1):
            TOTAL_POINTS += N_POINTS
            
            random_gen, exp_reshuffle = BFloat16.random_generator(seed=SEED, shared_exponent_bits=shared_bits)
            
            for _ in range(N_POINTS):
                exp_reshuffle()
                for i in range(N):
                    a[i].load_val(random_gen())
                    b[i].load_val(random_gen())
                
                t0 = time.perf_counter()
                con_res = conventional.evaluate().to_val()
                conventional_runtime_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                opt_res = optimized.evaluate().to_val()
                optimized_runtime_s += time.perf_counter() - t0

                spec_res = dot_product_spec(*a, *b)
                msg = (
                    f"Mismatch at pt:\nf{[x.val.to_val() for x in a + b]}",
                    f"optimized impl={opt_res}, conventional impl={con_res}, double-precision spec={spec_res}"
                )
                if ulp_distance(opt_res, con_res) != 0 or ulp_distance(opt_res, spec_res) != 0:
                    self.fail(str(msg))
                    
        TestFusedDotProduct.IMPL_REPORT = {
            "seed": SEED,
            "total_num_points": TOTAL_POINTS,
            "conventional_runtime_per_point": conventional_runtime_s/TOTAL_POINTS,
            "optimized_runtime_per_point": optimized_runtime_s/TOTAL_POINTS,
            "conventional_runtime_s_total": conventional_runtime_s,
            "optimized_runtime_s_total": optimized_runtime_s,
        }

        pprint(TestFusedDotProduct.IMPL_REPORT)

    def test_cpp_lowering_performance_multiplier(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
        design = fp32_mult(x, y)
        
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)
        
        try:
            jit_runtime = 0.0
            no_jit_runtime = 0.0
            reference_runtime = 0.0
            
            rnd = random.Random(self.SEED)
            for _ in range(self.N_POINTS):
                x_bits = rnd.getrandbits(32)
                y_bits = rnd.getrandbits(32)
                x_fp = float(np.float32(Float32(x_bits).to_val()))
                y_fp = float(np.float32(Float32(y_bits).to_val()))
                
                t0 = time.perf_counter()
                jit_bits = fn_jit(x_bits, y_bits)
                jit_runtime += time.perf_counter() - t0
                jit_fp32 = float(np.float32(Float32(jit_bits).to_val()))
                
                t0 = time.perf_counter()
                no_jit_bits = fn_no_jit(x_bits, y_bits)
                no_jit_runtime += time.perf_counter() - t0
                no_jit_fp32 = float(np.float32(Float32(no_jit_bits).to_val()))
                
                t0 = time.perf_counter()
                reference_fp64 = x_fp * y_fp
                reference_runtime += time.perf_counter() - t0
                reference_fp32 = float(np.float32(reference_fp64))
                
                with self.subTest(lhs=x_fp, rhs=y_fp):
                    self.assertEqual(ulp_distance(reference_fp32, jit_fp32), 0, msg=f"{reference_fp32} != {jit_fp32}")
                    self.assertEqual(ulp_distance(jit_fp32, no_jit_fp32), 0, msg=f"{jit_fp32} != {no_jit_fp32}")
                
            print(
                "cpp_lowering_performance_mult:",
                {
                    "jit_total": jit_runtime,
                    "no_jit_total": no_jit_runtime,
                    "reference_total": reference_runtime,
                    "jit_per_point": jit_runtime / self.N_POINTS,
                    "no_jit_per_point": no_jit_runtime / self.N_POINTS,
                    "reference_per_point": reference_runtime / self.N_POINTS,
                },
            )
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()


    def test_cpp_lowering_performance_adder(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
        design = fp32_add(x, y)
        
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)
        
        try:
            jit_runtime = 0.0
            no_jit_runtime = 0.0
            reference_runtime = 0.0
            
            rnd = random.Random(self.SEED)
            for _ in range(self.N_POINTS):
                x_bits = rnd.getrandbits(32)
                y_bits = rnd.getrandbits(32)
                x_fp = float(np.float32(Float32(x_bits).to_val()))
                y_fp = float(np.float32(Float32(y_bits).to_val()))
                
                t0 = time.perf_counter()
                jit_bits = fn_jit(x_bits, y_bits)
                jit_runtime += time.perf_counter() - t0
                jit_fp32 = float(np.float32(Float32(jit_bits).to_val()))
                
                t0 = time.perf_counter()
                no_jit_bits = fn_no_jit(x_bits, y_bits)
                no_jit_runtime += time.perf_counter() - t0
                no_jit_fp32 = float(np.float32(Float32(no_jit_bits).to_val()))
                
                t0 = time.perf_counter()
                reference_fp64 = x_fp + y_fp
                reference_runtime += time.perf_counter() - t0
                reference_fp32 = float(np.float32(reference_fp64))
                
                with self.subTest(lhs=x_fp, rhs=y_fp):
                    self.assertEqual(ulp_distance(reference_fp32, jit_fp32), 0, msg=f"{reference_fp32} != {jit_fp32}")
                    self.assertEqual(ulp_distance(jit_fp32, no_jit_fp32), 0, msg=f"{jit_fp32} != {no_jit_fp32}")
                
            print(
                "cpp_lowering_performance_adder:",
                {
                    "jit_total": jit_runtime,
                    "no_jit_total": no_jit_runtime,
                    "reference_total": reference_runtime,
                    "jit_per_point": jit_runtime / self.N_POINTS,
                    "no_jit_per_point": no_jit_runtime / self.N_POINTS,
                    "reference_per_point": reference_runtime / self.N_POINTS,
                },
            )
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()
        
    def test_cpp_lowering_via_jit_adder(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
    
        design = fp32_add(x, y)
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        try:
            rnd = random.Random(self.SEED)
            for _ in range(self.N_POINTS):
                x.load_rand(rnd)
                y.load_rand(rnd)
                with self.subTest(lhs=x.val, rhs=y.val):
                    expected = design.evaluate().val
                    self.assertEqual(fn_jit(x.val.val, y.val.val), expected)
                    self.assertEqual(fn_no_jit(x.val.val, y.val.val), expected)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

    def test_fp32_adder_zero_handling(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
        design = fp32_add(x, y)
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        cases = [
            # IEEE round-to-nearest produces -0 only when both zero inputs are -0.
            ("+0 + +0", 0x00000000, 0x00000000, 0x00000000),
            ("+0 + -0", 0x00000000, 0x80000000, 0x00000000),
            ("-0 + +0", 0x80000000, 0x00000000, 0x00000000),
            ("-0 + -0", 0x80000000, 0x80000000, 0x80000000),
            # Either zero sign must act as an identity for nonzero finite values.
            ("+0 + -1", 0x00000000, 0xbf800000, 0xbf800000),
            ("-0 + -1", 0x80000000, 0xbf800000, 0xbf800000),
            ("+0 + +1", 0x00000000, 0x3f800000, 0x3f800000),
            ("-0 + +1", 0x80000000, 0x3f800000, 0x3f800000),
            ("+0 + min-subnormal", 0x00000000, 0x00000001, 0x00000001),
            ("-0 + min-subnormal", 0x80000000, 0x00000001, 0x00000001),
            ("+0 + -min-subnormal", 0x00000000, 0x80000001, 0x80000001),
            ("-0 + -min-subnormal", 0x80000000, 0x80000001, 0x80000001),
            # Exact cancellation must pass an exact zero through fp32_encode as +0.
            ("min-subnormal cancellation", 0x00000001, 0x80000001, 0x00000000),
            ("reversed min-subnormal cancellation", 0x80000001, 0x00000001, 0x00000000),
            ("largest-subnormal cancellation", 0x007fffff, 0x807fffff, 0x00000000),
            ("min-normal cancellation", 0x00800000, 0x80800000, 0x00000000),
            # Results immediately around the subnormal/normal boundary.
            ("two min-subnormals", 0x00000001, 0x00000001, 0x00000002),
            ("two negative min-subnormals", 0x80000001, 0x80000001, 0x80000002),
            ("min-normal minus largest-subnormal", 0x00800000, 0x807fffff, 0x00000001),
            ("largest-subnormal minus min-normal", 0x007fffff, 0x80800000, 0x80000001),
            ("largest-subnormal plus min-subnormal", 0x007fffff, 0x00000001, 0x00800000),
            ("negative boundary carry", 0x807fffff, 0x80000001, 0x80800000),
        ]

        try:
            for name, lhs_bits, rhs_bits, expected_bits in cases:
                x.load_val(Float32(lhs_bits))
                y.load_val(Float32(rhs_bits))
                with self.subTest(name=name, lhs=hex(lhs_bits), rhs=hex(rhs_bits)):
                    self.assertEqual(design.evaluate().val, expected_bits)
                    self.assertEqual(fn_jit(lhs_bits, rhs_bits), expected_bits)
                    self.assertEqual(fn_no_jit(lhs_bits, rhs_bits), expected_bits)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

    def test_fp32_adder_exponent_alignment_regression(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
        design = fp32_add(x, y)
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        cases = [
            (0x40000000, 0x3fffffff, 0x40800000),
            (0x3fffffff, 0x40000000, 0x40800000),
        ]

        try:
            for lhs_bits, rhs_bits, expected_bits in cases:
                x.load_val(Float32(lhs_bits))
                y.load_val(Float32(rhs_bits))
                with self.subTest(lhs=hex(lhs_bits), rhs=hex(rhs_bits)):
                    self.assertEqual(design.evaluate().val, expected_bits)
                    self.assertEqual(fn_jit(lhs_bits, rhs_bits), expected_bits)
                    self.assertEqual(fn_no_jit(lhs_bits, rhs_bits), expected_bits)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

    def test_fp32_adder_nan_handling(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
        design = fp32_add(x, y)
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        cases = [
            (0x7f800000, 0xff800000, 0x7fc00000),
            (0x7fc00000, 0x3f800000, 0x7fc00000),
        ]

        try:
            for lhs_bits, rhs_bits, expected_bits in cases:
                x.load_val(Float32(lhs_bits))
                y.load_val(Float32(rhs_bits))
                with self.subTest(lhs=hex(lhs_bits), rhs=hex(rhs_bits)):
                    self.assertEqual(design.evaluate().val, expected_bits)
                    self.assertEqual(fn_jit(lhs_bits, rhs_bits), expected_bits)
                    self.assertEqual(fn_no_jit(lhs_bits, rhs_bits), expected_bits)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

    def test_fp32_adder_infinity_handling(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
        design = fp32_add(x, y)
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        cases = [
            (Float32.nInf().val, Float32.from_fields(0, 127, 0).val, Float32.nInf().val),
            (Float32.from_fields(1, 127, 0).val, Float32.nInf().val, Float32.nInf().val),
            (Float32.nInf().val, Float32.nInf().val, Float32.nInf().val),
            (Float32.Inf().val, Float32.Inf().val, Float32.Inf().val),
            (Float32.nInf().val, Float32.Inf().val, Float32.NaN().val),
        ]

        try:
            for lhs_bits, rhs_bits, expected_bits in cases:
                x.load_val(Float32(lhs_bits))
                y.load_val(Float32(rhs_bits))
                with self.subTest(lhs=hex(lhs_bits), rhs=hex(rhs_bits)):
                    self.assertEqual(design.evaluate().val, expected_bits)
                    self.assertEqual(fn_jit(lhs_bits, rhs_bits), expected_bits)
                    self.assertEqual(fn_no_jit(lhs_bits, rhs_bits), expected_bits)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

    def test_cpp_lowering_via_jit_conventional(self):
        a = [
            Var(name="a_0", sign=BFloat16T()),
            Var(name="a_1", sign=BFloat16T()),
            Var(name="a_2", sign=BFloat16T()),
            Var(name="a_3", sign=BFloat16T()),
        ]
        
        b = [
            Var(name="b_0", sign=BFloat16T()),
            Var(name="b_1", sign=BFloat16T()),
            Var(name="b_2", sign=BFloat16T()),
            Var(name="b_3", sign=BFloat16T()),
        ]
        
        design = bf16x8_dot_fp32_conventional(*a, *b)
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        try:
            random_gen, exp_shuffle = BFloat16.random_generator(seed=self.SEED)
            for _ in range(self.N_POINTS):
                exp_shuffle()
                args = []
                for i in range(4):
                    val = random_gen()
                    a[i].load_val(val)
                    args.append(val.val)
                for i in range(4):
                    val = random_gen()
                    b[i].load_val(val)
                    args.append(val.val)
                    
                with self.subTest(a=a, b=b):
                    expected = design.evaluate().val
                    self.assertEqual(fn_jit(*args), expected)
                    self.assertEqual(fn_no_jit(*args), expected)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

            
    def test_cpp_lowering_via_jit_csa(self):
        rnd = random.Random(self.SEED)
        args = [Var(f"arg_{i}", sign=QT(rnd.randint(1, 20), rnd.randint(1, 20))) for i in range(4)]
        
        design = CSA_tree4(*args)
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)
        try:
            for _ in range(self.N_POINTS):
                call_args = []
                for arg in args:
                    arg.load_rand(rnd)
                    call_args.append(arg.val.val)
                    
                with self.subTest(args=call_args):
                    expected = design.evaluate().val
                    self.assertEqual(fn_jit(*call_args), expected)
                    self.assertEqual(fn_no_jit(*call_args), expected)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()


    def test_cpp_lowering_via_jit_optimized(self):
        a = [
            Var(name="a_0", sign=BFloat16T()),
            Var(name="a_1", sign=BFloat16T()),
            Var(name="a_2", sign=BFloat16T()),
            Var(name="a_3", sign=BFloat16T()),
        ]
        
        b = [
            Var(name="b_0", sign=BFloat16T()),
            Var(name="b_1", sign=BFloat16T()),
            Var(name="b_2", sign=BFloat16T()),
            Var(name="b_3", sign=BFloat16T()),
        ]
        
        design = bf16x8_dot_fp32_optimized(*a, *b)
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        try:
            random_gen, exp_shuffle = BFloat16.random_generator(seed=self.SEED)
            for _ in range(self.N_POINTS):
                exp_shuffle()
                args = []
                for i in range(4):
                    val = random_gen()
                    a[i].load_val(val)
                    args.append(val.val)
                for i in range(4):
                    val = random_gen()
                    b[i].load_val(val)
                    args.append(val.val)
                    
                with self.subTest(a=a, b=b):
                    expected = design.evaluate().val
                    self.assertEqual(fn_jit(*args), expected)
                    self.assertEqual(fn_no_jit(*args), expected)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

    def test_cpp_lowering_via_jit_max_exponent(self):
        rnd = random.Random(self.SEED)
        args = [Var(f"arg_{i}", sign=UQT(rnd.randint(1, 10), 0)) for i in range(4)]
        
        design = OPTIMIZED_MAX_EXP4(*args)
        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)
        try:
            for _ in range(self.N_POINTS):
                call_args = []
                for arg in args:
                    arg.load_rand(rnd)
                    call_args.append(arg.val.val)
                    
                with self.subTest(args=call_args):
                    expected = design.evaluate().val
                    self.assertEqual(fn_jit(*call_args), expected)
                    self.assertEqual(fn_no_jit(*call_args), expected)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()


def build_unittest_report(seed: int, spec_report: dict, impl_report: dict):
    return {
        "seed": seed,
        "run_spec_report": spec_report,
        "impl_report": impl_report,
    }


def write_unittest_report(path: str, report: dict):
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unittests for fused dot product desings")
    parser.add_argument("-s", "--seed", help="Random seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("-pt", "--num-points", help=f"Number of points to run per shared exponent", default=DEFAULT_N_POINTS, type=int)
    parser.add_argument("--json-report", help="Write unittest summary report to this JSON file", default=None)
    
    args, unittest_args = parser.parse_known_args()

    TestFusedDotProduct.SEED = args.seed
    TestFusedDotProduct.N_POINTS = args.num_points
    TestFusedDotProduct.rnd = random.Random(args.seed)
    
    program = unittest.main(argv=[__file__, *unittest_args], exit=False)
    
    if args.json_report:
        report = build_unittest_report(
            seed=args.seed,
            spec_report=TestFusedDotProduct.SPEC_REPORT,
            impl_report=TestFusedDotProduct.IMPL_REPORT,
        )
        write_unittest_report(args.json_report, report)

    exit_code = 0 if program.result.wasSuccessful() else 1
    sys.exit(exit_code)
