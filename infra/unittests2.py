import unittest
import contextlib
import os
import pickle
import sys
import time
from unittest.mock import Mock, patch

import dreal
import math
import z3
from egglog import EGraph

from zolotone import *
from zolotone.ast import nodes as ast_nodes
from zolotone.egglog.rules import load_rules
from zolotone.smt import dreal_check_eq, z3_check_eq
from zolotone.solver import engine as solver_engine
from zolotone.solver.report import build_proof_report
from zolotone.rival import (
    RivalAnalysis,
    build_machine,
    collect_free_vars,
    get_rival_rects,
    rival_feasibility_check,
    rival_trim_context,
    to_rival_ir,
)
from zolotone.spec.spec_context import simplify_ctx
from zolotone.spec.spec_utils import from_egglog
from examples.fp32_add import fp32_add
from examples.fp32_mult import fp32_mult
from examples.common import xor_spec
from examples.conventional import Conventional
from examples.optimized import Optimized

from infra.compile_cpp import jit_compile, nonjit_compile


def _flat_trace_tool(ctx, timeout_ms):
    del timeout_ms
    report1 = build_proof_report(
        ctx,
        ctx.copy(),
        tool="branch-a",
        runtime_s=0.0,
        status="unknown",
    )
    report2 = build_proof_report(
        ctx,
        ctx.copy(),
        tool="branch-b",
        runtime_s=0.0,
        status="unsat",
    )
    return [report1, report2]


def _slow_tool(_ctx, timeout_ms):
    del timeout_ms
    time.sleep(1)


def _large_report_tool(ctx, timeout_ms):
    del timeout_ms
    return build_proof_report(
        ctx,
        ctx.copy(),
        tool="z3",
        runtime_s=0.0,
        status="unknown",
        supplementary_info=b"x" * 2048,
    )


class TestConstantFolding(unittest.TestCase):
    def assert_folded_value(self, node, runtime_type, expected_val):
        self.assertIsNotNone(node.node_type.runtime_val)
        self.assertIsInstance(node.node_type.runtime_val, runtime_type)
        self.assertEqual(node.node_type.runtime_val.val, expected_val)

        evaluated = node.evaluate()
        self.assertIsInstance(evaluated, runtime_type)
        self.assertEqual(evaluated.val, expected_val)

    def test_float32_nan_default_is_quiet(self):
        self.assertEqual(Float32.NaN().val, 0x7FC00000)
        self.assertEqual(Float32.NaN(1).val, 0x7F800001)
        
    def test_cpp_lowering_widening_add_sub(self):
        add_x = Var(name="add_x", sign=UQT(32, 0))
        add_y = Var(name="add_y", sign=UQT(32, 0))
        add_design = uq_add(add_x, add_y)

        sub_x = Var(name="sub_x", sign=UQT(32, 0))
        sub_y = Var(name="sub_y", sign=UQT(32, 0))
        sub_design = uq_sub(sub_x, sub_y)

        add_tempdir_jit, add_fn_jit = jit_compile(add_design)
        add_tempdir_no_jit, add_fn_no_jit = nonjit_compile(add_design)
        sub_tempdir_jit, sub_fn_jit = jit_compile(sub_design)
        sub_tempdir_no_jit, sub_fn_no_jit = nonjit_compile(sub_design)

        add_cases = [
            (0xFFFF_FFFF, 0x0000_0001, 0x1_0000_0000),
            (0x8000_0000, 0x8000_0000, 0x1_0000_0000),
        ]
        sub_cases = [
            (0x0000_0000, 0x0000_0001, 0x1_FFFF_FFFF),
            (0x8000_0000, 0x0000_0001, 0x07FFF_FFFF),
        ]

        try:
            for lhs_bits, rhs_bits, expected in add_cases:
                add_x.load_val(UQ(lhs_bits, 32, 0))
                add_y.load_val(UQ(rhs_bits, 32, 0))
                with self.subTest(op="uq_add", lhs=hex(lhs_bits), rhs=hex(rhs_bits)):
                    self.assertEqual(add_design.evaluate().val, expected)
                    self.assertEqual(add_fn_jit(lhs_bits, rhs_bits), expected)
                    self.assertEqual(add_fn_no_jit(lhs_bits, rhs_bits), expected)

            for lhs_bits, rhs_bits, expected in sub_cases:
                sub_x.load_val(UQ(lhs_bits, 32, 0))
                sub_y.load_val(UQ(rhs_bits, 32, 0))
                with self.subTest(op="uq_sub", lhs=hex(lhs_bits), rhs=hex(rhs_bits)):
                    self.assertEqual(sub_design.evaluate().val, expected)
                    self.assertEqual(sub_fn_jit(lhs_bits, rhs_bits), expected)
                    self.assertEqual(sub_fn_no_jit(lhs_bits, rhs_bits), expected)
        finally:
            add_tempdir_jit.cleanup()
            add_tempdir_no_jit.cleanup()
            sub_tempdir_jit.cleanup()
            sub_tempdir_no_jit.cleanup()


    def test_fp32_multiplier_special_cases(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
        design = fp32_mult(x, y)

        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        cases = [
            (Float32.Inf().val, Float32.Zero().val, Float32.NaN().val),
            (Float32.nInf().val, Float32.Zero().val, Float32.NaN().val),
            (Float32.Inf().val, Float32.from_fields(0, 128, 0).val, Float32.Inf().val),
            (Float32.nInf().val, Float32.from_fields(0, 128, 0).val, Float32.nInf().val),
            (Float32.nZero().val, Float32.from_fields(0, 128, 1 << 22).val, Float32.nZero().val),
            (Float32.nZero().val, Float32.from_fields(1, 128, 1 << 22).val, Float32.Zero().val),
            (Float32.NaN().val, Float32.from_fields(0, 128, 0).val, Float32.NaN().val),
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

    def test_fp32_multiplier_zero_handling(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
        design = fp32_mult(x, y)

        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        cases = [
            # Zero products use the XOR of the operand signs.
            ("+0 * +0", 0x00000000, 0x00000000, 0x00000000),
            ("+0 * -0", 0x00000000, 0x80000000, 0x80000000),
            ("-0 * +0", 0x80000000, 0x00000000, 0x80000000),
            ("-0 * -0", 0x80000000, 0x80000000, 0x00000000),
            ("+0 * +1", 0x00000000, 0x3f800000, 0x00000000),
            ("+0 * -1", 0x00000000, 0xbf800000, 0x80000000),
            ("-0 * +1", 0x80000000, 0x3f800000, 0x80000000),
            ("-0 * -1", 0x80000000, 0xbf800000, 0x00000000),
            ("+1 * -0", 0x3f800000, 0x80000000, 0x80000000),
            ("-1 * -0", 0xbf800000, 0x80000000, 0x00000000),
            ("+0 * min-subnormal", 0x00000000, 0x00000001, 0x00000000),
            ("-0 * min-subnormal", 0x80000000, 0x00000001, 0x80000000),
            ("-0 * -min-subnormal", 0x80000000, 0x80000001, 0x00000000),
            # Exact and rounded results at the bottom of the subnormal range.
            ("min-subnormal * +1", 0x00000001, 0x3f800000, 0x00000001),
            ("min-subnormal * -1", 0x00000001, 0xbf800000, 0x80000001),
            ("positive half-minimum tie", 0x00000001, 0x3f000000, 0x00000000),
            ("negative half-minimum tie", 0x80000001, 0x3f000000, 0x80000000),
            ("positive above-half minimum", 0x00000001, 0x3f400000, 0x00000001),
            ("negative above-half minimum", 0x80000001, 0x3f400000, 0x80000001),
            ("min-subnormal times 1.5 tie-to-even", 0x00000001, 0x3fc00000, 0x00000002),
            # Results around the subnormal/normal boundary.
            ("min-normal halved", 0x00800000, 0x3f000000, 0x00400000),
            ("min-normal times 2^-23", 0x00800000, 0x34000000, 0x00000001),
            ("min-normal times 2^-24 tie", 0x00800000, 0x33800000, 0x00000000),
            ("negative min-normal times 2^-24 tie", 0x80800000, 0x33800000, 0x80000000),
            ("normal-boundary tie-to-even", 0x00800000, 0x3f7fffff, 0x00800000),
            ("largest-subnormal doubled", 0x007fffff, 0x40000000, 0x00fffffe),
            # Zero times infinity is invalid regardless of either sign or order.
            ("+0 * +inf", 0x00000000, 0x7f800000, 0x7fc00000),
            ("-0 * +inf", 0x80000000, 0x7f800000, 0x7fc00000),
            ("+0 * -inf", 0x00000000, 0xff800000, 0x7fc00000),
            ("-0 * -inf", 0x80000000, 0xff800000, 0x7fc00000),
            ("+inf * -0", 0x7f800000, 0x80000000, 0x7fc00000),
            ("-inf * +0", 0xff800000, 0x00000000, 0x7fc00000),
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

    def test_basic_add_folds_for_float_bit_types(self):
        cases = [
            ("float32", Float32),
            ("bfloat16", BFloat16),
        ]

        for name, runtime_type in cases:
            with self.subTest(runtime_type=name):
                folded = basic_add(
                    Const(runtime_type(1)),
                    Const(runtime_type(2)),
                    out=Const(runtime_type(0)),
                )

                self.assert_folded_value(folded, runtime_type, 3)

    def test_basic_add_masks_to_output_width(self):
        folded = basic_add(
            Const(UQ(7, 3, 0)),
            Const(UQ(3, 3, 0)),
            out=Const(UQ(0, 3, 0)),
        )

        self.assert_folded_value(folded, UQ, 2)

    def test_comparison_folds_to_bool(self):
        folded = basic_less(
            Const(UQ(1, 2, 0)),
            Const(UQ(2, 2, 0)),
            out=Const(Bool(0)),
        )

        self.assert_folded_value(folded, Bool, 1)

    def test_tuple_get_item_is_constant_folded(self):
        folded = make_Tuple(
            Const(UQ(5, 3, 0)),
            Const(Bool(1)),
        )[0]

        self.assert_folded_value(folded, UQ, 5)

    def test_if_then_else_is_constant_folded(self):
        folded = if_then_else(
            Const(Bool(1)),
            Const(UQ(5, 3, 0)),
            Const(UQ(2, 3, 0)),
        )

        self.assert_folded_value(folded, UQ, 5)

    def test_non_constant_inputs_do_not_fold(self):
        x = Var("x", sign=UQT(3, 0))
        node = basic_add(
            x,
            Const(UQ(1, 3, 0)),
            out=Const(UQ(0, 3, 0)),
        )

        self.assertIsNone(node.node_type.runtime_val)

        x.load_val(UQ(6, 3, 0))
        self.assertEqual(node.evaluate().val, 7)
        self.assertIsNone(node.node_type.runtime_val)

    def test_uq_rshift_jam_sets_sticky_when_shifted_out_bits_are_nonzero(self):
        cases = [
            (8, 1, 4),
            (9, 1, 5),
        ]

        for x_val, amount_val, expected in cases:
            with self.subTest(x=x_val, amount=amount_val):
                folded = uq_rshift_jam(
                    Const(UQ(x_val, 4, 0)),
                    Const(UQ.from_int(amount_val)),
                )
                self.assert_folded_value(folded, UQ, expected)

    def test_conventional_and_optimized_match_on_cancellation_regression(self):
        vals = [43160, 10458, 11062, 10989, 10589, 10469, 11020, 11013]
        a = [Var(name=f"a_{i}", sign=BFloat16T()) for i in range(4)]
        b = [Var(name=f"b_{i}", sign=BFloat16T()) for i in range(4)]

        for i, bits in enumerate(vals[:4]):
            a[i].load_val(BFloat16(bits))
        for i, bits in enumerate(vals[4:]):
            b[i].load_val(BFloat16(bits))

        conventional = Conventional(*a, *b).evaluate()
        optimized = Optimized(*a, *b).evaluate()

        self.assertEqual(conventional.val, 388040612)
        self.assertEqual(conventional, optimized)


class TestFingerprint(unittest.TestCase):
    def test_runtime_type_fingerprint_depends_on_structure_and_value(self):
        self.assertEqual(UQ(3, 3, 0)._fingerprint(), UQ(3, 3, 0)._fingerprint())
        self.assertNotEqual(UQ(3, 3, 0)._fingerprint(), UQ(4, 3, 0)._fingerprint())
        self.assertNotEqual(UQ(3, 3, 0)._fingerprint(), UQ(3, 4, 0)._fingerprint())

        lhs = Tuple(UQ(5, 3, 0), Bool(1))
        rhs = Tuple(UQ(5, 3, 0), Bool(1))
        different = Tuple(UQ(5, 3, 0), Bool(0))
        self.assertEqual(lhs._fingerprint(), rhs._fingerprint())
        self.assertNotEqual(lhs._fingerprint(), different._fingerprint())

    def test_static_type_fingerprint_includes_runtime_val(self):
        plain = UQT(3, 0)
        same = UQT(3, 0)
        self.assertEqual(plain._fingerprint(), same._fingerprint())

        with_runtime = UQT(3, 0)
        with_runtime.runtime_val = UQ(1, 3, 0)
        same_runtime = UQT(3, 0)
        same_runtime.runtime_val = UQ(1, 3, 0)
        different_runtime = UQT(3, 0)
        different_runtime.runtime_val = UQ(2, 3, 0)

        self.assertEqual(with_runtime._fingerprint(), same_runtime._fingerprint())
        self.assertNotEqual(plain._fingerprint(), with_runtime._fingerprint())
        self.assertNotEqual(with_runtime._fingerprint(), different_runtime._fingerprint())

    def test_equivalent_graphs_have_equal_fingerprints(self):
        x1 = Var("x", sign=UQT(3, 0))
        y1 = Var("y", sign=UQT(3, 0))
        x2 = Var("x", sign=UQT(3, 0))
        y2 = Var("y", sign=UQT(3, 0))

        lhs = basic_add(x1, y1, Const(UQ(0, 3, 0)))
        rhs = basic_add(x2, y2, Const(UQ(0, 3, 0)))

        self.assertEqual(lhs._fingerprint(), rhs._fingerprint())

    def test_graph_fingerprint_changes_with_operation_and_constants(self):
        x = Var("x", sign=UQT(3, 0))
        y = Var("y", sign=UQT(3, 0))

        add_node = basic_add(x, y, Const(UQ(0, 3, 0)))
        sub_node = basic_sub(x, y, Const(UQ(0, 3, 0)))
        different_const = basic_add(x, y, Const(UQ(1, 3, 0)))

        self.assertNotEqual(add_node._fingerprint(), sub_node._fingerprint())
        self.assertNotEqual(add_node._fingerprint(), different_const._fingerprint())

    def test_fingerprint_depends_on_jittable_lowering_when_codegen_differs(self):
        node = make_Tuple(Const(UQ(1, 3, 0)), Const(UQ(2, 3, 0)))

        self.assertNotEqual(node._fingerprint(jittable=True), node._fingerprint(jittable=False))

    def test_fingerprint_is_stable_across_runtime_variable_bindings(self):
        x = Var("x", sign=UQT(3, 0))
        y = Var("y", sign=UQT(3, 0))
        node = basic_add(x, y, Const(UQ(0, 3, 0)))

        before = node._fingerprint()

        x.load_val(UQ(6, 3, 0))
        y.load_val(UQ(1, 3, 0))
        self.assertEqual(node.evaluate().val, 7)

        after = node._fingerprint()
        self.assertEqual(before, after)


class TestPowSpecOp(unittest.TestCase):
    def test_if_constant_fold_prunes_nonliteral_branch(self):
        x = RealVar("x")
        y = RealVar("y")

        self.assertEqual(
            If(BoolLit(False), x + RealLit(1), y + RealLit(2)).constant_fold(),
            Add(y, RealLit(2)),
        )
        self.assertEqual(
            If(BoolLit(True), x + RealLit(1), y + RealLit(2)).constant_fold(),
            Add(x, RealLit(1)),
        )

    def test_boolean_constant_fold_short_circuits_nonliteral_operands(self):
        p = BoolVar("p")

        self.assertEqual(
            And(BoolLit(False), p).constant_fold(),
            BoolLit(False),
        )
        self.assertEqual(
            And(BoolLit(True), p).constant_fold(),
            p,
        )
        self.assertEqual(
            Or(BoolLit(True), p).constant_fold(),
            BoolLit(True),
        )
        self.assertEqual(
            Or(BoolLit(False), p).constant_fold(),
            p,
        )

    def test_mul_by_zero_shortcuts_nonliteral_operand(self):
        x = RealVar("x")

        self.assertEqual(
            Mul(RealLit(0), x + RealLit(1)).constant_fold(),
            RealLit(0),
        )
        self.assertEqual(
            Mul(x + RealLit(1), RealLit(0)).constant_fold(),
            RealLit(0),
        )

    def test_pow_python_sugar_uses_negative_one_base(self):
        self.assertEqual(
            RealLit(-1) ** RealLit(3),
            Pow(RealLit(-1), RealLit(3)),
        )

    def test_pow_python_sugar_keeps_supported_base2_and_square_forms(self):
        self.assertEqual(
            RealLit(2) ** RealLit(3),
            Pow(RealLit(2), RealLit(3)),
        )
        self.assertEqual(
            RealLit(3) ** RealLit(2),
            Pow(RealLit(3), RealLit(2)),
        )

    def test_pow_round_trips_from_egglog(self):
        exprs = [
            Pow(RealLit(-1), RealLit(2)),
            Pow(RealLit(2), RealLit(3)),
            Pow(RealLit(3), RealLit(2)),
        ]

        for expr in exprs:
            with self.subTest(expr=str(expr)):
                self.assertEqual(from_egglog(expr.to_egglog()), expr)

    def test_pow_constant_folds_in_egglog(self):
        cases = [
            (Pow(RealLit(-1), RealLit(3)), RealLit(-1)),
            (Pow(RealLit(2), RealLit(3)), RealLit(8)),
            (Pow(RealLit(3), RealLit(2)), RealLit(9)),
        ]

        for expr, expected in cases:
            with self.subTest(expr=str(expr)):
                egraph = EGraph()
                load_rules(egraph)
                lowered = expr.to_egglog()

                egraph.register(lowered)
                egraph.run(1)

                self.assertEqual(from_egglog(egraph.extract(lowered)), expected)
    def test_minus_one_symbolic_power_lowers_as_plain_power(self):
        s = RealVar("s")
        expr = RealLit(-1) ** s

        z3_expr = expr.to_z3({})
        dreal_expr = expr.to_dreal({})

        self.assertNotEqual(z3_expr.decl().kind(), z3.Z3_OP_ITE)
        self.assertEqual(str(z3_expr), "-1**s")
        self.assertEqual(str(dreal_expr), "pow(-1, s)")
    def test_numeric_equality_constant_folds_in_egglog(self):
        cases = [
            (RealLit(3).eq(RealLit(3)), BoolLit(True)),
            (RealLit(3).ne(RealLit(4)), BoolLit(True)),
        ]

        for expr, expected in cases:
            with self.subTest(expr=str(expr)):
                egraph = EGraph()
                load_rules(egraph)
                lowered = expr.to_egglog()

                egraph.register(lowered)
                egraph.run(1)

                self.assertEqual(from_egglog(egraph.extract(lowered)), expected)

    def test_numeric_equality_constant_folds_in_simplify_mode(self):
        cases = [
            (RealLit(5).eq(RealLit(5)), BoolLit(True)),
            (RealLit(5).ne(RealLit(6)), BoolLit(True)),
        ]

        for expr, expected in cases:
            with self.subTest(expr=str(expr)):
                egraph = EGraph()
                load_rules(egraph, simplify=True)
                lowered = expr.to_egglog()

                egraph.register(lowered)
                egraph.run(1)

                self.assertEqual(from_egglog(egraph.extract(lowered)), expected)

    def test_context_simplify_uses_if_branch_pruning(self):
        ctx = SpecContext("if-pruning")
        x = ctx.real("x")

        ctx.check(If(BoolLit(False), x + RealLit(1), RealLit(3)).eq(RealLit(3)))

        simplified = ctx.simplify()
        self.assertEqual(simplified.checks, [])

    def test_context_simplify_uses_mul_by_zero_shortcut(self):
        ctx = SpecContext("mul-zero")
        x = ctx.real("x")

        ctx.check((RealLit(0) * (x + RealLit(5))).eq(RealLit(0)))

        simplified = ctx.simplify()
        self.assertEqual(simplified.checks, [])


class TestSpecContextLearning(unittest.TestCase):
    def test_learned_literals_reads_real_var_equalities(self):
        ctx = SpecContext("learn-real")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(ctx.real_val(1)))
        ctx.assume(ctx.real_val(2).eq(y))

        learned = ctx.learned_literals()
        self.assertEqual(len(learned), 2)
        self.assertEqual(learned[x], RealLit(1))
        self.assertEqual(learned[y], RealLit(2))

    def test_learned_literals_reads_foldable_real_equalities(self):
        ctx = SpecContext("learn-real-foldable")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(ctx.real_val(2) + ctx.real_val(3)))
        ctx.assume((ctx.real_val(10) - ctx.real_val(4)).eq(y))

        learned = ctx.learned_literals()
        self.assertEqual(len(learned), 2)
        self.assertEqual(learned[x], RealLit(5))
        self.assertEqual(learned[y], RealLit(6))

    def test_learned_literals_reads_bool_var_equalities(self):
        ctx = SpecContext("learn-bool")
        p = ctx.bool("p")
        q = ctx.bool("q")

        ctx.assume(p.eq(ctx.true()))
        ctx.assume(ctx.false().eq(q))

        learned = ctx.learned_literals()
        self.assertEqual(len(learned), 2)
        self.assertEqual(learned[p], BoolLit(True))
        self.assertEqual(learned[q], BoolLit(False))

    def test_learned_literals_reads_foldable_bool_equalities(self):
        ctx = SpecContext("learn-bool-foldable")
        p = ctx.bool("p")
        q = ctx.bool("q")

        ctx.assume(p.eq(ctx.real_val(2).eq(ctx.real_val(2))))
        ctx.assume((ctx.real_val(2).eq(ctx.real_val(3))).eq(q))

        learned = ctx.learned_literals()
        self.assertEqual(len(learned), 2)
        self.assertEqual(learned[p], BoolLit(True))
        self.assertEqual(learned[q], BoolLit(False))

    def test_learned_literals_reads_non_literal_equalities_as_boolean_facts(self):
        ctx = SpecContext("learn-ignore")
        x = ctx.real("x")
        y = ctx.real("y")
        p = ctx.bool("p")
        q = ctx.bool("q")

        ctx.assume(x.eq(y))
        ctx.assume(p.eq(q))

        self.assertEqual(
            ctx.learned_literals(),
            {
                x.eq(y): BoolLit(True),
                p.eq(q): BoolLit(True),
            },
        )

    def test_learned_literals_raises_on_conflicting_bindings(self):
        ctx = SpecContext("learn-conflict")
        x = ctx.real("x")
        p = ctx.bool("p")

        ctx.assume(x.eq(ctx.real_val(0)))
        ctx.assume(p.eq(ctx.true()))
        ctx.assume(x.eq(ctx.real_val(1)))

        with self.assertRaises(ValueError):
            ctx.learned_literals()

    def test_conflicting_assumptions_are_deferred_until_simplification(self):
        ctx = SpecContext("assume-deferred-conflict")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(ctx.real_val(0)))
        before = ctx.copy()
        ctx.assume(x.eq(ctx.real_val(1)))

        self.assertEqual(
            ctx.assumes,
            before.assumes + [Eq(x, RealLit(1))],
        )
        self.assertEqual(ctx.checks, before.checks)

        ctx.assume(y.eq(x))
        with self.assertRaises(ValueError):
            ctx.simplify()

        self.assertEqual(
            ctx.assumes,
            [Eq(x, RealLit(0)), Eq(x, RealLit(1)), Eq(y, x)],
        )

    def test_context_fixpoint_simplifies_assumptions_from_learned_literals(self):
        ctx = SpecContext("simplify-assumes")
        and_res = ctx.real("and_res")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(and_res.eq(x * y))
        ctx.assume(x.eq(ctx.real_val(0)))
        ctx.assume(y.eq(ctx.real_val(0)))

        simplified = ctx.simplify()

        self.assertEqual(
            ctx.assumes,
            [
                Eq(and_res, x * y),
                Eq(x, RealLit(0)),
                Eq(y, RealLit(0)),
            ],
        )
        self.assertEqual(
            simplified.assumes,
            [],
        )

    def test_context_fixpoint_inlines_non_literal_aliases(self):
        ctx = SpecContext("simplify-aliases")
        xor_res = ctx.real("xor_res")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(xor_res.eq(x + y))
        ctx.assume((xor_res * ctx.real_val(1)).eq(x + y))
        ctx.check((xor_res + ctx.real_val(1)).eq((x + y) + ctx.real_val(1)))

        simplified = ctx.simplify()

        self.assertNotIn("xor_res", str(simplified))
        self.assertEqual(simplified.assumes, [])
        self.assertEqual(simplified.checks, [])

    def test_context_fixpoint_preserves_duplicate_aliases_as_constraints(self):
        ctx = SpecContext("simplify-duplicate-aliases")
        alias = ctx.real("alias")
        y = ctx.real("y")
        z = ctx.real("z")

        ctx.assume(alias.eq(y + ctx.real_val(1)))
        ctx.assume(alias.eq(z + ctx.real_val(1)))

        simplified = ctx.simplify()

        self.assertEqual(
            simplified.assumes,
            [Eq(y + RealLit(1), z + RealLit(1))],
        )

    def test_context_fixpoint_keeps_self_referential_constraints(self):
        ctx = SpecContext("simplify-self-reference")
        x = ctx.real("x")

        ctx.assume(x.eq(abs(x)))
        ctx.check(x.eq(abs(x)))

        simplified = ctx.simplify()

        self.assertEqual(simplified.assumes, [Eq(x, Abs(x))])
        self.assertEqual(simplified.checks, [])

    def test_assume_records_alias_loops(self):
        ctx = SpecContext("assume-alias-loop")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(y + ctx.real_val(1)))
        ctx.assume(y.eq(x + ctx.real_val(1)))

        self.assertEqual(
            ctx.assumes,
            [
                Eq(x, y + RealLit(1)),
                Eq(y, x + RealLit(1)),
            ],
        )

    def test_check_records_alias_loops(self):
        ctx = SpecContext("check-alias-loop")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(y + ctx.real_val(1)))
        ctx.check(y.eq(x + ctx.real_val(1)))

        self.assertEqual(ctx.checks, [Eq(y, x + RealLit(1))])

    def test_context_fixpoint_substitutes_canonical_learned_assumptions(self):
        ctx = SpecContext("canonical-assumes")
        x = ctx.real("x")
        p = ctx.bool("p")

        ctx.assume(x.eq(ctx.real_val(1) + ctx.real_val(2)))
        ctx.assume(ctx.real_val(4).eq(x + ctx.real_val(1)))
        ctx.assume(p.eq(ctx.real_val(2).eq(ctx.real_val(2))))

        simplified = ctx.simplify()

        self.assertEqual(
            simplified.assumes,
            [],
        )
        self.assertEqual(simplified.learned_literals(), {})

    def test_context_fixpoint_propagates_through_multiple_rounds(self):
        ctx = SpecContext("simplify-multi-round")
        x = ctx.real("x")
        y = ctx.real("y")
        z = ctx.real("z")

        ctx.assume(x.eq(y + ctx.real_val(1)))
        ctx.assume(y.eq(z + ctx.real_val(1)))
        ctx.assume(z.eq(ctx.real_val(0)))

        simplified = ctx.simplify()

        self.assertEqual(
            simplified.assumes,
            [],
        )
        self.assertEqual(simplified.learned_literals(), {})

    def test_context_fixpoint_simplifies_checks_from_learned_literals(self):
        ctx = SpecContext("simplify-checks")
        x = ctx.real("x")

        ctx.assume(x.eq(ctx.real_val(1)))
        ctx.check((x + ctx.real_val(2)).eq(ctx.real_val(3)))

        simplified = ctx.simplify()

        self.assertEqual(ctx.checks, [Eq(x + RealLit(2), RealLit(3))])
        self.assertEqual(simplified.checks, [])

    def test_context_fixpoint_simplifies_bool_assumptions_and_checks(self):
        ctx = SpecContext("simplify-bool")
        p = ctx.bool("p")
        q = ctx.bool("q")

        ctx.assume(p.eq(ctx.true()))
        ctx.assume(q.eq(p))
        ctx.check(q.eq(ctx.true()))

        simplified = ctx.simplify()

        self.assertEqual(ctx.assumes, [BoolEq(p, BoolLit(True)), BoolEq(q, p)])
        self.assertEqual(simplified.assumes, [])
        self.assertEqual(simplified.checks, [])
        self.assertEqual(simplified.learned_literals(), {})

    def test_context_fixpoint_learns_negated_compound_boolean_fact(self):
        ctx = SpecContext("simplify-negated-compound-bool")
        x = ctx.real("x")
        overflow = abs(x) > ctx.real_val(1)

        ctx.assume(overflow.eq(ctx.false()))
        ctx.check(overflow.eq(ctx.false()))

        simplified = ctx.simplify()

        self.assertEqual(simplified.assumes, [~overflow])
        self.assertEqual(simplified.checks, [])
        self.assertEqual(
            simplified.learned_literals(),
            {overflow: BoolLit(False)},
        )

    def test_context_fixpoint_learns_positive_compound_boolean_fact(self):
        ctx = SpecContext("simplify-positive-compound-bool")
        x = ctx.real("x")
        in_range = (x >= ctx.real_val(-1)) & (x <= ctx.real_val(1))

        ctx.assume(in_range.eq(ctx.true()))
        ctx.check(in_range)

        simplified = ctx.simplify()

        self.assertEqual(simplified.assumes, [in_range])
        self.assertEqual(simplified.checks, [])
        self.assertEqual(
            simplified.learned_literals(),
            {in_range: BoolLit(True)},
        )

    def test_context_fixpoint_simplifies_assumptions_from_other_compound_facts(self):
        ctx = SpecContext("simplify-compound-assumptions")
        x = ctx.real("x")
        y = ctx.real("y")
        positive_x = x > ctx.real_val(0)
        positive_y = y > ctx.real_val(0)

        ctx.assume(positive_x)
        ctx.assume(positive_y)
        ctx.assume(~(positive_x & positive_y))

        with self.assertRaisesRegex(ValueError, "Assumption folds to false"):
            ctx.simplify()

    def test_context_fixpoint_keeps_one_anchor_for_duplicate_compound_facts(self):
        ctx = SpecContext("simplify-duplicate-compound")
        x = ctx.real("x")
        positive = x > ctx.real_val(0)

        ctx.assume(positive)
        ctx.assume(positive)
        ctx.check(positive)

        simplified = ctx.simplify()

        self.assertEqual(simplified.assumes, [positive])
        self.assertEqual(simplified.checks, [])

    def test_context_fixpoint_discharges_asserted_equalities(self):
        ctx = SpecContext("simplify-asserted-equalities")
        x = ctx.real("x")
        y = ctx.real("y")
        p = ctx.bool("p")
        q = ctx.bool("q")

        ctx.assume(x.eq(y))
        ctx.assume(p.eq(q))
        ctx.check(x.eq(y))
        ctx.check(p.eq(q))

        simplified = ctx.simplify()

        self.assertEqual(simplified.assumes, [x.eq(y), p.eq(q)])
        self.assertEqual(simplified.checks, [])

    def test_context_preserves_finite_if_after_alias_substitution(self):
        ctx = SpecContext("simplify-if-alias")
        selected = ctx.fresh_real("selected")
        condition = ctx.bool("condition")

        ctx.assume(selected.eq(If(condition, ctx.real_val(1), ctx.real_val(0))))
        ctx.assume((ctx.real_val(1) - selected).eq(ctx.real_val(1)))
        ctx.check(condition)

        simplified = ctx.simplify()

        self.assertEqual(
            simplified.assumes,
            [
                (
                    ctx.real_val(1)
                    - If(condition, ctx.real_val(1), ctx.real_val(0))
                ).eq(ctx.real_val(1))
            ],
        )
        self.assertEqual(simplified.checks, [condition])

    def test_context_fixpoint_accepts_duplicate_equivalent_bindings(self):
        ctx = SpecContext("simplify-duplicate")
        x = ctx.real("x")
        p = ctx.bool("p")

        ctx.assume(x.eq(ctx.real_val(1)))
        ctx.assume((ctx.real_val(2) - ctx.real_val(1)).eq(x))
        ctx.assume(p.eq(ctx.true()))
        ctx.assume(ctx.real_val(3).eq(ctx.real_val(3)).eq(p))

        simplified = ctx.simplify()

        self.assertEqual(
            simplified.assumes,
            [],
        )
        self.assertEqual(simplified.learned_literals(), {})

    def test_context_fixpoint_raises_on_conflicting_bindings(self):
        ctx = SpecContext("simplify-conflict")
        x = ctx.real("x")

        ctx.assume(x.eq(ctx.real_val(0)))
        ctx.assume(x.eq(ctx.real_val(1)))

        with self.assertRaises(ValueError):
            ctx.simplify()

    def test_simplify_returns_new_simplified_context(self):
        ctx = SpecContext("simplify-output")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(y + ctx.real_val(1)))
        ctx.assume(y.eq(ctx.real_val(2)))
        ctx.check(x.eq(ctx.real_val(3)))

        simplified = ctx.simplify()

        self.assertIsNot(simplified, ctx)
        self.assertEqual(ctx.assumes, [Eq(x, Add(y, RealLit(1))), Eq(y, RealLit(2))])
        self.assertEqual(ctx.checks, [Eq(x, RealLit(3))])
        self.assertEqual(
            simplified.assumes,
            [],
        )
        self.assertEqual(simplified.checks, [])

    def test_simplify_leaves_original_context_unchanged_on_conflict(self):
        ctx = SpecContext("simplify-conflict-output")
        x = ctx.real("x")

        ctx.assume(x.eq(ctx.real_val(0)))
        ctx.assume(x.eq(ctx.real_val(1)))

        with self.assertRaises(ValueError):
            ctx.simplify()

        self.assertEqual(ctx.assumes, [Eq(x, RealLit(0)), Eq(x, RealLit(1))])

    def test_simplify_ctx_skips_rival_for_false_assumption(self):
        ctx = SpecContext("infeasible-shortcut")
        ctx.assume(BoolLit(False))

        with (
            patch("zolotone.spec.spec_context.rival_feasibility_check") as feasibility,
            patch("zolotone.spec.spec_context.rival_trim_context") as trim,
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            report = simplify_ctx(ctx)

        feasibility.assert_not_called()
        trim.assert_not_called()
        self.assertEqual(report["feasibility_status"], "not feasible")
        self.assertEqual(report["status"], "sat")

class TestEgglogFloatLiterals(unittest.TestCase):
    def test_xnor_extracts_as_boolean_equality(self):
        p = BoolVar("p")
        q = BoolVar("q")
        xnor = (p & q) | ((~p) & (~q))

        egraph = EGraph()
        load_rules(egraph)
        lowered = xnor.to_egglog()

        egraph.register(lowered)
        egraph.run(1)

        self.assertEqual(from_egglog(egraph.extract(lowered)), p.eq(q))

    def test_real_lit_round_trips_exact_finite_float_values(self):
        values = [
            0.1,
            0.2,
            0.3,
            1.25,
            -2.5,
            1e-6,
            2.0 ** -20,
        ]

        for value in values:
            with self.subTest(value=value):
                round_tripped = from_egglog(RealLit(value).to_egglog())

                if isinstance(round_tripped, RealLit):
                    self.assertEqual(round_tripped.value.hex(), value.hex())
                else:
                    self.assertEqual(
                        round_tripped,
                        RealLit(4722366482869645) * (RealLit(2) ** RealLit(-72)),
                    )

    def test_fractional_literals_constant_fold_in_egglog_without_losing_float_bits(self):
        expr = RealLit(0.1) + RealLit(0.2)

        egraph = EGraph()
        load_rules(egraph)
        lowered = expr.to_egglog()

        egraph.register(lowered)
        egraph.run(1)

        folded = from_egglog(egraph.extract(lowered))
        self.assertEqual(folded, RealLit(0.1 + 0.2))
        self.assertEqual(folded.value.hex(), (0.1 + 0.2).hex())

    def test_fractional_pow_round_trips_through_egglog(self):
        expr = Pow(RealLit(2.5), RealLit(0.5))

        self.assertEqual(from_egglog(expr.to_egglog()), expr)

    def test_non_finite_real_lits_are_rejected_by_egglog_lowering(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RealLit(value).to_egglog()


class TestSpecAstConstantFolding(unittest.TestCase):
    def assert_check_simplifies_to(self, expr, expected):
        ctx = SpecContext("simplify-check-expression")
        ctx.check(expr)

        expected = expected.constant_fold()
        expected_checks = [] if expected == BoolLit(True) else [expected]
        self.assertEqual(ctx.simplify().checks, expected_checks)

    def test_constant_fold_method_folds_literal_tree(self):
        expr = (RealLit(2) + RealLit(3)) * RealLit(4)

        self.assertEqual(expr.constant_fold(), RealLit(20))

    def test_constant_fold_collapses_literal_subtrees(self):
        expr = ((RealLit(2) + RealLit(3)) * RealLit(4)).eq(RealLit(20))

        self.assertEqual(expr.constant_fold(), BoolLit(True))

    def test_constant_fold_can_produce_false(self):
        expr = (RealLit(2) + RealLit(3)).eq(RealLit(6))

        self.assertEqual(expr.constant_fold(), BoolLit(False))

    def test_constant_fold_keeps_symbolic_if_shape(self):
        expr = If(BoolLit(True), RealVar("x"), RealVar("y"))

        self.assertEqual(expr.constant_fold(), RealVar("x"))

    def test_if_keeps_real_branches_as_if(self):
        expr = If(BoolVar("condition"), RealVar("x"), RealVar("y"))

        self.assertIsInstance(expr, If)

    def test_if_lowers_bool_branches_to_boolean_logic(self):
        condition = BoolVar("condition")
        on_true = BoolVar("on_true")
        on_false = BoolVar("on_false")

        expr = If(condition, on_true, on_false)

        self.assertIsInstance(expr, BoolExpr)
        self.assertEqual(
            expr,
            (condition & on_true) | ((~condition) & on_false),
        )

    def test_if_constant_folds_bool_branches(self):
        self.assertEqual(
            If(BoolLit(True), BoolLit(False), BoolLit(True)).constant_fold(),
            BoolLit(False),
        )
        self.assertEqual(
            If(BoolLit(False), BoolLit(False), BoolLit(True)).constant_fold(),
            BoolLit(True),
        )

    def test_if_rejects_mixed_bool_and_real_branches(self):
        with self.assertRaisesRegex(TypeError, "If branches"):
            If(BoolVar("condition"), BoolVar("on_true"), RealLit(0))

    def test_if_selects_fp_fields_generically(self):
        condition = BoolVar("condition")
        selected = If(condition, fp32.nan(), fp32.ninf())

        self.assertIsInstance(selected, fp32)
        self.assertIsInstance(selected.value, If)
        self.assertEqual(
            If(BoolLit(True), fp32.nan(), fp32.ninf()).constant_fold(),
            fp32.nan(),
        )

    def test_if_rejects_mixed_real_and_fp_branches(self):
        with self.assertRaisesRegex(TypeError, "If branches"):
            If(BoolVar("condition"), fp32.nan(), RealLit(0))

    def test_cases_lower_to_ordered_nested_ifs(self):
        first = BoolVar("first")
        second = BoolVar("second")

        expr = Cases(
            case(first, RealLit(1)),
            case(second, RealLit(2)),
            default(RealLit(3)),
        )

        self.assertEqual(
            expr,
            If(first, RealLit(1), If(second, RealLit(2), RealLit(3))),
        )

    def test_cases_select_the_first_matching_case(self):
        expr = Cases(
            case(BoolLit(True), RealLit(1)),
            case(BoolLit(True), RealLit(2)),
            default(RealLit(3)),
        )

        self.assertEqual(expr.constant_fold(), RealLit(1))

    def test_cases_support_fp_values(self):
        expr = Cases(
            case(BoolLit(False), fp32.nan()),
            case(BoolLit(True), fp32.ninf()),
            default(fp32.inf()),
        )

        self.assertIsInstance(expr, fp32)
        self.assertEqual(expr.constant_fold(), fp32.ninf())

    def test_cases_support_bool_values(self):
        condition = BoolVar("condition")
        on_true = BoolVar("on_true")
        on_false = BoolVar("on_false")

        expr = Cases(
            case(condition, on_true),
            default(on_false),
        )

        self.assertIsInstance(expr, BoolExpr)
        self.assertEqual(
            expr,
            (condition & on_true) | ((~condition) & on_false),
        )

    def test_cases_allow_only_a_default(self):
        fallback = RealVar("fallback")

        self.assertIs(Cases(default(fallback)), fallback)

    def test_cases_require_exactly_one_final_default(self):
        condition = BoolVar("condition")
        value = RealLit(1)

        with self.assertRaisesRegex(ValueError, "requires a default"):
            Cases()
        with self.assertRaisesRegex(ValueError, "requires a default"):
            Cases(case(condition, value))
        with self.assertRaisesRegex(ValueError, "exactly one default"):
            Cases(default(value), default(value))
        with self.assertRaisesRegex(ValueError, "final entry"):
            Cases(default(value), case(condition, value))

    def test_cases_validate_entries_conditions_and_values(self):
        with self.assertRaisesRegex(TypeError, "created by case.*or default"):
            Cases([default(RealLit(1))])
        with self.assertRaisesRegex(TypeError, "created by case.*or default"):
            Cases(object())
        with self.assertRaisesRegex(TypeError, "Expected BoolExpr"):
            case(RealLit(1), RealLit(2))
        with self.assertRaisesRegex(TypeError, "Cases values"):
            default(object())

    def test_cases_reject_mismatched_branch_types(self):
        with self.assertRaisesRegex(TypeError, "If branches"):
            Cases(
                case(BoolVar("condition"), fp32.nan()),
                default(RealLit(0)),
            )

    def test_fp32_encoder_spec_canonicalizes_exact_zero(self):
        from examples.encode_Float32 import fp32_encode_spec

        for sign in (0, 1):
            with self.subTest(sign=sign):
                ctx = SpecContext(f"fp32-encode-zero-sign-{sign}")
                encoded = fp32_encode_spec(
                    RealLit(sign),
                    RealLit(0),
                    RealLit(0),
                    ctx,
                )
                ctx.check(encoded.is_pzero)

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_fp32_encode_design_canonicalizes_exact_zero(self):
        from examples.encode_Float32 import fp32_encode

        for sign in (0, 1):
            with self.subTest(sign=sign):
                design = fp32_encode(
                    Const(UQ(sign, 1, 0)),
                    Const(Q.from_int(127)),
                    Const(UQ.from_float(0, 1, Float32.mantissa_bits)),
                )
                self.assertEqual(design.evaluate(), Float32.Zero())

    def test_fp32_encode_spec_rounds_negative_underflow_to_negative_zero(self):
        # 2^-150 is exactly halfway between zero and the smallest binary32
        # subnormal. RNE selects the even encoding, zero, and keeps the sign.
        for name, value in (
            ("below-midpoint", -(2.0 ** -151)),
            ("midpoint", -(2.0 ** -150)),
        ):
            with self.subTest(name=name):
                ctx = SpecContext(f"fp32-encode-{name}")
                encoded = fp32.encode(RealLit(value), ctx)
                ctx.check(encoded.is_nzero)

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_fp32_encode_design_preserves_negative_underflow_sign(self):
        from examples.encode_Float32 import fp32_encode

        sign = Const(UQ(1, 1, 0))
        exponent = Const(Q.from_int(-22))

        cases = (
            (0.25, Float32.nZero()),
            (0.5, Float32.nZero()),
            (0.75, Float32.from_fields(1, 0, 1)),
        )
        for mantissa, expected in cases:
            with self.subTest(mantissa=mantissa):
                design = fp32_encode(
                    sign,
                    exponent,
                    Const(UQ.from_float(mantissa, 1, Float32.mantissa_bits)),
                )
                self.assertEqual(design.evaluate(), expected)

    def test_fp32_multiplier_spec_zero_handling(self):
        from examples.fp32_mult import spec_fp32_mult

        cases = (
            ("+0 * +0", 0x00000000, 0x00000000, "is_pzero"),
            ("+0 * -0", 0x00000000, 0x80000000, "is_nzero"),
            ("-0 * +0", 0x80000000, 0x00000000, "is_nzero"),
            ("-0 * -0", 0x80000000, 0x80000000, "is_pzero"),
            ("+0 * -1", 0x00000000, 0xbf800000, "is_nzero"),
            ("-0 * +1", 0x80000000, 0x3f800000, "is_nzero"),
            ("-1 * +0", 0xbf800000, 0x00000000, "is_nzero"),
            ("+1 * -0", 0x3f800000, 0x80000000, "is_nzero"),
            ("+0 * +inf", 0x00000000, 0x7f800000, "is_nan"),
            ("-0 * -inf", 0x80000000, 0xff800000, "is_nan"),
            ("+inf * -0", 0x7f800000, 0x80000000, "is_nan"),
            ("-inf * +0", 0xff800000, 0x00000000, "is_nan"),
        )

        for name, lhs_bits, rhs_bits, predicate in cases:
            with self.subTest(name=name):
                ctx = SpecContext(f"fp32-mult-{name}")
                result = spec_fp32_mult(
                    Float32(lhs_bits).to_spec(ctx),
                    Float32(rhs_bits).to_spec(ctx),
                    ctx,
                )
                ctx.check(getattr(result, predicate))

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_fp32_multiplier_spec_preserves_underflow_zero_sign(self):
        from examples.fp32_mult import spec_fp32_mult

        cases = (
            ("positive half-minimum tie", 0x00000001, 0x3f000000, "is_pzero"),
            ("negative half-minimum tie", 0x80000001, 0x3f000000, "is_nzero"),
            ("negative min-normal underflow", 0x80800000, 0x33800000, "is_nzero"),
            ("two negatives underflow", 0x80800000, 0xb3800000, "is_pzero"),
        )

        for name, lhs_bits, rhs_bits, predicate in cases:
            with self.subTest(name=name):
                ctx = SpecContext(f"fp32-mult-{name}")
                result = spec_fp32_mult(
                    Float32(lhs_bits).to_spec(ctx),
                    Float32(rhs_bits).to_spec(ctx),
                    ctx,
                )
                ctx.check(getattr(result, predicate))

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_fp32_adder_spec_preserves_single_infinity(self):
        from examples.fp32_add import spec_fp32_add

        cases = (
            (fp32.inf(), fp32.zero(), fp32.inf()),
            (fp32.zero(), fp32.inf(), fp32.inf()),
            (fp32.ninf(), fp32.zero(), fp32.ninf()),
            (fp32.zero(), fp32.ninf(), fp32.ninf()),
        )

        for lhs, rhs, expected in cases:
            with self.subTest(lhs=lhs, rhs=rhs):
                ctx = SpecContext("fp32-adder-single-infinity")
                result = spec_fp32_add(lhs, rhs, ctx)
                self.assertEqual(result.constant_fold(), expected)

    def test_constant_fold_partially_rebuilds_symbolic_real_expr(self):
        x = RealVar("x")
        expr = x + (RealLit(2) + RealLit(3))

        self.assertEqual(expr.constant_fold(), Add(x, RealLit(5)))

    def test_constant_fold_partially_rebuilds_symbolic_bool_expr(self):
        p = BoolVar("p")
        expr = p | RealLit(2).eq(RealLit(2))

        self.assertEqual(expr.constant_fold(), BoolLit(True))

    def test_constant_fold_folds_boolean_operator_trees(self):
        expr = RealLit(2).eq(RealLit(2)) & ~BoolLit(False)

        self.assertEqual(expr.constant_fold(), BoolLit(True))

    def test_constant_fold_factors_complementary_boolean_partitions(self):
        p = BoolVar("p")
        q = BoolVar("q")
        r = BoolVar("r")

        cases = (
            ((p & q) | (p & (~q)), p),
            ((q & p) | ((~q) & p), p),
            (((p & q) & r) | ((p & (~q)) & r), r & p),
        )

        for expr, expected in cases:
            with self.subTest(expr=str(expr)):
                self.assertEqual(expr.constant_fold(), expected.constant_fold())

    def test_context_simplify_leaves_finite_if_equality_unchanged(self):
        select = BoolVar("select")
        indicator_complement = RealLit(1) - If(
            select,
            RealLit(1),
            RealLit(0),
        )
        equality = indicator_complement.eq(RealLit(1))

        self.assert_check_simplifies_to(equality, equality)

    def test_constant_fold_leaves_finite_symbolic_equality_unchanged(self):
        equality = RealVar("x").eq(RealVar("y"))

        self.assertIs(equality.constant_fold(), equality)

    def test_constant_fold_leaves_unsupported_literal_pow_unchanged(self):
        expr = Pow(RealLit(-2), RealLit(0.5))

        self.assertEqual(expr.constant_fold(), expr)

    def test_non_finite_real_literals_are_rejected_at_construction(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-finite RealLit"):
                    RealLit(value)

    def test_float32_value_is_a_finite_only_real_variable(self):
        ctx = SpecContext("float32-finite-value")
        value = fp32.fresh("x", ctx)

        self.assertIsInstance(value.value, RealVar)
        self.assertTrue(
            any(
                isinstance(assume, Or)
                and value.is_zero in variables(assume)
                and value.value in variables(assume)
                for assume in ctx.assumes
            )
        )

    def test_fp32_uses_explicit_non_finite_constructors(self):
        cases = (
            ("nan", fp32.nan(), "is_nan"),
            ("inf", fp32.inf(), "is_pinf"),
            ("ninf", fp32.ninf(), "is_ninf"),
        )

        for name, encoded, expected_predicate in cases:
            with self.subTest(value=name):
                self.assertEqual(
                    getattr(encoded, expected_predicate).constant_fold(),
                    BoolLit(True),
                )

    def test_encode_fp32_infers_infinity_sign_from_value(self):
        cases = (
            ("positive", RealLit(1e300), "is_pinf"),
            ("negative", RealLit(-1e300), "is_ninf"),
        )

        for name, value, expected_predicate in cases:
            with self.subTest(value=name):
                ctx = SpecContext(f"encode-{name}-infinity")
                encoded = fp32.encode(value=value, ctx=ctx)
                ctx.check(getattr(encoded, expected_predicate))

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_float32_classification_observes_only_relevant_signs(self):
        def make_value(*, sign, norm=False, sub=False, zero=False, inf=False, nan=False):
            return fp32(
                value=RealVar("value"),
                sign=RealLit(sign),
                exponent=RealLit(0),
                mantissa=RealLit(0),
                is_norm=BoolLit(norm),
                is_sub=BoolLit(sub),
                is_zero=BoolLit(zero),
                is_inf=BoolLit(inf),
                is_nan=BoolLit(nan),
            )

        pinf = make_value(sign=0, inf=True).observables_for_classification("inf")
        ninf = make_value(sign=1, inf=True).observables_for_classification("inf")
        pzero = make_value(sign=0, zero=True).observables_for_classification("zero")
        nzero = make_value(sign=1, zero=True).observables_for_classification("zero")
        pnan = make_value(sign=0, nan=True).observables_for_classification("nan")
        nnan = make_value(sign=1, nan=True).observables_for_classification("nan")

        self.assertNotEqual(pinf[0].constant_fold(), ninf[0].constant_fold())
        self.assertNotEqual(pzero[0].constant_fold(), nzero[0].constant_fold())
        self.assertEqual(pnan[0].constant_fold(), nnan[0].constant_fold())

    def test_fp32_known_classification_exposes_only_relevant_fields(self):
        value = fp32(
            value=RealVar("value"),
            sign=RealVar("sign"),
            exponent=RealVar("exponent"),
            mantissa=RealVar("mantissa"),
            is_norm=BoolVar("is_norm"),
            is_sub=BoolVar("is_sub"),
            is_zero=BoolVar("is_zero"),
            is_inf=BoolVar("is_inf"),
            is_nan=BoolVar("is_nan"),
        )

        self.assertEqual(value.observables_for_classification("norm"), (value.value,))
        self.assertEqual(value.observables_for_classification("sub"), (value.value,))
        self.assertEqual(value.observables_for_classification("zero"), (value.sign,))
        self.assertEqual(value.observables_for_classification("inf"), (value.sign,))
        self.assertEqual(
            value.observables_for_classification("nan"),
            (BoolLit(True),),
        )
        with self.assertRaisesRegex(ValueError, "Unknown fp32 classification"):
            value.observables_for_classification("finite")

        ctx = SpecContext("different-output-classifications")
        ast_nodes._add_classification_case_checks(
            ctx,
            value,
            value,
            {"inner_spec": "inf", "outer_spec": "nan"},
        )
        self.assertEqual(
            ctx.checks,
            [flag.eq(flag) for flag in value.classification_flags().values()],
        )

    def test_nested_fp32_outputs_are_split_and_lowered_to_scalar_queries(self):
        inner = fp32.zero()
        outer = fp32.zero()
        ctx = SpecContext("nested-fp32")

        cases = ast_nodes._split_classification_cases(
            ctx,
            [],
            ((inner,), RealLit(7)),
            ((outer,), RealLit(7)),
        )
        case = next(
            case
            for case in cases
            if ast_nodes._case_labels(case.name)
            == {
                "inner_spec.0.0": "zero",
                "outer_spec.0.0": "zero",
            }
        )

        inner_query = tuple(inner.classification_flags().values()) + (inner.sign,)
        outer_query = tuple(outer.classification_flags().values()) + (outer.sign,)
        self.assertEqual(
            case.checks,
            [
                *(lhs.eq(rhs) for lhs, rhs in zip(inner_query, outer_query)),
                RealLit(7).eq(RealLit(7)),
            ],
        )
        self.assertTrue(
            ast_nodes._output_classifications_match(ast_nodes._case_labels(case.name))
        )

        status, _ = solver_engine.check_equivalence(
            case,
            schedule=[{"tool": "simplify"}],
        )
        self.assertEqual(status, "unsat")


class TestBFloat16Spec(unittest.TestCase):
    def test_bf16_format_and_explicit_non_finite_constructors(self):
        self.assertEqual(bf16.exponent_bits, 8)
        self.assertEqual(bf16.mantissa_bits, 7)
        self.assertEqual(bf16.exponent_bias, 127)

        cases = (
            (bf16.nan(), "is_nan"),
            (bf16.inf(), "is_pinf"),
            (bf16.ninf(), "is_ninf"),
            (bf16.zero(), "is_pzero"),
            (bf16.nzero(), "is_nzero"),
        )
        for value, predicate in cases:
            with self.subTest(predicate=predicate):
                self.assertEqual(
                    getattr(value, predicate).constant_fold(),
                    BoolLit(True),
                )

    def test_bf16_encode_classifies_representative_values(self):
        greatest_normal = (2 - 2 ** -bf16.mantissa_bits) * 2 ** 127
        cases = (
            ("positive-zero", 0.0, "is_pzero"),
            ("negative-underflow", -(2 ** -134), "is_nzero"),
            ("smallest-subnormal", 2 ** -133, "is_sub"),
            ("smallest-normal", 2 ** -126, "is_norm"),
            ("greatest-normal", greatest_normal, "is_norm"),
            ("positive-overflow", 4e38, "is_pinf"),
            ("negative-overflow", -4e38, "is_ninf"),
        )

        for name, real_value, predicate in cases:
            with self.subTest(value=name):
                ctx = SpecContext(f"bf16-encode-{name}")
                encoded = bf16.encode(RealLit(real_value), ctx)
                ctx.check(getattr(encoded, predicate))

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_bf16_known_classification_exposes_only_observable_fields(self):
        value = bf16(
            value=RealVar("value"),
            sign=RealVar("sign"),
            exponent=RealVar("exponent"),
            mantissa=RealVar("mantissa"),
            is_norm=BoolVar("is_norm"),
            is_sub=BoolVar("is_sub"),
            is_zero=BoolVar("is_zero"),
            is_inf=BoolVar("is_inf"),
            is_nan=BoolVar("is_nan"),
        )

        self.assertEqual(value.observables_for_classification("norm"), (value.value,))
        self.assertEqual(value.observables_for_classification("sub"), (value.value,))
        self.assertEqual(value.observables_for_classification("zero"), (value.sign,))
        self.assertEqual(value.observables_for_classification("inf"), (value.sign,))
        self.assertEqual(
            value.observables_for_classification("nan"),
            (BoolLit(True),),
        )
        with self.assertRaisesRegex(ValueError, "Unknown bf16 classification"):
            value.observables_for_classification("finite")

    def test_bf16_encode_requires_real_expression(self):
        with self.assertRaisesRegex(TypeError, "bf16.encode value must be RealExpr"):
            bf16.encode(1.0, SpecContext("bf16-invalid-encode"))


class TestRivalTranslation(unittest.TestCase):
    def test_real_expression_translates_to_rival_ir(self):
        x = RealVar("x")
        y = RealVar("y")
        expr = If(x < y, abs(x - RealLit(1)), RealLit(2) ** y)

        self.assertEqual(
            to_rival_ir(expr),
            {
                "op": "if",
                "cond": {
                    "op": "lt",
                    "lhs": {"op": "var", "name": "x"},
                    "rhs": {"op": "var", "name": "y"},
                },
                "on_true": {
                    "op": "abs",
                    "arg": {
                        "op": "sub",
                        "lhs": {"op": "var", "name": "x"},
                        "rhs": {"op": "real_lit", "num": "1", "den": "1"},
                    },
                },
                "on_false": {
                    "op": "pow",
                    "lhs": {"op": "real_lit", "num": "2", "den": "1"},
                    "rhs": {"op": "var", "name": "y"},
                },
            },
        )

    def test_bool_expression_translates_to_rival_ir(self):
        p = BoolVar("p")
        q = BoolVar("q")

        self.assertEqual(
            to_rival_ir((p & ~q).eq(BoolLit(True))),
            {
                "op": "bool_eq",
                "lhs": {
                    "op": "and",
                    "lhs": {"op": "var", "name": "p"},
                    "rhs": {"op": "not", "arg": {"op": "var", "name": "q"}},
                },
                "rhs": {"op": "bool_lit", "value": True},
            },
        )

    def test_float_literal_translates_exact_fraction(self):
        self.assertEqual(
            to_rival_ir(RealLit(0.5)),
            {"op": "real_lit", "num": "1", "den": "2"},
        )

    def test_collect_free_vars_is_sorted(self):
        exprs = [RealVar("z") + RealVar("a"), BoolVar("flag")]

        self.assertEqual(collect_free_vars(exprs), ["a", "flag", "z"])

    def test_build_machine_combines_exprs_into_single_asserted_and(self):
        x = RealVar("x")
        y = RealVar("y")
        raw_machine = object()
        native = Mock()
        native.build_machine.return_value = raw_machine

        with patch("zolotone.rival._load_native_module", return_value=native):
            machine = build_machine(
                [
                    x >= RealLit(0),
                    y <= RealLit(1),
                    x.eq(y),
                ],
                ["x", "y"],
            )

        self.assertIs(machine._raw_machine, raw_machine)
        native.build_machine.assert_called_once_with(
            [
                {
                    "op": "assert",
                    "arg": {
                        "op": "and",
                        "lhs": {
                            "op": "and",
                            "lhs": {
                                "op": "ge",
                                "lhs": {"op": "var", "name": "x"},
                                "rhs": {"op": "real_lit", "num": "0", "den": "1"},
                            },
                            "rhs": {
                                "op": "le",
                                "lhs": {"op": "var", "name": "y"},
                                "rhs": {"op": "real_lit", "num": "1", "den": "1"},
                            },
                        },
                        "rhs": {
                            "op": "eq",
                            "lhs": {"op": "var", "name": "x"},
                            "rhs": {"op": "var", "name": "y"},
                        },
                    },
                }
            ],
            ["x", "y"],
        )

    def test_rival_rects_default_to_unbounded(self):
        ctx = SpecContext("rival-rects-default")

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x", "y"]),
            [[(-math.inf, math.inf), (-math.inf, math.inf)]],
        )

    def test_rival_rects_extract_closed_bounds(self):
        ctx = SpecContext("rival-rects-closed")
        x = ctx.real("x")

        ctx.assume((x >= ctx.real_val(1)) & (x <= ctx.real_val(254)))

        self.assertEqual(get_rival_rects(ctx.assumes, ["x"]), [[(1.0, 254.0)]])

    def test_rival_rects_extract_bounds_with_literal_on_lhs(self):
        ctx = SpecContext("rival-rects-reversed")
        x = ctx.real("x")

        ctx.assume((ctx.real_val(1) <= x) & (ctx.real_val(254) >= x))

        self.assertEqual(get_rival_rects(ctx.assumes, ["x"]), [[(1.0, 254.0)]])

    def test_rival_rects_conservatively_close_strict_bounds(self):
        ctx = SpecContext("rival-rects-strict")
        x = ctx.real("x")

        ctx.assume((x > ctx.real_val(1)) & (x < ctx.real_val(254)))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x"]),
            [[(1.0, 254.0)]],
        )

    def test_rival_rects_conservatively_close_reversed_strict_bounds(self):
        ctx = SpecContext("rival-rects-reversed-strict")
        x = ctx.real("x")

        ctx.assume((ctx.real_val(1) < x) & (ctx.real_val(254) > x))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x"]),
            [[(1.0, 254.0)]],
        )

    def test_rival_rects_enclose_nonrepresentable_literals(self):
        ctx = SpecContext("rival-rects-rounded-literal")
        x = ctx.real("x")
        exact = (1 << 53) + 1

        ctx.assume(x.eq(ctx.real_val(exact)))

        rounded_down = float(1 << 53)
        rounded_up = math.nextafter(rounded_down, math.inf)
        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x"]),
            [[(rounded_down, rounded_up)]],
        )

    def test_rival_rects_extract_point_disjunction(self):
        ctx = SpecContext("rival-rects-point-or")
        sign = ctx.real("sign")

        ctx.assume(sign.eq(ctx.real_val(0)) | ctx.real_val(1).eq(sign))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["sign"]),
            [[(0.0, 0.0)], [(1.0, 1.0)]],
        )

    def test_rival_rects_cartesian_product_for_independent_disjunctions(self):
        ctx = SpecContext("rival-rects-cartesian-or")
        sign = ctx.real("sign")
        exponent = ctx.real("exponent")

        ctx.assume(sign.eq(ctx.real_val(0)) | sign.eq(ctx.real_val(1)))
        ctx.assume(exponent.eq(ctx.real_val(0)) | exponent.eq(ctx.real_val(255)))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["sign", "exponent"]),
            [
                [(0.0, 0.0), (0.0, 0.0)],
                [(0.0, 0.0), (255.0, 255.0)],
                [(1.0, 1.0), (0.0, 0.0)],
                [(1.0, 1.0), (255.0, 255.0)],
            ],
        )

    def test_rival_rects_preserve_free_var_order(self):
        ctx = SpecContext("rival-rects-order")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x >= ctx.real_val(1))
        ctx.assume(y <= ctx.real_val(2))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["y", "x", "z"]),
            [[(-math.inf, 2.0), (1.0, math.inf), (-math.inf, math.inf)]],
        )

    def test_rival_rects_drop_conflicting_bounds(self):
        ctx = SpecContext("rival-rects-conflict")
        x = ctx.real("x")

        ctx.assume(x >= ctx.real_val(2))
        ctx.assume(x <= ctx.real_val(1))

        self.assertEqual(get_rival_rects(ctx.assumes, ["x"]), [])

    def test_rival_rects_ignore_unsupported_assumptions(self):
        ctx = SpecContext("rival-rects-unsupported")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume((x + y).eq(ctx.real_val(1)))
        ctx.assume(x >= ctx.real_val(0))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x", "y"]),
            [[(0.0, math.inf), (-math.inf, math.inf)]],
        )

    def test_rival_rects_ignore_or_when_any_branch_is_unsupported(self):
        ctx = SpecContext("rival-rects-unsupported-or")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume((x >= ctx.real_val(0)) | (x + y).eq(ctx.real_val(1)))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x", "y"]),
            [[(-math.inf, math.inf), (-math.inf, math.inf)]],
        )

    def test_rival_rects_expand_independent_disjunctions_without_coalescing(self):
        ctx = SpecContext("rival-rects-no-coalesce")
        sign = ctx.real("sign")
        exponent = ctx.real("exponent")

        ctx.assume(sign.eq(ctx.real_val(0)) | sign.eq(ctx.real_val(1)))
        ctx.assume(exponent.eq(ctx.real_val(0)) | exponent.eq(ctx.real_val(255)))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["sign", "exponent"]),
            [
                [(0.0, 0.0), (0.0, 0.0)],
                [(0.0, 0.0), (255.0, 255.0)],
                [(1.0, 1.0), (0.0, 0.0)],
                [(1.0, 1.0), (255.0, 255.0)],
            ],
        )

    def test_rival_feasibility_splits_only_variables_from_maybe_exprs(self):
        ctx = SpecContext("rival-guided-split")
        x = ctx.real("x")
        y = ctx.real("y")
        z = ctx.real("z")
        good_x = x >= ctx.real_val(0)
        good_z = z >= ctx.real_val(0)
        maybe = x.eq(y)
        ctx.assume(good_x)
        ctx.assume(good_z)
        ctx.check(maybe)

        root = [(0.0, math.inf), (-math.inf, math.inf), (0.0, math.inf)]
        x_left = (0.0, sys.float_info.max / 2.0)
        x_right = (math.nextafter(sys.float_info.max / 2.0, math.inf), math.inf)
        y_left = (-math.inf, 0.0)
        y_right = (math.nextafter(0.0, math.inf), math.inf)
        combined_calls = []

        def build(exprs, free_vars):
            self.assertEqual(free_vars, ["x", "y", "z"])
            machine = Mock()
            if exprs == ctx.assumes + ctx.checks:
                def apply(rect, hints=None):
                    combined_calls.append((rect, hints))
                    if rect == root:
                        return RivalAnalysis(
                            status=(False, True),
                            hints="root-hints",
                            converged=False,
                        )
                    return RivalAnalysis(status=(True, True), hints=None, converged=True)

                machine.apply_with_hints.side_effect = apply
            elif exprs == [maybe]:
                machine.apply_with_hints.return_value = RivalAnalysis(
                    status=(False, True),
                    hints=None,
                    converged=False,
                )
            else:
                machine.apply_with_hints.return_value = RivalAnalysis(
                    status=(False, False),
                    hints=None,
                    converged=True,
                )
            return machine

        with patch("zolotone.rival.build_machine", side_effect=build):
            status = rival_feasibility_check(ctx, max_depth=1, checks=True)

        self.assertEqual(status, "not feasible")
        self.assertEqual(
            combined_calls,
            [
                (root, None),
                ([x_left, y_left, (0.0, math.inf)], "root-hints"),
                ([x_left, y_right, (0.0, math.inf)], "root-hints"),
                ([x_right, y_left, (0.0, math.inf)], "root-hints"),
                ([x_right, y_right, (0.0, math.inf)], "root-hints"),
            ],
        )

    def test_rival_feasibility_returns_first_clean_rect(self):
        ctx = SpecContext("rival-feasible-rect")
        x = ctx.real("x")
        ctx.assume(x.eq(ctx.real_val(0)) | x.eq(ctx.real_val(1)))
        ctx.check(x.eq(ctx.real_val(0)))

        clean_rect = [(0.0, 0.0)]
        bad_rect = [(1.0, 1.0)]
        combined_calls = []

        def build(exprs, free_vars):
            self.assertEqual(free_vars, ["x"])
            machine = Mock()
            if exprs == ctx.assumes + ctx.checks:
                def apply(rect, hints=None):
                    combined_calls.append((rect, hints))
                    if rect == clean_rect:
                        return RivalAnalysis(
                            status=(False, False),
                            hints=None,
                            converged=True,
                        )
                    if rect == bad_rect:
                        return RivalAnalysis(
                            status=(True, True),
                            hints=None,
                            converged=True,
                        )
                    raise AssertionError(f"unexpected rect {rect}")

                machine.apply_with_hints.side_effect = apply
            else:
                machine.apply_with_hints.return_value = RivalAnalysis(
                    status=(False, False),
                    hints=None,
                    converged=True,
                )
            return machine

        with (
            patch("zolotone.rival.get_rival_rects", return_value=[clean_rect, bad_rect]),
            patch("zolotone.rival.build_machine", side_effect=build),
        ):
            status = rival_feasibility_check(ctx, max_depth=1, checks=True)

        self.assertEqual(status, "feasible")
        self.assertEqual(combined_calls, [(clean_rect, None)])

    def test_rival_trim_context_uses_assumption_rects_only_for_checks(self):
        ctx = SpecContext("rival-trim-assumption-rects")
        x = ctx.real("x")
        bounded = x >= ctx.real_val(0)
        ctx.assume(bounded)
        ctx.check(bounded)

        unbounded = [(-math.inf, math.inf)]
        assumption_rect = [(0.0, math.inf)]
        seen_rects = []

        def build(exprs, free_vars):
            self.assertEqual(free_vars, ["x"])
            self.assertEqual(exprs, [bounded])
            machine = Mock()

            def apply(rect, hints=None):
                self.assertIsNone(hints)
                seen_rects.append(rect)
                return RivalAnalysis(
                    status=(False, False)
                    if rect == assumption_rect
                    else (False, True),
                    hints=None,
                    converged=True,
                )

            machine.apply_with_hints.side_effect = apply
            return machine

        with patch("zolotone.rival.build_machine", side_effect=build):
            trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.assumes, [bounded])
        self.assertEqual(trimmed.checks, [])
        self.assertEqual(seen_rects, [unbounded, assumption_rect])

    def test_rival_trim_context_keeps_maybe_exprs(self):
        ctx = SpecContext("rival-trim-maybe")
        x = ctx.real("x")
        y = ctx.real("y")
        assume = x >= ctx.real_val(0)
        check = x.eq(y)
        ctx.assume(assume)
        ctx.check(check)

        def build(exprs, free_vars):
            self.assertEqual(free_vars, ["x", "y"])
            machine = Mock()
            machine.apply_with_hints.return_value = RivalAnalysis(
                status=(False, True),
                hints=None,
                converged=True,
            )
            return machine

        with patch("zolotone.rival.build_machine", side_effect=build):
            trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.assumes, [assume])
        self.assertEqual(trimmed.checks, [check])

    def test_rival_trim_context_requires_every_assumption_rect(self):
        ctx = SpecContext("rival-trim-all-rects")
        sign = ctx.real("sign")
        bit_domain = sign.eq(ctx.real_val(0)) | sign.eq(ctx.real_val(1))
        check = sign.eq(ctx.real_val(0))
        ctx.assume(bit_domain)
        ctx.check(check)

        def build(exprs, free_vars):
            self.assertEqual(free_vars, ["sign"])
            machine = Mock()
            if exprs == [bit_domain]:
                machine.apply_with_hints.return_value = RivalAnalysis(
                    status=(False, True),
                    hints=None,
                    converged=True,
                )
            else:
                self.assertEqual(exprs, [check])

                def apply(rect, hints=None):
                    self.assertIsNone(hints)
                    return RivalAnalysis(
                        status=(False, False)
                        if rect == [(0.0, 0.0)]
                        else (True, True),
                        hints=None,
                        converged=True,
                    )

                machine.apply_with_hints.side_effect = apply
            return machine

        with patch("zolotone.rival.build_machine", side_effect=build):
            trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.assumes, [bit_domain])
        self.assertEqual(trimmed.checks, [check])

class TestSpecificationDeterminism(unittest.TestCase):
    def test_deterministic_primitive_uses_same_inputs_for_both_spec_runs(self):
        seen_inputs = []

        def deterministic_spec(x, ctx):
            seen_inputs.append(x)
            out = ctx.fresh_real("out")
            ctx.assume(out.eq(x + ctx.real_val(1)))
            return out

        @Primitive(name="deterministic_primitive", spec=deterministic_spec)
        def deterministic_primitive(x):
            return x.copy()

        node = deterministic_primitive(Var(name="x", sign=UQT(2, 0)))
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = node.check_determinism(
                schedule=[
                    {"tool": "simplify"},
                    {"tool": "z3", "timeout_ms": 1000},
                ]
            )

        self.assertTrue(result["proved"])
        self.assertEqual(len(result["proof_traces"]), 1)
        self.assertEqual(len(seen_inputs), 2)
        self.assertIs(seen_inputs[0], seen_inputs[1])

    def test_underconstrained_primitive_is_not_deterministic(self):
        def nondeterministic_spec(_x, ctx):
            out = ctx.fresh_real("out")
            zero = ctx.real_val(0)
            one = ctx.real_val(1)
            ctx.assume(out.eq(zero) | out.eq(one))
            return out

        @Primitive(name="nondeterministic_primitive", spec=nondeterministic_spec)
        def nondeterministic_primitive(x):
            return x.copy()

        node = nondeterministic_primitive(Var(name="x", sign=UQT(2, 0)))
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = node.check_determinism(
                schedule=[
                    {"tool": "simplify"},
                    {"tool": "z3", "timeout_ms": 1000},
                ]
            )

        self.assertFalse(result["proved"])
        self.assertEqual(result["proof_traces"][-1][-1]["status"], "sat")

    def test_combined_infeasibility_with_matching_side_feasibility_is_proved(self):
        base_ctx = SpecContext("matching_side_feasibility")
        x = base_ctx.real("x")
        collect_counts = {"zero": 0, "one": 0}

        def collect_zero(ctx):
            collect_counts["zero"] += 1
            ctx.assume(x.eq(ctx.real_val(0)))
            return x

        def collect_one(ctx):
            collect_counts["one"] += 1
            ctx.assume(x.eq(ctx.real_val(1)))
            return x

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = ast_nodes.check_equivalence(
                ast_nodes._Spec("zero_spec", collect_zero),
                ast_nodes._Spec("one_spec", collect_one),
                base_ctx=base_ctx,
                inputs=[],
                schedule=[{"tool": "simplify"}],
            )

        self.assertTrue(result["proved"])
        self.assertEqual(collect_counts, {"zero": 2, "one": 2})

    def test_combined_infeasibility_with_matching_infeasible_sides_is_proved(self):
        base_ctx = SpecContext("matching_infeasible_sides")
        x = base_ctx.real("x")

        def collect_infeasible(ctx):
            ctx.assume(ctx.false())
            return x

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = ast_nodes.check_equivalence(
                ast_nodes._Spec("first_spec", collect_infeasible),
                ast_nodes._Spec("second_spec", collect_infeasible),
                base_ctx=base_ctx,
                inputs=[],
                schedule=[{"tool": "simplify"}],
            )

        self.assertTrue(result["proved"])

    def test_side_feasibility_mismatch_is_not_equivalent(self):
        base_ctx = SpecContext("side_feasibility_mismatch")
        x = base_ctx.real("x")

        def collect_feasible(_ctx):
            return x

        def collect_infeasible(ctx):
            ctx.assume(ctx.false())
            return x

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = ast_nodes.check_equivalence(
                ast_nodes._Spec("feasible_spec", collect_feasible),
                ast_nodes._Spec("infeasible_spec", collect_infeasible),
                base_ctx=base_ctx,
                inputs=[],
                schedule=[{"tool": "simplify"}],
            )

        self.assertFalse(result["proved"])

    def test_unknown_side_feasibility_is_not_equivalent(self):
        base_ctx = SpecContext("unknown_side_feasibility")
        x = base_ctx.real("x")

        def collect_zero(ctx):
            ctx.assume(x.eq(ctx.real_val(0)))
            return x

        def collect_one(ctx):
            ctx.assume(x.eq(ctx.real_val(1)))
            return x

        with (
            patch.object(
                ast_nodes,
                "simplify_ctx",
                return_value={"feasibility_status": "unknown"},
            ),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            result = ast_nodes.check_equivalence(
                ast_nodes._Spec("zero_spec", collect_zero),
                ast_nodes._Spec("one_spec", collect_one),
                base_ctx=base_ctx,
                inputs=[],
                schedule=[{"tool": "simplify"}],
            )

        self.assertFalse(result["proved"])

    def test_composite_determinism_ignores_inner_proof_context(self):
        def identity_spec(x, ctx):
            del ctx
            return x

        @Composite(name="deterministic_composite", spec=identity_spec)
        def deterministic_composite(x):
            with context() as ctx:
                ctx.check(ctx.false())
            return x.copy()

        node = deterministic_composite(Var(name="x", sign=UQT(2, 0)))
        self.assertEqual(len(node.ctx.checks), 1)

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = node.check_determinism(schedule=[{"tool": "simplify"}])

        self.assertTrue(result["proved"])

    def test_nested_tuple_outputs_are_compared_recursively(self):
        def identity_spec(x, ctx):
            del ctx
            return x

        @Primitive(name="tuple_identity", spec=identity_spec)
        def tuple_identity(x):
            return x.copy()

        node = tuple_identity(
            Var(
                name="x",
                sign=TupleT(
                    UQT(2, 0),
                    TupleT(BoolT(), UQT(1, 1)),
                ),
            )
        )
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = node.check_determinism(schedule=[{"tool": "simplify"}])

        self.assertTrue(result["proved"])

    def test_output_shape_mismatch_raises_type_error(self):
        call_count = 0

        def unstable_shape_spec(x, ctx):
            del ctx
            nonlocal call_count
            call_count += 1
            return x if call_count == 1 else (x, x)

        @Primitive(name="unstable_shape", spec=unstable_shape_spec)
        def unstable_shape(x):
            return x.copy()

        node = unstable_shape(Var(name="x", sign=UQT(2, 0)))
        with self.assertRaisesRegex(TypeError, "Spec shape mismatch"):
            node.check_determinism(schedule=[{"tool": "simplify"}])

    def test_unobservable_nan_fields_do_not_make_spec_nondeterministic(self):
        def nan_spec(ctx):
            out = fp32.fresh("out", ctx)
            ctx.assume(out.is_nan.eq(ctx.true()))
            return out

        @Primitive(name="nan_primitive", spec=nan_spec)
        def nan_primitive():
            return Const(Float32.NaN())

        node = nan_primitive()
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = node.check_determinism(schedule=[{"tool": "simplify"}])

        self.assertTrue(result["proved"])
        self.assertEqual(len(result["proof_traces"]), 25)


class TestSolverApis(unittest.TestCase):
    def test_check_spec_preserves_mismatched_classification_case_verdict(self):
        def infinity_spec(ctx):
            del ctx
            return fp32.inf()

        @Composite(name="zero_vs_infinity", spec=infinity_spec)
        def zero_vs_infinity():
            return Const(Float32.Zero())

        split_classification_cases = ast_nodes._split_classification_cases

        def select_feasible_mismatch(*args, **kwargs):
            return [
                case
                for case in split_classification_cases(*args, **kwargs)
                if ast_nodes._case_labels(case.name)["inner_spec"] == "zero"
                and ast_nodes._case_labels(case.name)["outer_spec"] == "inf"
            ]

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_feasible_mismatch,
            ),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            result = zero_vs_infinity().check_spec(
                schedule=[{"tool": "simplify"}],
            )

        self.assertTrue(result["proved"])
        self.assertEqual(result["proof_traces"][-1][-1]["status"], "sat")

    def test_fp32_multiplier_check_spec_proves_zero_input_cases(self):
        multiplier = fp32_mult(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        zero_input_pairs = {
            ("zero", "zero"),
            ("zero", "norm"),
            ("norm", "zero"),
            ("zero", "sub"),
            ("sub", "zero"),
            ("zero", "inf"),
            ("inf", "zero"),
            ("zero", "nan"),
            ("nan", "zero"),
        }
        split_classification_cases = ast_nodes._split_classification_cases

        def select_zero_input_cases(
            ctx,
            inputs,
            spec_inner,
            spec_outer,
            output_names=("inner_spec", "outer_spec"),
        ):
            cases = split_classification_cases(
                ctx,
                inputs,
                spec_inner,
                spec_outer,
                output_names=output_names,
            )
            selected = [
                case
                for case in cases
                if (
                    ast_nodes._case_labels(case.name)["arg0"],
                    ast_nodes._case_labels(case.name)["arg1"],
                ) in zero_input_pairs
            ]
            self.assertEqual(len(selected), len(zero_input_pairs) * 25)
            return selected

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_zero_input_cases,
            ),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            check_result = multiplier.check_spec(schedule=[{"tool": "simplify"}])

        self.assertTrue(check_result["proved"])
        self.assertEqual(
            len(check_result["proof_traces"]),
            len(zero_input_pairs) * 25,
        )

    def test_fp32_adder_norm_inf_inf_inf_is_trimmed_before_egglog(self):
        adder = fp32_add(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        ctx = adder.ctx.copy()
        spec_inner = ctx.spec_of(adder.inner_tree)
        inputs = [ctx.spec_of(arg) for arg in adder.inner_args]
        spec_outer = adder.spec(*inputs, ctx=ctx)
        target_name = (
            "fp32_add["
            "arg0=norm,arg1=inf,inner_spec=inf,outer_spec=inf]"
        )
        case = next(
            case
            for case in ast_nodes._split_classification_cases(
                ctx,
                inputs,
                spec_inner,
                spec_outer,
            )
            if case.name == target_name
        )

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            status, trace = solver_engine.check_equivalence(
                case,
                schedule=[{"tool": "simplify"}],
            )

        self.assertEqual(status, "unsat")
        self.assertEqual(trace[-1]["new_ctx"].checks, [])


    def test_fp32_adder_inf_inf_norm_side_case_uses_only_real_and_bool_exprs(self):
        adder = fp32_add(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        labels = {
            "arg0": "inf",
            "arg1": "inf",
            "inner_spec": "norm",
            "outer_spec": "norm",
        }

        base_ctx = adder.ctx.copy()
        inputs = [base_ctx.spec_of(arg) for arg in adder.inner_args]
        simplified = ast_nodes._collect_classified_spec(
            ast_nodes._Spec(
                "inner_spec",
                lambda ctx: ctx.spec_of(adder.inner_tree),
            ),
            base_ctx=base_ctx,
            inputs=inputs,
            case_labels=labels,
        ).simplify()

        def uses_only_supported_exprs(node):
            return isinstance(node, (RealExpr, BoolExpr)) and all(
                uses_only_supported_exprs(child) for child in children(node)
            )

        self.assertTrue(all(uses_only_supported_exprs(expr) for expr in simplified.assumes))

    def test_fp32_adder_inf_inf_cannot_have_normal_outer_spec(self):
        adder = fp32_add(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        base_ctx = adder.ctx.copy()
        inputs = [base_ctx.spec_of(arg) for arg in adder.inner_args]
        simplified = ast_nodes._collect_classified_spec(
            ast_nodes._Spec(
                "outer_spec",
                lambda ctx: adder.spec(*inputs, ctx=ctx),
            ),
            base_ctx=base_ctx,
            inputs=inputs,
            case_labels={
                "arg0": "inf",
                "arg1": "inf",
                "outer_spec": "norm",
            },
        ).simplify()

        env = {}
        solver = z3.Solver()
        solver.add(*(assume.to_z3(env) for assume in simplified.assumes))
        self.assertEqual(solver.check(), z3.unsat)


    def test_check_equivalence_with_simplify_schedule_short_circuits(self):
        ctx = SpecContext("simplify-schedule")
        ctx.check((RealLit(1) + RealLit(2)).eq(RealLit(3)))

        status, proof_trace = solver_engine.check_equivalence(
            ctx,
            schedule=[{"tool": "simplify"}],
        )

        self.assertEqual(status, "unsat")
        self.assertEqual(len(proof_trace), 1)
        self.assertEqual(proof_trace[0]["tool"], "simplify")

    def test_check_equivalence_reports_poor_spec_from_simplify_schedule(self):
        ctx = SpecContext("simplify-conflict-schedule")
        x = ctx.real("x")

        ctx.assume(x.eq(ctx.real_val(0)))
        ctx.assume(x.eq(ctx.real_val(1)))
        ctx.check(x.eq(ctx.real_val(0)))

        status, proof_trace = solver_engine.check_equivalence(
            ctx,
            schedule=[{"tool": "simplify"}],
        )

        self.assertEqual(status, "sat")
        self.assertEqual(len(proof_trace), 1)
        self.assertEqual(proof_trace[0]["tool"], "simplify")
        self.assertEqual(proof_trace[0]["status"], "sat")
        self.assertIn("Conflicting learned literals", str(proof_trace[0]["info"]))

    def test_check_equivalence_returns_flat_proof_trace(self):
        ctx = SpecContext("flat-trace")
        ctx.check(RealLit(1).eq(RealLit(1)))

        with patch.dict(solver_engine.TOOL_FNS, {"z3": _flat_trace_tool}):
            status, proof_trace = solver_engine.check_equivalence(
                ctx,
                schedule=[{"tool": "z3", "timeout_ms": 1}],
            )

        self.assertEqual(status, "unsat")
        self.assertIsInstance(proof_trace, list)
        self.assertEqual(len(proof_trace), 1)
        self.assertIsInstance(proof_trace[0], dict)
        self.assertEqual(proof_trace[0]["tool"], "branch-b")

    def test_run_tool_has_hard_wall_clock_timeout(self):
        ctx = SpecContext("tool-timeout")

        with patch.dict(solver_engine.TOOL_FNS, {"z3": _slow_tool}):
            reports = solver_engine._run_tool(
                ctx,
                {"tool": "z3", "timeout_ms": 1},
                timeout=0.01,
            )

        self.assertEqual(reports[0]["status"], "unknown")
        self.assertEqual(reports[0]["wall_clock_timeout_s"], 0.01)

    def test_spec_context_is_pickleable(self):
        ctx = SpecContext("pickle-context")
        x = ctx.real("x")
        ctx.check((x + ctx.real_val(1)).eq(ctx.real_val(2)))

        restored = pickle.loads(pickle.dumps(ctx))

        self.assertEqual(str(restored), str(ctx))
        for transported_ctx in (restored, restored.copy()):
            with self.assertRaisesRegex(RuntimeError, "spec_cache was discarded"):
                transported_ctx.spec_of(object())

    def test_run_tool_rejects_large_report(self):
        ctx = SpecContext("large-report")

        with (
            patch.dict(solver_engine.TOOL_FNS, {"z3": _large_report_tool}),
            patch.object(solver_engine, "MAX_TOOL_REPORT_BYTES", 1024),
        ):
            with self.assertRaisesRegex(RuntimeError, "maximum is 1024 bytes"):
                solver_engine._run_tool(ctx, {"tool": "z3", "timeout_ms": 1})

    def test_check_equivalence_rejects_rival_feasibility_as_proof_tool(self):
        ctx = SpecContext("rival-schedule")
        ctx.check(RealLit(1).eq(RealLit(1)))

        with self.assertRaises(ValueError) as raised:
            solver_engine.check_equivalence(
                ctx,
                schedule=[
                    {
                        "tool": "rival_feasibility_check",
                        "max_depth": "1",
                    }
                ],
            )

        self.assertIn("Unknown schedule tool rival_feasibility_check", str(raised.exception))

    def test_fp32_adder_norm_norm_proves_with_egglog(self):
        adder = fp32_add(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        target_name = "fp32_add[arg0=norm,arg1=norm,inner_spec=norm,outer_spec=norm]"
        split_classification_cases = ast_nodes._split_classification_cases

        def select_norm_norm_case(*args, **kwargs):
            cases = split_classification_cases(*args, **kwargs)
            return [next(case for case in cases if case.name == target_name)]

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_norm_norm_case,
            ),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            check_result = adder.check_spec(
                schedule=[
                    {"tool": "simplify"},
                    {
                        "tool": "egglog-rewrite",
                        "iterations": 6,
                        "scheduler": {"match_limit": 500_000, "ban_length": 1},
                    },
                ]
            )

        self.assertTrue(check_result["proved"])
        matching_traces = [
            proof_trace
            for proof_trace in check_result["proof_traces"]
            if proof_trace and proof_trace[0]["name"] == target_name
        ]
        self.assertEqual(len(matching_traces), 1)
        self.assertTrue(
            any(
                report["tool"] == "egglog-rewrite" and report["status"] == "unsat"
                for report in matching_traces[0]
            ),
            matching_traces[0],
        )

    def test_conventional_zero_zero_proves_with_xor_sign_fact(self):
        a = [Var(name=f"a_{idx}", sign=BFloat16T()) for idx in range(4)]
        b = [Var(name=f"b_{idx}", sign=BFloat16T()) for idx in range(4)]
        conventional = Conventional(*a, *b)
        target_name = "Conventional[inner_spec=zero,outer_spec=zero]"
        split_classification_cases = ast_nodes._split_classification_cases

        def select_zero_zero_case(*args, **kwargs):
            cases = split_classification_cases(*args, **kwargs)
            return [next(case for case in cases if case.name == target_name)]

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_zero_zero_case,
            ),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            check_result = conventional.check_spec(
                schedule=[
                    {"tool": "simplify"},
                    {
                        "tool": "egglog-rewrite",
                        "iterations": 6,
                        "scheduler": {"match_limit": 500_000, "ban_length": 1},
                    },
                ]
            )

        self.assertTrue(check_result["proved"])
        proof_traces = check_result["proof_traces"]
        self.assertEqual(len(proof_traces), 1)
        self.assertTrue(
            any(
                report["tool"] == "egglog-rewrite" and report["status"] == "unsat"
                for report in proof_traces[0]
            ),
            proof_traces[0],
        )

    def test_z3_check_eq_returns_single_report(self):
        ctx = SpecContext("z3-api")
        ctx.check(RealLit(1).eq(RealLit(1)))

        report = z3_check_eq(ctx, timeout_ms=1000)

        self.assertIsInstance(report, dict)
        self.assertEqual(report["tool"], "z3")
        self.assertEqual(report["status"], "unsat")

    def test_dreal_check_eq_returns_single_report(self):
        ctx = SpecContext("dreal-api")
        ctx.check(RealLit(1).eq(RealLit(1)))

        report = dreal_check_eq(ctx, precision=0.001)

        self.assertIsInstance(report, dict)
        self.assertEqual(report["tool"], "dreal")
        self.assertEqual(report["status"], "unsat")


class TestSignSpecs(unittest.TestCase):
    def test_xor_sign_multiplier_fact_follows_from_existing_xor_constraints(self):
        ctx = SpecContext("xor-sign-multiplier")
        x = ctx.real("x")
        y = ctx.real("y")

        result = xor_spec(x, y, ctx)
        derived_fact = sign_multiplier(ctx, result).eq(
            sign_multiplier(ctx, x) * sign_multiplier(ctx, y)
        )
        self.assertEqual(ctx.assumes[-1], derived_fact)
        ctx.assumes.pop()
        ctx.check(derived_fact)

        report = z3_check_eq(ctx, timeout_ms=10000)

        self.assertEqual(report["status"], "unsat", report)

    def test_q_signs_xor_spec_matches_constant_sign_combinations(self):
        cases = [
            (-1, -1, 0),
            (-1, 1, 1),
            (1, -1, 1),
            (1, 1, 0),
        ]

        for lhs, rhs, expected in cases:
            with self.subTest(lhs=lhs, rhs=rhs):
                node = q_signs_xor(Const(Q.from_int(lhs)), Const(Q.from_int(rhs)))
                self.assertEqual(node.evaluate().val, expected)

                ctx = SpecContext("q_signs_xor")
                spec = ctx.spec_of(node)
                ctx.check(spec.eq(ctx.real_val(expected)))
                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(str(spec), "real(xored_signs_2)")
                self.assertEqual(report["status"], "unsat", report)


if __name__ == "__main__":
    unittest.main()
