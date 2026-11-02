import math
import unittest
from fractions import Fraction

from zolotone.rival import build_float_evaluator, RivalEvaluation
from zolotone.spec.spec_ast import RealLit, RealVar, If
from zolotone.spec.spec_context import SpecContext


class TestRivalFloatMachine(unittest.TestCase):
    def test_fractional_point_and_outward_singleton_bounds(self):
        x = RealVar('x')
        machine = build_float_evaluator(x / RealLit(3))
        self.assertEqual(machine.apply_point([1]), RivalEvaluation(True, 1 / 3))
        result = machine.apply_range([(1, 1)])
        self.assertEqual(result.precondition, (True, True))
        lo, hi = result.output
        self.assertLessEqual(Fraction(lo), Fraction(1, 3))
        self.assertGreaterEqual(Fraction(hi), Fraction(1, 3))
        self.assertLess(lo, hi)
        self.assertEqual(machine.apply_point([2]).output, 2 / 3)

    def test_exact_integer_inputs_above_binary64_precision(self):
        x = RealVar('x')
        origin = 1 << 80
        machine = build_float_evaluator(x - RealLit(origin))
        self.assertEqual(machine.apply_point([origin + 1]).output, 1.0)
        self.assertEqual(machine.apply_range([(origin + 1, origin + 3)]).output, (1.0, 3.0))

    def test_high_fractional_precision(self):
        machine = build_float_evaluator(
            RealVar('x') * RealLit(Fraction(1, 2**40)),
        )
        self.assertEqual(machine.apply_point([1]).output, 2**-40)
        self.assertEqual(machine.apply_range([(1, 2)]).output, (2**-40, 2**-39))

    def test_root_round_and_conditional_have_same_value_semantics(self):
        ctx = SpecContext('float-round')
        x = ctx.real('x')
        rounded = ctx.rne(x * ctx.real_val(Fraction(3, 8)), frac_bits=2)
        root = build_float_evaluator(rounded)
        conditional = build_float_evaluator(If(x > ctx.zero(), rounded, ctx.zero()))
        self.assertEqual(root.apply_point([1]).output, 0.5)
        self.assertEqual(conditional.apply_point([1]).output, 0.5)

    def test_multivariate_and_no_inputs(self):
        x, y = RealVar('x'), RealVar('y')
        machine = build_float_evaluator(x / y, x > RealLit(0), free_vars=['y', 'x'])
        self.assertEqual(machine.apply_point([2, 1]), RivalEvaluation(True, 0.5))
        result = machine.apply_range([(2, 4), (1, 2)])
        self.assertLessEqual(result.output[0], 0.25)
        self.assertGreaterEqual(result.output[1], 1.0)
        constant = build_float_evaluator(RealLit(Fraction(1, 8)), free_vars=[])
        self.assertEqual(constant.apply_point([]).output, 0.125)
        self.assertEqual(constant.apply_range([]).output, (0.125, 0.125))

    def test_error_flags_and_rejection(self):
        ctx = SpecContext('float-errors')
        x = ctx.real('x')
        invalid = ctx.zero() ** -ctx.one()
        machine = build_float_evaluator(invalid, ctx.false())
        result = machine.apply_point([0])
        self.assertFalse(result.precondition)
        self.assertFalse(result.precondition_error)
        self.assertTrue(result.output_error)
        self.assertTrue(result.output_total_error)
        result = machine.apply_range([(0, 1)])
        self.assertTrue(result.output_total_error)
        invalid_precondition = build_float_evaluator(x, invalid > ctx.zero())
        result = invalid_precondition.apply_point([1])
        self.assertTrue(result.precondition_error)
        self.assertTrue(result.precondition_total_error)
        self.assertFalse(result.output_error)
        rejected = build_float_evaluator(x, (x > ctx.zero()) & (x ** -ctx.one() > ctx.zero()))
        result = rejected.apply_point([0])
        self.assertFalse(result.precondition)
        self.assertFalse(result.precondition_error)

    def test_underflow_bounds_remain_outward(self):
        machine = build_float_evaluator(RealLit(Fraction(1, 2**1100)), free_vars=[])
        self.assertEqual(machine.apply_point([]).output, 0.0)
        self.assertEqual(machine.apply_range([]).output, (0.0, math.nextafter(0.0, math.inf)))

    def test_overflow_bounds_remain_outward(self):
        machine = build_float_evaluator(RealLit(2) ** RealLit(1100), free_vars=[])
        self.assertEqual(machine.apply_point([]).output, math.inf)
        lo, hi = machine.apply_range([]).output
        self.assertEqual(lo, math.nextafter(math.inf, 0.0))
        self.assertEqual(hi, math.inf)

    def test_input_validation(self):
        x = RealVar('x')
        with self.assertRaisesRegex(ValueError, 'missing referenced variables'):
            build_float_evaluator(x, free_vars=[])
        machine = build_float_evaluator(x)
        with self.assertRaisesRegex(ValueError, 'expected 1 inputs'):
            machine.apply_point([])
        with self.assertRaisesRegex(ValueError, 'reversed'):
            machine.apply_range([(2, 1)])
        with self.assertRaises(ValueError):
            machine.apply_point([1.5])
