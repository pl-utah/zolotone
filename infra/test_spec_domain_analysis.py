"""Typed-domain analysis replaces the continuous range and warning-only paths."""
import unittest
from fractions import Fraction
from unittest.mock import Mock, patch

from zolotone.errors import InfeasibleError, ZolotoneError
from zolotone.rival import RivalEvaluation
from zolotone.rival import analysis
from zolotone.spec.spec_ast import BoolVar, If, RealLit, RealVar, Sqrt
from zolotone.spec.spec_context import SpecContext
from zolotone.types import Bool, Q, UQ


class TestTypedDomainAnalysis(unittest.TestCase):
    def setUp(self):
        self.ctx = SpecContext('typed-domain')
        self.x = RealVar('x')

    def analyze(self, expression, dtype=Q(4, 0), inputs=None, obligations=()):
        if inputs is None:
            inputs = {self.x: UQ(2, 0)}
        return analysis.analyze_typed_domain(
            expression, self.ctx, inputs, dtype, obligations=obligations,
        )

    def test_integer_box_with_fractional_input_scale(self):
        self.assertEqual(self.analyze(self.x, Q(3, 2), {self.x: Q(3, 2)}),
                         (Fraction(-4), Fraction(15, 4)))

    def test_multivariate_range(self):
        y = RealVar('y')
        self.assertEqual(self.analyze(self.x + y, inputs={self.x: Q(3, 0), y: UQ(2, 0)}),
                         (-4, 6))

    def test_boolean_codes(self):
        flag = BoolVar('flag')
        self.assertEqual(self.analyze(~flag, Bool(), {flag: Bool()}), (0, 1))

    def test_constant_generic_output(self):
        self.assertEqual(self.analyze(RealLit(Fraction(3, 8)), Q, {}),
                         (Fraction(3, 8), Fraction(3, 8)))

    def test_generic_identity_preserves_high_input_precision(self):
        value = Fraction(1, 2**40)
        self.ctx.assume(self.x.eq(RealLit(value)))
        for family in (Q, UQ):
            with self.subTest(family=family):
                self.assertEqual(self.analyze(self.x, family, {self.x: UQ(1, 40)}),
                                 (value, value))

    def test_generic_constant_has_outward_bounds(self):
        value = Fraction(1, 3)
        lower, upper = self.analyze(RealLit(value), UQ, {})
        self.assertIsInstance(lower, Fraction)
        self.assertLessEqual(lower, value)
        self.assertGreaterEqual(upper, value)
        self.assertLess(lower, upper)

    def test_generic_root_round_returns_values_not_codes(self):
        expression = self.ctx.rne(RealLit(Fraction(3, 8)), frac_bits=2)
        self.assertEqual(self.analyze(expression, UQ, {}), (Fraction(1, 2),) * 2)

    def test_generic_domain_errors_remain_errors(self):
        with self.assertRaisesRegex(ZolotoneError, 'Domain error in output'):
            self.analyze(RealLit(0) ** RealLit(-1), UQ, {})

    def test_generic_nonfinite_enclosure_is_reported(self):
        with self.assertRaisesRegex(ZolotoneError, 'Non-finite output bounds'):
            self.analyze(RealLit(2) ** RealLit(1100), UQ, {})

    def test_untyped_free_variable_is_rejected(self):
        with self.assertRaisesRegex(ZolotoneError, 'needs input dtypes'):
            self.analyze(self.x + RealVar('y'))

    def test_integer_domain_excludes_noninteger_singularity(self):
        self.ctx.assume(self.x >= RealLit(Fraction(1, 2)))
        self.ctx.assume(self.x <= RealLit(1))
        expr = (self.x - RealLit(Fraction(3, 4))) ** RealLit(-1)
        self.assertEqual(self.analyze(expr), (4, 4))

    def test_unreachable_discrete_equality(self):
        self.ctx.assume(self.x.eq(RealLit(Fraction(1, 2))))
        with self.assertRaisesRegex(InfeasibleError, 'unreachable'):
            self.analyze(self.x)

    def test_total_error_and_partial_error_subdivision(self):
        with self.assertRaisesRegex(ZolotoneError, 'Domain error in output'):
            self.analyze(Sqrt(-self.x - RealLit(1)))
        with self.assertRaisesRegex(ZolotoneError, 'Domain error in output'):
            self.analyze(self.x ** RealLit(-1))

    def test_assumption_and_check_errors_are_reported(self):
        invalid = (self.x ** RealLit(-1)) > RealLit(0)
        with self.assertRaisesRegex(ZolotoneError, 'Domain error in checks'):
            self.analyze(self.x, obligations=[invalid])
        self.ctx.assume(invalid)
        with self.assertRaisesRegex(ZolotoneError, 'Domain error in assumptions'):
            self.analyze(self.x)

    def test_dead_branch_error_is_not_reported(self):
        self.ctx.assume(self.x.eq(RealLit(0)))
        expr = If(self.x.eq(RealLit(0)), RealLit(1), self.x ** RealLit(-1))
        self.assertEqual(self.analyze(expr), (1, 1))

    def test_unresolved_point_is_not_a_proof(self):
        assumes, checks, output = Mock(), Mock(), Mock()
        assumes.apply_point.return_value = RivalEvaluation(precondition=True)
        checks.apply_point.return_value = RivalEvaluation(output=1)
        output.apply_point.return_value = RivalEvaluation()
        with (
            patch.object(analysis, 'build_integer_evaluator', side_effect=[assumes, output]),
            patch.object(analysis, 'build_boolean_evaluator', return_value=checks),
            self.assertRaisesRegex(ZolotoneError, 'Unresolved output'),
        ):
            self.analyze(RealLit(0), inputs={})

    def test_error_range_names_and_decodes_inputs(self):
        y = RealVar('y')
        expr = Sqrt(-self.x - RealLit(1)) + y
        with self.assertRaisesRegex(
            ZolotoneError, r'Domain error in output at input range \[y=\[-1, 1/2\], x=\[0, 3/2\]\]',
        ):
            self.analyze(expr, inputs={y: Q(1, 1), self.x: UQ(1, 1)})

    def test_error_point_names_and_decodes_inputs(self):
        y = RealVar('y')
        self.ctx.assume(self.x.eq(RealLit(Fraction(1, 2))))
        self.ctx.assume(y.eq(RealLit(1)))
        expr = (self.x - RealLit(Fraction(1, 2))) ** RealLit(-1) + y
        with self.assertRaisesRegex(
            ZolotoneError, r'Domain error in output at input point \[y=1, x=1/2\]',
        ):
            self.analyze(expr, inputs={y: UQ(1, 0), self.x: UQ(1, 1)})


if __name__ == '__main__':
    unittest.main()
