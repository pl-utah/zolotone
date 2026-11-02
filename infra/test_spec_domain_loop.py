"""Focused coverage for the incremental spec-domain traversal."""
import unittest
from fractions import Fraction
from unittest.mock import Mock, patch

from zolotone.autogen.autogen import get_spec
from zolotone.autogen import spec_validation as validation
from zolotone.errors import InfeasibleError, ZolotoneError
from zolotone.rival import RivalEvaluation
from zolotone.rival import analysis as rival_analysis
from zolotone.spec.spec_ast import Sqrt
from zolotone.types import UQ


class TestSpecDomainLoop(unittest.TestCase):
    def analyze(self, function):
        return validation.spec_domain_analysis(get_spec('domain-loop', function))

    def test_fixed_point_bounds(self):
        def spec(x: UQ(2, 1), ctx) -> UQ(2, 1):
            ctx.check(x >= ctx.zero())
            return x
        self.assertEqual(self.analyze(spec).output_bounds, (0, Fraction(7, 2)))

    def test_cached_bounds_reused_without_range_or_obligation_queries(self):
        def spec(x: UQ(2, 0), ctx) -> UQ(2, 0):
            return x
        original = get_spec('cached', spec)
        analyzed = validation.spec_domain_analysis(original)
        self.assertIsNone(original.output_bounds)
        self.assertEqual(analyzed.output_bounds, (0, 3))
        with (
            patch.object(validation, 'analyze_typed_domain', side_effect=AssertionError('range recomputed')),
        ):
            validation._check_output_format(analyzed)
            guarded = validation.add_output_guards(analyzed)
        self.assertEqual(guarded.output_bounds, analyzed.output_bounds)
        self.assertEqual(len(guarded.spec_ctx.checks), 1)

    def test_autogenerate_runs_domain_analysis_once(self):
        from zolotone.autogen import autogen

        def spec(x: UQ(2, 0), ctx) -> UQ(2, 0):
            return x
        with (
            patch.object(autogen, 'spec_domain_analysis', wraps=validation.spec_domain_analysis) as analyze,
        ):
            generated = autogen.Autogenerate('cached-pipeline', spec)
        analyze.assert_called_once()
        self.assertEqual(generated.dtype, UQ(2, 0))

    def test_excluded_errors_do_not_contribute(self):
        def spec(x: UQ(1, 0), ctx) -> UQ(1, 0):
            ctx.assume(x > ctx.zero())
            ctx.check((x ** -ctx.one()) > ctx.zero())
            return x ** -ctx.one()
        self.assertEqual(self.analyze(spec).output_bounds, (1, 1))

    def test_false_checks_are_separate_from_errors(self):
        def spec(x: UQ(1, 0), ctx) -> UQ(1, 0):
            ctx.check(x < ctx.one())
            return x
        with self.assertRaisesRegex(InfeasibleError, "check that does not hold"):
            self.analyze(spec)

    def test_false_assumption_rejects_undefined_conjunct(self):
        for reverse in (False, True):
            def spec(x: UQ(1, 0), ctx) -> UQ(1, 0):
                predicates = [x > ctx.zero(), (x ** -ctx.one()) > ctx.zero()]
                for predicate in reversed(predicates) if reverse else predicates:
                    ctx.assume(predicate)
                return x
            with self.subTest(reverse=reverse):
                self.assertEqual(self.analyze(spec).output_bounds, (1, 1))

    def test_unreachable(self):
        def spec(x: UQ(1, 0), ctx) -> UQ(1, 0):
            ctx.assume(x < ctx.zero())
            return x
        with self.assertRaisesRegex(InfeasibleError, "unreachable"):
            self.analyze(spec)

    def test_total_error_raises_without_splitting(self):
        def spec(x: UQ(2, 0), ctx) -> UQ(2, 0):
            return Sqrt(-x - ctx.one())
        with self.assertRaisesRegex(ZolotoneError, r"Domain error in output.*x_0=\[0, 3\]"):
            self.analyze(spec)

    def test_unresolved_assumptions_cannot_prove_checks(self):
        def spec(ctx) -> UQ(1, 0):
            return ctx.zero()
        assumes, checks, output = Mock(), Mock(), Mock()
        assumes.apply_point.return_value = RivalEvaluation()
        with patch.object(rival_analysis, 'build_boolean_evaluator', return_value=checks), patch.object(rival_analysis, 'build_integer_evaluator', side_effect=[assumes, output]):
            with self.assertRaisesRegex(ZolotoneError, "Unresolved assumptions"):
                self.analyze(spec)
        checks.apply_point.assert_not_called()
        output.apply_point.assert_not_called()


if __name__ == '__main__':
    unittest.main()
