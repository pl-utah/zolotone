import contextlib
from fractions import Fraction
import io
import unittest
from unittest.mock import Mock, patch

from zolotone import Autogenerate, Bool, If, Q, UQ, log2, pi, sin
from zolotone.autogen import VerificationInconclusive, zfv
from zolotone.autogen.autogen import get_spec
from zolotone.errors import ZolotoneError


class TestFloatVerification(unittest.TestCase):
    def generate(self, function):
        with contextlib.redirect_stdout(io.StringIO()):
            return Autogenerate(function.__name__, function)

    def test_faithful_sine_checks_both_choices_against_unrounded_reference(self):
        def spec(x: UQ(2, 2), ctx) -> Q:
            return ctx.fr(sin(x * pi()) / ctx.real_val(4), frac_bits=4)

        original = get_spec('sine', spec)
        component = self.generate(spec)
        original_ast = original.spec_ast
        original_assumes = list(original.spec_ctx.assumes)
        result = zfv(original, component)
        self.assertIs(original.spec_ast, original_ast)
        self.assertEqual(original.spec_ctx.assumes, original_assumes)
        self.assertEqual(original.rounding_choices, {})
        self.assertEqual(result.checked, 32)
        self.assertEqual(result.skipped, 0)
        self.assertEqual(result.ulp, Fraction(1, 16))
        self.assertEqual(result.error_bound, Fraction(1, 16))

    def test_assumptions_skip_undefined_log_input(self):
        def spec(x: UQ(2, 1), ctx) -> Q:
            ctx.assume(x.ne(ctx.zero()))
            return ctx.fr(log2(x), frac_bits=3)

        result = zfv(get_spec('log', spec), self.generate(spec))
        self.assertEqual(result.checked, 14)
        self.assertEqual(result.skipped, 2)

    def test_signed_fractional_and_boolean_inputs_use_cpp_argument_order(self):
        def spec(x: Q(2, 1), select: Bool(), ctx) -> Q:
            return ctx.rne(If(select, x, -x), frac_bits=1)

        result = zfv(get_spec('signed', spec), self.generate(spec))
        self.assertEqual(result.checked, 16)
        self.assertEqual(result.error_bound, Fraction(1, 4))

    def test_reports_incorrect_cpp_result(self):
        def spec(x: UQ(2, 0), ctx) -> UQ(3, 0):
            return ctx.rne(x, frac_bits=0)

        def incorrect(x: UQ(2, 0), ctx) -> UQ(3, 0):
            return x + ctx.real_val(2)

        with self.assertRaisesRegex(ZolotoneError, r'zfv mismatch.*<= 1/2'):
            zfv(get_spec('reference', spec), self.generate(incorrect))

    def test_preserves_inner_rounding(self):
        def spec(x: UQ(1, 2), ctx) -> UQ:
            return ctx.rne(ctx.rtz(x, frac_bits=1), frac_bits=2)

        result = zfv(get_spec('nested', spec), self.generate(spec))
        self.assertEqual(result.checked, 8)
        self.assertEqual(result.error_bound, Fraction(1, 8))

    def test_faithful_assumptions_execute_every_choice(self):
        from infra.compile_cpp import jit_compile

        for allow_nonzero in (False, True):
            def spec(x: UQ(1, 3), ctx) -> UQ(2, 2):
                rounded = ctx.fr(x, frac_bits=2)
                ctx.assume(rounded.ne(ctx.zero()) if allow_nonzero else rounded.eq(ctx.zero()))
                return rounded

            original = get_spec('faithful_assume', spec)
            original_assumes = tuple(original.spec_ctx.assumes)
            component = self.generate(spec)
            calls = []

            def track_cpp(node, **kwargs):
                tempdir, cpp = jit_compile(node, **kwargs)

                def evaluate(raw, choice):
                    calls.append((raw, choice))
                    return cpp(raw, choice)

                return tempdir, evaluate

            with self.subTest(allow_nonzero=allow_nonzero), patch(
                'infra.compile_cpp.jit_compile', side_effect=track_cpp,
            ):
                result = zfv(original, component)
                expected = [(raw, choice) for raw in range(16) for choice in (0, 1)]
                self.assertEqual(calls, expected)
                reachable = 30 if allow_nonzero else 4
                self.assertEqual(result.checked, reachable)
                self.assertEqual(result.skipped, 32 - reachable)
                self.assertEqual(tuple(original.spec_ctx.assumes), original_assumes)
                self.assertEqual(original.rounding_choices, {})

    def test_preserves_nested_faithful_reference_for_each_choice(self):
        def spec(x: UQ(1, 2), ctx) -> UQ(2, 2):
            return ctx.rne(ctx.fr(x, frac_bits=1), frac_bits=2)

        result = zfv(get_spec('nested_faithful', spec), self.generate(spec))
        self.assertEqual(result.checked, 16)
        self.assertEqual(result.skipped, 0)
        self.assertEqual(result.error_bound, Fraction(1, 8))

    def test_unresolved_assumptions_are_not_skipped(self):
        def spec(x: UQ(1, 0), ctx) -> UQ:
            return ctx.rne(x, frac_bits=0)

        component = self.generate(spec)
        machine = Mock()
        machine.apply_point.side_effect = ValueError('Unsamplable input for rival machine')
        with patch('zolotone.autogen.spec_verification.build_integer_evaluator', return_value=machine):
            with self.assertRaisesRegex(VerificationInconclusive, 'assumption evaluation'):
                zfv(get_spec('unresolved_assumes', spec), component)

    def test_rne_rejects_error_above_half_ulp(self):
        def reference(x: Q(1, 2), ctx) -> Q(2, 0):
            return ctx.rne(x, frac_bits=0)

        def incorrect(x: Q(1, 2), ctx) -> Q(2, 0):
            return ctx.rtz(x, frac_bits=0)

        spec = get_spec('rne_reference', reference)
        result = zfv(spec, self.generate(reference))
        self.assertEqual(result.error_bound, Fraction(1, 2))
        with self.assertRaisesRegex(ZolotoneError, 'zfv mismatch'):
            zfv(spec, self.generate(incorrect))

    def test_directed_rounding_can_exceed_half_ulp(self):
        for mode in ('rtz', 'raz'):
            def spec(x: Q(1, 2), ctx) -> Q(2, 0):
                return getattr(ctx, mode)(x, frac_bits=0)

            with self.subTest(mode=mode):
                result = zfv(get_spec(mode, spec), self.generate(spec))
                self.assertEqual(result.checked, 8)
                self.assertEqual(result.error_bound, 1)

    def test_requires_original_root_round(self):
        def spec(x: UQ(2, 0), ctx) -> UQ:
            return x

        with self.assertRaisesRegex(NotImplementedError, 'round operation to be a root'):
            zfv(get_spec('unrounded', spec), self.generate(spec))

    def test_concrete_output_dtype_defines_unrounded_grid(self):
        def reference(x: UQ(1, 1), ctx) -> UQ(1, 1):
            return ctx.rtz(x / ctx.real_val(2))

        def implementation(x: UQ(1, 1), ctx) -> UQ(1, 1):
            return ctx.rtz(x / ctx.real_val(2), frac_bits=1)

        result = zfv(get_spec('dtype_precision', reference), self.generate(implementation))
        self.assertEqual(result.ulp, Fraction(1, 2))
        self.assertEqual(result.error_bound, Fraction(1, 2))

    def test_reference_errors_and_empty_domain_cannot_pass(self):
        def implementation(x: UQ(1, 0), ctx) -> UQ:
            return x

        def invalid(x: UQ(1, 0), ctx) -> Q:
            return ctx.rne(log2(x), frac_bits=1)

        def unreachable(x: UQ(1, 0), ctx) -> UQ:
            ctx.assume(ctx.false())
            return ctx.rne(x, frac_bits=0)

        component = self.generate(implementation)
        with self.assertRaisesRegex(ZolotoneError, 'spec comparison.*domain error'):
            zfv(get_spec('invalid', invalid), component)
        with self.assertRaisesRegex(ZolotoneError, 'C\\+\\+ returned despite false assumptions'):
            zfv(get_spec('unreachable', unreachable), component)

    def test_cpp_asserting_on_reachable_input_is_rejected(self):
        def reference(x: UQ(1, 0), ctx) -> UQ:
            return ctx.rne(x, frac_bits=0)

        def restricted(x: UQ(1, 0), ctx) -> UQ:
            ctx.assume(x.ne(ctx.zero()))
            return ctx.rne(x, frac_bits=0)

        with self.assertRaisesRegex(ZolotoneError, r'C\+\+ asserted despite true assumptions'):
            zfv(get_spec('reference', reference), self.generate(restricted))

    def test_cpp_without_required_assumption_assertion_is_rejected(self):
        def restricted(x: UQ(1, 0), ctx) -> UQ:
            ctx.assume(x.ne(ctx.zero()))
            return ctx.rne(x, frac_bits=0)

        def unrestricted(x: UQ(1, 0), ctx) -> UQ:
            return ctx.rne(x, frac_bits=0)

        with self.assertRaisesRegex(ZolotoneError, r'C\+\+ returned despite false assumptions'):
            zfv(get_spec('restricted', restricted), self.generate(unrestricted))

    def test_caught_cpp_assertion_allows_subsequent_calls(self):
        from infra.compile_cpp import CppAssertionError, jit_compile

        def spec(x: UQ(1, 0), ctx) -> UQ:
            ctx.assume(x.ne(ctx.zero()))
            return ctx.rne(x, frac_bits=0)

        tempdir, cpp = jit_compile(self.generate(spec), catch_assertions=True)
        try:
            for _ in range(2):
                with self.assertRaises(CppAssertionError):
                    cpp(0)
                self.assertEqual(cpp(1), 1)
        finally:
            tempdir.cleanup()

    def test_no_binary64_allowance_at_large_magnitude(self):
        origin = 1 << 53

        def reference(x: UQ(1, 0), ctx) -> UQ(54, 0):
            return ctx.fr(x + ctx.real_val(origin + 1), frac_bits=0)

        def adjacent(x: UQ(1, 0), ctx) -> UQ(54, 0):
            return x + ctx.real_val(origin + 2)

        def outside(x: UQ(1, 0), ctx) -> UQ(54, 0):
            return x + ctx.real_val(origin + 3)

        spec = get_spec('large_reference', reference)
        for implementation in (adjacent, outside):
            with self.subTest(implementation=implementation.__name__):
                with self.assertRaisesRegex(ZolotoneError, 'zfv mismatch'):
                    zfv(spec, self.generate(implementation))

    def test_exact_large_output_above_binary64_precision(self):
        def spec(x: UQ(1, 0), ctx) -> UQ:
            return ctx.rne(x + ctx.real_val((1 << 53) + 1), frac_bits=0)

        result = zfv(get_spec('exact_large', spec), self.generate(spec))
        self.assertEqual(result.checked, 2)
        self.assertEqual(result.error_bound, Fraction(1, 2))

    def test_incorrect_large_output_hidden_by_binary64_is_rejected(self):
        origin = 1 << 53

        def reference(x: UQ(1, 0), ctx) -> UQ(54, 0):
            return ctx.rne(x + ctx.real_val(origin + 1), frac_bits=0)

        def incorrect(x: UQ(1, 0), ctx) -> UQ:
            return x + ctx.real_val(origin)

        with self.assertRaisesRegex(ZolotoneError, 'zfv mismatch'):
            zfv(get_spec('hidden_error', reference), self.generate(incorrect))

    def test_refines_comparisons_near_half_ulp_boundary(self):
        epsilon = Fraction(1, 1 << 200)

        def implementation(x: UQ(1, 0), ctx) -> UQ:
            return x

        component = self.generate(implementation)
        for offset in (-epsilon, Fraction(0), epsilon):
            def reference(x: UQ(1, 0), ctx) -> UQ(2, 0):
                return ctx.rne(x + ctx.real_val(Fraction(1, 2) + offset), frac_bits=0)

            with self.subTest(offset=offset):
                original = get_spec('near_boundary', reference)
                if offset > 0:
                    with self.assertRaisesRegex(ZolotoneError, 'zfv mismatch'):
                        zfv(original, component)
                else:
                    result = zfv(original, component)
                    self.assertEqual(result.checked, 2)

    def test_only_cpp_choice_one_is_wrong(self):
        from infra.compile_cpp import jit_compile

        def spec(x: UQ(1, 1), ctx) -> UQ:
            return ctx.fr(x, frac_bits=0)

        original = get_spec('choice_one', spec)
        component = self.generate(spec)

        def corrupt_choice_one(node, **kwargs):
            tempdir, cpp = jit_compile(node, **kwargs)

            def evaluate(raw, choice):
                return 1 if raw == 0 and choice == 1 else cpp(raw, choice)

            return tempdir, evaluate

        with patch('infra.compile_cpp.jit_compile', side_effect=corrupt_choice_one):
            with self.assertRaisesRegex(ZolotoneError, 'zfv mismatch'):
                zfv(original, component)

    def test_unresolved_predicate_is_inconclusive(self):
        def spec(x: UQ(1, 0), ctx) -> UQ:
            return ctx.rne(x, frac_bits=0)

        original = get_spec('unresolved', spec)
        component = self.generate(spec)
        machine = Mock()
        machine.apply_point.side_effect = ValueError('Unsamplable input for rival machine')
        with patch('zolotone.autogen.spec_verification.build_boolean_evaluator', return_value=machine):
            with self.assertRaises(VerificationInconclusive):
                zfv(original, component)


if __name__ == '__main__':
    unittest.main()
