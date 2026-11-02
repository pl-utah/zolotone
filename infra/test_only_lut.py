"""LUT eligibility validation leaves the normal lowering pipeline intact."""
import unittest
from fractions import Fraction
from unittest.mock import patch

from zolotone.autogen.autogen import Autogenerate, get_spec
from zolotone.autogen import spec_validation as validation
from zolotone.errors import MissingError
from zolotone.spec.spec_ast import RealExpr
from zolotone.types import UQ


class TestOnlyLut(unittest.TestCase):
    def test_faithful_rounding_is_allowed(self):
        def root(x: UQ(2, 1), ctx) -> UQ(2, 0):
            return ctx.fr(x, frac_bits=0)

        def nested(x: UQ(2, 1), ctx) -> UQ(2, 0):
            return ctx.rne(ctx.fr(x, frac_bits=0), frac_bits=0)

        from zolotone.autogen.autogen import _expand_faithful_roundings
        for function in (root, nested):
            with self.subTest(function=function.__name__):
                spec = get_spec('faithful_lut', function)
                validation._validate_only_lut_spec(spec)
                expanded = _expand_faithful_roundings(spec)
                self.assertEqual(len(expanded.rounding_choices), 1)

    def test_concrete_dtype_uses_normal_search(self):
        def spec(x: UQ(2, 0), ctx) -> UQ(5, 0):
            return ctx.rne(x * x)

        from zolotone.autogen import spec_lowering
        with patch.object(spec_lowering, 'search_lower_spec_to_impl',
                          wraps=spec_lowering.search_lower_spec_to_impl) as search:
            result = Autogenerate('only_lut_square', spec, only_LUT=True)
        self.assertTrue(any(isinstance(call.args[0], RealExpr) for call in search.call_args_list))
        self.assertEqual(result.dtype, UQ(5, 0))
        for value in range(4):
            result.args[0].load_value(UQ(2, 0).from_float(value))
            self.assertEqual(result.evaluate().to_python(), value * value)

    def test_folded_constant_uses_normal_lowering(self):
        def spec(x: UQ(1, 0), ctx) -> UQ:
            return ctx.rne(ctx.real_val(Fraction(1, 4)), frac_bits=2)

        result = Autogenerate('only_lut_fraction', spec, only_LUT=True)
        self.assertEqual(result.dtype, UQ(0, 2))
        for value in range(2):
            result.args[0].load_value(UQ(1, 0).from_float(value))
            self.assertEqual(result.evaluate().to_python(), Fraction(1, 4))

    def test_invalid_specs(self):
        def unrounded(x: UQ(2, 0), ctx) -> UQ:
            return x

        def multiple(x: UQ(2, 0), y: UQ(2, 0), ctx) -> UQ:
            return ctx.rne(x + y, frac_bits=0)

        def oversized(x: UQ(21, 0), ctx) -> UQ:
            return ctx.rne(x, frac_bits=0)

        for function, error in ((unrounded, MissingError),
                                (multiple, MissingError), (oversized, MissingError)):
            with self.subTest(function=function.__name__), self.assertRaises(error):
                validation._validate_only_lut_spec(get_spec('invalid', function))
