import unittest
import math
from fractions import Fraction

from zolotone import Autogenerate, Bool, Q, UQ, sin
from zolotone.autogen.autogen import (
    get_spec, _expand_faithful_roundings,
)
from zolotone.spec.spec_ast import BoolLit, RealLit, substitute_spec_node
from zolotone.errors import MissingError


class TestFaithfulAssumptions(unittest.TestCase):
    def test_multiple_choices_with_assumptions_are_rejected(self):
        for nested in (False, True):
            def spec(x: UQ(0, 3), ctx) -> UQ:
                a = ctx.fr(x, 2)
                b = ctx.fr((a if nested else x) + ctx.real_val(Fraction(1, 8)), 1)
                ctx.assume(a.ne(b))
                return a + b

            with self.subTest(nested=nested), self.assertRaisesRegex(
                MissingError, 'supports only one choice; found 2 choices'
            ):
                _expand_faithful_roundings(get_spec('multiple', spec))

    def test_corrected_sine_tables_use_narrow_output(self):
        def spec(x: UQ(4, 2), ctx) -> Q:
            z = ctx.fr(sin(x), frac_bits=4)
            ctx.assume(abs(z).ne(ctx.one()))
            return z

        generated = Autogenerate('corrected_sine', spec)
        seen, tables = set(), []

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            if node.name.startswith('LUT_'):
                tables.append(node)
            for argument in node.args:
                visit(argument)
            if hasattr(node, 'inner_tree'):
                visit(node.inner_tree)

        visit(generated)
        self.assertEqual(len(tables), 2)
        self.assertTrue(all(table.dtype == Q(1, 4) for table in tables))
        for code in range(64):
            scaled = math.sin(code / 4) * 16
            toward = math.trunc(scaled)
            away = math.ceil(scaled) if scaled >= 0 else math.floor(scaled)
            generated.args[0].load_value(UQ(4, 2).from_bits(code))
            for direction in (0, 1):
                expected = away if direction else toward
                if abs(expected) == 16:
                    expected = toward if direction else away
                generated.args[1].load_value(Bool().from_bits(direction))
                self.assertEqual(generated.evaluate().to_python(), Fraction(expected, 16))

    def test_choice_redirection_and_excluded_inputs(self):
        for allow_nonzero in (True, False):
            def spec(x: UQ(1, 3), ctx) -> UQ(2, 2):
                z = ctx.fr(x, 2)
                condition = z.ne(ctx.zero()) if allow_nonzero else z.eq(ctx.zero())
                ctx.assume(condition)
                ctx.check(condition)
                return z

            resolved = _expand_faithful_roundings(get_spec('resolve', spec))
            x, choice = resolved.spec_inputs

            def evaluate(expr, value, direction):
                expr = substitute_spec_node(expr, x, RealLit(value))
                return substitute_spec_node(expr, choice, BoolLit(direction))._constant_fold().value

            for direction in (False, True):
                self.assertEqual(evaluate(resolved.spec_ast, Fraction(1, 8), direction),
                                 Fraction(1, 4) if allow_nonzero else 0)
                for condition in (*resolved.spec_ctx.assumes, *resolved.spec_ctx.checks):
                    self.assertTrue(evaluate(condition, Fraction(1, 8), direction))
                excluded = 0 if allow_nonzero else Fraction(1, 4)
                self.assertFalse(evaluate(resolved.spec_ctx.assumes[0], excluded, direction))
                if allow_nonzero:
                    self.assertEqual(evaluate(resolved.spec_ast, Fraction(3, 8), direction),
                                     Fraction(1, 2) if direction else Fraction(1, 4))

    def test_generated_output_redirects_invalid_choice(self):
        def spec(x: UQ(1, 3), ctx) -> UQ(2, 2):
            z = ctx.fr(x, 2)
            ctx.assume(z.ne(ctx.zero()))
            ctx.check(z.ne(ctx.zero()))
            return z

        generated = Autogenerate('redirect_nonzero', spec)
        generated.args[0].load_value(UQ(1, 3).from_bits(1))
        for direction in (0, 1):
            generated.args[1].load_value(Bool().from_bits(direction))
            self.assertEqual(generated.evaluate().to_python(), Fraction(1, 4))
