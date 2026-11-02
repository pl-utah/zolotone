import pickle
import unittest

from zolotone import Cos, Pi, RealLit, Sin, pi
from zolotone.rival import build_machine, collect_free_vars, to_rival_ir
from zolotone.spec.spec_ast import children, substitute_literals, variables


class TestSpecPi(unittest.TestCase):
    def test_pi_does_not_hide_lut_input(self):
        from zolotone.autogen.spec_lowering import lut_inputs
        from zolotone.spec.spec_ast import RealVar, Round, RoundingMode

        x = RealVar("x")
        expr = Round(Sin(x * pi()), RoundingMode.RTZ, 4)
        self.assertIn(x, lut_inputs(expr))

    def test_pi_stays_exact_through_ast_operations(self):
        node = pi()
        self.assertIsInstance(node, Pi)
        self.assertEqual(str(node), "pi()")
        self.assertEqual(children(node), ())
        self.assertEqual(variables(node), set())
        self.assertEqual(collect_free_vars([node]), [])
        self.assertIs(node._constant_fold(), node)
        self.assertTrue(pickle.loads(pickle.dumps(node)).identical(node))
        expr = Sin(node + RealLit(0))
        self.assertTrue(expr._constant_fold().identical(Sin(Pi())))
        self.assertTrue(substitute_literals(expr, {}).identical(Sin(Pi())))

    def test_unsupported_lowerings_match_trigonometric_nodes(self):
        node = pi()
        for lower in (node.to_egglog, lambda: node.to_z3({}),
                      lambda: node.to_dreal({})):
            with self.assertRaisesRegex(NotImplementedError, "lowering of Pi"):
                lower()

    def test_rival_evaluates_pi_and_trigonometric_expressions(self):
        self.assertEqual(to_rival_ir(pi()), {"op": "pi"})
        lower, upper = build_machine([pi()], free_vars=[]).apply_range([])
        self.assertGreater(lower, 3)
        self.assertLess(upper, 4)
        self.assertLessEqual(lower, upper)
        for expr, expected in ((Sin(pi()), 0.0), (Cos(pi()), -1.0)):
            with self.subTest(expr=str(expr)):
                lower, upper = build_machine([expr], free_vars=[]).apply_range([])
                self.assertLessEqual(lower, expected)
                self.assertGreaterEqual(upper, expected)
                self.assertLess(upper - lower, 1e-14)


if __name__ == "__main__":
    unittest.main()
