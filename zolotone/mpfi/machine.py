"""Recursive spec evaluation using native MPFI interval operations."""

from fractions import Fraction
import operator
from typing import Sequence

from .interval import IntervalField

from ..spec import spec_ast as ast


_UNARY = {
    ast.Neg: operator.neg,
    ast.Abs: abs,
    **{cls: operator.methodcaller("_unary", name) for cls, name in (
        (ast.Pow2, "sqr"), (ast.Sqrt, "sqrt"),
        (ast.Exp, "exp"), (ast.Exp2, "exp2"),
        (ast.Log, "log"), (ast.Log2, "log2"), (ast.Log10, "log10"),
        (ast.Sin, "sin"), (ast.Cos, "cos"), (ast.Tan, "tan"),
        (ast.Asin, "asin"), (ast.Acos, "acos"), (ast.Atan, "atan"),
        (ast.Sinh, "sinh"), (ast.Cosh, "cosh"), (ast.Tanh, "tanh"),
        (ast.Asinh, "asinh"), (ast.Acosh, "acosh"), (ast.Atanh, "atanh"),
    )},
}
_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mul: operator.mul,
    ast.Pow: operator.pow,
    ast.Min: lambda x, y: x.min(y),
    ast.Max: lambda x, y: x.max(y),
}


class MPFIMachine:
    """Compiled recursive functions; each call supplies one interval field."""

    def __init__(self, outputs, free_vars):
        self._outputs = tuple(outputs)
        self.free_vars = tuple(free_vars)

    def apply_point(
        self, point: Sequence[int | float | Fraction], *, precision: int = 128,
    ) -> list:
        """Evaluate at precision bits and return owned MPFIInterval objects.

        MPFI handles outward rounding. Its exceptional values
        pass through; there is no domain analysis or precision refinement.
        """
        if type(precision) is not int:
            raise TypeError("precision must be an integer number of bits")
        field = IntervalField(precision)
        values = list(point)
        if len(values) != len(self.free_vars):
            raise ValueError(f"expected {len(self.free_vars)} inputs, got {len(values)}")
        inputs = []
        for name, value in zip(self.free_vars, values):
            if isinstance(value, float):
                inputs.append(field(value))
            elif not isinstance(value, bool) and isinstance(value, (int, Fraction)):
                inputs.append(field(Fraction(value)))
            else:
                raise TypeError(f"input {name!r} requires int, float, or Fraction")
        return [output(field, inputs) for output in self._outputs]


def mpfi_compile(
    expressions: Sequence[ast.RealExpr], *, free_vars: Sequence[str],
) -> MPFIMachine:
    """Compile already typechecked numeric ASTs into recursive callables.

    No rewriting, shared-expression caching, or operand typechecking is done.
    Nodes without a supported interval operation raise NotImplementedError.
    """
    outputs, variables = list(expressions), list(free_vars)
    if not outputs:
        raise ValueError("expressions must be nonempty")
    if any(not isinstance(name, str) for name in variables):
        raise TypeError("free_vars must contain strings")
    if len(set(variables)) != len(variables):
        raise ValueError("free_vars must be unique")
    positions = {name: index for index, name in enumerate(variables)}

    def compile_node(node):
        cls = type(node)
        if cls is ast.RealVar:
            if node.name not in positions:
                raise ValueError(f"free_vars is missing {node.name!r}")
            index = positions[node.name]
            return lambda field, inputs: inputs[index]
        if cls is ast.RealLit:
            value = Fraction(node.value)
            return lambda field, inputs: field(value)
        if cls is ast.Pi:
            return lambda field, inputs: field.pi()
        if cls in _UNARY:
            op = _UNARY[cls]
            arg, = [compile_node(child) for child in ast.children(node)]
            return lambda field, inputs: op(arg(field, inputs))
        if cls in _BINARY:
            op = _BINARY[cls]
            lhs, rhs = [compile_node(child) for child in ast.children(node)]
            return lambda field, inputs: op(lhs(field, inputs), rhs(field, inputs))
        raise NotImplementedError(f"unsupported MPFI spec node: {cls.__name__}")

    return MPFIMachine([compile_node(output) for output in outputs], variables)
