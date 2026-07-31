from __future__ import annotations

from fractions import Fraction
from typing import Any

from egglog import Expr

from functools import reduce
from operator import or_, and_

from .spec_ast import (
    Abs,
    Add,
    And,
    BoolEq,
    BoolExpr,
    BoolLit,
    BoolVar,
    Eq,
    Ge,
    Gt,
    If,
    Le,
    Lt,
    Max,
    Min,
    Mul,
    Neg,
    Not,
    NotEq,
    Or,
    Pow,
    RealExpr,
    RealLit,
    RealVar,
)

_MATH_BOOL_METHODS = {"Eq", "NotEq", "Lt", "Le", "Gt", "Ge"}


def andmap(*args):
    return reduce(and_, args)

def ormap(*args):
    return reduce(or_, args)


def from_egglog(egg_node: Expr) -> RealExpr | BoolExpr:
    if not hasattr(egg_node, "__egg_typed_expr__"):
        raise TypeError(f"Expected egglog Expr, got {type(egg_node).__name__}")
    root = egg_node.__egg_typed_expr__

    def op(typed_expr: Any) -> tuple[str, str | None, Any]:
        expr_decl = typed_expr.expr
        if not hasattr(expr_decl, "callable") or not hasattr(expr_decl, "args"):
            raise TypeError(f"Expected call expression, got {type(expr_decl).__name__}")
        callable_decl = expr_decl.callable
        ident_name = getattr(getattr(callable_decl, "ident", None), "name", None)
        if ident_name is None:
            raise TypeError(f"Missing callable identifier on {type(callable_decl).__name__}")
        method_name = getattr(callable_decl, "method_name", None)
        return ident_name, method_name, expr_decl.args

    def lit(typed_expr: Any) -> str | int:
        expr_decl = typed_expr.expr
        if not hasattr(expr_decl, "value"):
            raise TypeError(f"Expected literal, got {type(expr_decl).__name__}")
        value = expr_decl.value
        if not isinstance(value, (str, int)):
            raise TypeError(f"Expected str|int literal, got {type(value).__name__}")
        return value

    def expect(args: Any, arity: int, name: str) -> None:
        if len(args) != arity:
            raise TypeError(f"{name} expects {arity} args, got {len(args)}")

    def parse_bigint(typed_expr: Any) -> int:
        ident_name, method_name, args = op(typed_expr)
        if ident_name != "BigInt":
            raise TypeError(f"Expected BigInt constructor, got {ident_name}")
        expect(args, 1, "BigInt")
        value = lit(args[0])
        if method_name == "from_string":
            if not isinstance(value, str):
                raise TypeError(f"Expected string literal, got {type(value).__name__}")
            return int(value)
        if method_name is None:
            return int(value)
        raise TypeError(f"Unsupported BigInt constructor: {ident_name}.{method_name}")

    def parse_bigrat(typed_expr: Any) -> float | int:
        ident_name, method_name, args = op(typed_expr)
        if ident_name != "BigRat" or method_name is not None:
            raise TypeError(f"Expected BigRat constructor, got {ident_name}.{method_name}")
        expect(args, 2, "BigRat")
        value = Fraction(parse_bigint(args[0]), parse_bigint(args[1]))
        if value.denominator == 1:
            return value.numerator
        return float(value)

    def parse_math(typed_expr: Any) -> RealExpr:
        ident_name, method_name, args = op(typed_expr)
        if ident_name != "Math":
            raise TypeError(f"Expected Math constructor, got {ident_name}.{method_name}")
        if method_name == "Var":
            expect(args, 1, "Math.Var")
            name = lit(args[0])
            if not isinstance(name, str):
                raise TypeError(f"Expected string literal, got {type(name).__name__}")
            return RealVar(name)
        if method_name == "Num":
            expect(args, 1, "Math.Num")
            return RealLit(parse_bigrat(args[0]))
        if method_name == "Add":
            expect(args, 2, "Math.Add")
            return parse_math(args[0]) + parse_math(args[1])
        if method_name == "Mul":
            expect(args, 2, "Math.Mul")
            return parse_math(args[0]) * parse_math(args[1])
        if method_name == "Pow":
            expect(args, 2, "Math.Pow")
            return parse_math(args[0]) ** parse_math(args[1])
        if method_name == "Neg":
            expect(args, 1, "Math.Neg")
            return (- parse_math(args[0]))
        if method_name == "Abs":
            expect(args, 1, "Math.Abs")
            return abs(parse_math(args[0]))
        if method_name == "Max":
            expect(args, 2, "Math.Max")
            return parse_math(args[0]).max(parse_math(args[1]))
        if method_name == "Min":
            expect(args, 2, "Math.Min")
            return parse_math(args[0]).min(parse_math(args[1]))
        if method_name == "If":
            expect(args, 3, "Math.If")
            return If(parse_bool(args[0]), parse_math(args[1]), parse_math(args[2]))
        raise NotImplementedError(f"Unsupported Math constructor: {method_name}")

    def parse_bool(typed_expr: Any) -> BoolExpr:
        ident_name, method_name, args = op(typed_expr)
        if ident_name == "MathBool":
            if method_name == "Var":
                expect(args, 1, "MathBool.Var")
                name = lit(args[0])
                if not isinstance(name, str):
                    raise TypeError(f"Expected string literal, got {type(name).__name__}")
                return BoolVar(name)
            if method_name == "True_":
                expect(args, 0, "MathBool.True_")
                return BoolLit(True)
            if method_name == "False_":
                expect(args, 0, "MathBool.False_")
                return BoolLit(False)
            if method_name == "Not":
                expect(args, 1, "MathBool.Not")
                return Not(parse_bool(args[0]))
            if method_name in {"Eq", "BoolEq"}:
                expect(args, 2, "MathBool.Eq")
                return BoolEq(parse_bool(args[0]), parse_bool(args[1]))
            if method_name == "And":
                expect(args, 2, "MathBool.And")
                return And(parse_bool(args[0]), parse_bool(args[1]))
            if method_name == "Or":
                expect(args, 2, "MathBool.Or")
                return Or(parse_bool(args[0]), parse_bool(args[1]))
            raise NotImplementedError(f"Unsupported MathBool constructor: {method_name}")

        if ident_name != "Math":
            raise TypeError(f"Expected boolean expression, got {ident_name}.{method_name}")
        if method_name == "Eq":
            expect(args, 2, "Math.Eq")
            return Eq(parse_math(args[0]), parse_math(args[1]))
        if method_name == "NotEq":
            expect(args, 2, "Math.NotEq")
            return NotEq(parse_math(args[0]), parse_math(args[1]))
        if method_name == "Lt":
            expect(args, 2, "Math.Lt")
            return Lt(parse_math(args[0]), parse_math(args[1]))
        if method_name == "Le":
            expect(args, 2, "Math.Le")
            return Le(parse_math(args[0]), parse_math(args[1]))
        if method_name == "Gt":
            expect(args, 2, "Math.Gt")
            return Gt(parse_math(args[0]), parse_math(args[1]))
        if method_name == "Ge":
            expect(args, 2, "Math.Ge")
            return Ge(parse_math(args[0]), parse_math(args[1]))
        raise TypeError(f"Math.{method_name} is not a boolean expression")

    class_name, method_name, _ = op(root)
    if class_name == "MathBool":
        return parse_bool(root)
    if class_name == "Math":
        if method_name in _MATH_BOOL_METHODS:
            return parse_bool(root)
        return parse_math(root)
    raise TypeError(f"Unsupported root expression: {class_name}.{method_name}")
