from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Iterable, NamedTuple, Sequence

from ..spec.spec_ast import *

RivalIR = dict[str, Any]

__all__ = [
    "RivalMachine", "RivalIntegerMachine", "RivalAnalysis", "RivalEvaluation",
    "build_machine", "build_integer_evaluator", "collect_free_vars", "to_rival_ir",
]


@dataclass(frozen=True)
class RivalAnalysis:
    status: tuple[bool, bool]
    hints: Any


class RivalMachine:
    def __init__(self, raw_machine: Any):
        self._raw_machine = raw_machine

    def apply_with_hints(
        self,
        rect: Sequence[tuple[float, float]],
        hints: Any | None = None,
    ) -> RivalAnalysis:
        status, next_hints = self._raw_machine.apply_with_hints(rect, hints)
        return RivalAnalysis(
            status=(bool(status[0]), bool(status[1])),
            hints=next_hints,
        )

    def apply_range(
        self,
        rect: Sequence[tuple[float, float]],
    ) -> tuple[float, float]:
        bounds = self._raw_machine.apply_range(rect)
        return float(bounds[0]), float(bounds[1])

    def domain_errors(
        self,
        rect: Sequence[tuple[float, float]],
    ) -> list[tuple[bool, bool]]:
        return [
            (bool(status[0]), bool(status[1]))
            for status in self._raw_machine.domain_errors(rect)
        ]


class RivalEvaluation(NamedTuple):
    """Point values or interval bounds; error-flagged values are not valid."""

    precondition: bool | tuple[bool, bool] | None = None
    output: int | tuple[Fraction, Fraction] | None = None
    precondition_error: bool = False
    output_error: bool = False


class RivalIntegerMachine:
    """Evaluate a Boolean precondition and an integer output code."""

    def __init__(self, raw_machine: Any):
        self._raw_machine = raw_machine

    def apply(self, raw_input: int) -> RivalEvaluation:
        precondition, output, precondition_error, output_error = self._raw_machine.apply(str(raw_input))
        return RivalEvaluation(
            bool(precondition), None if output is None else int(output),
            bool(precondition_error), bool(output_error),
        )


def build_machine(
    exprs: Sequence[SpecNode | RivalIR],
    free_vars: Sequence[str],
) -> RivalMachine:
    """Compile the supplied outputs; query construction belongs to callers."""
    expr_list = list(exprs)
    var_list = _validate_free_vars(free_vars)
    _validate_referenced_vars(expr_list, var_list)
    native = _load_native_module()
    translated_exprs = [to_rival_ir(expr) if isinstance(expr, SpecNode) else expr for expr in expr_list]
    raw_machine = native.build_machine(translated_exprs, var_list)
    return RivalMachine(raw_machine)


def _evaluator_precondition(precondition: BoolExpr) -> RivalIR:
    """Let an error-free false conjunct reject an input despite other errors."""
    pending = [precondition]
    predicates = []
    while pending:
        predicate = pending.pop()
        if isinstance(predicate, And):
            pending.extend((predicate.rhs, predicate.lhs))
        else:
            predicates.append(to_rival_ir(predicate))

    rejected = to_rival_ir(BoolLit(False))
    for predicate in predicates:
        rejected = {
            "op": "or", "lhs": rejected,
            "rhs": {
                "op": "if", "cond": {"op": "error", "arg": predicate},
                "on_true": to_rival_ir(BoolLit(False)),
                "on_false": {"op": "not", "arg": predicate},
            },
        }
    return {
        "op": "if", "cond": rejected,
        "on_true": to_rival_ir(BoolLit(False)),
        "on_false": _and_exprs(predicates),
    }


def build_integer_evaluator(
    expr: RealExpr,
    precondition: BoolExpr | None = None,
    *,
    input_total_bits: int,
    output_total_bits: int,
) -> RivalIntegerMachine:
    if precondition is None:
        precondition = BoolLit(True)
    if not isinstance(precondition, BoolExpr):
        raise TypeError("evaluation precondition must be a Boolean expression")
    _validate_referenced_vars([precondition], ["x"])
    if not isinstance(expr, RealExpr):
        raise TypeError("integer evaluation requires a real expression")
    _validate_referenced_vars([expr], ["x"])
    for label, total in (("input", input_total_bits), ("output", output_total_bits)):
        if type(total) is not int:
            raise TypeError(f"{label} bit width must be an integer")
        if total < 1:
            raise ValueError(f"invalid {label} integer bit width")

    rounding_mode = None
    if isinstance(expr, Round):
        if expr.mode is RoundingMode.FR:
            raise NotImplementedError("Rival lowering of faithful rounding is not implemented")
        rounding_mode = expr.mode.value
        expr = expr.value * (RealLit(2) ** RealLit(expr.frac_bits))

    native = _load_native_module()
    return RivalIntegerMachine(
        native.build_integer_evaluator(
            to_rival_ir(expr),
            input_total_bits,
            output_total_bits,
            rounding_mode,
            _evaluator_precondition(precondition),
        )
    )


def collect_free_vars(exprs: Iterable[SpecNode | RivalIR]) -> list[str]:
    found: set[str] = set()

    def visit(node: SpecNode | RivalIR) -> None:
        if isinstance(node, dict):
            if node.get("op") == "var":
                found.add(node["name"])
            for child in node.values():
                if isinstance(child, dict):
                    visit(child)
            return
        if isinstance(node, (RealVar, BoolVar)):
            found.add(node.name)
        for child in children(node):
            visit(child)

    for expr in exprs:
        if not isinstance(expr, (SpecNode, dict)):
            raise TypeError(f"Expected SpecNode or Rival IR, got {type(expr).__name__}")
        visit(expr)
    return sorted(found)


_RIVAL_UNARY_OPS = {
    Pow2: "pow2",
    Sqrt: "sqrt",
    Cbrt: "cbrt",
    Exp: "exp",
    Exp2: "exp2",
    Expm1: "expm1",
    Log: "log",
    Log2: "log2",
    Log10: "log10",
    Log1p: "log1p",
    Logb: "logb",
    Sin: "sin",
    Cos: "cos",
    Tan: "tan",
    Asin: "asin",
    Acos: "acos",
    Atan: "atan",
    Sinh: "sinh",
    Cosh: "cosh",
    Tanh: "tanh",
    Asinh: "asinh",
    Acosh: "acosh",
    Atanh: "atanh",
    Erf: "erf",
    Erfc: "erfc",
}


def to_rival_ir(node: SpecNode) -> RivalIR:
    unary_op = _RIVAL_UNARY_OPS.get(type(node))
    if unary_op is not None:
        return _unary(unary_op, node.value)
    if isinstance(node, Round):
        if node.mode is RoundingMode.FR:
            raise NotImplementedError("Rival lowering of faithful rounding is not implemented")
        scale = RealLit(2) ** RealLit(node.frac_bits)
        scaled = to_rival_ir(node.value * scale)
        if node.mode is RoundingMode.RNE:
            rounded = {"op": "round_even", "arg": scaled}
        elif node.mode is RoundingMode.RTZ:
            rounded = {"op": "trunc", "arg": scaled}
        else:
            rounded = {
                "op": "if",
                "cond": to_rival_ir(node.value >= RealLit(0)),
                "on_true": {"op": "ceil", "arg": scaled},
                "on_false": {"op": "neg", "arg": {
                    "op": "ceil", "arg": {"op": "neg", "arg": scaled},
                }},
            }
        return {"op": "mul", "lhs": rounded,
                "rhs": to_rival_ir(RealLit(2) ** RealLit(-node.frac_bits))}
    if isinstance(node, (RealVar, BoolVar)):
        return {"op": "var", "name": node.name}
    if isinstance(node, RealLit):
        ratio = _real_literal_ratio(node.value)
        return {
            "op": "real_lit",
            "num": str(ratio.numerator),
            "den": str(ratio.denominator),
        }
    if isinstance(node, BoolLit):
        return {"op": "bool_lit", "value": bool(node.value)}
    
    if isinstance(node, Add):
        return _binary("add", node.lhs, node.rhs)
    if isinstance(node, Sub):
        return _binary("sub", node.lhs, node.rhs)
    if isinstance(node, Mul):
        return _binary("mul", node.lhs, node.rhs)
    if isinstance(node, Neg):
        return _unary("neg", node.value)
    if isinstance(node, Abs):
        return _unary("abs", node.value)
    if isinstance(node, Pow):
        return _binary("pow", node.base, node.exponent)
    if isinstance(node, Max):
        return _binary("max", node.lhs, node.rhs)
    if isinstance(node, Min):
        return _binary("min", node.lhs, node.rhs)
    if isinstance(node, If):
        return {
            "op": "if",
            "cond": to_rival_ir(node.cond),
            "on_true": to_rival_ir(node.on_true),
            "on_false": to_rival_ir(node.on_false),
        }
    
    if isinstance(node, Eq):
        return _binary("eq", node.lhs, node.rhs)
    if isinstance(node, NotEq):
        return _binary("ne", node.lhs, node.rhs)
    if isinstance(node, Lt):
        return _binary("lt", node.lhs, node.rhs)
    if isinstance(node, Le):
        return _binary("le", node.lhs, node.rhs)
    if isinstance(node, Gt):
        return _binary("gt", node.lhs, node.rhs)
    if isinstance(node, Ge):
        return _binary("ge", node.lhs, node.rhs)
    if isinstance(node, BoolEq):
        return _binary("bool_eq", node.lhs, node.rhs)
    if isinstance(node, BoolNe):
        # Rival already supports Boolean equality and negation; retain the
        # compact AST node without requiring a new backend IR operation.
        return _unary("not", BoolEq(node.lhs, node.rhs))
    if isinstance(node, Not):
        return _unary("not", node.value)
    if isinstance(node, Or):
        return _binary("or", node.lhs, node.rhs)
    if isinstance(node, And):
        return _binary("and", node.lhs, node.rhs)
    
    raise TypeError(f"Unsupported SpecNode for Rival translation: {type(node).__name__}")


def _binary(op: str, lhs: SpecNode, rhs: SpecNode) -> RivalIR:
    return {"op": op, "lhs": to_rival_ir(lhs), "rhs": to_rival_ir(rhs)}


def _unary(op: str, arg: SpecNode) -> RivalIR:
    return {"op": op, "arg": to_rival_ir(arg)}


def _and_exprs(exprs: Iterable[RivalIR]) -> RivalIR:
    expr_iter = iter(exprs)
    result = next(expr_iter)
    for expr in expr_iter:
        result = {"op": "and", "lhs": result, "rhs": expr}
    return result


def _real_literal_ratio(value: int | float) -> Fraction:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Rival translation does not support non-finite RealLit values")
    return Fraction(value)


def _validate_free_vars(free_vars: Sequence[str]) -> list[str]:
    result = list(free_vars)
    if any(not isinstance(var, str) for var in result):
        raise TypeError("free_vars must contain only strings")
    if len(set(result)) != len(result):
        raise ValueError("free_vars must be unique")
    return result


def _validate_referenced_vars(exprs: Sequence[SpecNode | RivalIR], free_vars: Sequence[str]) -> None:
    if not exprs:
        raise ValueError("exprs must be non-empty")
    missing = sorted(set(collect_free_vars(exprs)) - set(free_vars))
    if missing:
        raise ValueError(f"free_vars is missing referenced variables: {missing}")


def _load_native_module():
    try:
        from .. import _rival3
    except ImportError:
        try:
            import _rival3  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                "Rival3 native bridge is not built. Build it with "
                "`maturin develop -m crates/rival_bridge/Cargo.toml` or an equivalent "
                "PyO3 build command."
            ) from exc
    return _rival3
