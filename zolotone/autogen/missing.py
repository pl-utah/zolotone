"""Create finite-domain runtime primitives for missing real operations."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math
import re

import z3

from ..errors import MissingError, ZolotoneError
from ..rival import build_integer_machine, rival_range_analysis
from ..spec.spec_ast import If, RealExpr, RealLit, RealVar, substitute_spec_node
from ..spec.spec_context import SpecContext
from ..types import Q, UQ
from ..ast.helpers import integer_lookup
from ..ast.node import Node
from ..ast.nodes import Primitive
from .spec_validation import (
    _fixed_point_result_fits,
    _output_format_suggestion_with_range_analysis,
    _output_format_suggestion_with_search,
)


MAX_GENERATED_TABLE_ENTRIES = 4_096


def _suggest_missing_output_dtype(
    expression: RealExpr, input_dtype: Q | UQ, range_ctx: SpecContext
) -> Q | UQ:
    conservative = _output_format_suggestion_with_range_analysis(
        expression, input_dtype, range_ctx
    )
    output = _output_format_suggestion_with_search(
        expression, conservative, range_ctx
    )
    if output is None:
        raise ZolotoneError(
            f"Could not prove a fixed-point output format for {expression!r}"
        )
    return output


def _code_bounds(
    expression: RealExpr, dtype: Q | UQ, ctx: SpecContext
) -> tuple[int, int]:
    value_range = rival_range_analysis(expression, ctx)
    if value_range is None or not all(math.isfinite(bound) for bound in value_range):
        raise ZolotoneError(
            f"Could not obtain a finite input range for component input {expression!r}"
        )
    scale = 1 << dtype.frac_bits
    lower = math.ceil(Fraction(value_range[0]) * scale)
    upper = math.floor(Fraction(value_range[1]) * scale)
    if isinstance(dtype, Q):
        minimum = -(1 << (dtype.total_bits() - 1))
        maximum = (1 << (dtype.total_bits() - 1)) - 1
    else:
        minimum = 0
        maximum = (1 << dtype.total_bits()) - 1
    lower = max(lower, minimum)
    upper = min(upper, maximum)
    if lower > upper:
        raise ZolotoneError(
            f"Generated component input {expression!r} has no valid encodings"
        )
    return lower, upper


def _evaluate_real_expression(
    expression: RealExpr,
    input_expr: RealExpr,
    code: int,
    dtype: Q | UQ,
) -> Fraction | None:
    concrete = substitute_spec_node(
        expression, input_expr, RealLit(Fraction(code, 1 << dtype.frac_bits))
    )
    try:
        folded = concrete.constant_fold()
    except (ArithmeticError, OverflowError, ValueError, TypeError):
        return None
    if isinstance(folded, RealLit):
        try:
            return Fraction(folded.value)
        except (OverflowError, ValueError):
            return None
    return None


def _decoded_output(raw: int, dtype: Q | UQ) -> Fraction:
    code = dtype._scaled_integer(raw) if isinstance(dtype, Q) else raw
    return Fraction(code, 1 << dtype.frac_bits)


def _build_missing_primitive(
    missing: RealExpr,
    input_expr: RealExpr,
    input_node: Node,
    range_ctx: SpecContext,
    output_dtype: Q | UQ | None = None,
) -> Node:
    """Generate a univariate typed LUT from one materialized input."""
    dtype = input_node.dtype
    if not isinstance(dtype, (Q, UQ)):
        raise MissingError("Missing component input requires a Q or UQ descriptor")
    if dtype.total_bits() > 63:
        raise MissingError("Generated LUT inputs must have at most 63 bits")

    generation_ctx = range_ctx.copy()
    generation_ctx.assumes.append(
        _fixed_point_result_fits(input_expr, dtype, generation_ctx)
    )
    if output_dtype is None:
        output_dtype = _suggest_missing_output_dtype(
            missing, dtype, generation_ctx
        )
    if output_dtype.total_bits() > 64:
        raise MissingError("Generated LUT outputs must have at most 64 bits")

    lower, upper = _code_bounds(input_expr, dtype, generation_ctx)
    table_size = upper - lower + 1
    if table_size > MAX_GENERATED_TABLE_ENTRIES:
        raise ZolotoneError(
            f"Generated component for {missing!r} has {table_size} input "
            f"encodings; the limit is {MAX_GENERATED_TABLE_ENTRIES}"
        )

    local_var = RealVar("x0")
    local_expression = substitute_spec_node(missing, input_expr, local_var)
    pattern = re.sub(r"[^a-z0-9]+", "_", type(missing).__name__.lower())
    dtype_part = "_".join(
        f"{type(format_).__name__.lower()}{format_.int_bits}_{format_.frac_bits}"
        for format_ in (dtype, output_dtype)
    )
    digest = hashlib.sha256(repr(local_expression).encode()).hexdigest()[:10]
    component_name = f"generated_{pattern}_{dtype_part}_{digest}"

    raw = RealVar("x")
    integer_expression = substitute_spec_node(
        local_expression,
        local_var,
        raw * (RealLit(2) ** RealLit(-dtype.frac_bits)),
    ) * (RealLit(2) ** RealLit(output_dtype.frac_bits))
    evaluator = build_integer_machine(
        integer_expression,
        input_total_bits=dtype.total_bits(),
        output_total_bits=output_dtype.total_bits(),
    )

    solver = z3.Solver()
    env: dict[object, object] = {}
    solver.add(*(assume.to_z3(env) for assume in generation_ctx.assumes))
    input_term = input_expr.to_z3(env)
    mask = (1 << output_dtype.total_bits()) - 1
    entries: list[int] = []
    reachable_entries: list[tuple[int, int]] = []
    exact = True
    try:
        for code in range(lower, upper + 1):
            solver.push()
            solver.add(input_term == z3.RealVal(f"{code}/{1 << dtype.frac_bits}"))
            reachable = solver.check() != z3.unsat
            solver.pop()
            if not reachable:
                entries.append(0)
                continue
            value = _evaluate_real_expression(
                local_expression, local_var, code, dtype
            )
            if value is None:
                try:
                    scaled = evaluator.apply(code)
                except ValueError as exc:
                    raise ZolotoneError(
                        f"Could not prove a finite output range for {missing!r}"
                    ) from exc
                exact = False
            else:
                scaled = round(value * (1 << output_dtype.frac_bits))
            raw_value = scaled & mask
            entries.append(raw_value)
            reachable_entries.append((code, raw_value))
            if value is not None and _decoded_output(raw_value, output_dtype) != value:
                exact = False
    except ValueError as exc:
        raise ZolotoneError(
            f"Could not evaluate generated component {component_name!r}"
        ) from exc
    table = tuple(entries)

    def generated_spec(x, ctx):
        scale = 1 << dtype.frac_bits
        ctx.assume(
            (x >= ctx.real_val(Fraction(lower, scale)))
            & (x <= ctx.real_val(Fraction(upper, scale)))
        )
        if exact:
            return substitute_spec_node(missing, input_expr, x)
        result: RealExpr = RealLit(0)
        for code, raw_value in reversed(reachable_entries):
            result = If(
                x.eq(RealLit(Fraction(code, scale))),
                RealLit(_decoded_output(raw_value, output_dtype)),
                result,
            )
        return result

    generated_spec.__annotations__ = {
        "x": dtype,
        "return": output_dtype,
    }
    generated_spec.__name__ = f"{component_name}_spec"
    generated_spec.__qualname__ = generated_spec.__name__

    @Primitive(name=component_name, spec=generated_spec)
    def generated_impl(node: Node) -> Node:
        return integer_lookup(
            node, table=table, lowest_code=lower, out=output_dtype
        )

    return generated_impl(input_node)
