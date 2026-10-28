"""Create finite-domain runtime primitives for missing real operations."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math

from ..errors import InfeasibleError, MissingError, ZolotoneError
from ..rival import (
    build_integer_evaluator,
    collect_free_vars,
)
from ..rival.analysis import analyze_typed_domain
from ..spec.spec_ast import RealExpr, RealLit, RealVar, Round, SpecNode, substitute_spec_node
from ..spec.spec_context import SpecContext
from ..types import DataType, Q, UQ
from ..ast.helpers import integer_lookup
from ..ast.node import Node
from ..ast.nodes import Primitive
from .spec_validation import (
    _condition_conjuncts,
    _fixed_point_result_fits,
    _output_format_suggestion_with_range_analysis,
    _output_format_suggestion_with_search,
)


MAX_GENERATED_TABLE_ENTRIES = 1 << 20


def _suggest_missing_output_dtype(
    expression: RealExpr, precision_dtype: Q | UQ, range_ctx: SpecContext,
    input_dtypes: dict[SpecNode, DataType],
) -> Q | UQ:
    """Keep the supplied fractional width and infer sign/integer bits from range."""
    conservative = _output_format_suggestion_with_range_analysis(
        expression, precision_dtype, range_ctx, input_dtypes=input_dtypes,
    )
    output = _output_format_suggestion_with_search(
        expression, conservative, range_ctx, input_dtypes
    )
    if output is None:
        raise ZolotoneError(
            f"Could not prove a fixed-point output format for {expression!r}"
        )
    return output


def _expression_bounds(
    expression: RealExpr, dtype: Q | UQ, ctx: SpecContext,
    input_dtypes: dict[SpecNode, DataType],
) -> tuple[int, int]:
    value_range = analyze_typed_domain(expression, ctx, input_dtypes, dtype)
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


def _build_missing_primitive(
    missing: RealExpr,
    input_expr: RealExpr,
    input_node: Node,
    range_ctx: SpecContext,
    output_dtype: Q | UQ,
    *,
    input_dtypes: dict[SpecNode, DataType],
) -> Node:
    """Generate a univariate typed LUT from one materialized input."""
    if not isinstance(missing, Round):
        raise MissingError(
            "LUT synthesis requires explicit ctx.rne, ctx.rtz, or ctx.raz with frac_bits"
        )
    dtype = input_node.dtype
    if not isinstance(dtype, (Q, UQ)):
        raise MissingError("Missing component input requires a Q or UQ descriptor")
    if not isinstance(output_dtype, (Q, UQ)):
        raise MissingError("Missing component output requires a Q or UQ descriptor")
    if dtype.total_bits() > 63:
        raise MissingError("Generated LUT inputs must have at most 63 bits")

    generation_ctx = range_ctx.copy()
    generation_ctx.assumes.append(
        _fixed_point_result_fits(input_expr, dtype, generation_ctx)
    )
    if output_dtype.total_bits() > 64:
        raise MissingError("Generated LUT outputs must have at most 64 bits")

    lower, upper = _expression_bounds(input_expr, dtype, generation_ctx, input_dtypes)
    table_size = 1 << dtype.total_bits()
    if table_size > MAX_GENERATED_TABLE_ENTRIES:
        raise ZolotoneError(
            f"Generated component for {missing!r} has a {table_size}-entry "
            f"input dtype; the limit is {MAX_GENERATED_TABLE_ENTRIES}"
        )

    raw = RealVar("x")
    integer_expression = substitute_spec_node(
        missing,
        input_expr,
        raw * (RealLit(2) ** RealLit(-dtype.frac_bits)),
    )
    pattern = f"{missing.mode.value.lower()}_f{missing.frac_bits}"
    dtype_part = "_".join(
        f"{type(format_).__name__.lower()}{format_.int_bits}_{format_.frac_bits}"
        for format_ in (dtype, output_dtype)
    )
    digest = hashlib.sha256(repr(integer_expression).encode()).hexdigest()[:10]
    component_name = f"LUT_{pattern}_{dtype_part}_{digest}"

    # Use a fresh placeholder so an original variable named x cannot be
    # mistaken for the materialized LUT input after substitution.
    original_names = set(collect_free_vars(generation_ctx.assumes))
    placeholder_name = "__lut_input"
    while placeholder_name in original_names:
        placeholder_name += "_"
    placeholder = RealVar(placeholder_name)
    normalized_input = placeholder * (RealLit(2) ** RealLit(-dtype.frac_bits))
    precondition = generation_ctx.true()
    for assume in (
        conjunct
        for condition in generation_ctx.assumes
        for conjunct in _condition_conjuncts(condition)
    ):
        substituted = substitute_spec_node(assume, input_expr, normalized_input)
        if set(collect_free_vars([substituted])) <= {placeholder_name}:
            precondition = precondition & substitute_spec_node(substituted, placeholder, raw)
        # Assumptions involving other variables cannot be decided at this
        # input alone, so conservatively leave those points reachable.
    evaluator = build_integer_evaluator(
        integer_expression,
        precondition,
        input_total_bits=dtype.total_bits(),
        output_total_bits=output_dtype.total_bits(),
    )

    output_bits = output_dtype.total_bits()
    mask = (1 << output_bits) - 1
    if isinstance(output_dtype, Q):
        output_minimum = -(1 << (output_bits - 1))
        output_maximum = (1 << (output_bits - 1)) - 1
    else:
        output_minimum = 0
        output_maximum = mask

    entries: list[int] = []
    unreachable_points: list[int] = []
    try:
        for point in range(lower, upper + 1):
            result = evaluator.apply_point([point])
            if result.precondition_error:
                raise ValueError("domain error in assumptions")
            if not result.precondition:
                entries.append(0)
                unreachable_points.append(point)
                continue
            if result.output_error:
                raise ValueError("domain error in output at reachable input")
            if result.output is None:
                raise ValueError("output at reachable input evaluated to inf/nan")
            scaled = result.output
            if not output_minimum <= scaled <= output_maximum:
                raise ZolotoneError(
                    f"Generated LUT output {scaled} at input point {point} "
                    f"does not fit {output_dtype!r} "
                    f"(allowed range [{output_minimum},{output_maximum}])"
                )
            # Converts signed int into two's complement
            raw_value = scaled & mask

            entries.append(raw_value)
    except ValueError as exc:
        raise ZolotoneError(
            f"Could not evaluate generated component {component_name!r} "
            f"at input code {point}: {exc}"
        ) from exc
    if len(unreachable_points) == len(entries):
        raise InfeasibleError(
            f"Generated component {component_name!r} has no reachable input codes"
        )
    dtype_minimum = (
        -(1 << (dtype.total_bits() - 1)) if isinstance(dtype, Q) else 0
    )
    full_table = [0] * table_size
    first_entry = lower - dtype_minimum
    full_table[first_entry : first_entry + len(entries)] = entries
    table = tuple(full_table)

    def generated_spec(x, ctx):
        """Describe the explicitly rounded function implemented by the table."""
        scale = 1 << dtype.frac_bits
        ctx.assume(
            (x >= ctx.real_val(Fraction(lower, scale)))
            & (x <= ctx.real_val(Fraction(upper, scale)))
        )
        for point in unreachable_points:
            ctx.assume(x.ne(ctx.real_val(Fraction(point, scale))))
        return substitute_spec_node(missing, input_expr, x)

    generated_spec.__annotations__ = {
        "x": dtype,
        "return": output_dtype,
    }
    generated_spec.__name__ = f"{component_name}_spec"
    generated_spec.__qualname__ = generated_spec.__name__

    @Primitive(name=component_name, spec=generated_spec)
    def generated_impl(node: Node) -> Node:
        return integer_lookup(
            node, table=table, out=output_dtype
        )

    return generated_impl(input_node)
