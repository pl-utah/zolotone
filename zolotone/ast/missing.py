"""Create runtime primitives for real specification expressions without components."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math
import re

from ..errors import MissingError, ZolotoneError
from ..rival import build_integer_machine, rival_range_analysis
from ..spec.spec_ast import RealExpr, RealLit, RealVar, substitute_spec_node
from ..spec.spec_context import SpecContext
from ..types import Q, UQ
from .helpers import integer_lookup
from .node import Node
from .nodes import Primitive
from .spec_validation import (
    _fixed_point_result_fits,
    _output_format_suggestion_with_range_analysis,
    _output_format_suggestion_with_search,
)


MAX_GENERATED_TABLE_ENTRIES = 4_096


def _create_missing_primitive(
    missing: RealExpr,
    input_expr: RealExpr,
    input_node: Node,
    range_ctx: SpecContext,
) -> Node:
    """Prove the format and create a runtime Primitive for a typed frontier."""
    input_dtype = input_node.dtype
    if not isinstance(input_dtype, (Q, UQ)):
        raise MissingError(
            f"Missing component input {input_expr!r} has no Q or UQ descriptor"
        )

    # The selected typed operand has valid descriptor bounds.
    generation_ctx = range_ctx.copy()
    generation_ctx.assumes.append(
        _fixed_point_result_fits(input_expr, input_dtype, generation_ctx)
    )

    input_range = rival_range_analysis(input_expr, generation_ctx)
    if input_range is None or not all(math.isfinite(bound) for bound in input_range):
        raise ZolotoneError(
            f"Could not obtain a finite input range for component input {input_expr!r}"
        )

    conservative = _output_format_suggestion_with_range_analysis(
        missing, input_dtype, generation_ctx
    )
    output_dtype = _output_format_suggestion_with_search(
        missing, conservative, generation_ctx
    )
    if output_dtype is None:
        raise ZolotoneError(
            f"Could not prove a fixed-point output format for {missing!r}"
        )

    expression = substitute_spec_node(missing, input_expr, RealVar("x"))
    pattern = re.sub(r"[^a-z0-9]+", "_", type(missing).__name__.lower())
    dtype_part = (
        f"{type(input_dtype).__name__.lower()}{input_dtype.int_bits}_{input_dtype.frac_bits}"
        f"_to_{type(output_dtype).__name__.lower()}{output_dtype.int_bits}_{output_dtype.frac_bits}"
    )
    digest = hashlib.sha256(repr(expression).encode()).hexdigest()[:10]
    component_name = f"generated_{pattern}_{dtype_part}_{digest}"
    lower, upper = input_range

    # Enumerate exactly the encodings within the inclusive requirement below.
    scale = 1 << input_dtype.frac_bits
    lower_code = math.ceil(Fraction(lower) * scale)
    upper_code = math.floor(Fraction(upper) * scale)
    if isinstance(input_dtype, Q):
        descriptor_lower = -(1 << (input_dtype.total_bits() - 1))
        descriptor_upper = (1 << (input_dtype.total_bits() - 1)) - 1
    else:
        descriptor_lower = 0
        descriptor_upper = (1 << input_dtype.total_bits()) - 1
    lower_code = max(lower_code, descriptor_lower)
    upper_code = min(upper_code, descriptor_upper)
    table_size = upper_code - lower_code + 1
    if table_size < 1:
        raise ZolotoneError(
            f"Generated component input {input_expr!r} has no valid encodings"
        )
    if table_size > MAX_GENERATED_TABLE_ENTRIES:
        raise ZolotoneError(
            f"Generated component input {input_expr!r} has {table_size} valid "
            f"encodings; the limit is {MAX_GENERATED_TABLE_ENTRIES}"
        )

    # Table entries are raw integer encodings. Keep this unit conversion local
    # so the generated primitive continues to expose its exact-real spec.
    x = RealVar("x")
    integer_expression = substitute_spec_node(
        expression, x, x * (RealLit(2) ** RealLit(-input_dtype.frac_bits))
    ) * (RealLit(2) ** RealLit(output_dtype.frac_bits))
    evaluator = build_integer_machine(
        integer_expression,
        input_total_bits=input_dtype.total_bits(),
        output_total_bits=output_dtype.total_bits(),
    )
    output_mask = (1 << output_dtype.total_bits()) - 1
    try:
        table = tuple(
            evaluator.apply(code) & output_mask
            for code in range(lower_code, upper_code + 1)
        )
    except ValueError as exc:
        # Do this before constructing the Primitive: callers must never see a
        # partly populated generated component.
        raise ZolotoneError(
            f"Could not evaluate generated component {component_name!r}"
        ) from exc

    def generated_spec(x: input_dtype, ctx) -> output_dtype:
        ctx.require((x >= ctx.real_val(lower)) & (x <= ctx.real_val(upper)))
        return substitute_spec_node(missing, input_expr, x)

    # This module postpones annotations; bind the exact descriptors now so
    # Primitive's contract resolver does not look for closure locals globally.
    generated_spec.__annotations__ = {"x": input_dtype, "return": output_dtype}
    generated_spec.__name__ = f"{component_name}_spec"
    generated_spec.__qualname__ = generated_spec.__name__

    @Primitive(name=component_name, spec=generated_spec)
    def generated_impl(x: Node) -> Node:
        return integer_lookup(
            x,
            table=table,
            lowest_code=lower_code,
            out=output_dtype,
        )

    return generated_impl(input_node)
