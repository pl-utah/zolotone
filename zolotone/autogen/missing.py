"""Create runtime primitives for real specification expressions without components."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math
import re

import z3

from ..errors import MissingError, ZolotoneError
from ..rival import (
    build_integer_machine,
    rival_feasibility_check,
    rival_range_analysis,
)
from ..spec.spec_ast import RealExpr, RealLit, RealVar, substitute_spec_node
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


def _code_is_reachable(
    generation_ctx: SpecContext,
    input_expr: RealExpr,
    code: int,
    scale: int,
    solver: z3.Solver,
    input_term: z3.ExprRef,
) -> bool:
    code_value = Fraction(code, scale)
    feasibility_ctx = generation_ctx.copy()
    feasibility_ctx.assumes.append(input_expr.eq(RealLit(code_value)))
    status = rival_feasibility_check(feasibility_ctx)
    if status == "not feasible":
        return False
    if status == "feasible":
        return True
    if status == "unknown":
        return solver.check(
            input_term == z3.RealVal(f"{code}/{scale}")
        ) != z3.unsat
    raise ValueError(f"Unexpected feasibility status: {status!r}")


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
    # The range may contain holes (for example, x == -1 or x == 1). Use Rival
    # first and Z3 only when interval analysis cannot decide reachability.
    env = {}
    reachable = z3.Solver()
    reachable.add(*(assume.to_z3(env) for assume in generation_ctx.assumes))
    input_term = input_expr.to_z3(env)
    output_mask = (1 << output_dtype.total_bits()) - 1
    try:
        entries = []
        for code in range(lower_code, upper_code + 1):
            code_is_possible = _code_is_reachable(
                generation_ctx,
                input_expr,
                code,
                scale,
                reachable,
                input_term,
            )
            entries.append(
                evaluator.apply(code) & output_mask if code_is_possible else 0
            )
        table = tuple(entries)
    except ValueError as exc:
        # Do this before constructing the Primitive: callers must never see a
        # partly populated generated component.
        raise ZolotoneError(
            f"Could not evaluate generated component {component_name!r}"
        ) from exc

    def generated_spec(x: input_dtype, ctx) -> output_dtype:
        ctx.assume((x >= ctx.real_val(lower)) & (x <= ctx.real_val(upper)))
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
