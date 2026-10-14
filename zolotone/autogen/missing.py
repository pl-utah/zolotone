"""Create finite-domain runtime primitives for missing real operations."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import inspect
import itertools
import math
import re

import z3

from ..errors import MissingError, ZolotoneError
from ..rival import build_integer_machine, rival_range_analysis
from ..spec.spec_ast import If, RealExpr, RealLit, RealVar, substitute_spec_node
from ..spec.spec_context import SpecContext
from ..types import Q, UQ
from ..ast.helpers import integer_lookup, integer_lookup_multi
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
    inputs: tuple[RealExpr, ...],
    codes: tuple[int, ...],
    dtypes: tuple[Q | UQ, ...],
) -> Fraction | None:
    concrete: RealExpr = expression
    for operand, code, dtype in zip(inputs, codes, dtypes, strict=True):
        concrete = substitute_spec_node(
            concrete, operand, RealLit(Fraction(code, 1 << dtype.frac_bits))
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


def _create_missing_primitive(
    missing: RealExpr,
    input_expr: RealExpr | tuple[RealExpr, ...],
    input_node: Node | tuple[Node, ...],
    range_ctx: SpecContext,
    *,
    output_dtype: Q | UQ | None = None,
) -> Node:
    """Generate a typed LUT for a missing expression and its materialized inputs."""
    input_exprs = (input_expr,) if isinstance(input_expr, RealExpr) else input_expr
    input_nodes = (input_node,) if isinstance(input_node, Node) else input_node
    if not input_exprs or len(input_exprs) != len(input_nodes):
        raise MissingError("A missing component needs matching typed inputs")
    if any(not isinstance(node.dtype, (Q, UQ)) for node in input_nodes):
        raise MissingError("Missing component inputs require Q or UQ descriptors")
    dtypes = tuple(node.dtype for node in input_nodes)
    if any(dtype.total_bits() > 63 for dtype in dtypes):
        raise MissingError("Generated LUT inputs must have at most 63 bits")

    generation_ctx = range_ctx.copy()
    for expression, dtype in zip(input_exprs, dtypes, strict=True):
        generation_ctx.assumes.append(
            _fixed_point_result_fits(expression, dtype, generation_ctx)
        )
    if output_dtype is None:
        output_dtype = _suggest_missing_output_dtype(
            missing, dtypes[0], generation_ctx
        )
    if output_dtype.total_bits() > 64:
        raise MissingError("Generated LUT outputs must have at most 64 bits")

    bounds = tuple(
        _code_bounds(expression, dtype, generation_ctx)
        for expression, dtype in zip(input_exprs, dtypes, strict=True)
    )
    extents = tuple(upper - lower + 1 for lower, upper in bounds)
    table_size = math.prod(extents)
    if table_size > MAX_GENERATED_TABLE_ENTRIES:
        raise ZolotoneError(
            f"Generated component for {missing!r} has {table_size} input "
            f"encodings; the limit is {MAX_GENERATED_TABLE_ENTRIES}"
        )

    local_vars = tuple(RealVar(f"x{i}") for i in range(len(input_exprs)))
    local_expression: RealExpr = missing
    for original, replacement in zip(input_exprs, local_vars, strict=True):
        local_expression = substitute_spec_node(
            local_expression, original, replacement
        )
    pattern = re.sub(r"[^a-z0-9]+", "_", type(missing).__name__.lower())
    dtype_part = "_".join(
        f"{type(dtype).__name__.lower()}{dtype.int_bits}_{dtype.frac_bits}"
        for dtype in (*dtypes, output_dtype)
    )
    digest = hashlib.sha256(repr(local_expression).encode()).hexdigest()[:10]
    component_name = f"generated_{pattern}_{dtype_part}_{digest}"

    evaluator = None
    if len(input_exprs) == 1:
        raw = RealVar("x")
        integer_expression = substitute_spec_node(
            local_expression,
            local_vars[0],
            raw * (RealLit(2) ** RealLit(-dtypes[0].frac_bits)),
        ) * (RealLit(2) ** RealLit(output_dtype.frac_bits))
        evaluator = build_integer_machine(
            integer_expression,
            input_total_bits=dtypes[0].total_bits(),
            output_total_bits=output_dtype.total_bits(),
        )

    solver = z3.Solver()
    env: dict[object, object] = {}
    solver.add(*(assume.to_z3(env) for assume in generation_ctx.assumes))
    input_terms = tuple(expression.to_z3(env) for expression in input_exprs)
    mask = (1 << output_dtype.total_bits()) - 1
    entries: list[int] = []
    reachable_entries: list[tuple[tuple[int, ...], int]] = []
    exact = True
    try:
        for codes in itertools.product(
            *(range(lower, upper + 1) for lower, upper in bounds)
        ):
            solver.push()
            solver.add(*(
                term == z3.RealVal(f"{code}/{1 << dtype.frac_bits}")
                for term, code, dtype in zip(input_terms, codes, dtypes, strict=True)
            ))
            reachable = solver.check() != z3.unsat
            solver.pop()
            if not reachable:
                entries.append(0)
                continue
            value = _evaluate_real_expression(
                local_expression, local_vars, codes, dtypes
            )
            if value is None:
                if evaluator is None:
                    raise ZolotoneError(
                        f"Could not evaluate generated component {component_name!r}"
                    )
                try:
                    scaled = evaluator.apply(codes[0])
                except ValueError as exc:
                    raise ZolotoneError(
                        f"Could not prove a finite output range for {missing!r}"
                    ) from exc
                exact = False
            else:
                scaled = round(value * (1 << output_dtype.frac_bits))
            raw_value = scaled & mask
            entries.append(raw_value)
            reachable_entries.append((codes, raw_value))
            if value is not None and _decoded_output(raw_value, output_dtype) != value:
                exact = False
    except ValueError as exc:
        raise ZolotoneError(
            f"Could not evaluate generated component {component_name!r}"
        ) from exc
    table = tuple(entries)

    def generated_spec(*args):
        *values, ctx = args
        for value, (lower, upper), dtype in zip(
            values, bounds, dtypes, strict=True
        ):
            scale = 1 << dtype.frac_bits
            ctx.assume(
                (value >= ctx.real_val(Fraction(lower, scale)))
                & (value <= ctx.real_val(Fraction(upper, scale)))
            )
        if exact:
            result = missing
            for original, value in zip(input_exprs, values, strict=True):
                result = substitute_spec_node(result, original, value)
            return result
        result: RealExpr = RealLit(0)
        for codes, raw_value in reversed(reachable_entries):
            condition = None
            for value, code, dtype in zip(values, codes, dtypes, strict=True):
                term = value.eq(RealLit(Fraction(code, 1 << dtype.frac_bits)))
                condition = term if condition is None else condition & term
            result = If(
                condition,
                RealLit(_decoded_output(raw_value, output_dtype)),
                result,
            )
        return result

    argument_names = (
        ("x",) if len(dtypes) == 1
        else tuple(f"x{i}" for i in range(len(dtypes)))
    )
    parameters = [
        inspect.Parameter(
            name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=dtype,
        )
        for name, dtype in zip(argument_names, dtypes, strict=True)
    ]
    parameters.append(
        inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    generated_spec.__signature__ = inspect.Signature(
        parameters=parameters, return_annotation=output_dtype
    )
    generated_spec.__annotations__ = {
        **dict(zip(argument_names, dtypes, strict=True)),
        "return": output_dtype,
    }
    generated_spec.__name__ = f"{component_name}_spec"
    generated_spec.__qualname__ = generated_spec.__name__

    @Primitive(name=component_name, spec=generated_spec)
    def generated_impl(*nodes: Node) -> Node:
        if len(nodes) == 1:
            return integer_lookup(
                nodes[0], table=table, lowest_code=bounds[0][0], out=output_dtype
            )
        return integer_lookup_multi(
            tuple(nodes),
            table=table,
            lowest_codes=tuple(lower for lower, _ in bounds),
            extents=extents,
            out=output_dtype,
        )

    node = generated_impl(*input_nodes)
    node._autogen_lut_exact = exact
    return node
