import math
import typing as tp
import warnings

from ..errors import InfeasibleError, MissingError, ZolotoneError
from ..rival import rival_domain_errors, rival_range_analysis
from ..solver import check_equivalence
from ..spec.spec_ast import (
    And,
    BoolEq,
    BoolExpr,
    BoolLit,
    Eq,
    Not,
    RealExpr,
    SpecNode,
    children,
    identical_nodes,
    variables,
)
from ..spec.spec_context import SpecContext, simplify_ctx
from ..types import Bool, DataType, Q, UQ
from .nodes import _SpecContract


FEASIBILITY_SCHEDULE = [
    {"tool": "simplify"},
    {"tool": "z3", "timeout_ms": 10_000},
]

REACHABILITY_FALLBACK_SCHEDULE = [
    {"tool": "z3", "timeout_ms": 10_000},
]


def reject_untyped_inputs(contract: _SpecContract):
    input_parameters = list(contract.signature.parameters.values())[:-1]
    for parameter in input_parameters:
        annotation = contract.annotations[parameter.name]
        if not isinstance(annotation, DataType):
            raise MissingError(
                f"Specification {contract.display_name} input "
                f"parameter {parameter.name!r} must have an exact DataType "
                f"descriptor, got {annotation!r}"
            )


def _spec_value_variables(value: tp.Any) -> set[tp.Any]:
    if isinstance(value, SpecNode):
        return variables(value)
    if isinstance(value, tuple):
        result = set()
        for item in value:
            result.update(_spec_value_variables(item))
        return result
    return set()


def _condition_conjuncts(condition: BoolExpr) -> tuple[BoolExpr, ...]:
    if isinstance(condition, And):
        return (
            *_condition_conjuncts(condition.lhs),
            *_condition_conjuncts(condition.rhs),
        )
    return (condition,)


def _spec_node_count(node: SpecNode) -> int:
    return 1 + sum(_spec_node_count(child) for child in children(node))


def _spec_simplification_size(spec_ast: SpecNode, ctx: SpecContext) -> int:
    return _spec_node_count(spec_ast) + sum(
        _spec_node_count(expression)
        for expressions in (ctx.assumes, ctx.checks)
        for expression in expressions
    )


def reject_undeclared_variables(
    spec_ast: SpecNode,
    spec_inputs: tuple[tp.Any, ...],
    ctx: SpecContext,
) -> None:
    input_variables = _spec_value_variables(spec_inputs)
    specification_variables = variables(spec_ast)
    for expressions in (ctx.assumes, ctx.checks, ctx.requirements):
        for expression in expressions:
            specification_variables.update(variables(expression))
    undeclared_variables = specification_variables - input_variables
    if undeclared_variables:
        rendered = ", ".join(
            sorted(str(variable) for variable in undeclared_variables)
        )
        raise MissingError(
            f"Undeclared variables in specification: {rendered}"
        )

    relevant_variables = variables(spec_ast)
    condition_variables = [
        variables(conjunct)
        for expressions in (ctx.assumes, ctx.checks, ctx.requirements)
        for expression in expressions
        for conjunct in _condition_conjuncts(expression)
    ]
    changed = True
    while changed:
        changed = False
        for connected_variables in condition_variables:
            if relevant_variables & connected_variables:
                enlarged = relevant_variables | connected_variables
                if enlarged != relevant_variables:
                    relevant_variables = enlarged
                    changed = True
    unused_variables = input_variables - relevant_variables
    if unused_variables:
        rendered = ", ".join(
            sorted(str(variable) for variable in unused_variables)
        )
        warnings.warn(
            f"Specification {ctx.name!r} has unused input variables: {rendered}",
            UserWarning,
            stacklevel=3,
        )


def _simplify_spec_ast(
    spec_ast: SpecNode,
    spec_inputs: tuple[tp.Any, ...],
    ctx: SpecContext,
) -> tuple[SpecNode, SpecContext]:
    size_before = _spec_simplification_size(spec_ast, ctx)
    # Checks are not getting simplified
    probe_ctx = ctx.copy(checks=[])
    if isinstance(spec_ast, RealExpr):
        marker = probe_ctx.fresh_real("simplified_spec_result")
    elif isinstance(spec_ast, BoolExpr):
        marker = probe_ctx.fresh_bool("simplified_spec_result")
    else:
        raise TypeError(
            "Specification simplification expects a real or Boolean "
            f"expression, got {type(spec_ast).__name__}"
        )

    probe_ctx.check(spec_ast.eq(marker))
    simplified_ctx = simplify_ctx(probe_ctx)["new_ctx"]
    simplified_checks = simplified_ctx.checks
    if len(simplified_checks) != 1:
        raise RuntimeError(
            "That's impossible, specification simplification lost its result marker"
        )

    # Finding marker in a simplified expression
    carrier = simplified_checks[0]
    if isinstance(carrier, (Eq, BoolEq)):
        if identical_nodes(carrier.rhs, marker):
            simplified = carrier.lhs
        elif identical_nodes(carrier.lhs, marker):
            simplified = carrier.rhs
        else:
            raise RuntimeError(
                "Specification simplification rewrote its result marker"
            )
    # expressions got simplied to marker itself because (True == marker) -> marker
    elif isinstance(carrier, type(marker)) and identical_nodes(carrier, marker):
        simplified = BoolLit(True)
    # expressions got simplied to marker itself because (False == marker) -> (not marker)
    elif isinstance(carrier, Not) and identical_nodes(carrier.value, marker):
        simplified = BoolLit(False)
    else:
        raise RuntimeError(
            "Specification simplification produced an invalid result carrier"
        )

    simplified_spec_ast = simplified.constant_fold()
    # Keep simplified assumptions, but restore user checks verbatim after
    # removing the probe's temporary result carrier.
    result_ctx = simplified_ctx.copy(checks=list(ctx.checks))
    size_after = _spec_simplification_size(simplified_spec_ast, result_ctx)
    reduction = size_before - size_after
    if reduction > 0:
        size_change = f"reduced node count by {reduction}"
    elif reduction < 0:
        size_change = f"increased node count by {-reduction}"
    if reduction != 0:
        warnings.warn(
            f"Specification {ctx.name} simplification {size_change}: "
            f"{size_before} -> {size_after}",
            UserWarning,
            stacklevel=3,
        )
    return simplified_spec_ast, result_ctx


def _fixed_point_real_bounds(dtype: Q | UQ) -> tuple[float, float]:
    quantum = 2.0 ** -dtype.frac_bits
    if isinstance(dtype, UQ):
        return 0.0, (2.0 ** dtype.int_bits) - quantum
    magnitude = 2.0 ** (dtype.int_bits - 1)
    return -magnitude, magnitude - quantum


def _fixed_point_result_fits(
    spec_ast: RealExpr,
    dtype: Q | UQ,
    ctx: SpecContext,
) -> BoolExpr:
    min_bound, max_bound = _fixed_point_real_bounds(dtype)
    return (spec_ast >= ctx.real_val(min_bound)) & (spec_ast <= ctx.real_val(max_bound))


def _prove_result_fits(spec_ast, output_type, ctx):
    if output_type is Q:
        return True
    if output_type is UQ:
        result_fits = spec_ast >= ctx.zero()
    elif isinstance(output_type, (Q, UQ)):
        result_fits = _fixed_point_result_fits(spec_ast, output_type, ctx)
    else:
        raise TypeError(
            "Result-fit proof expects a Q or UQ output type, got "
            f"{output_type!r}"
        )

    range_ctx = ctx.copy(checks=[result_fits])
    status, _proof_trace = check_equivalence(range_ctx, schedule=FEASIBILITY_SCHEDULE)
    return status == "unsat"


def _check_spec_reachability(ctx: SpecContext) -> None:
    report = simplify_ctx(ctx.copy(checks=[]))
    feasibility_status = report.get("feasibility_status", "unknown")

    if feasibility_status == "not feasible":
        raise InfeasibleError(
            f"Specification {ctx.name!r} is unreachable: "
            "no input satisfies its assumptions"
        )
    if feasibility_status == "feasible":
        return

    # Asking whether spec is satisfiable is equivalent to checking the
    # counterexample query for an always-false property:
    # assumes && !false == assumes.
    simplified_ctx = report["new_ctx"]
    reachability_ctx = simplified_ctx.copy(checks=[simplified_ctx.false()])
    status, _proof_trace = check_equivalence(
        reachability_ctx,
        schedule=REACHABILITY_FALLBACK_SCHEDULE,
    )
    if status == "unsat":
        raise InfeasibleError(
            f"Specification {ctx.name!r} is unreachable: "
            "no input satisfies its assumptions"
        )
    elif status != "sat":
        raise ZolotoneError(
            f"Could not determine whether specification {ctx.name!r} has a reachable input"
        )


def _check_spec_obligations(ctx: SpecContext) -> None:
    if not ctx.checks:
        return

    status, _proof_trace = check_equivalence(
        ctx,
        schedule=FEASIBILITY_SCHEDULE,
    )
    if status == "unsat":
        return
    if status == "sat":
        raise InfeasibleError(f"Specification {ctx.name!r} has a check that does not hold")
    raise ZolotoneError(f"Could not prove all checks in specification {ctx.name!r}")


def _domain_nodes(location: str, expression: SpecNode):
    for child in children(expression):
        yield from _domain_nodes(location, child)
    yield location, expression


def _warn_about_domain_errors(
    spec_ast: SpecNode,
    ctx: SpecContext,
) -> None:
    roots: list[tuple[str, SpecNode]] = [
        (f"assumption {index}", assume)
        for index, assume in enumerate(ctx.assumes, start=1)
    ]
    roots.append(("result", spec_ast))
    roots.extend(
        (f"check {index}", check)
        for index, check in enumerate(ctx.checks, start=1)
    )
    roots.extend(
        (f"requirement {index}", requirement)
        for index, requirement in enumerate(ctx.requirements, start=1)
    )
    tagged_nodes = [
        tagged_node
        for location, expression in roots
        for tagged_node in _domain_nodes(location, expression)
    ]
    findings = rival_domain_errors(
        [expression for _, expression in tagged_nodes],
        ctx.assumes,
    )
    if not findings:
        return

    for finding in findings:
        location, culprit = tagged_nodes[finding.expression_index]
        ranges = ", ".join(
            f"{name} in [{lower}, {upper}]"
            for name, (lower, upper) in zip(
                finding.free_vars,
                finding.rect,
                strict=True,
            )
        ) or "no input variables"
        warnings.warn(
            f"Specification {ctx.name!r} may trigger a domain error in "
            f"{location}: {culprit}; ranges: {ranges}",
            UserWarning,
            stacklevel=3,
        )


def _output_format_suggestion_with_search(
    spec_ast: RealExpr,
    conservative_format: Q | UQ,
    ctx: SpecContext,
) -> Q | UQ | None:
    dtype_type = UQ if _prove_result_fits(spec_ast, UQ, ctx) else Q

    def candidate(int_bits: int) -> Q | UQ:
        return dtype_type(int_bits, conservative_format.frac_bits)

    min_int_bits = max(0, 1 - conservative_format.frac_bits)
    max_int_bits = conservative_format.int_bits
    widest = candidate(max_int_bits)

    if not _prove_result_fits(spec_ast, widest, ctx):
        return None

    low = min_int_bits
    high = max_int_bits
    while low < high:
        middle = (low + high) // 2
        middle_dtype = candidate(middle)
        if _prove_result_fits(spec_ast, middle_dtype, ctx):
            high = middle
        else:
            low = middle + 1
    return candidate(low)


def _output_format_suggestion_with_range_analysis(
    spec_ast: RealExpr,
    return_annotation: Q | UQ,
    ctx: SpecContext,
) -> Q | UQ:
    output_range = rival_range_analysis(spec_ast, ctx)
    if output_range is None:
        raise ZolotoneError(f"Could not obtain output range for: {spec_ast}")

    lower, upper = output_range
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ZolotoneError(
            f"Could not obtain finite output range, got {output_range} "
            f"for {spec_ast}"
        )

    frac_bits = return_annotation.frac_bits
    scale = 1 << frac_bits
    if lower >= 0:
        required_raw = max(0, math.ceil(upper * scale))
        total_bits = max(1, frac_bits, required_raw.bit_length())
        int_bits = total_bits - frac_bits
        return UQ(int_bits, frac_bits)
    else:
        lower_raw = math.floor(lower * scale)
        upper_raw = math.ceil(upper * scale)
        required_magnitude = max(1, -lower_raw, upper_raw + 1)
        magnitude_bits = (required_magnitude - 1).bit_length()
        total_bits = max(1, frac_bits, magnitude_bits + 1)
        int_bits = total_bits - frac_bits
        return Q(int_bits, frac_bits)


def _output_format_suggestion(
    spec_ast: RealExpr,
    return_annotation: object,
    ctx: SpecContext,
) -> object | None:
    """Derive a conservative format with Rival, then shrink it by proof."""
    # TODO: provide a counterexample
    if return_annotation is UQ:
        if not _prove_result_fits(spec_ast, UQ, ctx):
            return Q
        return UQ
    elif return_annotation is Q:
        if _prove_result_fits(spec_ast, UQ, ctx):
            return UQ
        return Q
    elif isinstance(return_annotation, (Q, UQ)):
        conservative_format = _output_format_suggestion_with_range_analysis(
            spec_ast,
            return_annotation,
            ctx,
        )
        suggestion = _output_format_suggestion_with_search(
            spec_ast,
            conservative_format,
            ctx,
        )
        if suggestion is None:
            raise ValueError(
                "Rival-derived format does not fit the result: "
                f"{conservative_format}"
            )
        if not _prove_result_fits(spec_ast, suggestion, ctx):
            raise ValueError("That's a bug, derived dtype does not fit the result")
        return suggestion
    else:
        raise NotImplementedError(
            "Output format suggestion is not implemented for "
            f"{return_annotation!r}"
        )


def _format_output_annotation(annotation: object) -> str:
    if annotation in (Q, UQ):
        return annotation.__name__
    if isinstance(annotation, (Q, UQ)):
        return f"{type(annotation).__name__}({annotation.int_bits}, {annotation.frac_bits})"
    return repr(annotation)


def check_spec_feasibility(
    spec_ast: SpecNode,
    spec_inputs: tuple[tp.Any, ...],
    contract: _SpecContract,
    ctx: SpecContext,
) -> None:
    # Input ranges
    input_parameters = list(contract.signature.parameters.values())[:-1]
    for spec_input, parameter in zip(spec_inputs, input_parameters, strict=True):
        dtype = contract.annotations[parameter.name]
        if isinstance(dtype, (Q, UQ)):
            ctx.assume(_fixed_point_result_fits(spec_input, dtype, ctx))

    return_annotation = contract.annotations["return"]
    # Output validation
    if return_annotation is Bool or isinstance(return_annotation, Bool):
        if not isinstance(spec_ast, BoolExpr):
            raise TypeError(
                f"Specification returning {return_annotation!r} must produce "
                f"a Boolean expression, got {type(spec_ast).__name__}"
            )
        _check_spec_reachability(ctx)
        _warn_about_domain_errors(spec_ast, ctx)
        _check_spec_obligations(ctx)
        return

    if not isinstance(spec_ast, RealExpr):
        raise TypeError(
            f"Specification returning {return_annotation!r} must produce "
            f"a real expression, got {type(spec_ast).__name__}"
        )

    if return_annotation not in (Q, UQ) and not isinstance(return_annotation, (Q, UQ)):
        raise NotImplementedError(
            f"Output format {return_annotation!r} is not supported by "
            "the feasibility check"
        )

    _check_spec_reachability(ctx)
    _warn_about_domain_errors(spec_ast, ctx)
    _check_spec_obligations(ctx)

    # TODO: counterexample
    if not _prove_result_fits(spec_ast, return_annotation, ctx):
        suggestion = _output_format_suggestion(spec_ast, return_annotation, ctx)
        message = f"Specification result range does not fit {return_annotation!r}"
        if suggestion is not None:
            message += f"; try {_format_output_annotation(suggestion)} as the output format instead"
        else:
            message += "; could not find a fixed-point format that would fit the range"
        raise InfeasibleError(message)

    suggestion = _output_format_suggestion(spec_ast, return_annotation, ctx)
    if suggestion != return_annotation:
        warnings.warn(
            f"Output type {return_annotation} is wider than necessary; consider {suggestion}",
            UserWarning,
            stacklevel=3,
        )
    return
