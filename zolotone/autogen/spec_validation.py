from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
import typing as tp
import warnings

from ..errors import InfeasibleError, MissingError, ZolotoneError
from ..rival.analysis import (
    rival_domain_errors, rival_range_analysis,
    rival_prove_result_fits as _prove_result_fits,
)
from ..rival.machine import (
    build_boolean_evaluator, build_integer_evaluator, collect_free_vars,
)
from ..solver import check_equivalence
from ..spec.spec_ast import (
    And,
    BoolEq,
    BoolExpr,
    BoolLit,
    Eq,
    Not,
    RealExpr,
    RealLit,
    RealVar,
    Round,
    SpecNode,
    children,
    identical_nodes,
    substitute_spec_node,
    variables,
)
from ..spec.spec_context import SpecContext, normalize_ctx, simplify_ctx
from ..types import Bool, DataType, Q, UQ
from ..ast.nodes import _SpecContract

if tp.TYPE_CHECKING:
    from .autogen import Spec


FEASIBILITY_SCHEDULE = [
    {"tool": "simplify"},
    {"tool": "z3", "timeout_ms": 10_000},
]

REACHABILITY_FALLBACK_SCHEDULE = [
    {"tool": "z3", "timeout_ms": 10_000},
]



def _spec_input_dtypes(spec: "Spec") -> dict[SpecNode, DataType]:
    return {
        spec_input: spec.annotations[parameter.name]
        for spec_input, parameter in zip(
            spec.spec_inputs,
            tuple(spec.signature.parameters.values())[:-1],
            strict=True,
        )
    }


def spec_domain_analysis(spec: "Spec"):
    """Return reachability and output bounds, raising on failed or unresolved analysis.

    Inputs follow sorted variable-name order. Q inputs use signed codes and
    Boolean inputs use 0/1; fixed-point codes represent code * 2**-frac_bits.
    Generic Q/UQ outputs use 30 fractional bits without implicit rounding.
    """
    input_dtypes = _spec_input_dtypes(spec)
    output_dtype = spec.return_annotation
    generic_output = output_dtype is Q or output_dtype is UQ
    if not all(isinstance(dtype, (Q, UQ, Bool)) for dtype in input_dtypes.values()):
        raise NotImplementedError("spec domain analysis requires exact Q, UQ, or Bool inputs")
    if not generic_output and not isinstance(output_dtype, (Q, UQ, Bool)):
        raise NotImplementedError("spec domain analysis requires Q, UQ, or exact Bool output")

    assumes = BoolLit(True)
    for assume in spec.spec_ctx.assumes:
        assumes = assumes & assume
    checks = BoolLit(True)
    for check in spec.spec_ctx.checks:
        checks = checks & check
    output = spec.spec_ast

    # Scaling with respect to fractional bits
    for source, dtype in input_dtypes.items():
        code = RealVar(source.name)
        decoded = (
            code.ne(RealLit(0)) if isinstance(dtype, Bool)
            else code * (RealLit(2) ** RealLit(-dtype.frac_bits))
        )
        assumes = substitute_spec_node(assumes, source, decoded)
        checks = substitute_spec_node(checks, source, decoded)
        output = substitute_spec_node(output, source, decoded)

    # Creating Rival Machines
    kwargs = dict(
        free_vars=collect_free_vars([assumes, checks, output]),
        input_total_bits=max((dtype.total_bits() for dtype in input_dtypes.values()), default=1),
    )
    # The precondition path lets an error-free false conjunct reject the box.
    assumes_machine = build_integer_evaluator(
        RealLit(0), precondition=assumes, output_total_bits=1, **kwargs,
    )
    checks_machine = build_boolean_evaluator(checks, **kwargs)
    output_frac_bits = 30 if generic_output else (
        0 if isinstance(output_dtype, Bool) else output_dtype.frac_bits
    )
    output_scale = Fraction(2) ** output_frac_bits
    if isinstance(output_dtype, Bool):
        output_machine = build_boolean_evaluator(output, **kwargs)
    else:
        output_machine = build_integer_evaluator(
            output * RealLit(output_scale),
            output_total_bits=30 if generic_output else output_dtype.total_bits(), **kwargs,
        )

    # Creating Initial Rectangulars
    dtypes_by_name = {source.name: dtype for source, dtype in input_dtypes.items()}
    initial_box: list[tuple[int, int]] = []
    for name in kwargs["free_vars"]:
        dtype = dtypes_by_name[name]
        width = dtype.total_bits()
        if isinstance(dtype, Q):
            lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
        else:
            lower, upper = 0, (1 << width) - 1
        initial_box.append((lower, upper))

    def split_box(box: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """Bisect the widest integer-code interval; point boxes have no children."""
        axis = max(range(len(box)), key=lambda i: box[i][1] - box[i][0], default=None)
        if axis is None or box[axis][0] == box[axis][1]:
            return []
        lower, upper = box[axis]
        middle = (lower + upper) // 2
        left, right = list(box), list(box)
        left[axis] = (lower, middle)
        right[axis] = (middle + 1, upper)
        return [left, right]

    def evaluate_box(machine, box, is_point, location):
        try:
            if is_point:
                return machine.apply_point([lower for lower, _ in box])
            return machine.apply_range(box)
        except ValueError as exc:
            raise ZolotoneError(
                f"Could not evaluate {location} at input point {box}: {exc}"
            ) from exc

    reachable = False
    output_bounds = None
    pending = [initial_box]
    while pending:
        box = pending.pop()
        is_point = all(lower == upper for lower, upper in box)

        assumes = evaluate_box(assumes_machine, box, is_point, "assumptions")
        if not assumes.precondition_error and assumes.precondition in (False, (False, False)):
            # An error-free rejection excludes this box.
            continue
        if assumes.precondition_total_error:
            raise ZolotoneError(f"Domain error in assumptions at input point {box}")
        if assumes.precondition_error or assumes.precondition not in (True, (True, True)):
            if is_point:
                raise ZolotoneError(f"Unresolved assumptions at input point {box}")
            pending.extend(split_box(box))
            continue

        # Every point in this box satisfies all assumptions.
        reachable = True

        checks = evaluate_box(checks_machine, box, is_point, "checks")
        if checks.output_total_error:
            raise ZolotoneError(f"Domain error in checks at input box {box}")
        if checks.output_error or checks.output not in (0, 1, (0, 0), (1, 1)):
            if is_point:
                raise ZolotoneError(f"Unresolved checks at input point {box}")
            pending.extend(split_box(box))
            continue
        if checks.output in (0, (0, 0)):
            raise InfeasibleError(f"Specification check does not hold at input point {box}")

        output = evaluate_box(output_machine, box, is_point, "output")
        if output.output_total_error:
            raise ZolotoneError(f"Domain error in output at input point {box}")
        if output.output_error or output.output is None:
            if is_point:
                raise ZolotoneError(f"Unresolved output at input point {box}")
            pending.extend(split_box(box))
            continue

        lower, upper = (output.output, output.output) if is_point else output.output
        lower, upper = lower / output_scale, upper / output_scale
        output_bounds = (
            (lower, upper) if output_bounds is None else
            (min(output_bounds[0], lower), max(output_bounds[1], upper))
        )

    if not reachable:
        raise InfeasibleError(f"Specification {spec.name!r} is unreachable")
    return reachable, output_bounds


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
    spec: Spec,
) -> None:
    input_variables = _spec_value_variables(spec.spec_inputs)
    specification_variables = variables(spec.spec_ast)
    for expressions in (
        spec.spec_ctx.assumes,
        spec.spec_ctx.checks,
        spec.spec_ctx.requirements,
    ):
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

    relevant_variables = variables(spec.spec_ast)
    condition_variables = [
        variables(conjunct)
        for expressions in (
            spec.spec_ctx.assumes,
            spec.spec_ctx.checks,
            spec.spec_ctx.requirements,
        )
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
            f"Specification {spec.spec_ctx.name!r} has unused input variables: "
            f"{rendered}",
            UserWarning,
            stacklevel=3,
        )


def _simplify_spec_ast(
    spec: Spec,
) -> Spec:
    size_before = _spec_simplification_size(spec.spec_ast, spec.spec_ctx)
    # Normalize assumptions before simplification so complementary comparison
    # forms can expose redundant constraints. User checks remain outside the
    # probe so they are preserved rather than discharged.
    probe_ctx = normalize_ctx(spec.spec_ctx.copy(checks=[]))
    if isinstance(spec.spec_ast, RealExpr):
        marker = probe_ctx.fresh_real("simplified_spec_result")
    elif isinstance(spec.spec_ast, BoolExpr):
        marker = probe_ctx.fresh_bool("simplified_spec_result")
    else:
        raise TypeError(
            "Specification simplification expects a real or Boolean "
            f"expression, got {type(spec.spec_ast).__name__}"
        )

    probe_ctx.check(spec.spec_ast.eq(marker))
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

    simplified_spec_ast = simplified._constant_fold()
    # Keep assumptions that justified the simplified result. Rewrite checks to
    # refer to that result without folding them away; they will be lowered as
    # runtime implementation checks later.
    result_ctx = simplified_ctx.copy(
        checks=[
            substitute_spec_node(
                check,
                spec.spec_ast,
                simplified_spec_ast,
            )
            for check in spec.spec_ctx.checks
        ]
    )
    result_ctx = normalize_ctx(result_ctx)
    size_after = _spec_simplification_size(simplified_spec_ast, result_ctx)
    reduction = size_before - size_after
    if reduction > 0:
        size_change = f"reduced node count by {reduction}"
    elif reduction < 0:
        size_change = f"increased node count by {-reduction}"
    if reduction != 0:
        warnings.warn(
            f"Specification {spec.spec_ctx.name} simplification {size_change}: "
            f"{size_before} -> {size_after}",
            UserWarning,
            stacklevel=3,
        )
    return replace(
        spec,
        spec_ast=simplified_spec_ast,
        spec_ctx=result_ctx,
    )


def _fixed_point_real_bounds(dtype: Q | UQ) -> tuple[Fraction, Fraction]:
    width = dtype.total_bits()
    minimum = -(1 << (width - 1)) if isinstance(dtype, Q) else 0
    maximum = (1 << (width - isinstance(dtype, Q))) - 1
    return Fraction(minimum, 1 << dtype.frac_bits), Fraction(maximum, 1 << dtype.frac_bits)


def _fixed_point_result_fits(
    spec_ast: RealExpr,
    dtype: Q | UQ,
    ctx: SpecContext,
) -> BoolExpr:
    min_bound, max_bound = _fixed_point_real_bounds(dtype)
    return (spec_ast >= ctx.real_val(min_bound)) & (spec_ast <= ctx.real_val(max_bound))


def _check_spec_reachability(
    spec: Spec,
    analysis: SpecDomainAnalysis | None = None,
) -> None:
    """Prove user assumptions admit an input within the typed input domain."""
    if analysis is not None:
        if analysis.assumption_domain_error is not None:
            # Domain diagnostics below provide the more useful location; an
            # invalid assumption cannot establish a valid reachable point.
            return
        if not analysis.reachable:
            raise InfeasibleError(
                f"Specification {spec.spec_ctx.name!r} is unreachable: "
                "no input satisfies its assumptions"
            )
        return
    if not spec.has_user_assumptions:
        # The input domain is feasible by construction without user constraints.
        return
    report = simplify_ctx(spec.spec_ctx.copy(checks=[]))
    feasibility_status = report.get("feasibility_status", "unknown")

    if feasibility_status == "not feasible":
        raise InfeasibleError(
            f"Specification {spec.spec_ctx.name!r} is unreachable: "
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
            f"Specification {spec.spec_ctx.name!r} is unreachable: "
            "no input satisfies its assumptions"
        )
    elif status != "sat":
        raise ZolotoneError(
            "Could not determine whether specification "
            f"{spec.spec_ctx.name!r} has a reachable input"
        )


def _check_spec_obligations(
    spec: Spec,
    analysis: SpecDomainAnalysis | None = None,
) -> None:
    if analysis is not None:
        if any(witness is not None for witness in analysis.check_domain_errors):
            raise ZolotoneError(
                f"Specification {spec.spec_ctx.name!r} has a domain error in a check"
            )
        if any(witness is not None for witness in analysis.failed_checks):
            raise InfeasibleError(
                f"Specification {spec.spec_ctx.name!r} has a check that does not hold"
            )
        return
    if not spec.spec_ctx.checks:
        return

    status, _proof_trace = check_equivalence(
        spec.spec_ctx,
        schedule=FEASIBILITY_SCHEDULE,
    )
    if status == "unsat":
        return
    if status == "sat":
        raise InfeasibleError(
            f"Specification {spec.spec_ctx.name!r} has a check that does not hold"
        )
    raise ZolotoneError(
        f"Could not prove all checks in specification {spec.spec_ctx.name!r}"
    )


def _domain_nodes(location: str, expression: SpecNode):
    for child in children(expression):
        yield from _domain_nodes(location, child)
    yield location, expression


def _warn_about_domain_errors(
    spec: Spec,
    analysis: SpecDomainAnalysis | None = None,
) -> None:
    if analysis is not None:
        findings = [
            ("assumptions", analysis.assumption_domain_error),
            ("result", analysis.result_domain_error),
            *((f"check {index}", witness)
              for index, witness in enumerate(analysis.check_domain_errors, start=1)),
            *((f"requirement {index}", witness)
              for index, witness in enumerate(analysis.requirement_domain_errors, start=1)),
        ]
        for location, witness in findings:
            if witness is not None:
                rendered = ", ".join(str(code) for code in witness) or "no input variables"
                warnings.warn(
                    f"Specification {spec.spec_ctx.name!r} triggers a domain error "
                    f"in {location} at input codes [{rendered}]",
                    UserWarning,
                    stacklevel=3,
                )
        return
    roots: list[tuple[str, SpecNode]] = [
        (f"assumption {index}", assume)
        for index, assume in enumerate(spec.spec_ctx.assumes, start=1)
    ]
    roots.append(("result", spec.spec_ast))
    roots.extend(
        (f"check {index}", check)
        for index, check in enumerate(spec.spec_ctx.checks, start=1)
    )
    roots.extend(
        (f"requirement {index}", requirement)
        for index, requirement in enumerate(spec.spec_ctx.requirements, start=1)
    )
    tagged_nodes = [
        tagged_node
        for location, expression in roots
        for tagged_node in _domain_nodes(location, expression)
    ]
    findings = rival_domain_errors(
        [expression for _, expression in tagged_nodes],
        spec.spec_ctx.assumes,
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
            f"Specification {spec.spec_ctx.name!r} may trigger a domain error in "
            f"{location}: {culprit}; ranges: {ranges}",
            UserWarning,
            stacklevel=3,
        )


def _derive_output_guards(
    spec: Spec,
) -> tuple[BoolExpr, ...]:
    """Return runtime guards derived from the specification output."""
    if spec.return_annotation in (Q, UQ) or isinstance(
        spec.return_annotation,
        (Q, UQ),
    ):
        if not isinstance(spec.spec_ast, RealExpr):
            raise TypeError(
                "Numeric output assertions require a real expression, got "
                f"{type(spec.spec_ast).__name__}"
            )
        output_range = rival_range_analysis(spec.spec_ast, spec.spec_ctx)
        if output_range is None:
            raise ZolotoneError(
                f"Could not obtain output range for: {spec.spec_ast}"
            )

        lower, upper = output_range
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ZolotoneError(
                f"Could not obtain finite output range, got {output_range} "
                f"for {spec.spec_ast}"
            )
        return (
            (spec.spec_ast >= spec.spec_ctx.real_val(lower))
            & (spec.spec_ast <= spec.spec_ctx.real_val(upper)),
        )

    if spec.return_annotation is Bool or isinstance(spec.return_annotation, Bool):
        return (
            spec.spec_ast.eq(spec.spec_ctx.true())
            | spec.spec_ast.eq(spec.spec_ctx.false()),
        )

    raise NotImplementedError(
        "Output assertions are not implemented for "
        f"{spec.return_annotation!r}"
    )


def _output_format_suggestion_with_search(
    spec_ast: RealExpr,
    conservative_format: Q | UQ,
    ctx: SpecContext,
    input_dtypes: dict[SpecNode, DataType],
) -> Q | UQ | None:
    dtype_type = UQ if _prove_result_fits(spec_ast, UQ, ctx, input_dtypes) else Q

    def candidate(int_bits: int) -> Q | UQ:
        return dtype_type(int_bits, conservative_format.frac_bits)

    min_int_bits = max(0, 1 - conservative_format.frac_bits)
    max_int_bits = conservative_format.int_bits
    widest = candidate(max_int_bits)

    if not _prove_result_fits(spec_ast, widest, ctx, input_dtypes):
        return None

    low = min_int_bits
    high = max_int_bits
    while low < high:
        middle = (low + high) // 2
        middle_dtype = candidate(middle)
        if _prove_result_fits(spec_ast, middle_dtype, ctx, input_dtypes):
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


def _output_format_suggestion_from_exact_bounds(
    output_range: tuple[Fraction, Fraction],
    return_annotation: Q | UQ,
) -> Q | UQ:
    """Choose the smallest format from cached integer-machine bounds."""
    lower, upper = output_range
    frac_bits = return_annotation.frac_bits
    scale = 1 << frac_bits
    if lower >= 0:
        required_raw = max(0, math.ceil(upper * scale))
        total_bits = max(1, frac_bits, required_raw.bit_length())
        return UQ(total_bits - frac_bits, frac_bits)
    lower_raw = math.floor(lower * scale)
    upper_raw = math.ceil(upper * scale)
    required_magnitude = max(1, -lower_raw, upper_raw + 1)
    magnitude_bits = (required_magnitude - 1).bit_length()
    total_bits = max(1, frac_bits, magnitude_bits + 1)
    return Q(total_bits - frac_bits, frac_bits)


def _output_format_suggestion(
    spec_ast: RealExpr,
    return_annotation: object,
    ctx: SpecContext,
    input_dtypes: dict[SpecNode, DataType],
) -> object | None:
    """Derive a conservative format with Rival, then shrink it by search."""
    # TODO: provide a counterexample
    if return_annotation is Bool or isinstance(return_annotation, Bool):
        return return_annotation
    if return_annotation is UQ:
        if not _prove_result_fits(spec_ast, UQ, ctx, input_dtypes):
            return Q
        return UQ
    elif return_annotation is Q:
        if _prove_result_fits(spec_ast, UQ, ctx, input_dtypes):
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
            input_dtypes,
        )
        if suggestion is None:
            raise ValueError(
                "Rival-derived format does not fit the result: "
                f"{conservative_format}"
            )
        if not _prove_result_fits(spec_ast, suggestion, ctx, input_dtypes):
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


def _check_output_format(
    spec: Spec,
    analysis: SpecDomainAnalysis | None = None,
) -> None:
    input_dtypes = _spec_input_dtypes(spec)
    cached_fit: bool | None = None
    if (
        analysis is not None
        and isinstance(spec.return_annotation, (Q, UQ))
        and analysis.output_bounds is not None
        and analysis.assumption_domain_error is None
        and analysis.result_domain_error is None
    ):
        lower, upper = analysis.output_bounds
        minimum, maximum = _fixed_point_real_bounds(spec.return_annotation)
        if minimum <= lower <= upper <= maximum:
            cached_fit = True
        elif analysis.exact_output_bounds is not None:
            cached_fit = False
    # TODO: counterexample
    if cached_fit is False or (
        cached_fit is None and not _prove_result_fits(
            spec.spec_ast,
            spec.return_annotation,
            spec.spec_ctx,
            input_dtypes,
        )
    ):
        if (
            analysis is not None
            and analysis.exact_output_bounds is not None
            and isinstance(spec.return_annotation, (Q, UQ))
        ):
            suggestion = _output_format_suggestion_from_exact_bounds(
                analysis.exact_output_bounds, tp.cast(Q | UQ, spec.return_annotation),
            )
        else:
            suggestion = _output_format_suggestion(
                spec.spec_ast,
                spec.return_annotation,
                spec.spec_ctx,
                input_dtypes,
            )
        message = (
            "Specification result range does not fit "
            f"{spec.return_annotation!r}"
        )
        if suggestion is not None:
            message += f"; try {_format_output_annotation(suggestion)} as the output format instead"
        else:
            message += "; could not find a fixed-point format that would fit the range"
        raise InfeasibleError(message)

    if (
        analysis is not None
        and isinstance(spec.return_annotation, (Q, UQ))
        and analysis.exact_output_bounds is not None
        and analysis.assumption_domain_error is None
        and analysis.result_domain_error is None
    ):
        suggestion = _output_format_suggestion_from_exact_bounds(
            analysis.exact_output_bounds, spec.return_annotation,
        )
    else:
        suggestion = _output_format_suggestion(
            spec.spec_ast,
            spec.return_annotation,
            spec.spec_ctx,
            input_dtypes,
        )
    if suggestion != spec.return_annotation:
        warnings.warn(
            f"Output type {spec.return_annotation} is wider than necessary; "
            f"consider {suggestion}",
            UserWarning,
            stacklevel=3,
        )


def add_input_ranges(spec: Spec) -> Spec:
    """Prepend fixed-point input-domain facts to a copy of the spec."""
    input_parameters = list(spec.contract.signature.parameters.values())[:-1]
    input_range_assumes = []
    for spec_input, parameter in zip(
        spec.spec_inputs,
        input_parameters,
        strict=True,
    ):
        dtype = spec.annotations[parameter.name]
        if isinstance(dtype, (Q, UQ)):
            input_range_assumes.append(
                _fixed_point_result_fits(spec_input, dtype, spec.spec_ctx)
            )
    spec_ctx = spec.spec_ctx.copy()
    spec_ctx.assumes[:0] = input_range_assumes
    return replace(spec, spec_ctx=spec_ctx)


def _validate_spec_shape(
    spec: Spec,
) -> None:
    """Validate the specification result category before semantic analysis."""
    if spec.return_annotation is Bool or isinstance(spec.return_annotation, Bool):
        if not isinstance(spec.spec_ast, BoolExpr):
            raise TypeError(
                f"Specification returning {spec.return_annotation!r} must produce "
                f"a Boolean expression, got {type(spec.spec_ast).__name__}"
            )
        return

    if spec.return_annotation in (Q, UQ) or isinstance(
        spec.return_annotation,
        (Q, UQ),
    ):
        if not isinstance(spec.spec_ast, RealExpr):
            raise TypeError(
                f"Specification returning {spec.return_annotation!r} must produce "
                f"a real expression, got {type(spec.spec_ast).__name__}"
            )
        return

    raise NotImplementedError(
        f"Output format {spec.return_annotation!r} is not supported by "
        "autogeneration"
    )


def _validate_rounding_precision(spec: Spec) -> None:
    """Require resolved precision everywhere and reject narrowing root rounding."""
    pending = [
        spec.spec_ast,
        *spec.spec_ctx.assumes,
        *spec.spec_ctx.checks,
        *spec.spec_ctx.requirements,
    ]
    while pending:
        node = pending.pop()
        if isinstance(node, Round) and node.frac_bits is None:
            raise MissingError(
                "Cannot infer rounding precision for non-root rounding; "
                "provide frac_bits explicitly"
            )
        pending.extend(children(node))

    if (
        isinstance(spec.spec_ast, Round)
        and isinstance(spec.return_annotation, (Q, UQ))
        and spec.spec_ast.frac_bits > spec.return_annotation.frac_bits
    ):
        raise InfeasibleError(
            f"Root rounding uses frac_bits={spec.spec_ast.frac_bits}, but output "
            f"{spec.return_annotation!r} has frac_bits={spec.return_annotation.frac_bits}; "
            "the output cannot discard fractional precision. "
            "Round explicitly to the requested output precision instead"
        )


def _validate_spec_requirements(spec: Spec) -> None:
    spec.spec_ctx.validate_requirements()


def add_output_guards(spec: Spec) -> Spec:
    output_guards = _derive_output_guards(spec)
    guard_spec = replace(
        spec,
        spec_ctx=spec.spec_ctx.copy(checks=list(output_guards)),
    )
    _check_spec_obligations(guard_spec)
    spec_ctx = spec.spec_ctx.copy()
    spec_ctx.checks.extend(output_guards)
    return replace(spec, spec_ctx=spec_ctx)
