"""Lower validated specification expressions into implementation nodes."""

from __future__ import annotations

from dataclasses import replace
import itertools
import math
import typing as tp

from ..errors import MissingError, ZolotoneError
from ..rival import rival_range_analysis
from ..spec.spec_ast import (
    BoolExpr,
    BoolLit,
    BoolVar,
    Gt,
    Lt,
    RealExpr,
    RealLit,
    RealVar,
    SpecNode,
    children,
    substitute_spec_node,
)
from ..spec.spec_context import SpecContext
from ..types import Bool, Q, UQ, Value
from .node import Node
from .nodes import (
    Composite,
    Const,
    Var,
    _annotation_matches,
    _SpecContract,
)
from .spec_validation import _fixed_point_real_bounds, _prove_result_fits

if tp.TYPE_CHECKING:
    from .autogen import Spec


class _Candidate(tp.NamedTuple):
    node: Node
    spec: SpecNode
    depth: int


MAX_SEARCH_DEPTH = 30


# TODO: this is not too good
def _exact_fixed_point_value(value: int | float) -> Value[int]:
    numerator, denominator = value.as_integer_ratio()
    if denominator & (denominator - 1):
        raise NotImplementedError(
            f"Cannot exactly represent {value!r} as fixed point"
        )

    if denominator == 1:
        try:
            return UQ.from_int(numerator)
        except ValueError:
            return Q.from_int(numerator)

    frac_bits = denominator.bit_length() - 1
    if numerator >= 0:
        total_bits = max(frac_bits, numerator.bit_length())
        dtype = UQ(total_bits - frac_bits, frac_bits)
        raw = numerator
    else:
        signed_bits = (abs(numerator) - 1).bit_length() + 1
        total_bits = max(frac_bits, signed_bits)
        dtype = Q(total_bits - frac_bits, frac_bits)
        raw = (1 << total_bits) + numerator
    return dtype.from_bits(raw)


def _spec_subexpressions(root: SpecNode) -> tuple[SpecNode, ...]:
    result = []
    visited = set()

    def visit(expr: SpecNode) -> None:
        if expr in visited:
            return
        visited.add(expr)
        result.append(expr)
        for child in children(expr):
            visit(child)

    visit(root)
    return tuple(result)


def _matching_subexpression(
    expr: object,
    subexpressions: tuple[SpecNode, ...],
) -> SpecNode | None:
    if not isinstance(expr, SpecNode):
        return None
    return next(
        (
            subexpression
            for subexpression in subexpressions
            if expr.identical(subexpression)
        ),
        None,
    )


def _matches_result(
    candidate: _Candidate,
    spec_ast: SpecNode,
    return_annotation: object,
) -> bool:
    return (
        candidate.spec.identical(spec_ast)
        and _annotation_matches(candidate.node.dtype, return_annotation)
    )


def _candidate_output_range_fits(
    candidate: _Candidate,
    range_ctx: SpecContext,
    range_cache: dict[SpecNode, tuple[float, float] | None],
    fit_cache: dict[tuple[SpecNode, Q | UQ], bool],
) -> bool:
    if not isinstance(candidate.spec, RealExpr):
        return True
    if not isinstance(candidate.node.dtype, (Q, UQ)):
        raise NotImplementedError(
            "Real-valued autogeneration candidate must have a Q or UQ "
            f"output format, got {candidate.node.dtype!r} from "
            f"{candidate.node.name!r}"
        )

    if candidate.spec not in range_cache:
        range_cache[candidate.spec] = rival_range_analysis(
            candidate.spec,
            range_ctx,
        )
    output_range = range_cache[candidate.spec]
    if output_range is not None:
        lower, upper = output_range
        dtype_lower, dtype_upper = _fixed_point_real_bounds(candidate.node.dtype)
        if (
            math.isfinite(lower)
            and math.isfinite(upper)
            and lower >= dtype_lower
            and upper <= dtype_upper
        ):
            return True

    fit_key = (candidate.spec, candidate.node.dtype)
    if fit_key not in fit_cache:
        fit_cache[fit_key] = _prove_result_fits(
            candidate.spec,
            candidate.node.dtype,
            range_ctx,
        )
    return fit_cache[fit_key]


def search_lower_spec_to_impl(
    spec_ast: SpecNode,
    spec_input_nodes: dict[tp.Any, Node],
    return_annotation: object,
    range_ctx: SpecContext,
) -> Node:
    from ..components import LOSSLESS_COMPONENTS

    # Lowering leaves first
    subexpressions = _spec_subexpressions(spec_ast)
    range_cache: dict[SpecNode, tuple[float, float] | None] = {}
    fit_cache: dict[tuple[SpecNode, Q | UQ], bool] = {}
    candidates = [
        _Candidate(node=node, spec=expr, depth=0)
        for expr, node in spec_input_nodes.items()
    ]
    states = {(candidate.spec, candidate.node.dtype) for candidate in candidates}

    for expr in subexpressions:
        if isinstance(expr, RealLit):
            literal = _Candidate(
                node=Const(_exact_fixed_point_value(expr.value)),
                spec=expr,
                depth=0,
            )
            state = (literal.spec, literal.node.dtype)
        elif isinstance(expr, BoolLit):
            literal = _Candidate(
                node=Const(Bool().from_bits(int(expr.value))),
                spec=expr,
                depth=0,
            )
            state = (literal.spec, literal.node.dtype)
        elif isinstance(expr, (BoolVar, RealVar)):
            if expr not in spec_input_nodes:
                raise MissingError(
                    f"Undeclared variable in specification: {expr}"
                )
            continue
        else:
            continue

        if state not in states:
            states.add(state)
            candidates.append(literal)

    # Shortcut
    for candidate in candidates:
        if _matches_result(candidate, spec_ast, return_annotation):
            return candidate.node

    depth = 1
    while True:
        generated = []
        for component in LOSSLESS_COMPONENTS:
            component_contract = getattr(component, "_spec_contract", None)
            if not isinstance(component_contract, _SpecContract):
                raise MissingError(
                    f"Autogeneration component {component!r} does not expose "
                    "a specification contract"
                )
            component_parameters = list(
                component_contract.signature.parameters.values()
            )[:-1]
            input_pools = [
                tuple(
                    candidate
                    for candidate in candidates
                    if _annotation_matches(
                        candidate.node.dtype,
                        component_contract.annotations[parameter.name],
                    )
                )
                for parameter in component_parameters
            ]
            component_name = component_contract.display_name
            if any(not pool for pool in input_pools):
                continue

            for arguments in itertools.product(*input_pools):
                if max(
                    (argument.depth for argument in arguments),
                    default=0,
                ) != depth - 1:
                    continue
                try:
                    node = component(*(argument.node for argument in arguments))
                    component_ctx = SpecContext(
                        f"autogen candidate {component_name}"
                    )
                    candidate_spec = node.spec(
                        *(argument.spec for argument in arguments),
                        component_ctx,
                    )
                except (TypeError, ValueError, NotImplementedError, OverflowError):
                    continue

                matched_spec = _matching_subexpression(
                    candidate_spec,
                    subexpressions,
                )
                if matched_spec is None:
                    continue

                candidate = _Candidate(
                    node=node,
                    spec=matched_spec,
                    depth=depth,
                )
                if not _candidate_output_range_fits(
                    candidate,
                    range_ctx,
                    range_cache,
                    fit_cache,
                ):
                    continue
                state = (candidate.spec, candidate.node.dtype)
                if state in states:
                    continue
                states.add(state)
                if _matches_result(candidate, spec_ast, return_annotation):
                    return candidate.node
                generated.append(candidate)

        if not generated:
            raise ZolotoneError(
                f"Cannot lower specification {spec_ast!r} to "
                f"{return_annotation!r}; search reached a fixpoint after "
                f"depth {depth} with {len(states)} candidate states"
            )
        if depth == MAX_SEARCH_DEPTH:
            raise ZolotoneError(
                f"Cannot lower specification {spec_ast!r} to "
                f"{return_annotation!r}; search reached the maximum depth "
                f"of {MAX_SEARCH_DEPTH} with {len(states)} candidate states"
            )

        candidates.extend(generated)
        depth += 1


def _make_conditions_bit_precise(
    conditions: tp.Iterable[BoolExpr],
    spec_inputs: tuple[tp.Any, ...],
    contract: _SpecContract,
    range_ctx: SpecContext,
) -> tuple[BoolExpr, ...]:
    """Rewrite strict comparisons using formats selected by lowering."""
    input_parameters = list(contract.signature.parameters.values())[:-1]
    spec_input_nodes = dict(
        zip(
            spec_inputs,
            (
                Var(parameter.name, contract.annotations[parameter.name])
                for parameter in input_parameters
            ),
            strict=True,
        )
    )

    rewritten_conditions = []
    for condition in conditions:
        pending = [condition]
        strict_comparisons = []
        while pending:
            node = pending.pop()
            pending.extend(children(node))
            if isinstance(node, (Lt, Gt)):
                strict_comparisons.append(node)

        rewritten = condition
        for comparison in reversed(strict_comparisons):
            lowered = search_lower_spec_to_impl(
                comparison,
                spec_input_nodes,
                Bool(),
                range_ctx,
            )
            lowered_args = getattr(lowered, "args", ())
            if len(lowered_args) != 2:
                continue
            lhs_dtype = lowered_args[0].dtype
            rhs_dtype = lowered_args[1].dtype
            if not isinstance(lhs_dtype, (Q, UQ)) or not isinstance(
                rhs_dtype,
                (Q, UQ),
            ):
                continue

            frac_bits = max(lhs_dtype.frac_bits, rhs_dtype.frac_bits)
            quantum = range_ctx.real_val(2.0 ** -frac_bits)
            if isinstance(comparison, Gt):
                precise = comparison.lhs >= comparison.rhs + quantum
            else:
                precise = comparison.lhs <= comparison.rhs - quantum
            rewritten = substitute_spec_node(
                rewritten,
                comparison,
                precise.constant_fold(),
            )
        rewritten_conditions.append(rewritten)
    return tuple(rewritten_conditions)


def rewrite_strict_conditions(
    spec: Spec,
) -> Spec:
    """Rewrite strict assumptions and checks for fixed-point semantics.

    Assumptions are rewritten in context order so each one is lowered using
    only the preceding precise assumptions. Requirements stay in source form
    because Z3's continuous-real model cannot exclude gaps between adjacent
    fixed-point values.
    """
    condition_ctx = spec.spec_ctx.copy(
        assumes=[],
        checks=[],
        requirements=[],
    )
    # TODO: This lowering can produce wider types that neccessary.
    # But only wider integer bits though, fractional bits are as tight as possible
    # So, this code cannot be used for general lowering, but for computing quantums it is okay
    for assumption in spec.spec_ctx.assumes:
        precise_assumption, = _make_conditions_bit_precise(
            (assumption,),
            spec.spec_inputs,
            spec.contract,
            condition_ctx,
        )
        condition_ctx.assumes.append(precise_assumption)

    precise_checks = _make_conditions_bit_precise(
        spec.spec_ctx.checks,
        spec.spec_inputs,
        spec.contract,
        condition_ctx,
    )
    return replace(
        spec,
        spec_ctx=spec.spec_ctx.copy(
            assumes=list(condition_ctx.assumes),
            checks=list(precise_checks),
        ),
    )


def lower_spec_result(
    spec: Spec,
) -> Node:
    @Composite(name=spec.name, spec=spec.function)
    def generated_impl(*impl_inputs: Node) -> Node:
        spec_input_nodes = dict(
            zip(spec.spec_inputs, impl_inputs, strict=True)
        )
        return search_lower_spec_to_impl(
            spec.spec_ast,
            spec_input_nodes,
            spec.return_annotation,
            spec.spec_ctx,
        )

    input_parameters = list(spec.contract.signature.parameters.values())[:-1]
    impl_inputs = [
        Var(
            name=parameter.name,
            dtype=spec.annotations[parameter.name],
        )
        for parameter in input_parameters
    ]
    return generated_impl(*impl_inputs)


def attach_lowered_conditions(
    lowered_composite: Node,
    spec: Spec,
) -> None:
    """Lower and attach specification assumptions and runtime checks."""
    spec_input_nodes = dict(
        zip(spec.spec_inputs, lowered_composite.inner_args, strict=True)
    )
    lowered_assumes = []
    prior_assumes = []
    for assumption in spec.spec_ctx.assumes:
        range_ctx = spec.spec_ctx.copy(
            assumes=list(prior_assumes),
            checks=[],
        )
        lowered_assumes.append(
            (
                assumption,
                search_lower_spec_to_impl(
                    assumption,
                    spec_input_nodes,
                    Bool(),
                    range_ctx,
                ),
            )
        )
        prior_assumes.append(assumption)

    spec_input_nodes[spec.spec_ast] = lowered_composite.inner_tree
    lowered_checks = tuple(
        (
            check,
            search_lower_spec_to_impl(
                check,
                spec_input_nodes,
                Bool(),
                spec.spec_ctx,
            ),
        )
        for check in spec.spec_ctx.checks
    )
    lowered_composite.set_impl_conditions(
        assumes=tuple(lowered_assumes),
        checks=lowered_checks,
    )
