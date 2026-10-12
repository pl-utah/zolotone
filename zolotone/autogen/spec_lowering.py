"""Lower validated specification expressions into implementation nodes."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import itertools
import math
import typing as tp

from ..errors import MissingError, ZolotoneError
from ..rival import rival_range_analysis
from ..spec.spec_ast import (
    BoolExpr,
    BoolLit,
    BoolVar,
    Eq,
    FPExpr,
    Ge,
    Gt,
    Le,
    Lt,
    NotEq,
    RealExpr,
    RealLit,
    RealVar,
    SpecNode,
    children,
    substitute_spec_node,
)
from ..spec.spec_context import SpecContext
from ..types import Bool, DataType, Q, UQ, Value
from .missing import _create_missing_primitive as _build_missing_primitive
from ..ast.node import Node
from ..ast.nodes import (
    Composite,
    Const,
    Var,
    _annotation_matches,
    _SpecContract,
)
from .spec_validation import (
    _fixed_point_real_bounds,
    _fixed_point_result_fits,
    _prove_result_fits,
)

if tp.TYPE_CHECKING:
    from .autogen import Spec


class _Candidate(tp.NamedTuple):
    node: Node
    spec: SpecNode
    depth: int


MAX_SEARCH_DEPTH = 30
MAX_SPEC_PROBES = 5_000
MAX_TYPED_SPEC_PROBES = 5_000


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


def _probe_input_category(annotation: object) -> type[SpecNode] | None:
    if annotation is Bool or isinstance(annotation, Bool):
        return BoolExpr
    if annotation in (Q, UQ) or isinstance(annotation, (Q, UQ)):
        return RealExpr
    if annotation is DataType:  # in case of if statement, annotation is just any type
        return SpecNode
    if not (
        isinstance(annotation, DataType)
        or isinstance(annotation, type) and issubclass(annotation, DataType)
    ):
        return None
    try:
        dtype = annotation if isinstance(annotation, DataType) else annotation()
        sample = dtype.to_spec("probe", SpecContext("lowering input type"))
        return type(sample) if isinstance(sample, FPExpr) else None
    except Exception:
        return None


def _prefilter_component_matches(
    subexpressions: tuple[SpecNode, ...],
    available: tuple[SpecNode, ...],
    components: tuple[tp.Callable[..., Node], ...],
    contracts: list[_SpecContract],
    component_categories: list[tuple[type[SpecNode] | None, ...]],
) -> tuple[dict[tuple[int, tuple[SpecNode, ...]], SpecNode], frozenset[SpecNode]] | None:
    """Find relevant spec shapes; this does not establish typed reachability."""
    if any(not isinstance(expr, SpecNode) for expr in available):
        return None
    possible = set(available)
    entries = []
    for index, (component, contract, categories) in enumerate(
        zip(components, contracts, component_categories, strict=True)
    ):
        if any(category is None for category in categories):
            return None
        entries.append((index, component, contract, categories))

    matches: dict[tuple[int, tuple[SpecNode, ...]], SpecNode] = {}
    probed: set[tuple[int, tuple[SpecNode, ...]]] = set()
    while True:
        old_count = len(possible)
        for index, component, contract, categories in entries:
            pools = [
                tuple(expr for expr in possible if isinstance(expr, category))
                for category in categories
            ]
            for argument_specs in itertools.product(*pools):
                key = (index, argument_specs)
                if key in probed:
                    continue
                if len(probed) >= MAX_SPEC_PROBES:
                    return None
                probed.add(key)
                try:
                    result = component._spec_fn(
                        *argument_specs,
                        SpecContext(f"autogen candidate {contract.display_name}"),
                    )
                except (TypeError, ValueError, NotImplementedError, OverflowError):
                    continue
                except Exception:
                    return None
                matched = _matching_subexpression(result, subexpressions)
                if matched is not None:
                    matches[key] = matched
                    possible.add(matched)
        if len(possible) == old_count:
            return matches, frozenset(possible)


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
) -> bool:
    if not isinstance(candidate.spec, RealExpr):
        return True
    if not isinstance(candidate.node.dtype, (Q, UQ)):
        raise NotImplementedError(
            "Real-valued autogeneration candidate must have a Q or UQ "
            f"output format, got {candidate.node.dtype!r} from "
            f"{candidate.node.name!r}"
        )

    output_range = rival_range_analysis(candidate.spec, range_ctx)
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

    return _prove_result_fits(candidate.spec, candidate.node.dtype, range_ctx)


def _create_missing_primitive(
    missing: RealExpr,
    input_candidate: _Candidate,
    range_ctx: SpecContext,
) -> Node:
    """Compatibility wrapper for missing-component construction."""
    return _build_missing_primitive(
        missing,
        tp.cast(RealExpr, input_candidate.spec),
        input_candidate.node,
        range_ctx,
    )


def _missing_frontiers(
    subexpressions: tuple[SpecNode, ...],
    candidates: list[_Candidate],
) -> tp.Iterator[tuple[RealExpr, _Candidate]]:
    """Yield expressions whose entire nonliteral boundary is one typed operand."""
    for expression in reversed(subexpressions):
        if not isinstance(expression, RealExpr) or isinstance(expression, RealLit):
            continue
        nonliteral = tuple(
            child for child in children(expression)
            if not isinstance(child, (RealLit, BoolLit))
        )
        if not nonliteral or not isinstance(nonliteral[0], RealExpr):
            continue
        input_expr = nonliteral[0]
        if any(not child.identical(input_expr) for child in nonliteral[1:]):
            continue
        for candidate in candidates:
            if (
                isinstance(candidate.node.dtype, (Q, UQ))
                and candidate.spec.identical(input_expr)
            ):
                yield expression, candidate


def search_lower_spec_to_impl(
    spec_ast: SpecNode,
    spec_input_nodes: dict[tp.Any, Node],
    return_annotation: object,
    range_ctx: SpecContext,
) -> Node:
    """Lower with known components, generating one typed gap at a time."""
    from ..components import LOSSLESS_COMPONENTS

    components = tuple(LOSSLESS_COMPONENTS)
    component_contracts = []
    component_categories = []
    for component in components:
        contract = getattr(component, "_spec_contract", None)
        if not isinstance(contract, _SpecContract) or not callable(
            getattr(component, "_spec_fn", None)
        ):
            raise MissingError(
                f"Autogeneration component {component!r} does not expose "
                "a specification contract"
            )
        component_contracts.append(contract)
        parameters = tuple(contract.signature.parameters.values())[:-1]
        component_categories.append(
            tuple(
                _probe_input_category(contract.annotations[parameter.name])
                for parameter in parameters
            )
        )

    # Turn leaves into candidates
    subexpressions = _spec_subexpressions(spec_ast)
    sources = [
        _Candidate(node=node, spec=expr, depth=0)
        for expr, node in spec_input_nodes.items()
    ]
    source_states = {(candidate.spec, candidate.node.dtype) for candidate in sources}
    for expr in subexpressions:
        if isinstance(expr, RealLit):
            literal = _Candidate(Const(_exact_fixed_point_value(expr.value)), expr, 0)
        elif isinstance(expr, BoolLit):
            literal = _Candidate(Const(Bool().from_bits(int(expr.value))), expr, 0)
        elif isinstance(expr, (BoolVar, RealVar)):
            if expr not in spec_input_nodes:
                raise MissingError(f"Undeclared variable in specification: {expr}")
            continue
        else:
            continue
        state = (literal.spec, literal.node.dtype)
        if state not in source_states:
            source_states.add(state)
            sources.append(literal)

    known_depth_exhausted = False

    def saturate(typed_ctx: SpecContext) -> tuple[_Candidate | None, list[_Candidate]]:
        nonlocal known_depth_exhausted
        candidates = list(sources)
        probe_count = 0
        states = {(candidate.spec, candidate.node.dtype) for candidate in candidates}
        for candidate in candidates:
            if _matches_result(candidate, spec_ast, return_annotation):
                return candidate, candidates

        prefilter = _prefilter_component_matches(
            subexpressions,
            tuple(candidate.spec for candidate in candidates),
            components,
            component_contracts,
            component_categories,
        )
        spec_matches = prefilter[0] if prefilter is not None else None
        relevant_components = (
            {index for index, _ in spec_matches} if spec_matches is not None else None
        )
        possible_specs = prefilter[1] if prefilter is not None else None
        # If the root is structurally impossible with known components, only
        # build enough typed nodes to expose every possible missing input.
        if (
            possible_specs is not None
            and spec_ast not in possible_specs
            and possible_specs.issubset(candidate.spec for candidate in candidates)
        ):
            return None, candidates

        for depth in range(1, MAX_SEARCH_DEPTH + 1):
            generated: list[_Candidate] = []
            for index, (component, contract) in enumerate(
                zip(components, component_contracts, strict=True)
            ):
                if relevant_components is not None and index not in relevant_components:
                    continue
                parameters = tuple(contract.signature.parameters.values())[:-1]
                pools = [
                    tuple(
                        candidate for candidate in candidates
                        if _annotation_matches(
                            candidate.node.dtype,
                            contract.annotations[parameter.name],
                        )
                        and (
                            component_categories[index][position] is None
                            or isinstance(
                                candidate.spec, component_categories[index][position]
                            )
                        )
                    )
                    for position, parameter in enumerate(parameters)
                ]
                if any(not pool for pool in pools):
                    continue
                for arguments in itertools.product(*pools):
                    if max((arg.depth for arg in arguments), default=0) != depth - 1:
                        continue
                    argument_specs = tuple(arg.spec for arg in arguments)
                    if spec_matches is not None:
                        matched = spec_matches.get((index, argument_specs))
                    else:
                        if probe_count >= MAX_TYPED_SPEC_PROBES:
                            # TODO: we can generate missing lossy component here. it can be a truncation really
                            raise ZolotoneError(
                                f"Cannot lower {spec_ast!r}: component spec probe "
                                f"limit of {MAX_TYPED_SPEC_PROBES} was reached"
                            )
                        probe_count += 1
                        try:
                            result = component._spec_fn(
                                *argument_specs,
                                SpecContext(f"autogen candidate {contract.display_name}"),
                            )
                        except (TypeError, ValueError, NotImplementedError, OverflowError):
                            result = None
                        matched = _matching_subexpression(result, subexpressions)
                    if matched is None:
                        continue
                    try:
                        node = component(*(arg.node for arg in arguments))
                    except (TypeError, ValueError, NotImplementedError, OverflowError):
                        continue
                    candidate = _Candidate(
                        node=node,
                        spec=matched,
                        depth=depth,
                    )
                    try:
                        fits = _candidate_output_range_fits(candidate, typed_ctx)
                    except (NotImplementedError, ZolotoneError):
                        fits = False
                    if not fits:
                        continue
                    state = (matched, node.dtype)
                    if state in states:
                        continue
                    states.add(state)
                    if _matches_result(candidate, spec_ast, return_annotation):
                        return candidate, candidates + generated
                    generated.append(candidate)
            if not generated:
                return None, candidates
            candidates.extend(generated)
            if (
                possible_specs is not None
                and spec_ast not in possible_specs
                and possible_specs.issubset(candidate.spec for candidate in candidates)
            ):
                return None, candidates
        known_depth_exhausted = True
        return None, candidates

    generated_specs: set[SpecNode] = set()
    while True:
        # Source descriptors, including newly generated nodes, are valid range
        # facts even when this entry point is called outside Autogenerate.
        typed_ctx = range_ctx.copy()
        for source in sources:
            if isinstance(source.spec, RealExpr) and isinstance(source.node.dtype, (Q, UQ)):
                typed_ctx.assumes.append(
                    _fixed_point_result_fits(source.spec, source.node.dtype, typed_ctx)
                )

        result, candidates = saturate(typed_ctx)
        if result is not None:
            return result.node

        # Missing component
        errors: list[Exception] = []
        created = False
        for expression, operand in _missing_frontiers(subexpressions, candidates):
            if expression in generated_specs:
                continue
            if expression is not spec_ast and any(
                candidate.spec.identical(expression) for candidate in candidates
            ):
                continue
            try:
                node = _create_missing_primitive(expression, operand, typed_ctx)
            except (MissingError, ZolotoneError, NotImplementedError) as exc:
                errors.append(exc)
                continue
            generated_specs.add(expression)
            state = (expression, node.dtype)
            if state in source_states:
                continue
            source_states.add(state)
            sources.append(_Candidate(node=node, spec=expression, depth=0))
            created = True
            break
        if created:
            continue
        if errors:
            raise errors[0]
        if known_depth_exhausted:
            raise ZolotoneError(
                f"Cannot lower {spec_ast!r}: known-component search reached "
                f"the maximum depth of {MAX_SEARCH_DEPTH}"
            )
        raise ZolotoneError(
            f"Cannot lower specification {spec_ast!r} to {return_annotation!r}; "
            "no typed component or supported missing primitive was found"
        )


def _tighten_literal_comparison(
    comparison: Lt | Le | Gt | Ge | Eq | NotEq,
    spec_input_nodes: dict[tp.Any, Node],
) -> BoolExpr | None:
    """Tighten a var–literal comparison using its representable input values."""
    lhs = comparison.lhs.constant_fold()
    rhs = comparison.rhs.constant_fold()
    operator = type(comparison)
    if isinstance(lhs, RealLit):
        lhs, rhs = rhs, lhs
        operator = {Lt: Gt, Le: Ge, Gt: Lt, Ge: Le, Eq: Eq, NotEq: NotEq}[operator]
    input_node = spec_input_nodes.get(lhs)
    if (
        not isinstance(lhs, RealVar)
        or not isinstance(rhs, RealLit)
        or input_node is None
        or not isinstance(input_node.dtype, (Q, UQ))
    ):
        return None

    scale = 1 << input_node.dtype.frac_bits
    scaled = Fraction(rhs.value) * scale
    if operator in (Eq, NotEq):
        signed = isinstance(input_node.dtype, Q)
        magnitude = 1 << (input_node.dtype.total_bits() - int(signed))
        minimum = -magnitude if signed else 0
        if scaled.denominator != 1 or not minimum <= scaled < magnitude:
            return BoolLit(operator is NotEq)
        return None
    if operator is Ge:
        bound = math.ceil(scaled)
    elif operator is Gt:
        bound = math.floor(scaled) + 1
    elif operator is Le:
        bound = math.floor(scaled)
    else:
        bound = math.ceil(scaled) - 1

    exact = Fraction(bound, scale)
    if exact.denominator == 1:
        literal = RealLit(exact.numerator)
    elif Fraction(float(exact)) == exact:
        literal = RealLit(float(exact))
    else:
        # Keep bounds beyond float precision exact as a dyadic expression.
        literal = RealLit(bound) * (RealLit(2) ** RealLit(-input_node.dtype.frac_bits))
    return Ge(lhs, literal) if operator in (Gt, Ge) else Le(lhs, literal)


def _make_conditions_bit_precise(
    conditions: tp.Iterable[BoolExpr],
    spec_inputs: tuple[tp.Any, ...],
    contract: _SpecContract,
    range_ctx: SpecContext,
) -> tuple[BoolExpr, ...]:
    """Tighten var–literal bounds and lower other strict comparisons."""
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
        comparisons = []
        while pending:
            node = pending.pop()
            pending.extend(children(node))
            if isinstance(node, (Lt, Le, Gt, Ge, Eq, NotEq)):
                comparisons.append(node)

        rewritten = condition
        for comparison in reversed(comparisons):
            precise = _tighten_literal_comparison(comparison, spec_input_nodes)
            if precise is not None:
                rewritten = substitute_spec_node(rewritten, comparison, precise)
                if isinstance(precise, BoolLit):
                    rewritten = rewritten.constant_fold()
                continue
            if not isinstance(comparison, (Lt, Gt)):
                continue

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
    """Tighten assumptions and checks for fixed-point semantics.

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
