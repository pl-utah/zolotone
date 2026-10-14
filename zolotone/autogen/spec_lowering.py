"""Lower validated specification expressions into implementation nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class _Component:
    contract: _SpecContract
    categories: tuple[type[SpecNode], ...]
    component: tp.Callable[..., Node]

    @classmethod
    def from_callable(cls, component: tp.Callable[..., Node]) -> _Component:
        contract = getattr(component, "_spec_contract", None)
        if not isinstance(contract, _SpecContract) or not callable(
            getattr(component, "_spec_fn", None)
        ):
            raise MissingError(
                f"Autogeneration component {component!r} does not expose "
                "a specification contract"
            )
        parameters = tuple(contract.signature.parameters.values())[:-1]
        categories = []
        for parameter in parameters:
            annotation = contract.annotations[parameter.name]
            category = _probe_input_category(annotation)
            if category is None:
                raise MissingError(
                    f"Autogeneration component {contract.display_name} has an "
                    f"unsupported input category for parameter {parameter.name!r} "
                    f"with annotation {annotation!r}"
                )
            categories.append(category)
        return cls(
            contract=contract, categories=tuple(categories), component=component
        )


class _Candidate(tp.NamedTuple):
    node: Node
    spec: SpecNode
    depth: int = 0
    cost: int = 0
    parents: tuple[_Candidate, ...] = ()
    operation: str = "source"
    exact: bool = True
    rank: int = -1
    lossy_count: int = 0
    steps: frozenset[tuple[int, int]] = frozenset()


MAX_SPEC_PROBES = 5_000
MAX_TYPED_SPEC_PROBES = 5_000
MAX_EXTENSION_BITS = 32
LUT_COST = 100


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
    components: tuple[_Component, ...],
) -> tuple[dict[tuple[int, tuple[SpecNode, ...]], SpecNode], frozenset[SpecNode]] | None:
    """Find relevant spec shapes; this does not establish typed reachability."""
    if any(not isinstance(expr, SpecNode) for expr in available):
        return None
    possible = set(available)

    matches: dict[tuple[int, tuple[SpecNode, ...]], SpecNode] = {}
    probed: set[tuple[int, tuple[SpecNode, ...]]] = set()
    while True:
        old_count = len(possible)
        for index, entry in enumerate(components):
            pools = [
                tuple(expr for expr in possible if isinstance(expr, category))
                for category in entry.categories
            ]
            for argument_specs in itertools.product(*pools):
                key = (index, argument_specs)
                if key in probed:
                    continue
                if len(probed) >= MAX_SPEC_PROBES:
                    return None
                probed.add(key)
                try:
                    result = entry.component._spec_fn(
                        *argument_specs,
                        SpecContext(f"autogen candidate {entry.contract.display_name}"),
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


def _initial_candidates(subexpressions, spec_input_nodes):
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
    return sources, source_states


def _typed_context(range_ctx: SpecContext, sources: list[_Candidate]) -> SpecContext:
    typed_ctx = range_ctx.copy()
    for source in sources:
        if isinstance(source.spec, RealExpr) and isinstance(source.node.dtype, (Q, UQ)):
            typed_ctx.assumes.append(
                _fixed_point_result_fits(source.spec, source.node.dtype, typed_ctx)
            )
    return typed_ctx


def search_lower_spec_to_impl(
    spec_ast: SpecNode,
    spec_input_nodes: dict[tp.Any, Node],
    return_annotation: object,
    range_ctx: SpecContext,
) -> Node:
    """Find the cheapest typed derivation of a specification expression."""
    from ..components import LOSSLESS_COMPONENTS
    from .missing import _suggest_missing_output_dtype

    components = tuple(
        _Component.from_callable(component) for component in LOSSLESS_COMPONENTS
    )
    subexpressions = _spec_subexpressions(spec_ast)
    sources, _ = _initial_candidates(subexpressions, spec_input_nodes)
    typed_ctx = _typed_context(range_ctx, sources)
    candidates: dict[tuple[SpecNode, DataType], _Candidate] = {
        (source.spec, source.node.dtype): source for source in sources
    }
    spec_matches = None
    component_attempts = {}
    lut_attempts = {}
    probe_count = 0
    errors = []

    initial_prefilter = _prefilter_component_matches(
        subexpressions,
        tuple(source.spec for source in sources),
        components,
    )
    prefilter_cache = {
        frozenset(source.spec for source in sources): initial_prefilter
    }
    root_known_without_missing = (
        initial_prefilter is None or spec_ast in initial_prefilter[1]
    )
    extensions_enabled = False

    def current_result() -> _Candidate | None:
        matching = (
            candidate for candidate in candidates.values()
            if _matches_result(candidate, spec_ast, return_annotation)
        )
        return min(
            matching,
            key=lambda candidate: (
                candidate.cost,
                candidate.lossy_count,
                candidate.node.dtype.total_bits(),
            ),
            default=None,
        )

    def add(candidate: _Candidate) -> bool:
        state = (candidate.spec, candidate.node.dtype)
        previous = candidates.get(state)
        if previous is not None and (
            previous.cost < candidate.cost
            or previous.cost == candidate.cost and (
                previous.lossy_count < candidate.lossy_count
                or previous.lossy_count == candidate.lossy_count
                and previous.rank <= candidate.rank
            )
        ):
            return False
        candidates[state] = candidate
        return True

    def known_pass() -> bool:
        nonlocal probe_count
        changed = False
        if spec_matches is None:
            work = (
                (index, None, None)
                for index in range(len(components))
            )
        else:
            work = (
                (index, argument_specs, matched)
                for (index, argument_specs), matched in spec_matches.items()
            )
        for index, argument_specs, known_match in work:
            entry = components[index]
            component = entry.component
            contract = entry.contract
            if not extensions_enabled and contract.display_name.split(" ", 1)[0].strip("'") in {
                "_uq_zero_extend", "_uq_fraction_extend",
                "_q_sign_extend", "_q_fraction_extend",
            }:
                continue
            parameters = tuple(contract.signature.parameters.values())[:-1]
            pools = [
                tuple(
                    candidate for candidate in candidates.values()
                    if (argument_specs is None or candidate.spec == argument_specs[position])
                    and _annotation_matches(
                        candidate.node.dtype, contract.annotations[parameter.name]
                    )
                    and isinstance(candidate.spec, entry.categories[position])
                )
                for position, parameter in enumerate(parameters)
            ]
            if any(not pool for pool in pools):
                continue
            for arguments in itertools.product(*pools):
                parent_steps = frozenset().union(
                    *(argument.steps for argument in arguments)
                )
                total_cost = 1 + sum(weight for _, weight in parent_steps)
                best = current_result()
                if best is not None and total_cost > best.cost:
                    continue
                key = (index, tuple((arg.spec, arg.node.dtype) for arg in arguments))
                if component_attempts.get(key, math.inf) <= total_cost:
                    continue
                component_attempts[key] = total_cost
                matched = known_match
                if matched is None:
                    if probe_count >= MAX_TYPED_SPEC_PROBES:
                        raise ZolotoneError(
                            f"Cannot lower {spec_ast!r}: component spec probe "
                            f"limit of {MAX_TYPED_SPEC_PROBES} was reached"
                        )
                    probe_count += 1
                    try:
                        result = component._spec_fn(
                            *(arg.spec for arg in arguments),
                            SpecContext(f"autogen candidate {contract.display_name}"),
                        )
                    except (TypeError, ValueError, NotImplementedError, OverflowError):
                        continue
                    matched = _matching_subexpression(result, subexpressions)
                if matched is None:
                    continue
                try:
                    node = component(*(arg.node for arg in arguments))
                except (TypeError, ValueError, NotImplementedError, OverflowError):
                    continue
                if (
                    isinstance(node.dtype, (Q, UQ))
                    and node.dtype.total_bits() > MAX_EXTENSION_BITS
                    and any(
                        arg.spec.identical(matched)
                        and isinstance(arg.node.dtype, (Q, UQ))
                        and node.dtype.total_bits() > arg.node.dtype.total_bits()
                        for arg in arguments
                    )
                ):
                    continue
                candidate = _Candidate(
                    node=node,
                    spec=matched,
                    depth=max((arg.depth for arg in arguments), default=0) + 1,
                    cost=total_cost,
                    parents=arguments,
                    operation=contract.display_name,
                    exact=all(arg.exact for arg in arguments),
                    rank=index,
                    lossy_count=sum(arg.lossy_count for arg in arguments),
                    steps=parent_steps | {(id(node), 1)},
                )
                try:
                    if not _candidate_output_range_fits(candidate, typed_ctx):
                        continue
                except (NotImplementedError, ZolotoneError):
                    continue
                changed |= add(candidate)
        return changed

    def missing_pass() -> bool:
        changed = False
        available = tuple(candidates.values())
        for expression in reversed(subexpressions):
            if not isinstance(expression, RealExpr) or isinstance(expression, RealLit):
                continue
            operands: list[RealExpr] = []
            for child in children(expression):
                if isinstance(child, RealExpr) and not isinstance(child, RealLit):
                    if not any(child.identical(existing) for existing in operands):
                        operands.append(child)
                elif not isinstance(child, (RealLit, BoolLit)):
                    operands = []
                    break
            if not operands:
                continue
            demanded_types: set[Q | UQ] = set()
            if expression.identical(spec_ast) and isinstance(return_annotation, (Q, UQ)):
                demanded_types.add(return_annotation)
            if spec_matches is not None:
                for (index, argument_specs), _ in spec_matches.items():
                    contract = components[index].contract
                    parameters = tuple(contract.signature.parameters.values())[:-1]
                    for argument_spec, parameter in zip(
                        argument_specs, parameters, strict=True
                    ):
                        if argument_spec.identical(expression):
                            annotation = contract.annotations[parameter.name]
                            if isinstance(annotation, (Q, UQ)):
                                demanded_types.add(annotation)
            existing = tuple(
                candidate for candidate in available
                if candidate.spec.identical(expression)
            )
            if (
                any(candidate.cost < LUT_COST for candidate in existing)
                and all(
                    any(candidate.node.dtype == dtype for candidate in existing)
                    for dtype in demanded_types
                )
                and (
                    not expression.identical(spec_ast)
                    or any(
                        _annotation_matches(candidate.node.dtype, return_annotation)
                        for candidate in existing
                    )
                )
            ):
                continue
            pools = [
                tuple(
                    candidate for candidate in available
                    if candidate.spec.identical(operand)
                    and isinstance(candidate.node.dtype, (Q, UQ))
                )
                for operand in operands
            ]
            if any(not pool for pool in pools):
                continue
            for arguments in itertools.product(*pools):
                parent_steps = frozenset().union(
                    *(arg.steps for arg in arguments)
                )
                total_cost = LUT_COST + sum(weight for _, weight in parent_steps)
                best = current_result()
                if best is not None and total_cost > best.cost:
                    continue
                output_types: set[Q | UQ] = {
                    tp.cast(Q | UQ, arg.node.dtype) for arg in arguments
                }
                output_types.update(demanded_types)
                suggested = None
                try:
                    suggested = _suggest_missing_output_dtype(
                        expression,
                        tp.cast(Q | UQ, arguments[0].node.dtype),
                        typed_ctx,
                    )
                    output_types.add(suggested)
                except (ZolotoneError, NotImplementedError):
                    pass
                if existing and min(candidate.cost for candidate in existing) < LUT_COST:
                    output_types = {
                        dtype for dtype in demanded_types
                        if all(candidate.node.dtype != dtype for candidate in existing)
                    }
                argument_states = tuple((arg.spec, arg.node.dtype) for arg in arguments)
                for output_dtype in sorted(
                    output_types,
                    key=lambda dtype: (
                        dtype != suggested,
                        dtype.total_bits(),
                        dtype.frac_bits,
                        type(dtype).__name__,
                    ),
                ):
                    if expression.identical(spec_ast) and not _annotation_matches(
                        output_dtype, return_annotation
                    ):
                        continue
                    state = (expression, output_dtype)
                    if state in candidates and candidates[state].cost <= total_cost:
                        continue
                    key = (expression, argument_states, output_dtype)
                    if lut_attempts.get(key, math.inf) <= total_cost:
                        continue
                    lut_attempts[key] = total_cost
                    try:
                        node = _build_missing_primitive(
                            expression,
                            tuple(operands),
                            tuple(arg.node for arg in arguments),
                            typed_ctx,
                            output_dtype=output_dtype,
                        )
                    except (MissingError, ZolotoneError, NotImplementedError) as exc:
                        errors.append(exc)
                        continue
                    changed |= add(_Candidate(
                        node=node,
                        spec=expression,
                        depth=max(arg.depth for arg in arguments) + 1,
                        cost=total_cost,
                        parents=arguments,
                        operation="generated_lut",
                        exact=getattr(node, "_autogen_lut_exact", False)
                        and all(arg.exact for arg in arguments),
                        rank=len(components),
                        lossy_count=sum(arg.lossy_count for arg in arguments)
                        + int(not getattr(node, "_autogen_lut_exact", False)),
                        steps=parent_steps | {(id(node), LUT_COST)},
                    ))
        return changed

    initial = current_result()
    if initial is not None:
        return initial.node
    while True:
        available_key = frozenset(candidate.spec for candidate in candidates.values())
        if available_key not in prefilter_cache:
            prefilter_cache[available_key] = _prefilter_component_matches(
                subexpressions,
                tuple(available_key),
                components,
            )
        prefilter = prefilter_cache[available_key]
        spec_matches = prefilter[0] if prefilter is not None else None
        possible_specs = prefilter[1] if prefilter is not None else None
        available_specs = {candidate.spec for candidate in candidates.values()}
        known_can_advance = (
            extensions_enabled
            or
            possible_specs is None
            or bool(possible_specs - available_specs)
            or spec_ast in available_specs and current_result() is None
        )
        if known_can_advance:
            while known_pass():
                pass
        result = current_result()
        exact_component_demands = any(
            isinstance(annotation, (Q, UQ))
            for entry in components
            for name, annotation in entry.contract.annotations.items()
            if name != "return"
        )
        if result is not None and result.cost <= 2:
            return result.node
        if missing_pass():
            continue
        if result is not None and (
            not root_known_without_missing
            or not isinstance(return_annotation, (Q, UQ))
            and not exact_component_demands
        ):
            return result.node
        if not extensions_enabled:
            if errors and not root_known_without_missing:
                raise errors[0]
            extensions_enabled = True
            continue
        if result is not None:
            return result.node
        if errors:
            raise errors[0]
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
