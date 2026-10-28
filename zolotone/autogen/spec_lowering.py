"""Lower validated specification expressions into implementation nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import itertools
import math
import typing as tp

from ..errors import MissingError, ZolotoneError, _SpecProbeLimitExceeded
from ..rival.analysis import analyze_typed_domain
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
    Round,
    SpecNode,
    children,
    substitute_spec_node,
)
from ..spec.spec_context import SpecContext
from ..types import Bool, DataType, Q, UQ, Value
from .missing import _build_missing_primitive
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
    input_categories: tuple[type[SpecNode], ...]
    node: tp.Callable[..., Node]

    @property
    def input_types(self):
        parameters = tuple(self.contract.signature.parameters.values())[:-1]
        return tuple(self.contract.annotations[parameter.name] for parameter in parameters)

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
        input_categories = []
        for parameter in parameters:
            annotation = contract.annotations[parameter.name]
            category = _probe_input_category(annotation)
            if category is None:
                raise MissingError(
                    f"Autogeneration component {contract.display_name} has an "
                    f"unsupported input category for parameter {parameter.name!r} "
                    f"with annotation {annotation!r}"
                )
            input_categories.append(category)
        return cls(
            contract=contract, input_categories=tuple(input_categories), node=component
        )


class _Candidate(tp.NamedTuple):
    node: Node
    spec: SpecNode
    cost: int = 0
    steps: frozenset[tuple[int, int]] = frozenset()

    @classmethod
    def from_operation(cls, node, spec, arguments, operation_cost):
        steps = _shared_steps(arguments) | {(id(node), operation_cost)}
        return cls(node, spec, sum(cost for _, cost in steps), steps)


def _shared_steps(arguments):
    """Charge a shared implementation node once, even when reused by inputs."""
    return frozenset().union(*(argument.steps for argument in arguments))


def _operation_cost(arguments, operation_cost):
    return operation_cost + sum(cost for _, cost in _shared_steps(arguments))


MAX_SPEC_PROBES = 5_000
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


def _candidate_output_range_fits(
    candidate: _Candidate,
    range_ctx: SpecContext,
    input_dtypes: dict[SpecNode, DataType] | None = None,
    output_bounds: tuple[Fraction, Fraction] | None = None,
) -> bool:
    if not isinstance(candidate.spec, RealExpr):
        return True
    if not isinstance(candidate.node.dtype, (Q, UQ)):
        raise NotImplementedError(
            "Real-valued autogeneration candidate must have a Q or UQ "
            f"output format, got {candidate.node.dtype!r} from "
            f"{candidate.node.name!r}"
        )

    if output_bounds is None:
        output_bounds = analyze_typed_domain(
            candidate.spec, range_ctx, input_dtypes or {}, candidate.node.dtype,
        )
    lower, upper = output_bounds
    dtype_lower, dtype_upper = _fixed_point_real_bounds(candidate.node.dtype)
    if lower >= dtype_lower and upper <= dtype_upper:
        return True

    return _prove_result_fits(candidate.spec, candidate.node.dtype, range_ctx, input_dtypes or {})


def _initial_candidates(subexpressions, spec_input_nodes):
    sources = [
        _Candidate(node=node, spec=expr)
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
    return sources


def _typed_context(range_ctx: SpecContext, sources: list[_Candidate]) -> SpecContext:
    typed_ctx = range_ctx.copy()
    for source in sources:
        if isinstance(source.spec, RealExpr) and isinstance(source.node.dtype, (Q, UQ)):
            typed_ctx.assumes.append(
                _fixed_point_result_fits(source.spec, source.node.dtype, typed_ctx)
            )
    return typed_ctx


class _LoweringSearch:
    """One candidate table, improved by known components and generated LUTs."""

    def __init__(self, spec_ast, spec_input_nodes, return_annotation, range_ctx, output_bounds=None):
        from ..components import LOSSLESS_COMPONENTS

        self.root = spec_ast
        self.output_bounds = output_bounds
        self.input_dtypes = {expr: node.dtype for expr, node in spec_input_nodes.items()}
        self.return_annotation = return_annotation
        self.expressions = _spec_subexpressions(spec_ast)
        self.components = tuple(
            _Component.from_callable(component) for component in LOSSLESS_COMPONENTS
        )
        # self.candidates = {
        #     expr: {
        #         dtype: candidate,
        #     },
        # }
        self.candidates = {}
        # Final adaptations never enter the operand pools for further search.
        self.result = None

        # Constants/Variables
        sources = _initial_candidates(self.expressions, spec_input_nodes)
        self.ctx = _typed_context(range_ctx, sources)
        for source in sources:
            self.add(source)

        # Cache both matches and misses across all format variants.
        self.component_matches = {}
        self.component_attempts = {}
        self.lut_attempts = {}
        self.suggested_lut_types = {}
        self.range_checks = {}
        self.errors = []

    def run(self) -> Node:
        # Run up until a fixed point
        while True:
            try:
                changed = self.add_component_candidates()
            except _SpecProbeLimitExceeded:
                if self.result is None:
                    raise
                break
            changed |= self.add_lut_candidates()
            if not changed:
                break

        result = self.result
        if result is not None:
            return result.node
        if (
            isinstance(self.root, Round)
            and isinstance(self.return_annotation, (Q, UQ))
            and self.return_annotation.frac_bits < self.root.frac_bits
        ):
            raise ZolotoneError(
                "Requested output cannot discard the spec's fractional precision; "
                "add explicit rounding to the requested precision"
            )
        if self.errors:
            raise self.errors[0]
        raise ZolotoneError(
            f"Cannot lower specification {self.root!r} to "
            f"{self.return_annotation!r}; no typed component or supported "
            "missing primitive was found. LUT synthesis requires explicit "
            "ctx.rne, ctx.rtz, or ctx.raz with frac_bits."
        )

    def consider_result(self, candidate):
        if not candidate.spec.identical(self.root):
            return
        if not _annotation_matches(candidate.node.dtype, self.return_annotation):
            source = candidate.node.dtype
            target = self.return_annotation
            if not (
                isinstance(source, (Q, UQ))
                and type(target) is type(source)
                and target.int_bits >= source.int_bits
                and target.frac_bits >= source.frac_bits
            ):
                return
            if not self.worth_trying(candidate.cost + 1):
                return
            from ..components.Q import q_resize
            from ..components.UQ import uq_resize

            resize = q_resize if isinstance(source, Q) else uq_resize
            node = resize(candidate.node, target.int_bits, target.frac_bits)
            candidate = _Candidate.from_operation(node, self.root, (candidate,), 1)
        if self.result is None or self.result_rank(candidate) < self.result_rank(self.result):
            self.result = candidate

    def result_rank(self, candidate):
        """Generic fixed-point outputs minimize width, then computation cost."""
        if self.return_annotation in (Q, UQ):
            return candidate.node.dtype.total_bits(), candidate.cost
        return (candidate.cost,)

    # Add candidate if it is cheaper than previous one
    def add(self, candidate):
        formats = self.candidates.setdefault(candidate.spec, {})
        previous = formats.get(candidate.node.dtype)
        if previous is not None and previous.cost <= candidate.cost:
            return False
        formats[candidate.node.dtype] = candidate
        self.consider_result(candidate)
        return True

    def worth_trying(self, cost, *, operation_cost=1):
        # Free shrinking is always eligible; cost cannot bound width-first results.
        if operation_cost == 0 or self.return_annotation in (Q, UQ):
            return True
        result = self.result
        return result is None or cost < result.cost

    def match_component(self, index, argument_specs):
        key = (index, argument_specs)
        if key not in self.component_matches:
            if len(self.component_matches) >= MAX_SPEC_PROBES:
                raise _SpecProbeLimitExceeded(
                    f"Cannot lower {self.root!r}: component spec probe "
                    f"limit of {MAX_SPEC_PROBES} was reached"
                )
            component = self.components[index]
            try:
                expression = component.node._spec_fn(
                    *argument_specs,
                    SpecContext(f"autogen candidate {component.contract.display_name}"),
                )
            except (TypeError, ValueError, NotImplementedError, OverflowError):
                expression = None
            matched = _matching_subexpression(expression, self.expressions)
            self.component_matches[key] = matched

        return self.component_matches[key]

    # Checks that we do not widen formats unless it is a final root
    def allows_format(self, node, expression, arguments):
        """Leave same-format widening to final adaptation, not operand search."""
        if not isinstance(node.dtype, (Q, UQ)):
            return True
        for arg in arguments:
            if (
                arg.spec != expression
                or type(arg.node.dtype) is not type(node.dtype)
                or node.dtype.total_bits() <= arg.node.dtype.total_bits()
            ):
                continue
            return False
        return True

    def fits(self, candidate):
        key = (candidate.spec, candidate.node.dtype)
        if key not in self.range_checks:
            try:
                bounds = self.output_bounds if candidate.spec.identical(self.root) else None
                self.range_checks[key] = _candidate_output_range_fits(
                    candidate, self.ctx, self.input_dtypes, bounds,
                )
            except (NotImplementedError, ZolotoneError):
                self.range_checks[key] = False
        return self.range_checks[key]

    def add_component_candidates(self):
        from ..components import q_truncate_msb, uq_truncate_msb

        changed = False
        expressions = tuple(self.candidates.keys())  # Resolved expressions
        for index, component in enumerate(self.components):
            operation_cost = (
                0 if component.node is q_truncate_msb or component.node is uq_truncate_msb else 1
            )
            # Select expressions that can be inputs to new component
            expression_pools = [
                tuple(
                    expression for expression in expressions
                    if isinstance(expression, category)
                )
                for category in component.input_categories
            ]
            for argument_specs in itertools.product(*expression_pools):
                # Try to see whether component implements some subexpressions
                matched = self.match_component(index, argument_specs)
                if matched is None:
                    continue
                # Select candidates which dtypes do match component input types
                pools = [
                    tuple(
                        candidate
                        for candidate in self.candidates[expression].values()
                        if _annotation_matches(candidate.node.dtype, annotation)
                    )
                    for expression, annotation in zip(
                        argument_specs, component.input_types, strict=True
                    )
                ]
                for arguments in itertools.product(*pools):
                    cost = _operation_cost(arguments, operation_cost)
                    if not self.worth_trying(cost, operation_cost=operation_cost):
                        continue
                    key = (index, tuple((arg.spec, arg.node.dtype) for arg in arguments))
                    if self.component_attempts.get(key, math.inf) <= cost:
                        continue
                    self.component_attempts[key] = cost
                    try:
                        node = component.node(*(arg.node for arg in arguments))
                    except (TypeError, ValueError, NotImplementedError, OverflowError):
                        continue
                    if not self.allows_format(node, matched, arguments):
                        continue
                    candidate = _Candidate.from_operation(
                        node, matched, arguments, operation_cost=operation_cost
                    )
                    if self.fits(candidate):
                        changed |= self.add(candidate)
        return changed

    def lut_output_type(self, expression):
        """Honor the spec's explicit rounding precision; infer integer width."""
        from .missing import _suggest_missing_output_dtype

        key = expression
        if key not in self.suggested_lut_types:
            try:
                precision_dtype = UQ(1, expression.frac_bits)
                self.suggested_lut_types[key] = _suggest_missing_output_dtype(
                    expression, precision_dtype, self.ctx, self.input_dtypes
                )
            except (ZolotoneError, NotImplementedError) as exc:
                self.errors.append(exc)
                self.suggested_lut_types[key] = None
        return self.suggested_lut_types[key]

    def add_lut_candidates(self):
        changed = False
        # Consider whole regions before their children so a cheap fused LUT can
        # prune more expensive chains. Components still compete by total cost.
        for expression in self.expressions:
            if not isinstance(expression, Round):
                continue
            # Does it worth trying a LUT
            existing = self.candidates.get(expression, {})
            if any(candidate.cost < LUT_COST for candidate in existing.values()):
                continue
            
            pool = sorted(
                (
                    candidate
                    for operand in lut_inputs(expression)
                    for candidate in self.candidates.get(operand, {}).values()
                    if isinstance(candidate.node.dtype, (Q, UQ))
                ),
                key=lambda candidate: candidate.cost,
            )
            for argument in pool:
                cost = argument.cost + LUT_COST
                if not self.worth_trying(cost):
                    continue
                dtype = self.lut_output_type(expression)
                if dtype is None:
                    continue
                existing = self.candidates.get(expression, {}).get(dtype)
                # Does it worth trying a LUT
                if existing is not None and existing.cost <= cost:
                    continue
                key = (
                    expression,
                    (argument.spec, argument.node.dtype),
                    dtype,
                )
                if self.lut_attempts.get(key, math.inf) <= cost:
                    continue
                self.lut_attempts[key] = cost
                try:
                    node = _build_missing_primitive(
                        expression, argument.spec, argument.node,
                        self.ctx, output_dtype=dtype, input_dtypes=self.input_dtypes,
                    )
                except (MissingError, ZolotoneError, NotImplementedError) as exc:
                    self.errors.append(exc)
                    continue
                changed |= self.add(
                    _Candidate.from_operation(node, expression, (argument,), LUT_COST)
                )
        return changed


def lut_inputs(expression):
    """Find single-input cuts, ordered from largest to smallest LUT region."""
    if not isinstance(expression, RealExpr) or isinstance(expression, RealLit):
        return ()

    cuts = {}

    def visit(node):
        if isinstance(node, (RealLit, BoolLit)):
            return None
        if node not in cuts:
            node_children = children(node)
            dependent_children = [
                child_cuts
                for child in node_children
                if (child_cuts := visit(child)) is not None
            ]
            if node_children and not dependent_children:
                cuts[node] = None
                return None
            # A boundary must account for every nonliteral dependency, including
            # repeated occurrences. Independent inputs have no common cut.
            common = (
                set.intersection(*dependent_children) if dependent_children else set()
            )
            if isinstance(node, RealExpr):
                common.add(node)
            cuts[node] = common
        return cuts[node]

    boundaries = visit(expression) or set()
    return tuple(
        node for node in reversed(_spec_subexpressions(expression)[1:])
        if node in boundaries
    )


def search_lower_spec_to_impl(
    spec_ast: SpecNode,
    spec_input_nodes: dict[tp.Any, Node],
    return_annotation: object,
    range_ctx: SpecContext,
    output_bounds: tuple[Fraction, Fraction] | None = None,
) -> Node:
    """Find the cheapest typed derivation of a specification expression."""
    return _LoweringSearch(
        spec_ast, spec_input_nodes, return_annotation, range_ctx, output_bounds
    ).run()


def _tighten_literal_comparison(
    comparison: Lt | Le | Gt | Ge | Eq | NotEq,
    spec_input_nodes: dict[tp.Any, Node],
) -> BoolExpr | None:
    """Tighten a var–literal comparison using its representable input values."""
    lhs = comparison.lhs._constant_fold()
    rhs = comparison.rhs._constant_fold()
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
                    rewritten = rewritten._constant_fold()
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
                precise._constant_fold(),
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

    @Composite(name=spec.name, spec=spec.to_function())
    def generated_impl(*impl_inputs: Node) -> Node:
        spec_input_nodes = dict(
            zip(spec.spec_inputs, impl_inputs, strict=True)
        )
        return search_lower_spec_to_impl(
            spec.spec_ast,
            spec_input_nodes,
            spec.return_annotation,
            spec.spec_ctx,
            output_bounds=spec.output_bounds,
        )

    input_parameters = list(spec.contract.signature.parameters.values())[:-1]
    impl_inputs = [
        Var(
            name=parameter.name,
            dtype=spec.annotations[parameter.name],
        )
        for parameter in input_parameters
    ]
    result = generated_impl(*impl_inputs)
    result.rounding_choice_count = len(spec.rounding_choices)
    result._fingerprint_cache.clear()
    return result


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
