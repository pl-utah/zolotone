import itertools
import math
import typing as tp

from ..spec.spec_ast import (
    BoolExpr,
    BoolLit,
    BoolVar,
    RealLit,
    RealExpr,
    RealVar,
    SpecNode,
    children,
)
from ..spec.spec_context import SpecContext
from ..rival import rival_range_analysis
from ..solver import check_equivalence
from ..types import Bool, DataType, Q, UQ
from .node import Node
from .nodes import (
    Composite,
    Const,
    Var,
    _annotation_matches,
    _build_spec_contract,
    _SpecContract,
)


class _Candidate(tp.NamedTuple):
    node: Node
    spec: SpecNode
    depth: int


MAX_SEARCH_DEPTH = 30
FEASIBILITY_SCHEDULE = [
    {"tool": "simplify"},
    {"tool": "z3", "timeout_ms": 10_000},
]


def reject_untyped_inputs(contract: _SpecContract):
    input_parameters = list(contract.signature.parameters.values())[:-1]
    for parameter in input_parameters:
        annotation = contract.annotations[parameter.name]
        if not isinstance(annotation, DataType):
            raise TypeError(
                f"Autogenerate specification {contract.display_name} input "
                f"parameter {parameter.name!r} must have an exact DataType "
                f"descriptor, got {annotation!r}"
            )


def get_spec_ast(
    spec: tp.Callable[..., tp.Any],
    contract: _SpecContract,
) -> tuple[SpecNode, tuple[tp.Any, ...], SpecContext]:
    ctx = SpecContext(contract.display_name)
    input_parameters = list(contract.signature.parameters.values())[:-1]
    spec_inputs = []
    for parameter in input_parameters:
        dtype = contract.annotations[parameter.name]
        spec_input = dtype.to_spec(
            name=parameter.name,
            ctx=ctx,
        )
        spec_inputs.append(spec_input)

    spec_ast = spec(*spec_inputs, ctx)
    return spec_ast, tuple(spec_inputs), ctx


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
    return (spec_ast >= ctx.real_val(min_bound)) & (
        spec_ast <= ctx.real_val(max_bound)
    )


def _prove_result_fits(ctx: SpecContext, result_fits: BoolExpr) -> bool:
    range_ctx = ctx.copy(checks=[result_fits])
    status, _proof_trace = check_equivalence(range_ctx, schedule=FEASIBILITY_SCHEDULE)
    return status == "unsat"


# TODO: this method is mostly for checking range analysis now, to be deleted later
def output_format_suggestion_with_search(
    spec_ast: RealExpr,
    return_annotation: Q | UQ,
    ctx: SpecContext,
) -> Q | UQ | None:
    dtype_type = type(return_annotation)

    def candidate(int_bits: int) -> Q | UQ:
        return dtype_type(int_bits, return_annotation.frac_bits)

    # As few integer bits as possible
    min_int_bits = max(0, 1 - return_annotation.frac_bits)
    max_int_bits = return_annotation.int_bits + 32
    widest = candidate(max_int_bits)

    if not _prove_result_fits(ctx, _fixed_point_result_fits(spec_ast, widest, ctx)):
        # If the result cannot be stored even with a wider UQ output format - try Q
        if isinstance(return_annotation, UQ):
            signed = Q(max(0, 1 - return_annotation.frac_bits), return_annotation.frac_bits)
            return output_format_debugging(spec_ast, signed, ctx)
        return None

    low = min_int_bits
    high = max_int_bits
    while low < high:
        middle = (low + high) // 2
        middle_dtype = candidate(middle)
        if _prove_result_fits(ctx, _fixed_point_result_fits(spec_ast, middle_dtype, ctx)):
            high = middle
        else:
            low = middle + 1
    return candidate(low)

def output_format_suggestion_with_range_analysis(
    spec_ast: RealExpr,
    return_annotation: Q | UQ,
    ctx: SpecContext,
) -> Q | UQ:
    output_range = rival_range_analysis(spec_ast, ctx)
    if output_range is None:
        raise RuntimeError("Could not obtain output range for:", str(spec_ast))

    lower, upper = output_range
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise RuntimeError(
            "Could not obtain finite output range, got:",
            output_range,
            "for",
            spec_ast,
        )

    frac_bits = return_annotation.frac_bits
    scale = 1 << frac_bits
    unsigned = _prove_result_fits(ctx, spec_ast >= ctx.zero())
    # It should be an unsigned fixed-point
    if unsigned:
        required_raw = max(0, math.ceil(upper * scale))
        total_bits = max(1, frac_bits, required_raw.bit_length())
        int_bits = total_bits - frac_bits
        return UQ(int_bits, frac_bits)
    # It clearly should be a signed fixed-point
    else:
        lower_raw = math.floor(lower * scale)
        upper_raw = math.ceil(upper * scale)
        required_magnitude = max(1, -lower_raw, upper_raw + 1)
        magnitude_bits = (required_magnitude - 1).bit_length()
        total_bits = max(1, frac_bits, magnitude_bits + 1)
        int_bits = total_bits - frac_bits
        return Q(int_bits, frac_bits)


def output_format_debugging(
    spec_ast: RealExpr,
    return_annotation: object,
    ctx: SpecContext,
) -> object | None:
    # TODO: provide a counterexample
    if return_annotation is UQ:
        if not _prove_result_fits(ctx, spec_ast >= ctx.zero()):
            return Q
        return UQ
    elif return_annotation is Q:
        if _prove_result_fits(ctx, spec_ast >= ctx.zero()):
            return UQ
        return Q
    elif isinstance(return_annotation, (Q, UQ)):
        dtype1 = output_format_suggestion_with_range_analysis(spec_ast, return_annotation, ctx)
        dtype2 = output_format_suggestion_with_search(spec_ast, return_annotation, ctx)
        if not dtype1 == dtype2:
            raise AssertionError("derived dtypes are not equal! ranges: " + str(dtype1) + ", search: " + str(dtype2))
        if not _prove_result_fits(ctx, _fixed_point_result_fits(spec_ast, dtype1, ctx)):
            raise AssertionError("That's a crime, derived dtype does not fit the result")
        return dtype1
    else:
        raise NotImplementedError("Output range debugging is not implemented for", return_annotation)


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
    # Output checks
    if return_annotation is Bool or isinstance(return_annotation, Bool):
        if not isinstance(spec_ast, BoolExpr):
            raise TypeError(
                f"Specification returning {return_annotation!r} must produce "
                f"a Boolean expression, got {type(spec_ast).__name__}"
            )
        # At this point there is nothing to check
        return

    elif return_annotation in (Q, UQ) or isinstance(return_annotation, (Q, UQ)):
        if not isinstance(spec_ast, RealExpr):
            raise TypeError(
                f"Specification returning {return_annotation!r} must produce "
                f"a real expression, got {type(spec_ast).__name__}"
            )
        # If output is just Q - there is nothing to check for feasibility really
        if return_annotation is Q:
            return
        # If output is UQ - we can check that output is non-negative
        if return_annotation is UQ:
            result_fits = spec_ast >= ctx.zero()
        # If it is UQ/Q with bit-widths - we can check that range covers all outputs
        else:
            result_fits = _fixed_point_result_fits(
                spec_ast,
                return_annotation,
                ctx,
            )
        
        if not _prove_result_fits(ctx, result_fits):
            suggestion = output_format_debugging(
                spec_ast,
                return_annotation,
                ctx,
            )
            message = f"Specification result range does not fit {return_annotation!r}"
            if suggestion is not None:
                message += f"; try {_format_output_annotation(suggestion)} as the output format instead"
            else:
                message += "; could not find a fixed-point format that would fit the range"
            raise TypeError(message)
        return
    else:
        raise NotImplementedError(
            "Not supporting",
            return_annotation,
            "output format for feasibility check",
        )


def search_lower_spec_to_impl(
    spec_ast: SpecNode,
    spec_input_nodes: dict[tp.Any, Node],
    return_annotation: object,
) -> Node:
    from ..components import LOSSLESS_COMPONENTS

    # Lowering leaves first
    subexpressions = _spec_subexpressions(spec_ast)
    candidates = [
        _Candidate(node=node, spec=expr, depth=0)
        for expr, node in spec_input_nodes.items()
    ]
    states = {(candidate.spec, candidate.node.dtype) for candidate in candidates}

    for expr in subexpressions:
        if isinstance(expr, RealLit):
            try:
                literal = _Candidate(
                    node=Const(UQ.from_int(expr.value)),
                    spec=expr,
                    depth=0,
                )
            # value is negative
            except ValueError:
                literal = _Candidate(
                    node=Const(Q.from_int(expr.value)),
                    spec=expr,
                    depth=0,
                )
            # it is not an integer
            except TypeError:
                raise TypeError("Cannot currently lower floats into a fixed point")
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
                raise TypeError(
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
                raise TypeError(
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
                state = (candidate.spec, candidate.node.dtype)
                if state in states:
                    continue
                states.add(state)
                generated.append(candidate)

        for candidate in generated:
            if _matches_result(candidate, spec_ast, return_annotation):
                return candidate.node
        if not generated:
            raise TypeError(
                f"Cannot lower specification {spec_ast!r} to "
                f"{return_annotation!r}; search reached a fixpoint after "
                f"depth {depth} with {len(states)} candidate states"
            )
        if depth == MAX_SEARCH_DEPTH:
            raise TypeError(
                f"Cannot lower specification {spec_ast!r} to "
                f"{return_annotation!r}; search reached the maximum depth "
                f"of {MAX_SEARCH_DEPTH} with {len(states)} candidate states"
            )

        candidates.extend(generated)
        depth += 1


def lower_spec_to_impl(
    name: str,
    spec: tp.Callable[..., tp.Any],
    contract: _SpecContract,
    spec_ast: SpecNode,
    spec_inputs: tuple[tp.Any, ...],
) -> Node:
    @Composite(name=name, spec=spec)
    def generated_impl(*impl_inputs: Node) -> Node:
        spec_input_nodes = dict(
            zip(spec_inputs, impl_inputs, strict=True)
        )
        return search_lower_spec_to_impl(
            spec_ast,
            spec_input_nodes,
            contract.annotations["return"],
        )

    input_parameters = list(contract.signature.parameters.values())[:-1]
    impl_inputs = [
        Var(
            name=parameter.name,
            dtype=contract.annotations[parameter.name],
        )
        for parameter in input_parameters
    ]
    return generated_impl(*impl_inputs)


def Autogenerate(name: str, spec: tp.Callable[..., tp.Any]):
    contract = _build_spec_contract(name, spec)
    reject_untyped_inputs(contract)

    spec_ast, spec_inputs, spec_ctx = get_spec_ast(spec, contract)
    check_spec_feasibility(
        spec_ast,
        spec_inputs,
        contract,
        spec_ctx,
    )
    lowered_composite = lower_spec_to_impl(
        name,
        spec,
        contract,
        spec_ast,
        spec_inputs,
    )

    return lowered_composite
