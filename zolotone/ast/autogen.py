import itertools
import typing as tp

from ..errors import MissingError, ZolotoneError
from ..spec.spec_ast import (
    BoolLit,
    BoolVar,
    RealLit,
    RealVar,
    SpecNode,
    children,
)
from ..spec.spec_context import SpecContext
from ..types import Bool, Q, UQ
from .node import Node
from .nodes import (
    Composite,
    Const,
    Var,
    _annotation_matches,
    _build_spec_contract,
    _SpecContract,
)
from .spec_validation import (
    _simplify_spec_ast,
    check_spec_feasibility,
    reject_untyped_inputs,
    reject_undeclared_variables,
)


class _Candidate(tp.NamedTuple):
    node: Node
    spec: SpecNode
    depth: int


MAX_SEARCH_DEPTH = 30


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
                raise NotImplementedError(
                    "Cannot currently lower floats into a fixed point"
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
                state = (candidate.spec, candidate.node.dtype)
                if state in states:
                    continue
                states.add(state)
                generated.append(candidate)

        for candidate in generated:
            if _matches_result(candidate, spec_ast, return_annotation):
                return candidate.node
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
    reject_undeclared_variables(spec_ast, spec_inputs, spec_ctx)

    # Step 1. Spec Validation
    check_spec_feasibility(spec_ast, spec_inputs, contract, spec_ctx)

    print(spec_ctx)
    # Step 2. Spec Simplification
    spec_ast, spec_ctx = _simplify_spec_ast(spec_ast, spec_inputs, spec_ctx)
    print(spec_ctx)
    print("----------------------------")

    # Step 3. Spec Exploration
    lowered_composite = lower_spec_to_impl(name, spec, contract, spec_ast, spec_inputs)

    return lowered_composite
