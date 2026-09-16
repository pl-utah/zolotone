import typing as tp

from ..spec.spec_ast import Add, Mul, RealLit, SpecNode, children
from ..spec.spec_context import SpecContext
from ..types import DataType, Q, UQ
from .node import Node
from .nodes import Composite, Const, Var, _build_spec_contract, _SpecContract


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


# TODO: inputs of contract have already their own ranges - asserts essentially
def get_spec_ast(
    spec: tp.Callable[..., tp.Any],
    contract: _SpecContract,
) -> tuple[SpecNode, tuple[tp.Any, ...]]:
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
    return spec_ast, tuple(spec_inputs)


def recurse_lower_spec_to_impl(
    spec_ast: SpecNode,
    spec_input_nodes: dict[tp.Any, Node],
) -> Node:
    def recurse(expr: SpecNode) -> Node:
        if expr in spec_input_nodes:
            return spec_input_nodes[expr]

        spec_children = children(expr)
        if not spec_children and not isinstance(expr, RealLit):
            raise TypeError(
                f"Cannot lower unsupported specification leaf {expr!r}"
            )

        lowered_children = tuple(recurse(child) for child in spec_children)
        child_types = tuple(child.dtype for child in lowered_children)

        if isinstance(expr, RealLit):
            return Const(UQ.from_int(expr.value))

        if isinstance(expr, Add):
            if all(isinstance(dtype, UQ) for dtype in child_types):
                from ..components.UQ import uq_add
                return uq_add(*lowered_children)
            if all(isinstance(dtype, Q) for dtype in child_types):
                from ..components.Q import q_add
                return q_add(*lowered_children)

        if isinstance(expr, Mul):
            if all(isinstance(dtype, UQ) for dtype in child_types):
                from ..components.UQ import uq_mul
                return uq_mul(*lowered_children)
            if all(isinstance(dtype, Q) for dtype in child_types):
                from ..components.Q import q_mul
                return q_mul(*lowered_children)

        raise TypeError(
            f"Cannot lower {type(expr).__name__} with input types "
            f"{child_types}"
        )

    return recurse(spec_ast)


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
        return recurse_lower_spec_to_impl(
            spec_ast,
            spec_input_nodes,
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

    spec_ast, spec_inputs = get_spec_ast(spec, contract)
    lowered_composite = lower_spec_to_impl(name, spec, contract, spec_ast, spec_inputs)
    
    return lowered_composite
