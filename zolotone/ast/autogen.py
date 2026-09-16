import typing as tp

from ..spec.spec_ast import SpecNode
from ..spec.spec_context import SpecContext
from ..types import DataType
from .nodes import _build_spec_contract, _SpecContract


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
) -> tuple[SpecNode, dict[tp.Any, DataType]]:
    ctx = SpecContext(contract.display_name)
    input_parameters = list(contract.signature.parameters.values())[:-1]
    spec_inputs = []
    spec_input_types = {}
    for parameter in input_parameters:
        dtype = contract.annotations[parameter.name]
        spec_input = dtype.to_spec(
            name=parameter.name,
            ctx=ctx,
        )
        spec_inputs.append(spec_input)
        spec_input_types[spec_input] = dtype

    spec_ast = spec(*spec_inputs, ctx)
    return spec_ast, spec_input_types

def Autogenerate(name: str, spec: tp.Callable[..., tp.Any]):
    contract = _build_spec_contract(name, spec)
    reject_untyped_inputs(contract)

    spec_ast, spec_input_types = get_spec_ast(spec, contract)
    print(contract)
    print(spec_ast)
    print(spec_input_types)
