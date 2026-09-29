import typing as tp

from ..spec.spec_ast import RealExpr, SpecNode
from ..spec.spec_context import SpecContext
from .nodes import _build_spec_contract, _SpecContract
from .spec_lowering import (
    attach_lowered_conditions,
    lower_spec_result,
)
from .spec_validation import (
    _check_output_format,
    _check_spec_obligations,
    _check_spec_reachability,
    _derive_input_ranges,
    _derive_output_guards,
    _simplify_spec_ast,
    _validate_spec_shape,
    _warn_about_domain_errors,
    reject_untyped_inputs,
    reject_undeclared_variables,
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


def Autogenerate(name: str, spec: tp.Callable[..., tp.Any]):
    # Step 1: Obtain spec AST
    contract = _build_spec_contract(name, spec)
    reject_untyped_inputs(contract)
    spec_ast, spec_inputs, spec_ctx = get_spec_ast(spec, contract)
    return_annotation = contract.annotations["return"]

    # Step 2: Validate spec
    ## Error on undeclared variables, warn about unused variables
    reject_undeclared_variables(spec_ast, spec_inputs, spec_ctx)
    ## Exact ranges of inputs given its types
    input_ranges = _derive_input_ranges(spec_inputs, contract, spec_ctx)
    spec_ctx.assumes[:0] = input_ranges
    ## Catching type errors like RealExpr vs BoolExpr
    _validate_spec_shape(spec_ast, return_annotation)
    ## Reachibility with user-provided assumes
    _check_spec_reachability(spec_ctx)
    ## Domain errors given user-provided/derived assumes
    _warn_about_domain_errors(spec_ast, spec_ctx)
    ## Prove determinism of "Cases"
    spec_ctx.validate_requirements()
    ## Prove that user-defined asserts are satisfied
    _check_spec_obligations(spec_ctx)
    ## Check that output range fits output format; warn if it can be narrowed
    _check_output_format(spec_ast, return_annotation, spec_ctx)

    # Step 3: simplify before exploring implementation candidates.
    spec_ast, spec_ctx = _simplify_spec_ast(spec_ast, spec_inputs, spec_ctx)

    # Step 4: lower the result before deriving implementation guards.
    lowered_composite = lower_spec_result(
        name,
        spec,
        contract,
        spec_ast,
        spec_inputs,
        spec_ctx=spec_ctx,
    )

    # Step 5: derive guards for the selected implementation result.
    output_guards = _derive_output_guards(
        spec_ast,
        return_annotation,
        spec_ctx,
    )
    _check_spec_obligations(spec_ctx.copy(checks=list(output_guards)))
    spec_ctx.checks.extend(output_guards)

    # Step 6: lower and attach assumptions, user checks, and output guards.
    attach_lowered_conditions(
        lowered_composite,
        spec_ast,
        spec_inputs,
        spec_ctx,
    )
    return lowered_composite
