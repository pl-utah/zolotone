from dataclasses import dataclass
import inspect
import typing as tp

from ..spec.spec_ast import SpecNode
from ..spec.spec_context import SpecContext
from .nodes import _build_spec_contract, _SpecContract
from .spec_lowering import (
    attach_lowered_conditions,
    lower_spec_result,
    rewrite_strict_conditions,
)
from .spec_validation import (
    _check_output_format,
    _check_spec_obligations,
    _check_spec_reachability,
    _simplify_spec_ast,
    _validate_spec_requirements,
    _validate_spec_shape,
    _warn_about_domain_errors,
    add_input_ranges,
    add_output_guards,
    reject_untyped_inputs,
    reject_undeclared_variables,
)


def _format_annotation(annotation: object) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return repr(annotation)


@dataclass(frozen=True)
class Spec:
    name: str
    function: tp.Callable[..., tp.Any]
    annotations: dict[str, tp.Any]
    signature: inspect.Signature
    display_name: str
    spec_ast: SpecNode
    spec_inputs: tuple[tp.Any, ...]
    spec_ctx: SpecContext

    @property
    def contract(self) -> _SpecContract:
        return _SpecContract(
            signature=self.signature,
            annotations=self.annotations,
            display_name=self.display_name,
        )

    @property
    def return_annotation(self) -> object:
        return self.annotations["return"]

    def __str__(self) -> str:
        input_parameters = list(self.signature.parameters.values())[:-1]
        arguments = ", ".join(
            f"{parameter.name}: "
            f"{_format_annotation(self.annotations[parameter.name])}"
            for parameter in input_parameters
        )
        signature = (
            f"({arguments}) -> {_format_annotation(self.return_annotation)}"
        )

        def format_section(
            title: str,
            expressions: tp.Sequence[SpecNode],
        ) -> list[str]:
            if not expressions:
                return [f"  {title}:", "    <none>"]
            return [f"  {title}:"] + [
                f"    {expression}" for expression in expressions
            ]

        lines = [
            f"Specification {self.display_name}",
            f"  signature: {signature}",
            f"  result: {self.spec_ast}",
            "",
        ]
        lines.extend(format_section("assumes", self.spec_ctx.assumes))
        lines.append("")
        lines.extend(format_section("checks", self.spec_ctx.checks))
        lines.append("")
        lines.extend(
            format_section("requirements", self.spec_ctx.requirements)
        )
        return "\n".join(lines)


def get_spec(
    name: str,
    function: tp.Callable[..., tp.Any],
) -> Spec:
    contract = _build_spec_contract(name, function)
    reject_untyped_inputs(contract)
    spec_ctx = SpecContext(contract.display_name)
    input_parameters = list(contract.signature.parameters.values())[:-1]
    spec_inputs = tuple(
        contract.annotations[parameter.name].to_spec(
            name=parameter.name,
            ctx=spec_ctx,
        )
        for parameter in input_parameters
    )
    spec_ast = function(*spec_inputs, spec_ctx)
    return Spec(
        name=name,
        function=function,
        annotations=dict(contract.annotations),
        signature=contract.signature,
        display_name=contract.display_name,
        spec_ast=spec_ast,
        spec_inputs=spec_inputs,
        spec_ctx=spec_ctx,
    )


def Autogenerate(name: str, spec: tp.Callable[..., tp.Any]):
    # Step 1: Obtain spec AST
    current_spec = get_spec(name, spec)

    print(current_spec)

    # Step 2: establish the abstract input domain and validate its shape.
    # Error on undeclared variables, warn about unused variables.
    reject_undeclared_variables(current_spec)
    # Add exact input ranges implied by their types.
    current_spec = add_input_ranges(current_spec)
    # Catch type errors such as RealExpr versus BoolExpr.
    _validate_spec_shape(current_spec)
    # Step 3: simplify before making conditions type-precise.
    current_spec = _simplify_spec_ast(current_spec)

    # Step 4: rewrite strict assumptions and checks for fixed-point semantics.
    current_spec = rewrite_strict_conditions(current_spec)

    print(current_spec)

    # Step 5: validate the complete, eventually bit-precise specification.
    # Every check below requires bit-precise assumptions.
    _check_spec_reachability(current_spec)
    _warn_about_domain_errors(current_spec)
    _validate_spec_requirements(current_spec)
    _check_spec_obligations(current_spec)
    _check_output_format(current_spec)

    # Step 6: lower the result before deriving implementation guards.
    lowered_composite = lower_spec_result(current_spec)

    # Step 7: derive guards for the selected implementation result.
    current_spec = add_output_guards(current_spec)

    # Step 8: lower and attach assumptions, user checks, and output guards.
    attach_lowered_conditions(lowered_composite, current_spec)

    # Step 9: lower composite to C++
    cpp = lowered_composite.to_cpp()
    return lowered_composite
