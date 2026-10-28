from dataclasses import dataclass, field, replace
from fractions import Fraction
import inspect
import typing as tp

from ..spec.spec_ast import (
    BoolVar,
    FPExpr,
    If,
    Round,
    RoundingMode,
    SpecNode,
    children,
    rebuild_spec_node,
    substitute_spec_node,
)
from ..spec.spec_context import SpecContext
from ..errors import MissingError
from ..types import Bool, Q, UQ
from ..ast.nodes import _build_spec_contract, _SpecContract
from .spec_lowering import (
    attach_lowered_conditions,
    lower_spec_result,
    rewrite_strict_conditions,
)
from .spec_validation import (
    _check_output_format,
    spec_domain_analysis,
    _simplify_spec_ast,
    _spec_value_variables,
    _validate_spec_requirements,
    _validate_rounding_precision,
    _validate_spec_shape,
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
    annotations: dict[str, tp.Any]
    signature: inspect.Signature
    display_name: str
    spec_ast: SpecNode
    spec_inputs: tuple[tp.Any, ...]
    spec_ctx: SpecContext
    rounding_choices: dict[Round, BoolVar] = field(default_factory=dict)
    has_user_assumptions: bool = False
    output_bounds: tuple[Fraction, Fraction] | None = None

    def to_function(self) -> tp.Callable[..., tp.Any]:
        """Derive a callable from this AST and contract, without replaying source."""
        signature = self.signature
        parameters = tuple(signature.parameters.values())
        inputs = self.spec_inputs
        spec_ast = self.spec_ast
        source_ctx = self.spec_ctx.copy()

        def derived_spec(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            ctx = bound.arguments[parameters[-1].name]
            replacements = {}

            def bind_input(source, target):
                if isinstance(source, tuple):
                    for old, new in zip(source, target, strict=True):
                        bind_input(old, new)
                    return
                replacements[source] = target
                if isinstance(source, FPExpr):
                    for old, new in zip(children(source), children(target), strict=True):
                        bind_input(old, new)

            for source, parameter in zip(inputs, parameters[:-1], strict=True):
                bind_input(source, bound.arguments[parameter.name])

            rewritten = {}

            def substitute(node):
                # Return replacements directly: never substitute into a new input.
                if node in replacements:
                    return replacements[node]
                if node not in rewritten:
                    old_children = children(node)
                    new_children = tuple(substitute(child) for child in old_children)
                    rewritten[node] = (
                        node
                        if all(old is new for old, new in zip(old_children, new_children))
                        else rebuild_spec_node(node, new_children)
                    )
                return rewritten[node]

            for location in ("assumes", "checks", "requirements"):
                getattr(ctx, location).extend(
                    substitute(condition) for condition in getattr(source_ctx, location)
                )
            ctx.case_partitions.extend(
                replace(
                    partition,
                    entries=tuple(
                        replace(entry, condition=substitute(entry.condition), value=substitute(entry.value))
                        for entry in partition.entries
                    ),
                    value=substitute(partition.value),
                )
                for partition in source_ctx.case_partitions
            )
            return substitute(spec_ast)

        derived_spec.__name__ = f"{self.name}_spec"
        derived_spec.__qualname__ = derived_spec.__name__
        derived_spec.__signature__ = signature
        derived_spec.__annotations__ = {
            **self.annotations, parameters[-1].name: SpecContext,
        }
        return derived_spec

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
        if self.rounding_choices:
            lines.extend(["", "  rounding choices:"])
            lines.extend(
                f"    {choice}: {expression}"
                for expression, choice in self.rounding_choices.items()
            )
        return "\n".join(lines)


def get_spec(
    name: str,
    function: tp.Callable[..., tp.Any],
) -> Spec:
    contract = _build_spec_contract(name, function)
    reject_untyped_inputs(contract)
    output_dtype = contract.annotations["return"]
    spec_ctx = SpecContext(contract.display_name)
    input_parameters = list(contract.signature.parameters.values())[:-1]
    spec_inputs = tuple(
        contract.annotations[parameter.name].to_spec(
            name=parameter.name,
            ctx=spec_ctx,
        )
        for parameter in input_parameters
    )
    # Input descriptors can introduce their own well-formedness assumptions.
    # Only assumptions added by the specification need a reachability proof.
    input_assumption_count = len(spec_ctx.assumes)
    spec_ast = function(*spec_inputs, spec_ctx)
    has_user_assumptions = len(spec_ctx.assumes) > input_assumption_count
    if isinstance(spec_ast, Round) and spec_ast.frac_bits is None:
        if isinstance(output_dtype, (Q, UQ)):
            unresolved = spec_ast
            inferred = replace(spec_ast, frac_bits=output_dtype.frac_bits)
            spec_ast = inferred
            for location in ("assumes", "checks", "requirements"):
                conditions = getattr(spec_ctx, location)
                conditions[:] = [
                    substitute_spec_node(condition, unresolved, inferred)
                    for condition in conditions
                ]
            spec_ctx.case_partitions = [
                replace(
                    partition,
                    entries=tuple(
                        replace(
                            entry,
                            condition=substitute_spec_node(entry.condition, unresolved, inferred),
                            value=substitute_spec_node(entry.value, unresolved, inferred),
                        )
                        for entry in partition.entries
                    ),
                    value=substitute_spec_node(partition.value, unresolved, inferred),
                )
                for partition in spec_ctx.case_partitions
            ]
        else:
            raise MissingError(
                "Cannot infer rounding precision; provide frac_bits explicitly "
                "or use a concrete Q or UQ return annotation"
            )
    return Spec(
        name=name,
        annotations=dict(contract.annotations),
        signature=contract.signature,
        display_name=contract.display_name,
        spec_ast=spec_ast,
        spec_inputs=spec_inputs,
        spec_ctx=spec_ctx,
        has_user_assumptions=has_user_assumptions,
    )


def _expand_faithful_roundings(spec: Spec) -> Spec:
    """Expand FR into explicit choices without adding ordinary function inputs."""
    # Check the original root before an If could hide narrowing or missing precision.
    _validate_rounding_precision(spec)
    ctx = spec.spec_ctx.copy()
    choices = dict(spec.rounding_choices)
    occupied_names = {
        variable.name
        for variable in _spec_value_variables((
            spec.spec_inputs, spec.spec_ast,
            *ctx.assumes, *ctx.checks, *ctx.requirements,
            *choices.values(),
        ))
    }
    occupied_names.update(spec.signature.parameters)
    rewritten: dict[SpecNode, SpecNode] = {}
    expanded = False

    def visit(node: SpecNode) -> SpecNode:
        nonlocal expanded
        if node in rewritten:
            return rewritten[node]
        old_children = children(node)
        new_children = tuple(visit(child) for child in old_children)
        result = (
            node
            if all(old is new for old, new in zip(old_children, new_children))
            else rebuild_spec_node(node, new_children)
        )
        if isinstance(node, Round) and node.mode is RoundingMode.FR:
            if node.frac_bits is None:
                raise MissingError("Faithful rounding requires frac_bits explicitly")
            if node not in choices:
                choice = ctx.fresh_bool("rnd")
                while choice.name in occupied_names:
                    choice = ctx.fresh_bool("rnd")
                occupied_names.add(choice.name)
                choices[node] = choice
            result = If(
                choices[node],
                replace(result, mode=RoundingMode.RAZ),
                replace(result, mode=RoundingMode.RTZ),
            )
            expanded = True
        rewritten[node] = result
        return result

    spec_ast = visit(spec.spec_ast)
    for location in ("assumes", "checks", "requirements"):
        conditions = getattr(ctx, location)
        conditions[:] = [visit(condition) for condition in conditions]
    ctx.case_partitions = [
        replace(
            partition,
            entries=tuple(
                replace(entry, condition=visit(entry.condition), value=visit(entry.value))
                for entry in partition.entries
            ),
            value=visit(partition.value),
        )
        for partition in ctx.case_partitions
    ]
    if not expanded:
        return spec
    new_choices = tuple(choice for choice in choices.values() if choice not in spec.spec_inputs)
    parameters = tuple(spec.signature.parameters.values())
    choice_parameters = tuple(
        inspect.Parameter(choice.name, parameters[-1].kind, annotation=Bool())
        for choice in new_choices
    )
    return replace(
        spec,
        spec_ast=spec_ast,
        spec_ctx=ctx,
        spec_inputs=spec.spec_inputs + new_choices,
        annotations={
            **{name: dtype for name, dtype in spec.annotations.items() if name != "return"},
            **{choice.name: Bool() for choice in new_choices},
            "return": spec.return_annotation,
        },
        signature=spec.signature.replace(parameters=(*parameters[:-1], *choice_parameters, parameters[-1])),
        rounding_choices=choices,
    )


def Autogenerate(
    name: str,
    spec: tp.Callable[..., tp.Any],
):
    """Lower a specification, creating missing real Primitives in memory."""
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
    current_spec = _expand_faithful_roundings(current_spec)
    # Step 3: simplify before making conditions type-precise.
    current_spec = _simplify_spec_ast(current_spec)

    # Step 4: tighten comparisons and resolve unrepresentable equalities in
    # normalized/simplified assumptions and checks for fixed-point semantics.
    current_spec = rewrite_strict_conditions(current_spec)

    # Step 5: validate the complete, eventually bit-precise specification.
    # Every check below requires bit-precise assumptions.
    current_spec = spec_domain_analysis(current_spec)
    _validate_spec_requirements(current_spec)
    _check_output_format(current_spec)

    # Step 6: lower the result before deriving implementation guards.
    lowered_composite = lower_spec_result(current_spec)

    if current_spec.return_annotation is Q or current_spec.return_annotation is UQ:
        # Resolve family annotations without mutating the user's function.
        result_dtype = lowered_composite.dtype
        current_spec = replace(
            current_spec,
            annotations={**current_spec.annotations, "return": result_dtype},
            signature=current_spec.signature.replace(return_annotation=result_dtype),
        )

    # Step 7: derive guards for the selected implementation result.
    current_spec = add_output_guards(current_spec)

    print(current_spec)


    # Step 8: lower and attach assumptions, user checks, and output guards.
    attach_lowered_conditions(lowered_composite, current_spec)

    # Step 9: lower composite to C++
    cpp = lowered_composite.to_cpp()
    return lowered_composite
