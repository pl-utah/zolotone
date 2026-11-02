"""Exhaustive checks of generated C++ using adaptive-precision Rival predicates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

from ..errors import MissingError, ZolotoneError
from ..rival import build_boolean_evaluator, build_integer_evaluator, collect_free_vars
from ..spec.spec_ast import (
    BoolLit, BoolVar, If, RealExpr, RealLit, RealVar, Round, RoundingMode,
    substitute_spec_node,
)
from ..types import Bool, Q, UQ

if TYPE_CHECKING:
    from .autogen import Spec
    from ..ast.nodes import Composite


@dataclass(frozen=True)
class VerificationResult:
    """Counts refer to input/rounding-choice assignments."""

    checked: int
    skipped: int
    ulp: Fraction
    error_bound: Fraction


# Retain the previous import name; verification no longer uses a float reference.
FloatVerificationResult = VerificationResult


class VerificationInconclusive(ZolotoneError):
    """Rival could not resolve a verification predicate within its limits."""


def _signed_pt(raw: int, dtype: Q | UQ | Bool) -> int:
    if isinstance(dtype, Q) and raw & (1 << (dtype.total_bits() - 1)):
        return raw - (1 << dtype.total_bits())
    return raw


def _evaluate_point(machine, pt, label, stage):
    try:
        result = machine.apply_point(pt)
    except ValueError as exc:
        if "Unsamplable" in str(exc):
            raise VerificationInconclusive(
                f"ZFV: {stage} inconclusive at {label}: {exc}"
            ) from exc
        raise ZolotoneError(f"ZFV: {stage} failed at {label}: {exc}") from exc
    if result.precondition_total_error or result.output_total_error:
        raise ZolotoneError(f"ZFV: {stage} reported a domain error at {label}")
    if result.precondition_error or result.output_error:
        raise VerificationInconclusive(f"ZFV: {stage} unresolved at {label}")
    return result


def zfv(spec: Spec, component: Composite) -> VerificationResult:
    """Check each reachable input/choice against the original root error budget.

    Faithful rounding is expanded in assumptions and intermediate reference
    expressions, sharing synthesis's choice redirection. Only proven-false
    assumptions exclude an assignment; evaluation errors never count as skips.
    """
    from .autogen import _expand_faithful_roundings
    from .spec_validation import _spec_input_dtypes
    from infra.compile_cpp import jit_compile

    output_dtype = component.dtype
    if not isinstance(output_dtype, (Q, UQ)):
        raise NotImplementedError("zfv requires a Q or UQ output")
    if output_dtype.total_bits() > 64:
        raise NotImplementedError("zfv C++ outputs must have at most 64 bits")

    # getting precision at output
    root = spec.spec_ast
    if not isinstance(root, RealExpr):
        raise NotImplementedError("zfv requires a scalar real specification")
    if not isinstance(root, Round):
        raise NotImplementedError("zfv requires a round operation to be a root")
    precision = root.frac_bits
    if precision is None:
        dtype = spec.return_annotation
        if isinstance(dtype, (Q, UQ)):
            precision = dtype.frac_bits 
        else:
            raise MissingError("not provided frac_bits!!")
    
    # calculating error budget at output
    ulp = Fraction(2) ** -precision
    if root.mode is RoundingMode.RNE:
        error_bound = ulp / 2
        inclusive = True
    elif root.mode in (RoundingMode.FR, RoundingMode.RTZ, RoundingMode.RAZ):
        error_bound = ulp
        inclusive = False
    else:
        raise NotImplementedError(f"zfv does not support rounding mode {root.mode!r}")

    # Checking inputs for correctness
    inputs = list(_spec_input_dtypes(spec).items())
    rounding_choices = getattr(component, "rounding_choice_count", 0)
    if len(inputs) + rounding_choices != len(component.inner_args):
        raise ValueError("Spec inputs do not match the C++ component arguments")
    original_arguments = component.inner_args[:len(inputs)]
    choice_arguments = component.inner_args[len(inputs):]
    if any(argument.dtype != Bool() for argument in choice_arguments):
        raise ValueError("Generated rounding-choice arguments must be Boolean")
    for (source, dtype), argument in zip(inputs, original_arguments, strict=True):
        if not isinstance(dtype, (Q, UQ, Bool)) or not isinstance(source, (RealVar, BoolVar)):
            raise NotImplementedError("zfv supports scalar Q, UQ, and Bool inputs")
        if dtype != argument.dtype:
            raise ValueError("Spec and C++ input dtypes differ")
        if dtype.total_bits() > 64:
            raise NotImplementedError("zfv C++ inputs must have at most 64 bits")

    # Expand a copy, preserving the original root's mode, precision and
    # unrounded numerical reference rather than using the expanded output.
    expanded_spec = _expand_faithful_roundings(spec)
    # Root FR becomes an If selecting RTZ/RAZ, possibly through another If
    # redirecting a disallowed choice. All those outer Round nodes share the
    # same pre-round argument. Take that argument, retaining expanded inner
    # roundings while discarding only the root's choice and correction guards.
    expanded_root = expanded_spec.spec_ast
    while isinstance(expanded_root, If):
        expanded_root = expanded_root.on_true
    if not isinstance(expanded_root, Round):
        raise ValueError("Faithful expansion did not preserve the root rounding")
    reference = expanded_root.value
    precondition = BoolLit(True)
    for assumption in expanded_spec.spec_ctx.assumes:
        precondition = precondition & assumption
    occupied = set(collect_free_vars([reference, precondition]))
    expanded_choices = tuple(expanded_spec.rounding_choices.values())
    referenced_choices = occupied & {choice.name for choice in expanded_choices}
    if referenced_choices and len(expanded_choices) != rounding_choices:
        raise ValueError("Spec rounding choices do not match the C++ component arguments")

    # Rename scaled inputs
    def fresh_name(prefix):
        name = prefix
        while name in occupied:
            name += "_"
        occupied.add(name)
        return RealVar(name)

    free_vars = []
    for index, (source, dtype) in enumerate(inputs):
        code = fresh_name(f"__zfv_code_{index}")
        free_vars.append(code.name)
        decoded = code.ne(RealLit(0)) if isinstance(dtype, Bool) else (
            code * RealLit(Fraction(2) ** -dtype.frac_bits)
        )
        reference = substitute_spec_node(reference, source, decoded)
        precondition = substitute_spec_node(precondition, source, decoded)

    for index in range(rounding_choices):
        code = fresh_name(f"__zfv_choice_{index}")
        free_vars.append(code.name)
        if len(expanded_choices) == rounding_choices:
            decoded = code.ne(RealLit(0))
            reference = substitute_spec_node(reference, expanded_choices[index], decoded)
            precondition = substitute_spec_node(precondition, expanded_choices[index], decoded)

    # Keep reachability separate so unreachable inputs never execute C++.
    assumptions_machine = build_integer_evaluator(
        RealLit(0), precondition, free_vars=free_vars,
    )

    # One predicate serves every assignment, including nested faithful choices.
    choices = tuple(product((0, 1), repeat=rounding_choices))
    cpp_result = fresh_name("__zfv_output")
    actual = cpp_result * RealLit(Fraction(2) ** -output_dtype.frac_bits)
    error = abs(actual - reference)
    predicate = error <= RealLit(error_bound) if inclusive else error < RealLit(error_bound)
    machine = build_boolean_evaluator(predicate, free_vars=free_vars + [cpp_result.name])

    checked = skipped = 0
    # Compile C++
    tempdir, evaluate_cpp = jit_compile(component)
    try:
        # Exhaustively iterate over every input
        for raw_inputs in product(*(range(1 << dtype.total_bits()) for _, dtype in inputs)):
            signed_pt = [_signed_pt(raw, dtype) for raw, (_, dtype) in zip(raw_inputs, inputs)]
            label = {source.name: code for (source, _), code in zip(inputs, signed_pt)}
            for choice in choices:
                point = signed_pt + list(choice)
                assignment_label = {"inputs": label, "choices": choice}
                assumptions = _evaluate_point(
                    assumptions_machine, point, assignment_label, "assumption evaluation",
                )
                if assumptions.precondition is False:
                    skipped += 1
                    continue
                if assumptions.precondition is not True:
                    raise VerificationInconclusive(f"ZFV: assumptions unresolved at {assignment_label}")

                raw_output = int(evaluate_cpp(*raw_inputs, *choice))
                output_dtype.validate_raw(raw_output)
                cpp_code = _signed_pt(raw_output, output_dtype)
                result = _evaluate_point(
                    machine, point + [cpp_code], assignment_label, "spec comparison",
                )
                if result.output == 0:
                    value = Fraction(cpp_code, 1 << output_dtype.frac_bits)
                    relation = "<=" if inclusive else "<"
                    raise ZolotoneError(
                        f"zfv mismatch at input point {label} (raw={raw_inputs}): "
                        f"C++ result (choices, value)={(choice, value)}; "
                        f"required abs(C++ - reference) {relation} {error_bound} "
                    )
                if result.output != 1:
                    raise VerificationInconclusive(f"ZFV: spec comparison unresolved at {assignment_label}")
                checked += 1
    finally:
        tempdir.cleanup()
    if checked == 0:
        raise ZolotoneError("zfv found no reachable input points")
    return VerificationResult(checked, skipped, ulp, error_bound)
