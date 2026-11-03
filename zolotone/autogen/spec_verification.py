"""Exhaustive checks of generated C++ using adaptive-precision Rival predicates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

from ..errors import MissingError, ZolotoneError
from ..mpfi import mpfi_compile
from ..rival import collect_free_vars, rival_compile, wrap_precondition
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
    """Counts refer to assignments; skipped means C++ asserted as expected."""

    checked: int
    skipped: int
    ulp: Fraction
    error_bound: Fraction


# Retain the previous import name; verification no longer uses a float reference.
FloatVerificationResult = VerificationResult


class VerificationInconclusive(ZolotoneError):
    """Rival could not resolve a verification predicate within its limits."""


class _RivalDomainError(ZolotoneError):
    """Rival established that an expression is undefined at this point."""


def _signed_pt(raw: int, dtype: Q | UQ | Bool) -> int:
    if isinstance(dtype, Q) and raw & (1 << (dtype.total_bits() - 1)):
        return raw - (1 << dtype.total_bits())
    return raw


def _apply_point(machine, pt, label, stage):
    try:
        return machine.apply_point(pt)
    except ValueError as exc:
        if "Unsamplable" in str(exc):
            raise VerificationInconclusive(
                f"ZFV: {stage} inconclusive at {label}: {exc}"
            ) from exc
        if "Invalid input" in str(exc):
            raise _RivalDomainError(f"ZFV: {stage} reported a domain error at {label}") from exc
        raise ZolotoneError(f"ZFV: {stage} failed at {label}: {exc}") from exc


def strip_root_rounding(root: RealExpr) -> RealExpr:
    """Extract the unrounded reference from a faithfully expanded root."""
    while isinstance(root, If):
        root = root.on_true
    if not isinstance(root, Round):
        raise ValueError("Faithful expansion did not preserve the root rounding")
    return root.value


def zfv(spec: Spec, component: Composite) -> VerificationResult:
    """Check each reachable input/choice against rounding and the error budget.

    Faithful rounding is expanded in assumptions and intermediate reference
    expressions, sharing synthesis's choice redirection. False or undefined
    assumptions require a C++ assertion; true assumptions require a result.
    Unresolved assumptions and output evaluation errors fail verification.
    """
    from .autogen import _expand_faithful_roundings
    from .spec_validation import _spec_input_dtypes
    from infra.compile_cpp import CppAssertionError, jit_compile

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

    # Expand faithful rounding in both the reference (due to nested roundings) and assumptions.
    expanded_spec = _expand_faithful_roundings(spec)
    rounded_reference = expanded_spec.spec_ast
    if isinstance(rounded_reference, Round) and rounded_reference.frac_bits is None:
        rounded_reference = replace(rounded_reference, frac_bits=precision)
    unrounded_reference = strip_root_rounding(rounded_reference)
    precondition = BoolLit(True)
    for assumption in expanded_spec.spec_ctx.assumes:
        precondition = precondition & assumption
    occupied = set(collect_free_vars([unrounded_reference, rounded_reference, precondition]))
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

    # Scale inputs
    free_vars = []
    for index, (source, dtype) in enumerate(inputs):
        code = fresh_name(f"__zfv_code_{index}")
        free_vars.append(code.name)
        decoded = code.ne(RealLit(0)) if isinstance(dtype, Bool) else (
            code * RealLit(Fraction(2) ** -dtype.frac_bits)
        )
        unrounded_reference = substitute_spec_node(unrounded_reference, source, decoded)
        rounded_reference = substitute_spec_node(rounded_reference, source, decoded)
        precondition = substitute_spec_node(precondition, source, decoded)

    # Rounding choices
    for index in range(rounding_choices):
        code = fresh_name(f"__zfv_choice_{index}")
        free_vars.append(code.name)
        if len(expanded_choices) == rounding_choices:
            decoded = code.ne(RealLit(0))
            unrounded_reference = substitute_spec_node(unrounded_reference, expanded_choices[index], decoded)
            rounded_reference = substitute_spec_node(rounded_reference, expanded_choices[index], decoded)
            precondition = substitute_spec_node(precondition, expanded_choices[index], decoded)

    # Specs assumptions
    assumptions_machine = rival_compile(
        [wrap_precondition(precondition)], ["boolean"], free_vars=free_vars,
    )

    # CPP results inside a machine variable
    choices = tuple(product((0, 1), repeat=rounding_choices))
    cpp_result = fresh_name("__zfv_output")
    actual = cpp_result * RealLit(Fraction(2) ** -output_dtype.frac_bits)
    error = abs(actual - unrounded_reference)
    predicate = error <= RealLit(error_bound) if inclusive else error < RealLit(error_bound)
    machine = rival_compile(
        [predicate, actual.eq(rounded_reference)], ["boolean", "boolean"],
        free_vars=free_vars + [cpp_result.name],
    )
    # Independently evaluate only the numeric error-budget LHS with MPFI.
    # Unsupported nodes such as nested Round fail explicitly in the translator.
    mpfi_error = mpfi_compile(
        [error], free_vars=free_vars + [cpp_result.name],
    )

    checked = skipped = 0
    # Compile C++
    tempdir, evaluate_cpp = jit_compile(component, catch_assertions=True)
    try:
        # Exhaustively iterate over every input
        for raw_inputs in product(*(range(1 << dtype.total_bits()) for _, dtype in inputs)):
            signed_pt = [_signed_pt(raw, dtype) for raw, (_, dtype) in zip(raw_inputs, inputs)]
            label = {source.name: code for (source, _), code in zip(inputs, signed_pt)}
            for choice in choices:
                # Assumes evaluation
                point = signed_pt + list(choice)
                assignment_label = {"inputs": label, "choices": choice}
                assumption_failure = "undefined assumptions"
                try:
                    assumptions, = _apply_point(
                        assumptions_machine, point, assignment_label, "assumption evaluation",
                    )
                except _RivalDomainError:
                    pass
                else:
                    if not assumptions.total_error:
                        if assumptions.error or not assumptions.resolved:
                            raise VerificationInconclusive(
                                f"ZFV: assumption evaluation unresolved at {assignment_label}"
                            )
                        if assumptions.value is not True and assumptions.value is not False:
                            raise VerificationInconclusive(f"ZFV: assumptions unresolved at {assignment_label}")
                        assumption_failure = None if assumptions.value else "false assumptions"

                # C++ evaluation
                try:
                    raw_output = int(evaluate_cpp(*raw_inputs, *choice))
                except CppAssertionError as exc:
                    if assumption_failure is not None:
                        skipped += 1
                        continue
                    raise ZolotoneError(
                        f"ZFV: C++ asserted despite true assumptions at {assignment_label}: {exc}"
                    ) from exc
                if assumption_failure is not None:
                    raise ZolotoneError(
                        f"ZFV: C++ returned despite {assumption_failure} at {assignment_label}: {raw_output}"
                    )

                # At this point C++ and assumes should agree that the point belongs to domain

                output_dtype.validate_raw(raw_output)
                cpp_code = _signed_pt(raw_output, output_dtype)
                results = _apply_point(machine, point + [cpp_code], assignment_label, "spec comparison")
                names = ("error budget", "rounded reference")
                for name, result in zip(names, results, strict=True):
                    if result.total_error:
                        raise ZolotoneError(
                            f"ZFV: spec comparison ({name}) reported a domain error at {assignment_label}"
                        )
                failures = [
                    name for name, result in zip(names, results, strict=True)
                    if result.resolved and not result.error and result.value is False
                ]
                if failures:
                    value = Fraction(cpp_code, 1 << output_dtype.frac_bits)
                    relation = "<=" if inclusive else "<"
                    raise ZolotoneError(
                        f"zfv mismatch at input point {label} (raw={raw_inputs}): "
                        f"C++ result (choices, value)={(choice, value)}; "
                        f"failed checks: {', '.join(failures)}; "
                        f"required abs(C++ - unrounded_reference) {relation} {error_bound} "
                        "and C++ == rounded_reference for the selected rounding choices"
                    )
                for name, result in zip(names, results, strict=True):
                    if result.error or not result.resolved:
                        raise VerificationInconclusive(
                            f"ZFV: spec comparison ({name}) unresolved at {assignment_label}"
                        )
                precision = 128
                try:
                    mpfi_value, = mpfi_error.apply_point(point + [cpp_code], precision=precision)
                except NotImplementedError as exc:
                    raise VerificationInconclusive(
                        f"ZFV: MPFI error-budget evaluation unsupported at {assignment_label}: {exc}"
                    ) from exc
                mpfi_lower, mpfi_upper = mpfi_value.lower(), mpfi_value.upper()
                rival_error_budget = results[0]
                if inclusive:
                    mpfi_proved = mpfi_upper <= error_bound
                    mpfi_disproved = mpfi_lower > error_bound
                else:
                    mpfi_proved = mpfi_upper < error_bound
                    mpfi_disproved = mpfi_lower >= error_bound
                if not mpfi_proved and not mpfi_disproved:
                    raise VerificationInconclusive(
                        f"ZFV: MPFI error-budget interval straddles {error_bound} "
                        f"at {assignment_label}: [{mpfi_lower}, {mpfi_upper}]"
                    )
                if mpfi_proved != rival_error_budget.value:
                    raise ZolotoneError(
                        f"ZFV: Rival/MPFI error-budget disagreement at {assignment_label}: "
                        f"precision={precision} could not resolve error bound {error_bound}, MPFI=[{mpfi_lower}, {mpfi_upper}]"
                    )
                checked += 1
    finally:
        tempdir.cleanup()
    if checked == 0:
        raise ZolotoneError("zfv found no reachable input points")
    return VerificationResult(checked, skipped, ulp, error_bound)
