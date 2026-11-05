"""Exhaustive checks of generated C++ using adaptive-precision Rival predicates."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from math import ceil, floor
from typing import TYPE_CHECKING

from .zfv_certificate import (
    ZFVCertificate, _RunCollector, _decode_assignment, _manifest,
)

from ..errors import MissingError, ZolotoneError
from ..mpfi import mpfi_compile
from ..rival import collect_free_vars, rival_compile, wrap_precondition
from ..spec.spec_ast import (
    BoolLit, BoolVar, If, NotEq, RealExpr, RealLit, RealVar, Round, RoundingMode,
    substitute_spec_node,
)
from ..types import Bool, Q, UQ

if TYPE_CHECKING:
    from .autogen import Spec
    from ..ast.nodes import Composite


class VerificationInconclusive(ZolotoneError):
    """A numerical checker could not resolve a verification obligation."""


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


def _highest_precision(machine, label, stage):
    try:
        return machine.highest_precision()
    except ValueError as exc:
        raise VerificationInconclusive(
            f"ZFV: Rival precision unavailable for {stage} at {label}: {exc}"
        ) from exc


def _selected_rounding_mode(root, values, label):
    """Follow the expanded root's choice branches without using the C++ result."""
    while isinstance(root, If):
        if isinstance(root.cond, BoolLit):
            selected = root.cond.value
        elif isinstance(root.cond, BoolVar) and root.cond.name in values:
            selected = bool(values[root.cond.name])
        elif (isinstance(root.cond, NotEq) and isinstance(root.cond.lhs, RealVar)
              and root.cond.lhs.name in values and isinstance(root.cond.rhs, RealLit)
              and root.cond.rhs.value == 0):
            # Boolean inputs and rounding choices are lowered to code != 0.
            selected = values[root.cond.lhs.name] != 0
        else:
            raise VerificationInconclusive(
                f"ZFV: MPFI rounding-mode selection requires an unsupported condition at {label}"
            )
        root = root.on_true if selected else root.on_false
    if not isinstance(root, Round) or root.mode is RoundingMode.FR:
        raise VerificationInconclusive(f"ZFV: MPFI rounding mode is unresolved at {label}")
    return root.mode


# Used by MPFI to round values
def _round_endpoint(value: Fraction, mode: RoundingMode, precision: int) -> Fraction:
    """Round an exact rational to the output grid without floating-point conversion."""
    scale = 1 << precision
    scaled = value * scale
    if mode is RoundingMode.RNE:
        code = round(scaled)  # Fraction implements exact ties-to-even.
    elif mode is RoundingMode.RTZ:
        code = int(scaled)
    elif mode is RoundingMode.RAZ:
        code = floor(scaled) if scaled < 0 else ceil(scaled)
    else:
        raise VerificationInconclusive(f"ZFV: unsupported MPFI rounding mode {mode}")
    return Fraction(code, scale)


def _check_mpfi_reference(reference, actual, mode, precision, error_bound, inclusive, label):
    """Check error and round MPFI endpoints to precision fractional bits."""
    low, high = reference.lower(), reference.upper()
    if not isinstance(low, Fraction) or not isinstance(high, Fraction) or low > high:
        raise VerificationInconclusive(f"ZFV: MPFI reference is not a finite enclosure at {label}")

    max_error = max(abs(actual - low), abs(actual - high))
    proved = max_error <= error_bound if inclusive else max_error < error_bound
    if not proved:
        raise VerificationInconclusive(
            f"ZFV: MPFI error budget not established at {label}: "
            f"max_error={max_error}, bound={error_bound}, inclusive={inclusive}"
        )

    rounded_low, rounded_high = (_round_endpoint(endpoint, mode, precision) for endpoint in (low, high))
    if rounded_low != rounded_high:
        raise VerificationInconclusive(
            f"ZFV: MPFI {mode.value} rounding unresolved at {label}: "
            f"[{low}, {high}] rounds to [{rounded_low}, {rounded_high}]"
        )
    if rounded_low != actual:
        raise ZolotoneError(
            f"ZFV: MPFI/C++ rounded-reference mismatch at {label}: "
            f"MPFI=[{low}, {high}], rouded MPFI={rounded_low}, C++={actual}, mode={mode.value}"
        )


def zfv(spec: Spec, component: Composite) -> ZFVCertificate:
    """Check each reachable input/choice against rounding and the error budget.

    Faithful rounding is expanded in assumptions and intermediate reference
    expressions, sharing synthesis's choice redirection. False or undefined
    assumptions require a C++ assertion; true assumptions require a result.
    Unresolved assumptions and output evaluation errors fail verification.
    Return a certificate only after checking every assignment.
    """
    from .autogen import _expand_faithful_roundings
    from .spec_validation import _spec_input_dtypes
    from ..codegen.compile_cpp import CppAssertionError, jit_compile

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
    rounding_allowed = expanded_spec.rounding_allowed if root.mode is RoundingMode.FR else {}
    occupied = set(collect_free_vars([
        unrounded_reference, rounded_reference, precondition,
        *(condition for _, condition in rounding_allowed.values()),
    ]))
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
        rounding_allowed = {
            selected: (mode, substitute_spec_node(condition, source, decoded))
            for selected, (mode, condition) in rounding_allowed.items()
        }

    # Rounding choices
    for index in range(rounding_choices):
        code = fresh_name(f"__zfv_choice_{index}")
        free_vars.append(code.name)
        if len(expanded_choices) == rounding_choices:
            decoded = code.ne(RealLit(0))
            unrounded_reference = substitute_spec_node(unrounded_reference, expanded_choices[index], decoded)
            rounded_reference = substitute_spec_node(rounded_reference, expanded_choices[index], decoded)
            precondition = substitute_spec_node(precondition, expanded_choices[index], decoded)
            rounding_allowed = {
                selected: (mode, substitute_spec_node(condition, expanded_choices[index], decoded))
                for selected, (mode, condition) in rounding_allowed.items()
            }

    # CPP results inside a machine variable
    cpp_result = fresh_name("__zfv_output")
    actual = cpp_result * RealLit(Fraction(2) ** -output_dtype.frac_bits)
    error = abs(actual - unrounded_reference)
    predicate = error <= RealLit(error_bound) if inclusive else error < RealLit(error_bound)
    manifest = _manifest(
        spec, component, inputs, expanded_choices, rounding_choices, ulp, error_bound, inclusive,
    )
    assumptions_machine = rival_compile(
        [wrap_precondition(precondition), *(condition for _, condition in rounding_allowed.values())],
        ["boolean"] * (1 + len(rounding_allowed)), free_vars=free_vars,
    )
    machine = rival_compile(
        [predicate, actual.eq(rounded_reference)], ["boolean", "boolean"],
        free_vars=free_vars + [cpp_result.name],
    )
    # MPFI evaluates only the unrounded reference; all checks use exact Python arithmetic.
    try:
        mpfi_reference = mpfi_compile(
            [unrounded_reference], free_vars=free_vars,
        )
    except NotImplementedError as exc:
        raise VerificationInconclusive(f"ZFV: MPFI does not support this expression: {exc}") from exc

    collector = _RunCollector()

    # Compile C++
    tempdir, evaluate_cpp = jit_compile(component, catch_assertions=True)
    try:
        # Enumerate indices without materializing any input or choice domain.
        input_widths = tuple(item["total_bits"] for item in manifest["enumeration"]["inputs"])
        for assignment_index in range(manifest["enumeration"]["total_assignments"]):
            assignment = _decode_assignment(assignment_index, input_widths)
            raw_inputs, choice = assignment[:len(inputs)], assignment[len(inputs):]
            signed_pt = [_signed_pt(raw, dtype) for raw, (_, dtype) in zip(raw_inputs, inputs)]
            label = {source.name: code for (source, _), code in zip(inputs, signed_pt)}
            # Assumes evaluation
            point = signed_pt + list(choice)
            assignment_label = {"inputs": label, "choices": choice}
            assumption_failure = "undefined assumptions"
            try:
                assumptions, *rounding_results = _apply_point(
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

            assumption_precision = _highest_precision(
                assumptions_machine, assignment_label, "assumption evaluation",
            )

            # C++ evaluation
            try:
                raw_output = int(evaluate_cpp(*raw_inputs, *choice))
            except CppAssertionError as exc:
                if assumption_failure is not None:
                    collector.append(assignment_index, "rejected", assumption_precision)
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
            results = _apply_point(
                machine, point + [cpp_code], assignment_label, "spec comparison",
            )
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
            # Resolve the reference rounding mode after assumption-based shrinking.
            if rounding_allowed:
                allowed_modes = []
                for result, (direction, _) in zip(rounding_results, rounding_allowed.values(), strict=True):
                    if (result.total_error or result.error or not result.resolved
                            or (result.value is not True and result.value is not False)):
                        raise VerificationInconclusive(
                            f"ZFV: {direction.value} assumption evaluation unresolved at {assignment_label}"
                        )
                    if result.value:
                        allowed_modes.append(direction)
                if not allowed_modes:
                    raise VerificationInconclusive(
                        f"ZFV: neither rounding direction satisfies assumptions at {assignment_label}"
                    )
                # Assumption-based correction supports exactly one faithful choice.
                selected_choice, = choice
                mode, _ = rounding_allowed[bool(selected_choice)]
                if mode not in allowed_modes:
                    mode = allowed_modes[0]
            else:
                mode = _selected_rounding_mode(
                    rounded_reference, dict(zip(free_vars, point, strict=True)), assignment_label,
                )

            working_precision = max(assumption_precision, _highest_precision(
                machine, assignment_label, "spec comparison",
            ))
            try:
                mpfi_value, = mpfi_reference.apply_point(point, precision=working_precision)
            except NotImplementedError as exc:
                raise VerificationInconclusive(
                    f"ZFV: MPFI evaluation inconclusive at {assignment_label}: {exc}"
                ) from exc
            _check_mpfi_reference(
                mpfi_value, Fraction(cpp_code, 1 << output_dtype.frac_bits),
                mode, precision, error_bound, inclusive, assignment_label,
            )
            collector.append(assignment_index, "valid", working_precision)
    finally:
        tempdir.cleanup()
    certificate = ZFVCertificate(manifest, tuple(collector.runs))
    if certificate.checked == 0:
        raise ZolotoneError("zfv found no reachable input points")
    return certificate


def replay_zfv(
    certificate: ZFVCertificate, component: Composite, *,
    require_same_checker: bool = True,
) -> ZFVCertificate:
    """Reconstruct the saved spec, generate a fresh certificate, and compare claims."""
    certificate.validate()
    spec = certificate.to_spec()
    fresh = zfv(spec, component)
    for field, expected in fresh.manifest.items():
        if field == "checker" and not require_same_checker:
            continue
        recorded = certificate.manifest[field]
        if field == "enumeration":
            # Packing order may differ; positional C++ argument descriptors may not.
            recorded = {**recorded, "inputs_order": expected["inputs_order"]}
        if recorded != expected:
            raise ValueError(f"ZFV replay manifest mismatch: {field}")

    recorded_runs = sorted(certificate.runs, key=lambda run: run.start)
    fresh_runs = sorted(fresh.runs, key=lambda run: run.start)

    def compare(recorded, observed, index):
        if recorded.outcome != observed.outcome:
            raise ZolotoneError(
                f"ZFV replay outcome mismatch at assignment {index}: "
                f"recorded {recorded.outcome}, observed {observed.outcome}"
            )

    layout = certificate.manifest["enumeration"]
    if layout["inputs_order"] == fresh.manifest["enumeration"]["inputs_order"]:
        # Walk overlapping runs without enumerating their individual assignments.
        left = right = 0
        while left < len(recorded_runs) and right < len(fresh_runs):
            recorded, observed = recorded_runs[left], fresh_runs[right]
            compare(recorded, observed, max(recorded.start, observed.start))
            left += recorded.stop <= observed.stop
            right += observed.stop <= recorded.stop
    else:
        from bisect import bisect_right

        # Map saved packed indices into the fresh certificate's argument order.
        starts = [run.start for run in fresh_runs]
        widths = [item["total_bits"] for item in fresh.manifest["enumeration"]["inputs"]]
        for recorded in recorded_runs:
            for index in range(recorded.start, recorded.stop + 1):
                packed = 0
                for value, width in zip(certificate.decode_assignment(index), widths, strict=True):
                    packed = (packed << width) | value
                compare(recorded, fresh_runs[bisect_right(starts, packed) - 1], index)
    return fresh
