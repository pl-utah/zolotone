import math
import typing as tp
import warnings

from ..errors import InfeasibleError, MissingError, ZolotoneError
from ..rival import rival_range_analysis
from ..solver import check_equivalence
from ..spec.spec_ast import BoolExpr, RealExpr, SpecNode
from ..spec.spec_context import SpecContext
from ..types import Bool, DataType, Q, UQ
from .nodes import _SpecContract


FEASIBILITY_SCHEDULE = [
    {"tool": "simplify"},
    {"tool": "z3", "timeout_ms": 10_000},
]


def reject_untyped_inputs(contract: _SpecContract):
    input_parameters = list(contract.signature.parameters.values())[:-1]
    for parameter in input_parameters:
        annotation = contract.annotations[parameter.name]
        if not isinstance(annotation, DataType):
            raise MissingError(
                f"Specification {contract.display_name} input "
                f"parameter {parameter.name!r} must have an exact DataType "
                f"descriptor, got {annotation!r}"
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


def _prove_result_fits(spec_ast, output_type, ctx):
    if output_type is Q:
        return True
    if output_type is UQ:
        result_fits = spec_ast >= ctx.zero()
    elif isinstance(output_type, (Q, UQ)):
        result_fits = _fixed_point_result_fits(spec_ast, output_type, ctx)
    else:
        raise TypeError(
            "Result-fit proof expects a Q or UQ output type, got "
            f"{output_type!r}"
        )

    range_ctx = ctx.copy(checks=[result_fits])
    status, _proof_trace = check_equivalence(range_ctx, schedule=FEASIBILITY_SCHEDULE)
    return status == "unsat"


def _output_format_suggestion_with_search(
    spec_ast: RealExpr,
    conservative_format: Q | UQ,
    ctx: SpecContext,
) -> Q | UQ | None:
    """Shrink a conservative format using proofs over the full context."""
    dtype_type = UQ if _prove_result_fits(spec_ast, UQ, ctx) else Q

    def candidate(int_bits: int) -> Q | UQ:
        return dtype_type(int_bits, conservative_format.frac_bits)

    min_int_bits = max(0, 1 - conservative_format.frac_bits)
    max_int_bits = conservative_format.int_bits
    widest = candidate(max_int_bits)

    if not _prove_result_fits(spec_ast, widest, ctx):
        return None

    low = min_int_bits
    high = max_int_bits
    while low < high:
        middle = (low + high) // 2
        middle_dtype = candidate(middle)
        if _prove_result_fits(spec_ast, middle_dtype, ctx):
            high = middle
        else:
            low = middle + 1
    return candidate(low)


def _output_format_suggestion_with_range_analysis(
    spec_ast: RealExpr,
    return_annotation: Q | UQ,
    ctx: SpecContext,
) -> Q | UQ:
    output_range = rival_range_analysis(spec_ast, ctx)
    if output_range is None:
        raise ZolotoneError(f"Could not obtain output range for: {spec_ast}")

    lower, upper = output_range
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ZolotoneError(
            f"Could not obtain finite output range, got {output_range} "
            f"for {spec_ast}"
        )

    frac_bits = return_annotation.frac_bits
    scale = 1 << frac_bits
    if lower >= 0:
        required_raw = max(0, math.ceil(upper * scale))
        total_bits = max(1, frac_bits, required_raw.bit_length())
        int_bits = total_bits - frac_bits
        return UQ(int_bits, frac_bits)

    lower_raw = math.floor(lower * scale)
    upper_raw = math.ceil(upper * scale)
    required_magnitude = max(1, -lower_raw, upper_raw + 1)
    magnitude_bits = (required_magnitude - 1).bit_length()
    total_bits = max(1, frac_bits, magnitude_bits + 1)
    int_bits = total_bits - frac_bits
    return Q(int_bits, frac_bits)


def _output_format_suggestion(
    spec_ast: RealExpr,
    return_annotation: object,
    ctx: SpecContext,
) -> object | None:
    """Derive a conservative format with Rival, then shrink it by proof."""
    # TODO: provide a counterexample
    if return_annotation is UQ:
        if not _prove_result_fits(spec_ast, UQ, ctx):
            return Q
        return UQ
    elif return_annotation is Q:
        if _prove_result_fits(spec_ast, UQ, ctx):
            return UQ
        return Q
    elif isinstance(return_annotation, (Q, UQ)):
        conservative_format = _output_format_suggestion_with_range_analysis(
            spec_ast,
            return_annotation,
            ctx,
        )
        suggestion = _output_format_suggestion_with_search(
            spec_ast,
            conservative_format,
            ctx,
        )
        if suggestion is None:
            raise ValueError(
                "Rival-derived format does not fit the result: "
                f"{conservative_format}"
            )
        if not _prove_result_fits(spec_ast, suggestion, ctx):
            raise ValueError("That's a bug, derived dtype does not fit the result")
        return suggestion
    else:
        raise NotImplementedError(
            "Output format suggestion is not implemented for "
            f"{return_annotation!r}"
        )


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

    if not isinstance(spec_ast, RealExpr):
        raise TypeError(
            f"Specification returning {return_annotation!r} must produce "
            f"a real expression, got {type(spec_ast).__name__}"
        )

    if return_annotation not in (Q, UQ) and not isinstance(return_annotation, (Q, UQ)):
        raise NotImplementedError(
            f"Output format {return_annotation!r} is not supported by "
            "the feasibility check"
        )

    # TODO: counterexample
    if not _prove_result_fits(spec_ast, return_annotation, ctx):
        suggestion = _output_format_suggestion(spec_ast, return_annotation, ctx)
        message = f"Specification result range does not fit {return_annotation!r}"
        if suggestion is not None:
            message += f"; try {_format_output_annotation(suggestion)} as the output format instead"
        else:
            message += "; could not find a fixed-point format that would fit the range"
        raise InfeasibleError(message)

    suggestion = _output_format_suggestion(spec_ast, return_annotation, ctx)
    if suggestion != return_annotation:
        warnings.warn(
            f"Output type {return_annotation} is wider than necessary; consider {suggestion}",
            UserWarning,
            stacklevel=3,
        )
    return
