from zolotone import *

from .common import *
from .encode_Float32 import fp32_encode


def spec_fp16_to_fp32(x: fp16, ctx):
    return Cases(
        case(x.is_nan, fp32.nan(ctx)),
        case(x.is_ninf, fp32.ninf(ctx)),
        case(x.is_pinf, fp32.inf(ctx)),
        case(x.is_nzero, fp32.nzero(ctx)),
        case(x.is_finite, fp32.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="fp16_to_fp32", spec=spec_fp16_to_fp32)
def fp16_to_fp32(x: Node) -> Node:
    X = fp16_decode(x)

    mantissa_fraction = integer_to_fraction(X.mantissa)
    significand = if_then_else(
        X.is_norm,
        add_implicit_bit(mantissa_fraction),
        uq_resize(mantissa_fraction, 1, Float16.mantissa_bits),
    )
    # Widen the exact ten-bit FP16 significand to FP32's 23-bit precision.
    # The encoder's additional bits are reserved for G/R/S rounding state.
    significand = uq_resize(significand, 1, Float32.mantissa_bits)

    subnormal_exponent = Const(
        UQ(1, X.exponent.node_type.int_bits, X.exponent.node_type.frac_bits)
    )
    effective_exponent = if_then_else(
        X.is_sub,
        subnormal_exponent,
        X.exponent,
    )
    target_exponent = q_add(
        uq_to_q(effective_exponent),
        Const(Q.from_int(Float32.exponent_bias - Float16.exponent_bias)),
    )

    finite_result = fp32_encode(X.sign, target_exponent, significand)

    encode_ninf = bit_and(X.is_inf, X.sign)
    encode_pinf = bit_and(X.is_inf, bit_neg(X.sign))
    encode_nzero = bit_and(X.is_zero, X.sign)

    return if_then_else(
        X.is_nan,
        Const(Float32.NaN()),
        if_then_else(
            encode_ninf,
            Const(Float32.nInf()),
            if_then_else(
                encode_pinf,
                Const(Float32.Inf()),
                if_then_else(
                    encode_nzero,
                    Const(Float32.nZero()),
                    finite_result,
                ),
            ),
        ),
    )


if __name__ == "__main__":
    cast = fp16_to_fp32(Var(name="x", sign=Float16T()))

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/fp16_to_fp32_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/fp16_to_fp32_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
