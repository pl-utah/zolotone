from zolotone import *

from .common import *
from .encode_BFloat16 import *


def spec_fp32_to_bf16(x: fp32, ctx):
    return Cases(
        case(x.is_nan, bf16.nan(ctx)),
        case(x.is_ninf, bf16.ninf(ctx)),
        case(x.is_pinf, bf16.inf(ctx)),
        case(x.is_nzero, bf16.nzero(ctx)),
        case(x.is_finite, bf16.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="fp32_to_bf16", spec=spec_fp32_to_bf16)
def fp32_to_bf16(x: Node) -> Node:
    X = fp32_decode(x)

    # UQ<23, 0> -> UQ<0, 23>.
    mantissa_fraction = integer_to_fraction(X.mantissa)
    significand = if_then_else(
        X.is_norm,
        add_implicit_bit(mantissa_fraction),
        uq_resize(mantissa_fraction, 1, Float32.mantissa_bits),
    )

    # Subnormals store exponent zero but behave as exponent 1-bias.
    subnormal_exponent = Const(
        UQ(1, X.exponent.node_type.int_bits, X.exponent.node_type.frac_bits)
    )
    effective_exponent = if_then_else(
        X.is_sub,
        subnormal_exponent,
        X.exponent,
    )

    finite_result = bf16_encode(
        X.sign,
        uq_to_q(effective_exponent),
        significand,
    )

    encode_ninf = bit_and(X.is_inf, X.sign)
    encode_pinf = bit_and(X.is_inf, bit_neg(X.sign))
    encode_nzero = bit_and(X.is_zero, X.sign)

    return if_then_else(
        X.is_nan,
        Const(BFloat16.NaN()),
        if_then_else(
            encode_ninf,
            Const(BFloat16.nInf()),
            if_then_else(
                encode_pinf,
                Const(BFloat16.Inf()),
                if_then_else(
                    encode_nzero,
                    Const(BFloat16.nZero()),
                    finite_result,
                ),
            ),
        ),
    )


if __name__ == "__main__":
    cast = fp32_to_bf16(Var(name="x", sign=Float32T()))

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/fp32_to_bf16_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/fp32_to_bf16_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
