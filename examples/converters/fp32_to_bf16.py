from zolotone import *


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

    mantissa_fraction = uq_integer_to_fraction(X.mantissa)
    significand = if_then_else(
        X.is_norm,
        add_implicit_bit(mantissa_fraction),
        uq_resize(mantissa_fraction, 1, Float32.mantissa_bits),
    )
    significand = uq_resize(
        significand,
        1,
        max(Float32.mantissa_bits, BFloat16.mantissa_bits),
    )

    subnormal_exponent = Const(
        UQ(
            1,
            X.exponent.node_type.int_bits,
            X.exponent.node_type.frac_bits,
        )
    )
    effective_exponent = if_then_else(
        X.is_sub,
        subnormal_exponent,
        X.exponent,
    )
    target_exponent = q_add(
        uq_to_q(effective_exponent),
        Const(Q.from_int(BFloat16.exponent_bias - Float32.exponent_bias)),
    )

    result = bf16_encode(X.sign, target_exponent, significand)

    encode_negative_zero = bit_and(X.is_zero, X.sign)
    result = if_then_else(
        encode_negative_zero,
        Const(BFloat16.nZero()),
        result,
    )

    encode_positive_infinity = bit_and(X.is_inf, bit_neg(X.sign))
    result = if_then_else(
        encode_positive_infinity,
        Const(BFloat16.Inf()),
        result,
    )

    encode_negative_infinity = bit_and(X.is_inf, X.sign)
    result = if_then_else(
        encode_negative_infinity,
        Const(BFloat16.nInf()),
        result,
    )

    result = if_then_else(
        X.is_nan,
        Const(BFloat16.NaN()),
        result,
    )
    return result


if __name__ == "__main__":
    cast = fp32_to_bf16(
        Var(name="x", sign=Float32T()),
    )

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/fp32_to_bf16_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/fp32_to_bf16_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
