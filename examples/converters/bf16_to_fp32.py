from zolotone import *


def spec_bf16_to_fp32(x: bf16, ctx):
    return Cases(
        case(x.is_nan, fp32.nan(ctx)),
        case(x.is_ninf, fp32.ninf(ctx)),
        case(x.is_pinf, fp32.inf(ctx)),
        case(x.is_nzero, fp32.nzero(ctx)),
        case(x.is_finite, fp32.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="bf16_to_fp32", spec=spec_bf16_to_fp32)
def bf16_to_fp32(x: Node) -> Node:
    X = bf16_decode(x)

    significand = effective_significand(X)
    significand = uq_resize(
        significand,
        1,
        max(BFloat16.mantissa_bits, Float32.mantissa_bits),
    )

    source_exponent = effective_exponent(X)
    target_exponent = q_add(
        uq_to_q(source_exponent),
        Const(Q.from_int(Float32.exponent_bias - BFloat16.exponent_bias)),
    )

    result = fp32_encode(X.sign, target_exponent, significand)

    encode_negative_zero = bit_and(X.is_zero, X.sign)
    result = if_then_else(
        encode_negative_zero,
        Const(Float32().nZero()),
        result,
    )

    encode_positive_infinity = bit_and(X.is_inf, bit_neg(X.sign))
    result = if_then_else(
        encode_positive_infinity,
        Const(Float32().Inf()),
        result,
    )

    encode_negative_infinity = bit_and(X.is_inf, X.sign)
    result = if_then_else(
        encode_negative_infinity,
        Const(Float32().nInf()),
        result,
    )

    result = if_then_else(
        X.is_nan,
        Const(Float32().NaN()),
        result,
    )
    return result


if __name__ == "__main__":
    cast = bf16_to_fp32(
        Var(name="x", dtype=BFloat16()),
    )

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/bf16_to_fp32_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/bf16_to_fp32_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
