from zolotone import *


def spec_fp32_to_e5m2(x: fp32, ctx):
    return Cases(
        case(x.is_nan, e5m2.nan(ctx)),
        case(x.is_ninf, e5m2.ninf(ctx)),
        case(x.is_pinf, e5m2.inf(ctx)),
        case(x.is_nzero, e5m2.nzero(ctx)),
        case(x.is_finite, e5m2.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="fp32_to_e5m2", spec=spec_fp32_to_e5m2)
def fp32_to_e5m2(x: Node) -> Node:
    X = fp32_decode(x)

    significand = effective_significand(X)
    significand = uq_resize(
        significand,
        1,
        max(Float32.mantissa_bits, E5M2.mantissa_bits),
    )

    source_exponent = effective_exponent(X)
    target_exponent = q_add(
        uq_to_q(source_exponent),
        Const(Q.from_int(E5M2.exponent_bias - Float32.exponent_bias)),
    )

    result = e5m2_encode(X.sign, target_exponent, significand)

    encode_negative_zero = bit_and(X.is_zero, X.sign)
    result = if_then_else(
        encode_negative_zero,
        Const(E5M2.nZero()),
        result,
    )

    encode_positive_infinity = bit_and(X.is_inf, bit_neg(X.sign))
    result = if_then_else(
        encode_positive_infinity,
        Const(E5M2.Inf()),
        result,
    )

    encode_negative_infinity = bit_and(X.is_inf, X.sign)
    result = if_then_else(
        encode_negative_infinity,
        Const(E5M2.nInf()),
        result,
    )

    result = if_then_else(
        X.is_nan,
        Const(E5M2.NaN()),
        result,
    )
    return result


if __name__ == "__main__":
    cast = fp32_to_e5m2(
        Var(name="x", sign=Float32T()),
    )

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/fp32_to_e5m2_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/fp32_to_e5m2_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
