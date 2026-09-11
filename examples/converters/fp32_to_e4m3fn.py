from zolotone import *


def spec_fp32_to_e4m3fn(x: fp32, ctx):
    return Cases(
        case(x.is_nan, e4m3fn.nan(ctx)),
        case(x.is_ninf, e4m3fn.encode(ctx.real_val(-448), ctx)),
        case(x.is_pinf, e4m3fn.encode(ctx.real_val(448), ctx)),
        case(x.is_nzero, e4m3fn.nzero(ctx)),
        case(x.is_finite, e4m3fn.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="fp32_to_e4m3fn", spec=spec_fp32_to_e4m3fn)
def fp32_to_e4m3fn(x: Node) -> Node:
    X = fp32_decode(x)

    significand = effective_significand(X)
    significand = uq_resize(
        significand,
        1,
        max(Float32.mantissa_bits, E4M3FN.mantissa_bits),
    )

    source_exponent = effective_exponent(X)
    target_exponent = q_add(
        uq_to_q(source_exponent),
        Const(Q.from_int(E4M3FN.exponent_bias - Float32.exponent_bias)),
    )

    result = e4m3fn_encode(X.sign, target_exponent, significand)

    encode_negative_zero = bit_and(X.is_zero, X.sign)
    result = if_then_else(
        encode_negative_zero,
        Const(E4M3FN().nZero()),
        result,
    )

    encode_positive_infinity = bit_and(X.is_inf, bit_neg(X.sign))
    result = if_then_else(
        encode_positive_infinity,
        Const(E4M3FN().from_fields(sign=0, exponent=15, mantissa=6)),
        result,
    )

    encode_negative_infinity = bit_and(X.is_inf, X.sign)
    result = if_then_else(
        encode_negative_infinity,
        Const(E4M3FN().from_fields(sign=1, exponent=15, mantissa=6)),
        result,
    )

    result = if_then_else(
        X.is_nan,
        Const(E4M3FN().NaN()),
        result,
    )
    return result


if __name__ == "__main__":
    cast = fp32_to_e4m3fn(
        Var(name="x", dtype=Float32()),
    )

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/fp32_to_e4m3fn_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/fp32_to_e4m3fn_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
