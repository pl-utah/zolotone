from zolotone import *


def spec_e4m3fn_to_fp32(x: e4m3fn, ctx):
    return Cases(
        case(x.is_nan, fp32.nan(ctx)),
        case(x.is_nzero, fp32.nzero(ctx)),
        case(x.is_finite, fp32.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="e4m3fn_to_fp32", spec=spec_e4m3fn_to_fp32)
def e4m3fn_to_fp32(x: Node) -> Node:
    X = e4m3fn_decode(x)

    mantissa_fraction = uq_integer_to_fraction(X.mantissa)
    significand = if_then_else(
        X.is_norm,
        add_implicit_bit(mantissa_fraction),
        uq_resize(mantissa_fraction, 1, E4M3FN.mantissa_bits),
    )
    significand = uq_resize(
        significand,
        1,
        max(E4M3FN.mantissa_bits, Float32.mantissa_bits),
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
        Const(Q.from_int(Float32.exponent_bias - E4M3FN.exponent_bias)),
    )

    result = fp32_encode(X.sign, target_exponent, significand)

    encode_negative_zero = bit_and(X.is_zero, X.sign)
    result = if_then_else(
        encode_negative_zero,
        Const(Float32.nZero()),
        result,
    )

    result = if_then_else(
        X.is_nan,
        Const(Float32.NaN()),
        result,
    )
    return result


if __name__ == "__main__":
    cast = e4m3fn_to_fp32(
        Var(name="x", sign=E4M3FNT()),
    )

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/e4m3fn_to_fp32_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/e4m3fn_to_fp32_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
