from zolotone import *


def spec_fp32_to_e5m2fnuz(x: fp32, ctx):
    return Cases(
        case(x.is_nan, e5m2fnuz.nan(ctx)),
        case(x.is_ninf, e5m2fnuz.encode(ctx.real_val(-57344), ctx)),
        case(x.is_pinf, e5m2fnuz.encode(ctx.real_val(57344), ctx)),
        case(x.is_finite, e5m2fnuz.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="fp32_to_e5m2fnuz", spec=spec_fp32_to_e5m2fnuz)
def fp32_to_e5m2fnuz(x: Node) -> Node:
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
        max(Float32.mantissa_bits, E5M2FNUZ.mantissa_bits),
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
        Const(Q.from_int(E5M2FNUZ.exponent_bias - Float32.exponent_bias)),
    )

    result = e5m2fnuz_encode(X.sign, target_exponent, significand)

    encode_positive_infinity = bit_and(X.is_inf, bit_neg(X.sign))
    result = if_then_else(
        encode_positive_infinity,
        Const(E5M2FNUZ.from_fields(sign=0, exponent=31, mantissa=3)),
        result,
    )

    encode_negative_infinity = bit_and(X.is_inf, X.sign)
    result = if_then_else(
        encode_negative_infinity,
        Const(E5M2FNUZ.from_fields(sign=1, exponent=31, mantissa=3)),
        result,
    )

    result = if_then_else(
        X.is_nan,
        Const(E5M2FNUZ.NaN()),
        result,
    )
    return result


if __name__ == "__main__":
    cast = fp32_to_e5m2fnuz(
        Var(name="x", sign=Float32T()),
    )

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/fp32_to_e5m2fnuz_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/fp32_to_e5m2fnuz_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
