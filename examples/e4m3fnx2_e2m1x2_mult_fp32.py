from zolotone import *


def spec_e4m3fnx2_e2m1x2_mult_fp32(a0, a1, b0, b1, ctx):
    negative_sign = a0.sign.ne(a1.sign).ne(b0.sign.ne(b1.sign))
    nan_case = a0.is_nan | a1.is_nan
    zero_case = a0.is_zero | a1.is_zero | b0.is_zero | b1.is_zero
    finite_value = a0.value * a1.value * b0.value * b1.value

    return Cases(
        case(nan_case, fp32.nan(ctx)),
        case(zero_case & negative_sign, fp32.nzero(ctx)),
        case(
            a0.is_finite & a1.is_finite & b0.is_finite & b1.is_finite,
            fp32.encode(finite_value, ctx),
        ),
        ctx=ctx,
    )


def _significand(decoded, mantissa_bits):
    fraction = uq_integer_to_fraction(decoded.mantissa)
    return if_then_else(
        decoded.is_norm,
        add_implicit_bit(fraction),
        uq_resize(fraction, 1, mantissa_bits),
    )


def _effective_exponent(decoded):
    subnormal_exponent = Const(
        UQ(
            1,
            decoded.exponent.node_type.int_bits,
            decoded.exponent.node_type.frac_bits,
        )
    )
    return if_then_else(
        decoded.is_sub,
        subnormal_exponent,
        decoded.exponent,
    )


@Composite(
    name="e4m3fnx2_e2m1x2_mult_fp32",
    spec=spec_e4m3fnx2_e2m1x2_mult_fp32,
)
def e4m3fnx2_e2m1x2_mult_fp32(
    a0: Node,
    a1: Node,
    b0: Node,
    b1: Node,
) -> Node:
    A0 = e4m3fn_decode(a0)
    A1 = e4m3fn_decode(a1)
    B0 = e2m1_decode(b0)
    B1 = e2m1_decode(b1)

    sign_bit = bit_xor(
        bit_xor(A0.sign, A1.sign),
        bit_xor(B0.sign, B1.sign),
    )
    any_nan = bit_or(A0.is_nan, A1.is_nan)
    any_zero = bit_or(
        bit_or(A0.is_zero, A1.is_zero),
        bit_or(B0.is_zero, B1.is_zero),
    )

    # UQ1.3 x UQ1.3 x UQ1.1 x UQ1.1 is an exact UQ4.8 product.
    significand_product = uq_mul(
        uq_mul(
            _significand(A0, E4M3FN.mantissa_bits),
            _significand(A1, E4M3FN.mantissa_bits),
        ),
        uq_mul(
            _significand(B0, E2M1.mantissa_bits),
            _significand(B1, E2M1.mantissa_bits),
        ),
    )
    significand_product = uq_resize(
        significand_product,
        4,
        Float32.mantissa_bits,
    )

    effective_exponent_sum = uq_add(
        uq_add(_effective_exponent(A0), _effective_exponent(A1)),
        uq_add(_effective_exponent(B0), _effective_exponent(B1)),
    )
    fp32_exponent = q_add(
        uq_to_q(effective_exponent_sum),
        Const(
            Q.from_int(
                Float32.exponent_bias
                - 2 * E4M3FN.exponent_bias
                - 2 * E2M1.exponent_bias
            )
        ),
    )

    finite_result = fp32_encode(
        sign_bit,
        fp32_exponent,
        significand_product,
    )
    signed_zero = if_then_else(
        sign_bit,
        Const(Float32.nZero()),
        Const(Float32.Zero()),
    )
    return if_then_else(
        any_nan,
        Const(Float32.NaN()),
        if_then_else(any_zero, signed_zero, finite_result),
    )


if __name__ == "__main__":
    multiplier = e4m3fnx2_e2m1x2_mult_fp32(
        Var(name="a0", sign=E4M3FNT()),
        Var(name="a1", sign=E4M3FNT()),
        Var(name="b0", sign=E2M1T()),
        Var(name="b1", sign=E2M1T()),
    )

    # multiplier.check_determinism()
    multiplier.check_spec()

    with open(
        "examples/c_models/e4m3fnx2_e2m1x2_mult_fp32_jit.hpp",
        "w",
    ) as file:
        file.write(multiplier.to_cpp(jittable=True))

    with open(
        "examples/c_models/e4m3fnx2_e2m1x2_mult_fp32_no_jit.hpp",
        "w",
    ) as file:
        file.write(multiplier.to_cpp(jittable=False))
