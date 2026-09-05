from zolotone import *


def spec_ue4m3x2_e2m1x2_mult_fp32(a0, a1, b0, b1, ctx):
    negative_sign = b0.sign.ne(b1.sign)
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


@Composite(
    name="ue4m3x2_e2m1x2_mult_fp32",
    spec=spec_ue4m3x2_e2m1x2_mult_fp32,
)
def ue4m3x2_e2m1x2_mult_fp32(
    a0: Node,
    a1: Node,
    b0: Node,
    b1: Node,
) -> Node:
    A0 = ue4m3_decode(a0)
    A1 = ue4m3_decode(a1)
    B0 = e2m1_decode(b0)
    B1 = e2m1_decode(b1)

    sign_bit = bit_xor(B0.sign, B1.sign)
    any_nan = bit_or(A0.is_nan, A1.is_nan)
    any_zero = bit_or(
        bit_or(A0.is_zero, A1.is_zero),
        bit_or(B0.is_zero, B1.is_zero),
    )

    # UQ1.3 x UQ1.3 x UQ1.1 x UQ1.1 is an exact UQ4.8 product.
    significand_product = uq_mul(
        uq_mul(
            effective_significand(A0),
            effective_significand(A1),
        ),
        uq_mul(
            effective_significand(B0),
            effective_significand(B1),
        ),
    )
    significand_product = uq_resize(
        significand_product,
        4,
        Float32.mantissa_bits,
    )

    effective_exponent_sum = uq_add(
        uq_add(effective_exponent(A0), effective_exponent(A1)),
        uq_add(effective_exponent(B0), effective_exponent(B1)),
    )
    fp32_exponent = q_add(
        uq_to_q(effective_exponent_sum),
        Const(
            Q.from_int(
                Float32.exponent_bias
                - 2 * UE4M3.exponent_bias
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
        Const(Float32().nZero()),
        Const(Float32().Zero()),
    )
    return if_then_else(
        any_nan,
        Const(Float32().NaN()),
        if_then_else(any_zero, signed_zero, finite_result),
    )


if __name__ == "__main__":
    multiplier = ue4m3x2_e2m1x2_mult_fp32(
        Var(name="a0", dtype=UE4M3()),
        Var(name="a1", dtype=UE4M3()),
        Var(name="b0", dtype=E2M1()),
        Var(name="b1", dtype=E2M1()),
    )

    multiplier.check_determinism()
    multiplier.check_spec()

    with open(
        "examples/c_models/ue4m3x2_e2m1x2_mult_fp32_jit.hpp",
        "w",
    ) as file:
        file.write(multiplier.to_cpp(jittable=True))

    with open(
        "examples/c_models/ue4m3x2_e2m1x2_mult_fp32_no_jit.hpp",
        "w",
    ) as file:
        file.write(multiplier.to_cpp(jittable=False))
