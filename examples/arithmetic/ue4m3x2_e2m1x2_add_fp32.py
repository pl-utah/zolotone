from zolotone import *


def spec_ue4m3x2_e2m1x2_add_fp32(scale0, scale1, x0, x1, ctx):
    nan_case = scale0.is_nan | scale1.is_nan
    term0_is_zero = scale0.is_zero | x0.is_zero
    term1_is_zero = scale1.is_zero | x1.is_zero
    negative_zero_case = (
        term0_is_zero
        & term1_is_zero
        & x0.sign.eq(ctx.one())
        & x1.sign.eq(ctx.one())
    )
    finite_value = scale0.value * x0.value + scale1.value * x1.value
    
    return Cases(
        case(nan_case, fp32.nan(ctx)),
        case(negative_zero_case, fp32.nzero(ctx)),
        case(
            scale0.is_finite
            & scale1.is_finite
            & x0.is_finite
            & x1.is_finite,
            fp32.encode(finite_value, ctx),
        ),
        ctx=ctx,
    )


@Composite(
    name="ue4m3x2_e2m1x2_add_fp32",
    spec=spec_ue4m3x2_e2m1x2_add_fp32,
)
def ue4m3x2_e2m1x2_add_fp32(
    scale0: Node,
    scale1: Node,
    x0: Node,
    x1: Node,
) -> Node:
    Scale0 = ue4m3_decode(scale0)
    Scale1 = ue4m3_decode(scale1)
    X0 = e2m1_decode(x0)
    X1 = e2m1_decode(x1)
    
    any_nan = bit_or(Scale0.is_nan, Scale1.is_nan)
    term0_is_zero = bit_or(Scale0.is_zero, X0.is_zero)
    term1_is_zero = bit_or(Scale1.is_zero, X1.is_zero)
    encode_negative_zero = bit_and(
        bit_and(term0_is_zero, term1_is_zero),
        bit_and(X0.sign, X1.sign),
    )
    
    scale0_exponent = effective_exponent(Scale0)
    scale1_exponent = effective_exponent(Scale1)
    x0_exponent = effective_exponent(X0)
    x1_exponent = effective_exponent(X1)
    product0_exponent = uq_add(scale0_exponent, x0_exponent)
    product1_exponent = uq_add(scale1_exponent, x1_exponent)
    common_product_exponent = uq_max(product0_exponent, product1_exponent)
    
    # Each UE4M3/E2M1 product is an exact UQ2.4 value. The maximum exponent
    # difference between nonzero products is 16, so UQ2.20 retains every bit
    # while aligning the two products.
    product0 = uq_resize(
        uq_mul(
            effective_significand(Scale0),
            effective_significand(X0),
        ),
        2,
        20,
    )
    product1 = uq_resize(
        uq_mul(
            effective_significand(Scale1),
            effective_significand(X1),
        ),
        2,
        20,
    )
    product0 = uq_rshift(
        product0,
        uq_sub(common_product_exponent, product0_exponent),
    )
    product1 = uq_rshift(
        product1,
        uq_sub(common_product_exponent, product1_exponent),
    )
    
    signed_product0 = q_add_sign(uq_to_q(product0), X0.sign)
    signed_product1 = q_add_sign(uq_to_q(product1), X1.sign)
    product_sum = q_add(signed_product0, signed_product1)
    
    # The product exponent is biased by UE4M3 bias 7 plus E2M1 bias 1.
    fp32_exponent = uq_add(
        common_product_exponent,
        Const(
            UQ.from_int(
                Float32.exponent_bias
                - UE4M3.exponent_bias
                - E2M1.exponent_bias
            )
        ),
    )
    finite_result = fp32_encode(
        q_sign_bit(product_sum),
        uq_to_q(fp32_exponent),
        q_to_uq(q_abs(product_sum)),
    )
    
    return if_then_else(
        any_nan,
        Const(Float32.NaN()),
        if_then_else(
            encode_negative_zero,
            Const(Float32.nZero()),
            finite_result,
        ),
    )


if __name__ == "__main__":
    design = ue4m3x2_e2m1x2_add_fp32(
        Var(name="scale0", sign=UE4M3T()),
        Var(name="scale1", sign=UE4M3T()),
        Var(name="x0", sign=E2M1T()),
        Var(name="x1", sign=E2M1T()),
    )
    
    design.check_determinism()
    design.check_spec()
    
    with open(
        "examples/c_models/ue4m3x2_e2m1x2_add_fp32_jit.hpp",
        "w",
    ) as file:
        file.write(design.to_cpp(jittable=True))
    
    with open(
        "examples/c_models/ue4m3x2_e2m1x2_add_fp32_no_jit.hpp",
        "w",
    ) as file:
        file.write(design.to_cpp(jittable=False))
