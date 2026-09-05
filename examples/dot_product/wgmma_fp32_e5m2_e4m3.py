"""Reduced fused ``wgmma ... f32.e5m2.e4m3`` dot-accumulate model."""

from pathlib import Path

from zolotone import *


N = 4


def spec_wgmma_fp32_e5m2_e4m3(
    a0: e5m2, a1: e5m2, a2: e5m2, a3: e5m2,
    b0: e4m3fn, b1: e4m3fn, b2: e4m3fn, b3: e4m3fn,
    c: fp32, ctx
):
    A = (a0, a1, a2, a3)
    B = (b0, b1, b2, b3)
    product_signs = [A[i].sign.ne(B[i].sign) for i in range(N)]
    positive_infinity = ormap(
        c.is_inf & c.sign.eq(ctx.zero()),
        *[A[i].is_inf & (~product_signs[i]) for i in range(N)],
    )
    negative_infinity = ormap(
        c.is_inf & c.sign.eq(ctx.one()),
        *[A[i].is_inf & product_signs[i] for i in range(N)],
    )
    nan_case = ormap(
        c.is_nan,
        *[value.is_nan for value in (*A, *B)],
        *[A[i].is_inf & B[i].is_zero for i in range(N)],
        positive_infinity & negative_infinity,
    )
    finite_value =  sum(
        (A[i].value * B[i].value for i in range(N)), c.value
    )
    return Cases(
        case(nan_case, fp32.nan(ctx)),
        case(negative_infinity, fp32.ninf(ctx)),
        case(positive_infinity, fp32.inf(ctx)),
        case(andmap(c.is_finite, *[value.is_finite for value in (*A, *B)]),
            fp32.encode(finite_value, ctx),
        ),
        ctx=ctx,
    )


@Composite(name="wgmma_fp32_e5m2_e4m3", spec=spec_wgmma_fp32_e5m2_e4m3)
def wgmma_fp32_e5m2_e4m3(
    a0: Node,
    a1: Node,
    a2: Node,
    a3: Node,
    b0: Node,
    b1: Node,
    b2: Node,
    b3: Node,
    c: Node,
) -> Node:
    A = tuple(e5m2_decode(value) for value in (a0, a1, a2, a3))
    B = tuple(e4m3fn_decode(value) for value in (b0, b1, b2, b3))
    C = fp32_decode(c)

    product_signs = [bit_xor(A[i].sign, B[i].sign) for i in range(N)]
    positive_infinity = bit_and(C.is_inf, bit_neg(C.sign))
    negative_infinity = bit_and(C.is_inf, C.sign)
    invalid_product = Const(UQ(1, 0).from_bits(0))
    encode_nan = Const(UQ(1, 0).from_bits(0))
    for i in range(N):
        positive_infinity = bit_or(
            positive_infinity,
            bit_and(A[i].is_inf, bit_neg(product_signs[i])),
        )
        negative_infinity = bit_or(
            negative_infinity,
            bit_and(A[i].is_inf, product_signs[i]),
        )
        invalid_product = bit_or(
            invalid_product, bit_and(A[i].is_inf, B[i].is_zero)
        )
        encode_nan = bit_or(
            encode_nan, bit_or(A[i].is_nan, B[i].is_nan)
        )
    encode_nan = bit_or(
        C.is_nan,
        bit_or(
            encode_nan,
            bit_or(
                invalid_product,
                bit_and(positive_infinity, negative_infinity),
            ),
        ),
    )
    encode_ninf = bit_and(bit_neg(encode_nan), negative_infinity)
    encode_pinf = bit_and(bit_neg(encode_nan), positive_infinity)

    product_exponents = [
        uq_add(effective_exponent(A[i]), effective_exponent(B[i])) for i in range(N)
    ]
    product_exponents_for_alignment = [
        if_then_else(
            bit_or(A[i].is_zero, B[i].is_zero),
            Const(
                UQ(product_exponents[i].dtype.int_bits, product_exponents[i].dtype.frac_bits).from_bits(0)
            ),
            product_exponents[i],
        )
        for i in range(N)
    ]
    maximum_product_exponent = uq_max(
        uq_max(
            product_exponents_for_alignment[0],
            product_exponents_for_alignment[1],
        ),
        uq_max(
            product_exponents_for_alignment[2],
            product_exponents_for_alignment[3],
        ),
    )
    
    # Each exact UQ2.5 product receives 43 low zero bits, enough for the
    # complete finite E5M2/E4M3 product-exponent span.
    products = [
        uq_resize(
            uq_mul(effective_significand(A[i]), effective_significand(B[i])),
            2,
            48,
        )
        for i in range(N)
    ]
    aligned_products = [
        uq_rshift(
            products[i],
            uq_sub(
                maximum_product_exponent,
                product_exponents_for_alignment[i],
            ),
        )
        for i in range(N)
    ]
    signed_products = [
        q_add_sign(uq_to_q(aligned_products[i]), product_signs[i])
        for i in range(N)
    ]
    product_sum = q_add(
        q_add(signed_products[0], signed_products[1]),
        q_add(signed_products[2], signed_products[3]),
    )

    # Product exponents have bias 22. Convert them to FP32's bias 127.
    product_exponent = uq_add(
        maximum_product_exponent,
        Const(
            UQ.from_int(
                Float32.exponent_bias
                - E5M2.exponent_bias
                - E4M3FN.exponent_bias
            )
        ),
    )
    product_exponent = if_then_else(
        q_is_zero(product_sum),
        Const(
            UQ(product_exponent.dtype.int_bits, product_exponent.dtype.frac_bits).from_bits(0)
        ),
        product_exponent,
    )
    c_exponent = effective_exponent(C)
    common_exponent = uq_max(product_exponent, c_exponent)

    product_sum_with_grs = q_resize(
        product_sum,
        product_sum.dtype.int_bits,
        product_sum.dtype.frac_bits + 3,
    )
    
    c_significand = effective_significand(C)
    signed_c = q_resize(
        q_add_sign(uq_to_q(c_significand), C.sign),
        product_sum_with_grs.dtype.int_bits,
        product_sum_with_grs.dtype.frac_bits,
    )
    
    aligned_product_sum = q_rshift_jam(
        product_sum_with_grs,
        uq_sub(common_exponent, product_exponent),
    )
    aligned_c = q_rshift_jam(
        signed_c,
        uq_sub(common_exponent, c_exponent),
    )
    finite_sum = q_add(aligned_product_sum, aligned_c)
    finite_result = fp32_encode(
        q_sign_bit(finite_sum),
        uq_to_q(common_exponent),
        q_to_uq(q_abs(finite_sum)),
    )

    return if_then_else(
        encode_nan,
        Const(Float32().NaN()),
        if_then_else(
            encode_ninf,
            Const(Float32().nInf()),
            if_then_else(encode_pinf, Const(Float32().Inf()), finite_result),
        ),
    )


if __name__ == "__main__":
    design = wgmma_fp32_e5m2_e4m3(
        *[Var(name=f"a{index}", dtype=E5M2()) for index in range(N)],
        *[Var(name=f"b{index}", dtype=E4M3FN()) for index in range(N)],
        Var(name="c", dtype=Float32()),
    )
    design.check_determinism()
    design.check_spec()
    output_directory = Path(__file__).resolve().parents[1] / "c_models"
    output_directory.mkdir(exist_ok=True)
    for jittable, suffix in ((True, "jit"), (False, "no_jit")):
        (output_directory / f"wgmma_fp32_e5m2_e4m3_{suffix}.hpp").write_text(
            design.to_cpp(jittable=jittable), encoding="utf-8"
        )


__all__ = [
    "spec_wgmma_fp32_e5m2_e4m3",
    "wgmma_fp32_e5m2_e4m3",
]
