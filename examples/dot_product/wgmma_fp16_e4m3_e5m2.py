"""Reduced fused ``wgmma ... f16.e4m3.e5m2`` dot-accumulate model."""

from pathlib import Path

from zolotone import *


N = 4


def spec_wgmma_fp16_e4m3_e5m2(
    a0: e4m3fn, a1: e4m3fn, a2: e4m3fn, a3: e4m3fn,
    b0: e5m2, b1: e5m2, b2: e5m2, b3: e5m2,
    c: fp16, ctx
):
    A = (a0, a1, a2, a3)
    B = (b0, b1, b2, b3)
    product_signs = [A[i].sign.ne(B[i].sign) for i in range(N)]
    positive_infinity = ormap(
        c.is_pinf,
        *[B[i].is_inf & (~product_signs[i]) for i in range(N)],
    )
    negative_infinity = ormap(
        c.is_ninf,
        *[B[i].is_inf & product_signs[i] for i in range(N)],
    )
    nan_case = ormap(
        c.is_nan,
        *[value.is_nan for value in (*A, *B)],
        *[B[i].is_inf & A[i].is_zero for i in range(N)],
        positive_infinity & negative_infinity,
    )
    finite_value = sum([A[i].value * B[i].value for i in range(N)], c.value)
    return Cases(
        case(nan_case, fp16.nan(ctx)),
        case(negative_infinity, fp16.ninf(ctx)),
        case(positive_infinity, fp16.inf(ctx)),
        case(andmap(c.is_finite, *[value.is_finite for value in (*A, *B)]),
            fp16.encode(finite_value, ctx),
        ),
        ctx=ctx,
    )


@Composite(name="wgmma_fp16_e4m3_e5m2", spec=spec_wgmma_fp16_e4m3_e5m2)
def wgmma_fp16_e4m3_e5m2(
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
    A = tuple(e4m3fn_decode(value) for value in (a0, a1, a2, a3))
    B = tuple(e5m2_decode(value) for value in (b0, b1, b2, b3))
    C = fp16_decode(c)

    # Special case handling
    product_signs = [bit_xor(A[i].sign, B[i].sign) for i in range(N)]
    positive_infinity = bit_and(C.is_inf, bit_neg(C.sign))
    negative_infinity = bit_and(C.is_inf, C.sign)
    invalid_product = Const(UQ(1, 0).value(0))
    encode_nan = Const(UQ(1, 0).value(0))
    for i in range(N):
        positive_infinity = bit_or(
            positive_infinity,
            bit_and(B[i].is_inf, bit_neg(product_signs[i])),
        )
        negative_infinity = bit_or(
            negative_infinity,
            bit_and(B[i].is_inf, product_signs[i]),
        )
        invalid_product = bit_or(
            invalid_product, bit_and(B[i].is_inf, A[i].is_zero)
        )
        encode_nan = bit_or(
            encode_nan, bit_or(A[i].is_nan, B[i].is_nan)
        )

    encode_nan = bit_or(
        bit_or(encode_nan, C.is_nan),
        bit_or(
            invalid_product,
            bit_and(positive_infinity, negative_infinity),
        ),
    )
    encode_ninf = bit_and(bit_neg(encode_nan), negative_infinity)
    encode_pinf = bit_and(bit_neg(encode_nan), positive_infinity)

    # Exponents
    A_exponents = [effective_exponent(value) for value in A]
    B_exponents = [effective_exponent(value) for value in B]
    product_exponents = [
        uq_add(A_exponents[i], B_exponents[i]) for i in range(N)
    ]
    product_exponents_for_alignment = [
        if_then_else(
            bit_or(A[i].is_zero, B[i].is_zero),
            Const(
                UQ(product_exponents[i].dtype.int_bits, product_exponents[i].dtype.frac_bits).value(0)
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

    # Significands
    A_significands = [effective_significand(value) for value in A]
    B_significands = [effective_significand(value) for value in B]

    products = [
        uq_resize(
            uq_mul(A_significands[i], B_significands[i]),
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

    # The product bias is 22. Re-bias FP16 C from 15 to that common domain.
    product_exponent = if_then_else(
        q_is_zero(product_sum),
        Const(
            UQ(maximum_product_exponent.dtype.int_bits, maximum_product_exponent.dtype.frac_bits).value(0)
        ),
        maximum_product_exponent,
    )

    # Accumulator' exponent
    c_exponent = effective_exponent(C)
    c_exponent = uq_add(
        c_exponent,
        Const(
            UQ.from_int(
                E4M3FN.exponent_bias
                + E5M2.exponent_bias
                - Float16.exponent_bias
            )
        ),
    )
    c_exponent = if_then_else(
        C.is_zero,
        Const(UQ(c_exponent.dtype.int_bits, 0).value(0)),
        c_exponent,
    )
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
    destination_exponent = q_add(
        uq_to_q(common_exponent),
        Const(
            Q.from_int(
                Float16.exponent_bias
                - E4M3FN.exponent_bias
                - E5M2.exponent_bias
            )
        ),
    )
    finite_result = fp16_encode(
        q_sign_bit(finite_sum),
        destination_exponent,
        q_to_uq(q_abs(finite_sum)),
    )

    return if_then_else(
        encode_nan,
        Const(Float16().NaN()),
        if_then_else(
            encode_ninf,
            Const(Float16().nInf()),
            if_then_else(encode_pinf, Const(Float16().Inf()), finite_result),
        ),
    )


if __name__ == "__main__":
    design = wgmma_fp16_e4m3_e5m2(
        *[Var(name=f"a{index}", dtype=E4M3FN()) for index in range(N)],
        *[Var(name=f"b{index}", dtype=E5M2()) for index in range(N)],
        Var(name="c", dtype=Float16()),
    )
    design.check_determinism()
    design.check_spec()
    output_directory = Path(__file__).resolve().parents[1] / "c_models"
    output_directory.mkdir(exist_ok=True)
    for jittable, suffix in ((True, "jit"), (False, "no_jit")):
        (output_directory / f"wgmma_fp16_e4m3_e5m2_{suffix}.hpp").write_text(
            design.to_cpp(jittable=jittable), encoding="utf-8"
        )


__all__ = [
    "spec_wgmma_fp16_e4m3_e5m2",
    "wgmma_fp16_e4m3_e5m2",
]
