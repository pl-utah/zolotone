from zolotone import *



def spec_bf16_to_fp16(x: bf16, ctx):
    return Cases(
        case(x.is_nan, fp16.nan(ctx)),
        case(x.is_ninf, fp16.ninf(ctx)),
        case(x.is_pinf, fp16.inf(ctx)),
        case(x.is_nzero, fp16.nzero(ctx)),
        case(x.is_finite, fp16.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="bf16_to_fp16", spec=spec_bf16_to_fp16)
def bf16_to_fp16(x: Node) -> Node:
    X = bf16_decode(x)

    mantissa_fraction = uq_integer_to_fraction(X.mantissa)
    significand = if_then_else(
        X.is_norm,
        add_implicit_bit(mantissa_fraction),
        uq_resize(mantissa_fraction, 1, BFloat16.mantissa_bits),
    )
    # Widen the exact seven-bit BF16 significand to FP16's ten-bit precision.
    # The encoder's additional bits are reserved for G/R/S rounding state.
    significand = uq_resize(significand, 1, Float16.mantissa_bits)

    subnormal_exponent = Const(
        UQ(1, X.exponent.node_type.int_bits, X.exponent.node_type.frac_bits)
    )
    effective_exponent = if_then_else(
        X.is_sub,
        subnormal_exponent,
        X.exponent,
    )
    target_exponent = q_add(
        uq_to_q(effective_exponent),
        Const(Q.from_int(Float16.exponent_bias - BFloat16.exponent_bias)),
    )

    finite_result = fp16_encode(X.sign, target_exponent, significand)

    encode_ninf = bit_and(X.is_inf, X.sign)
    encode_pinf = bit_and(X.is_inf, bit_neg(X.sign))
    encode_nzero = bit_and(X.is_zero, X.sign)

    return if_then_else(
        X.is_nan,
        Const(Float16.NaN()),
        if_then_else(
            encode_ninf,
            Const(Float16.nInf()),
            if_then_else(
                encode_pinf,
                Const(Float16.Inf()),
                if_then_else(
                    encode_nzero,
                    Const(Float16.nZero()),
                    finite_result,
                ),
            ),
        ),
    )


if __name__ == "__main__":
    cast = bf16_to_fp16(Var(name="x", sign=BFloat16T()))

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/bf16_to_fp16_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/bf16_to_fp16_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
