from zolotone import *

from .common import *
from .encode_BFloat16 import *


def spec_bf16_add(x: bf16, y: bf16, ctx):
    nan_case = (
        x.is_nan
        | y.is_nan
        | (x.is_pinf & y.is_ninf)
        | (x.is_ninf & y.is_pinf)
    )
    neg_inf_case = (x.is_ninf | y.is_ninf) & (~nan_case)
    pos_inf_case = (x.is_pinf | y.is_pinf) & (~nan_case)
    neg_zero_case = x.is_nzero & y.is_nzero

    return Cases(
        case(nan_case, bf16.nan(ctx)),
        case(neg_inf_case, bf16.ninf(ctx)),
        case(pos_inf_case, bf16.inf(ctx)),
        case(neg_zero_case, bf16.nzero(ctx)),
        case(
            x.is_finite & y.is_finite,
            bf16.encode(value=x.value + y.value, ctx=ctx),
        ),
        ctx=ctx,
    )


@Composite(name="bf16_add", spec=spec_bf16_add)
def bf16_add(x: Node, y: Node) -> Node:
    (
        x_sign,
        x_exponent,
        x_mantissa,
        x_is_normal,
        x_is_subnormal,
        x_is_zero,
        x_is_inf,
        x_is_nan,
    ) = bf16_decode(x)
    (
        y_sign,
        y_exponent,
        y_mantissa,
        y_is_normal,
        y_is_subnormal,
        y_is_zero,
        y_is_inf,
        y_is_nan,
    ) = bf16_decode(y)

    x_is_ninf = bit_and(x_is_inf, x_sign)
    y_is_ninf = bit_and(y_is_inf, y_sign)
    x_is_pinf = bit_and(x_is_inf, bit_neg(x_sign))
    y_is_pinf = bit_and(y_is_inf, bit_neg(y_sign))

    infinities_with_opposite_signs = bit_or(
        bit_and(x_is_ninf, y_is_pinf),
        bit_and(x_is_pinf, y_is_ninf),
    )

    any_input_is_nan = bit_or(x_is_nan, y_is_nan)
    encode_nan = bit_or(infinities_with_opposite_signs, any_input_is_nan)
    not_encode_nan = bit_neg(encode_nan)

    encode_ninf = bit_and(not_encode_nan, bit_or(x_is_ninf, y_is_ninf))
    encode_pinf = bit_and(not_encode_nan, bit_or(x_is_pinf, y_is_pinf))
    encode_nzero = bit_and(
        bit_and(x_is_zero, y_is_zero),
        bit_and(x_sign, y_sign),
    )

    x_mantissa_fraction = integer_to_fraction(x_mantissa)
    y_mantissa_fraction = integer_to_fraction(y_mantissa)

    x_significand = if_then_else(
        x_is_normal,
        add_implicit_bit(x_mantissa_fraction),
        uq_resize(x_mantissa_fraction, 1, BFloat16.mantissa_bits),
    )
    y_significand = if_then_else(
        y_is_normal,
        add_implicit_bit(y_mantissa_fraction),
        uq_resize(y_mantissa_fraction, 1, BFloat16.mantissa_bits),
    )

    effective_subnormal_exponent = Const(
        UQ(
            1,
            x_exponent.node_type.int_bits,
            x_exponent.node_type.frac_bits,
        )
    )
    x_effective_exponent = if_then_else(
        x_is_subnormal,
        effective_subnormal_exponent,
        x_exponent,
    )
    y_effective_exponent = if_then_else(
        y_is_subnormal,
        effective_subnormal_exponent,
        y_exponent,
    )

    aligned_exponent = uq_max(x_effective_exponent, y_effective_exponent)
    x_shift_amount = uq_sub(aligned_exponent, x_effective_exponent)
    y_shift_amount = uq_sub(aligned_exponent, y_effective_exponent)

    x_significand_wide = uq_resize(
        x_significand,
        1,
        BFloat16.mantissa_bits + 3,
    )
    y_significand_wide = uq_resize(
        y_significand,
        1,
        BFloat16.mantissa_bits + 3,
    )

    x_aligned_significand = uq_rshift_jam(
        x_significand_wide,
        x_shift_amount,
    )
    y_aligned_significand = uq_rshift_jam(
        y_significand_wide,
        y_shift_amount,
    )

    x_signed_significand = q_add_sign(
        uq_to_q(x_aligned_significand),
        x_sign,
    )
    y_signed_significand = q_add_sign(
        uq_to_q(y_aligned_significand),
        y_sign,
    )
    significand_sum = q_add(
        x_signed_significand,
        y_signed_significand,
    )

    finite_sign = q_sign_bit(significand_sum)
    finite_mantissa = q_to_uq(q_abs(significand_sum))
    finite_exponent = uq_to_q(aligned_exponent)

    finite_result = bf16_encode(
        finite_sign,
        finite_exponent,
        finite_mantissa,
    )
    return if_then_else(
        encode_nan,
        Const(BFloat16.NaN()),
        if_then_else(
            encode_ninf,
            Const(BFloat16.nInf()),
            if_then_else(
                encode_pinf,
                Const(BFloat16.Inf()),
                if_then_else(
                    encode_nzero,
                    Const(BFloat16.nZero()),
                    finite_result,
                ),
            ),
        ),
    )


if __name__ == "__main__":
    adder = bf16_add(
        Var(name="a", sign=BFloat16T()),
        Var(name="b", sign=BFloat16T()),
    )

    adder.check_spec()

    with open("examples/bf16_adder_jit.hpp", "w") as file:
        file.write(adder.to_cpp(jittable=True))

    with open("examples/bf16_adder_no_jit.hpp", "w") as file:
        file.write(adder.to_cpp(jittable=False))
