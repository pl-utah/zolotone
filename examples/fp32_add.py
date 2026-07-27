from zolotone import *
from .common import *
from .encode_Float32 import *


# Demo navigation map:
# - FP32 decode helpers: zolotone/components/Float.py
# - Bit helpers and significand formatting: examples/common.py
# - Unsigned fixed-point operations: zolotone/components/UQ.py
# - Signed fixed-point operations: zolotone/components/Q.py
# - Final FP32 packing and rounding: examples/encode_Float32.py


# Non-bit-precise concept specification of a single-precision IEEE adder.
def spec_fp32_add(x: fp32, y: fp32, ctx):
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
        case(nan_case, fp32.nan()),
        case(neg_inf_case, fp32.ninf()),
        case(pos_inf_case, fp32.inf()),
        case(neg_zero_case, fp32.nzero()),
        case(
            x.is_finite & y.is_finite,
            fp32.encode(value=x.value + y.value, ctx=ctx),
        ),
        ctx=ctx,
    )


# Implementation of a single-precision IEEE adder.
@Composite(name="fp32_add", spec=spec_fp32_add)
def fp32_add(x: Node, y: Node) -> Node:
    # 1. Decode both packed FP32 inputs into sign, exponent, mantissa, and flags.
    (
        x_sign,
        x_exponent,
        x_mantissa,
        x_is_normal,
        x_is_subnormal,
        x_is_zero,
        x_is_inf,
        x_is_nan,
    ) = fp32_decode(x)
    (
        y_sign,
        y_exponent,
        y_mantissa,
        y_is_normal,
        y_is_subnormal,
        y_is_zero,
        y_is_inf,
        y_is_nan,
    ) = fp32_decode(y)

    # 2. Resolve IEEE special-result flags before the finite-number datapath.
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

    # 3. Format and align significands.
    # Decode gives the mantissa as UQ<23, 0>; the adder datapath wants UQ<0, 23>.
    x_mantissa_fraction = integer_to_fraction(x_mantissa)
    y_mantissa_fraction = integer_to_fraction(y_mantissa)

    # Normal numbers have an implicit leading 1. Subnormals do not.
    x_significand = if_then_else(
        x_is_normal,
        add_implicit_bit(x_mantissa_fraction),
        uq_resize(x_mantissa_fraction, 1, Float32.mantissa_bits),
    )
    y_significand = if_then_else(
        y_is_normal,
        add_implicit_bit(y_mantissa_fraction),
        uq_resize(y_mantissa_fraction, 1, Float32.mantissa_bits),
    )

    # Subnormals store exponent 0 but behave as exponent 1-bias.
    effective_subnormal_exponent = Const(
        UQ(1, x_exponent.node_type.int_bits, x_exponent.node_type.frac_bits)
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

    # Keep three extra low bits so right-shift-jam can preserve rounding evidence.
    x_significand_wide = uq_resize(x_significand, 1, Float32.mantissa_bits + 3)
    y_significand_wide = uq_resize(y_significand, 1, Float32.mantissa_bits + 3)

    x_aligned_significand = uq_rshift_jam(x_significand_wide, x_shift_amount)
    y_aligned_significand = uq_rshift_jam(y_significand_wide, y_shift_amount)

    # 4. Apply signs, then add the aligned signed significands.
    x_signed_significand = q_add_sign(uq_to_q(x_aligned_significand), x_sign)
    y_signed_significand = q_add_sign(uq_to_q(y_aligned_significand), y_sign)
    significand_sum = q_add(x_signed_significand, y_signed_significand)

    # 5. Encode FP32.
    finite_sign = q_sign_bit(significand_sum)
    finite_mantissa = q_to_uq(q_abs(significand_sum))
    finite_exponent = uq_to_q(aligned_exponent)

    # 6. Normalize, round, and pack the final FP32 result.
    finite_result = fp32_encode(
        finite_sign,
        finite_exponent,
        finite_mantissa,
    )
    return if_then_else(
        encode_nan,
        Const(Float32.NaN()),
        if_then_else(
            encode_ninf,
            Const(Float32.nInf()),
            if_then_else(
                encode_pinf,
                Const(Float32.Inf()),
                if_then_else(
                    encode_nzero,
                    Const(Float32.nZero()),
                    finite_result,
                ),
            ),
        ),
    )


if __name__ == '__main__':
    adder = fp32_add(
        Var(name="a", sign=Float32T()),
        Var(name="b", sign=Float32T()),
    )
    adder.check_determinism()
    adder.check_spec()

    with open("examples/adder_jit.hpp", "w") as file:
        file.write(adder.to_cpp(jittable=True))

    with open("examples/adder_no_jit.hpp", "w") as file:
        file.write(adder.to_cpp(jittable=False))
