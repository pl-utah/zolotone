"""Shared normalization and RNE building blocks for floating-point encoders."""

import math

from ..ast import *
from ..types import *
from .Q import *
from .Tuple import *
from .UQ import *
from .basics import *

def uq_RNE_IEEE(m: Node, bits_to_cut: int):
    """Round an unsigned fixed-point value to nearest, ties to even."""

    if bits_to_cut < 0:
        raise ValueError("Cannot cut a negative number of bits")
    if bits_to_cut >= m.dtype.total_bits():
        raise ValueError("Cannot cut all bits of a fixed-point value")

    def spec(x, ctx):
        increment = ctx.fresh_real("increment")
        ctx.assume(increment.eq(ctx.zero()))
        return x, increment

    @Primitive(name="uq_RNE_IEEE", spec=spec)
    def impl(m: Node):
        total_bits = m.dtype.total_bits()
        target_bits = total_bits - bits_to_cut
        target_int_bits = min(m.dtype.int_bits, target_bits)
        target_frac_bits = max(target_bits - target_int_bits, 0)
        bit = Const(UQ(1, 0).from_bits(0))

        guard = bit.copy()
        round_bit = bit.copy()
        sticky = bit.copy()
        if bits_to_cut:
            guard = uq_select(m, bits_to_cut - 1, bits_to_cut - 1)
            if bits_to_cut >= 2:
                round_bit = uq_select(m, bits_to_cut - 2, bits_to_cut - 2)
                if bits_to_cut > 2:
                    sticky = basic_or_reduce(
                        uq_select(m, bits_to_cut - 3, 0), sticky.dtype
                    )

        lsb = uq_select(m, bits_to_cut, bits_to_cut)
        # Add increment?
        tail = basic_or(basic_or(round_bit, sticky, bit.dtype), lsb, bit.dtype)
        increment = basic_and(guard, tail, bit.dtype)
        
        truncated = uq_resize(m, target_int_bits, target_frac_bits)
        # Overflow after incrementing?
        overflow = basic_and(
            basic_and_reduce(truncated, bit.dtype), increment, bit.dtype
        )
        
        incremented = basic_add(
            truncated,
            increment,
            UQ(target_int_bits, target_frac_bits),
        )
        return make_Tuple(incremented, overflow)

    return impl(m)


def round_mantissa_spec(m, e, ctx):
    rounded_m = ctx.fresh_real("rounded_m")
    rounded_e = ctx.fresh_real("rounded_e")
    ctx.assume((m * ctx.two() ** e).eq(rounded_m * ctx.two() ** rounded_e))
    return rounded_m, rounded_e


def round_mantissa(
    m: Node,
    e: Node,
    target_bits: int = Float32.mantissa_bits,
    rounding_mode: str = "RNE",
):
    if target_bits <= 0:
        raise ValueError("target_bits must be positive")
    if rounding_mode != "RNE":
        raise NotImplementedError(rounding_mode)

    @Primitive(name="round_mantissa", spec=round_mantissa_spec)
    def impl(m: Node, e: Node):
        bits_to_cut = max(m.dtype.total_bits() - target_bits, 0)
        rounded, overflow = uq_RNE_IEEE(m, bits_to_cut)
        incremented_e = uq_add(e, overflow)
        was_subnormal = uq_is_zero(e)
        became_normal = basic_and(
            was_subnormal,
            basic_or_reduce(incremented_e, UQ(1, 0)),
            UQ(1, 0),
        )
        rounded = basic_mux_2_1(
            became_normal,
            rounded,
            Const(UQ(rounded.dtype.int_bits, rounded.dtype.frac_bits).from_bits(0)),
            rounded.dtype,
        )
        return make_Tuple(rounded, incremented_e)

    return impl(m, e)


def lzc_spec(x, ctx):
    raise NotImplementedError


def lzc(x: Node) -> Node:
    width = x.dtype.total_bits()
    count_bits = max(1, math.ceil(math.log2(width + 1)))

    def lowering(args, jittable):
        arg = args[0]
        if width <= 32:
            return (
                f"(({arg}) == 0 ? {width} : "
                f"(__builtin_clz(static_cast<uint32_t>({arg})) - {32 - width}))"
            )
        if width <= 64:
            return (
                f"(({arg}) == 0 ? {width} : "
                f"(__builtin_clzll(static_cast<uint64_t>({arg})) - {64 - width}))"
            )
        raise TypeError("lzc supports values up to 64 bits")

    @Primitive(name="lzc", spec=lzc_spec, c_inline=True, c_lowering=lowering)
    def impl(x):
        count = Const(UQ(count_bits, 0).from_bits(0))
        still_zero = Const(UQ(1, 0).from_bits(1))
        for pos in range(width - 1, -1, -1):
            is_zero = basic_invert(uq_select(x, pos, pos), UQ(1, 0))
            still_zero = basic_and(still_zero, is_zero, still_zero.dtype)
            count = basic_add(count, still_zero, count.dtype)
        return count

    return impl(x)


def normalize_to_1_xxx_spec(m, e, ctx):
    normalized_m = ctx.fresh_real("normalized_m")
    normalized_e = ctx.fresh_real("normalized_e")
    ctx.assume((m * ctx.two() ** e).eq(normalized_m * ctx.two() ** normalized_e))
    return normalized_m, normalized_e


def normalize_to_1_xxx(m: Node, e: Node):
    @Primitive(name="normalize_to_1_xxx", spec=normalize_to_1_xxx_spec)
    def impl(m: Node, e: Node):
        target_frac_bits = max(m.dtype.int_bits - 1, 0) + m.dtype.frac_bits
        leading_zeros = uq_to_q(lzc(m))
        shift = q_add(
            q_sub(leading_zeros, uq_to_q(uq_int_bits(m))),
            Const(Q.from_int(1)),
        )
        shift_sign = q_sign_bit(shift)
        shift_magnitude_q = q_abs(shift)
        shift_magnitude = q_to_uq(shift_magnitude_q)
        resized = uq_resize(m, max(1, m.dtype.int_bits), target_frac_bits)
        normalized_m = basic_mux_2_1(
            shift_sign,
            uq_lshift(resized, shift_magnitude),
            uq_rshift_jam(resized, shift_magnitude),
            UQ(1, target_frac_bits),
        )
        right_e = q_add(e, shift_magnitude_q)
        normalized_e = basic_mux_2_1(
            shift_sign,
            q_sub(e, shift_magnitude_q),
            right_e,
            right_e.dtype,
        )
        return make_Tuple(normalized_m, normalized_e)

    return impl(m, e)


@Primitive(name="drop_implicit_bit", spec=lambda x, ctx: x - ctx.one())
def drop_implicit_bit(x: Node):
    return uq_select(x, x.dtype.frac_bits - 1, 0)


def shift_if_subnormal_spec(m, e, ctx):
    shifted_m = ctx.fresh_real("classified_m")
    shifted_e = ctx.fresh_real("classified_e")
    ctx.assume((m * ctx.two() ** e).eq(shifted_m * ctx.two() ** shifted_e))
    return shifted_m, shifted_e


def shift_if_subnormal(
    normalized_m: Node,
    normalized_e: Node,
    subnormal_extra_bits: int = 3,
):
    if not isinstance(subnormal_extra_bits, int):
        raise TypeError("subnormal_extra_bits must be an int")
    if subnormal_extra_bits < 0:
        raise ValueError("subnormal_extra_bits must be non-negative")

    @Primitive(name="shift_if_subnormal", spec=shift_if_subnormal_spec)
    def impl(m: Node, e: Node):
        is_subnormal = basic_or(
            q_is_zero(e), q_sign_bit(e), UQ(1, 0)
        )
        exponent_magnitude = q_to_uq(q_abs(e))
        shifted_e = basic_mux_2_1(
            is_subnormal,
            exponent_magnitude,
            Const(UQ(1, 0).from_bits(0)),
            exponent_magnitude.dtype,
        )
        subnormal_shift = uq_add(Const(UQ(1, 0).from_bits(1)), exponent_magnitude)
        shift_amount = basic_mux_2_1(
            is_subnormal,
            Const(UQ(1, 0).from_bits(0)),
            subnormal_shift,
            subnormal_shift.dtype,
        )
        widened = basic_lshift(
            m,
            Const(UQ.from_int(subnormal_extra_bits)),
            UQ(m.dtype.int_bits, m.dtype.frac_bits + subnormal_extra_bits),
        )
        return make_Tuple(uq_rshift_jam(widened, shift_amount), shifted_e)

    return impl(normalized_m, normalized_e)
