from typing import NamedTuple

from ..ast import *
from ..spec import If, fp32, sign_multiplier
from ..types import *
from .Tuple import make_Tuple
from .UQ import uq_fraction_to_integer, uq_is_zero, uq_min
from .basics import *
from .rounding_routines import (
    drop_implicit_bit,
    normalize_to_1_xxx,
    round_mantissa,
    shift_if_subnormal,
)


def _fp32_mantissa(x: Node) -> Op:
    def impl(x: Float32) -> UQ:
        return UQ(Float32.mantissa_bits, 0).from_bits(x.mantissa)

    def sign(x: Float32) -> UQ:
        return UQ(Float32.mantissa_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"({args[0]} & 8388607)",
        args=[x],
        name="_fp32_mantissa",
    )


def _fp32_exponent(x: Node) -> Op:
    def impl(x: Float32) -> UQ:
        return UQ(Float32.exponent_bits, 0).from_bits(x.exponent)

    def sign(x: Float32) -> UQ:
        return UQ(Float32.exponent_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 23) & 255)",
        args=[x],
        name="_fp32_exponent",
    )


def _fp32_sign(x: Node) -> Op:
    def impl(x: Float32) -> UQ:
        return UQ(1, 0).from_bits(x.sign)

    def sign(x: Float32) -> UQ:
        return UQ(1, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 31) & 1)",
        args=[x],
        name="_fp32_sign",
    )


def _fp32_alloc(sign_bit: Node, exponent: Node, mantissa: Node) -> Op:
    def sign(
        sign_bit: DataType,
        exponent: DataType,
        mantissa: DataType,
    ) -> Float32:
        return Float32()

    def impl(
        sign_bit: RuntimeValue,
        exponent: RuntimeValue,
        mantissa: RuntimeValue,
    ) -> Float32:
        return Float32().from_fields(sign_bit.raw, exponent.raw, mantissa.raw)

    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda args, jittable: (
            f"(({Float32().to_cpp_type(jittable=jittable)}({args[0]}) << 31) | "
            f"({Float32().to_cpp_type(jittable=jittable)}({args[1]}) << 23) | "
            f"{Float32().to_cpp_type(jittable=jittable)}({args[2]}))"
        ),
        args=[sign_bit, exponent, mantissa],
        name="_fp32_alloc",
    )


class DecodedFP32(NamedTuple):
    sign: Node
    exponent: Node
    mantissa: Node
    is_norm: Node
    is_sub: Node
    is_zero: Node
    is_inf: Node
    is_nan: Node


def fp32_decode_spec(x: fp32, ctx):
    decoded = x.decode()[1:]
    classification_count = len(x.classification_flags())
    fields = decoded[:-classification_count]
    classifications = decoded[-classification_count:]
    return fields + tuple(
        If(flag, ctx.one(), ctx.zero()) for flag in classifications
    )


def fp32_pack_spec(s, e, m, ctx):
    zero = ctx.zero()
    one = ctx.one()
    two = ctx.two()
    max_exponent = ctx.real_val(Float32.inf_code)

    ctx.assume(s.eq(zero) | s.eq(one))

    exponent_is_zero = e.eq(zero)
    exponent_is_max = e.eq(max_exponent)
    mantissa_is_zero = m.eq(zero)
    is_zero = exponent_is_zero & mantissa_is_zero
    is_sub = exponent_is_zero & (~mantissa_is_zero)
    is_inf = exponent_is_max & mantissa_is_zero
    is_nan = exponent_is_max & (~mantissa_is_zero)
    is_norm = (~exponent_is_zero) & (~exponent_is_max)

    signed = sign_multiplier(ctx, s)
    normal_value = (
        signed
        * (one + m * two ** (-ctx.real_val(Float32.mantissa_bits)))
        * two ** (e - ctx.real_val(Float32.exponent_bias))
    )
    subnormal_value = (
        signed
        * m
        * two ** (-ctx.real_val(Float32.mantissa_bits))
        * two ** (one - ctx.real_val(Float32.exponent_bias))
    )
    value = If(
        is_norm,
        normal_value,
        If(is_sub, subnormal_value, If(is_zero, zero, ctx.fresh_real("special"))),
    )
    return fp32(
        value=value,
        sign=s,
        exponent=e,
        mantissa=m,
        is_norm=is_norm,
        is_sub=is_sub,
        is_zero=is_zero,
        is_inf=is_inf,
        is_nan=is_nan,
    )


@Primitive(name="fp32_pack", spec=fp32_pack_spec)
def fp32_pack(sign: Node, exponent: Node, mantissa: Node) -> Node:
    return _fp32_alloc(sign, exponent, mantissa)


def fp32_decode(x: Node) -> DecodedFP32:
    @Primitive(name="fp32_decode", spec=fp32_decode_spec)
    def decode(x: Node) -> Node:
        sign = _fp32_sign(x)
        exponent = _fp32_exponent(x)
        mantissa = _fp32_mantissa(x)

        bit = UQ(1, 0)
        mantissa_is_nonzero = basic_or_reduce(mantissa, bit)
        mantissa_is_zero = basic_invert(mantissa_is_nonzero, bit)
        exponent_is_all_ones = basic_and_reduce(exponent, bit)
        exponent_is_not_all_ones = basic_invert(
            exponent_is_all_ones,
            bit,
        )
        exponent_is_nonzero = basic_or_reduce(exponent, bit)
        exponent_is_zero = basic_invert(exponent_is_nonzero, bit)

        is_normal = basic_and(
            exponent_is_nonzero,
            exponent_is_not_all_ones,
            bit,
        )
        is_subnormal = basic_and(
            exponent_is_zero,
            mantissa_is_nonzero,
            bit,
        )
        is_zero = basic_and(exponent_is_zero, mantissa_is_zero, bit)
        is_inf = basic_and(
            exponent_is_all_ones,
            mantissa_is_zero,
            bit,
        )
        is_nan = basic_and(
            exponent_is_all_ones,
            mantissa_is_nonzero,
            bit,
        )
        return make_Tuple(
            sign,
            exponent,
            mantissa,
            is_normal,
            is_subnormal,
            is_zero,
            is_inf,
            is_nan,
        )

    decoded = decode(x)
    return DecodedFP32(
        sign=decoded[0],
        exponent=decoded[1],
        mantissa=decoded[2],
        is_norm=decoded[3],
        is_sub=decoded[4],
        is_zero=decoded[5],
        is_inf=decoded[6],
        is_nan=decoded[7],
    )


def fp32_encodings_spec(m, e, ctx):
    return m * ctx.two() ** ctx.real_val(Float32.mantissa_bits), e


@Primitive(name="fp32_encodings", spec=fp32_encodings_spec)
def fp32_encodings(m_rounded: Node, e_rounded: Node):
    final_e_wide = uq_min(
        e_rounded,
        Const(UQ.from_int(Float32.inf_code)),
    )
    final_e = basic_identity(
        final_e_wide,
        UQ(Float32.exponent_bits, 0),
    )
    is_inf = basic_and_reduce(final_e, UQ(1, 0))
    final_m = basic_mux_2_1(
        is_inf,
        m_rounded,
        Const(UQ(1, 0).from_bits(0)),
        m_rounded.dtype,
    )
    return make_Tuple(uq_fraction_to_integer(final_m), final_e)


def fp32_encode_spec(s, e, m, ctx):
    signed = sign_multiplier(ctx, s)
    finite_value = (
        signed
        * m
        * ctx.two() ** (e - ctx.real_val(Float32.exponent_bias))
    )
    return fp32.encode(finite_value, ctx)


@Composite(name="fp32_encode", spec=fp32_encode_spec)
def fp32_encode(s: Node, e: Node, m: Node) -> Node:
    if e.dtype.frac_bits != 0:
        raise ValueError("fp32_encode exponent must have zero fractional bits")

    encode_exact_zero = uq_is_zero(m)
    normalized_m, normalized_e = normalize_to_1_xxx(m, e)
    shifted_m, shifted_e = shift_if_subnormal(
        normalized_m,
        normalized_e,
        subnormal_extra_bits=3,
    )
    rounded_m, rounded_e = round_mantissa(
        drop_implicit_bit(shifted_m),
        shifted_e,
        target_bits=Float32.mantissa_bits,
    )
    final_m, final_e = fp32_encodings(rounded_m, rounded_e)
    return if_then_else(
        encode_exact_zero,
        Const(Float32().Zero()),
        fp32_pack(s, final_e, final_m),
    )
