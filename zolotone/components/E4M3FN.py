from typing import NamedTuple

from ..ast import *
from ..spec import *
from ..types import *
from .Tuple import *
from .rounding_routines import *
from .UQ import *
from .basics import *


def _e4m3fn_mantissa(x: Node) -> Op:
    def impl(x: E4M3FN) -> UQ:
        return UQ(x.mantissa, E4M3FN.mantissa_bits, 0)

    def sign(x: E4M3FNT) -> UQT:
        return UQT(E4M3FN.mantissa_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"({args[0]} & 7)",
        args=[x],
        name="_e4m3fn_mantissa",
    )


def _e4m3fn_exponent(x: Node) -> Op:
    def impl(x: E4M3FN) -> UQ:
        return UQ(x.exponent, E4M3FN.exponent_bits, 0)

    def sign(x: E4M3FNT) -> UQT:
        return UQT(E4M3FN.exponent_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 3) & 15)",
        args=[x],
        name="_e4m3fn_exponent",
    )


def _e4m3fn_sign(x: Node) -> Op:
    def impl(x: E4M3FN) -> UQ:
        return UQ(x.sign, 1, 0)

    def sign(x: E4M3FNT) -> UQT:
        return UQT(1, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 7) & 1)",
        args=[x],
        name="_e4m3fn_sign",
    )


def _e4m3fn_alloc(sign_bit: Node, exponent: Node, mantissa: Node) -> Op:
    def sign(
        sign_bit: StaticType,
        exponent: StaticType,
        mantissa: StaticType,
    ) -> E4M3FNT:
        return E4M3FNT()
    
    def impl(
        sign_bit: RuntimeType,
        exponent: RuntimeType,
        mantissa: RuntimeType,
    ) -> E4M3FN:
        return E4M3FN.from_fields(sign_bit.val, exponent.val, mantissa.val)
    
    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda args, jittable: (
            f"(({E4M3FNT().to_cpp_type(jittable=jittable)}({args[0]}) << 7) | "
            f"({E4M3FNT().to_cpp_type(jittable=jittable)}({args[1]}) << 3) | "
            f"{E4M3FNT().to_cpp_type(jittable=jittable)}({args[2]}))"
        ),
        args=[sign_bit, exponent, mantissa],
        name="_e4m3fn_alloc",
    )


class DecodedE4M3FN(NamedTuple):
    sign: Node
    exponent: Node
    mantissa: Node
    is_norm: Node
    is_sub: Node
    is_zero: Node
    is_nan: Node


def e4m3fn_decode_spec(x: e4m3fn, ctx):
    decoded = x.decode()[1:]
    classification_count = len(x.classification_flags())
    fields = decoded[:-classification_count]
    classifications = decoded[-classification_count:]
    return fields + tuple(
        If(flag, ctx.one(), ctx.zero()) for flag in classifications
    )


def e4m3fn_pack_spec(s, e, m, ctx):
    zero = ctx.zero()
    one = ctx.one()
    two = ctx.two()
    max_exponent = ctx.real_val(15)
    max_mantissa = ctx.real_val(7)
    
    ctx.assume(s.eq(zero) | s.eq(one))
    
    exponent_is_zero = e.eq(zero)
    mantissa_is_zero = m.eq(zero)
    is_zero = exponent_is_zero & mantissa_is_zero
    is_sub = exponent_is_zero & (~mantissa_is_zero)
    is_nan = e.eq(max_exponent) & m.eq(max_mantissa)
    is_norm = (~exponent_is_zero) & (~is_nan)
    
    signed = sign_multiplier(ctx, s)
    normal_value = (
        signed
        * (one + m * two ** (-ctx.real_val(E4M3FN.mantissa_bits)))
        * two ** (e - ctx.real_val(E4M3FN.exponent_bias))
    )
    subnormal_value = (
        signed
        * m
        * two ** (-ctx.real_val(E4M3FN.mantissa_bits))
        * two ** (one - ctx.real_val(E4M3FN.exponent_bias))
    )
    value = If(
        is_norm,
        normal_value,
        If(is_sub, subnormal_value, If(is_zero, zero, ctx.fresh_real("special"))),
    )
    return e4m3fn(
        value=value,
        sign=s,
        exponent=e,
        mantissa=m,
        is_norm=is_norm,
        is_sub=is_sub,
        is_zero=is_zero,
        is_nan=is_nan,
    )


@Primitive(name="e4m3fn_pack", spec=e4m3fn_pack_spec)
def e4m3fn_pack(sign: Node, exponent: Node, mantissa: Node) -> Node:
    return _e4m3fn_alloc(sign, exponent, mantissa)


def e4m3fn_decode(x: Node) -> DecodedE4M3FN:
    @Primitive(name="e4m3fn_decode", spec=e4m3fn_decode_spec)
    def decode(x: Node) -> Node:
        sign = _e4m3fn_sign(x)
        exponent = _e4m3fn_exponent(x)
        mantissa = _e4m3fn_mantissa(x)
        
        bit = UQ(0, 1, 0)
        mantissa_is_nonzero = basic_or_reduce(mantissa, out=Const(bit))
        mantissa_is_zero = basic_invert(mantissa_is_nonzero, out=Const(bit))
        mantissa_is_all_ones = basic_and_reduce(mantissa, out=Const(bit))
        exponent_is_nonzero = basic_or_reduce(exponent, out=Const(bit))
        exponent_is_zero = basic_invert(exponent_is_nonzero, out=Const(bit))
        exponent_is_all_ones = basic_and_reduce(exponent, out=Const(bit))
        is_nan = basic_and(exponent_is_all_ones, mantissa_is_all_ones, Const(bit))
        not_nan = basic_invert(is_nan, Const(bit))
        is_normal = basic_and(exponent_is_nonzero, not_nan, Const(bit))
        is_subnormal = basic_and(exponent_is_zero, mantissa_is_nonzero, Const(bit))
        is_zero = basic_and(exponent_is_zero, mantissa_is_zero, Const(bit))
        return make_Tuple(
            sign,
            exponent,
            mantissa,
            is_normal,
            is_subnormal,
            is_zero,
            is_nan,
        )
    
    decoded = decode(x)
    return DecodedE4M3FN(
        sign=decoded[0],
        exponent=decoded[1],
        mantissa=decoded[2],
        is_norm=decoded[3],
        is_sub=decoded[4],
        is_zero=decoded[5],
        is_nan=decoded[6],
    )


def e4m3fn_encodings_spec(m, e, ctx):
    integer_m = m * ctx.two() ** ctx.real_val(E4M3FN.mantissa_bits)
    max_e = ctx.real_val(E4M3FN.max_finite_code)
    max_m = ctx.real_val(E4M3FN.max_finite_mantissa)
    reserved_or_overflow = (e > max_e) | (e.eq(max_e) & (integer_m > max_m))
    return (
        If(reserved_or_overflow, max_m, integer_m),
        If(e > max_e, max_e, e),
    )


@Primitive(name="e4m3fn_encodings", spec=e4m3fn_encodings_spec)
def e4m3fn_encodings(m_rounded: Node, e_rounded: Node):
    """Clamp rounded fields to E4M3FN's greatest finite encoding."""

    max_exponent = Const(UQ.from_int(E4M3FN.max_finite_code))
    exponent_overflow = uq_gt(e_rounded, max_exponent)
    clamped_e_wide = uq_min(e_rounded, max_exponent)
    final_e = basic_identity(
        clamped_e_wide,
        Const(UQ(0, E4M3FN.exponent_bits, 0)),
    )
    final_m = uq_fraction_to_integer(m_rounded)
    
    exponent_is_15 = uq_eq(final_e, max_exponent)
    mantissa_is_7 = basic_and_reduce(final_m, Const(UQ(0, 1, 0)))
    reserved_nan = basic_and(
        exponent_is_15,
        mantissa_is_7,
        Const(UQ(0, 1, 0)),
    )
    saturate = basic_or(
        exponent_overflow,
        reserved_nan,
        Const(UQ(0, 1, 0)),
    )
    final_m = basic_mux_2_1(
        saturate,
        final_m,
        Const(UQ.from_int(E4M3FN.max_finite_mantissa)),
        final_m.copy(),
    )
    return make_Tuple(final_m, final_e)


def e4m3fn_encode_spec(s, e, m, ctx):
    signed = sign_multiplier(ctx, s)
    finite_value = (
        signed
        * m
        * ctx.two() ** (e - ctx.real_val(E4M3FN.exponent_bias))
    )
    encoded = e4m3fn.encode(finite_value, ctx)
    # The bit-level interface carries a sign independently from the magnitude,
    # so it can preserve negative zero even though mathematical reals cannot.
    return e4m3fn(
        value=encoded.value,
        sign=s,
        exponent=encoded.exponent,
        mantissa=encoded.mantissa,
        is_norm=encoded.is_norm,
        is_sub=encoded.is_sub,
        is_zero=encoded.is_zero,
        is_nan=encoded.is_nan,
    )


@Composite(name="e4m3fn_encode", spec=e4m3fn_encode_spec)
def e4m3fn_encode(s: Node, e: Node, m: Node) -> Node:
    """Encode sign, biased exponent, and unsigned magnitude using RNE."""
    
    if e.node_type.frac_bits != 0:
        raise ValueError("e4m3fn_encode exponent must have zero fractional bits")
    
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
        target_bits=E4M3FN.mantissa_bits,
    )
    final_m, final_e = e4m3fn_encodings(rounded_m, rounded_e)
    packed = e4m3fn_pack(s, final_e, final_m)
    signed_zero = e4m3fn_pack(
        s,
        Const(UQ(0, E4M3FN.exponent_bits, 0)),
        Const(UQ(0, E4M3FN.mantissa_bits, 0)),
    )
    return if_then_else(encode_exact_zero, signed_zero, packed)
