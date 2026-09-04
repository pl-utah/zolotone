from typing import NamedTuple

from ..ast import *
from ..spec import *
from ..types import *
from .Tuple import *
from .common import *
from .rounding_routines import *
from .UQ import *
from .basics import *


def _ue4m3_mantissa(x: Node) -> Op:
    def impl(x: UE4M3) -> UQ:
        return UQ(UE4M3.mantissa_bits, 0).value(x.mantissa)

    def sign(x: UE4M3) -> UQ:
        return UQ(UE4M3.mantissa_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"({args[0]} & 7)",
        args=[x],
        name="_ue4m3_mantissa",
    )


def _ue4m3_exponent(x: Node) -> Op:
    def impl(x: UE4M3) -> UQ:
        return UQ(UE4M3.exponent_bits, 0).value(x.exponent)

    def sign(x: UE4M3) -> UQ:
        return UQ(UE4M3.exponent_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 3) & 15)",
        args=[x],
        name="_ue4m3_exponent",
    )


def _ue4m3_alloc(exponent: Node, mantissa: Node) -> Op:
    def sign(exponent: DataType, mantissa: DataType) -> UE4M3:
        return UE4M3()

    def impl(exponent: RuntimeValue, mantissa: RuntimeValue) -> UE4M3:
        return UE4M3().from_fields(exponent.raw, mantissa.raw)

    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda args, jittable: (
            f"(({UE4M3().to_cpp_type(jittable=jittable)}({args[0]}) << 3) | "
            f"{UE4M3().to_cpp_type(jittable=jittable)}({args[1]}))"
        ),
        args=[exponent, mantissa],
        name="_ue4m3_alloc",
    )


class DecodedUE4M3(NamedTuple):
    exponent: Node
    mantissa: Node
    is_norm: Node
    is_sub: Node
    is_zero: Node
    is_nan: Node


def ue4m3_decode_spec(x: ue4m3, ctx):
    decoded = x.decode()[1:]
    classification_count = len(x.classification_flags())
    fields = decoded[:-classification_count]
    classifications = decoded[-classification_count:]
    return fields + tuple(
        If(flag, ctx.one(), ctx.zero()) for flag in classifications
    )


def ue4m3_pack_spec(e, m, ctx):
    zero = ctx.zero()
    one = ctx.one()
    two = ctx.two()
    max_exponent = ctx.real_val(15)
    max_mantissa = ctx.real_val(7)

    exponent_is_zero = e.eq(zero)
    mantissa_is_zero = m.eq(zero)
    is_zero = exponent_is_zero & mantissa_is_zero
    is_sub = exponent_is_zero & (~mantissa_is_zero)
    is_nan = e.eq(max_exponent) & m.eq(max_mantissa)
    is_norm = (~exponent_is_zero) & (~is_nan)
    normal_value = (
        (one + m * two ** (-ctx.real_val(UE4M3.mantissa_bits)))
        * two ** (e - ctx.real_val(UE4M3.exponent_bias))
    )
    subnormal_value = (
        m
        * two ** (-ctx.real_val(UE4M3.mantissa_bits))
        * two ** (one - ctx.real_val(UE4M3.exponent_bias))
    )
    value = If(
        is_norm,
        normal_value,
        If(is_sub, subnormal_value, If(is_zero, zero, ctx.fresh_real("special"))),
    )
    return ue4m3(
        value=value,
        exponent=e,
        mantissa=m,
        is_norm=is_norm,
        is_sub=is_sub,
        is_zero=is_zero,
        is_nan=is_nan,
    )


@Primitive(name="ue4m3_pack", spec=ue4m3_pack_spec)
def ue4m3_pack(exponent: Node, mantissa: Node) -> Node:
    return _ue4m3_alloc(exponent, mantissa)


def ue4m3_decode(x: Node) -> DecodedUE4M3:
    @Primitive(name="ue4m3_decode", spec=ue4m3_decode_spec)
    def decode(x: Node) -> Node:
        exponent = _ue4m3_exponent(x)
        mantissa = _ue4m3_mantissa(x)

        bit = UQ(1, 0).value(0)
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
            exponent,
            mantissa,
            is_normal,
            is_subnormal,
            is_zero,
            is_nan,
        )

    decoded = decode(x)
    return DecodedUE4M3(
        exponent=decoded[0],
        mantissa=decoded[1],
        is_norm=decoded[2],
        is_sub=decoded[3],
        is_zero=decoded[4],
        is_nan=decoded[5],
    )


def ue4m3_encodings_spec(m, e, ctx):
    integer_m = m * ctx.two() ** ctx.real_val(UE4M3.mantissa_bits)
    max_e = ctx.real_val(UE4M3.max_finite_code)
    max_m = ctx.real_val(UE4M3.max_finite_mantissa)
    reserved_or_overflow = (e > max_e) | (e.eq(max_e) & (integer_m > max_m))
    return (
        If(reserved_or_overflow, max_m, integer_m),
        If(e > max_e, max_e, e),
    )


@Primitive(name="ue4m3_encodings", spec=ue4m3_encodings_spec)
def ue4m3_encodings(m_rounded: Node, e_rounded: Node):
    """Clamp rounded fields to UE4M3's greatest finite encoding."""

    max_exponent = Const(UQ.from_int(UE4M3.max_finite_code))
    exponent_overflow = uq_gt(e_rounded, max_exponent)
    final_e = basic_identity(
        uq_min(e_rounded, max_exponent),
        Const(UQ(UE4M3.exponent_bits, 0).value(0)),
    )
    final_m = uq_fraction_to_integer(m_rounded)

    exponent_is_15 = uq_eq(final_e, max_exponent)
    mantissa_is_7 = basic_and_reduce(final_m, Const(UQ(1, 0).value(0)))
    reserved_nan = basic_and(
        exponent_is_15,
        mantissa_is_7,
        Const(UQ(1, 0).value(0)),
    )
    saturate = basic_or(
        exponent_overflow,
        reserved_nan,
        Const(UQ(1, 0).value(0)),
    )
    final_m = basic_mux_2_1(
        saturate,
        final_m,
        Const(UQ.from_int(UE4M3.max_finite_mantissa)),
        final_m.copy(),
    )
    return make_Tuple(final_m, final_e)


def ue4m3_encode_spec(e, m, ctx):
    finite_value = m * ctx.two() ** (e - ctx.real_val(UE4M3.exponent_bias))
    return ue4m3.encode(finite_value, ctx)


@Composite(name="ue4m3_encode", spec=ue4m3_encode_spec)
def ue4m3_encode(e: Node, m: Node) -> Node:
    """Encode an unsigned magnitude using RNE and finite saturation."""

    if e.dtype.frac_bits != 0:
        raise ValueError("ue4m3_encode exponent must have zero fractional bits")

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
        target_bits=UE4M3.mantissa_bits,
    )
    final_m, final_e = ue4m3_encodings(rounded_m, rounded_e)
    rounded_zero = bit_and(uq_is_zero(final_e), uq_is_zero(final_m))
    use_zero = bit_or(encode_exact_zero, rounded_zero)
    return if_then_else(
        use_zero,
        Const(UE4M3().Zero()),
        ue4m3_pack(final_e, final_m),
    )
