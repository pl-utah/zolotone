from typing import NamedTuple

from ..ast import *
from ..spec import *
from ..types import *
from .Tuple import *
from .common import *
from .rounding_routines import *
from .UQ import *
from .basics import *


def _e5m2fnuz_mantissa(x: Node) -> Op:
    def impl(value: E5M2FNUZ) -> UQ:
        return UQ(value.mantissa, E5M2FNUZ.mantissa_bits, 0)

    def sign(value_type: E5M2FNUZT) -> UQT:
        return UQT(E5M2FNUZ.mantissa_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"({args[0]} & 3)",
        args=[x],
        name="_e5m2fnuz_mantissa",
    )


def _e5m2fnuz_exponent(x: Node) -> Op:
    def impl(value: E5M2FNUZ) -> UQ:
        return UQ(value.exponent, E5M2FNUZ.exponent_bits, 0)

    def sign(value_type: E5M2FNUZT) -> UQT:
        return UQT(E5M2FNUZ.exponent_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 2) & 31)",
        args=[x],
        name="_e5m2fnuz_exponent",
    )


def _e5m2fnuz_sign(x: Node) -> Op:
    def impl(value: E5M2FNUZ) -> UQ:
        return UQ(value.sign, 1, 0)

    def sign(value_type: E5M2FNUZT) -> UQT:
        return UQT(1, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 7) & 1)",
        args=[x],
        name="_e5m2fnuz_sign",
    )


def _e5m2fnuz_alloc(sign_bit: Node, exponent: Node, mantissa: Node) -> Op:
    def sign(
        sign_bit: StaticType,
        exponent: StaticType,
        mantissa: StaticType,
    ) -> E5M2FNUZT:
        return E5M2FNUZT()

    def impl(
        sign_bit: RuntimeType,
        exponent: RuntimeType,
        mantissa: RuntimeType,
    ) -> E5M2FNUZ:
        return E5M2FNUZ.from_fields(sign_bit.val, exponent.val, mantissa.val)

    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda args, jittable: (
            f"(({E5M2FNUZT().to_cpp_type(jittable=jittable)}({args[0]}) << 7) | "
            f"({E5M2FNUZT().to_cpp_type(jittable=jittable)}({args[1]}) << 2) | "
            f"{E5M2FNUZT().to_cpp_type(jittable=jittable)}({args[2]}))"
        ),
        args=[sign_bit, exponent, mantissa],
        name="_e5m2fnuz_alloc",
    )


class DecodedE5M2FNUZ(NamedTuple):
    sign: Node
    exponent: Node
    mantissa: Node
    is_norm: Node
    is_sub: Node
    is_zero: Node
    is_nan: Node


def e5m2fnuz_decode_spec(x: e5m2fnuz, ctx):
    decoded = x.decode()[1:]
    classification_count = len(x.classification_flags())
    fields = decoded[:-classification_count]
    classifications = decoded[-classification_count:]
    return fields + tuple(
        If(flag, ctx.one(), ctx.zero()) for flag in classifications
    )


def e5m2fnuz_pack_spec(s, e, m, ctx):
    zero = ctx.zero()
    one = ctx.one()
    two = ctx.two()
    ctx.assume(s.eq(zero) | s.eq(one))
    exponent_is_zero = e.eq(zero)
    mantissa_is_zero = m.eq(zero)
    is_nan = s.eq(one) & exponent_is_zero & mantissa_is_zero
    is_zero = s.eq(zero) & exponent_is_zero & mantissa_is_zero
    is_sub = exponent_is_zero & (~mantissa_is_zero)
    is_norm = ~exponent_is_zero
    signed = sign_multiplier(ctx, s)
    normal_value = (
        signed
        * (one + m * two ** (-ctx.real_val(E5M2FNUZ.mantissa_bits)))
        * two ** (e - ctx.real_val(E5M2FNUZ.exponent_bias))
    )
    subnormal_value = (
        signed
        * m
        * two ** (-ctx.real_val(E5M2FNUZ.mantissa_bits))
        * two ** (one - ctx.real_val(E5M2FNUZ.exponent_bias))
    )
    value = If(
        is_norm,
        normal_value,
        If(is_sub, subnormal_value, If(is_zero, zero, ctx.fresh_real("special"))),
    )
    return e5m2fnuz(
        value=value,
        sign=s,
        exponent=e,
        mantissa=m,
        is_norm=is_norm,
        is_sub=is_sub,
        is_zero=is_zero,
        is_nan=is_nan,
    )


@Primitive(name="e5m2fnuz_pack", spec=e5m2fnuz_pack_spec)
def e5m2fnuz_pack(sign: Node, exponent: Node, mantissa: Node) -> Node:
    return _e5m2fnuz_alloc(sign, exponent, mantissa)


def e5m2fnuz_decode(x: Node) -> DecodedE5M2FNUZ:
    @Primitive(name="e5m2fnuz_decode", spec=e5m2fnuz_decode_spec)
    def decode(x: Node) -> Node:
        sign = _e5m2fnuz_sign(x)
        exponent = _e5m2fnuz_exponent(x)
        mantissa = _e5m2fnuz_mantissa(x)
        bit = Const(UQ(0, 1, 0))
        exponent_is_nonzero = basic_or_reduce(exponent, bit.copy())
        exponent_is_zero = basic_invert(exponent_is_nonzero, bit.copy())
        mantissa_is_nonzero = basic_or_reduce(mantissa, bit.copy())
        mantissa_is_zero = basic_invert(mantissa_is_nonzero, bit.copy())
        sign_is_zero = basic_invert(sign, bit.copy())
        zero_fields = basic_and(exponent_is_zero, mantissa_is_zero, bit.copy())
        is_zero = basic_and(sign_is_zero, zero_fields, bit.copy())
        is_nan = basic_and(sign, zero_fields, bit.copy())
        is_sub = basic_and(exponent_is_zero, mantissa_is_nonzero, bit.copy())
        return make_Tuple(
            sign,
            exponent,
            mantissa,
            exponent_is_nonzero,
            is_sub,
            is_zero,
            is_nan,
        )

    decoded = decode(x)
    return DecodedE5M2FNUZ(
        sign=decoded[0],
        exponent=decoded[1],
        mantissa=decoded[2],
        is_norm=decoded[3],
        is_sub=decoded[4],
        is_zero=decoded[5],
        is_nan=decoded[6],
    )


def e5m2fnuz_encodings_spec(m, e, ctx):
    integer_m = m * ctx.two() ** ctx.real_val(E5M2FNUZ.mantissa_bits)
    overflow = e > ctx.real_val(E5M2FNUZ.max_finite_code)
    return (
        If(overflow, ctx.real_val(E5M2FNUZ.max_finite_mantissa), integer_m),
        If(overflow, ctx.real_val(E5M2FNUZ.max_finite_code), e),
    )


@Primitive(name="e5m2fnuz_encodings", spec=e5m2fnuz_encodings_spec)
def e5m2fnuz_encodings(m_rounded: Node, e_rounded: Node):
    max_exponent = Const(UQ.from_int(E5M2FNUZ.max_finite_code))
    overflow = uq_gt(e_rounded, max_exponent)
    final_e = basic_identity(
        uq_min(e_rounded, max_exponent),
        Const(UQ(0, E5M2FNUZ.exponent_bits, 0)),
    )
    final_m = uq_fraction_to_integer(m_rounded)
    final_m = basic_mux_2_1(
        overflow,
        final_m,
        Const(UQ.from_int(E5M2FNUZ.max_finite_mantissa)),
        final_m.copy(),
    )
    return make_Tuple(final_m, final_e)


def e5m2fnuz_encode_spec(s, e, m, ctx):
    finite_value = (
        sign_multiplier(ctx, s)
        * m
        * ctx.two() ** (e - ctx.real_val(E5M2FNUZ.exponent_bias))
    )
    return e5m2fnuz.encode(finite_value, ctx)


@Composite(name="e5m2fnuz_encode", spec=e5m2fnuz_encode_spec)
def e5m2fnuz_encode(s: Node, e: Node, m: Node) -> Node:
    """Encode using RNE, canonical unsigned zero, and finite saturation."""

    if e.node_type.frac_bits != 0:
        raise ValueError("e5m2fnuz_encode exponent must have zero fractional bits")
    encode_exact_zero = uq_is_zero(m)
    normalized_m, normalized_e = normalize_to_1_xxx(m, e)
    shifted_m, shifted_e = shift_if_subnormal(
        normalized_m, normalized_e, subnormal_extra_bits=3
    )
    rounded_m, rounded_e = round_mantissa(
        drop_implicit_bit(shifted_m),
        shifted_e,
        target_bits=E5M2FNUZ.mantissa_bits,
    )
    final_m, final_e = e5m2fnuz_encodings(rounded_m, rounded_e)
    rounded_zero = bit_and(uq_is_zero(final_e), uq_is_zero(final_m))
    use_zero = bit_or(encode_exact_zero, rounded_zero)
    return if_then_else(
        use_zero,
        Const(E5M2FNUZ.Zero()),
        e5m2fnuz_pack(s, final_e, final_m),
    )
