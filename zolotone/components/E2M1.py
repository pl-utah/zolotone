from typing import NamedTuple

from ..ast import *
from ..spec import *
from ..types import *
from .Tuple import *
from .rounding_routines import *
from .UQ import *
from .basics import *


def _e2m1_mantissa(x: Node) -> Op:
    def impl(value: E2M1) -> UQ:
        return UQ(value.mantissa, E2M1.mantissa_bits, 0)

    def sign(value_type: E2M1T) -> UQT:
        return UQT(E2M1.mantissa_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"({args[0]} & 1)",
        args=[x],
        name="_e2m1_mantissa",
    )


def _e2m1_exponent(x: Node) -> Op:
    def impl(value: E2M1) -> UQ:
        return UQ(value.exponent, E2M1.exponent_bits, 0)

    def sign(value_type: E2M1T) -> UQT:
        return UQT(E2M1.exponent_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 1) & 3)",
        args=[x],
        name="_e2m1_exponent",
    )


def _e2m1_sign(x: Node) -> Op:
    def impl(value: E2M1) -> UQ:
        return UQ(value.sign, 1, 0)

    def sign(value_type: E2M1T) -> UQT:
        return UQT(1, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 3) & 1)",
        args=[x],
        name="_e2m1_sign",
    )


def _e2m1_alloc(sign_bit: Node, exponent: Node, mantissa: Node) -> Op:
    def sign(
        sign_bit: StaticType,
        exponent: StaticType,
        mantissa: StaticType,
    ) -> E2M1T:
        return E2M1T()

    def impl(
        sign_bit: RuntimeType,
        exponent: RuntimeType,
        mantissa: RuntimeType,
    ) -> E2M1:
        return E2M1.from_fields(sign_bit.val, exponent.val, mantissa.val)

    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda args, jittable: (
            f"(({E2M1T().to_cpp_type(jittable=jittable)}({args[0]}) << 3) | "
            f"({E2M1T().to_cpp_type(jittable=jittable)}({args[1]}) << 1) | "
            f"{E2M1T().to_cpp_type(jittable=jittable)}({args[2]}))"
        ),
        args=[sign_bit, exponent, mantissa],
        name="_e2m1_alloc",
    )


class DecodedE2M1(NamedTuple):
    sign: Node
    exponent: Node
    mantissa: Node
    is_norm: Node
    is_sub: Node
    is_zero: Node


def e2m1_decode_spec(x: e2m1, ctx):
    decoded = x.decode()[1:]
    classification_count = len(x.classification_flags())
    fields = decoded[:-classification_count]
    classifications = decoded[-classification_count:]
    return fields + tuple(
        If(flag, ctx.one(), ctx.zero()) for flag in classifications
    )


def e2m1_pack_spec(s, e, m, ctx):
    zero = ctx.zero()
    one = ctx.one()
    two = ctx.two()
    ctx.assume(s.eq(zero) | s.eq(one))
    exponent_is_zero = e.eq(zero)
    mantissa_is_zero = m.eq(zero)
    is_zero = exponent_is_zero & mantissa_is_zero
    is_sub = exponent_is_zero & (~mantissa_is_zero)
    is_norm = ~exponent_is_zero
    signed = sign_multiplier(ctx, s)
    normal_value = (
        signed
        * (one + m * two ** (-ctx.real_val(E2M1.mantissa_bits)))
        * two ** (e - ctx.real_val(E2M1.exponent_bias))
    )
    subnormal_value = (
        signed
        * m
        * two ** (-ctx.real_val(E2M1.mantissa_bits))
        * two ** (one - ctx.real_val(E2M1.exponent_bias))
    )
    return e2m1(
        value=If(is_norm, normal_value, If(is_sub, subnormal_value, zero)),
        sign=s,
        exponent=e,
        mantissa=m,
        is_norm=is_norm,
        is_sub=is_sub,
        is_zero=is_zero,
    )


@Primitive(name="e2m1_pack", spec=e2m1_pack_spec)
def e2m1_pack(sign: Node, exponent: Node, mantissa: Node) -> Node:
    return _e2m1_alloc(sign, exponent, mantissa)


def e2m1_decode(x: Node) -> DecodedE2M1:
    @Primitive(name="e2m1_decode", spec=e2m1_decode_spec)
    def decode(x: Node) -> Node:
        sign = _e2m1_sign(x)
        exponent = _e2m1_exponent(x)
        mantissa = _e2m1_mantissa(x)
        bit = Const(UQ(0, 1, 0))
        exponent_is_nonzero = basic_or_reduce(exponent, bit.copy())
        exponent_is_zero = basic_invert(exponent_is_nonzero, bit.copy())
        mantissa_is_nonzero = basic_or_reduce(mantissa, bit.copy())
        mantissa_is_zero = basic_invert(mantissa_is_nonzero, bit.copy())
        is_sub = basic_and(exponent_is_zero, mantissa_is_nonzero, bit.copy())
        is_zero = basic_and(exponent_is_zero, mantissa_is_zero, bit.copy())
        return make_Tuple(
            sign, exponent, mantissa, exponent_is_nonzero, is_sub, is_zero
        )

    decoded = decode(x)
    return DecodedE2M1(
        sign=decoded[0],
        exponent=decoded[1],
        mantissa=decoded[2],
        is_norm=decoded[3],
        is_sub=decoded[4],
        is_zero=decoded[5],
    )


def e2m1_encodings_spec(m, e, ctx):
    integer_m = m * ctx.two() ** ctx.real_val(E2M1.mantissa_bits)
    overflow = e > ctx.real_val(E2M1.max_finite_code)
    return (
        If(overflow, ctx.real_val(E2M1.max_finite_mantissa), integer_m),
        If(overflow, ctx.real_val(E2M1.max_finite_code), e),
    )


@Primitive(name="e2m1_encodings", spec=e2m1_encodings_spec)
def e2m1_encodings(m_rounded: Node, e_rounded: Node):
    max_exponent = Const(UQ.from_int(E2M1.max_finite_code))
    overflow = uq_gt(e_rounded, max_exponent)
    final_e = basic_identity(
        uq_min(e_rounded, max_exponent),
        Const(UQ(0, E2M1.exponent_bits, 0)),
    )
    final_m = uq_fraction_to_integer(m_rounded)
    final_m = basic_mux_2_1(
        overflow,
        final_m,
        Const(UQ.from_int(E2M1.max_finite_mantissa)),
        final_m.copy(),
    )
    return make_Tuple(final_m, final_e)


def e2m1_encode_spec(s, e, m, ctx):
    finite_value = (
        sign_multiplier(ctx, s)
        * m
        * ctx.two() ** (e - ctx.real_val(E2M1.exponent_bias))
    )
    encoded = e2m1.encode(finite_value, ctx)
    return e2m1(
        value=encoded.value,
        sign=s,
        exponent=encoded.exponent,
        mantissa=encoded.mantissa,
        is_norm=encoded.is_norm,
        is_sub=encoded.is_sub,
        is_zero=encoded.is_zero,
    )


@Composite(name="e2m1_encode", spec=e2m1_encode_spec)
def e2m1_encode(s: Node, e: Node, m: Node) -> Node:
    """Encode using RNE, signed zero, and finite saturation."""

    if e.node_type.frac_bits != 0:
        raise ValueError("e2m1_encode exponent must have zero fractional bits")
    encode_exact_zero = uq_is_zero(m)
    normalized_m, normalized_e = normalize_to_1_xxx(m, e)
    shifted_m, shifted_e = shift_if_subnormal(
        normalized_m, normalized_e, subnormal_extra_bits=3
    )
    rounded_m, rounded_e = round_mantissa(
        drop_implicit_bit(shifted_m), shifted_e, target_bits=E2M1.mantissa_bits
    )
    final_m, final_e = e2m1_encodings(rounded_m, rounded_e)
    signed_zero = e2m1_pack(
        s,
        Const(UQ(0, E2M1.exponent_bits, 0)),
        Const(UQ(0, E2M1.mantissa_bits, 0)),
    )
    return if_then_else(
        encode_exact_zero,
        signed_zero,
        e2m1_pack(s, final_e, final_m),
    )
