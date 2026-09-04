from typing import NamedTuple

from ..ast import *
from ..spec import *
from ..types import *
from .Tuple import make_Tuple
from .basics import *
from .common import *
from .rounding_routines import *
from .UQ import *


def _e5m2_mantissa(x: Node) -> Op:
    def impl(value: E5M2) -> UQ:
        return UQ(E5M2.mantissa_bits, 0).value(value.mantissa)
    
    def sign(value_type: E5M2) -> UQ:
        return UQ(E5M2.mantissa_bits, 0)
    
    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"({args[0]} & 3)",
        args=[x],
        name="_e5m2_mantissa",
    )


def _e5m2_exponent(x: Node) -> Op:
    def impl(value: E5M2) -> UQ:
        return UQ(E5M2.exponent_bits, 0).value(value.exponent)
    
    def sign(value_type: E5M2) -> UQ:
        return UQ(E5M2.exponent_bits, 0)
    
    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 2) & 31)",
        args=[x],
        name="_e5m2_exponent",
    )


def _e5m2_sign(x: Node) -> Op:
    def impl(value: E5M2) -> UQ:
        return UQ(1, 0).value(value.sign)
    
    def sign(value_type: E5M2) -> UQ:
        return UQ(1, 0)
    
    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda args, jittable: f"(({args[0]} >> 7) & 1)",
        args=[x],
        name="_e5m2_sign",
    )


def _e5m2_alloc(sign_bit: Node, exponent: Node, mantissa: Node) -> Op:
    def sign(
        sign_bit: DataType,
        exponent: DataType,
        mantissa: DataType,
    ) -> E5M2:
        return E5M2()
    
    def impl(
        sign_bit: RuntimeValue,
        exponent: RuntimeValue,
        mantissa: RuntimeValue,
    ) -> E5M2:
        return E5M2().from_fields(sign_bit.raw, exponent.raw, mantissa.raw)
    
    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda args, jittable: (
            f"(({E5M2().to_cpp_type(jittable=jittable)}({args[0]}) << 7) | "
            f"({E5M2().to_cpp_type(jittable=jittable)}({args[1]}) << 2) | "
            f"{E5M2().to_cpp_type(jittable=jittable)}({args[2]}))"
        ),
        args=[sign_bit, exponent, mantissa],
        name="_e5m2_alloc",
    )


class DecodedE5M2(NamedTuple):
    sign: Node
    exponent: Node
    mantissa: Node
    is_norm: Node
    is_sub: Node
    is_zero: Node
    is_inf: Node
    is_nan: Node


def e5m2_decode_spec(x: e5m2, ctx):
    decoded = x.decode()[1:]
    classification_count = len(x.classification_flags())
    fields = decoded[:-classification_count]
    classifications = decoded[-classification_count:]
    return fields + tuple(
        If(flag, ctx.one(), ctx.zero()) for flag in classifications
    )


def e5m2_pack_spec(s, e, m, ctx):
    zero = ctx.zero()
    one = ctx.one()
    two = ctx.two()
    max_exponent = ctx.real_val(E5M2.inf_code)
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
        * (one + m * two ** (-ctx.real_val(E5M2.mantissa_bits)))
        * two ** (e - ctx.real_val(E5M2.exponent_bias))
    )
    subnormal_value = (
        signed
        * m
        * two ** (-ctx.real_val(E5M2.mantissa_bits))
        * two ** (one - ctx.real_val(E5M2.exponent_bias))
    )
    value = If(
        is_norm,
        normal_value,
        If(is_sub, subnormal_value, If(is_zero, zero, ctx.fresh_real("special"))),
    )
    return e5m2(
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


@Primitive(name="e5m2_pack", spec=e5m2_pack_spec)
def e5m2_pack(sign: Node, exponent: Node, mantissa: Node) -> Node:
    return _e5m2_alloc(sign, exponent, mantissa)


def e5m2_decode(x: Node) -> DecodedE5M2:
    @Primitive(name="e5m2_decode", spec=e5m2_decode_spec)
    def decode(x: Node) -> Node:
        sign = _e5m2_sign(x)
        exponent = _e5m2_exponent(x)
        mantissa = _e5m2_mantissa(x)
        bit = Const(UQ(1, 0).value(0))
        exponent_is_all_ones = basic_and_reduce(exponent, bit.copy())
        exponent_is_not_all_ones = basic_invert(
            exponent_is_all_ones, bit.copy()
        )
        exponent_is_nonzero = basic_or_reduce(exponent, bit.copy())
        exponent_is_zero = basic_invert(exponent_is_nonzero, bit.copy())
        mantissa_is_nonzero = basic_or_reduce(mantissa, bit.copy())
        mantissa_is_zero = basic_invert(mantissa_is_nonzero, bit.copy())
        is_norm = basic_and(
            exponent_is_nonzero, exponent_is_not_all_ones, bit.copy()
        )
        is_sub = basic_and(exponent_is_zero, mantissa_is_nonzero, bit.copy())
        is_zero = basic_and(exponent_is_zero, mantissa_is_zero, bit.copy())
        is_inf = basic_and(exponent_is_all_ones, mantissa_is_zero, bit.copy())
        is_nan = basic_and(exponent_is_all_ones, mantissa_is_nonzero, bit.copy())
        return make_Tuple(
            sign, exponent, mantissa, is_norm, is_sub, is_zero, is_inf, is_nan
        )
    
    decoded = decode(x)
    return DecodedE5M2(
        sign=decoded[0],
        exponent=decoded[1],
        mantissa=decoded[2],
        is_norm=decoded[3],
        is_sub=decoded[4],
        is_zero=decoded[5],
        is_inf=decoded[6],
        is_nan=decoded[7],
    )


def e5m2_encodings_spec(m, e, ctx):
    is_inf = e >= ctx.real_val(E5M2.inf_code)
    return (
        If(is_inf, ctx.zero(), m * ctx.two() ** ctx.real_val(E5M2.mantissa_bits)),
        If(is_inf, ctx.real_val(E5M2.inf_code), e),
    )


@Primitive(name="e5m2_encodings", spec=e5m2_encodings_spec)
def e5m2_encodings(m_rounded: Node, e_rounded: Node):
    final_e = basic_identity(
        uq_min(e_rounded, Const(UQ.from_int(E5M2.inf_code))),
        Const(UQ.from_int(E5M2.inf_code)),
    )
    is_inf = basic_and_reduce(final_e, Const(UQ(1, 0).value(0)))
    final_m = basic_mux_2_1(
        is_inf,
        m_rounded,
        Const(UQ(1, 0).value(0)),
        m_rounded.copy(),
    )
    return make_Tuple(uq_fraction_to_integer(final_m), final_e)


def e5m2_encode_spec(s, e, m, ctx):
    finite_value = (
        sign_multiplier(ctx, s)
        * m
        * ctx.two() ** (e - ctx.real_val(E5M2.exponent_bias))
    )
    return e5m2.encode(finite_value, ctx)


@Composite(name="e5m2_encode", spec=e5m2_encode_spec)
def e5m2_encode(s: Node, e: Node, m: Node) -> Node:
    """Encode E5M2 with RNE, canonical exact zero, and signed infinity."""
    
    if e.dtype.frac_bits != 0:
        raise ValueError("e5m2_encode exponent must have zero fractional bits")
    encode_exact_zero = uq_is_zero(m)
    normalized_m, normalized_e = normalize_to_1_xxx(m, e)
    shifted_m, shifted_e = shift_if_subnormal(
        normalized_m, normalized_e, subnormal_extra_bits=3
    )
    rounded_m, rounded_e = round_mantissa(
        drop_implicit_bit(shifted_m),
        shifted_e,
        target_bits=E5M2.mantissa_bits,
    )
    final_m, final_e = e5m2_encodings(rounded_m, rounded_e)
    return if_then_else(
        encode_exact_zero,
        Const(E5M2().Zero()),
        e5m2_pack(s, final_e, final_m),
    )
