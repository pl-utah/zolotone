from typing import NamedTuple

from ..ast import *
from ..spec import *
from ..types import *
from .Tuple import make_Tuple
from .basics import *
from .rounding_routines import (
    drop_implicit_bit,
    normalize_to_1_xxx,
    round_mantissa,
    shift_if_subnormal,
)
from .UQ import uq_fraction_to_integer, uq_is_zero, uq_min


########### Private Helpers ############


def _fp16_mantissa(x: Node) -> Op:
    def impl(x: Float16) -> UQ:
        return UQ(Float16.mantissa_bits, 0).value(x.mantissa)

    def sign(x: Float16) -> UQ:
        return UQ(Float16.mantissa_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda lowered_args, jittable: f"({lowered_args[0]} & 1023)",
        args=[x],
        name="_fp16_mantissa",
    )


def _fp16_exponent(x: Node) -> Op:
    def impl(x: Float16) -> UQ:
        return UQ(Float16.exponent_bits, 0).value(x.exponent)

    def sign(x: Float16) -> UQ:
        return UQ(Float16.exponent_bits, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda lowered_args, jittable: f"(({lowered_args[0]} >> 10) & 31)",
        args=[x],
        name="_fp16_exponent",
    )


def _fp16_sign(x: Node) -> Op:
    def impl(x: Float16) -> UQ:
        return UQ(1, 0).value(x.sign)

    def sign(x: Float16) -> UQ:
        return UQ(1, 0)

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda lowered_args, jittable: f"(({lowered_args[0]} >> 15) & 1)",
        args=[x],
        name="_fp16_sign",
    )


def _fp16_alloc(
    sign_bit: Node,
    exponent: Node,
    mantissa: Node,
) -> Op:
    def sign(
        sign_bit: DataType,
        exponent: DataType,
        mantissa: DataType,
    ) -> Float16:
        return Float16()

    def impl(
        sign_bit: RuntimeValue,
        exponent: RuntimeValue,
        mantissa: RuntimeValue,
    ) -> Float16:
        return Float16().from_fields(
            sign=sign_bit.raw,
            exponent=exponent.raw,
            mantissa=mantissa.raw,
        )

    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda lowered_args, jittable: (
            f"(({Float16().to_cpp_type(jittable=jittable)}({lowered_args[0]}) << 15) | "
            f"({Float16().to_cpp_type(jittable=jittable)}({lowered_args[1]}) << 10) | "
            f"{Float16().to_cpp_type(jittable=jittable)}({lowered_args[2]}))"
        ),
        args=[sign_bit, exponent, mantissa],
        name="_fp16_alloc",
    )


############## Public API ##############


class DecodedFP16(NamedTuple):
    sign: Node
    exponent: Node
    mantissa: Node
    is_norm: Node
    is_sub: Node
    is_zero: Node
    is_inf: Node
    is_nan: Node


def fp16_decode_spec(x: fp16, ctx):
    decoded = x.decode()[1:]
    classification_count = len(x.classification_flags())
    fields = decoded[:-classification_count]
    classifications = decoded[-classification_count:]
    return fields + tuple(
        If(flag, ctx.one(), ctx.zero()) for flag in classifications
    )


def fp16_pack_spec(s, e, m, ctx):
    zero = ctx.zero()
    one = ctx.one()
    two = ctx.two()
    max_exponent = ctx.real_val(Float16.inf_code)

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
        * (one + m * two ** (-ctx.real_val(Float16.mantissa_bits)))
        * two ** (e - ctx.real_val(Float16.exponent_bias))
    )
    subnormal_value = (
        signed
        * m
        * two ** (-ctx.real_val(Float16.mantissa_bits))
        * two ** (one - ctx.real_val(Float16.exponent_bias))
    )
    value = If(
        is_norm,
        normal_value,
        If(
            is_sub,
            subnormal_value,
            If(is_zero, zero, ctx.fresh_real("special")),
        ),
    )

    return fp16(
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


@Primitive(name="fp16_pack", spec=fp16_pack_spec)
def fp16_pack(sign: Node, exponent: Node, mantissa: Node) -> Node:
    return _fp16_alloc(sign, exponent, mantissa)


def fp16_decode(x: Node) -> DecodedFP16:
    @Primitive(name="fp16_decode", spec=fp16_decode_spec)
    def decode(x: Node) -> Node:
        sign = _fp16_sign(x)
        exponent = _fp16_exponent(x)
        mantissa = _fp16_mantissa(x)

        mantissa_is_nonzero = basic_or_reduce(
            mantissa,
            out=Const(UQ(1, 0).value(0)),
        )
        mantissa_is_zero = basic_invert(
            mantissa_is_nonzero,
            out=Const(UQ(1, 0).value(0)),
        )

        exponent_is_all_ones = basic_and_reduce(
            exponent,
            out=Const(UQ(1, 0).value(0)),
        )
        exponent_is_not_all_ones = basic_invert(
            exponent_is_all_ones,
            out=Const(UQ(1, 0).value(0)),
        )
        exponent_is_nonzero = basic_or_reduce(
            exponent,
            out=Const(UQ(1, 0).value(0)),
        )
        exponent_is_zero = basic_invert(
            exponent_is_nonzero,
            out=Const(UQ(1, 0).value(0)),
        )

        is_normal = basic_and(
            exponent_is_nonzero,
            exponent_is_not_all_ones,
            Const(UQ(1, 0).value(0)),
        )
        is_subnormal = basic_and(
            exponent_is_zero,
            mantissa_is_nonzero,
            Const(UQ(1, 0).value(0)),
        )
        is_zero = basic_and(
            exponent_is_zero,
            mantissa_is_zero,
            Const(UQ(1, 0).value(0)),
        )
        is_inf = basic_and(
            exponent_is_all_ones,
            mantissa_is_zero,
            Const(UQ(1, 0).value(0)),
        )
        is_nan = basic_and(
            exponent_is_all_ones,
            mantissa_is_nonzero,
            Const(UQ(1, 0).value(0)),
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
    return DecodedFP16(
        sign=decoded[0],
        exponent=decoded[1],
        mantissa=decoded[2],
        is_norm=decoded[3],
        is_sub=decoded[4],
        is_zero=decoded[5],
        is_inf=decoded[6],
        is_nan=decoded[7],
    )


def fp16_encodings_spec(m, e, ctx):
    return m * ctx.two() ** ctx.real_val(Float16.mantissa_bits), e


@Primitive(name="fp16_encodings", spec=fp16_encodings_spec)
def fp16_encodings(m_rounded: Node, e_rounded: Node):
    final_e_wide = uq_min(
        e_rounded,
        Const(UQ.from_int(Float16.inf_code)),
    )
    final_e = basic_identity(
        final_e_wide,
        Const(UQ.from_int(Float16.inf_code)),
    )
    is_inf = basic_and_reduce(final_e, Const(UQ(1, 0).value(0)))
    final_m = basic_mux_2_1(
        is_inf,
        m_rounded,
        Const(UQ(1, 0).value(0)),
        m_rounded.copy(),
    )
    return make_Tuple(uq_fraction_to_integer(final_m), final_e)


def fp16_encode_spec(s, e, m, ctx):
    signed = sign_multiplier(ctx, s)
    finite_value = (
        signed
        * m
        * ctx.two() ** (e - ctx.real_val(Float16.exponent_bias))
    )
    return fp16.encode(finite_value, ctx)


@Composite(name="fp16_encode", spec=fp16_encode_spec)
def fp16_encode(s: Node, e: Node, m: Node) -> Node:
    if e.dtype.frac_bits != 0:
        raise ValueError("fp16_encode exponent must have zero fractional bits")

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
        target_bits=Float16.mantissa_bits,
    )
    final_m, final_e = fp16_encodings(rounded_m, rounded_e)
    return if_then_else(
        encode_exact_zero,
        Const(Float16().Zero()),
        fp16_pack(s, final_e, final_m),
    )
