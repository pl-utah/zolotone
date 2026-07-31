from typing import NamedTuple

from ..types import *
from ..ast import *
from ..spec import If, fp32, sign_multiplier
from .Tuple import make_Tuple
from .basics import *


########### Private Helpers ############

def _fp32_mantissa(x: Node) -> Op:
    def impl(x: Float32) -> UQ:
        return UQ(x.mantissa, Float32.mantissa_bits, 0)
    
    def sign(x: Float32T) -> UQT:
        return UQT(Float32.mantissa_bits, 0)
    
    return Op(
            impl=impl,
            sign=sign,
            c_lowering=lambda lowered_args, jittable: f"({lowered_args[0]} & 8388607)",
            args=[x],
            name="_fp32_mantissa")

def _fp32_exponent(x: Node) -> Op:
    def impl(x: Float32) -> UQ:
        return UQ(x.exponent, Float32.exponent_bits, 0)
    
    def sign(x: Float32T) -> UQT:
        return UQT(Float32.exponent_bits, 0)
    
    return Op(
            impl=impl,
            sign=sign,
            c_lowering=lambda lowered_args, jittable: f"(({lowered_args[0]} >> 23) & 255)",
            args=[x],
            name="_fp32_exponent")

def _fp32_sign(x: Node) -> Op:
    def impl(x: Float32) -> UQ:
        return UQ(x.sign, 1, 0)
    
    def sign(x: Float32T) -> UQT:
        return UQT(1, 0)
    
    return Op(
            impl=impl,
            sign=sign,
            c_lowering=lambda lowered_args, jittable: f"(({lowered_args[0]} >> 31) & 1)",
            args=[x],
            name="_fp32_sign")

def _fp32_alloc(sign_bit: Node,
                  exponent: Node,
                  mantissa: Node) -> Op:
    def sign(sign_bit: StaticType, exponent: StaticType, mantissa: StaticType) -> Float32T:
        return Float32T()
    
    def impl(sign_bit: RuntimeType, exponent: RuntimeType, mantissa: RuntimeType) -> Float32:
        return Float32.from_fields(sign_bit.val, exponent.val, mantissa.val)
    
    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda lowered_args, jittable: (
            f"(({Float32T().to_cpp_type(jittable=jittable)}({lowered_args[0]}) << 31) | "
            f"({Float32T().to_cpp_type(jittable=jittable)}({lowered_args[1]}) << 23) | "
            f"{Float32T().to_cpp_type(jittable=jittable)}({lowered_args[2]}))"
        ),
        args=[sign_bit, exponent, mantissa],
        name="_fp32_alloc")

############## Public API ##############


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
    (
        sign,
        exponent,
        mantissa,
        is_normal,
        is_subnormal,
        is_zero,
        is_inf,
        is_nan,
    ) = x.decode()[1:]

    def bool_to_real(flag):
        return If(flag, ctx.one(), ctx.zero())

    return (
        sign,
        exponent,
        mantissa,
        bool_to_real(is_normal),
        bool_to_real(is_subnormal),
        bool_to_real(is_zero),
        bool_to_real(is_inf),
        bool_to_real(is_nan),
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
    value = If(is_norm, normal_value, If(is_sub, subnormal_value, If(is_zero, zero, ctx.fresh_real("special"))))

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

        mantissa_is_nonzero = basic_or_reduce(mantissa, out=Const(UQ(0, 1, 0)))
        mantissa_is_zero = basic_invert(mantissa_is_nonzero, out=Const(UQ(0, 1, 0)))

        exponent_is_all_ones = basic_and_reduce(exponent, out=Const(UQ(0, 1, 0)))
        exponent_is_not_all_ones = basic_invert(exponent_is_all_ones, out=Const(UQ(0, 1, 0)))
        exponent_is_nonzero = basic_or_reduce(exponent, out=Const(UQ(0, 1, 0)))
        exponent_is_zero = basic_invert(exponent_is_nonzero, out=Const(UQ(0, 1, 0)))

        is_normal = basic_and(exponent_is_nonzero, exponent_is_not_all_ones, Const(UQ(0, 1, 0)))
        is_subnormal = basic_and(exponent_is_zero, mantissa_is_nonzero, Const(UQ(0, 1, 0)))
        is_zero = basic_and(exponent_is_zero, mantissa_is_zero, Const(UQ(0, 1, 0)))
        is_inf = basic_and(exponent_is_all_ones, mantissa_is_zero, Const(UQ(0, 1, 0)))
        is_nan = basic_and(exponent_is_all_ones, mantissa_is_nonzero, Const(UQ(0, 1, 0)))

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
