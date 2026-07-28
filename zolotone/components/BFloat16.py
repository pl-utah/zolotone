from ..types import *
from .Tuple import make_Tuple
from .basics import *
from ..ast import *
from ..spec import *

########### Private Helpers ############

def _bf16_mantissa(x: Node) -> Op:
    def impl(x: BFloat16) -> UQ:
        return UQ(x.mantissa, 7, 0)
    
    def sign(x: BFloat16T) -> UQT:
        return UQT(7, 0)
    
    return Op(
            impl=impl,
            sign=sign,
            c_lowering=lambda lowered_args, jittable: f"({lowered_args[0]} & 127)",
            args=[x],
            name="_bf16_mantissa")

def _bf16_exponent(x: Node) -> Op:
    def impl(x: BFloat16) -> UQ:
        return UQ(x.exponent, 8, 0)
    
    def sign(x: BFloat16T) -> UQT:
        return UQT(8, 0)
    
    return Op(
            impl=impl,
            sign=sign,
            c_lowering=lambda lowered_args, jittable: f"(({lowered_args[0]} >> 7) & 255)",
            args=[x],
            name="_bf16_exponent")

def _bf16_sign(x: Node) -> Op:
    def impl(x: BFloat16) -> UQ:
        return UQ(x.sign, 1, 0)
    
    def sign(x: BFloat16T) -> UQT:
        return UQT(1, 0)
    
    return Op(
            impl=impl,
            sign=sign,
            c_lowering=lambda lowered_args, jittable: f"(({lowered_args[0]} >> 15) & 1)",
            args=[x],
            name="_bf16_sign")

def _bf16_alloc(
    sign_bit: Node,
    exponent: Node,
    mantissa: Node,
) -> Op:
    def sign(
        sign_bit: StaticType,
        exponent: StaticType,
        mantissa: StaticType,
    ) -> BFloat16T:
        return BFloat16T()

    def impl(
        sign_bit: RuntimeType,
        exponent: RuntimeType,
        mantissa: RuntimeType,
    ) -> BFloat16:
        return BFloat16.from_fields(
            sign=sign_bit.val,
            exponent=exponent.val,
            mantissa=mantissa.val,
        )

    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda lowered_args, jittable: (
            f"(({BFloat16T().to_cpp_type(jittable=jittable)}({lowered_args[0]}) << 15) | "
            f"({BFloat16T().to_cpp_type(jittable=jittable)}({lowered_args[1]}) << 7) | "
            f"{BFloat16T().to_cpp_type(jittable=jittable)}({lowered_args[2]}))"
        ),
        args=[sign_bit, exponent, mantissa],
        name="_bf16_alloc",
    )

############## Public API ##############

def bf16_decode_spec(x: bf16, ctx):
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
        return If(flag, ctx.real_val(1), ctx.real_val(0))

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


def bf16_pack_spec(s, e, m, ctx):
    zero = ctx.real_val(0)
    one = ctx.real_val(1)
    two = ctx.real_val(2)
    max_exponent = ctx.real_val(BFloat16.inf_code)

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
        * (one + m * two ** (-ctx.real_val(BFloat16.mantissa_bits)))
        * two ** (e - ctx.real_val(BFloat16.exponent_bias))
    )
    subnormal_value = (
        signed
        * m
        * two ** (-ctx.real_val(BFloat16.mantissa_bits))
        * two ** (one - ctx.real_val(BFloat16.exponent_bias))
    )
    value = If(is_norm, normal_value, If(is_sub, subnormal_value, zero))

    return bf16(
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


@Primitive(name="bf16_pack", spec=bf16_pack_spec)
def bf16_pack(sign: Node, exponent: Node, mantissa: Node) -> Node:
    return _bf16_alloc(sign, exponent, mantissa)


@Primitive(name="bf16_decode", spec=bf16_decode_spec)
def bf16_decode(x: Node) -> Node:
    sign = _bf16_sign(x)
    exponent = _bf16_exponent(x)
    mantissa = _bf16_mantissa(x)

    mantissa_is_nonzero = basic_or_reduce(
        mantissa,
        out=Const(UQ(0, 1, 0)),
    )
    mantissa_is_zero = basic_invert(
        mantissa_is_nonzero,
        out=Const(UQ(0, 1, 0)),
    )

    exponent_is_all_ones = basic_and_reduce(
        exponent,
        out=Const(UQ(0, 1, 0)),
    )
    exponent_is_not_all_ones = basic_invert(
        exponent_is_all_ones,
        out=Const(UQ(0, 1, 0)),
    )
    exponent_is_nonzero = basic_or_reduce(
        exponent,
        out=Const(UQ(0, 1, 0)),
    )
    exponent_is_zero = basic_invert(
        exponent_is_nonzero,
        out=Const(UQ(0, 1, 0)),
    )

    is_normal = basic_and(
        exponent_is_nonzero,
        exponent_is_not_all_ones,
        Const(UQ(0, 1, 0)),
    )
    is_subnormal = basic_and(
        exponent_is_zero,
        mantissa_is_nonzero,
        Const(UQ(0, 1, 0)),
    )
    is_zero = basic_and(
        exponent_is_zero,
        mantissa_is_zero,
        Const(UQ(0, 1, 0)),
    )
    is_inf = basic_and(
        exponent_is_all_ones,
        mantissa_is_zero,
        Const(UQ(0, 1, 0)),
    )
    is_nan = basic_and(
        exponent_is_all_ones,
        mantissa_is_nonzero,
        Const(UQ(0, 1, 0)),
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
