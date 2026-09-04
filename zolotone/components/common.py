"""Reusable fixed-point and single-bit helpers for component designs."""

from ..ast import *
from ..spec import *
from ..types import *
from .basics import *

def add_implicit_bit(x: Node) -> Primitive:
    assert x.dtype.int_bits == 0
    frac_bits = x.dtype.frac_bits

    def spec(x, ctx):
        return x + ctx.one()

    @Primitive(name="add_implicit_bit", spec=spec)
    def impl(x):
        return basic_concat(
            x=Const(UQ(1, 0).value(1)),
            y=x,
            out=Const(UQ(1, frac_bits).value(0)),
        )

    return impl(x)


def effective_significand(value) -> Node:
    """Return a decoded floating-point value's effective significand."""

    from .UQ import uq_integer_to_fraction, uq_resize

    assert hasattr(value, "is_sub")
    assert value.mantissa.dtype.frac_bits == 0
    fraction = uq_integer_to_fraction(value.mantissa)
    return if_then_else(
        value.is_norm,
        add_implicit_bit(fraction),
        uq_resize(fraction, 1, value.mantissa.dtype.int_bits),
    )


def effective_exponent(value) -> Node:
    """Return a decoded floating-point value's effective stored exponent."""

    assert hasattr(value, "is_sub")
    exponent_type = value.exponent.dtype
    assert exponent_type.frac_bits == 0
    return if_then_else(
        value.is_sub,
        Const(UQ(exponent_type.int_bits, exponent_type.frac_bits).value(1)),
        value.exponent,
    )


def and_spec(x, y, ctx):
    return If(
        x.eq(ctx.one()) & y.eq(ctx.one()),
        ctx.one(),
        ctx.zero(),
    )


@Primitive(name="bit_and", spec=and_spec, c_inline=True)
def bit_and(x: Node, y: Node) -> Node:
    assert x.dtype.total_bits() == 1, (
        f"bit_and expects single bit as an input, given: {x.dtype.total_bits()}"
    )
    assert y.dtype.total_bits() == 1, (
        f"bit_and expects single bit as an input, given: {y.dtype.total_bits()}"
    )
    return basic_and(x, y, Const(UQ(1, 0).value(0)))


def xor_spec(x, y, ctx):
    return If(x.ne(y), ctx.one(), ctx.zero())


@Primitive(name="bit_xor", spec=xor_spec, c_inline=True)
def bit_xor(x: Node, y: Node) -> Node:
    assert x.dtype.total_bits() == 1, (
        f"bit_xor expects single bit as an input, given: {x.dtype.total_bits()}"
    )
    assert y.dtype.total_bits() == 1, (
        f"bit_xor expects single bit as an input, given: {y.dtype.total_bits()}"
    )
    return basic_xor(x, y, Const(UQ(1, 0).value(0)))


def or_spec(x, y, ctx):
    return If(
        x.eq(ctx.one()) | y.eq(ctx.one()),
        ctx.one(),
        ctx.zero(),
    )


@Primitive(name="bit_or", spec=or_spec, c_inline=True)
def bit_or(x: Node, y: Node) -> Node:
    assert x.dtype.total_bits() == 1, (
        f"bit_or expects single bit as an input, given: {x.dtype.total_bits()}"
    )
    assert y.dtype.total_bits() == 1, (
        f"bit_or expects single bit as an input, given: {y.dtype.total_bits()}"
    )
    return basic_or(x, y, Const(UQ(1, 0).value(0)))


def neg_spec(x, ctx):
    return If(
        x.eq(ctx.one()),
        ctx.zero(),
        ctx.one(),
    )


@Primitive(name="bit_neg", spec=neg_spec, c_inline=True)
def bit_neg(x: Node) -> Node:
    assert x.dtype.total_bits() == 1, (
        f"bit_neg expects single bit as an input, given: {x.dtype.total_bits()}"
    )
    return basic_invert(x, Const(UQ(1, 0).value(0)))
