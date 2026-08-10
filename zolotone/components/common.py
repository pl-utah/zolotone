"""Reusable fixed-point and single-bit helpers for component designs."""

from ..ast import *
from ..spec import *
from ..types import *
from .basics import *

def add_implicit_bit(x: Node) -> Primitive:
    assert x.node_type.int_bits == 0
    frac_bits = x.node_type.frac_bits

    def spec(x, ctx):
        return x + ctx.one()

    @Primitive(name="add_implicit_bit", spec=spec)
    def impl(x):
        return basic_concat(
            x=Const(UQ(1, 1, 0)),
            y=x,
            out=Const(UQ(0, 1, frac_bits)),
        )

    return impl(x)


def and_spec(x, y, ctx):
    return If(
        x.eq(ctx.one()) & y.eq(ctx.one()),
        ctx.one(),
        ctx.zero(),
    )


@Primitive(name="bit_and", spec=and_spec, c_inline=True)
def bit_and(x: Node, y: Node) -> Node:
    assert x.node_type.total_bits() == 1, (
        f"bit_and expects single bit as an input, given: {x.node_type.total_bits()}"
    )
    assert y.node_type.total_bits() == 1, (
        f"bit_and expects single bit as an input, given: {y.node_type.total_bits()}"
    )
    return basic_and(x, y, Const(UQ(0, 1, 0)))


def xor_spec(x, y, ctx):
    return If(x.ne(y), ctx.one(), ctx.zero())


@Primitive(name="bit_xor", spec=xor_spec, c_inline=True)
def bit_xor(x: Node, y: Node) -> Node:
    assert x.node_type.total_bits() == 1, (
        f"bit_xor expects single bit as an input, given: {x.node_type.total_bits()}"
    )
    assert y.node_type.total_bits() == 1, (
        f"bit_xor expects single bit as an input, given: {y.node_type.total_bits()}"
    )
    return basic_xor(x, y, Const(UQ(0, 1, 0)))


def or_spec(x, y, ctx):
    return If(
        x.eq(ctx.one()) | y.eq(ctx.one()),
        ctx.one(),
        ctx.zero(),
    )


@Primitive(name="bit_or", spec=or_spec, c_inline=True)
def bit_or(x: Node, y: Node) -> Node:
    assert x.node_type.total_bits() == 1, (
        f"bit_or expects single bit as an input, given: {x.node_type.total_bits()}"
    )
    assert y.node_type.total_bits() == 1, (
        f"bit_or expects single bit as an input, given: {y.node_type.total_bits()}"
    )
    return basic_or(x, y, Const(UQ(0, 1, 0)))


def neg_spec(x, ctx):
    return If(
        x.eq(ctx.one()),
        ctx.zero(),
        ctx.one(),
    )


@Primitive(name="bit_neg", spec=neg_spec, c_inline=True)
def bit_neg(x: Node) -> Node:
    assert x.node_type.total_bits() == 1, (
        f"bit_neg expects single bit as an input, given: {x.node_type.total_bits()}"
    )
    return basic_invert(x, Const(UQ(0, 1, 0)))
