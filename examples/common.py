from zolotone import *


def add_implicit_bit(x: Node) -> Primitive:
    assert x.node_type.int_bits == 0
    frac_bits = x.node_type.frac_bits
    
    def spec(x, ctx):
        return x + ctx.real_val(1)
    
    @Primitive(name="add_implicit_bit", spec=spec)
    def impl(x):
        return basic_concat(x=Const(UQ(1, 1, 0)), y=x, out=Const(UQ(0, 1, frac_bits)))
    
    return impl(x)

def fraction_to_integer(x: Node) -> Primitive:
    if x.node_type.int_bits != 0:
        raise ValueError("fraction_to_integer expects a UQ value with zero integer bits")

    frac_bits = x.node_type.frac_bits

    def spec(x, ctx):
        return x * (ctx.real_val(2) ** ctx.real_val(frac_bits))

    @Primitive(name="fraction_to_integer", spec=spec, c_inline=True)
    def impl(x: Node):
        return basic_identity(x=x, out=Const(UQ(0, frac_bits, 0)))

    return impl(x)

def integer_to_fraction(x: Node) -> Primitive:
    if x.node_type.frac_bits != 0:
        raise ValueError("integer_to_fraction expects a UQ value with zero fractional bits")

    int_bits = x.node_type.int_bits

    def spec(x, ctx):
        return x * (ctx.real_val(2) ** (-ctx.real_val(int_bits)))

    @Primitive(name="integer_to_fraction", spec=spec, c_inline=True)
    def impl(x: Node):
        return basic_identity(x=x, out=Const(UQ(0, 0, int_bits)))

    return impl(x)

def and_spec(x, y, ctx):
    return If(
        x.eq(ctx.real_val(1)) & y.eq(ctx.real_val(1)),
        ctx.real_val(1),
        ctx.real_val(0),
    )

@Primitive(name="bit_and", spec=and_spec, c_inline=True)
def bit_and(x: Node, y: Node) -> Node:
    assert x.node_type.total_bits() == 1, f"bit_and expects single bit as an input, given: {x.node_type.total_bits()}"
    assert y.node_type.total_bits() == 1, f"bit_and expects single bit as an input, given: {y.node_type.total_bits()}"
    return basic_and(x, y, Const(UQ(0, 1, 0)))

def xor_spec(x, y, ctx):
    return If(x.ne(y), ctx.real_val(1), ctx.real_val(0))

@Primitive(name="bit_xor", spec=xor_spec, c_inline=True)
def bit_xor(x: Node, y: Node) -> Node:
    assert x.node_type.total_bits() == 1, f"bit_xor expects single bit as an input, given: {x.node_type.total_bits()}"
    assert y.node_type.total_bits() == 1, f"bit_xor expects single bit as an input, given: {y.node_type.total_bits()}"
    return basic_xor(x, y, Const(UQ(0, 1, 0)))

def or_spec(x, y, ctx):
    return If(
        x.eq(ctx.real_val(1)) | y.eq(ctx.real_val(1)),
        ctx.real_val(1),
        ctx.real_val(0),
    )

@Primitive(name="bit_or", spec=or_spec, c_inline=True)
def bit_or(x: Node, y: Node) -> Node:
    assert x.node_type.total_bits() == 1, f"bit_or expects single bit as an input, given: {x.node_type.total_bits()}"
    assert y.node_type.total_bits() == 1, f"bit_or expects single bit as an input, given: {y.node_type.total_bits()}"
    return basic_or(x, y, Const(UQ(0, 1, 0)))

def neg_spec(x, ctx):
    res = ctx.fresh_real('neg_res')
    ctx.assume(res.eq(ctx.real_val(1) - x))
    ctx.assume(res.eq(If(x.eq(ctx.real_val(1)), ctx.real_val(0), ctx.real_val(1))))
    ctx.assume(res.eq(If(x.eq(ctx.real_val(0)), ctx.real_val(1), ctx.real_val(0))))
    ctx.assume(res.eq(If(x.ne(ctx.real_val(0)), ctx.real_val(0), ctx.real_val(1))))
    ctx.assume(res.eq(If(x.ne(ctx.real_val(1)), ctx.real_val(1), ctx.real_val(0))))
    ctx.assume(res.eq(ctx.real_val(0)) | res.eq(ctx.real_val(1)))
    return res

@Primitive(name="bit_neg", spec=neg_spec, c_inline=True)
def bit_neg(x: Node) -> Node:
    assert x.node_type.total_bits() == 1, f"bit_neg expects single bit as an input, given: {x.node_type.total_bits()}"
    return basic_invert(x, Const(UQ(0, 1, 0)))
