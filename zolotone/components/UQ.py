import typing as tp

from ..types import *
from .basics import *
from .Q import q_alloc
from .Tuple import make_Tuple
from ..ast import *
from ..spec import *


############## Public API ##############

# Function does not care about int_bits/frac_bits types, it takes their values
# Allocates UQ at runtime
def uq_alloc(int_bits: Node,
             frac_bits: Node) -> Op:
    def sign(int_bits: StaticType, frac_bits: StaticType) -> UQT:
        if int_bits.runtime_val is not None and frac_bits.runtime_val is not None:
            return UQT(int_bits.runtime_val.val, frac_bits.runtime_val.val)
        raise TypeError("uq_alloc's arguments depend on a variable")
    
    def impl(int_bits: RuntimeType, frac_bits: RuntimeType) -> UQ:
        return UQ(0, int_bits.val, frac_bits.val)
    
    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda lowered_args, jittable: "0",
        args=[int_bits, frac_bits],
        name="uq_alloc")


@Primitive(name="uq_lt", spec=lambda x, y, ctx: x < y)
def uq_lt(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = uq_aligner(x, y, None, max)
    return basic_less(aligned_x, aligned_y, out=Const(Bool(0)))


@Primitive(name="uq_le", spec=lambda x, y, ctx: x <= y)
def uq_le(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = uq_aligner(x, y, None, max)
    return basic_less_or_equal(aligned_x, aligned_y, out=Const(Bool(0)))


@Primitive(name="uq_gt", spec=lambda x, y, ctx: x > y)
def uq_gt(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = uq_aligner(x, y, None, max)
    return basic_greater(aligned_x, aligned_y, out=Const(Bool(0)))


@Primitive(name="uq_ge", spec=lambda x, y, ctx: x >= y)
def uq_ge(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = uq_aligner(x, y, None, max)
    return basic_greater_or_equal(aligned_x, aligned_y, out=Const(Bool(0)))


@Primitive(name="uq_eq", spec=lambda x, y, ctx: x.eq(y))
def uq_eq(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = uq_aligner(x, y, None, max)
    return basic_equal(aligned_x, aligned_y, out=Const(Bool(0)))


@Primitive(name="uq_ne", spec=lambda x, y, ctx: x.ne(y))
def uq_ne(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = uq_aligner(x, y, None, max)
    return basic_not_equal(aligned_x, aligned_y, out=Const(Bool(0)))


def uq_aligner(x: Node,
               y: Node,
               int_aggr: tp.Callable | None,
               frac_aggr: tp.Callable | None) -> Node:

    _int_aggr = lambda arg: int_aggr(x.node_type.int_bits, y.node_type.int_bits) if int_aggr is not None else arg.node_type.int_bits
    _frac_aggr = lambda arg: frac_aggr(x.node_type.frac_bits, y.node_type.frac_bits) if frac_aggr is not None else arg.node_type.frac_bits
    
    @Primitive(name="uq_aligner", spec=lambda x, y, ctx: (x, y))
    def impl(x: Node, y: Node) -> Node:
        def align(x):
            int_bits = _int_aggr(x)
            frac_bits = _frac_aggr(x)
            
            shift = frac_bits - x.node_type.frac_bits
            if shift < 0:
                raise NotImplementedError("truncation is not implemented yet")  # truncation
            
            # frac bits extension
            if shift > 0:
                return basic_lshift(x, Const(UQ.from_int(shift)), Const(UQ(0, int_bits, frac_bits)))

            # no extension
            if int_bits == x.node_type.int_bits and frac_bits == x.node_type.frac_bits:
                return x
            
            # int bits extension
            return basic_identity(x, Const(UQ(0, int_bits, frac_bits)))
        return make_Tuple(align(x), align(y))
    
    return impl(x, y)


def uq_frac_bits(x: Node) -> Node:
    if not isinstance(x.node_type, UQT):
        raise TypeError(f"Expected UQT node, got {type(x.node_type).__name__}")
    return Const(UQ.from_int(x.node_type.frac_bits))


def uq_int_bits(x: Node) -> Node:
    if not isinstance(x.node_type, UQT):
        raise TypeError(f"Expected UQT node, got {type(x.node_type).__name__}")
    return Const(UQ.from_int(x.node_type.int_bits))


def uq_fraction_to_integer(x: Node) -> Primitive:
    """Reinterpret a UQ fractional field as an integer field."""

    if x.node_type.int_bits != 0:
        raise ValueError(
            "uq_fraction_to_integer expects a UQ value with zero integer bits"
        )
    frac_bits = x.node_type.frac_bits

    def spec(x, ctx):
        return x * ctx.two() ** ctx.real_val(frac_bits)

    @Primitive(name="uq_fraction_to_integer", spec=spec, c_inline=True)
    def impl(x: Node):
        return basic_identity(x=x, out=Const(UQ(0, frac_bits, 0)))

    return impl(x)


def uq_integer_to_fraction(x: Node) -> Primitive:
    """Reinterpret a UQ integer field as a fractional field."""

    if x.node_type.frac_bits != 0:
        raise ValueError(
            "uq_integer_to_fraction expects a UQ value with zero fractional bits"
        )
    int_bits = x.node_type.int_bits

    def spec(x, ctx):
        return x * ctx.two() ** (-ctx.real_val(int_bits))

    @Primitive(name="uq_integer_to_fraction", spec=spec, c_inline=True)
    def impl(x: Node):
        return basic_identity(x=x, out=Const(UQ(0, 0, int_bits)))

    return impl(x)


def uq_zero_extend(x: Node, n: int) -> Node:
    if not isinstance(n, int):
        raise TypeError(f"n must be int, given: {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, given: {n}")
    
    @Primitive(name="uq_zero_extend", spec=lambda x, ctx: x)
    def impl(x: Node) -> Node:
        int_bits = uq_add(uq_int_bits(x), Const(UQ.from_int(n)))
        frac_bits = uq_frac_bits(x)
        out = uq_alloc(int_bits, frac_bits)
        return basic_identity(x=x, out=out)
    
    return impl(x)


@Primitive(name="uq_add", spec=lambda x, y, ctx: x + y)
def uq_add(x: Node, y: Node) -> Node:
    target_int_bits = max(x.node_type.int_bits, y.node_type.int_bits) + 1
    target_frac_bits = max(x.node_type.frac_bits, y.node_type.frac_bits)
    
    x_adj, y_adj = uq_aligner(
        x=x,
        y=y,
        int_aggr=None,
        frac_aggr=lambda x, y: max(x, y),
    )
    root = basic_add(
        x=x_adj,
        y=y_adj,
        out=Const(UQ(0, target_int_bits, target_frac_bits)),
    )
    return root


@Primitive(name="uq_sub", spec=lambda x, y, ctx: x - y)
def uq_sub(x: Node, y: Node) -> Node:
    target_int_bits = max(x.node_type.int_bits, y.node_type.int_bits) + 1
    target_frac_bits = max(x.node_type.frac_bits, y.node_type.frac_bits)
    
    x_adj, y_adj = uq_aligner(
        x=x,
        y=y,
        int_aggr=None,
        frac_aggr=lambda x, y: max(x, y),
    )
    root = basic_sub(
        x=x_adj, 
        y=y_adj,
        out=Const(UQ(0, target_int_bits, target_frac_bits)),
    )
    return root


@Primitive(name="uq_max", spec=lambda x, y, ctx: x.max(y))
def uq_max(x: Node, y: Node) -> Node:
    target_int_bits = max(x.node_type.int_bits, y.node_type.int_bits)
    target_frac_bits = max(x.node_type.frac_bits, y.node_type.frac_bits)
    
    x_adj, y_adj = uq_aligner(
        x=x, 
        y=y, 
        int_aggr=None, 
        frac_aggr=lambda x, y: max(x, y),
    )
    root = basic_max(
        x=x_adj, 
        y=y_adj,
        out=Const(UQ(0, target_int_bits, target_frac_bits)),
    )
    return root


@Primitive(name="uq_min", spec=lambda x, y, ctx: x.min(y))
def uq_min(x: Node, y: Node) -> Node:
    target_int_bits = max(x.node_type.int_bits, y.node_type.int_bits)
    target_frac_bits = max(x.node_type.frac_bits, y.node_type.frac_bits)
    
    x_adj, y_adj = uq_aligner(
        x=x,
        y=y,
        int_aggr=None, 
        frac_aggr=lambda x, y: max(x, y),
    )
    root = basic_min(
        x=x_adj, 
        y=y_adj,
        out=Const(UQ(0, target_int_bits, target_frac_bits)),
    )
    return root


@Primitive(name="uq_mul", spec=lambda x, y, ctx: x * y)
def uq_mul(x: Node, y: Node) -> Node:
    target_int_bits = x.node_type.int_bits + y.node_type.int_bits
    target_frac_bits = x.node_type.frac_bits + y.node_type.frac_bits
    
    root = basic_mul(
        x=x,
        y=y,
        out=Const(UQ(0, target_int_bits, target_frac_bits)),
    )
    return root


@Primitive(name="uq_to_q", spec=lambda x, ctx: x, c_inline=True)
def uq_to_q(x: Node) -> Node:
    int_bits = uq_add(uq_int_bits(x), Const(UQ.from_int(1)))
    frac_bits = uq_frac_bits(x)
    out = q_alloc(int_bits, frac_bits)
    return basic_identity(x=x, out=out)


@Primitive(name="uq_rshift", spec=lambda x, amount, ctx: x * (ctx.two() ** (-amount)))
def uq_rshift(x: Node, amount: Node) -> Node:
    return basic_rshift(
        x=x,
        amount=amount,
        out=x.copy(),
    )


@Primitive(name="uq_rshift_jam", spec=lambda x, amount, ctx: x * (ctx.two() ** (-amount)))
def uq_rshift_jam(x: Node, amount: Node) -> Node:
    one = Const(
        UQ(1, x.node_type.int_bits, x.node_type.frac_bits)
    )
    shifted_bit_mask = basic_sub(
        basic_lshift(
            one,
            amount,
            out=x.copy(),
        ),
        one,
        out=x.copy(),
    )
    sticky_masked_bits = basic_and(x, shifted_bit_mask, out=x.copy())
    sticky_bit = basic_or_reduce(sticky_masked_bits, Const(UQ(0, 1, 0)))
    return basic_or(uq_rshift(x, amount), sticky_bit, x.copy())


# TODO: truncation
@Primitive(name="uq_lshift", spec=lambda x, amount, ctx: x * (ctx.two() ** amount))
def uq_lshift(x: Node, amount: Node) -> Node:
    root = basic_lshift(
        x=x,
        amount=amount,
        out=x.copy(),
    )
    return root


def uq_select(x: Node, start: int, end: int) -> Node:
    width = start - end + 1
    x_frac_bits = x.node_type.frac_bits
    
    frac_bits = max(0, min(start, x_frac_bits - 1) - end + 1) if x_frac_bits > 0 else 0
    int_bits = width - frac_bits
    
    def spec(x, ctx):
        slice1 = ctx.fresh_real("slice1")
        slice2 = ctx.fresh_real("slice2")
        ctx.assume(x.eq(slice1 + slice2))
        return slice1
    
    @Primitive(name="uq_select", spec=spec, c_inline=True)
    def impl(x: Node) -> Node:
        out = Const(UQ(0, int_bits, frac_bits))
        root = basic_select(x, start, end, out)
        return root
    
    return impl(x)


def uq_split(x: Node, idx: int) -> Node:
    # Returns Tuple(lo, hi), where lo are the lowest `idx` bits.
    if not isinstance(idx, int):
        raise TypeError(f"idx must be int, given: {idx}")
    total_bits = x.node_type.total_bits()
    if idx <= 0 or idx >= total_bits:
        raise ValueError(f"idx must be in (0, {total_bits}), given: {idx}")
    
    x_int_bits = x.node_type.int_bits
    x_frac_bits = x.node_type.frac_bits
    
    lo_width = idx
    lo_frac_bits = min(x_frac_bits, lo_width)
    lo_int_bits = lo_width - lo_frac_bits
    
    hi_width = total_bits - lo_width
    hi_frac_bits = x_frac_bits - lo_frac_bits
    hi_int_bits = hi_width - hi_frac_bits
    
    def spec(x, ctx):
        lo = ctx.fresh_real("lo")
        hi = ctx.fresh_real("hi")
        
        two = ctx.two()
        zero = ctx.zero()
        
        # x = hi * 2^lo_int_bits + lo * 2^-hi_frac_bits
        ctx.assume(
            x.eq(
                (hi * two ** ctx.real_val(lo_int_bits))
                + (lo * two ** (-ctx.real_val(hi_frac_bits)))
            )
        )
        # Preserve the exact zero behavior explicitly. Although lo and hi are
        # unsigned, the simplifier does not always derive both equalities from
        # the decomposition equation when x has simplified to zero.
        ctx.assume(
            (~x.eq(zero))
            | (lo.eq(zero) & hi.eq(zero))
        )
        return (lo, hi)
    
    @Primitive(name="uq_split", spec=spec)
    def impl(x: Node) -> Node:
        lo = Const(UQ(0, lo_int_bits, lo_frac_bits))
        lo = basic_select(x, idx - 1, 0, lo)
        
        hi = Const(UQ(0, hi_int_bits, hi_frac_bits))
        hi = basic_select(x, total_bits - 1, idx, hi)
        
        return make_Tuple(lo, hi)
    
    return impl(x)


def uq_resize(x: Node, int_bits: int, frac_bits: int) -> Node:
    if not isinstance(int_bits, int) or not isinstance(frac_bits, int):
        raise TypeError("int_bits and frac_bits must be integers")
    if int_bits < 0 or frac_bits < 0:
        raise ValueError("int_bits and frac_bits must be non-negative")

    frac_shift = frac_bits - x.node_type.frac_bits
    
    @Primitive(name="uq_resize", spec=lambda x, ctx: x)
    def impl(x: Node) -> Node:
        out = Const(UQ(0, int_bits, frac_bits))
        if frac_shift > 0:
            return basic_lshift(
                x=x,
                amount=Const(UQ.from_int(frac_shift)),
                out=out,
            )
        if frac_shift < 0:
            return basic_rshift(
                x=x,
                amount=Const(UQ.from_int(-frac_shift)),
                out=out,
            )
        return basic_identity(x=x, out=out)
    return impl(x)


@Primitive(name="uq_is_zero", spec=lambda x, ctx: x.eq(ctx.zero()))
def uq_is_zero(x: Node) -> Node:
    return basic_invert(basic_or_reduce(x, Const(UQ(0, 1, 0))), Const(UQ(0, 1, 0)))
