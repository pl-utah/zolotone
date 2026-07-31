from ..types import *
from .basics import *
from .Tuple import make_Tuple
from ..ast import *
from ..spec import *


########### Private Helpers ############

# Does not have spec
def _q_is_min_val(x: Node) -> Op:
    def impl(x: Q) -> UQ:
        if x.val == (1 << (x.total_bits() - 1)):
            res = 1
        else:
            res = 0
        return UQ(res, 1, 0)
    
    def sign(x: QT) -> UQT:
        return UQT(1, 0)
    
    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda lowered_args, jittable: f"({lowered_args[0]} == {1 << (x.node_type.total_bits() - 1)})",
        args=[x],
        name="_q_is_min_val")
        

############# Public API ###############

# Function does not care about int_bits/frac_bits types, it takes their values
def q_alloc(int_bits: Node, frac_bits: Node) -> Op:
    def sign(x: StaticType, y: StaticType) -> QT:
        if x.runtime_val is None or y.runtime_val is None:
            raise TypeError("q_alloc's arguments depend on a variable")
        return QT(x.runtime_val.val, y.runtime_val.val)

    def impl(x: RuntimeType, y: RuntimeType) -> Q:
        return Q(0, x.val, y.val)

    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda lowered_args, jittable: "0",
        args=[int_bits, frac_bits],
        name="q_alloc")


def q_signs_xor_spec(x, y, ctx):
    x_sign = ctx.fresh_real("x_sign")
    y_sign = ctx.fresh_real("y_sign")
    res = ctx.fresh_real("xored_signs")

    ctx.assume(x_sign.eq(ctx.one()) | x_sign.eq(ctx.zero()))
    ctx.assume(y_sign.eq(ctx.one()) | y_sign.eq(ctx.zero()))
    ctx.assume(res.eq(ctx.one()) | res.eq(ctx.zero()))
    x_sign_value = sign_multiplier(ctx, x_sign)
    y_sign_value = sign_multiplier(ctx, y_sign)
    ctx.assume(x.eq(x_sign_value * abs(x)))
    ctx.assume(y.eq(y_sign_value * abs(y)))
    ctx.assume(res.eq(If(x_sign.ne(y_sign), ctx.one(), ctx.zero())))
    return res

@Primitive(name="q_signs_xor", spec=q_signs_xor_spec)
def q_signs_xor(x: Node, y: Node) -> Node:
    return basic_xor(
        q_sign_bit(x),
        q_sign_bit(y),
        out=Const(UQ(0, 1, 0))
    )


@Primitive(name="q_lt", spec=lambda x, y, ctx: x < y)
def q_lt(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_mux_2_1(
        sel=q_signs_xor(x, y),
        in0=basic_less(aligned_x, aligned_y, Const(Bool(0))),  # same signs
        in1=basic_mux_2_1(  # different signs!
            sel=q_sign_bit(x),
            in0=Const(Bool(0)),  # x > y
            in1=Const(Bool(1)),  # x < y
            out=Const(Bool(0))),
        out=Const(Bool(0)))


@Primitive(name="q_le", spec=lambda x, y, ctx: x <= y)
def q_le(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_mux_2_1(
        sel=q_signs_xor(x, y),
        in0=basic_less_or_equal(aligned_x, aligned_y, Const(Bool(0))),  # same signs
        in1=basic_mux_2_1(  # different signs!
            sel=q_sign_bit(x),
            in0=Const(Bool(0)),  # x > y
            in1=Const(Bool(1)),  # x < y
            out=Const(Bool(0))),
        out=Const(Bool(0)))


@Primitive(name="q_gt", spec=lambda x, y, ctx: x > y)
def q_gt(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_mux_2_1(
        sel=q_signs_xor(x, y),
        in0=basic_greater(aligned_x, aligned_y, Const(Bool(0))),  # same signs
        in1=basic_mux_2_1(  # different signs!
            sel=q_sign_bit(x),
            in0=Const(Bool(1)),  # x > y
            in1=Const(Bool(0)),  # x < y
            out=Const(Bool(0))),
        out=Const(Bool(0)))


@Primitive(name="q_ge", spec=lambda x, y, ctx: x >= y)
def q_ge(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_mux_2_1(
        sel=q_signs_xor(x, y),
        in0=basic_greater_or_equal(aligned_x, aligned_y, Const(Bool(0))),  # same signs
        in1=basic_mux_2_1(  # different signs!
            sel=q_sign_bit(x),
            in0=Const(Bool(1)),  # x > y
            in1=Const(Bool(0)),  # x < y
            out=Const(Bool(0))),
        out=Const(Bool(0)))


@Primitive(name="q_eq", spec=lambda x, y, ctx: x.eq(y))
def q_eq(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_equal(aligned_x, aligned_y, out=Const(Bool(0)))


@Primitive(name="q_ne", spec=lambda x, y, ctx: x.ne(y))
def q_ne(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_not_equal(aligned_x, aligned_y, out=Const(Bool(0)))


def q_aligner(x: Node,
              y: Node,
              int_aggr: tp.Callable,
              frac_aggr: tp.Callable) -> Primitive:
    int_bits = int_aggr(x.node_type.int_bits, y.node_type.int_bits)
    frac_bits = frac_aggr(x.node_type.frac_bits, y.node_type.frac_bits)

    @Primitive(name="q_aligner", spec=lambda x, y, ctx: (x, y))
    def impl(x: Node, y: Node) -> Node:
        def align(x):
            # Step 1. Align frac bits
            shift = frac_bits - x.node_type.frac_bits
            if shift < 0:
                raise NotImplementedError("truncation is not implemented yet")
            if shift > 0:
                x = basic_lshift(
                    x,
                    Const(UQ.from_int(shift)), 
                    Const(Q(0, x.node_type.int_bits, frac_bits)))
            
            # Step 2. Align integer bits
            shift = int_bits - x.node_type.int_bits
            if shift < 0:
                raise NotImplementedError("truncation is not implemented yet")
            if shift > 0:
                return q_sign_extend(x, shift)
            return x

        return make_Tuple(align(x), align(y))

    return impl(x, y)


def q_sign_bit_spec(x, ctx):
    sign = ctx.fresh_real("sign")
    ctx.assume(sign.eq(ctx.zero()) | sign.eq(ctx.one()))
    ctx.assume(x.eq(sign_multiplier(ctx, sign) * abs(x)))
    return sign

@Primitive(name="q_sign_bit", spec=q_sign_bit_spec, c_inline=True)
def q_sign_bit(x: Node) -> Node:
    start = x.node_type.int_bits + x.node_type.frac_bits - 1
    return basic_select(
        x=x,
        start=start,
        end=start,
        out=Const(UQ(0, 1, 0)),
    )


def q_sign_extend(x: Node, n: int) -> Node:
    if not isinstance(n, int):
        raise TypeError(f"n should be an int, {type(n).__name__} is given")
    if n < 0:
        raise ValueError(f"n should be a non-negative integer, {n} is given")

    @Primitive(name="q_sign_extend", spec=lambda x, ctx: x)
    def impl(x: Node) -> Node:
        if n == 0:
            return x.copy()

        sign_bit = q_sign_bit(x)
        shift_amount = Const(UQ.from_int(n))

        shifted = basic_lshift(
            x=sign_bit,
            amount=shift_amount,
            out=Const(UQ(0, n + 1, 0)),
        )

        upper_bits = basic_sub(
            x=shifted,
            y=sign_bit,
            out=Const(UQ(0, n, 0)),
        )

        int_bits = Const(UQ.from_int(x.node_type.int_bits + n))
        frac_bits = Const(UQ.from_int(x.node_type.frac_bits))
        out = q_alloc(int_bits, frac_bits)

        res = basic_concat(
            x=upper_bits,
            y=x,
            out=out,
        )
        return res

    return impl(x)


@Primitive(name="q_neg", spec=lambda x, ctx: -x)
def q_neg(x: Node) -> Node:
    x_inv = basic_invert(x, x.copy())
    x_neg = basic_add(x_inv, Const(UQ.from_int(1)), x.copy())

    x_is_min = _q_is_min_val(x)
    x_overflow = basic_invert(basic_xor(x, x, x.copy()), x.copy())

    return basic_mux_2_1(sel=x_is_min, in0=x_neg, in1=x_overflow, out=x.copy())


@Primitive(name="q_add", spec=lambda x, y, ctx: x + y)
def q_add(x: Node, y: Node) -> Node:
    x_adj, y_adj = q_aligner(
        x=x,
        y=y,
        int_aggr=lambda x, y: max(x, y) + 1,
        frac_aggr=lambda x, y: max(x, y),
    )
    return basic_add(x_adj, y_adj, x_adj.copy())


@Primitive(name="q_sub", spec=lambda x, y, ctx: x - y)
def q_sub(x: Node, y: Node) -> Node:
    x_adj, y_adj = q_aligner(
        x=x,
        y=y,
        int_aggr=lambda x, y: max(x, y) + 1,
        frac_aggr=lambda x, y: max(x, y),
    )
    root = basic_sub(x_adj, y_adj, x_adj.copy())
    return root


@Primitive(name="q_mul", spec=lambda x, y, ctx: x * y)
def q_mul(x: Node, y: Node) -> Node:
    target_int_bits = x.node_type.int_bits + y.node_type.int_bits
    target_frac_bits = x.node_type.frac_bits + y.node_type.frac_bits
    x_adj = q_sign_extend(x, y.node_type.total_bits())
    y_adj = q_sign_extend(y, x.node_type.total_bits())
    out = Const(Q(0, target_int_bits, target_frac_bits))
    return basic_mul(x=x_adj, y=y_adj, out=out)


@Primitive(name="q_lshift", spec=lambda x, n, ctx: x * (ctx.two() ** n))
def q_lshift(x: Node, n: Node) -> Node:
    return basic_lshift(x=x, amount=n, out=x.copy())


# Assumes that x is positive
@Primitive(name="q_to_uq", spec=lambda x, ctx: x, c_inline=True)
def q_to_uq(x: Node) -> Node:
    int_bits = x.node_type.int_bits - 1
    frac_bits = x.node_type.frac_bits
    return basic_identity(x=x, out=Const(UQ(0, int_bits, frac_bits)))


@Primitive(name="q_rshift", spec=lambda x, n, ctx: x * (ctx.two() ** (-n)))
def q_rshift(x: Node, n: Node) -> Node:
    return basic_rshift(x=x, amount=n, out=x.copy())


def q_add_sign_spec(x, s, ctx):
    return sign_multiplier(ctx, s) * x

@Primitive(name="q_add_sign", spec=q_add_sign_spec)
def q_add_sign(x: Node, s: Node) -> Node:
    return basic_mux_2_1(
        sel=s,
        in0=x.copy(),
        in1=q_neg(x),
        out=x.copy(),
    )


@Primitive(name="q_abs", spec=lambda x, ctx: abs(x))
def q_abs(x: Node) -> Node:
    sign_bit = q_sign_bit(x)  # UQ1.0
    return q_add_sign(x, sign_bit)


def q_is_zero_spec(x, ctx):
    result = ctx.fresh_bool("q_is_zero")
    ctx.assume(result.eq(x.eq(ctx.zero())))
    return result

@Primitive(name="q_is_zero", spec=q_is_zero_spec)
def q_is_zero(x: Node) -> Node:
    return basic_invert(basic_or_reduce(x, Const(UQ(0, 1, 0))), Const(UQ(0, 1, 0)))
