from ..types import *
from .basics import *
from .Tuple import make_Tuple
from ..ast import *
from ..spec import *


########### Private Helpers ############

# Does not have spec
def _q_is_min_val(x: Node) -> Op:
    def impl(value: int) -> int:
        if value == (1 << (x.dtype.total_bits() - 1)):
            res = 1
        else:
            res = 0
        return res
    
    def sign(x: Q) -> UQ:
        return UQ(1, 0)
    
    return Op(
        impl=impl,
        sign=sign,
        c_lowering=lambda lowered_args, jittable: f"({lowered_args[0]} == {1 << (x.dtype.total_bits() - 1)})",
        args=[x],
        name="_q_is_min_val")
        

############# Public API ###############

# Function does not care about int_bits/frac_bits types, it takes their values
def q_alloc(int_bits: Node, frac_bits: Node) -> Op:
    if int_bits.constant is None or frac_bits.constant is None:
        raise TypeError("q_alloc's arguments must be constant")
    result_dtype = Q(int_bits.constant.raw, frac_bits.constant.raw)

    def sign(x: UQ, y: UQ) -> Q:
        return result_dtype

    def impl(x: int, y: int) -> int:
        return 0

    return Op(
        sign=sign,
        impl=impl,
        c_lowering=lambda lowered_args, jittable: "0",
        args=[int_bits, frac_bits],
        name="q_alloc")


def q_signs_xor_spec(x: Q, y: Q, ctx) -> UQ:
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
        out=UQ(1, 0)
    )


def q_lt_spec(x: Q, y: Q, ctx) -> Bool:
    return x < y


@Primitive(name="q_lt", spec=q_lt_spec)
def q_lt(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_mux_2_1(
        sel=q_signs_xor(x, y),
        in0=basic_less(aligned_x, aligned_y, Bool()),  # same signs
        in1=basic_mux_2_1(  # different signs!
            sel=q_sign_bit(x),
            in0=Const(Bool().from_bits(0)),  # x > y
            in1=Const(Bool().from_bits(1)),  # x < y
            out=Bool()),
        out=Bool())


def q_le_spec(x: Q, y: Q, ctx) -> Bool:
    return x <= y


@Primitive(name="q_le", spec=q_le_spec)
def q_le(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_mux_2_1(
        sel=q_signs_xor(x, y),
        in0=basic_less_or_equal(aligned_x, aligned_y, Bool()),  # same signs
        in1=basic_mux_2_1(  # different signs!
            sel=q_sign_bit(x),
            in0=Const(Bool().from_bits(0)),  # x > y
            in1=Const(Bool().from_bits(1)),  # x < y
            out=Bool()),
        out=Bool())


def q_gt_spec(x: Q, y: Q, ctx) -> Bool:
    return x > y


@Primitive(name="q_gt", spec=q_gt_spec)
def q_gt(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_mux_2_1(
        sel=q_signs_xor(x, y),
        in0=basic_greater(aligned_x, aligned_y, Bool()),  # same signs
        in1=basic_mux_2_1(  # different signs!
            sel=q_sign_bit(x),
            in0=Const(Bool().from_bits(1)),  # x > y
            in1=Const(Bool().from_bits(0)),  # x < y
            out=Bool()),
        out=Bool())


def q_ge_spec(x: Q, y: Q, ctx) -> Bool:
    return x >= y


@Primitive(name="q_ge", spec=q_ge_spec)
def q_ge(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_mux_2_1(
        sel=q_signs_xor(x, y),
        in0=basic_greater_or_equal(aligned_x, aligned_y, Bool()),  # same signs
        in1=basic_mux_2_1(  # different signs!
            sel=q_sign_bit(x),
            in0=Const(Bool().from_bits(1)),  # x > y
            in1=Const(Bool().from_bits(0)),  # x < y
            out=Bool()),
        out=Bool())


def q_eq_spec(x: Q, y: Q, ctx) -> Bool:
    return x.eq(y)


@Primitive(name="q_eq", spec=q_eq_spec)
def q_eq(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_equal(aligned_x, aligned_y, out=Bool())


def q_ne_spec(x: Q, y: Q, ctx) -> Bool:
    return x.ne(y)


@Primitive(name="q_ne", spec=q_ne_spec)
def q_ne(x: Node, y: Node) -> Node:
    aligned_x, aligned_y = q_aligner(x, y, max, max)
    return basic_not_equal(aligned_x, aligned_y, out=Bool())


def q_aligner(x: Node,
              y: Node,
              int_aggr: tp.Callable,
              frac_aggr: tp.Callable) -> Primitive:
    int_bits = int_aggr(x.dtype.int_bits, y.dtype.int_bits)
    frac_bits = frac_aggr(x.dtype.frac_bits, y.dtype.frac_bits)

    def q_aligner_spec(x: Q, y: Q, ctx) -> Tuple:
        return x, y

    @Primitive(name="q_aligner", spec=q_aligner_spec)
    def impl(x: Node, y: Node) -> Node:
        def align(x):
            # Step 1. Align frac bits
            shift = frac_bits - x.dtype.frac_bits
            if shift < 0:
                raise NotImplementedError("truncation is not implemented yet")
            if shift > 0:
                x = basic_lshift(
                    x,
                    Const(UQ.from_int(shift)), 
                    Q(x.dtype.int_bits, frac_bits))
            
            # Step 2. Align integer bits
            shift = int_bits - x.dtype.int_bits
            if shift < 0:
                raise NotImplementedError("truncation is not implemented yet")
            if shift > 0:
                return q_sign_extend(x, shift)
            return x

        return make_Tuple(align(x), align(y))

    return impl(x, y)


def q_sign_bit_spec(x: Q, ctx) -> UQ:
    sign = ctx.fresh_real("sign")
    ctx.assume(sign.eq(ctx.zero()) | sign.eq(ctx.one()))
    ctx.assume(x.eq(sign_multiplier(ctx, sign) * abs(x)))
    return sign

@Primitive(name="q_sign_bit", spec=q_sign_bit_spec, c_inline=True)
def q_sign_bit(x: Node) -> Node:
    start = x.dtype.int_bits + x.dtype.frac_bits - 1
    return basic_select(
        x=x,
        start=start,
        end=start,
        out=UQ(1, 0),
    )


def q_sign_extend(x: Node, n: int) -> Node:
    if not isinstance(n, int):
        raise TypeError(f"n should be an int, {type(n).__name__} is given")
    if n < 0:
        raise ValueError(f"n should be a non-negative integer, {n} is given")

    def q_sign_extend_spec(x: Q, ctx) -> Q:
        return x

    @Primitive(name="q_sign_extend", spec=q_sign_extend_spec)
    def impl(x: Node) -> Node:
        if n == 0:
            return x.copy()

        sign_bit = q_sign_bit(x)
        shift_amount = Const(UQ.from_int(n))

        shifted = basic_lshift(
            x=sign_bit,
            amount=shift_amount,
            out=UQ(n + 1, 0),
        )

        upper_bits = basic_sub(
            x=shifted,
            y=sign_bit,
            out=UQ(n, 0),
        )

        res = basic_concat(
            x=upper_bits,
            y=x,
            out=Q(x.dtype.int_bits + n, x.dtype.frac_bits),
        )
        return res

    return impl(x)


def q_resize_spec(x: Q, ctx) -> Q:
    return x


def q_resize(x: Node, int_bits: int, frac_bits: int) -> Node:
    """Exactly widen a signed fixed-point value to the requested shape."""
    
    if not isinstance(int_bits, int) or not isinstance(frac_bits, int):
        raise TypeError("int_bits and frac_bits must be integers")
    if int_bits < x.dtype.int_bits:
        raise ValueError("q_resize cannot narrow the integer field")
    if frac_bits < x.dtype.frac_bits:
        raise ValueError("q_resize cannot narrow the fractional field")
    if int_bits + frac_bits < 1:
        raise ValueError("q_resize requires at least one total bit")
    
    @Primitive(name="q_resize", spec=q_resize_spec)
    def impl(x: Node) -> Node:
        resized = q_sign_extend(x, int_bits - x.dtype.int_bits)
        fractional_extension = frac_bits - x.dtype.frac_bits
        if fractional_extension == 0:
            return resized
        return basic_lshift(
            resized,
            Const(UQ.from_int(fractional_extension)),
            Q(int_bits, frac_bits),
        )
    
    return impl(x)


def q_neg_spec(x: Q, ctx) -> Q:
    return -x


@Primitive(name="q_neg", spec=q_neg_spec)
def q_neg(x: Node) -> Node:
    x_inv = basic_invert(x, x.dtype)
    x_neg = basic_add(x_inv, Const(UQ.from_int(1)), x.dtype)

    x_is_min = _q_is_min_val(x)
    x_overflow = basic_invert(basic_xor(x, x, x.dtype), x.dtype)

    return basic_mux_2_1(sel=x_is_min, in0=x_neg, in1=x_overflow, out=x.dtype)


def q_add_spec(x: Q, y: Q, ctx) -> Q:
    return x + y


@Primitive(name="q_add", spec=q_add_spec)
def q_add(x: Node, y: Node) -> Node:
    x_adj, y_adj = q_aligner(
        x=x,
        y=y,
        int_aggr=lambda x, y: max(x, y) + 1,
        frac_aggr=lambda x, y: max(x, y),
    )
    return basic_add(x_adj, y_adj, x_adj.dtype)


def q_sub_spec(x: Q, y: Q, ctx) -> Q:
    return x - y


@Primitive(name="q_sub", spec=q_sub_spec)
def q_sub(x: Node, y: Node) -> Node:
    x_adj, y_adj = q_aligner(
        x=x,
        y=y,
        int_aggr=lambda x, y: max(x, y) + 1,
        frac_aggr=lambda x, y: max(x, y),
    )
    root = basic_sub(x_adj, y_adj, x_adj.dtype)
    return root


def q_mul_spec(x: Q, y: Q, ctx) -> Q:
    return x * y


@Primitive(name="q_mul", spec=q_mul_spec)
def q_mul(x: Node, y: Node) -> Node:
    target_int_bits = x.dtype.int_bits + y.dtype.int_bits
    target_frac_bits = x.dtype.frac_bits + y.dtype.frac_bits
    x_adj = q_sign_extend(x, y.dtype.total_bits())
    y_adj = q_sign_extend(y, x.dtype.total_bits())
    out = Q(target_int_bits, target_frac_bits)
    return basic_mul(x=x_adj, y=y_adj, out=out)


def q_lshift_spec(x: Q, n: DataType, ctx) -> Q:
    return x * (ctx.two() ** n)


@Primitive(name="q_lshift", spec=q_lshift_spec)
def q_lshift(x: Node, n: Node) -> Node:
    return basic_lshift(x=x, amount=n, out=x.dtype)


# Assumes that x is positive
def q_to_uq_spec(x: Q, ctx) -> UQ:
    return x


@Primitive(name="q_to_uq", spec=q_to_uq_spec, c_inline=True)
def q_to_uq(x: Node) -> Node:
    int_bits = x.dtype.int_bits - 1
    frac_bits = x.dtype.frac_bits
    return basic_identity(x=x, out=UQ(int_bits, frac_bits))


def q_rshift_spec(x: Q, n: DataType, ctx) -> Q:
    return x * (ctx.two() ** (-n))


@Primitive(name="q_rshift", spec=q_rshift_spec)
def q_rshift(x: Node, n: Node) -> Node:
    return basic_rshift(x=x, amount=n, out=x.dtype)


def q_rshift_jam_spec(x: Q, n: DataType, ctx) -> Q:
    """Ideal signed scaling represented by a sign-symmetric jammed shift."""

    return x * (ctx.two() ** (-n))


@Primitive(name="q_rshift_jam", spec=q_rshift_jam_spec)
def q_rshift_jam(x: Node, n: Node) -> Node:
    """Right-shift a signed value and jam discarded magnitude bits.

    Jamming is performed on the absolute magnitude and the original sign is
    restored afterward.  This avoids both a logical shift of the packed
    two's-complement value and the negative-infinity bias of an arithmetic
    shift.  Sign-extending before taking the absolute value also makes the
    minimum representable two's-complement input safe.
    """

    # UQ imports q_alloc from this module, so keep this reverse dependency
    # local to avoid a module-import cycle.
    from .UQ import uq_rshift_jam, uq_to_q

    widened = q_sign_extend(x, 1)
    sign = q_sign_bit(widened)
    magnitude = q_to_uq(q_abs(widened))
    shifted_magnitude = uq_rshift_jam(magnitude, n)
    signed_result = q_add_sign(uq_to_q(shifted_magnitude), sign)
    return basic_identity(
        signed_result,
        Q(x.dtype.int_bits, x.dtype.frac_bits),
    )


def q_add_sign_spec(x: Q, s: UQ, ctx) -> Q:
    return sign_multiplier(ctx, s) * x

@Primitive(name="q_add_sign", spec=q_add_sign_spec)
def q_add_sign(x: Node, s: Node) -> Node:
    return basic_mux_2_1(
        sel=s,
        in0=x.copy(),
        in1=q_neg(x),
        out=x.dtype,
    )


def q_abs_spec(x: Q, ctx) -> Q:
    return abs(x)


@Primitive(name="q_abs", spec=q_abs_spec)
def q_abs(x: Node) -> Node:
    sign_bit = q_sign_bit(x)  # UQ1.0
    return q_add_sign(x, sign_bit)


def q_is_zero_spec(x: Q, ctx) -> UQ:
    result = ctx.fresh_bool("q_is_zero")
    ctx.assume(result.eq(x.eq(ctx.zero())))
    return result

@Primitive(name="q_is_zero", spec=q_is_zero_spec)
def q_is_zero(x: Node) -> Node:
    return basic_invert(basic_or_reduce(x, UQ(1, 0)), UQ(1, 0))
