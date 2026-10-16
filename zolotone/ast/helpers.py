from ..types import DataType, Q, Tuple, UQ
from ..spec import BoolExpr, Cases, case, FPExpr, If, RealExpr
from .node import Node
from .nodes import Const, Op, Primitive


def integer_lookup(
    x: Node,
    table: tuple[int, ...],
    out: Q | UQ,
) -> Node:
    """Select a typed tuple entry using the input's integer code."""
    from ..components.basics import basic_sub

    input_dtype = x.dtype
    if not isinstance(input_dtype, (Q, UQ)):
        raise TypeError("integer lookup input must be Q or UQ")
    bits = input_dtype.total_bits()
    minimum = -(1 << (bits - 1)) if isinstance(input_dtype, Q) else 0
    maximum = (1 << (bits - int(isinstance(input_dtype, Q)))) - 1
    if len(table) != maximum - minimum + 1:
        raise ValueError("integer lookup table must cover every input code")

    values = Const(Tuple(*(out for _ in table)).from_values(
        *(out.from_bits(raw) for raw in table)
    ))
    index_dtype = UQ(bits, 0)
    # Shift signed codes so the dtype's minimum value maps to tuple index zero.
    index = basic_sub(
        x,
        Const(index_dtype.from_bits(minimum & ((1 << bits) - 1))),
        out=index_dtype,
    )
    return Tuple_get_item(values, index)


def Copy(x: Node) -> Node:
    if not isinstance(x, Node):
        raise TypeError(f"Copy argument must be a Node, got {type(x).__name__}")

    dtype = x.dtype

    def copy_spec(x: dtype, ctx) -> dtype:
        return x

    @Primitive(name="Copy", spec=copy_spec, c_inline=True)
    def impl(x: Node) -> Node:
        return x

    return impl(x)


def _basic_get_item(x: Node, idx: int) -> Op:
    if not isinstance(x.dtype, Tuple):
        raise TypeError(f"Expected a Tuple node, got {x.dtype}")
    if idx >= len(x.dtype.items) or idx < 0:
        raise IndexError(f"Index is out of range for tuple {str(x)}, given {str(idx)}")
    
    def sign(x: Tuple) -> DataType:
        return x.items[idx]
    
    def op(x: tuple) -> object:
        return x[idx]
    
    return Op(
        impl=op,
        sign=sign,
        c_lowering=lambda lowered_args, jittable: (
            f"{lowered_args[0]}.at({idx})"
            if jittable
            else f"std::get<{idx}>({lowered_args[0]})"
        ),
        args=[x],
        name=f"_basic_get_item_{idx}",
    )

def _dynamic_get_item(x: Node, idx: Node) -> Node:
    input_type = x.dtype
    output_type = input_type.items[0]
    if not isinstance(output_type, (Q, UQ)) or not input_type.is_homogeneous:
        raise TypeError("Dynamic tuple indexing requires matching Q or UQ element types")
    if not isinstance(idx.dtype, UQ) or idx.dtype.frac_bits != 0:
        raise TypeError("Dynamic tuple indexing requires an unsigned integer index")
    index_type = idx.dtype
    length = len(input_type.items)

    def get_item_spec(values: input_type, index: index_type, ctx) -> output_type:
        ctx.assume((index >= ctx.zero()) & (index < ctx.real_val(length)))

        def select(start, end):
            if end - start == 1:
                return values[start]
            middle = (start + end) // 2
            return If(
                index < ctx.real_val(middle),
                select(start, middle), select(middle, end),
            )

        return select(0, length)

    def impl(values: tuple, index: int) -> int:
        return values[index]

    def sign(values: Tuple, index: UQ) -> DataType:
        return output_type

    def c_lowering(args: list[str], jittable: bool) -> str:
        return f"({args[0]}).at(static_cast<uint64_t>({args[1]}))"

    @Primitive(name="Tuple_get_item", spec=get_item_spec, c_inline=True)
    def get_item(values: Node, index: Node) -> Node:
        return Op(
            impl=impl, sign=sign, c_lowering=c_lowering,
            args=[values, index], name="_basic_dynamic_get_item",
        )

    return get_item(x, idx)


def Tuple_get_item(x: Node, idx: int | Node) -> Node:
    if not isinstance(x.dtype, Tuple):
        raise TypeError(f"Expected a Tuple node, got {x.dtype}")
    if isinstance(idx, Node):
        return _dynamic_get_item(x, idx)
    if idx >= len(x.dtype.items) or idx < 0:
        raise IndexError(f"Index is out of range for tuple {str(x)}, given {str(idx)}")

    input_type = x.dtype
    output_type = x.dtype.items[idx]

    def get_item_spec(x: input_type, ctx) -> output_type:
        return x[idx]

    @Primitive(name=f"Tuple_get_item_{idx}", spec=get_item_spec, c_inline=True)
    def impl(x: Node) -> Node:
        return _basic_get_item(x, idx)
    
    return impl(x)

def _if_then_else_spec(sel, in1, in0, ctx):
    branches_are_real = isinstance(in1, RealExpr) and isinstance(in0, RealExpr)
    branches_are_fp = (
        isinstance(in1, FPExpr)
        and isinstance(in0, FPExpr)
        and type(in1) is type(in0)
    )
    if not (branches_are_real or branches_are_fp):
        raise TypeError(
            "if_then_else spec branches must be matching real or floating-point "
            f"expressions, got {type(in1).__name__} and {type(in0).__name__}"
        )
    
    if isinstance(sel, BoolExpr):
        condition = sel
    elif isinstance(sel, RealExpr):
        condition = sel.ne(ctx.zero())
    else:
        raise TypeError()

    if branches_are_fp:
        return Cases(
            case(condition, in1),
            case(~condition, in0),
            ctx=ctx,
        )
    return If(condition, in1, in0)

def if_then_else(sel: Node, in1: Node, in0: Node) -> Node:
    args = (sel, in1, in0)
    if not all(isinstance(arg, Node) for arg in args):
        bad_args = [type(arg).__name__ for arg in args if not isinstance(arg, Node)]
        raise TypeError(
            f"if_then_else arguments must be Node instances, got {bad_args}"
        )
    if in1.dtype != in0.dtype:
        raise TypeError(
            "if_then_else branches must have matching descriptors, "
            f"got {in1.dtype} and {in0.dtype}"
        )

    selector_type = sel.dtype
    branch_type = in0.dtype

    def spec(
        sel: selector_type,
        in1: branch_type,
        in0: branch_type,
        ctx,
    ) -> branch_type:
        return _if_then_else_spec(sel, in1, in0, ctx)

    @Primitive(name="if_then_else", spec=spec, c_inline=True)
    def impl(sel: Node, in1: Node, in0: Node) -> Node:
        from ..components.basics import basic_mux_2_1
        return basic_mux_2_1(sel=sel, in0=in0, in1=in1, out=branch_type)

    return impl(sel, in1, in0)
