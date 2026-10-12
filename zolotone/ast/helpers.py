from ..types import DataType, Q, Tuple, UQ
from ..spec import BoolExpr, Cases, case, FPExpr, If, RealExpr
from .node import Node
from .nodes import Op, Primitive


def integer_lookup(
    x: Node,
    *,
    table: tuple[int, ...],
    lowest_code: int,
    out: Q | UQ,
) -> Op:
    """Look up raw output bits by the input's integer encoding."""
    input_dtype = x.dtype
    if not isinstance(input_dtype, (Q, UQ)):
        raise TypeError("integer lookup input must be Q or UQ")

    def impl(raw: int) -> int:
        code = input_dtype._scaled_integer(raw) if isinstance(input_dtype, Q) else raw
        index = code - lowest_code
        if not 0 <= index < len(table):
            raise ValueError("integer lookup input is outside its required range")
        return table[index]

    def sign(value: DataType) -> DataType:
        return out

    values = ", ".join(map(str, table))
    code = "static_cast<int64_t>(raw)"
    if isinstance(input_dtype, Q) and input_dtype.total_bits() != 64:
        shift = 64 - input_dtype.total_bits()
        code = f"(static_cast<int64_t>(raw << {shift}) >> {shift})"

    def c_lowering(args: list[str], jittable: bool) -> str:
        return (
            "([](uint64_t raw) { "
            f"static constexpr uint64_t table[] = {{{values}}}; "
            f"const int64_t code = {code}; "
            f"return code >= {lowest_code} && "
            f"static_cast<uint64_t>(code - {lowest_code}) < {len(table)} "
            f"? table[static_cast<uint64_t>(code - {lowest_code})] : 0; "
            f"}})({args[0]})"
        )

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=c_lowering,
        args=[x],
        name="integer_lookup",
    )


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

def Tuple_get_item(x: Node, idx: int) -> Primitive:
    if not isinstance(x.dtype, Tuple):
        raise TypeError(f"Expected a Tuple node, got {x.dtype}")
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
