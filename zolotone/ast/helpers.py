from ..types import DataType, RuntimeValue, Tuple, TupleValue
from ..spec import BoolExpr, Cases, case, FPExpr, If, RealExpr
from .node import Node
from .nodes import Op, Primitive


@Primitive(name="Copy", spec=lambda x, ctx: x, c_inline=True)
def Copy(x: Node) -> Node:
    return x


def _basic_get_item(x: Node, idx: int) -> Op:
    if not isinstance(x.dtype, Tuple):
        raise TypeError(f"Expected a Tuple node, got {x.dtype}")
    if idx >= len(x.dtype.items) or idx < 0:
        raise IndexError(f"Index is out of range for tuple {str(x)}, given {str(idx)}")
    
    def sign(x: Tuple) -> DataType:
        return x.items[idx]
    
    def op(x: TupleValue) -> RuntimeValue:
        return x.items[idx]
    
    return Op(
        impl=op,
        sign=sign,
        c_lowering=lambda lowered_args, jittable: f"{lowered_args[0]}[{idx}]" if jittable else f"std::get<{idx}>({lowered_args[0]})",
        args=[x],
        name=f"_basic_get_item_{idx}",
    )

def Tuple_get_item(x: Node, idx: int) -> Primitive:
    @Primitive(name=f"Tuple_get_item_{idx}", spec=lambda x, ctx: x[idx], c_inline=True)
    def impl(x: Node) -> Node:
        return _basic_get_item(x, idx)
    
    return impl(x)

def if_then_else_spec(sel, in1, in0, ctx):
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

@Primitive(name="if_then_else", spec=if_then_else_spec, c_inline=True)
def if_then_else(sel: Node, in1: Node, in0: Node) -> Node:
    from ..components.basics import basic_mux_2_1
    assert in1.dtype == in0.dtype, "Non-deterministic type"
    return basic_mux_2_1(sel=sel, in0=in0, in1=in1, out=in0.dtype)
