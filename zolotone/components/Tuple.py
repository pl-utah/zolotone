from ..types import *
from ..ast import *
from ..utils import make_fixed_arguments


def basic_tuple_maker(*args) -> Op:
    def sign(*args: DataType) -> Tuple:
        return Tuple(*args)
    
    def op(*vals: RuntimeValue) -> TupleValue:
        dtype = Tuple(*(value.dtype for value in vals))
        return dtype.value(*vals)
    
    return Op(
        impl=make_fixed_arguments(op, [RuntimeValue] * len(args)),
        sign=make_fixed_arguments(sign, [DataType] * len(args)),
        c_lowering=lambda lowered_args, jittable: (
            f"std::array<uint64_t, {len(args)}>{{"
            + ", ".join(f"static_cast<uint64_t>({arg})" for arg in lowered_args)
            + "}"
        ) if jittable else f"std::make_tuple({', '.join(lowered_args)})",
        args=[*args],
        name=f"basic_tuple_maker_{len(args)}",
    )


@Primitive(name="make_Tuple", spec=lambda *args, ctx: tuple(args), c_inline=True)
def make_Tuple(*args: Node) -> Node:
    return basic_tuple_maker(*args)
