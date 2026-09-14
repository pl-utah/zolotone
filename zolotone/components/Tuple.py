from ..types import *
from ..ast import *
from ..spec import SpecContext
from ..utils import make_fixed_arguments


def basic_tuple_maker(*args) -> Op:
    def sign(*args: DataType) -> Tuple:
        return Tuple(*args)
    
    def op(*vals: object) -> tuple:
        return tuple(vals)
    
    return Op(
        impl=make_fixed_arguments(op, [object] * len(args)),
        sign=make_fixed_arguments(sign, [DataType] * len(args)),
        c_lowering=lambda lowered_args, jittable: (
            f"std::array<uint64_t, {len(args)}>{{"
            + ", ".join(f"static_cast<uint64_t>({arg})" for arg in lowered_args)
            + "}"
        ) if jittable else f"std::make_tuple({', '.join(lowered_args)})",
        args=[*args],
        name=f"basic_tuple_maker_{len(args)}",
    )


def make_Tuple(*args: Node) -> Node:
    if not all(isinstance(arg, Node) for arg in args):
        bad_args = [type(arg).__name__ for arg in args if not isinstance(arg, Node)]
        raise TypeError(f"make_Tuple arguments must be Node instances, got {bad_args}")

    output_type = Tuple(*(arg.dtype for arg in args))

    def make_tuple_spec(*values_and_ctx):
        return tuple(values_and_ctx[:-1])

    make_tuple_spec = make_fixed_arguments(
        make_tuple_spec,
        [arg.dtype for arg in args] + [SpecContext],
        return_type=output_type,
    )

    @Primitive(name="make_Tuple", spec=make_tuple_spec, c_inline=True)
    def impl(*values: Node) -> Node:
        return basic_tuple_maker(*values)

    return impl(*args)
