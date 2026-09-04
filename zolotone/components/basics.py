import typing as tp

from ..types import *
from ..utils import *
from ..ast import *


############ Constructors ##############

def _format_c_lowering(template: str, *args_ids: list[int]):
    def lower(args: list[str], jittable: bool) -> str:
        return template.format(*[args[idx] for idx in args_ids])
    return lower

def _cpp_cast(type_: DataType, expr: str, jittable: bool) -> str:
    return f"{type_.to_cpp_type(jittable=jittable)}({expr})"


def _cpp_zero(type_: DataType, jittable: bool) -> str:
    return _cpp_cast(type_, "0", jittable=jittable)


def _mask_literal(bits: int) -> str:
    return str((1 << bits) - 1)


def _impl_constructor(op):
    # To be called with op's arguments and output
    def impl(*args: RuntimeValue) -> RuntimeValue:
        *op_args, out = args
        val = op(*op_args)
        # TODO: check for truncation
        val = mask(val, out.dtype.total_bits())
        return out.dtype.value(val)
    return impl

def _sign_constructor() -> DataType:
    # To be called with arguments and output
    def sign(*args: DataType) -> DataType:
        return args[-1]
    return sign

def _ternary_operator(
    op: tp.Callable,
    x: Node,
    y: Node,
    z: Node,
    out: Node,
    c_lowering,
    name: str,
) -> Op:
    return Op(
        impl=make_fixed_arguments(_impl_constructor(op), [RuntimeValue] * 4),
        sign=make_fixed_arguments(_sign_constructor(), [DataType] * 4),
        c_lowering=c_lowering,
        args=[x, y, z, out],
        name=name)

def _binary_operator(op: tp.Callable, x: Node, y: Node, out: Node, c_lowering, name: str) -> Op:
    return Op(
        impl=make_fixed_arguments(_impl_constructor(op), [RuntimeValue] * 3),
        sign=make_fixed_arguments(_sign_constructor(), [DataType] * 3),
        c_lowering=c_lowering,
        args=[x, y, out],
        name=name)

def _unary_operator(op: tp.Callable, x: Node, out: Node, c_lowering, name: str) -> Op:
    return Op(
        impl=make_fixed_arguments(_impl_constructor(op), [RuntimeValue] * 2),
        sign=make_fixed_arguments(_sign_constructor(), [DataType] * 2),
        c_lowering=c_lowering,
        args=[x, out],
        name=name)

########## Ternary Operators ###########

def basic_mux_2_1(sel: Node, in0: Node, in1: Node, out: Node) -> Op:
    def op(sel: RuntimeValue, in0: RuntimeValue, in1: RuntimeValue) -> int:
        if sel.raw not in (0, 1):
            raise ValueError(f"Selector must be 0 or 1, got {sel.raw}")
        return in1.raw if sel.raw == 1 else in0.raw
    return _ternary_operator(
        op=op,
        x=sel,
        y=in0,
        z=in1,
        out=out,
        c_lowering=lambda lowered_args, jittable: (
            f"({lowered_args[0]} != 0 ? "
            f"{_cpp_cast(out.dtype, lowered_args[2], jittable=jittable)} : "
            f"{_cpp_cast(out.dtype, lowered_args[1], jittable=jittable)})"
        ),
        name="basic_mux_2_1",
    )

########### Binary Operators ###########

def basic_add(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: x.raw + y.raw,
        x=x,
        y=y,
        out=out,
        c_lowering=lambda lowered_args, jittable: (
            f"({_cpp_cast(out.dtype, lowered_args[0], jittable=jittable)} + "
            f"{_cpp_cast(out.dtype, lowered_args[1], jittable=jittable)})"
        ),
        name="basic_add",
    )

def basic_sub(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: x.raw - y.raw,
        x=x,
        y=y,
        out=out,
        c_lowering=lambda lowered_args, jittable: (
            f"({_cpp_cast(out.dtype, lowered_args[0], jittable=jittable)} - "
            f"{_cpp_cast(out.dtype, lowered_args[1], jittable=jittable)})"
        ),
        name="basic_sub",
    )

def basic_mul(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: x.raw * y.raw,
        x=x,
        y=y,
        out=out,
        c_lowering=lambda lowered_args, jittable: (
            f"({_cpp_cast(out.dtype, lowered_args[0], jittable=jittable)} * "
            f"{_cpp_cast(out.dtype, lowered_args[1], jittable=jittable)})"
        ),
        name="basic_mul",
    )

def basic_max(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: max(x.raw, y.raw),
        x=x,
        y=y,
        out=out,
        c_lowering=lambda lowered_args, jittable: (
            f"({lowered_args[0]} > {lowered_args[1]} ? "
            f"{_cpp_cast(out.dtype, lowered_args[0], jittable=jittable)} : "
            f"{_cpp_cast(out.dtype, lowered_args[1], jittable=jittable)})"
        ),
        name="basic_max",
    )

def basic_min(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: min(x.raw, y.raw),
        x=x,
        y=y,
        out=out,
        c_lowering=lambda lowered_args, jittable: (
            f"({lowered_args[0]} < {lowered_args[1]} ? "
            f"{_cpp_cast(out.dtype, lowered_args[0], jittable=jittable)} : "
            f"{_cpp_cast(out.dtype, lowered_args[1], jittable=jittable)})"
        ),
        name="basic_min",
    )

def basic_rshift(x: Node, amount: Node, out: Node) -> Op:
    width = x.dtype.total_bits()
    return _binary_operator(
        op=lambda x, amount: x.raw >> amount.raw,
        x=x,
        y=amount,
        out=out,
        c_lowering=lambda lowered_args, jittable: (
            f"({lowered_args[1]} >= {width} ? {_cpp_zero(x.dtype, jittable=jittable)} : "
            f"({lowered_args[0]} >> {lowered_args[1]}))"
        ),  # Shifting more than bitwidth is undefined behavior.
        name="basic_rshift",
    )

def basic_lshift(x: Node, amount: Node, out: Node) -> Op:
    out_width = out.dtype.total_bits()
    return _binary_operator(
        op=lambda x, amount: x.raw << amount.raw,
        x=x,
        y=amount,
        out=out,
        c_lowering=lambda lowered_args, jittable: (
            f"({lowered_args[1]} >= {out_width} ? {_cpp_zero(out.dtype, jittable=jittable)} : "
            f"({_cpp_cast(out.dtype, lowered_args[0], jittable=jittable)} << {lowered_args[1]}))"
        ),  # Avoid undefined behavior when shifting.
        name="basic_lshift",
    )

def basic_or(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: x.raw | y.raw,
        x=x,
        y=y,
        out=out,
        c_lowering=_format_c_lowering("({} | {})", 0, 1),
        name="basic_or",
    )
 
def basic_xor(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: x.raw ^ y.raw,
        x=x,
        y=y,
        out=out,
        c_lowering=_format_c_lowering(f"({{}} ^ {{}})", 0, 1),
        name="basic_xor",
    )

def basic_and(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: x.raw & y.raw,
        x=x,
        y=y,
        out=out,
        c_lowering=_format_c_lowering("({} & {})", 0, 1),
        name="basic_and",
    )

def basic_concat(x: Node, y: Node, out: Node) -> Op:
    shift = y.dtype.total_bits()
    return _binary_operator(
        op=lambda x, y: (x.raw << y.dtype.total_bits()) | y.raw,
        x=x,
        y=y,
        out=out,
        c_lowering=lambda lowered_args, jittable: (
            f"(({out.dtype.to_cpp_type(jittable=jittable)}({lowered_args[0]}) << {shift}) | {lowered_args[1]})"
        ),
        name="basic_concat",
    )

def basic_less(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: 1 if x.raw < y.raw else 0,
        x=x,
        y=y,
        out=out,
        c_lowering=_format_c_lowering("({} < {})", 0, 1),
        name="basic_less",
    )

def basic_less_or_equal(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: 1 if x.raw <= y.raw else 0,
        x=x,
        y=y,
        out=out,
        c_lowering=_format_c_lowering("({} <= {})", 0, 1),
        name="basic_less_or_equal",
    )

def basic_greater(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: 1 if x.raw > y.raw else 0,
        x=x,
        y=y,
        out=out,
        c_lowering=_format_c_lowering("({} > {})", 0, 1),
        name="basic_greater",
    )

def basic_greater_or_equal(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: 1 if x.raw >= y.raw else 0,
        x=x,
        y=y,
        out=out,
        c_lowering=_format_c_lowering("({} >= {})", 0, 1),
        name="basic_greater_or_equal",
    )

def basic_equal(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: 1 if x.raw == y.raw else 0,
        x=x,
        y=y,
        out=out,
        c_lowering=_format_c_lowering("({} == {})", 0, 1),
        name="basic_equal",
    )

def basic_not_equal(x: Node, y: Node, out: Node) -> Op:
    return _binary_operator(
        op=lambda x, y: 1 if x.raw != y.raw else 0,
        x=x,
        y=y,
        out=out,
        c_lowering=_format_c_lowering("({} != {})", 0, 1),
        name="basic_not_equal",
    )

########### Unary Operators ###########

# TODO: Truncation is possible if out is too small
def basic_select(x: Node, start: int, end: int, out: Node) -> Op:
    if start < end or end < 0:
        raise ValueError(f"Bad indexing: start={start}, end={end}")
    select_mask = _mask_literal(start - end + 1)
    return _unary_operator(
        op=lambda x: mask(x.raw >> end, start - end + 1),
        x=x,
        out=out,
        c_lowering=_format_c_lowering(
            f"(({{}} >> {end}) & {select_mask})",
            0,
        ),
        name="basic_select",
    )

# TODO: Truncation is possible if out is too small
def basic_invert(x: Node, out: Node) -> Op:
    invert_mask = _mask_literal(x.dtype.total_bits())
    return _unary_operator(
        op=lambda x: ((1 << x.dtype.total_bits()) - 1) - x.raw,
        x=x,
        out=out,
        c_lowering=lambda lowered_args, jittable: f"((~{lowered_args[0]}) & {invert_mask})",
        name="basic_invert",
    )

# TODO: Truncation is possible if out is too small
def basic_identity(x: Node, out: Node) -> Op:
    return _unary_operator(
        op=lambda x: x.raw,
        x=x,
        out=out,
        c_lowering=_format_c_lowering("{}", 0),
        name="basic_identity",
    )

def basic_or_reduce(x: Node, out: Node) -> Op:
    return _unary_operator(
        op=lambda x: 1 if x.raw > 0 else 0,
        x=x,
        out=out,
        c_lowering=_format_c_lowering("({} != 0)", 0),
        name="basic_or_reduce",
    )

def basic_and_reduce(x: Node, out: Node) -> Op:
    all_ones = _mask_literal(x.dtype.total_bits())
    return _unary_operator(
        op=lambda x: 1 if x.raw == ((1 << x.dtype.total_bits()) - 1) else 0,
        x=x,
        out=out,
        c_lowering=_format_c_lowering(f"({{}} == {all_ones})", 0),
        name="basic_and_reduce",
    )
