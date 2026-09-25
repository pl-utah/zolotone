from .common import *
from .rounding_routines import *
from .Float32 import *
from .Float16 import *
from .BFloat16 import *
from .E4M3FN import *
from .UE4M3 import *
from .E5M2 import *
from .E5M2FNUZ import *
from .E2M1 import *
from .Bool import bool_and, bool_eq, bool_or, negate
from ..ast import Composite, Node
from ..ast.helpers import _if_then_else_spec, if_then_else
from ..types import Bool, DataType, Q, UQ
from .Q import q_to_uq
from .UQ import uq_add, uq_mul


def _q_sign_extend_spec(x: Q, ctx) -> Q:
    return x

@Composite(name="_q_sign_extend", spec=_q_sign_extend_spec)
def _q_sign_extend(x: Node) -> Node:
    return q_sign_extend(x, n=1)

def _uq_zero_extend_spec(x: UQ, ctx) -> UQ:
    return x

@Composite(name="_uq_zero_extend", spec=_uq_zero_extend_spec)
def _uq_zero_extend(x: Node) -> Node:
    return uq_zero_extend(x, n=1)


def _if_then_else_component_spec(
    sel: Bool,
    in1: DataType,
    in0: DataType,
    ctx,
) -> DataType:
    return _if_then_else_spec(sel, in1, in0, ctx)


@Composite(name="_if_then_else", spec=_if_then_else_component_spec)
def _if_then_else(sel: Node, in1: Node, in0: Node) -> Node:
    return if_then_else(sel, in1, in0)


LOSSLESS_COMPONENTS = (
    bool_and,
    bool_eq,
    bool_or,
    negate,
    bool_to_uq,
    uq_to_bool,

    uq_lt,
    uq_gt,
    uq_le,
    uq_ge,
    uq_eq,
    uq_sub,
    uq_add,
    uq_mul,
    uq_max,
    uq_min,
    uq_to_q,
    uq_is_zero,
    _uq_zero_extend,

    q_lt,
    q_le,
    q_gt,
    q_ge,
    q_eq,
    q_ne,
    q_neg,
    q_add,
    q_sub,
    q_mul,
    q_abs,
    _q_sign_extend,
    q_to_uq,

    _if_then_else,
    bit_and,
    bit_or,
    bit_xor,
    bit_neg,
)
