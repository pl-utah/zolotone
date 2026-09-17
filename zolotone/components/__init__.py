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
from ..ast import Composite, Node
from ..types import Q, UQ
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


LOSSLESS_COMPONENTS = (
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
    q_to_uq,
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
)
