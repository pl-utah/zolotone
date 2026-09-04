from .basics import _unary_operator, _binary_operator, basic_invert
from ..types import *
from ..ast import *
from ..egglog import *
from ..spec import *

@Primitive(name="negate", spec=lambda x, ctx: ~x)
def negate(x: Node) -> Node:
    return basic_invert(x, out=Const(Bool().value(0)))
