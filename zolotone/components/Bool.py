from .basics import _unary_operator, _binary_operator, basic_invert
from ..types import *
from ..ast import *
from ..egglog import *
from ..spec import *

def negate_spec(x: Bool, ctx) -> Bool:
    return ~x


@Primitive(name="negate", spec=negate_spec)
def negate(x: Node) -> Node:
    return basic_invert(x, out=Bool())
