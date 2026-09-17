from .basics import basic_equal, basic_invert
from ..types import *
from ..ast import *
from ..egglog import *
from ..spec import *

def negate_spec(x: Bool, ctx) -> Bool:
    return ~x

@Primitive(name="negate", spec=negate_spec)
def negate(x: Node) -> Node:
    return basic_invert(x, out=Bool())


def bool_eq_spec(x: Bool, y: Bool, ctx) -> Bool:
    return x.eq(y)

@Primitive(name="bool_eq", spec=bool_eq_spec)
def bool_eq(x: Node, y: Node) -> Node:
    return basic_equal(x, y, out=Bool())
