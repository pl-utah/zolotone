from .basics import basic_and, basic_equal, basic_invert, basic_or
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


def bool_and_spec(x: Bool, y: Bool, ctx) -> Bool:
    return x & y


@Primitive(name="bool_and", spec=bool_and_spec, c_inline=True)
def bool_and(x: Node, y: Node) -> Node:
    return basic_and(x, y, out=Bool())


def bool_or_spec(x: Bool, y: Bool, ctx) -> Bool:
    return x | y


@Primitive(name="bool_or", spec=bool_or_spec, c_inline=True)
def bool_or(x: Node, y: Node) -> Node:
    return basic_or(x, y, out=Bool())
