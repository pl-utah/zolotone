from zolotone import *


# this should pass
def spec1(x: UQ(2,0), y: UQ(3,0), ctx) -> UQ:
    return x + y

# mixed-type lossless conversion
def spec2(x: UQ(2,0), y: Q(3,0), ctx) -> Q:
    return x + y

# this should not pass
def spec3(x: UQ, y: Q(3,0), ctx) -> Q:
    return x + y

# this should fail
def spec4(x: UQ(2,0), y: Q(3,0), ctx) -> UQ(3,0):
    return x + y

# this should pass
def spec5(x: UQ(2,0), ctx) -> UQ:
    return x + ctx.one()

# this should pass
def spec6(x: Q(2,0), ctx) -> Q(4,0):
    return x + ctx.one()

def spec7(x: Q(3,0), y: Bool(), ctx) -> Q(3,0):
    return If(y.eq(ctx.false()), x, ctx.one())

def spec8(x: Q(3,0), ctx) -> Q:
    y = ctx.fresh_real('x')
    return x + ctx.real_val(2) + y

def spec9(x: Q(3,0), ctx) -> Q:
    return x + ctx.real_val(2.5)

# conditional selection
def spec10(sel: Bool(), in1: UQ(2,0), in0: UQ(2,0), ctx) -> UQ(2,0):
    return If(sel, in1, in0)

# this should fail before search because the result can reach 4
def spec11(x: UQ(2,0), ctx) -> UQ(2,0):
    return x + ctx.one()

Autogenerate(name="spec1", spec=spec1).print_tree(depth=1)
Autogenerate(name="spec2", spec=spec2).print_tree(depth=1)
try:
    Autogenerate(name="spec3", spec=spec3)
except TypeError:
    pass
try:
    Autogenerate(name="spec4", spec=spec4).print_tree(depth=1)
except TypeError:
    pass
Autogenerate(name="spec5", spec=spec5).print_tree(depth=1)
Autogenerate(name="spec6", spec=spec6).print_tree(depth=1)
Autogenerate(name="spec7", spec=spec7).print_tree(depth=1)
try:
    Autogenerate(name="spec8", spec=spec8).print_tree(depth=1)
except TypeError:
    pass
try:
    Autogenerate(name="spec9", spec=spec9).print_tree(depth=1)
except TypeError:
    pass
Autogenerate(name="spec10", spec=spec10).print_tree(depth=1)
try:
    Autogenerate(name="spec11", spec=spec11).print_tree(depth=1)
except TypeError:
    pass
