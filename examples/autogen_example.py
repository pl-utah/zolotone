from zolotone import *


# this should pass
def spec1(x: UQ(2,0), y: UQ(3,0), ctx) -> UQ:
    return x + y

# this should pass
def spec2(x: UQ(2,0), y: Q(3,0), ctx) -> Q:
    return x + y

# this should not pass
def spec3(x: UQ, y: Q(3,0), ctx) -> Q:
    return x + y

# this should error
def spec4(x: UQ(2,0), y: Q(3,0), ctx) -> UQ:
    return x + y

Autogenerate(name="spec1", spec=spec1)
Autogenerate(name="spec2", spec=spec2)
try:
    Autogenerate(name="spec3", spec=spec3)
except TypeError:
    pass
Autogenerate(name="spec4", spec=spec4)