from zolotone import *


def spec(x: UQ(3, 0), y: UQ(2, 0), ctx) -> UQ:
    ctx.assume(x >= ctx.real_val(0.5))
    denominator = x + y
    return ctx.rne((denominator ** ctx.real_val(-1)) * ctx.real_val(2), frac_bits=4)


generated = Autogenerate(name=spec.__name__, spec=spec)
print(generated.print_tree(depth=1))
print(generated.to_cpp())
with open("examples/c_models/test.cpp", "w") as file:
    file.write(generated.to_cpp())
