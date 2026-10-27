from zolotone import *


def spec(x: UQ(1, 3), ctx) -> UQ:
    ctx.assume(x < ctx.real_val(1))
    return x + ctx.one()

generated = Autogenerate(name=spec.__name__, spec=spec)
print(generated.print_tree(depth=1))
print(generated.to_cpp())
with open("examples/c_models/test.cpp", "w") as file:
    file.write(generated.to_cpp())
