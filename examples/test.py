from zolotone import *


def spec(x: UQ(4, 2), ctx) -> Q:
    z = ctx.fr(sin(x), frac_bits=4)
    ctx.assume(abs(z).ne(ctx.one()))
    return z

generated = Autogenerate(name=spec.__name__, spec=spec, only_LUT=False)
print(generated.print_tree(depth=1))
print(generated.to_cpp())
with open("examples/c_models/test.cpp", "w") as file:
    file.write(generated.to_cpp())
