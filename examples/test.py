from zolotone import *



def spec(x: UQ(2, 3), ctx) -> UQ(3, 2):
    result = ctx.fr(x, frac_bits=1)
    ctx.check(result >= ctx.real_val(0))
    return result

generated = Autogenerate(name=spec.__name__, spec=spec)
print(generated.print_tree(depth=1))
print(generated.to_cpp())
with open("examples/c_models/test.cpp", "w") as file:
    file.write(generated.to_cpp())
