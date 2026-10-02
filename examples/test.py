from zolotone import *

def spec(x: UQ(1, 0), ctx) -> UQ:
    ctx.assume(x >= ctx.real_val(0.5))
    denominator = x - ctx.real_val(0.75)
    return x + denominator

generated = Autogenerate(name=spec.__name__, spec=spec)
print(generated.print_tree(depth=1))
print(generated.to_cpp())