from zolotone import *

def spec(x: Q(3, 2), ctx) -> UQ(3, 1):
    ctx.assume(x - ctx.one() >= ctx.zero())
    return x

generated = Autogenerate(name=spec.__name__, spec=spec)
print(generated.print_tree(depth=1))
print(generated.to_cpp())