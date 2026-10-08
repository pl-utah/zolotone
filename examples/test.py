from zolotone import *

def spec(x: Q(3, 0), ctx) -> Q:
    ctx.assume(x + ctx.one() >= ctx.zero())
    return x ** ctx.real_val(-1)

generated = Autogenerate(name=spec.__name__, spec=spec)
print(generated.print_tree(depth=1))
print(generated.to_cpp())