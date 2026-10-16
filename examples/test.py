from zolotone import *


def spec(x: Q(3, 2), ctx) -> Q(3, 1):
    return z ** ctx.real_val(-1)


generated = Autogenerate(name=spec.__name__, spec=spec)
print(generated.print_tree(depth=1))
print(generated.to_cpp())