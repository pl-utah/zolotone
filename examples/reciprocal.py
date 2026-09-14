from zolotone import *


# n bit unsigned x
# 2/1.x = 2^(n+1)/(2^n+x) in (1,2]
# 2^(n+1)/1.x = 2^(2*n+1)/(2^n+x) in (2^n,2^(n+1)]
def python_reciprocal(x, n, round_up) -> int:
    a = 1 << (2*n + 1)
    b = (1 << n) + x
    r = a % b
    l = a // b
    u = l + 1 if r > 0 else l
    return u if round_up else l


def _reciprocal_bound(x: Node, round_up: Node) -> Node:
    if x.dtype.int_bits != 0:
        raise ValueError(
            "reciprocal expects UQ<0,n>, whose bits represent the fractional "
            "part of 1.x"
        )

    n = x.dtype.frac_bits
    a = 1 << (2 * n + 1)
    implicit_one = 1 << n

    def sign(x: UQ, round_up: Bool) -> UQ:
        return UQ(1, n + 1)

    def impl(raw_x: int, round_up: int) -> int:
        return python_reciprocal(raw_x, n, round_up)

    def c_lowering(args: list[str], jittable: bool) -> str:
        del jittable
        b = f"({implicit_one} + {args[0]})"
        lower = f"({a} / {b})"
        has_remainder = f"(({a} % {b}) != 0)"
        return f"({lower} + (({args[1]} != 0) && {has_remainder}))"

    return Op(
        impl=impl,
        sign=sign,
        c_lowering=c_lowering,
        args=[x, round_up],
        name="reciprocal",
    )


def reciprocal_value_spec(x: UQ(0,5), round_up: Bool, ctx) -> UQ(1,6):
    return (ctx.one() + x) ** (-ctx.one())

@Primitive(name="reciprocal", spec=reciprocal_value_spec)
def reciprocal(x: Node, round_up: Node) -> Node:
    return _reciprocal_bound(x, round_up)


if __name__ == "__main__":
    x = Var(name="x", dtype=UQ(0, 5))
    round_up = Var(name="round_up", dtype=Bool())
    design = reciprocal(x, round_up)

    with open("examples/c_models/reciprocal.hpp", "w") as file:
        file.write(design.to_cpp())


