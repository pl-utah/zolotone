from zolotone import *


def spec_bf16_relu(x: bf16, ctx):
    negative_case = x.sign.eq(ctx.one())
    nonnegative_case = x.sign.eq(ctx.zero())

    return Cases(
        case(x.is_nan, bf16.nan(ctx)),
        case(negative_case, bf16.zero(ctx)),
        case(nonnegative_case, x),
        ctx=ctx,
    )


@Composite(name="bf16_relu", spec=spec_bf16_relu)
def bf16_relu(x: Node) -> Node:
    X = bf16_decode(x)

    return if_then_else(
        X.is_nan,
        Const(BFloat16().NaN()),
        if_then_else(
            X.sign,
            Const(BFloat16().Zero()),
            x,
        ),
    )


if __name__ == "__main__":
    relu = bf16_relu(Var(name="x", dtype=BFloat16()))

    relu.check_determinism()
    relu.check_spec()

    with open("examples/c_models/bf16_relu_jit.hpp", "w") as file:
        file.write(relu.to_cpp(jittable=True))

    with open("examples/c_models/bf16_relu_no_jit.hpp", "w") as file:
        file.write(relu.to_cpp(jittable=False))
