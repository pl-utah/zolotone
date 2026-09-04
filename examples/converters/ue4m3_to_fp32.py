from zolotone import *


def spec_ue4m3_to_fp32(x: ue4m3, ctx):
    return Cases(
        case(x.is_nan, fp32.nan(ctx)),
        case(x.is_finite, fp32.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="ue4m3_to_fp32", spec=spec_ue4m3_to_fp32)
def ue4m3_to_fp32(x: Node) -> Node:
    X = ue4m3_decode(x)

    significand = effective_significand(X)
    significand = uq_resize(
        significand,
        1,
        max(UE4M3.mantissa_bits, Float32.mantissa_bits),
    )

    source_exponent = effective_exponent(X)
    target_exponent = q_add(
        uq_to_q(source_exponent),
        Const(Q.from_int(Float32.exponent_bias - UE4M3.exponent_bias)),
    )

    result = fp32_encode(
        Const(UQ(1, 0).value(0)),
        target_exponent,
        significand,
    )
    return if_then_else(
        X.is_nan,
        Const(Float32().NaN()),
        result,
    )


if __name__ == "__main__":
    cast = ue4m3_to_fp32(Var(name="x", dtype=UE4M3()))

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/ue4m3_to_fp32_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/ue4m3_to_fp32_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
