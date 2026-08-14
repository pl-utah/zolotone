from zolotone import *


def spec_e5m2fnuz_to_fp32(x: e5m2fnuz, ctx):
    return Cases(
        case(x.is_nan, fp32.nan(ctx)),
        case(x.is_finite, fp32.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="e5m2fnuz_to_fp32", spec=spec_e5m2fnuz_to_fp32)
def e5m2fnuz_to_fp32(x: Node) -> Node:
    X = e5m2fnuz_decode(x)

    significand = effective_significand(X)
    significand = uq_resize(
        significand,
        1,
        max(E5M2FNUZ.mantissa_bits, Float32.mantissa_bits),
    )

    source_exponent = effective_exponent(X)
    target_exponent = q_add(
        uq_to_q(source_exponent),
        Const(Q.from_int(Float32.exponent_bias - E5M2FNUZ.exponent_bias)),
    )

    result = fp32_encode(X.sign, target_exponent, significand)

    result = if_then_else(
        X.is_nan,
        Const(Float32.NaN()),
        result,
    )
    return result


if __name__ == "__main__":
    cast = e5m2fnuz_to_fp32(
        Var(name="x", sign=E5M2FNUZT()),
    )

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/e5m2fnuz_to_fp32_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/e5m2fnuz_to_fp32_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
