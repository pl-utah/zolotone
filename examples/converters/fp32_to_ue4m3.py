from zolotone import *


def spec_fp32_to_ue4m3(x: fp32, ctx):
    return Cases(
        case(x.is_nan, ue4m3.nan(ctx)),
        case(x.is_inf, ue4m3.encode(ctx.real_val(448), ctx)),
        case(x.is_finite, ue4m3.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="fp32_to_ue4m3", spec=spec_fp32_to_ue4m3)
def fp32_to_ue4m3(x: Node) -> Node:
    X = fp32_decode(x)

    significand = effective_significand(X)
    significand = uq_resize(
        significand,
        1,
        max(Float32.mantissa_bits, UE4M3.mantissa_bits),
    )

    source_exponent = effective_exponent(X)
    target_exponent = q_add(
        uq_to_q(source_exponent),
        Const(Q.from_int(UE4M3.exponent_bias - Float32.exponent_bias)),
    )

    result = ue4m3_encode(target_exponent, significand)
    result = if_then_else(
        X.is_zero,
        Const(UE4M3().Zero()),
        result,
    )
    result = if_then_else(
        X.is_inf,
        Const(
            UE4M3().from_fields(
                exponent=UE4M3.max_finite_code,
                mantissa=UE4M3.max_finite_mantissa,
            )
        ),
        result,
    )
    return if_then_else(
        X.is_nan,
        Const(UE4M3().NaN()),
        result,
    )


if __name__ == "__main__":
    cast = fp32_to_ue4m3(Var(name="x", dtype=Float32()))

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/fp32_to_ue4m3_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/fp32_to_ue4m3_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
