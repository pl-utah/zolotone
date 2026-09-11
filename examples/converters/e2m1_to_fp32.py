from zolotone import *


def spec_e2m1_to_fp32(x: e2m1, ctx):
    return Cases(
        case(x.is_nzero, fp32.nzero(ctx)),
        case(x.is_finite, fp32.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="e2m1_to_fp32", spec=spec_e2m1_to_fp32)
def e2m1_to_fp32(x: Node) -> Node:
    X = e2m1_decode(x)

    significand = effective_significand(X)
    significand = uq_resize(
        significand,
        1,
        max(E2M1.mantissa_bits, Float32.mantissa_bits),
    )

    source_exponent = effective_exponent(X)
    target_exponent = q_add(
        uq_to_q(source_exponent),
        Const(Q.from_int(Float32.exponent_bias - E2M1.exponent_bias)),
    )

    result = fp32_encode(X.sign, target_exponent, significand)

    encode_negative_zero = bit_and(X.is_zero, X.sign)
    result = if_then_else(
        encode_negative_zero,
        Const(Float32().nZero()),
        result,
    )
    return result


if __name__ == "__main__":
    cast = e2m1_to_fp32(
        Var(name="x", dtype=E2M1()),
    )

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/e2m1_to_fp32_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/e2m1_to_fp32_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
