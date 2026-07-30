from zolotone import *


def spec_bf16_to_fp32(x: bf16, ctx):
    return Cases(
        case(x.is_nan, fp32.nan(ctx)),
        case(x.is_ninf, fp32.ninf(ctx)),
        case(x.is_pinf, fp32.inf(ctx)),
        case(x.is_nzero, fp32.nzero(ctx)),
        case(x.is_finite, fp32.encode(x.value, ctx)),
        ctx=ctx,
    )


@Composite(name="bf16_to_fp32", spec=spec_bf16_to_fp32)
def bf16_to_fp32(x: Node) -> Node:
    X = bf16_decode(x)

    # BF16 and FP32 use the same exponent width and bias. Widening is exact:
    # retain sign/exponent and append 16 zeros to the seven-bit fraction.
    fp32_mantissa = uq_resize(X.mantissa, Float32.mantissa_bits, 0)
    fp32_mantissa = uq_lshift(fp32_mantissa, Const(UQ.from_int(Float32.mantissa_bits - BFloat16.mantissa_bits)))
    return fp32_pack(X.sign, X.exponent, fp32_mantissa)


if __name__ == "__main__":
    cast = bf16_to_fp32(Var(name="x", sign=BFloat16T()))

    cast.check_determinism()
    cast.check_spec()

    with open("examples/c_models/bf16_to_fp32_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=True))

    with open("examples/c_models/bf16_to_fp32_no_jit.hpp", "w") as file:
        file.write(cast.to_cpp(jittable=False))
