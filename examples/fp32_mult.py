from zolotone import *
from .common import *
from .encode_Float32 import *

def spec_fp32_mult(x: fp32, y: fp32, ctx):
    invalid = (x.is_inf & y.is_zero) | (y.is_inf & x.is_zero)
    nan_case = x.is_nan | y.is_nan | invalid
    
    negative_sign = x.sign.ne(y.sign)
    inf_case = (x.is_inf | y.is_inf) & (~nan_case)
    neg_inf_case = inf_case & negative_sign
    pos_inf_case = inf_case & (~negative_sign)
    zero_case = (
        (x.is_zero | y.is_zero)
        & (~nan_case)
        & (~inf_case)
    )
    neg_zero_case = zero_case & negative_sign
    
    return Cases(
        case(nan_case, fp32.nan()),
        case(neg_inf_case, fp32.ninf()),
        case(pos_inf_case, fp32.inf()),
        case(neg_zero_case, fp32.nzero()),
        default(fp32.encode(x.value * y.value, ctx)),
    )


@Composite(name="fp32_mult", spec=spec_fp32_mult)
def fp32_mult(x: Node, y: Node) -> Node:
    x_s, x_e, x_m, x_norm, x_sub, x_zero, x_inf, x_nan = fp32_decode(x)
    y_s, y_e, y_m, y_norm, y_sub, y_zero, y_inf, y_nan = fp32_decode(y)
    
    sign_bit = bit_xor(x_s, y_s)
    
    # IEEE-754 invalid cases for multiplication are NaN operands and 0 * inf.
    inf_times_zero = bit_or(
        bit_and(x_inf, y_zero),
        bit_and(y_inf, x_zero),
    )
    any_is_nan = bit_or(x_nan, y_nan)
    encode_nan = bit_or(inf_times_zero, any_is_nan)
    encode_inf = bit_and(bit_neg(encode_nan), bit_or(x_inf, y_inf))
    encode_ninf = bit_and(encode_inf, sign_bit)
    encode_pinf = bit_and(encode_inf, bit_neg(sign_bit))
    encode_nzero = bit_and(
        bit_and(
            bit_neg(encode_nan),
            bit_neg(encode_inf),
        ),
        bit_and(bit_or(x_zero, y_zero), sign_bit),
    )
    
    # UQ<23, 0> -> UQ<0, 23>
    x_m_fraction = integer_to_fraction(x_m)
    y_m_fraction = integer_to_fraction(y_m)
    
    # UQ<1, 23>
    x_m_formatted = if_then_else(x_norm, add_implicit_bit(x_m_fraction), uq_resize(x_m_fraction, 1, Float32.mantissa_bits))
    y_m_formatted = if_then_else(y_norm, add_implicit_bit(y_m_fraction), uq_resize(y_m_fraction, 1, Float32.mantissa_bits))
    
    # Subnormals have exponent field 0 but effective exponent 1-bias.
    effective_subnormal_e = Const(UQ(1, x_e.node_type.int_bits, x_e.node_type.frac_bits))
    x_effective_e = if_then_else(x_sub, effective_subnormal_e, x_e)
    y_effective_e = if_then_else(y_sub, effective_subnormal_e, y_e)
    
    # Keep the full 24x24-bit significand product exact and let fp32_encode
    # handle normalization, subnormal shifting, and final IEEE rounding.
    m_prod = uq_mul(x_m_formatted, y_m_formatted)
    e_prod = q_sub(
        q_add(uq_to_q(x_effective_e), uq_to_q(y_effective_e)),
        Const(Q.from_int(Float32.exponent_bias)),
    )
    
    finite_result = fp32_encode(
        sign_bit,               # sign: UQ
        e_prod,                 # exponent: Q, biased
        m_prod,                 # mantissa: exact UQ product
    )
    return if_then_else(
        encode_nan,
        Const(Float32.NaN()),
        if_then_else(
            encode_ninf,
            Const(Float32.nInf()),
            if_then_else(
                encode_pinf,
                Const(Float32.Inf()),
                if_then_else(
                    encode_nzero,
                    Const(Float32.nZero()),
                    finite_result,
                ),
            ),
        ),
    )


if __name__ == '__main__':
    from pprint import pprint
    multiplier = fp32_mult(
        Var(name="a", sign=Float32T()),
        Var(name="b", sign=Float32T()),
    )
    multiplier.check_spec()
    # pprint(multiplier.check_spec())
    with open("examples/multiplier_jit.hpp", "w") as file:
        file.write(multiplier.to_cpp(jittable=True))
    
    with open("examples/multiplier_no_jit.hpp", "w") as file:
        file.write(multiplier.to_cpp(jittable=False))
