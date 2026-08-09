from zolotone import *


def bf16_encodings_spec(m, e, ctx):
    return m * ctx.two() ** ctx.real_val(BFloat16.mantissa_bits), e


@Primitive(name="bf16_encodings", spec=bf16_encodings_spec)
def bf16_encodings(m_rounded_uq: Node, e_rounded_uq: Node):
    final_e_uq_wide = uq_min(
        e_rounded_uq,
        Const(UQ.from_int(BFloat16.inf_code)),
    )
    final_e_uq = basic_identity(
        x=final_e_uq_wide,
        out=Const(UQ.from_int(BFloat16.inf_code)),
    )
    is_inf = basic_and_reduce(
        final_e_uq,
        out=Const(UQ(0, 1, 0)),
    )
    final_m_uq = basic_mux_2_1(
        sel=is_inf,
        in0=m_rounded_uq,
        in1=Const(UQ(0, 1, 0)),
        out=m_rounded_uq.copy(),
    )

    final_m_uq = fraction_to_integer(final_m_uq)
    return make_Tuple(final_m_uq, final_e_uq)


def bf16_encode_spec(s, e, m, ctx):
    sign = sign_multiplier(ctx, s)
    finite_value = (
        sign
        * m
        * (
            ctx.two()
            ** (e - ctx.real_val(BFloat16.exponent_bias))
        )
    )
    return bf16.encode(finite_value, ctx)


@Composite(name="bf16_encode", spec=bf16_encode_spec)
def bf16_encode(s_uq: Node, e_q: Node, m_uq: Node) -> Node:
    assert e_q.node_type.frac_bits == 0

    encode_exact_zero = uq_is_zero(m_uq)

    normalized_m_uq, normalized_e_q = normalize_to_1_xxx(m_uq, e_q)
    shifted_m_uq, shifted_e_uq = shift_if_subnormal(
        normalized_m_uq,
        normalized_e_q,
        subnormal_extra_bits=3,
    )
    shifted_dropped_bit_m_uq = drop_implicit_bit(shifted_m_uq)

    m_rounded_uq, e_rounded_uq = round_mantissa(
        shifted_dropped_bit_m_uq,
        shifted_e_uq,
        target_bits=BFloat16.mantissa_bits,
    )

    final_m_uq, final_e_uq = bf16_encodings(
        m_rounded_uq,
        e_rounded_uq,
    )

    packed_bf16 = bf16_pack(s_uq, final_e_uq, final_m_uq)
    return if_then_else(
        encode_exact_zero,
        Const(BFloat16.Zero()),
        packed_bf16,
    )
