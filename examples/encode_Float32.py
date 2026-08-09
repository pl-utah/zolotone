from zolotone import *


def fp32_encodings_spec(m, e, ctx):
    return m * ctx.two() ** ctx.real_val(Float32.mantissa_bits), e


@Primitive(name="fp32_encodings", spec=fp32_encodings_spec)
def fp32_encodings(m_rounded_uq: Node, e_rounded_uq: Node):
    final_e_uq_wide = uq_min(
        e_rounded_uq,
        Const(UQ.from_int(Float32.inf_code)),
    )
    final_e_uq = basic_identity(
        x=final_e_uq_wide,
        out=Const(UQ.from_int(Float32.inf_code)),
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
    return make_Tuple(fraction_to_integer(final_m_uq), final_e_uq)


def fp32_encode_spec(s, e, m, ctx):
    sign = sign_multiplier(ctx, s)
    finite_value = (
        sign
        * m
        * ctx.two() ** (e - ctx.real_val(Float32.exponent_bias))
    )
    return fp32.encode(finite_value, ctx)


@Composite(name="fp32_encode", spec=fp32_encode_spec)
def fp32_encode(s_uq: Node, e_q: Node, m_uq: Node) -> Primitive:
    assert e_q.node_type.frac_bits == 0

    encode_exact_zero = uq_is_zero(m_uq)
    normalized_m_uq, normalized_e_q = normalize_to_1_xxx(m_uq, e_q)
    shifted_m_uq, shifted_e_uq = shift_if_subnormal(
        normalized_m_uq,
        normalized_e_q,
        subnormal_extra_bits=3,
    )
    m_rounded_uq, e_rounded_uq = round_mantissa(
        drop_implicit_bit(shifted_m_uq),
        shifted_e_uq,
    )
    final_m_uq, final_e_uq = fp32_encodings(m_rounded_uq, e_rounded_uq)
    packed_fp32 = fp32_pack(s_uq, final_e_uq, final_m_uq)
    return if_then_else(
        encode_exact_zero,
        Const(Float32.Zero()),
        packed_fp32,
    )


if __name__ == "__main__":
    from pprint import pprint

    m = Const(UQ.from_float(4.02923583984375, 5, 28))
    e = Const(Q.from_float(-25.0, 11, 0))
    s = Const(UQ(1, 1, 0))
    design = fp32_encode(s, e, m)
    design.print_tree(depth=1)
    pprint(design.check_spec())
    print(design.evaluate())
