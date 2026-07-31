from zolotone import *
from .encode_Float32 import *
from .common import *

N = 4
Wf = 30


def dot_product_spec(a0, a1, a2, a3,
                     b0, b1, b2, b3, ctx):
    A = (a0, a1, a2, a3)
    B = (b0, b1, b2, b3)

    product_is_inf = [A[i].is_inf | B[i].is_inf for i in range(N)]
    product_is_negative = [A[i].sign.ne(B[i].sign) for i in range(N)]
    has_positive_inf = ormap(*[product_is_inf[i] & (~product_is_negative[i]) for i in range(N)])
    has_negative_inf = ormap(*[product_is_inf[i] & product_is_negative[i] for i in range(N)])

    invalid_product = ormap(*[(A[i].is_inf & B[i].is_zero) | (B[i].is_inf & A[i].is_zero) for i in range(N)])
    any_input_is_nan = ormap(*[value.is_nan for value in (*A, *B)])
    nan_case = (
        any_input_is_nan
        | invalid_product
        | (has_positive_inf & has_negative_inf)
    )
    negative_inf_case = has_negative_inf & (~nan_case)
    positive_inf_case = has_positive_inf & (~nan_case)
    finite_case = andmap(*[value.is_finite for value in (*A, *B)])

    finite_value = sum([A[i].value * B[i].value for i in range(N)], ctx.zero())
    finite_result = fp32.encode(finite_value, ctx)

    return Cases(
        case(nan_case, fp32.nan(ctx)),
        case(negative_inf_case, fp32.ninf(ctx)),
        case(positive_inf_case, fp32.inf(ctx)),
        case(finite_case, finite_result),
        ctx=ctx,
    )

@Composite(name="bf16x8_dot_fp32_conventional", spec=dot_product_spec)
def bf16x8_dot_fp32_conventional(a0: Node, a1: Node, a2: Node, a3: Node,
                 b0: Node, b1: Node, b2: Node, b3: Node) -> Node:
    A = tuple(bf16_decode(value) for value in (a0, a1, a2, a3))
    B = tuple(bf16_decode(value) for value in (b0, b1, b2, b3))

    ############ SPECIAL VALUES ########

    S_p = [bit_xor(A[i].sign, B[i].sign) for i in range(N)]
    product_is_inf = [bit_or(A[i].is_inf, B[i].is_inf) for i in range(N)]

    has_positive_inf = product_is_inf[0]
    has_negative_inf = bit_and(product_is_inf[0], S_p[0])
    has_positive_inf = bit_and(has_positive_inf, bit_neg(S_p[0]))
    invalid_product = bit_or(
        bit_and(A[0].is_inf, B[0].is_zero),
        bit_and(B[0].is_inf, A[0].is_zero),
    )
    any_input_is_nan = bit_or(A[0].is_nan, B[0].is_nan)
    
    for i in range(1, N):
        has_positive_inf = bit_or(
            has_positive_inf,
            bit_and(product_is_inf[i], bit_neg(S_p[i])),
        )
        has_negative_inf = bit_or(
            has_negative_inf,
            bit_and(product_is_inf[i], S_p[i]),
        )
        invalid_product = bit_or(
            invalid_product,
            bit_or(
                bit_and(A[i].is_inf, B[i].is_zero),
                bit_and(B[i].is_inf, A[i].is_zero),
            ),
        )
        any_input_is_nan = bit_or(
            any_input_is_nan,
            bit_or(A[i].is_nan, B[i].is_nan),
        )

    opposing_infinities = bit_and(has_positive_inf, has_negative_inf)
    encode_nan = bit_or(
        any_input_is_nan,
        bit_or(invalid_product, opposing_infinities),
    )
    not_encode_nan = bit_neg(encode_nan)
    encode_ninf = bit_and(not_encode_nan, has_negative_inf)
    encode_pinf = bit_and(not_encode_nan, has_positive_inf)
    
    ############ CONSTANTS #############
    
    bf16_bias = Const(Q.from_int(BFloat16.exponent_bias))
    subnormal_exponent = Const(
        UQ(
            1,
            A[0].exponent.node_type.int_bits,
            A[0].exponent.node_type.frac_bits,
        )
    )
    
    ############ EXPONENTS #############

    # Subnormals store exponent zero but behave as exponent 1-bias.
    E_a, E_b = [0] * N, [0] * N
    for i in range(N):
        E_a[i] = if_then_else(A[i].is_sub, subnormal_exponent, A[i].exponent)
        E_b[i] = if_then_else(B[i].is_sub, subnormal_exponent, B[i].exponent)

    # Step 1. Exponents add. Each E_p is shifted by bias twice!
    E_p = [uq_add(E_a[i], E_b[i]) for i in range(N)]

    # A zero product has no meaningful exponent and must not control
    # alignment of the nonzero products.
    zero_product_exponent = Const(
        UQ(
            0,
            E_p[0].node_type.int_bits,
            E_p[0].node_type.frac_bits,
        )
    )
    # This alignment does not affect verification - it is only affective for implementation
    E_p_for_alignment = [
        if_then_else(
            bit_or(A[i].is_zero, B[i].is_zero),
            zero_product_exponent,
            E_p[i],
        )
        for i in range(N)
    ]
    
    # Step 2. Calculate maximum exponent
    E_m = uq_max(
        uq_max(E_p_for_alignment[0], E_p_for_alignment[1]),
        uq_max(E_p_for_alignment[2], E_p_for_alignment[3]),
    )
    
    # Step 3. Calculate global shifts
    Sh_p = [uq_sub(E_m, E_p_for_alignment[i]) for i in range(N)]
    
    ############ MANTISSAS #############
    
    # Step 1. Convert mantissas to UQ1.7. Only normal values have an
    # implicit leading bit; subnormals and zero use the stored fraction.
    M_a, M_b = [0] * N, [0] * N
    for i in range(N):
        M_a[i] = if_then_else(
            A[i].is_norm,
            add_implicit_bit(integer_to_fraction(A[i].mantissa)),
            uq_resize(
                integer_to_fraction(A[i].mantissa),
                1,
                BFloat16.mantissa_bits,
            ),
        )
        M_b[i] = if_then_else(
            B[i].is_norm,
            add_implicit_bit(integer_to_fraction(B[i].mantissa)),
            uq_resize(
                integer_to_fraction(B[i].mantissa),
                1,
                BFloat16.mantissa_bits,
            ),
        )
    
    # Step 2. Multiply mantissas
    M_p = [uq_mul(M_a[i], M_b[i]) for i in range(N)]
    
    # Step 3. Shift mantissas
    # Make room for the right shift first, accuracy requirement is Wf
    M_p_resized = [uq_resize(M_p[i], 2, Wf - 2) for i in range(N)]
    M_p_shifted = [uq_rshift_jam(M_p_resized[i], Sh_p[i]) for i in range(N)]
    
    # Step 4. Adjust sign for mantissas using xor operation
    M_p_q = [uq_to_q(M_p_shifted[i]) for i in range(N)]
    
    M_p_q = [q_add_sign(M_p_q[i], S_p[i]) for i in range(N)]
    
    # Step 5. Adder tree
    M_sum = q_add(
        q_add(M_p_q[0], M_p_q[1]),
        q_add(M_p_q[2], M_p_q[3]),
    )

    # Paranoid check #2
    with context() as ctx:
        lhs = ctx.spec_of(M_sum)
        rhs = (
            ctx.spec_of(M_p_q[0])
            + ctx.spec_of(M_p_q[1])
            + ctx.spec_of(M_p_q[2])
            + ctx.spec_of(M_p_q[3])
        )
        ctx.check(lhs.eq(rhs))
    
    ############ RESULT ################
    
    # Subtract bias that is left!
    E_m_q = uq_to_q(E_m)
    
    E_m_q_biased = q_sub(E_m_q, bf16_bias)

    sign_bit = q_sign_bit(M_sum)
    M_sum_uq = q_to_uq(q_abs(M_sum))

    finite_result = fp32_encode(sign_bit, E_m_q_biased, M_sum_uq)

    return if_then_else(
        encode_nan,
        Const(Float32.NaN()),
        if_then_else(
            encode_ninf,
            Const(Float32.nInf()),
            if_then_else(
                encode_pinf,
                Const(Float32.Inf()),
                finite_result,
            ),
        ),
    )


if __name__ == '__main__':
    from time import time
    from pprint import pprint
    # Compile design
    a = [
        Var(name="a_0", sign=BFloat16T()),
        Var(name="a_1", sign=BFloat16T()),
        Var(name="a_2", sign=BFloat16T()),
        Var(name="a_3", sign=BFloat16T()),
    ]
    
    b = [
        Var(name="b_0", sign=BFloat16T()),
        Var(name="b_1", sign=BFloat16T()),
        Var(name="b_2", sign=BFloat16T()),
        Var(name="b_3", sign=BFloat16T()),
    ]
    
    design = bf16x8_dot_fp32_conventional(*a, *b)
    print(design)
    design.print_tree(depth=1)
    report = design.check_spec()
    pprint(report)
    with open("examples/c_models/bf16x8_dot_fp32_conventional.hpp", "w") as file:
        file.write(design.to_cpp())
