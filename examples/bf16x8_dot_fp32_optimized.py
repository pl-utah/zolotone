from zolotone import *
from .encode_Float32 import *
from .CSA import CSA_tree4
from .common import *
from .max_exponent import *

from functools import reduce
from operator import or_, and_

s = 2
N = 4
Wf = 30


def spec_est_global_shift(E_max, E_p, ctx):
    return (E_max - E_p) * (ctx.real_val(2) ** ctx.real_val(s))

@Primitive(name="_est_global_shift", spec=spec_est_global_shift)
def _est_global_shift(E_max: Node, E_p: Node) -> Node:
    out_int_bits = uq_add(uq_int_bits(E_max), Const(UQ.from_int(s)))
    out_frac_bits = Const(UQ.from_int(0))
    out = uq_alloc(out_int_bits, out_frac_bits)

    return basic_concat(uq_sub(E_max, E_p), Const(UQ(0, s, 0)), out)


def spec_est_local_shift(E_trail, ctx):
    two = ctx.real_val(2)
    one = ctx.real_val(1)
    return (two ** ctx.real_val(s)) - one - E_trail

@Primitive(name="_est_local_shift", spec=spec_est_local_shift)
def _est_local_shift(E_trail: Node) -> Node:
    return basic_invert(x=E_trail, out=E_trail.copy())


def spec_prepend_ones(x, ctx):
    two = ctx.real_val(2)
    one = ctx.real_val(1)
    real_s = ctx.real_val(s)
    return x * (two ** real_s) + (two ** real_s) - one

@Primitive(name="_prepend_ones", spec=spec_prepend_ones)
def _prepend_ones(x: Node) -> Node:
    out_int_bits = uq_add(uq_int_bits(x), Const(UQ.from_int(s)))
    out_frac_bits = uq_frac_bits(x)
    out = uq_alloc(out_int_bits, out_frac_bits)
    return basic_concat(x, Const(UQ.from_int((1 << s) - 1)), out)


def dot_product_spec(a0, a1, a2, a3,
                     b0, b1, b2, b3, ctx):
    A = (a0, a1, a2, a3)
    B = (b0, b1, b2, b3)

    product_is_inf = [A[i].is_inf | B[i].is_inf for i in range(N)]
    product_is_negative = [A[i].sign.ne(B[i].sign) for i in range(N)]
    has_positive_inf = reduce(or_, [product_is_inf[i] & (~product_is_negative[i]) for i in range(N)])
    has_negative_inf = reduce(or_, [product_is_inf[i] & product_is_negative[i] for i in range(N)])

    invalid_product = reduce(or_, [(A[i].is_inf & B[i].is_zero) | (B[i].is_inf & A[i].is_zero) for i in range(N)])
    any_input_is_nan = reduce(or_, [value.is_nan for value in (*A, *B)])
    nan_case = (
        any_input_is_nan
        | invalid_product
        | (has_positive_inf & has_negative_inf)
    )
    negative_inf_case = has_negative_inf & (~nan_case)
    positive_inf_case = has_positive_inf & (~nan_case)
    finite_case = reduce(and_, [value.is_finite for value in (*A, *B)])

    finite_value = sum([A[i].value * B[i].value for i in range(N)], ctx.real_val(0))
    finite_result = fp32.encode(finite_value, ctx)

    return Cases(
        case(nan_case, fp32.nan(ctx)),
        case(negative_inf_case, fp32.ninf(ctx)),
        case(positive_inf_case, fp32.inf(ctx)),
        case(finite_case, finite_result),
        ctx=ctx,
    )

@Composite(name="bf16x8_dot_fp32_optimized", spec=dot_product_spec)
def bf16x8_dot_fp32_optimized(a0: Node, a1: Node, a2: Node, a3: Node,
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

    E_lead, E_trail = [0] * N, [0] * N
    for i in range(N):
        E_trail[i], E_lead[i] = uq_split(E_p[i], s)
    
    # Step 2. Estimate local shifts
    L_shifts = [_est_local_shift(E_trail[i]) for i in range(N)]
    
    # Step 4. Take max exponent
    E_m = OPTIMIZED_MAX_EXP4(*E_lead)
    
    # Step 5. Calculate global shifts as {(max_exp - exp) * 2**s}
    G_shifts = [_est_global_shift(E_m, E_lead[i]) for i in range(N)]
    
    ############# MANTISSAS ############
    
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
    
    # Step 2. Multiply mantissas into UQ2.14
    M_p = [uq_mul(M_a[i], M_b[i]) for i in range(N)]
    
    # Step 3. Locally shift mantissas by the inverted last {s} bits of E_p
    # Make room for the right shift
    M_p = [uq_resize(M_p[i], 2, 14 + 2**s - 1) for i in range(N)]
    
    M_p = [uq_rshift(M_p[i], L_shifts[i]) for i in range(N)]
    
    # Step 4. Globally shift mantissas by G_shifts[i] amount
    # Make room for the right shift
    M_p = [uq_resize(M_p[i], 2, Wf - 2 + 2**s - 1) for i in range(N)]
    
    M_p = [uq_rshift(M_p[i], G_shifts[i]) for i in range(N)]
    
    with context() as ctx:
        for i in range(N):
            # M_p[i] * 2 ** (E_m * 2**s + 2**s - 1)
            lhs = ctx.spec_of(M_p[i]) * ctx.real_val(2) ** (
                ctx.spec_of(E_m) * ctx.real_val(2) ** ctx.real_val(s)
                + ctx.real_val(2) ** ctx.real_val(s)
                - ctx.real_val(1)
            )
            rhs = (
                ctx.spec_of(M_a[i])
                * ctx.spec_of(M_b[i])
                * ctx.real_val(2) ** ctx.spec_of(E_p[i])
            )
            ctx.check(lhs.eq(rhs))
    
    # Step 5. Adjust signs using xor operation
    M_p = [uq_to_q(M_p[i]) for i in range(N)]
    
    M_p = [q_add_sign(M_p[i], S_p[i]) for i in range(N)]
    
    # Step 6. Adder Tree
    M_sum = CSA_tree4(*M_p)
    
    ############# RESULT ###############
    # Append {s} 1s at the end of the max exponent for a normalization
    E_m = _prepend_ones(E_m)
    
    # Subtract bias since E_m is biased twice
    E_m = uq_to_q(E_m)
    
    E_m = q_sub(E_m, bf16_bias)

    sign_bit = q_sign_bit(M_sum)
    M_sum_uq = q_to_uq(q_abs(M_sum))

    finite_result = fp32_encode(sign_bit, E_m, M_sum_uq)

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
    
    design = bf16x8_dot_fp32_optimized(*a, *b)
    print(design)
    design.print_tree(depth=1)
    report = design.check_spec()
    pprint(report)
    with open("examples/c_models/bf16x8_dot_fp32_optimized.hpp", "w") as file:
        file.write(design.to_cpp())
