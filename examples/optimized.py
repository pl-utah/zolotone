from zolotone import *
from .encode_Float32 import *
from .CSA import CSA_tree4
from .common import *
from .max_exponent import *

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


def optimized_spec(a0, a1, a2, a3,
                   b0, b1, b2, b3, ctx):
    return fp32.encode(a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3, ctx)

@Composite(name="Optimized", spec=optimized_spec)
def Optimized(a0: Node, a1: Node, a2: Node, a3: Node,
              b0: Node, b1: Node, b2: Node, b3: Node) -> Node:
    ############## INPUT ###############
    
    S_a, M_a, E_a = [0] * N, [0] * N, [0] * N
    S_b, M_b, E_b = [0] * N, [0] * N, [0] * N
    
    E_lead, E_trail = [0] * N, [0] * N
    
    for idx, value in enumerate((a0, a1, a2, a3)):
        decoded = bf16_decode(value)
        S_a[idx], E_a[idx], M_a[idx] = decoded[0], decoded[1], decoded[2]

    for idx, value in enumerate((b0, b1, b2, b3)):
        decoded = bf16_decode(value)
        S_b[idx], E_b[idx], M_b[idx] = decoded[0], decoded[1], decoded[2]
    
    ############ CONSTANTS #############
    
    bf16_bias = Const(Q.from_int(BFloat16.exponent_bias))
    
    ############ EXPONENTS #############
    
    # Step 1. Exponents add. Each E_p is shifted by bias twice!
    E_p = [uq_add(E_a[i], E_b[i]) for i in range(N)]
    
    for i in range(N):
        E_trail[i], E_lead[i] = uq_split(E_p[i], s)
    
    # Step 2. Estimate local shifts
    L_shifts = [_est_local_shift(E_trail[i]) for i in range(N)]
    
    # Step 4. Take max exponent
    E_m = OPTIMIZED_MAX_EXP4(*E_lead)
    
    # Step 5. Calculate global shifts as {(max_exp - exp) * 2**s}
    G_shifts = [_est_global_shift(E_m, E_lead[i]) for i in range(N)]
    
    ############# MANTISSAS ############
    
    # Step 1. Convert mantissas to UQ1.7
    M_a = [add_implicit_bit(integer_to_fraction(M_a[i])) for i in range(N)]
    M_b = [add_implicit_bit(integer_to_fraction(M_b[i])) for i in range(N)]
    
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
    S_p = [bit_xor(S_a[i], S_b[i]) for i in range(N)]
    
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

    return fp32_encode(sign_bit, E_m, M_sum_uq)


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
    
    design = Optimized(*a, *b)
    print(design)
    design.print_tree(depth=1)
    report = design.check_spec()
    pprint(report)
    with open("examples/optimized.hpp", "w") as file:
        file.write(design.to_cpp())
