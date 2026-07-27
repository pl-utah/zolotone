from zolotone import *
from .encode_Float32 import *
from .common import *

N = 4
Wf = 30

def conventional_spec(a0, a1, a2, a3,
                      b0, b1, b2, b3, ctx):
    return fp32.encode(a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3, ctx)

@Composite(name="Conventional", spec=conventional_spec)
def Conventional(a0: Node, a1: Node, a2: Node, a3: Node,
                 b0: Node, b1: Node, b2: Node, b3: Node) -> Node:
    S_a, M_a, E_a = [0] * N, [0] * N, [0] * N
    S_b, M_b, E_b = [0] * N, [0] * N, [0] * N
    
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
    
    # Step 2. Calculate maximum exponent
    E_m = uq_max(uq_max(E_p[0], E_p[1]), uq_max(E_p[2], E_p[3]))
    
    # Step 3. Calculate global shifts
    Sh_p = [uq_sub(E_m, E_p[i]) for i in range(N)]
    
    ############ MANTISSAS #############
    
    # Step 1. Convert mantissas to UQ1.7
    M_a = [add_implicit_bit(integer_to_fraction(M_a[i])) for i in range(N)]
    M_b = [add_implicit_bit(integer_to_fraction(M_b[i])) for i in range(N)]
    
    # Step 2. Multiply mantissas
    M_p = [uq_mul(M_a[i], M_b[i]) for i in range(N)]
    
    # Step 3. Shift mantissas
    # Make room for the right shift first, accuracy requirement is Wf
    M_p_resized = [uq_resize(M_p[i], 2, Wf - 2) for i in range(N)]
    M_p_shifted = [uq_rshift(M_p_resized[i], Sh_p[i]) for i in range(N)]

    with context() as ctx:
        for i in range(N):
            lhs = ctx.spec_of(M_p_shifted[i]) * ctx.real_val(2) ** ctx.spec_of(E_m)
            rhs = ctx.spec_of(M_p[i]) * ctx.real_val(2) ** ctx.spec_of(E_p[i])
            ctx.check(lhs.eq(rhs))
    
    # Step 4. Adjust sign for mantissas using xor operation
    S_p = [bit_xor(S_a[i], S_b[i]) for i in range(N)]
    
    M_p_q = [uq_to_q(M_p_shifted[i]) for i in range(N)]
    
    M_p_q = [q_add_sign(M_p_q[i], S_p[i]) for i in range(N)]
    
    # Step 5. Adder tree
    M_sum = q_add(
        q_add(M_p_q[0], M_p_q[1]),
        q_add(M_p_q[2], M_p_q[3]),
    )
    
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
    
    return fp32_encode(sign_bit, E_m_q_biased, M_sum_uq)


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
    
    design = Conventional(*a, *b)
    print(design)
    design.print_tree(depth=1)
    report = design.check_spec()
    pprint(report)
    with open("examples/conventional.hpp", "w") as file:
        file.write(design.to_cpp())
