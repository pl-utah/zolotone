from zolotone import *

############## HELPERS #################
def _aligner(a: Node, b: Node):
    return q_aligner(a, b, max, max)

def _exact_xor(a: Node, b: Node):
    a_, b_ = _aligner(a, b)
    return basic_xor(a_, b_, a_.copy())

def _exact_and(a: Node, b: Node):
    a_, b_ = _aligner(a, b)
    return basic_and(a_, b_, a_.copy())

def _exact_or(a: Node, b: Node):
    a_, b_ = _aligner(a, b)
    return basic_or(a_, b_, a_.copy())


def csa_spec(x, y, z, ctx):
    carry = ctx.fresh_real("carry")
    sum_ = ctx.fresh_real("sum")
    ctx.assume((x + y + z).eq(carry + sum_))
    return sum_, carry

@Primitive(name="CSA", spec=csa_spec)
def CSA(x: Node, y: Node, z: Node) -> Node:
    sum_  = _exact_xor(_exact_xor(x, y), z)
    carry = _exact_or(_exact_or(_exact_and(x, y), _exact_and(x, z)), _exact_and(y, z))
    one = Const(UQ.from_int(1))
    carry = q_sign_extend(carry, 1)
    carry = q_lshift(carry, one)
    return make_Tuple(sum_, carry)


def CSA_tree4_spec(m0, m1, m2, m3, ctx):
        return m0 + m1 + m2 + m3

@Composite(name="CSA_tree4", spec=CSA_tree4_spec)
def CSA_tree4(m0: Node, m1: Node, m2: Node, m3: Node) -> Node:
    s1, c1 = CSA(m0, m1, m2)
    s2, c2 = CSA(m3, s1, c1)
    return q_add(s2, c2)


if __name__ == '__main__':
    from pprint import pprint
    # Compile design
    args = [
        Var(name="a_0", dtype=Q(3, 4)),
        Var(name="a_1", dtype=Q(8, 3)),
        Var(name="a_2", dtype=Q(5, 0)),
        Var(name="a_3", dtype=Q(1, 5)),
    ]
    
    design = CSA_tree4(*args)
    print(design)
    design.print_tree(depth=1)
    report = design.check_spec()
    pprint(report)
    with open("examples/c_models/csa.hpp", "w") as file:
        file.write(design.to_cpp())
