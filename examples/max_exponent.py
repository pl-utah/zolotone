from zolotone import *

def max_exp4_spec(e0, e1, e2, e3, ctx):
    return e0.max(e1).max(e2).max(e3)


@Primitive(name="OPTIMIZED_MAX_EXP4", spec=max_exp4_spec)
def OPTIMIZED_MAX_EXP4(e0: Node, e1: Node, e2: Node, e3: Node) -> Node:
    def bit_not(x: Node) -> Node:
        return basic_invert(x=x, out=Const(UQ(1, 0).value(0)))

    def bit_and(x: Node, y: Node) -> Node:
        return basic_and(x=x, y=y, out=Const(UQ(1, 0).value(0)))

    def bit_or(x: Node, y: Node) -> Node:
        return basic_or(x=x, y=y, out=Const(UQ(1, 0).value(0)))

    def concat(high: Node, low: Node) -> Node:
        return basic_concat(
            x=high,
            y=low,
            out=Const(UQ(high.dtype.int_bits + low.dtype.int_bits, 0).value(0)),
        )

    inputs = [e0, e1, e2, e3]

    # This uses statically known widths to build the comparison tree shape.
    int_bits = max(x.dtype.int_bits for x in inputs)
    frac_bits = max(x.dtype.frac_bits for x in inputs)
    assert frac_bits == 0, "not implemented yet"
    
    bit_width = int_bits + frac_bits

    zero_bit = Const(UQ(1, 0).value(0))

    prev_bits = [zero_bit] * len(inputs)
    smaller = [zero_bit] * len(inputs)
    max_prev = zero_bit
    bits = None

    for pos in range(bit_width - 1, -1, -1):
        curr_bits = [uq_select(exp, pos, pos) for exp in inputs]
        next_smaller = []
        candidates = zero_bit

        for i, (prev_bit, curr_bit, was_smaller) in enumerate(zip(prev_bits, curr_bits, smaller)):
            is_smaller = bit_or(bit_and(max_prev, bit_not(prev_bit)), was_smaller)
            next_smaller.append(is_smaller)
            
            candidate = bit_and(curr_bit, bit_not(is_smaller))
            candidates = concat(candidates, candidate)

        max_prev = basic_or_reduce(candidates, out=Const(UQ(1, 0).value(0)))  # or tree
        if bits is None:
            bits = max_prev
        else:
            bits = concat(bits, max_prev)
        prev_bits = curr_bits
        smaller = next_smaller
    return bits

if __name__ == '__main__':
    inputs = [Var(f"arg_{i}", dtype=UQ(10, 0)) for i in range(4)]
    design = OPTIMIZED_MAX_EXP4(*inputs)
    # design.print_tree(depth=1)
    with open("examples/c_models/max_exp.hpp", "w") as file:
        file.write(design.to_cpp())
    
