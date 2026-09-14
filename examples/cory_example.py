from zolotone import *

@Primitive(name="example", spec=lambda x, ctx: x)
def example(x: Node) -> Node:
    ######################### ARITHMETIC #################################
    # Here x is a signed fixed point
    # x.dtype.int_bits or x.dtype.frac_bits would give integer/fractional bits
    print("x:", x.dtype)

    one = Const(UQ.from_int(1)) # U<1,0>
    print("one:", one.dtype)
    one_as_signed_fixedpoint = uq_to_q(one) # Q<2,0>
    print("one_as_signed_fixedpoint:", one_as_signed_fixedpoint.dtype)
    # Multiplication in full precision, so, you do not worry about bit-widths
    mult1 = q_mul(one_as_signed_fixedpoint, x) # Q<5,4>
    print("mult1:", mult1.dtype)

    # this will take absolute value of x
    x_as_unsigned_fixp = q_to_uq(x)
    print("x_as_unsigned_fixp:", x_as_unsigned_fixp.dtype)

    adder = uq_add(x_as_unsigned_fixp, one)
    print("adder", adder.dtype)

    # trying to right shift by two bits
    shift_amount = 2
    shift = uq_rshift(adder, Const(UQ.from_int(shift_amount)))
    print("shift:", shift.dtype)

    # The right shift introduced two zero MSBs. Drop those packed bits without
    # changing the position or scale of any remaining bit.
    retained_msb = shift.dtype.total_bits() - shift_amount - 1
    shift_truncated = uq_select(shift, retained_msb, 0)
    print("shift_truncated:", shift_truncated.dtype)

    ############################### ARRAYS #################################
    # well, not like array, but rather tuples - you can essentially store any types inside

    # Tuple<UQ<1,0>, Q<2,0>, Q<3,4>, UQ<2,4>>
    tuple_ = make_Tuple(one, one_as_signed_fixedpoint, x, x_as_unsigned_fixp)
    print("\ntuple of 4 elems", tuple_.dtype)
    print("make a tuple by accessing another tuple:", make_Tuple(tuple_[1], tuple_[3]).dtype)

    print("make a tuple be accessing another tuple + some arithmetic:",
        make_Tuple(
            uq_add(tuple_[0], tuple_[3]),
            q_add(tuple_[1], tuple_[2]),
        ).dtype
    )

    sum_of_elems_in_tuple = q_add(
        uq_to_q(uq_add(tuple_[0], tuple_[3])),
        q_add(tuple_[1], tuple_[2]),
    )
    print("sum of tuple elements:", sum_of_elems_in_tuple.dtype)

    nested_tuples = make_Tuple(tuple_, tuple_)
    print("nested tuples:", nested_tuples.dtype)
    print("accessing nested_tuple:", nested_tuples[0][1].dtype)

    ################################ Bitwise operations #####################
    # Basic bitwise operations work directly on packed integer bits. Their
    # output descriptor is explicit, and overflow is masked to that width.
    bit_pattern = Const(UQ(4, 0).from_bits(0b1010))
    bit_mask = Const(UQ(4, 0).from_bits(0b1100))
    one_bit = Const(UQ.from_int(1))

    rshifted = basic_rshift(bit_pattern, one_bit, out=bit_pattern.dtype)
    lshifted = basic_lshift(bit_pattern, one_bit, out=bit_pattern.dtype)
    xored = basic_xor(bit_pattern, bit_mask, out=bit_pattern.dtype)
    ored = basic_or(bit_pattern, bit_mask, out=bit_pattern.dtype)
    anded = basic_and(bit_pattern, bit_mask, out=bit_pattern.dtype)

    def print_bitwise_value(name: str, node: Node) -> None:
        value = node.evaluate()
        print(
            f"{name:<14} "
            f"dtype={value.dtype} "
            f"decimal={value.to_dec():<2} "
            f"bits={value.to_bitstring()} "
            f"hex={value.to_hex()}"
        )

    print()
    print_bitwise_value("bit pattern:", bit_pattern)
    print_bitwise_value("bit mask:", bit_mask)
    print_bitwise_value("basic_rshift:", rshifted)
    print_bitwise_value("basic_lshift:", lshifted)
    print_bitwise_value("basic_xor:", xored)
    print_bitwise_value("basic_or:", ored)
    print_bitwise_value("basic_and:", anded)

    return sum_of_elems_in_tuple


if __name__ == '__main__':
    from pprint import pprint
    # Compile design
    arg = Var(name="x", dtype=Q(3, 4))
    design = example(arg)
    design.print_tree(depth=1)

    with open("examples/c_models/cory_example.hpp", "w") as file:
        file.write(design.to_cpp())
