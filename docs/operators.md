### Type notation
- `Q<I,F>`: signed fixed-point with `I` integer and `F` fractional bits; `UQ<I,F>` unsigned.
- `Float16`, `BFloat16`, `Float32`: IEEE754 formats.
- `T`: arbitrary bitvector type, `Any`: unconstrained node chosen by caller.
- `x`: tuple: explicit output node supplied by caller (shape must match the result).
- `n<int>, s<int> etc.`: literal integers passed as Python args, not nodes.

### Public API

## Basic bitwise/arithmetic ops (`zolotone/numtypes/basics.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `basic_mux_2_1` | Op | `Any -> Any -> Any -> T -> T` | 2:1 mux. |
| `basic_add` | Op | `Any -> Any -> T -> T` | Aligned addition. |
| `basic_sub` | Op | `Any -> Any -> T -> T` | Aligned subtraction. |
| `basic_mul` | Op | `Any -> Any -> T -> T` | Aligned multiplication. |
| `basic_max` | Op | `Any -> Any -> T -> T` | Bitwise max. |
| `basic_min` | Op | `Any -> Any -> T -> T` | Bitwise min. |
| `basic_rshift` | Op | `Any -> Any -> T -> T` | Logical right shift. |
| `basic_lshift` | Op | `Any -> Any -> T -> T` | Logical left shift. |
| `basic_or` | Op | `Any -> Any -> T -> T` | Bitwise OR. |
| `basic_xor` | Op | `Any -> Any -> T -> T` | Bitwise XOR. |
| `basic_and` | Op | `Any -> Any -> T -> T` | Bitwise AND. |
| `basic_concat` | Op | `Any -> Any -> T -> T` | Concatenate bits. |
| `basic_less` | Op | `Any -> Any -> T -> T` | Bitwise less-than. |
| `basic_less_or_equal` | Op | `Any -> Any -> T -> T` | Bitwise less-or-equal. |
| `basic_greater` | Op | `Any -> Any -> T -> T` | Bitwise greater-than. |
| `basic_greater_or_equal` | Op | `Any -> Any -> T -> T` | Bitwise greater-or-equal. |
| `basic_equal` | Op | `Any -> Any -> T -> T` | Bitwise equality. |
| `basic_not_equal` | Op | `Any -> Any -> T -> T` | Bitwise inequality. |
| `basic_select` | Op | `Any -> int -> int -> T -> T` | Inclusive slice. |
| `basic_invert` | Op | `Any -> T -> T` | Bitwise NOT. |
| `basic_identity` | Op | `Any -> T -> T` | Identity/cast. |
| `basic_or_reduce` | Op | `Any -> T -> T` | OR-reduce across bits. |
| `basic_and_reduce` | Op | `Any -> T -> T` | AND-reduce across bits. |

## Unsigned fixed-point primitives (`zolotone/numtypes/UQ.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `uq_alloc` | Op | `int_bits<Any> -> frac_bits<Any> -> UQ<int_bits.val, frac_bits.val>` | Assemble UQ from integer and fractional bits. |
| `uq_aligner` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> int_aggr<int -> int -> int> -> frac_aggr<int -> int -> int> -> (UQ<int_aggr(I1,I2), frac_aggr(F1,F2)> x UQ<int_aggr(I1,I2), frac_aggr(F1,F2)>)` | Align two UQs to a common width. |
| `uq_zero_extend` | Primitive | `UQ<I,F> -> n<int> -> UQ<I+n,F>` | Pad high bits with `n` zeros. |
| `uq_add` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> UQ<max(I1,I2)+1, max(F1,F2)>` | Unsigned add with alignment and carry bit. |
| `uq_sub` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> UQ<max(I1,I2)+1, max(F1,F2)>` | Unsigned subtract with alignment and borrow bit. |
| `uq_max` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> UQ<max(I1,I2), max(F1,F2)>` | Unsigned max with alignment. |
| `uq_min` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> UQ<max(I1,I2), max(F1,F2)>` | Unsigned min with alignment. |
| `uq_mul` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> UQ<I1+I2, F1+F2>` | Unsigned multiply (full precision). |
| `uq_less` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> Bool<1>` | Unsigned less-than. |
| `uq_less_or_equal` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> Bool<1>` | Unsigned less-or-equal. |
| `uq_greater` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> Bool<1>` | Unsigned greater-than. |
| `uq_greater_or_equal` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> Bool<1>` | Unsigned greater-or-equal. |
| `uq_equal` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> Bool<1>` | Unsigned equality. |
| `uq_not_equal` | Primitive | `UQ<I1,F1> -> UQ<I2,F2> -> Bool<1>` | Unsigned inequality. |
| `uq_to_q` | Primitive | `UQ<I,F> -> Q<I+1,F>` | Convert to signed (adds sign bit). |
| `uq_rshift` | Primitive | `UQ<I,F> -> Any -> UQ<I,F>` | Logical right shift without resize. |
| `uq_lshift` | Primitive | `UQ<I,F> -> Any -> UQ<I,F>` | Logical left shift without resize. |
| `uq_select` | Primitive | `UQ<I,F> -> start<int> -> end<int> -> UQ<(start-end+1)-k, k>` where `k = max(0, min(start, F-1)-end+1)` | Bit slice with fractional portion preserved when slicing frac bits. |
| `uq_resize` | Primitive | `UQ<I,F> -> int_bits<int> -> frac_bits<int> -> UQ<int_bits, frac_bits>` | Resize/round toward zero (no truncation allowed). |

## Signed fixed-point primitives (`zolotone/numtypes/Q.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `q_alloc` | Op | `int_bits<Any> -> frac_bits<Any> -> Q<int_bits.val, frac_bits.val>` | Assemble Q from integer and fractional bits. |
| `q_aligner` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> int_aggr<int -> int -> int> -> frac_aggr<int -> int -> int> -> (Q<int_aggr(I1,I2), frac_aggr(F1,F2)> x Q<int_aggr(I1,I2), frac_aggr(F1,F2)>)` | Align two Qs to a common width. |
| `q_sign_bit` | Primitive | `Q<I,F> -> UQ<1,0>` | MSB of two's complement value. |
| `q_sign_extend` | Primitive | `Q<I,F> -> n<int> -> Q<I+n, F>` | Extend sign into high bits. |
| `q_neg` | Primitive | `Q<I,F> -> Q<I,F>` | Two's complement negate (special-case overflow at min). |
| `q_signs_xor` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> Bool<1>` | Sign mismatch (xor of sign bits). |
| `q_add` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> Q<max(I1,I2)+1, max(F1,F2)>` | Signed addition with alignment. |
| `q_sub` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> Q<max(I1,I2)+1, max(F1,F2)>` | Signed subtraction with alignment. |
| `q_mul` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> Q<I1+I2, F1+F2>` | Signed multiply (full precision). |
| `q_less` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> Bool<1>` | Signed less-than. |
| `q_less_or_equal` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> Bool<1>` | Signed less-or-equal. |
| `q_greater` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> Bool<1>` | Signed greater-than. |
| `q_greater_or_equal` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> Bool<1>` | Signed greater-or-equal. |
| `q_equal` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> Bool<1>` | Signed equality. |
| `q_not_equal` | Primitive | `Q<I1,F1> -> Q<I2,F2> -> Bool<1>` | Signed inequality. |
| `q_lshift` | Primitive | `Q<I,F> -> Any -> Q<I,F>` | Logical left shift without resize. |
| `q_rshift` | Primitive | `Q<I,F> -> Any -> Q<I,F>` | Arithmetic right shift without resize. |
| `q_to_uq` | Primitive | `Q<I,F> -> UQ<I-1, F>` | Drop sign bit (assumes non-negative). |
| `q_add_sign` | Primitive | `Q<I,F> -> UQ<1,0> -> Q<I,F>` | Apply sign bit to unsigned magnitude. |
| `q_abs` | Primitive | `Q<I,F> -> Q<I,F>` | Absolute value (safe at min). |

## BF16 helpers (`zolotone/numtypes/BFloat16.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `_bf16_mantissa` | Op | `BFloat16 -> UQ<7,0>` | Extract mantissa bits. |
| `_bf16_exponent` | Op | `BFloat16 -> UQ<8,0>` | Extract exponent bits. |
| `_bf16_sign` | Op | `BFloat16 -> UQ<1,0>` | Extract sign bit. |
| `bf16_decode` | Primitive | `BFloat16 -> (UQ<1,0> x UQ<7,0> x UQ<8,0>)` | Split BF16 into sign/mantissa/exponent. |

## FP16 helpers (`zolotone/components/Float16.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `fp16_pack` | Primitive | `UQ<1,0> -> UQ<5,0> -> UQ<10,0> -> Float16` | Assemble binary16 from sign, exponent, and mantissa fields. |
| `fp16_decode` | Primitive | `Float16 -> DecodedFP16` | Split binary16 into fields and zero/normal/subnormal/infinity/NaN flags. |

## Float helpers (`zolotone/numtypes/Float.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `float32_alloc` | Op | `UQ<1,0> -> UQ<23,0> -> UQ<8,0> -> Float32` | Assemble FP32 from sign, mantissa, exponent fields. |

## Tuple helpers (`zolotone/numtypes/Tuple.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `make_Tuple` | Op | `T0 -> T1 -> ... -> (T0 x T1 x ...)` | Variadic tuple constructor (arity = number of args). |

## Bool helpers (`zolotone/numtypes/Bool.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `is_typeof` | Op | `Any -> StaticType -> Bool<1>` | Static type check (returns bool node). |
| `negate` | Op | `Bool<1> -> Bool<1>` | Boolean negation. |

## General helpers (`zolotone/ast/AST.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `Copy` | Op | `T -> T` | Node copy. |
| `Tuple_get_item` | Op | `(T0 x T1 x ...) -> idx<int> -> T_idx` | Tuple selection by constant index. |

### Examples

## Common helpers (`examples/common.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `mantissa_add_implicit_bit` | Composite | `UQ<I,0> -> UQ<1,I>` | Prefix implicit leading 1. |
| `sign_xor` | Primitive | `UQ<1,0> -> UQ<1,0> -> UQ<1,0>` | XOR sign bits. |

## Optimized max exponent (`examples/max_exponent.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `OPTIMIZED_MAX_EXP4` | Primitive | `UQ<I0,0> -> UQ<I1,0> -> UQ<I2,0> -> UQ<I3,0> -> UQ<max(I0,I1,I2,I3), 0>` | Max of four exponents via bitwise tree. |

## Carry-save adder (`examples/CSA.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `CSA` | Primitive | `QT<I0, F0> -> QT<I1, F1> -> QT<I2, F2> -> (QT<max(I0, I1, I2), max(F0, F1, F2)> x QT<max(I0, I1, I2) + 1, max(F0, F1, F2)>)` | Carry save adder. |
| `CSA_tree4` | Composite | `Q<I0,F0> -> Q<I1,F1> -> Q<I2,F2> -> Q<I3,F3> -> Q<max(I0, I1, I2, I3)+3, max(F0, F1, F2, F3)>` | 4-input CSA tree with width alignment. |

## Float32 encoder (`examples/encode_Float32.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `round_to_the_nearest_even` | Primitive | `UQ<I,F> -> UQ<E,0> -> target_bits<int> -> (UQ<min(I, target_bits), max(target_bits - I, 0)> x UQ<E+1,0>)` | Guard/round/sticky rounding of mantissa and exponent bump. |
| `lzc` | Primitive | `UQ<I,F> -> UQ<ceil(log2(I+F+1)), 0>` | Leading-zero count. |
| `normalize_to_1_xxx` | Primitive | `UQ<I,F> -> Q<E,0> -> (UQ<1, max(I-1,0)+F> x Q<max(E, max(max(1, ceil(log2(I+F+1)))+1, bitlen(I)+1)+2)+1, 0>)` | Normalize mantissa to `1.xxx`; exponent stays signed with fractional bits preserved. |
| `encode_Float32` | Primitive | `Q<I,F> -> Q<E,0> -> Float32` | Pack sign, exponent, mantissa with subnormal/inf/nan handling. |

## Optimized design helpers (`examples/optimized.py`)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `_est_global_shift` | Primitive | `UQ<I,0> -> UQ<J,0> -> s<int> -> UQ<I+s,0>` | Compute global shift `(E_max - E_p) * 2^s`. |
| `_est_local_shift` | Primitive | `UQ<I,0> -> s<int> -> UQ<s,0>` | Invert low `s` exponent bits for local alignment. |
| `_prepend_ones` | Primitive | `UQ<I,0> -> s<int> -> UQ<I+s,0>` | Append `s` ones to exponent for normalization. |

## Examples (full composites)
| Name | Kind | Type | Purpose/Notes |
| --- | --- | --- | --- |
| `Conventional` | Composite | `BFloat16 -> BFloat16 -> BFloat16 -> BFloat16 -> BFloat16 -> BFloat16 -> BFloat16 -> BFloat16 -> Float32` | Baseline BF16 dot product (max exponent + global shift). |
| `Optimized` | Composite | `BFloat16 -> BFloat16 -> BFloat16 -> BFloat16 -> BFloat16 -> BFloat16 -> BFloat16 -> BFloat16 -> Float32` | Fused design with shared exponent estimation and local/global shifting. |
