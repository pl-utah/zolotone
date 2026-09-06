"""Stateless bit-manipulation helpers for concrete type modules."""

from __future__ import annotations


def scalar_cpp_type(total_bits: int, jittable: bool = True) -> str:
    if jittable:
        if total_bits <= 8:
            return "uint8_t"
        if total_bits <= 16:
            return "uint16_t"
        if total_bits <= 32:
            return "uint32_t"
        if total_bits <= 64:
            return "uint64_t"
        raise TypeError("Can not find an ABI-safe type with more than 64 bits in C")
    return f"ac_uint<{total_bits}>"


def validate_fixed_raw(format_name: str, raw: int, width: int) -> None:
    if not isinstance(raw, int):
        raise TypeError(
            f"{format_name} raw value must be int, got {type(raw).__name__}"
        )
    if not (0 <= raw < (1 << width)):
        raise ValueError(
            f"{format_name} value {raw} does not fit into {width} bits"
        )


def validate_float_raw(format_name: str, raw: int, width: int) -> None:
    if not isinstance(raw, int):
        raise TypeError(
            f"{format_name} expects packed bits as int, got {type(raw).__name__}"
        )
    if not (0 <= raw < (1 << width)):
        raise ValueError(
            f"{format_name} packed bits must fit in {width} bits, got {raw}"
        )


def validate_float_fields(
    format_name: str,
    sign: int,
    exponent: int,
    mantissa: int,
    *,
    sign_bits: int,
    exponent_bits: int,
    mantissa_bits: int,
) -> None:
    if sign_bits:
        if not isinstance(sign, int) or sign not in (0, 1):
            raise ValueError(f"{format_name} sign must be 0 or 1, got {sign}")
    elif sign != 0:
        raise ValueError(f"{format_name} has no sign bit")
    if not isinstance(exponent, int) or not (0 <= exponent < (1 << exponent_bits)):
        raise ValueError(f"{format_name} exponent out of range: {exponent}")
    if not isinstance(mantissa, int) or not (0 <= mantissa < (1 << mantissa_bits)):
        raise ValueError(f"{format_name} mantissa out of range: {mantissa}")


def pack_float_fields(
    sign: int, exponent: int, mantissa: int, exponent_bits: int, mantissa_bits: int
) -> int:
    return (
        (sign << (exponent_bits + mantissa_bits))
        | (exponent << mantissa_bits)
        | mantissa
    )


def extract_float_fields(
    raw: int, sign_bits: int, exponent_bits: int, mantissa_bits: int
) -> tuple[int, int, int]:
    sign = (
        (raw >> (exponent_bits + mantissa_bits)) & 1 if sign_bits else 0
    )
    exponent = (raw >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = raw & ((1 << mantissa_bits) - 1)
    return sign, exponent, mantissa
