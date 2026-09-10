"""Unsigned E4M3 data-format descriptor."""

from dataclasses import dataclass
import random
from typing import ClassVar

from ._helpers import (
    extract_float_fields,
    pack_float_fields,
    scalar_cpp_type,
    validate_float_fields,
    validate_float_raw,
)
from .base import DataType, Value, value_method


@dataclass(frozen=True, repr=False)
class UE4M3(DataType[int]):
    sign_bits: ClassVar[int] = 0
    exponent_bits: ClassVar[int] = 4
    mantissa_bits: ClassVar[int] = 3
    exponent_bias: ClassVar[int] = 7
    nan_code: ClassVar[int] = 15
    nan_mantissa: ClassVar[int] = 7
    max_finite_code: ClassVar[int] = 15
    max_finite_mantissa: ClassVar[int] = 6
    min_subnormal: ClassVar[float] = 2 ** -9
    min_normal: ClassVar[float] = 2 ** -6
    max_finite: ClassVar[float] = 448.0
    raw_bits: ClassVar[int] = 7
    zero_code: ClassVar[int] = 0
    sub_code: ClassVar[int] = 0
    inf_code: ClassVar[int | None] = None

    def total_bits(self) -> int:
        return 8

    def validate_raw(self, raw: int) -> None:
        validate_float_raw(type(self).__name__, raw, self.raw_bits)

    def from_bits(self, raw: int) -> Value[int]:
        return Value(self, raw)

    def from_fields(self, exponent: int, mantissa: int) -> Value[int]:
        validate_float_fields(
            type(self).__name__,
            0,
            exponent,
            mantissa,
            sign_bits=self.sign_bits,
            exponent_bits=self.exponent_bits,
            mantissa_bits=self.mantissa_bits,
        )
        return self.from_bits(
            pack_float_fields(
                0, exponent, mantissa, self.exponent_bits, self.mantissa_bits
            )
        )

    def _extract_fields(self, raw: int) -> tuple[int, int, int]:
        self.validate_raw(raw)
        return extract_float_fields(
            raw, self.sign_bits, self.exponent_bits, self.mantissa_bits
        )

    @value_method
    def fields(self, raw: int) -> tuple[int, int]:
        _, exponent, mantissa = self._extract_fields(raw)
        return exponent, mantissa

    @value_method
    def sign(self, raw: int) -> int:
        return self._extract_fields(raw)[0]

    @value_method
    def exponent(self, raw: int) -> int:
        return self._extract_fields(raw)[1]

    @value_method
    def mantissa(self, raw: int) -> int:
        return self._extract_fields(raw)[2]

    @value_method
    def significand(self, raw: int) -> int:
        return self.mantissa(raw)

    @value_method
    def is_nan(self, raw: int) -> bool:
        return (
            self.exponent(raw) == self.nan_code
            and self.mantissa(raw) == self.nan_mantissa
        )

    @value_method
    def is_zero(self, raw: int) -> bool:
        return (
            not self.is_nan(raw)
            and self.exponent(raw) == self.zero_code
            and self.mantissa(raw) == 0
        )

    @value_method
    def is_sub(self, raw: int) -> bool:
        return not (self.is_nan(raw) or self.is_zero(raw)) and self.exponent(
            raw
        ) == self.sub_code

    @value_method
    def is_norm(self, raw: int) -> bool:
        return not (self.is_nan(raw) or self.is_zero(raw) or self.is_sub(raw))

    def to_python(self, raw: int) -> float:
        self.validate_raw(raw)
        if self.is_nan(raw):
            return float("nan")
        exponent = self.exponent(raw)
        mantissa = self.mantissa(raw)
        if exponent == self.sub_code:
            fraction = mantissa / (2 ** self.mantissa_bits)
            return float(fraction * (2 ** (1 - self.exponent_bias)))
        fraction = 1.0 + mantissa / (2 ** self.mantissa_bits)
        return float(fraction * (2 ** (exponent - self.exponent_bias)))

    def random_value(self, rng: random.Random) -> Value[int]:
        return self.from_bits(rng.getrandbits(self.raw_bits))

    def random_generator(self, seed=None, shared_exponent_bits: int = 0):
        if not (0 <= shared_exponent_bits <= self.exponent_bits):
            raise ValueError(
                f"shared_exponent_bits must be between 0 and {self.exponent_bits}, "
                f"got {shared_exponent_bits}"
            )
        rng = random.Random(seed)
        unshared_bits = self.exponent_bits - shared_exponent_bits
        shared_exponent = rng.getrandbits(shared_exponent_bits) << unshared_bits

        def generate():
            return self.from_fields(
                shared_exponent + rng.getrandbits(unshared_bits),
                rng.getrandbits(self.mantissa_bits),
            )

        def generate_shared_exponent():
            nonlocal shared_exponent
            shared_exponent = rng.getrandbits(shared_exponent_bits) << unshared_bits
            return shared_exponent

        return generate, generate_shared_exponent

    def to_spec(self, name, ctx):
        from ..spec.custom_specs.ue4m3 import ue4m3
        return ue4m3.fresh(name, ctx)

    def to_spec_value(self, raw: int, ctx):
        from ..spec.custom_specs.ue4m3 import ue4m3

        self.validate_raw(raw)
        if self.is_nan(raw):
            return ue4m3.nan(ctx)
        if self.is_zero(raw):
            return ue4m3.zero(ctx)
        return ue4m3(
            value=ctx.real_val(self.to_python(raw)),
            exponent=ctx.real_val(self.exponent(raw)),
            mantissa=ctx.real_val(self.mantissa(raw)),
            is_norm=ctx.bool_val(self.is_norm(raw)),
            is_sub=ctx.bool_val(self.is_sub(raw)),
            is_zero=ctx.bool_val(False),
            is_nan=ctx.bool_val(False),
        )

    def to_bitstring(self, raw: int) -> str:
        self.validate_raw(raw)
        return format(raw, f"0{self.total_bits()}b")

    def to_cpp_type(self, jittable: bool = True) -> str:
        return scalar_cpp_type(self.total_bits(), jittable)

    def format_value(self, raw: int) -> str:
        return f"{type(self).__name__}({self.to_python(raw)})"

    def Zero(self) -> Value[int]:
        return self.from_fields(self.zero_code, 0)

    def NaN(self, payload: int | None = None) -> Value[int]:
        if payload is not None:
            raise ValueError("UE4M3 has a fixed NaN encoding")
        return self.from_fields(self.nan_code, self.nan_mantissa)

    def __repr__(self) -> str:
        return "UE4M3<8>"

    __str__ = __repr__


__all__ = ["UE4M3"]
