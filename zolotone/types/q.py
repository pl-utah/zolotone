"""Signed two's-complement fixed-point descriptor."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from ._helpers import scalar_cpp_type, validate_fixed_raw
from .base import DataType, Value


@dataclass(frozen=True)
class Q(DataType[int]):
    int_bits: int
    frac_bits: int

    def __post_init__(self) -> None:
        if not isinstance(self.int_bits, int) or not isinstance(self.frac_bits, int):
            raise TypeError("Q bit widths must be integers")
        if self.int_bits < 0 or self.frac_bits < 0:
            raise ValueError(
                "Q bit widths must be non-negative, "
                f"got int_bits={self.int_bits}, frac_bits={self.frac_bits}"
            )
        if self.total_bits() < 1:
            raise ValueError("Q requires at least one total bit")

    def total_bits(self) -> int:
        return self.int_bits + self.frac_bits

    def validate_raw(self, raw: int) -> None:
        validate_fixed_raw("Q", raw, self.total_bits())

    def from_bits(self, raw: int) -> Value[int]:
        return Value(self, raw)

    @classmethod
    def from_int(cls, value: int) -> Value[int]:
        if not isinstance(value, int):
            raise TypeError(f"Q.from_int expects int, got {type(value).__name__}")
        if value < 0:
            magnitude = abs(value)
            int_bits = max(2, (magnitude - 1).bit_length() + 1)
            raw = (1 << int_bits) + value
        else:
            int_bits = max(2, value.bit_length() + 1)
            raw = value
        dtype = cls(int_bits, 0)
        return dtype.from_bits(raw)

    def from_float(self, value: float) -> Value[int]:
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Q.from_float expects int or float, got {type(value).__name__}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Q.from_float expects a finite number, got {value}")
        width = self.total_bits()
        scaled = int(round(value * (1 << self.frac_bits)))
        scaled = min(max(scaled, -(1 << (width - 1))), (1 << (width - 1)) - 1)
        return self.from_bits(scaled & ((1 << width) - 1))

    def _scaled_integer(self, value: object) -> int:
        self.validate_raw(value)
        if value >> (self.total_bits() - 1):
            return value - (1 << self.total_bits())
        return value

    def to_python(self, value: object) -> float:
        return float(self._scaled_integer(value)) / (2 ** self.frac_bits)

    def to_spec_value(self, value: object, ctx):
        scaled = ctx.real_val(self._scaled_integer(value))
        if self.frac_bits == 0:
            return scaled
        return scaled * (ctx.two() ** ctx.real_val(-self.frac_bits))

    def to_spec(self, name, ctx):
        return ctx.fresh_real(name)

    def random_value(self, rng: random.Random) -> Value[int]:
        return self.from_bits(rng.getrandbits(self.total_bits()))

    def to_bitstring(self, value: object) -> str:
        self.validate_raw(value)
        return format(value, f"0{self.total_bits()}b")

    def to_cpp_type(self, jittable: bool = True) -> str:
        return scalar_cpp_type(self.total_bits(), jittable)

    def format_value(self, value: object) -> str:
        return f"Q{self.int_bits}.{self.frac_bits}({self.to_python(value)})"

    def __repr__(self) -> str:
        return f"Q<{self.int_bits},{self.frac_bits}>"

    __str__ = __repr__


__all__ = ["Q"]
