"""Signed two's-complement fixed-point descriptor and runtime value."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from ._helpers import scalar_cpp_type, validate_fixed_raw
from .base import DataType, RuntimeValue


@dataclass(frozen=True)
class Q(DataType):
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

    def from_bits(self, raw: int) -> "QValue":
        return QValue(self, raw)

    @classmethod
    def from_int(cls, x: int) -> "QValue":
        if not isinstance(x, int):
            raise TypeError(f"Q.from_int expects int, got {type(x).__name__}")
        if x < 0:
            magnitude = abs(x)
            int_bits = max(2, (magnitude - 1).bit_length() + 1)
            raw = (1 << int_bits) + x
        else:
            int_bits = max(2, x.bit_length() + 1)
            raw = x
        return cls(int_bits, 0).from_bits(raw)

    def from_float(self, x: float) -> "QValue":
        if not isinstance(x, (int, float)):
            raise TypeError(f"Q.from_float expects int or float, got {type(x).__name__}")
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError(f"Q.from_float expects a finite number, got {x}")
        width = self.total_bits()
        scaled = int(round(x * (1 << self.frac_bits)))
        scaled = min(max(scaled, -(1 << (width - 1))), (1 << (width - 1)) - 1)
        return self.from_bits(scaled & ((1 << width) - 1))

    def to_spec(self, name, ctx):
        return ctx.fresh_real(name)

    def random_value(self, rng: random.Random) -> "QValue":
        return self.from_bits(rng.getrandbits(self.total_bits()))

    def to_cpp_type(self, jittable: bool = True) -> str:
        return scalar_cpp_type(self.total_bits(), jittable)

    def __repr__(self) -> str:
        return f"Q<{self.int_bits},{self.frac_bits}>"

    __str__ = __repr__


@dataclass(frozen=True)
class QValue(RuntimeValue):
    dtype: Q
    raw: int

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, Q):
            raise TypeError("QValue requires a Q descriptor")
        validate_fixed_raw("Q", self.raw, self.dtype.total_bits())

    def _scaled_integer(self) -> int:
        raw = self.raw
        if raw >> (self.dtype.total_bits() - 1):
            raw -= 1 << self.dtype.total_bits()
        return raw

    def to_python(self) -> float:
        return float(self._scaled_integer()) / (2 ** self.dtype.frac_bits)

    def to_spec(self, ctx):
        scaled = ctx.real_val(self._scaled_integer())
        if self.dtype.frac_bits == 0:
            return scaled
        return scaled * (ctx.two() ** ctx.real_val(-self.dtype.frac_bits))

    def to_bitstring(self) -> str:
        return format(self.raw, f"0{self.dtype.total_bits()}b")

    def __str__(self) -> str:
        return f"Q{self.dtype.int_bits}.{self.dtype.frac_bits}({self.to_python()})"


__all__ = ["Q", "QValue"]
