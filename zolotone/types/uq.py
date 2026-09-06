"""Unsigned fixed-point descriptor and runtime value."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from ._helpers import scalar_cpp_type, validate_fixed_raw
from .base import DataType, RuntimeValue


@dataclass(frozen=True)
class UQ(DataType):
    int_bits: int
    frac_bits: int

    def __post_init__(self) -> None:
        if not isinstance(self.int_bits, int) or not isinstance(self.frac_bits, int):
            raise TypeError("UQ bit widths must be integers")
        if self.int_bits < 0 or self.frac_bits < 0:
            raise ValueError(
                "UQ bit widths must be non-negative, "
                f"got int_bits={self.int_bits}, frac_bits={self.frac_bits}"
            )
        if self.total_bits() < 1:
            raise ValueError("UQ requires at least one total bit")

    def total_bits(self) -> int:
        return self.int_bits + self.frac_bits

    def from_bits(self, raw: int) -> "UQValue":
        return UQValue(self, raw)

    @classmethod
    def from_int(cls, x: int) -> "UQValue":
        if not isinstance(x, int):
            raise TypeError(f"UQ.from_int expects int, got {type(x).__name__}")
        if x < 0:
            raise ValueError(f"UQ.from_int expects a non-negative integer, got {x}")
        return cls(max(1, x.bit_length()), 0).from_bits(x)

    def from_float(self, x: float) -> "UQValue":
        if not isinstance(x, (int, float)):
            raise TypeError(f"UQ.from_float expects int or float, got {type(x).__name__}")
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError(f"UQ.from_float expects a finite number, got {x}")
        scaled = int(round(x * (1 << self.frac_bits)))
        scaled = min(max(scaled, 0), (1 << self.total_bits()) - 1)
        return self.from_bits(scaled)

    def to_spec(self, name, ctx):
        variable = ctx.fresh_real(name)
        ctx.assume(variable.eq(abs(variable)))
        return variable

    def random_value(self, rng: random.Random) -> "UQValue":
        return self.from_bits(rng.getrandbits(self.total_bits()))

    def to_cpp_type(self, jittable: bool = True) -> str:
        return scalar_cpp_type(self.total_bits(), jittable)

    def __repr__(self) -> str:
        return f"UQ<{self.int_bits},{self.frac_bits}>"

    __str__ = __repr__


@dataclass(frozen=True)
class UQValue(RuntimeValue):
    dtype: UQ
    raw: int

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, UQ):
            raise TypeError("UQValue requires a UQ descriptor")
        validate_fixed_raw("UQ", self.raw, self.dtype.total_bits())

    def _scaled_integer(self) -> int:
        return self.raw

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
        return f"UQ{self.dtype.int_bits}.{self.dtype.frac_bits}({self.to_python()})"


__all__ = ["UQ", "UQValue"]
