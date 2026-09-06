"""Boolean descriptor and runtime value."""

from __future__ import annotations

from dataclasses import dataclass
import random

from ._helpers import scalar_cpp_type
from .base import DataType, RuntimeValue


@dataclass(frozen=True)
class Bool(DataType):
    def total_bits(self) -> int:
        return 1

    def from_bits(self, raw: int) -> "BoolValue":
        return BoolValue(self, raw)

    def to_spec(self, name, ctx):
        return ctx.fresh_bool(name)

    def random_value(self, rng: random.Random) -> "BoolValue":
        return self.from_bits(rng.getrandbits(1))

    def to_cpp_type(self, jittable: bool = True) -> str:
        return scalar_cpp_type(self.total_bits(), jittable)

    def __repr__(self) -> str:
        return "Bool<1>"

    __str__ = __repr__


@dataclass(frozen=True)
class BoolValue(RuntimeValue):
    dtype: Bool
    raw: int

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, Bool):
            raise TypeError("BoolValue requires a Bool descriptor")
        if not isinstance(self.raw, int):
            raise TypeError(f"Bool raw value must be int, got {type(self.raw).__name__}")
        if self.raw not in (0, 1):
            raise ValueError(f"Bool value must be 0 or 1, got {self.raw}")

    def to_python(self) -> bool:
        return bool(self.raw)

    def to_spec(self, ctx):
        return ctx.bool_val(self.to_python())

    def to_bitstring(self) -> str:
        return format(self.raw, "01b")

    def __str__(self) -> str:
        return f"Bool({self.raw})"


__all__ = ["Bool", "BoolValue"]
