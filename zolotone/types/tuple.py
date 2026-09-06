"""Tuple descriptor and runtime value."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .base import DataType, RuntimeValue


@dataclass(frozen=True, init=False)
class Tuple(DataType):
    items: tuple[DataType, ...]

    def __init__(self, *items: DataType):
        if not items:
            raise ValueError("Tuple cannot be empty")
        if not all(isinstance(item, DataType) for item in items):
            raise TypeError("Tuple must contain DataType descriptors")
        object.__setattr__(self, "items", tuple(items))

    def total_bits(self) -> int:
        return sum(item.total_bits() for item in self.items)

    def from_values(self, *values: RuntimeValue) -> "TupleValue":
        return TupleValue(self, tuple(values))

    def to_spec(self, name, ctx):
        return tuple(
            item.to_spec(name=f"{name}_{index}", ctx=ctx)
            for index, item in enumerate(self.items)
        )

    def random_value(self, rng: random.Random) -> "TupleValue":
        return self.from_values(*(item.random_value(rng) for item in self.items))

    def to_cpp_type(self, jittable: bool = True) -> str:
        if jittable:
            return f"std::array<uint64_t, {len(self.items)}>"
        return f"std::tuple<{', '.join(item.to_cpp_type(False) for item in self.items)}>"

    def __repr__(self) -> str:
        return f"Tuple<{', '.join(repr(item) for item in self.items)}>"

    __str__ = __repr__


@dataclass(frozen=True)
class TupleValue(RuntimeValue):
    dtype: Tuple
    items: tuple[RuntimeValue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, Tuple):
            raise TypeError("TupleValue requires a Tuple descriptor")
        if not isinstance(self.items, tuple):
            object.__setattr__(self, "items", tuple(self.items))
        if len(self.items) != len(self.dtype.items):
            raise ValueError(
                f"Tuple value has {len(self.items)} items; expected {len(self.dtype.items)}"
            )
        for index, (value, dtype) in enumerate(zip(self.items, self.dtype.items)):
            if not isinstance(value, RuntimeValue):
                raise TypeError(f"Tuple item {index} is not a RuntimeValue")
            if value.dtype != dtype:
                raise TypeError(
                    f"Tuple item {index} has descriptor {value.dtype}; expected {dtype}"
                )

    @property
    def raw(self) -> tuple:
        return tuple(item.raw for item in self.items)

    def to_python(self) -> tuple:
        return tuple(item.to_python() for item in self.items)

    def to_spec(self, ctx):
        return tuple(item.to_spec(ctx) for item in self.items)

    def __str__(self) -> str:
        return f"TupleValue[{', '.join(str(item) for item in self.items)}]"


__all__ = ["Tuple", "TupleValue"]
