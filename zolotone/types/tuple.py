"""Tuple data-format descriptor."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .base import DataType, Value


@dataclass(frozen=True, init=False)
class Tuple(DataType[tuple]):
    items: tuple[DataType, ...]

    def __init__(self, *items: DataType):
        if not items:
            raise ValueError("Tuple cannot be empty")
        if not all(isinstance(item, DataType) for item in items):
            raise TypeError("Tuple must contain DataType descriptors")
        object.__setattr__(self, "items", tuple(items))

    def total_bits(self) -> int:
        return sum(item.total_bits() for item in self.items)

    def validate_raw(self, raw: tuple) -> None:
        if not isinstance(raw, tuple):
            raise TypeError(
                f"Tuple raw value must be tuple, got {type(raw).__name__}"
            )
        if len(raw) != len(self.items):
            raise ValueError(
                f"Tuple value has {len(raw)} items; expected {len(self.items)}"
            )
        for item_type, item_raw in zip(self.items, raw):
            item_type.validate_raw(item_raw)

    def from_values(self, *values: Value) -> Value[tuple]:
        if len(values) != len(self.items):
            raise ValueError(
                f"Tuple value has {len(values)} items; expected {len(self.items)}"
            )
        raw_items = []
        for index, (value, expected_dtype) in enumerate(zip(values, self.items)):
            if not isinstance(value, Value):
                raise TypeError(f"Tuple item {index} is not a Value")
            if value.dtype != expected_dtype:
                raise TypeError(
                    f"Tuple item {index} has descriptor {value.dtype}; "
                    f"expected {expected_dtype}"
                )
            raw_items.append(value.raw)
        return Value(self, tuple(raw_items))

    def to_python(self, value: object) -> tuple:
        self.validate_raw(value)
        return tuple(
            item_type.to_python(item_value)
            for item_type, item_value in zip(self.items, value)
        )

    def to_spec_value(self, value: object, ctx):
        self.validate_raw(value)
        return tuple(
            item_type.to_spec_value(item_value, ctx)
            for item_type, item_value in zip(self.items, value)
        )

    def to_spec(self, name, ctx):
        return tuple(
            item.to_spec(name=f"{name}_{index}", ctx=ctx)
            for index, item in enumerate(self.items)
        )

    def random_value(self, rng: random.Random) -> Value[tuple]:
        return self.from_values(*(item.random_value(rng) for item in self.items))

    def to_cpp_type(self, jittable: bool = True) -> str:
        if jittable:
            return f"std::array<uint64_t, {len(self.items)}>"
        return f"std::tuple<{', '.join(item.to_cpp_type(False) for item in self.items)}>"

    def format_value(self, value: object) -> str:
        self.validate_raw(value)
        rendered = (
            item_type.format_value(item_value)
            for item_type, item_value in zip(self.items, value)
        )
        return f"Tuple[{', '.join(rendered)}]"

    def __repr__(self) -> str:
        return f"Tuple<{', '.join(repr(item) for item in self.items)}>"

    __str__ = __repr__


__all__ = ["Tuple"]
