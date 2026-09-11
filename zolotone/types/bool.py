"""Boolean data-format descriptor."""

from dataclasses import dataclass
import random

from ._helpers import scalar_cpp_type
from .base import DataType, Value


@dataclass(frozen=True)
class Bool(DataType[int]):
    def total_bits(self) -> int:
        return 1

    def validate_raw(self, raw: int) -> None:
        if not isinstance(raw, int):
            raise TypeError(f"Bool raw value must be int, got {type(raw).__name__}")
        if raw not in (0, 1):
            raise ValueError(f"Bool value must be 0 or 1, got {raw}")

    def from_bits(self, raw: int) -> Value[int]:
        return Value(self, raw)

    def to_python(self, value: object) -> bool:
        self.validate_raw(value)
        return bool(value)

    def to_spec_value(self, value: object, ctx):
        return ctx.bool_val(self.to_python(value))

    def to_spec(self, name, ctx):
        return ctx.fresh_bool(name)

    def random_value(self, rng: random.Random) -> Value[int]:
        return self.from_bits(rng.getrandbits(1))

    def to_bitstring(self, value: object) -> str:
        self.validate_raw(value)
        return format(value, "01b")

    def to_cpp_type(self, jittable: bool = True) -> str:
        return scalar_cpp_type(self.total_bits(), jittable)

    def format_value(self, value: object) -> str:
        self.validate_raw(value)
        return f"Bool({value})"

    def __repr__(self) -> str:
        return "Bool<1>"

    __str__ = __repr__


__all__ = ["Bool"]
