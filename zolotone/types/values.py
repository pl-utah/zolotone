"""Immutable concrete values for Zolotone data-format descriptors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .descriptors import DataType, Bool, Q, UQ, _FloatDescriptor, Tuple


class RuntimeValue:
    """Base class for immutable values consumed and produced by implementations."""

    dtype: "DataType"
    raw: object

    def to_python(self):
        raise NotImplementedError

    def to_spec(self, ctx):
        raise NotImplementedError

    def to_bitstring(self) -> str:
        """Return the packed encoding, padded to the descriptor width."""
        if not isinstance(self.raw, int):
            raise TypeError(
                f"{type(self).__name__} does not have a scalar bit encoding"
            )
        return format(self.raw, f"0{self.dtype.total_bits()}b")

    def _fingerprint(self):
        return (type(self).__name__, self.dtype._fingerprint(), self.raw)


@dataclass(frozen=True)
class BoolValue(RuntimeValue):
    dtype: "Bool"
    raw: int

    def __post_init__(self) -> None:
        from .descriptors import Bool
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

    def __str__(self) -> str:
        return f"Bool({self.raw})"


@dataclass(frozen=True)
class FixedValue(RuntimeValue):
    dtype: "Q | UQ"
    raw: int

    def __post_init__(self) -> None:
        from .descriptors import Q, UQ
        if not isinstance(self.dtype, (Q, UQ)):
            raise TypeError("FixedValue requires a Q or UQ descriptor")
        if not isinstance(self.raw, int):
            raise TypeError(
                f"{type(self.dtype).__name__} raw value must be int, "
                f"got {type(self.raw).__name__}"
            )
        if not (0 <= self.raw < (1 << self.dtype.total_bits())):
            raise ValueError(
                f"{type(self.dtype).__name__} value {self.raw} does not fit into "
                f"{self.dtype.total_bits()} bits"
            )

    def _scaled_integer(self) -> int:
        from .descriptors import Q
        raw = self.raw
        if isinstance(self.dtype, Q) and raw >> (self.dtype.total_bits() - 1):
            raw -= 1 << self.dtype.total_bits()
        return raw

    def to_python(self) -> float:
        return float(self._scaled_integer()) / (2 ** self.dtype.frac_bits)

    def to_spec(self, ctx):
        scaled = ctx.real_val(self._scaled_integer())
        if self.dtype.frac_bits == 0:
            return scaled
        return scaled * (ctx.two() ** ctx.real_val(-self.dtype.frac_bits))

    def __str__(self) -> str:
        return (
            f"{type(self.dtype).__name__}{self.dtype.int_bits}."
            f"{self.dtype.frac_bits}({self.to_python()})"
        )


@dataclass(frozen=True)
class FloatValue(RuntimeValue):
    dtype: "_FloatDescriptor"
    raw: int

    def __post_init__(self) -> None:
        from .descriptors import _FloatDescriptor
        if not isinstance(self.dtype, _FloatDescriptor):
            raise TypeError("FloatValue requires a floating-point descriptor")
        self.dtype._validate_raw(self.raw)

    @property
    def sign(self) -> int:
        return self.dtype._fields(self.raw)[0]

    @property
    def exponent(self) -> int:
        return self.dtype._fields(self.raw)[1]

    @property
    def mantissa(self) -> int:
        return self.dtype._fields(self.raw)[2]

    @property
    def significand(self) -> int:
        return self.mantissa

    @property
    def is_nan(self) -> bool:
        return self.dtype._is_nan(self.raw)

    @property
    def is_inf(self) -> bool:
        return self.dtype._is_inf(self.raw)

    @property
    def is_zero(self) -> bool:
        return self.dtype._is_zero(self.raw)

    @property
    def is_sub(self) -> bool:
        return not (self.is_nan or self.is_inf or self.is_zero) and self.exponent == 0

    @property
    def is_norm(self) -> bool:
        return not (self.is_nan or self.is_inf or self.is_zero or self.is_sub)

    def to_python(self) -> float:
        return self.dtype._to_python(self.raw)

    def to_spec(self, ctx):
        return self.dtype._value_to_spec(self, ctx)

    def __str__(self) -> str:
        return f"{type(self.dtype).__name__}({self.to_python()})"


@dataclass(frozen=True)
class TupleValue(RuntimeValue):
    dtype: "Tuple"
    items: tuple[RuntimeValue, ...]

    def __post_init__(self) -> None:
        from .descriptors import Tuple
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


__all__ = ["RuntimeValue", "BoolValue", "FixedValue", "FloatValue", "TupleValue"]
