"""Generic data-format and concrete-value contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import wraps
import random
from typing import Callable, Generic, TypeVar


RawT = TypeVar("RawT")
ResultT = TypeVar("ResultT")
_VALUE_METHOD_MARKER = "__zolotone_value_method__"


def value_method(method: Callable[..., ResultT]) -> Callable[..., ResultT]:
    """Allow a descriptor method to be called through a :class:`Value`.

    Marking is deliberately opt-in: descriptors can expose factories and
    metadata without turning ``Value`` into an unrestricted proxy.
    """

    setattr(method, _VALUE_METHOD_MARKER, True)
    return method


class DataType(ABC, Generic[RawT]):
    """Stateless descriptor for validating and interpreting raw payloads."""

    @abstractmethod
    def total_bits(self) -> int:
        """Return the descriptor's storage width."""

    @abstractmethod
    def validate_raw(self, raw: RawT) -> None:
        """Raise when ``raw`` is not valid data for this descriptor."""

    @abstractmethod
    def to_python(self, raw: RawT):
        """Convert raw data to its closest Python representation."""

    @abstractmethod
    def to_spec_value(self, raw: RawT, ctx):
        """Convert raw data to a symbolic specification constant."""

    @abstractmethod
    def to_spec(self, name, ctx):
        """Create a fresh symbolic specification value."""

    @abstractmethod
    def random_value(self, rng: random.Random) -> "Value[RawT]":
        """Create a typed concrete value using ``rng``."""

    @abstractmethod
    def to_cpp_type(self, jittable: bool = True) -> str:
        """Return the C++ storage type used by code generation."""

    def format_value(self, raw: RawT) -> str:
        self.validate_raw(raw)
        return str(self.to_python(raw))

    def _fingerprint(self):
        descriptor_fields = ()
        if hasattr(self, "__dataclass_fields__"):
            descriptor_fields = tuple(
                (field.name, getattr(self, field.name)) for field in fields(self)
            )
        return (type(self).__name__, descriptor_fields)


@dataclass(frozen=True)
class Value(Generic[RawT]):
    """A concrete raw payload paired with its exact data-format descriptor."""

    dtype: DataType[RawT]
    raw: RawT

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DataType):
            raise TypeError(
                f"Value dtype must be a DataType, got {type(self.dtype).__name__}"
            )
        self.dtype.validate_raw(self.raw)

    def to_python(self):
        return self.dtype.to_python(self.raw)

    def to_spec(self, ctx):
        return self.dtype.to_spec_value(self.raw, ctx)

    def to_bitstring(self) -> str:
        method = getattr(self.dtype, "to_bitstring", None)
        if method is None or not callable(method):
            raise AttributeError(
                f"{type(self.dtype).__name__} values do not support to_bitstring"
            )
        return method(self.raw)

    def _fingerprint(self):
        return ("Value", self.dtype._fingerprint(), self.raw)

    def __getattr__(self, name: str):
        descriptor_attribute = getattr(self.dtype, name, None)
        if not callable(descriptor_attribute) or not getattr(
            descriptor_attribute, _VALUE_METHOD_MARKER, False
        ):
            raise AttributeError(
                f"{type(self).__name__} object has no attribute {name!r}"
            )

        @wraps(descriptor_attribute)
        def forwarded(*args, **kwargs):
            return descriptor_attribute(self.raw, *args, **kwargs)

        return forwarded

    def __dir__(self) -> list[str]:
        forwarded = {
            name
            for name in dir(self.dtype)
            if callable(attribute := getattr(self.dtype, name, None))
            and getattr(attribute, _VALUE_METHOD_MARKER, False)
        }
        return sorted(set(super().__dir__()) | forwarded)

    def __str__(self) -> str:
        return self.dtype.format_value(self.raw)


__all__ = ["DataType", "RawT", "Value", "value_method"]
