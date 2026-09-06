"""Abstract contracts used by the graph engine.

Concrete data formats own all construction, encoding, conversion, and
formatting behavior.  These base classes only provide the common identities
that the graph engine uses for validation and stable fingerprints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
import random


class DataType(ABC):
    """Abstract descriptor contract understood by the graph engine."""

    @abstractmethod
    def total_bits(self) -> int:
        """Return the descriptor's storage width."""

    @abstractmethod
    def to_spec(self, name, ctx):
        """Create a symbolic specification value."""

    @abstractmethod
    def random_value(self, rng: random.Random) -> "RuntimeValue":
        """Create a concrete value using ``rng``."""

    @abstractmethod
    def to_cpp_type(self, jittable: bool = True) -> str:
        """Return the C++ storage type used by code generation."""

    def _fingerprint(self):
        descriptor_fields = ()
        if hasattr(self, "__dataclass_fields__"):
            descriptor_fields = tuple(
                (field.name, getattr(self, field.name)) for field in fields(self)
            )
        return (type(self).__name__, descriptor_fields)


class RuntimeValue(ABC):
    """Abstract concrete-value contract understood by the graph engine."""

    @abstractmethod
    def to_python(self):
        """Convert this value to its closest Python representation."""

    @abstractmethod
    def to_spec(self, ctx):
        """Convert this value to a symbolic specification constant."""

    def _fingerprint(self):
        return (type(self).__name__, self.dtype._fingerprint(), self.raw)


__all__ = ["DataType", "RuntimeValue"]
