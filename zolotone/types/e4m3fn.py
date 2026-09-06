"""E4M3FN descriptor and runtime value."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import ClassVar

from ._helpers import (
    extract_float_fields,
    pack_float_fields,
    scalar_cpp_type,
    validate_float_fields,
    validate_float_raw,
)
from .base import DataType, RuntimeValue


@dataclass(frozen=True, repr=False)
class E4M3FN(DataType):
    sign_bits: ClassVar[int] = 1
    exponent_bits: ClassVar[int] = 4
    mantissa_bits: ClassVar[int] = 3
    exponent_bias: ClassVar[int] = 7
    zero_code: ClassVar[int] = 0
    sub_code: ClassVar[int] = 0
    inf_code: ClassVar[int | None] = None
    nan_code: ClassVar[int] = 15
    nan_mantissa: ClassVar[int] = 7
    max_finite_code: ClassVar[int] = 15
    max_finite_mantissa: ClassVar[int] = 6

    def total_bits(self) -> int:
        return 8

    def from_bits(self, raw: int) -> "E4M3FNValue":
        return E4M3FNValue(self, raw)

    def from_fields(self, sign: int, exponent: int, mantissa: int) -> "E4M3FNValue":
        validate_float_fields(
            type(self).__name__, sign, exponent, mantissa,
            sign_bits=self.sign_bits,
            exponent_bits=self.exponent_bits,
            mantissa_bits=self.mantissa_bits,
        )
        return self.from_bits(
            pack_float_fields(sign, exponent, mantissa, self.exponent_bits, self.mantissa_bits)
        )

    def to_spec(self, name, ctx):
        from ..spec.custom_specs.e4m3fn import e4m3fn
        return e4m3fn.fresh(name, ctx)

    def random_value(self, rng: random.Random) -> "E4M3FNValue":
        return self.from_bits(rng.getrandbits(self.total_bits()))

    def random_generator(self, seed=None, shared_exponent_bits: int = 0):
        if seed is None:
            seed = int(time.time())
        if not (0 <= shared_exponent_bits <= self.exponent_bits):
            raise ValueError(
                f"shared_exponent_bits must be between 0 and {self.exponent_bits}, "
                f"got {shared_exponent_bits}"
            )
        rng = random.Random(seed)
        unshared_bits = self.exponent_bits - shared_exponent_bits
        shared_exponent = rng.getrandbits(shared_exponent_bits) << unshared_bits

        def generate():
            return self.from_fields(
                rng.getrandbits(self.sign_bits),
                shared_exponent + rng.getrandbits(unshared_bits),
                rng.getrandbits(self.mantissa_bits),
            )

        def generate_shared_exponent():
            nonlocal shared_exponent
            shared_exponent = rng.getrandbits(shared_exponent_bits) << unshared_bits
            return shared_exponent

        return generate, generate_shared_exponent

    def Zero(self) -> "E4M3FNValue":
        return self.from_fields(0, self.zero_code, 0)

    def nZero(self) -> "E4M3FNValue":
        return self.from_fields(1, self.zero_code, 0)

    def NaN(self, payload: int | None = None) -> "E4M3FNValue":
        if payload is not None:
            raise ValueError("E4M3FN has a fixed NaN encoding")
        return self.from_fields(0, self.nan_code, self.nan_mantissa)

    def to_cpp_type(self, jittable: bool = True) -> str:
        return scalar_cpp_type(self.total_bits(), jittable)

    def __repr__(self) -> str:
        return "E4M3FN<8>"

    __str__ = __repr__


@dataclass(frozen=True)
class E4M3FNValue(RuntimeValue):
    dtype: E4M3FN
    raw: int

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, E4M3FN):
            raise TypeError("E4M3FNValue requires an E4M3FN descriptor")
        validate_float_raw("E4M3FN", self.raw, self.dtype.total_bits())

    def _fields(self) -> tuple[int, int, int]:
        return extract_float_fields(
            self.raw, self.dtype.sign_bits, self.dtype.exponent_bits, self.dtype.mantissa_bits
        )

    @property
    def sign(self) -> int:
        return self._fields()[0]

    @property
    def exponent(self) -> int:
        return self._fields()[1]

    @property
    def mantissa(self) -> int:
        return self._fields()[2]

    @property
    def significand(self) -> int:
        return self.mantissa

    @property
    def is_nan(self) -> bool:
        return self.exponent == self.dtype.nan_code and self.mantissa == self.dtype.nan_mantissa

    @property
    def is_inf(self) -> bool:
        return False

    @property
    def is_zero(self) -> bool:
        return self.exponent == self.dtype.zero_code and self.mantissa == 0

    @property
    def is_sub(self) -> bool:
        return not (self.is_nan or self.is_zero) and self.exponent == self.dtype.sub_code

    @property
    def is_norm(self) -> bool:
        return not (self.is_nan or self.is_zero or self.is_sub)

    def to_python(self) -> float:
        if self.is_nan:
            return float("nan")
        multiplier = -1.0 if self.sign else 1.0
        if self.exponent == self.dtype.sub_code:
            fraction = self.mantissa / (2 ** self.dtype.mantissa_bits)
            return float(multiplier * fraction * (2 ** (1 - self.dtype.exponent_bias)))
        fraction = 1.0 + self.mantissa / (2 ** self.dtype.mantissa_bits)
        return float(multiplier * fraction * (2 ** (self.exponent - self.dtype.exponent_bias)))

    def to_spec(self, ctx):
        from ..spec.custom_specs.e4m3fn import e4m3fn
        if self.is_nan:
            return e4m3fn.nan(ctx)
        if self.is_zero:
            return e4m3fn.nzero(ctx) if self.sign else e4m3fn.zero(ctx)
        return e4m3fn(
            value=ctx.real_val(self.to_python()),
            sign=ctx.real_val(self.sign),
            exponent=ctx.real_val(self.exponent),
            mantissa=ctx.real_val(self.mantissa),
            is_norm=ctx.bool_val(self.is_norm),
            is_sub=ctx.bool_val(self.is_sub),
            is_zero=ctx.bool_val(False),
            is_nan=ctx.bool_val(False),
        )

    def to_bitstring(self) -> str:
        return format(self.raw, f"0{self.dtype.total_bits()}b")

    def __str__(self) -> str:
        return f"E4M3FN({self.to_python()})"


__all__ = ["E4M3FN", "E4M3FNValue"]
