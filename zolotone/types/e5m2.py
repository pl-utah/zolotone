"""IEEE-style E5M2 data-format descriptor."""

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
from .base import DataType, Value, value_method


@dataclass(frozen=True, repr=False)
class E5M2(DataType[int]):
    sign_bits: ClassVar[int] = 1
    exponent_bits: ClassVar[int] = 5
    mantissa_bits: ClassVar[int] = 2
    exponent_bias: ClassVar[int] = 15
    inf_code: ClassVar[int] = 31
    nan_code: ClassVar[int] = 31
    max_finite_code: ClassVar[int] = 30
    max_finite_mantissa: ClassVar[int] = 3
    min_subnormal: ClassVar[float] = 2 ** -16
    min_normal: ClassVar[float] = 2 ** -14
    max_finite: ClassVar[float] = 57344.0
    zero_code: ClassVar[int] = 0
    sub_code: ClassVar[int] = 0

    def total_bits(self) -> int:
        return self.sign_bits + self.exponent_bits + self.mantissa_bits

    def validate_raw(self, raw: int) -> None:
        validate_float_raw(type(self).__name__, raw, self.total_bits())

    def from_bits(self, raw: int) -> Value[int]:
        return Value(self, raw)

    def from_fields(self, sign: int, exponent: int, mantissa: int) -> Value[int]:
        validate_float_fields(
            type(self).__name__,
            sign,
            exponent,
            mantissa,
            sign_bits=self.sign_bits,
            exponent_bits=self.exponent_bits,
            mantissa_bits=self.mantissa_bits,
        )
        return self.from_bits(
            pack_float_fields(
                sign, exponent, mantissa, self.exponent_bits, self.mantissa_bits
            )
        )

    def _extract_fields(self, raw: int) -> tuple[int, int, int]:
        self.validate_raw(raw)
        return extract_float_fields(
            raw, self.sign_bits, self.exponent_bits, self.mantissa_bits
        )

    @value_method
    def fields(self, raw: int) -> tuple[int, int, int]:
        return self._extract_fields(raw)

    @value_method
    def sign(self, raw: int) -> int:
        return self._extract_fields(raw)[0]

    @value_method
    def exponent(self, raw: int) -> int:
        return self._extract_fields(raw)[1]

    @value_method
    def mantissa(self, raw: int) -> int:
        return self._extract_fields(raw)[2]

    @value_method
    def significand(self, raw: int) -> int:
        return self.mantissa(raw)

    @value_method
    def is_nan(self, raw: int) -> bool:
        return self.exponent(raw) == self.nan_code and self.mantissa(raw) != 0

    @value_method
    def is_inf(self, raw: int) -> bool:
        return self.exponent(raw) == self.inf_code and self.mantissa(raw) == 0

    @value_method
    def is_zero(self, raw: int) -> bool:
        return (
            not self.is_nan(raw)
            and self.exponent(raw) == self.zero_code
            and self.mantissa(raw) == 0
        )

    @value_method
    def is_sub(self, raw: int) -> bool:
        return not (
            self.is_nan(raw) or self.is_inf(raw) or self.is_zero(raw)
        ) and self.exponent(raw) == self.sub_code

    @value_method
    def is_norm(self, raw: int) -> bool:
        return not (
            self.is_nan(raw)
            or self.is_inf(raw)
            or self.is_zero(raw)
            or self.is_sub(raw)
        )

    def to_python(self, raw: int) -> float:
        self.validate_raw(raw)
        if self.is_inf(raw):
            return float("-inf") if self.sign(raw) else float("inf")
        if self.is_nan(raw):
            return float("nan")
        multiplier = -1.0 if self.sign(raw) else 1.0
        exponent = self.exponent(raw)
        mantissa = self.mantissa(raw)
        if exponent == self.sub_code:
            fraction = mantissa / (2 ** self.mantissa_bits)
            return float(multiplier * fraction * (2 ** (1 - self.exponent_bias)))
        fraction = 1.0 + mantissa / (2 ** self.mantissa_bits)
        return float(multiplier * fraction * (2 ** (exponent - self.exponent_bias)))

    def to_spec(self, name, ctx):
        from ..spec.custom_specs.e5m2 import e5m2
        return e5m2.fresh(name, ctx)

    def to_spec_value(self, raw: int, ctx):
        from ..spec.custom_specs.e5m2 import e5m2

        self.validate_raw(raw)
        if self.is_inf(raw):
            return e5m2.ninf(ctx) if self.sign(raw) else e5m2.inf(ctx)
        if self.is_nan(raw):
            return e5m2.nan(ctx)
        if self.is_zero(raw):
            return e5m2.nzero(ctx) if self.sign(raw) else e5m2.zero(ctx)
        return e5m2(
            value=ctx.real_val(self.to_python(raw)),
            exponent=ctx.real_val(self.exponent(raw)),
            mantissa=ctx.real_val(self.mantissa(raw)),
            sign=ctx.real_val(self.sign(raw)),
            is_norm=ctx.bool_val(self.is_norm(raw)),
            is_sub=ctx.bool_val(self.is_sub(raw)),
            is_zero=ctx.bool_val(False),
            is_inf=ctx.bool_val(False),
            is_nan=ctx.bool_val(False),
        )

    def random_value(self, rng: random.Random) -> Value[int]:
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

    def to_bitstring(self, raw: int) -> str:
        self.validate_raw(raw)
        return format(raw, f"0{self.total_bits()}b")

    def to_cpp_type(self, jittable: bool = True) -> str:
        return scalar_cpp_type(self.total_bits(), jittable)

    def format_value(self, raw: int) -> str:
        return f"{type(self).__name__}({self.to_python(raw)})"

    def Zero(self) -> Value[int]:
        return self.from_fields(0, self.zero_code, 0)

    def nZero(self) -> Value[int]:
        return self.from_fields(1, self.zero_code, 0)

    def Inf(self) -> Value[int]:
        return self.from_fields(0, self.inf_code, 0)

    def nInf(self) -> Value[int]:
        return self.from_fields(1, self.inf_code, 0)

    def NaN(self, payload: int | None = None) -> Value[int]:
        if payload is None:
            payload = 1 << (self.mantissa_bits - 1)
        if not isinstance(payload, int):
            raise TypeError(f"NaN payload must be int, got {type(payload).__name__}")
        if not (1 <= payload < (1 << self.mantissa_bits)):
            raise ValueError(
                f"NaN payload must fit in {self.mantissa_bits} mantissa bits "
                f"and be non-zero, got {payload}"
            )
        return self.from_fields(0, self.nan_code, payload)

    def __repr__(self) -> str:
        return "E5M2<8>"

    __str__ = __repr__


__all__ = ["E5M2"]
