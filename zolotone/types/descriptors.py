"""Immutable data-format descriptors.

Descriptors describe the shape and interpretation of data.  Concrete packed
bits live in :mod:`zolotone.types.values`; a descriptor never carries a value.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import random
import time
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .values import RuntimeValue, BoolValue, FixedValue, FloatValue, TupleValue


class DataType:
    """Base class for immutable data-format descriptors."""

    def total_bits(self) -> int:
        raise NotImplementedError

    def from_bits(self, raw):
        """Construct a concrete value from its packed bit encoding."""
        raise NotImplementedError

    def to_spec(self, name, ctx):
        raise NotImplementedError

    def random_value(self, rng: random.Random) -> "RuntimeValue":
        return self.from_bits(rng.getrandbits(self.total_bits()))

    def to_cpp_type(self, jittable: bool = True) -> str:
        total_bits = self.total_bits()
        if jittable:
            if total_bits <= 8:
                return "uint8_t"
            if total_bits <= 16:
                return "uint16_t"
            if total_bits <= 32:
                return "uint32_t"
            if total_bits <= 64:
                return "uint64_t"
            raise TypeError("Can not find an ABI-safe type with more than 64 bits in C")
        return f"ac_uint<{total_bits}>"

    def _fingerprint(self):
        descriptor_fields = ()
        if hasattr(self, "__dataclass_fields__"):
            descriptor_fields = tuple(
                (field.name, getattr(self, field.name)) for field in fields(self)
            )
        return (type(self).__name__, descriptor_fields)


@dataclass(frozen=True)
class Bool(DataType):
    def total_bits(self) -> int:
        return 1

    def from_bits(self, raw: int) -> "BoolValue":
        """Construct a Boolean from its one-bit encoding."""
        from .values import BoolValue
        return BoolValue(self, raw)

    def to_spec(self, name, ctx):
        return ctx.fresh_bool(name)

    def __repr__(self) -> str:
        return "Bool<1>"

    __str__ = __repr__


@dataclass(frozen=True)
class Q(DataType):
    """Signed two's-complement fixed-point descriptor."""

    int_bits: int
    frac_bits: int

    def __post_init__(self) -> None:
        if not isinstance(self.int_bits, int) or not isinstance(self.frac_bits, int):
            raise TypeError("Q bit widths must be integers")
        if self.int_bits < 0 or self.frac_bits < 0:
            raise ValueError(
                "Q bit widths must be non-negative, "
                f"got int_bits={self.int_bits}, frac_bits={self.frac_bits}"
            )
        if self.total_bits() < 1:
            raise ValueError("Q requires at least one total bit")

    def total_bits(self) -> int:
        return self.int_bits + self.frac_bits

    def from_bits(self, raw: int) -> "FixedValue":
        """Construct a value from packed two's-complement bits."""
        from .values import FixedValue
        return FixedValue(self, raw)

    @classmethod
    def from_int(cls, x: int) -> "FixedValue":
        """Infer the smallest signed, zero-fraction format containing ``x``."""
        if not isinstance(x, int):
            raise TypeError(f"Q.from_int expects int, got {type(x).__name__}")
        if x < 0:
            magnitude = abs(x)
            int_bits = max(2, (magnitude - 1).bit_length() + 1)
            raw = (1 << int_bits) + x
        else:
            int_bits = max(2, x.bit_length() + 1)
            raw = x
        return cls(int_bits, 0).from_bits(raw)

    def from_float(self, x: float) -> "FixedValue":
        """Quantize a number using ties-to-even rounding and saturation."""
        if not isinstance(x, (int, float)):
            raise TypeError(f"Q.from_float expects int or float, got {type(x).__name__}")
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError(f"Q.from_float expects a finite number, got {x}")
        width = self.total_bits()
        scaled = int(round(x * (1 << self.frac_bits)))
        scaled = min(max(scaled, -(1 << (width - 1))), (1 << (width - 1)) - 1)
        return self.from_bits(scaled & ((1 << width) - 1))

    def to_spec(self, name, ctx):
        return ctx.fresh_real(name)

    def __repr__(self) -> str:
        return f"Q<{self.int_bits},{self.frac_bits}>"

    __str__ = __repr__


@dataclass(frozen=True)
class UQ(DataType):
    """Unsigned fixed-point descriptor."""

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

    def from_bits(self, raw: int) -> "FixedValue":
        """Construct a value from packed unsigned bits."""
        from .values import FixedValue
        return FixedValue(self, raw)

    @classmethod
    def from_int(cls, x: int) -> "FixedValue":
        """Infer the smallest unsigned, zero-fraction format containing ``x``."""
        if not isinstance(x, int):
            raise TypeError(f"UQ.from_int expects int, got {type(x).__name__}")
        if x < 0:
            raise ValueError(f"UQ.from_int expects a non-negative integer, got {x}")
        return cls(max(1, x.bit_length()), 0).from_bits(x)

    def from_float(self, x: float) -> "FixedValue":
        """Quantize a number using ties-to-even rounding and saturation."""
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

    def __repr__(self) -> str:
        return f"UQ<{self.int_bits},{self.frac_bits}>"

    __str__ = __repr__


class _FloatDescriptor(DataType):
    sign_bits: ClassVar[int] = 1
    exponent_bits: ClassVar[int]
    mantissa_bits: ClassVar[int]
    exponent_bias: ClassVar[int]
    zero_code: ClassVar[int] = 0
    sub_code: ClassVar[int] = 0
    inf_code: ClassVar[int | None] = None
    nan_code: ClassVar[int | None] = None
    nan_mantissa: ClassVar[int | None] = None
    raw_bits: ClassVar[int | None] = None
    spec_name: ClassVar[str]
    display_name: ClassVar[str]

    def total_bits(self) -> int:
        return self.sign_bits + self.exponent_bits + self.mantissa_bits

    def _packed_bits(self) -> int:
        return self.total_bits() if self.raw_bits is None else self.raw_bits

    def from_bits(self, raw: int) -> "FloatValue":
        """Construct a floating-point value from its packed encoding."""
        from .values import FloatValue
        return FloatValue(self, raw)

    def _validate_raw(self, raw: int) -> None:
        if not isinstance(raw, int):
            raise TypeError(
                f"{type(self).__name__} expects packed bits as int, "
                f"got {type(raw).__name__}"
            )
        if not (0 <= raw < (1 << self._packed_bits())):
            raise ValueError(
                f"{type(self).__name__} packed bits must fit in "
                f"{self._packed_bits()} bits, got {raw}"
            )

    def from_fields(self, sign: int, exponent: int, mantissa: int) -> "FloatValue":
        if self.sign_bits:
            if not isinstance(sign, int) or sign not in (0, 1):
                raise ValueError(f"{type(self).__name__} sign must be 0 or 1, got {sign}")
        elif sign != 0:
            raise ValueError(f"{type(self).__name__} has no sign bit")
        if not isinstance(exponent, int) or not (0 <= exponent < (1 << self.exponent_bits)):
            raise ValueError(f"{type(self).__name__} exponent out of range: {exponent}")
        if not isinstance(mantissa, int) or not (0 <= mantissa < (1 << self.mantissa_bits)):
            raise ValueError(f"{type(self).__name__} mantissa out of range: {mantissa}")
        return self.from_bits(
            (sign << (self.exponent_bits + self.mantissa_bits))
            | (exponent << self.mantissa_bits)
            | mantissa
        )

    def _fields(self, raw: int) -> tuple[int, int, int]:
        sign = (
            (raw >> (self.exponent_bits + self.mantissa_bits)) & 1
            if self.sign_bits else 0
        )
        exponent = (raw >> self.mantissa_bits) & ((1 << self.exponent_bits) - 1)
        mantissa = raw & ((1 << self.mantissa_bits) - 1)
        return sign, exponent, mantissa

    def _is_nan(self, raw: int) -> bool:
        sign, exponent, mantissa = self._fields(raw)
        if isinstance(self, E5M2FNUZ):
            return raw == 0x80
        if self.nan_code is None:
            return False
        if self.nan_mantissa is not None:
            return exponent == self.nan_code and mantissa == self.nan_mantissa
        return exponent == self.nan_code and mantissa != 0

    def _is_inf(self, raw: int) -> bool:
        _, exponent, mantissa = self._fields(raw)
        return self.inf_code is not None and exponent == self.inf_code and mantissa == 0

    def _is_zero(self, raw: int) -> bool:
        if isinstance(self, E5M2FNUZ):
            return raw == 0
        _, exponent, mantissa = self._fields(raw)
        return exponent == self.zero_code and mantissa == 0

    def _to_python(self, raw: int) -> float:
        sign, exponent, mantissa = self._fields(raw)
        if self._is_inf(raw):
            return float("-inf") if sign else float("inf")
        if self._is_nan(raw):
            return float("nan")
        multiplier = -1.0 if sign else 1.0
        if exponent == self.sub_code:
            fraction = mantissa / (2 ** self.mantissa_bits)
            return float(multiplier * fraction * (2 ** (1 - self.exponent_bias)))
        fraction = 1.0 + mantissa / (2 ** self.mantissa_bits)
        return float(multiplier * fraction * (2 ** (exponent - self.exponent_bias)))

    def _spec_type(self):
        if self.spec_name == "fp32":
            from ..spec.custom_specs.fp32 import fp32
            return fp32
        if self.spec_name == "fp16":
            from ..spec.custom_specs.fp16 import fp16
            return fp16
        if self.spec_name == "bf16":
            from ..spec.custom_specs.bf16 import bf16
            return bf16
        if self.spec_name == "e4m3fn":
            from ..spec.custom_specs.e4m3fn import e4m3fn
            return e4m3fn
        if self.spec_name == "ue4m3":
            from ..spec.custom_specs.ue4m3 import ue4m3
            return ue4m3
        if self.spec_name == "e5m2":
            from ..spec.custom_specs.e5m2 import e5m2
            return e5m2
        if self.spec_name == "e5m2fnuz":
            from ..spec.custom_specs.e5m2fnuz import e5m2fnuz
            return e5m2fnuz
        if self.spec_name == "e2m1":
            from ..spec.custom_specs.e2m1 import e2m1
            return e2m1
        raise AssertionError(self.spec_name)

    def to_spec(self, name, ctx):
        return self._spec_type().fresh(name, ctx)

    def _value_to_spec(self, value: "FloatValue", ctx):
        spec_type = self._spec_type()
        if value.is_inf:
            return spec_type.ninf(ctx) if value.sign else spec_type.inf(ctx)
        if value.is_nan:
            return spec_type.nan(ctx)
        if value.is_zero:
            if self.sign_bits and value.sign and not isinstance(self, E5M2FNUZ):
                return spec_type.nzero(ctx)
            return spec_type.zero(ctx)

        kwargs = {
            "value": ctx.real_val(value.to_python()),
            "exponent": ctx.real_val(value.exponent),
            "mantissa": ctx.real_val(value.mantissa),
            "is_norm": ctx.bool_val(value.is_norm),
            "is_sub": ctx.bool_val(value.is_sub),
            "is_zero": ctx.bool_val(False),
        }
        if self.sign_bits:
            kwargs["sign"] = ctx.real_val(value.sign)
        if self.inf_code is not None:
            kwargs["is_inf"] = ctx.bool_val(False)
        if self.nan_code is not None:
            kwargs["is_nan"] = ctx.bool_val(False)
        return spec_type(**kwargs)

    def random_value(self, rng: random.Random) -> "FloatValue":
        return self.from_bits(rng.getrandbits(self._packed_bits()))

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
                rng.getrandbits(self.sign_bits) if self.sign_bits else 0,
                shared_exponent + rng.getrandbits(unshared_bits),
                rng.getrandbits(self.mantissa_bits),
            )

        def generate_shared_exponent():
            nonlocal shared_exponent
            shared_exponent = rng.getrandbits(shared_exponent_bits) << unshared_bits
            return shared_exponent

        return generate, generate_shared_exponent

    def Zero(self) -> "FloatValue":
        return self.from_fields(0, self.zero_code, 0)

    def nZero(self) -> "FloatValue":
        if not self.sign_bits or isinstance(self, E5M2FNUZ):
            raise AttributeError(f"{type(self).__name__} has no negative zero")
        return self.from_fields(1, self.zero_code, 0)

    def Inf(self) -> "FloatValue":
        if self.inf_code is None:
            raise AttributeError(f"{type(self).__name__} has no infinity")
        return self.from_fields(0, self.inf_code, 0)

    def nInf(self) -> "FloatValue":
        if self.inf_code is None:
            raise AttributeError(f"{type(self).__name__} has no infinity")
        return self.from_fields(1, self.inf_code, 0)

    def NaN(self, payload: int | None = None) -> "FloatValue":
        if isinstance(self, E5M2FNUZ):
            if payload is not None:
                raise ValueError("E5M2FNUZ has a single NaN encoding")
            return self.from_bits(0x80)
        if self.nan_code is None:
            raise AttributeError(f"{type(self).__name__} has no NaN")
        if self.nan_mantissa is not None:
            if payload is not None:
                raise ValueError(f"{type(self).__name__} has a fixed NaN encoding")
            payload = self.nan_mantissa
        elif payload is None:
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
        return self.display_name

    __str__ = __repr__


@dataclass(frozen=True, repr=False)
class Float16(_FloatDescriptor):
    exponent_bits: ClassVar[int] = 5
    mantissa_bits: ClassVar[int] = 10
    exponent_bias: ClassVar[int] = 15
    inf_code: ClassVar[int] = 31
    nan_code: ClassVar[int] = 31
    spec_name: ClassVar[str] = "fp16"
    display_name: ClassVar[str] = "Float<16>"


@dataclass(frozen=True, repr=False)
class Float32(_FloatDescriptor):
    exponent_bits: ClassVar[int] = 8
    mantissa_bits: ClassVar[int] = 23
    exponent_bias: ClassVar[int] = 127
    inf_code: ClassVar[int] = 255
    nan_code: ClassVar[int] = 255
    spec_name: ClassVar[str] = "fp32"
    display_name: ClassVar[str] = "Float<32>"


@dataclass(frozen=True, repr=False)
class BFloat16(_FloatDescriptor):
    exponent_bits: ClassVar[int] = 8
    mantissa_bits: ClassVar[int] = 7
    exponent_bias: ClassVar[int] = 127
    inf_code: ClassVar[int] = 255
    nan_code: ClassVar[int] = 255
    spec_name: ClassVar[str] = "bf16"
    display_name: ClassVar[str] = "BFloat<16>"

    # Preserve the established bfloat factory's positional field order.
    def from_fields(self, sign: int, mantissa: int, exponent: int) -> "FloatValue":
        return super().from_fields(sign, exponent, mantissa)

    def Zero(self) -> "FloatValue":
        return self.from_fields(0, 0, self.zero_code)

    def nZero(self) -> "FloatValue":
        return self.from_fields(1, 0, self.zero_code)

    def Inf(self) -> "FloatValue":
        return self.from_fields(0, 0, self.inf_code)

    def nInf(self) -> "FloatValue":
        return self.from_fields(1, 0, self.inf_code)

    def NaN(self, payload: int | None = None) -> "FloatValue":
        if payload is None:
            payload = 1 << (self.mantissa_bits - 1)
        if not isinstance(payload, int):
            raise TypeError(f"NaN payload must be int, got {type(payload).__name__}")
        if not (1 <= payload < (1 << self.mantissa_bits)):
            raise ValueError(
                f"NaN payload must fit in {self.mantissa_bits} mantissa bits "
                f"and be non-zero, got {payload}"
            )
        return self.from_fields(0, payload, self.nan_code)

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
                rng.getrandbits(1),
                rng.getrandbits(self.mantissa_bits),
                shared_exponent + rng.getrandbits(unshared_bits),
            )

        def generate_shared_exponent():
            nonlocal shared_exponent
            shared_exponent = rng.getrandbits(shared_exponent_bits) << unshared_bits
            return shared_exponent

        return generate, generate_shared_exponent


@dataclass(frozen=True, repr=False)
class E4M3FN(_FloatDescriptor):
    exponent_bits: ClassVar[int] = 4
    mantissa_bits: ClassVar[int] = 3
    exponent_bias: ClassVar[int] = 7
    nan_code: ClassVar[int] = 15
    nan_mantissa: ClassVar[int] = 7
    max_finite_code: ClassVar[int] = 15
    max_finite_mantissa: ClassVar[int] = 6
    spec_name: ClassVar[str] = "e4m3fn"
    display_name: ClassVar[str] = "E4M3FN<8>"


@dataclass(frozen=True, repr=False)
class UE4M3(_FloatDescriptor):
    sign_bits: ClassVar[int] = 0
    exponent_bits: ClassVar[int] = 4
    mantissa_bits: ClassVar[int] = 3
    exponent_bias: ClassVar[int] = 7
    nan_code: ClassVar[int] = 15
    nan_mantissa: ClassVar[int] = 7
    max_finite_code: ClassVar[int] = 15
    max_finite_mantissa: ClassVar[int] = 6
    min_subnormal: ClassVar[float] = 2 ** -9
    min_normal: ClassVar[float] = 2 ** -6
    max_finite: ClassVar[float] = 448.0
    raw_bits: ClassVar[int] = 7
    spec_name: ClassVar[str] = "ue4m3"
    display_name: ClassVar[str] = "UE4M3<8>"

    def total_bits(self) -> int:
        return 8

    def from_fields(self, exponent: int, mantissa: int) -> "FloatValue":
        return super().from_fields(0, exponent, mantissa)

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
                shared_exponent + rng.getrandbits(unshared_bits),
                rng.getrandbits(self.mantissa_bits),
            )

        def generate_shared_exponent():
            nonlocal shared_exponent
            shared_exponent = rng.getrandbits(shared_exponent_bits) << unshared_bits
            return shared_exponent

        return generate, generate_shared_exponent

    def Zero(self) -> "FloatValue":
        return self.from_fields(self.zero_code, 0)

    def NaN(self, payload: int | None = None) -> "FloatValue":
        if payload is not None:
            raise ValueError("UE4M3 has a fixed NaN encoding")
        return self.from_fields(self.nan_code, self.nan_mantissa)


@dataclass(frozen=True, repr=False)
class E5M2(_FloatDescriptor):
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
    spec_name: ClassVar[str] = "e5m2"
    display_name: ClassVar[str] = "E5M2<8>"


@dataclass(frozen=True, repr=False)
class E5M2FNUZ(_FloatDescriptor):
    exponent_bits: ClassVar[int] = 5
    mantissa_bits: ClassVar[int] = 2
    exponent_bias: ClassVar[int] = 16
    nan_code: ClassVar[int] = 0
    nan_mantissa: ClassVar[int] = 0
    max_finite_code: ClassVar[int] = 31
    max_finite_mantissa: ClassVar[int] = 3
    spec_name: ClassVar[str] = "e5m2fnuz"
    display_name: ClassVar[str] = "E5M2FNUZ<8>"


@dataclass(frozen=True, repr=False)
class E2M1(_FloatDescriptor):
    exponent_bits: ClassVar[int] = 2
    mantissa_bits: ClassVar[int] = 1
    exponent_bias: ClassVar[int] = 1
    max_finite_code: ClassVar[int] = 3
    max_finite_mantissa: ClassVar[int] = 1
    spec_name: ClassVar[str] = "e2m1"
    display_name: ClassVar[str] = "E2M1<4>"


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

    def from_values(self, *values: "RuntimeValue") -> "TupleValue":
        """Construct a tuple value from concrete component values."""
        from .values import TupleValue
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


__all__ = [
    "DataType", "Bool", "Q", "UQ", "Float16", "Float32", "BFloat16",
    "E4M3FN", "UE4M3", "E5M2", "E5M2FNUZ", "E2M1", "Tuple",
]
