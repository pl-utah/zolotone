from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..spec_ast import BoolExpr, BoolLit, FPExpr, If, RealExpr, RealLit
from .fp32 import sign_multiplier

@dataclass(frozen=True)
class e5m2(FPExpr):
    """Symbolic OCP E5M2 value with signed zeros and infinities."""

    exponent_bits: ClassVar[int] = 5
    mantissa_bits: ClassVar[int] = 2
    exponent_bias: ClassVar[int] = 15

    value: RealExpr
    sign: RealExpr
    exponent: RealExpr
    mantissa: RealExpr
    is_norm: BoolExpr
    is_sub: BoolExpr
    is_zero: BoolExpr
    is_inf: BoolExpr
    is_nan: BoolExpr

    @classmethod
    def fresh(cls, name: str, ctx) -> "e5m2":
        name = ctx.fresh_name(name)
        sign = ctx.fresh_real(f"{name}_sign")
        exponent = ctx.fresh_real(f"{name}_exponent")
        mantissa = ctx.fresh_real(f"{name}_mantissa")
        is_norm = ctx.fresh_bool(f"{name}_is_norm")
        is_sub = ctx.fresh_bool(f"{name}_is_sub")
        is_zero = ctx.fresh_bool(f"{name}_is_zero")
        is_inf = ctx.fresh_bool(f"{name}_is_inf")
        is_nan = ctx.fresh_bool(f"{name}_is_nan")

        zero = ctx.zero()
        one = ctx.one()
        two = ctx.two()
        max_exponent = ctx.real_val((1 << cls.exponent_bits) - 1)
        max_mantissa = ctx.real_val((1 << cls.mantissa_bits) - 1)
        signed = sign_multiplier(ctx, sign)
        normal_value = (
            signed
            * (one + mantissa * two ** (-ctx.real_val(cls.mantissa_bits)))
            * two ** (exponent - ctx.real_val(cls.exponent_bias))
        )
        subnormal_value = (
            signed
            * mantissa
            * two ** (-ctx.real_val(cls.mantissa_bits))
            * two ** (one - ctx.real_val(cls.exponent_bias))
        )
        value = If(
            is_norm,
            normal_value,
            If(
                is_sub,
                subnormal_value,
                If(is_zero, zero, ctx.fresh_real("special")),
            ),
        )
        out = cls(
            value=value,
            sign=sign,
            exponent=exponent,
            mantissa=mantissa,
            is_norm=is_norm,
            is_sub=is_sub,
            is_zero=is_zero,
            is_inf=is_inf,
            is_nan=is_nan,
        )

        ctx.assume(sign.eq(zero) | sign.eq(one))
        ctx.assume((exponent >= zero) & (exponent <= max_exponent))
        ctx.assume((mantissa >= zero) & (mantissa <= max_mantissa))
        out._assume_exclusive_classification(ctx)
        ctx.assume(
            is_norm.implies((exponent >= one) & (exponent < max_exponent))
        )
        ctx.assume(is_sub.implies(exponent.eq(zero) & (mantissa >= one)))
        ctx.assume(is_zero.implies(exponent.eq(zero) & mantissa.eq(zero)))
        ctx.assume(is_inf.implies(exponent.eq(max_exponent) & mantissa.eq(zero)))
        ctx.assume(is_nan.implies(exponent.eq(max_exponent) & (mantissa >= one)))
        return out

    @classmethod
    def encode(cls, value: RealExpr, ctx) -> "e5m2":
        """Round a real value to non-saturating E5M2 using RNE."""

        if not isinstance(value, RealExpr):
            raise TypeError(
                f"e5m2.encode value must be RealExpr, got {type(value).__name__}"
            )

        zero = ctx.zero()
        one = ctx.one()
        two = ctx.two()
        magnitude = abs(value)
        sign = If(value < zero, one, zero)
        exponent = ctx.fresh_real("encoded_e5m2_exponent")
        mantissa = ctx.fresh_real("encoded_e5m2_mantissa")
        max_exponent = ctx.real_val((1 << cls.exponent_bits) - 1)
        max_mantissa = ctx.real_val((1 << cls.mantissa_bits) - 1)
        mantissa_bits = ctx.real_val(cls.mantissa_bits)
        exponent_bias = ctx.real_val(cls.exponent_bias)
        smallest_normal = two ** (one - exponent_bias)
        smallest_subnormal = two ** (one - exponent_bias - mantissa_bits)
        zero_boundary = smallest_subnormal * two ** ctx.real_val(-1)
        greatest_normal = ctx.real_val(57344)

        is_zero = magnitude <= zero_boundary
        is_sub = (magnitude > zero_boundary) & (magnitude < smallest_normal)
        is_norm = (magnitude >= smallest_normal) & (magnitude <= greatest_normal)
        is_inf = magnitude > greatest_normal
        is_nan = ctx.false()
        normal_magnitude = (
            (one + mantissa * two ** (-mantissa_bits))
            * two ** (exponent - exponent_bias)
        )
        subnormal_magnitude = (
            mantissa
            * two ** (-mantissa_bits)
            * two ** (one - exponent_bias)
        )
        encoded_value = If(
            is_norm | is_sub,
            value,
            If(is_zero, zero, ctx.fresh_real("special")),
        )
        out = cls(
            value=encoded_value,
            sign=sign,
            exponent=exponent,
            mantissa=mantissa,
            is_norm=is_norm,
            is_sub=is_sub,
            is_zero=is_zero,
            is_inf=is_inf,
            is_nan=is_nan,
        )

        ctx.assume((exponent >= zero) & (exponent <= max_exponent))
        ctx.assume((mantissa >= zero) & (mantissa <= max_mantissa))
        ctx.assume(is_zero.implies(exponent.eq(zero) & mantissa.eq(zero)))
        ctx.assume(is_sub.implies(exponent.eq(zero) & (mantissa >= one)))
        ctx.assume(
            is_norm.implies((exponent >= one) & (exponent < max_exponent))
        )
        ctx.assume(is_inf.implies(exponent.eq(max_exponent) & mantissa.eq(zero)))
        ctx.assume(is_norm.implies(magnitude.eq(normal_magnitude)))
        ctx.assume(is_sub.implies(magnitude.eq(subnormal_magnitude)))
        return out

    @classmethod
    def nan(cls, ctx) -> "e5m2":
        return cls(
            value=ctx.fresh_real("special"),
            sign=RealLit(0),
            exponent=RealLit(31),
            mantissa=RealLit(2),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(False),
            is_inf=BoolLit(False),
            is_nan=BoolLit(True),
        )

    @classmethod
    def inf(cls, ctx) -> "e5m2":
        return cls(
            value=ctx.fresh_real("special"),
            sign=RealLit(0),
            exponent=RealLit(31),
            mantissa=RealLit(0),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(False),
            is_inf=BoolLit(True),
            is_nan=BoolLit(False),
        )

    @classmethod
    def ninf(cls, ctx) -> "e5m2":
        return cls(
            value=ctx.fresh_real("special"),
            sign=RealLit(1),
            exponent=RealLit(31),
            mantissa=RealLit(0),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(False),
            is_inf=BoolLit(True),
            is_nan=BoolLit(False),
        )

    @classmethod
    def zero(cls, ctx) -> "e5m2":
        return cls(
            value=ctx.zero(),
            sign=RealLit(0),
            exponent=RealLit(0),
            mantissa=RealLit(0),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(True),
            is_inf=BoolLit(False),
            is_nan=BoolLit(False),
        )

    @classmethod
    def nzero(cls, ctx) -> "e5m2":
        return cls(
            value=ctx.zero(),
            sign=RealLit(1),
            exponent=RealLit(0),
            mantissa=RealLit(0),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(True),
            is_inf=BoolLit(False),
            is_nan=BoolLit(False),
        )

    def observables_for_classification(self, classification: str):
        if classification in {"norm", "sub"}:
            return (self.value,)
        if classification in {"zero", "inf"}:
            return (self.sign,)
        if classification == "nan":
            return (BoolLit(True),)
        raise ValueError(f"Unknown e5m2 classification {classification!r}")

    def decode(self):
        return (
            self.value,
            self.sign,
            self.exponent,
            self.mantissa,
            self.is_norm,
            self.is_sub,
            self.is_zero,
            self.is_inf,
            self.is_nan,
        )

    def classification_flags(self):
        return {
            "norm": self.is_norm,
            "sub": self.is_sub,
            "zero": self.is_zero,
            "inf": self.is_inf,
            "nan": self.is_nan,
        }

    @property
    def is_finite(self):
        return self.is_norm | self.is_sub | self.is_zero

    @property
    def is_ninf(self):
        return self.is_inf & self.sign.eq(RealLit(1))

    @property
    def is_pinf(self):
        return self.is_inf & self.sign.eq(RealLit(0))

    @property
    def is_nzero(self):
        return self.is_zero & self.sign.eq(RealLit(1))

    @property
    def is_pzero(self):
        return self.is_zero & self.sign.eq(RealLit(0))
