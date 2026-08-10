from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..spec_ast import BoolExpr, BoolLit, FPExpr, If, RealExpr, RealLit
from .fp32 import sign_multiplier


def _implies(lhs: BoolExpr, rhs: BoolExpr) -> BoolExpr:
    return (~lhs) | rhs


@dataclass(frozen=True)
class e5m2fnuz(FPExpr):
    """Symbolic AMD E5M2FNUZ value (bias 16, unsigned zero)."""

    exponent_bits: ClassVar[int] = 5
    mantissa_bits: ClassVar[int] = 2
    exponent_bias: ClassVar[int] = 16

    value: RealExpr
    sign: RealExpr
    exponent: RealExpr
    mantissa: RealExpr
    is_norm: BoolExpr
    is_sub: BoolExpr
    is_zero: BoolExpr
    is_nan: BoolExpr

    @classmethod
    def fresh(cls, name: str, ctx) -> "e5m2fnuz":
        name = ctx.fresh_name(name)
        sign = ctx.fresh_real(f"{name}_sign")
        exponent = ctx.fresh_real(f"{name}_exponent")
        mantissa = ctx.fresh_real(f"{name}_mantissa")
        is_norm = ctx.fresh_bool(f"{name}_is_norm")
        is_sub = ctx.fresh_bool(f"{name}_is_sub")
        is_zero = ctx.fresh_bool(f"{name}_is_zero")
        is_nan = ctx.fresh_bool(f"{name}_is_nan")

        zero = ctx.zero()
        one = ctx.one()
        two = ctx.two()
        max_exponent = ctx.real_val(31)
        max_mantissa = ctx.real_val(3)
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
            If(is_sub, subnormal_value, If(is_zero, zero, ctx.fresh_real("special"))),
        )
        out = cls(
            value=value,
            sign=sign,
            exponent=exponent,
            mantissa=mantissa,
            is_norm=is_norm,
            is_sub=is_sub,
            is_zero=is_zero,
            is_nan=is_nan,
        )

        ctx.assume(sign.eq(zero) | sign.eq(one))
        ctx.assume((exponent >= zero) & (exponent <= max_exponent))
        ctx.assume((mantissa >= zero) & (mantissa <= max_mantissa))
        out._assume_exclusive_classification(ctx)
        ctx.assume(_implies(is_norm, exponent >= one))
        ctx.assume(_implies(is_sub, exponent.eq(zero) & (mantissa >= one)))
        ctx.assume(
            _implies(
                is_zero,
                sign.eq(zero) & exponent.eq(zero) & mantissa.eq(zero),
            )
        )
        ctx.assume(
            _implies(
                is_nan,
                sign.eq(one) & exponent.eq(zero) & mantissa.eq(zero),
            )
        )
        return out

    @classmethod
    def encode(cls, value: RealExpr, ctx) -> "e5m2fnuz":
        """Round a real using RNE, unsigned underflow, and finite saturation."""

        if not isinstance(value, RealExpr):
            raise TypeError(
                f"e5m2fnuz.encode value must be RealExpr, got {type(value).__name__}"
            )

        zero = ctx.zero()
        one = ctx.one()
        two = ctx.two()
        magnitude = abs(value)
        negative_sign = If(value < zero, one, zero)
        exponent = ctx.fresh_real("encoded_e5m2fnuz_exponent")
        mantissa = ctx.fresh_real("encoded_e5m2fnuz_mantissa")
        max_exponent = ctx.real_val(31)
        max_mantissa = ctx.real_val(3)
        mantissa_bits = ctx.real_val(cls.mantissa_bits)
        exponent_bias = ctx.real_val(cls.exponent_bias)
        smallest_normal = two ** (one - exponent_bias)
        smallest_subnormal = two ** (one - exponent_bias - mantissa_bits)
        zero_boundary = smallest_subnormal * two ** ctx.real_val(-1)
        max_finite = ctx.real_val(57344)

        is_zero = magnitude <= zero_boundary
        is_sub = (magnitude > zero_boundary) & (magnitude < smallest_normal)
        is_norm = magnitude >= smallest_normal
        is_nan = ctx.false()
        sign = If(is_zero, zero, negative_sign)
        signed = sign_multiplier(ctx, sign)
        normal_magnitude = (
            (one + mantissa * two ** (-mantissa_bits))
            * two ** (exponent - exponent_bias)
        )
        subnormal_magnitude = (
            mantissa
            * two ** (-mantissa_bits)
            * two ** (one - exponent_bias)
        )
        clamped_magnitude = If(magnitude > max_finite, max_finite, magnitude)
        encoded_value = If(is_zero, zero, signed * clamped_magnitude)
        out = cls(
            value=encoded_value,
            sign=sign,
            exponent=exponent,
            mantissa=mantissa,
            is_norm=is_norm,
            is_sub=is_sub,
            is_zero=is_zero,
            is_nan=is_nan,
        )

        ctx.assume((exponent >= zero) & (exponent <= max_exponent))
        ctx.assume((mantissa >= zero) & (mantissa <= max_mantissa))
        ctx.assume(_implies(is_zero, exponent.eq(zero) & mantissa.eq(zero)))
        ctx.assume(_implies(is_sub, (mantissa >= one) & (mantissa <= max_mantissa)))
        ctx.assume(
            _implies(
                is_norm & (magnitude > max_finite),
                exponent.eq(max_exponent) & mantissa.eq(max_mantissa),
            )
        )
        ctx.assume(
            _implies(
                is_norm & (magnitude <= max_finite),
                magnitude.eq(normal_magnitude),
            )
        )
        ctx.assume(_implies(is_sub, magnitude.eq(subnormal_magnitude)))
        return out

    @classmethod
    def nan(cls, ctx) -> "e5m2fnuz":
        return cls(
            value=ctx.fresh_real("special"),
            sign=RealLit(1),
            exponent=RealLit(0),
            mantissa=RealLit(0),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(False),
            is_nan=BoolLit(True),
        )

    @classmethod
    def zero(cls, ctx) -> "e5m2fnuz":
        return cls(
            value=ctx.zero(),
            sign=RealLit(0),
            exponent=RealLit(0),
            mantissa=RealLit(0),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(True),
            is_nan=BoolLit(False),
        )

    def observables_for_classification(self, classification: str):
        if classification in {"norm", "sub"}:
            return (self.value,)
        if classification in {"zero", "nan"}:
            return (BoolLit(True),)
        raise ValueError(f"Unknown e5m2fnuz classification {classification!r}")

    def decode(self):
        return (
            self.value,
            self.sign,
            self.exponent,
            self.mantissa,
            self.is_norm,
            self.is_sub,
            self.is_zero,
            self.is_nan,
        )

    def classification_flags(self):
        return {
            "norm": self.is_norm,
            "sub": self.is_sub,
            "zero": self.is_zero,
            "nan": self.is_nan,
        }

    @property
    def is_finite(self):
        return self.is_norm | self.is_sub | self.is_zero

    @property
    def is_pzero(self):
        return self.is_zero

    @property
    def is_nzero(self):
        return BoolLit(False)
