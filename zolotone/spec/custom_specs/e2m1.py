from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..spec_ast import BoolExpr, FPExpr, If, RealExpr, RealLit
from .fp32 import sign_multiplier


def _implies(lhs: BoolExpr, rhs: BoolExpr) -> BoolExpr:
    return (~lhs) | rhs


@dataclass(frozen=True)
class e2m1(FPExpr):
    """Symbolic finite-only E2M1 value with signed zeros."""

    exponent_bits: ClassVar[int] = 2
    mantissa_bits: ClassVar[int] = 1
    exponent_bias: ClassVar[int] = 1

    value: RealExpr
    sign: RealExpr
    exponent: RealExpr
    mantissa: RealExpr
    is_norm: BoolExpr
    is_sub: BoolExpr
    is_zero: BoolExpr

    @classmethod
    def fresh(cls, name: str, ctx) -> "e2m1":
        name = ctx.fresh_name(name)
        sign = ctx.fresh_real(f"{name}_sign")
        exponent = ctx.fresh_real(f"{name}_exponent")
        mantissa = ctx.fresh_real(f"{name}_mantissa")
        is_norm = ctx.fresh_bool(f"{name}_is_norm")
        is_sub = ctx.fresh_bool(f"{name}_is_sub")
        is_zero = ctx.fresh_bool(f"{name}_is_zero")
        zero = ctx.zero()
        one = ctx.one()
        two = ctx.two()
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
        value = If(is_norm, normal_value, If(is_sub, subnormal_value, zero))
        out = cls(
            value=value,
            sign=sign,
            exponent=exponent,
            mantissa=mantissa,
            is_norm=is_norm,
            is_sub=is_sub,
            is_zero=is_zero,
        )

        ctx.assume(sign.eq(zero) | sign.eq(one))
        ctx.assume((exponent >= zero) & (exponent <= ctx.real_val(3)))
        ctx.assume((mantissa >= zero) & (mantissa <= one))
        out._assume_exclusive_classification(ctx)
        ctx.assume(_implies(is_norm, exponent >= one))
        ctx.assume(_implies(is_sub, exponent.eq(zero) & mantissa.eq(one)))
        ctx.assume(_implies(is_zero, exponent.eq(zero) & mantissa.eq(zero)))
        return out

    @classmethod
    def encode(cls, value: RealExpr, ctx) -> "e2m1":
        """Round a real using RNE and saturate finite overflow to signed six."""

        if not isinstance(value, RealExpr):
            raise TypeError(
                f"e2m1.encode value must be RealExpr, got {type(value).__name__}"
            )
        zero = ctx.zero()
        one = ctx.one()
        two = ctx.two()
        magnitude = abs(value)
        sign = If(value < zero, one, zero)
        exponent = ctx.fresh_real("encoded_e2m1_exponent")
        mantissa = ctx.fresh_real("encoded_e2m1_mantissa")
        smallest_normal = ctx.real_val(1)
        smallest_subnormal = ctx.real_val(0.5)
        zero_boundary = ctx.real_val(0.25)
        max_finite = ctx.real_val(6)
        is_zero = magnitude <= zero_boundary
        is_sub = (magnitude > zero_boundary) & (magnitude < smallest_normal)
        is_norm = magnitude >= smallest_normal
        signed = sign_multiplier(ctx, sign)
        normal_magnitude = (
            (one + mantissa * two ** (-ctx.real_val(cls.mantissa_bits)))
            * two ** (exponent - ctx.real_val(cls.exponent_bias))
        )
        subnormal_magnitude = mantissa * smallest_subnormal
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
        )

        ctx.assume((exponent >= zero) & (exponent <= ctx.real_val(3)))
        ctx.assume((mantissa >= zero) & (mantissa <= one))
        ctx.assume(_implies(is_zero, exponent.eq(zero) & mantissa.eq(zero)))
        ctx.assume(_implies(is_sub, mantissa.eq(one)))
        ctx.assume(
            _implies(
                is_norm & (magnitude > max_finite),
                exponent.eq(ctx.real_val(3)) & mantissa.eq(one),
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
    def zero(cls, ctx) -> "e2m1":
        return cls(
            value=ctx.zero(),
            sign=RealLit(0),
            exponent=RealLit(0),
            mantissa=RealLit(0),
            is_norm=ctx.false(),
            is_sub=ctx.false(),
            is_zero=ctx.true(),
        )

    @classmethod
    def nzero(cls, ctx) -> "e2m1":
        return cls(
            value=ctx.zero(),
            sign=RealLit(1),
            exponent=RealLit(0),
            mantissa=RealLit(0),
            is_norm=ctx.false(),
            is_sub=ctx.false(),
            is_zero=ctx.true(),
        )

    def observables_for_classification(self, classification: str):
        if classification in {"norm", "sub"}:
            return (self.value,)
        if classification == "zero":
            return (self.sign,)
        raise ValueError(f"Unknown e2m1 classification {classification!r}")

    def decode(self):
        return (
            self.value,
            self.sign,
            self.exponent,
            self.mantissa,
            self.is_norm,
            self.is_sub,
            self.is_zero,
        )

    def classification_flags(self):
        return {"norm": self.is_norm, "sub": self.is_sub, "zero": self.is_zero}

    @property
    def is_finite(self):
        return self.is_norm | self.is_sub | self.is_zero

    @property
    def is_nzero(self):
        return self.is_zero & self.sign.eq(RealLit(1))

    @property
    def is_pzero(self):
        return self.is_zero & self.sign.eq(RealLit(0))
