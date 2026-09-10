from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..spec_ast import BoolExpr, BoolLit, FPExpr, If, RealExpr, RealLit


def sign_multiplier(ctx, sign: RealExpr) -> RealExpr:
    one = ctx.one()
    return If(sign.eq(one), ctx.real_val(-1), one)


@dataclass(frozen=True)
class fp32(FPExpr):
    """A symbolic IEEE-754 binary32 value and its format operations.

    ``value`` is meaningful only for normal, subnormal, and zero values.
    NaN and infinity are represented by classification fields. NaN payload
    and sign are intentionally not part of the observable semantics.
    """
    
    exponent_bits: ClassVar[int] = 8
    mantissa_bits: ClassVar[int] = 23
    exponent_bias: ClassVar[int] = 127

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
    def fresh(cls, name: str, ctx) -> fp32:
        """Create an unconstrained, well-formed fp32 value."""
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
        max_exponent = ctx.real_val((1 << cls.exponent_bits) - 1)
        max_mantissa = ctx.real_val((1 << cls.mantissa_bits) - 1)

        two = ctx.two()
        mantissa_bits = ctx.real_val(cls.mantissa_bits)
        exponent_bias = ctx.real_val(cls.exponent_bias)

        signed_value = sign_multiplier(ctx, sign)
        normal_magnitude = (
            (one + mantissa * (two ** (-mantissa_bits)))
            * (two ** (exponent - exponent_bias))
        )
        normal_value = signed_value * normal_magnitude
        subnormal_magnitude = (
            mantissa
            * (two ** (-mantissa_bits))
            * (two ** (one - exponent_bias))
        )
        subnormal_value = signed_value * subnormal_magnitude
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

        ctx.assume(is_norm.implies(exponent >= one))
        ctx.assume(is_norm.implies(exponent <= max_exponent - one))
        ctx.assume((is_sub | is_zero).implies(exponent.eq(zero)))
        ctx.assume((is_zero | is_inf).implies(mantissa.eq(zero)))
        ctx.assume((is_inf | is_nan).implies(exponent.eq(max_exponent)))
        ctx.assume((is_sub | is_nan).implies(mantissa >= one))

        return out
    
    @classmethod
    def encode(cls, value: RealExpr, ctx) -> fp32:
        if not isinstance(value, RealExpr):
            raise TypeError(
                f"fp32.encode value must be RealExpr, got {type(value).__name__}"
            )
        
        zero = ctx.zero()
        one = ctx.one()
        two = ctx.two()
        mantissa_bits = ctx.real_val(cls.mantissa_bits)
        exponent_bits = ctx.real_val(cls.exponent_bits)
        exponent_bias = ctx.real_val(cls.exponent_bias)
        
        smallest_normal = two ** (one - exponent_bias)
        greatest_normal = (
            (two - two ** (-mantissa_bits))
            * two ** (two ** exponent_bits - two - exponent_bias)
        )
        smallest_subnormal = two ** (one - exponent_bias - mantissa_bits)
        zero_rounding_boundary = smallest_subnormal * (two ** ctx.real_val(-1))
        
        magnitude = abs(value)
        sign = If(value < zero, one, zero)
        
        # Under RNE, values at the midpoint between zero and the smallest
        # subnormal tie to the even encoding (zero). The sign still comes
        # from the nonzero input, so negative underflow produces -0.
        is_zero = magnitude <= zero_rounding_boundary
        is_sub = (
            (magnitude < smallest_normal)
            & (magnitude > zero_rounding_boundary)
        )
        is_norm = (
            (magnitude >= smallest_normal)
            & (magnitude <= greatest_normal)
        )
        is_inf = magnitude > greatest_normal
        is_nan = ctx.false()

        exponent = ctx.fresh_real("encoded_fp32_exponent")
        mantissa = ctx.fresh_real("encoded_fp32_mantissa")
        special = ctx.fresh_real("special")
        encoded_value = If(
            is_norm | is_sub,
            value,
            If(is_zero, zero, special),
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

        max_exponent = ctx.real_val((1 << cls.exponent_bits) - 1)
        max_mantissa = ctx.real_val((1 << cls.mantissa_bits) - 1)
        ctx.assume((exponent >= zero) & (exponent <= max_exponent))
        ctx.assume((mantissa >= zero) & (mantissa <= max_mantissa))
        ctx.assume(is_norm.implies(exponent >= one))
        ctx.assume(is_norm.implies(exponent <= max_exponent - one))
        ctx.assume((is_sub | is_zero).implies(exponent.eq(zero)))
        ctx.assume(is_inf.implies(exponent.eq(max_exponent)))
        ctx.assume((is_zero | is_inf).implies(mantissa.eq(zero)))
        ctx.assume(is_sub.implies(mantissa >= one))

        signed_value = sign_multiplier(ctx, sign)
        normal_value = (
            signed_value
            * (one + mantissa * (two ** (-mantissa_bits)))
            * (two ** (exponent - exponent_bias))
        )
        subnormal_value = (
            signed_value
            * mantissa
            * (two ** (-mantissa_bits))
            * (two ** (one - exponent_bias))
        )
        ctx.assume(is_norm.implies(value.eq(normal_value)))
        ctx.assume(is_sub.implies(value.eq(subnormal_value)))
        return out



    @classmethod
    def nan(cls, ctx) -> fp32:
        return cls(
            value=ctx.fresh_real("special"),
            sign=RealLit(0),
            exponent=RealLit((1 << cls.exponent_bits) - 1),
            mantissa=RealLit(1),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(False),
            is_inf=BoolLit(False),
            is_nan=BoolLit(True),
        )


    @classmethod
    def inf(cls, ctx) -> fp32:
        return cls(
            value=ctx.fresh_real("special"),
            sign=RealLit(0),
            exponent=RealLit((1 << cls.exponent_bits) - 1),
            mantissa=RealLit(0),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(False),
            is_inf=BoolLit(True),
            is_nan=BoolLit(False),
        )

    @classmethod
    def ninf(cls, ctx) -> fp32:
        return cls(
            value=ctx.fresh_real("special"),
            sign=RealLit(1),
            exponent=RealLit((1 << cls.exponent_bits) - 1),
            mantissa=RealLit(0),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(False),
            is_inf=BoolLit(True),
            is_nan=BoolLit(False),
        )

    @classmethod
    def zero(cls, ctx) -> fp32:
        return cls(
            value=RealLit(0),
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
    def nzero(cls, ctx) -> fp32:
        return cls(
            value=RealLit(0),
            sign=RealLit(1),
            exponent=RealLit(0),
            mantissa=RealLit(0),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(True),
            is_inf=BoolLit(False),
            is_nan=BoolLit(False),
        )

    def observables_for_classification(
        self,
        classification: str,
    ) -> tuple[RealExpr | BoolExpr, ...]:
        """Return only the fields observable for a known classification."""

        if classification in {"norm", "sub"}:
            return (self.value,)
        if classification in {"zero", "inf"}:
            return (self.sign,)
        if classification == "nan":
            # NaN has no observable payload in this model. Keep one trivial
            # Boolean query so schedules that start with egglog still receive
            # a non-empty checks list.
            return (BoolLit(True),)
        raise ValueError(f"Unknown fp32 classification {classification!r}")

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

    def classification_flags(self) -> dict[str, BoolExpr]:
        return dict(
            zip(
                ("norm", "sub", "zero", "inf", "nan"),
                (self.is_norm, self.is_sub, self.is_zero, self.is_inf, self.is_nan),
            )
        )

    @property
    def is_finite(self) -> BoolExpr:
        return self.is_norm | self.is_sub | self.is_zero

    @property
    def is_ninf(self) -> BoolExpr:
        return self.is_inf & self.sign.eq(RealLit(1))

    @property
    def is_pinf(self) -> BoolExpr:
        return self.is_inf & self.sign.eq(RealLit(0))

    @property
    def is_nzero(self) -> BoolExpr:
        return self.is_zero & self.sign.eq(RealLit(1))

    @property
    def is_pzero(self) -> BoolExpr:
        return self.is_zero & self.sign.eq(RealLit(0))
