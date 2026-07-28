from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..spec_ast import BoolExpr, BoolLit, FPExpr, If, RealExpr, RealLit


def _implies(lhs: BoolExpr, rhs: BoolExpr) -> BoolExpr:
    return (~lhs) | rhs


def sign_multiplier(ctx, sign: RealExpr) -> RealExpr:
    one = ctx.real_val(1)
    return If(sign.eq(one), ctx.real_val(-1), one)


def _assume_exclusive_classification(ctx, fp32_value):
    flags = (fp32_value.is_norm, fp32_value.is_sub, fp32_value.is_zero, fp32_value.is_inf, fp32_value.is_nan)
    
    at_least_one = flags[0]
    for flag in flags[1:]:
        at_least_one = at_least_one | flag
    ctx.assume(at_least_one)

    for i, lhs in enumerate(flags):
        for rhs in flags[i + 1:]:
            ctx.assume((~lhs) | (~rhs))


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
    def _symbolic(cls, name: str, ctx) -> fp32:
        name = ctx.fresh_name(name)
        return cls(
            value=ctx.fresh_real(f"{name}_value"),
            sign=ctx.fresh_real(f"{name}_sign"),
            exponent=ctx.fresh_real(f"{name}_exponent"),
            mantissa=ctx.fresh_real(f"{name}_mantissa"),
            is_norm=ctx.fresh_bool(f"{name}_is_norm"),
            is_sub=ctx.fresh_bool(f"{name}_is_sub"),
            is_zero=ctx.fresh_bool(f"{name}_is_zero"),
            is_inf=ctx.fresh_bool(f"{name}_is_inf"),
            is_nan=ctx.fresh_bool(f"{name}_is_nan"),
        )
    
    @classmethod
    def fresh(cls, name: str, ctx) -> fp32:
        """Create an unconstrained, well-formed fp32 value."""
        out = cls._symbolic(name, ctx)
        
        zero = ctx.real_val(0)
        one = ctx.real_val(1)
        max_exponent = ctx.real_val((1 << cls.exponent_bits) - 1)
        max_mantissa = ctx.real_val((1 << cls.mantissa_bits) - 1)
        
        ctx.assume(out.sign.eq(zero) | out.sign.eq(one))
        ctx.assume((out.exponent >= zero) & (out.exponent <= max_exponent))
        ctx.assume((out.mantissa >= zero) & (out.mantissa <= max_mantissa))
        
        _assume_exclusive_classification(ctx, out)
        
        ctx.assume(_implies(out.is_norm, out.exponent >= one))
        ctx.assume(_implies(out.is_norm, out.exponent <= max_exponent - one))
        ctx.assume(_implies(out.is_sub | out.is_zero, out.exponent.eq(zero)))
        ctx.assume(_implies(out.is_zero | out.is_inf, out.mantissa.eq(zero)))
        ctx.assume(_implies(out.is_inf | out.is_nan, out.exponent.eq(max_exponent)))
        ctx.assume(_implies(out.is_sub | out.is_nan, out.mantissa >= one))

        two = ctx.real_val(2)
        mantissa_bits = ctx.real_val(cls.mantissa_bits)
        exponent_bias = ctx.real_val(cls.exponent_bias)

        signed_value = sign_multiplier(ctx, out.sign)
        normal_value = signed_value * (one + out.mantissa * (two ** (-mantissa_bits))) * (two ** (out.exponent - exponent_bias))
        subnormal_value = signed_value * out.mantissa  * (two ** (-mantissa_bits)) * (two ** (one - exponent_bias))

        ctx.assume(_implies(out.is_norm, out.value.eq(normal_value)))
        ctx.assume(_implies(out.is_sub, out.value.eq(subnormal_value)))
        ctx.assume(_implies(out.is_zero, out.value.eq(zero)))
        
        return out
    
    @classmethod
    def encode(cls, value: RealExpr, ctx) -> fp32:
        if not isinstance(value, RealExpr):
            raise TypeError(
                f"fp32.encode value must be RealExpr, got {type(value).__name__}"
            )
        
        zero = ctx.real_val(0)
        one = ctx.real_val(1)
        two = ctx.real_val(2)
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
        
        out = cls.fresh("encoded_fp32", ctx)
        
        magnitude = ctx.fresh_real("encoded_fp32_magnitude")
        ctx.assume(magnitude.eq(abs(value)))
        ctx.assume(out.sign.eq(If(value < zero, one, zero)))
        
        # Under RNE, values at the midpoint between zero and the smallest
        # subnormal tie to the even encoding (zero). The sign still comes
        # from the nonzero input, so negative underflow produces -0.
        ctx.assume(out.is_zero.eq(magnitude <= zero_rounding_boundary))
        ctx.assume(
            out.is_sub.eq(
                (magnitude < smallest_normal)
                & (magnitude > zero_rounding_boundary)
            )
        )
        ctx.assume(
            out.is_norm.eq(
                (magnitude >= smallest_normal)
                & (magnitude <= greatest_normal)
            )
        )
        ctx.assume(out.is_inf.eq(magnitude > greatest_normal))
        ctx.assume(out.is_nan.eq(ctx.false()))
        
        ctx.assume(_implies(out.is_norm | out.is_sub, out.value.eq(value)))
        ctx.assume(_implies(out.is_zero, out.value.eq(zero)))
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
            value=ctx.real_val(0),
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
            value=ctx.real_val(0),
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

    def constant_fold(self) -> fp32:
        """Fold each scalar field of this structured expression."""

        old_fields = self.decode()
        folded_fields = tuple(field.constant_fold() for field in old_fields)
        if all(old is new for old, new in zip(old_fields, folded_fields)):
            return self
        return type(self)(*folded_fields)

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
