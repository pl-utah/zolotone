from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction
import math
import sys
from typing import Any

from ..egglog import *

import z3
import dreal


class SpecNode:
    def __reduce__(self):
        return type(self), tuple(getattr(self, field.name) for field in fields(self))

    def to_egglog(self):
        raise NotImplementedError
    
    def to_z3(self):
        raise NotImplementedError
    
    def to_dreal(self):
        raise NotImplementedError
    
    def identical(self, other: object) -> bool:
        return identical_nodes(self, other)
    
    def constant_fold(self) -> "SpecNode":
        args = children(self)
        if args == ():
            return self
        
        folded_args = tuple(arg.constant_fold() for arg in args)
        
        # Try shortcut
        shortcut = _shortcut_fold(self, folded_args)
        if shortcut is not None:
            return shortcut.constant_fold()
        
        # Else, general case
        output_type = _literal_type(self)
        fold = self.fold()
        if all(isinstance(arg, (RealLit, BoolLit)) for arg in folded_args):
            folded_value = fold(*(arg.value for arg in folded_args))
            if folded_value is not None and _can_constant_fold_literal(output_type, folded_value):
                # Successfully folded
                return output_type(folded_value)
        
        # Nothing got folded
        if all(old is new for old, new in zip(args, folded_args)):
            return self
        # Partially folded
        return type(self)(*folded_args)
    
    def fold(self):
        raise NotImplementedError


class RealExpr(SpecNode):
    @staticmethod
    def _coerce_real_expr(value):
        if isinstance(value, RealExpr):
            return value
        raise TypeError(f"Expected RealExpr, got {type(value).__name__}")
    
    def __add__(self, other: "RealExpr") -> "RealExpr":
        return Add(self, other)
    
    def __iadd__(self, other: "RealExpr") -> "RealExpr":
        return self + other
    
    def __sub__(self, other: "RealExpr") -> "RealExpr":
        return Sub(self, other)
    
    def __isub__(self, other: "RealExpr") -> "RealExpr":
        return self - other
    
    def __mul__(self, other: "RealExpr") -> "RealExpr":
        return Mul(self, other)
    
    def __imul__(self, other: "RealExpr") -> "RealExpr":
        return self * other
    
    def __pow__(self, other: "RealExpr", modulo=None) -> "RealExpr":
        if modulo is not None:
            raise NotImplementedError("pow(..., modulo) is not supported for spec AST")
        return Pow(self, other)
    
    def __ipow__(self, other: "RealExpr") -> "RealExpr":
        return self ** other
    
    def __neg__(self) -> "RealExpr":
        return Neg(self)
    
    def __pos__(self) -> "RealExpr":
        return self
    
    def __abs__(self) -> "RealExpr":
        return Abs(self)
    
    def __lt__(self, other: "RealExpr") -> "BoolExpr":
        return Lt(self, other)
        
    def __le__(self, other: "RealExpr") -> "BoolExpr":
        return Le(self, other)
    
    def __gt__(self, other: "RealExpr") -> "BoolExpr":
        return Gt(self, other)
    
    def __ge__(self, other: "RealExpr") -> "BoolExpr":
        return Ge(self, other)
    
    def eq(self, other: "RealExpr") -> "BoolExpr":
        return Eq(self, other)
    
    def ne(self, other: "RealExpr") -> "BoolExpr":
        return NotEq(self, other)
    
    def max(self, other: "RealExpr") -> "RealExpr":
        return Max(self, other)
    
    def min(self, other: "RealExpr") -> "RealExpr":
        return Min(self, other)


class FPExpr(SpecNode, ABC):
    value: "RealExpr"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        declares_value = any(
            base is not FPExpr
            and "value" in base.__dict__.get("__annotations__", {})
            for base in cls.__mro__
        )
        if not declares_value:
            raise TypeError(
                f"{cls.__name__} must declare a value: RealExpr field"
            )

    def __post_init__(self) -> None:
        if not is_dataclass(self):
            raise TypeError(
                f"{type(self).__name__} must be a dataclass FPExpr"
            )
        if "value" not in {field.name for field in fields(self)}:
            raise TypeError(
                f"{type(self).__name__} must define a value dataclass field"
            )
        if not isinstance(self.value, RealExpr):
            raise TypeError(
                f"{type(self).__name__}.value must be RealExpr, got "
                f"{type(self.value).__name__}"
            )

    @classmethod
    @abstractmethod
    def fresh(cls, name: str, ctx) -> "FPExpr":
        """Create an unconstrained, well-formed value of this FP format."""

    @classmethod
    @abstractmethod
    def encode(cls, value: "RealExpr", ctx) -> "FPExpr":
        """Encode a mathematical real value into this FP format."""

    @abstractmethod
    def decode(self) -> tuple[SpecNode, ...]:
        """Return this FP value's mathematical value and format fields."""

    def constant_fold(self) -> "FPExpr":
        """Fold each scalar field of this structured expression."""

        old_fields = self.decode()
        folded_fields = tuple(field.constant_fold() for field in old_fields)
        if all(old is new for old, new in zip(old_fields, folded_fields)):
            return self
        return type(self)(*folded_fields)

    @property
    @abstractmethod
    def is_finite(self) -> "BoolExpr":
        """Return whether this FP value has a finite classification."""

    @abstractmethod
    def classification_flags(self) -> dict[str, "BoolExpr"]:
        """Return the mutually exclusive classification predicates."""

    def _assume_exclusive_classification(self, ctx) -> None:
        """Require exactly one of this format's classifications to hold."""

        flags = tuple(self.classification_flags().values())
        if not flags:
            raise ValueError(
                f"{type(self).__name__}.classification_flags() must not be empty"
            )

        at_least_one = flags[0]
        for flag in flags[1:]:
            at_least_one = at_least_one | flag
        ctx.assume(at_least_one)

        for idx, lhs in enumerate(flags):
            for rhs in flags[idx + 1:]:
                ctx.assume((~lhs) | (~rhs))

    @abstractmethod
    def observables_for_classification(
        self,
        classification: str,
    ) -> tuple[SpecNode, ...]:
        """Return observables when the classification is already known."""

    # Internal implementation used by Cases to combine two FP branches.
    @classmethod
    def _select(
        cls,
        condition: "BoolExpr",
        on_true: "FPExpr",
        on_false: "FPExpr",
    ) -> "FPExpr":
        """Select two same-format FP expressions field by field for Cases."""

        BoolExpr._coerce_bool_expr(condition)
        if type(on_true) is not type(on_false):
            raise TypeError(
                "Cases FP branches must have the same type, got "
                f"{type(on_true).__name__} and {type(on_false).__name__}"
            )
        if not isinstance(on_true, cls) or not is_dataclass(on_true):
            raise TypeError(
                f"Cases FP branches must be dataclass {cls.__name__} values"
            )

        selected_fields: dict[str, object] = {}
        for field in fields(on_true):
            if not field.init:
                continue

            true_value = getattr(on_true, field.name)
            false_value = getattr(on_false, field.name)

            if isinstance(true_value, RealExpr):
                if not isinstance(false_value, RealExpr):
                    raise TypeError(f"Mismatched FP field types for {field.name}")
                selected_value = If(condition, true_value, false_value)
            elif isinstance(true_value, BoolExpr):
                if not isinstance(false_value, BoolExpr):
                    raise TypeError(f"Mismatched FP field types for {field.name}")
                selected_value = (
                    (condition & true_value)
                    | ((~condition) & false_value)
                )
            elif isinstance(true_value, FPExpr):
                if type(true_value) is not type(false_value):
                    raise TypeError(f"Mismatched FP field types for {field.name}")
                selected_value = type(true_value)._select(
                    condition,
                    true_value,
                    false_value,
                )
            else:
                if true_value != false_value:
                    raise TypeError(
                        f"Cannot select differing FP metadata field {field.name}"
                    )
                selected_value = true_value

            selected_fields[field.name] = selected_value

        return type(on_true)(**selected_fields)


def special_encoding(value: Any, ctx) -> Any:
    """Give each spec collection an independent nonfinite FP value.

    An FPExpr's real ``value`` is shared for finite inputs, but is not
    semantically defined for infinities or NaNs.  Materialize that undefined
    projection as a fresh real in the current collection while preserving all
    classification fields.  Tuple-shaped spec inputs are handled recursively.
    """

    if isinstance(value, tuple):
        return tuple(special_encoding(item, ctx) for item in value)
    if not isinstance(value, FPExpr):
        return value
    if not is_dataclass(value):
        raise TypeError(
            "special_encoding requires dataclass FPExpr values, got "
            f"{type(value).__name__}"
        )

    raw_value = getattr(value, "value", None)
    if not isinstance(raw_value, RealExpr):
        raise TypeError(
            f"{type(value).__name__} must define a RealExpr value field "
            "for special_encoding"
        )

    updates: dict[str, object] = {
        "value": If(
            value.is_finite,
            raw_value,
            ctx.fresh_real(f"{type(value).__name__}_special"),
        )
    }

    for field in fields(value):
        if not field.init or field.name == "value":
            continue
        field_value = getattr(value, field.name)
        encoded_value = special_encoding(field_value, ctx)
        if encoded_value is not field_value:
            updates[field.name] = encoded_value

    return replace(value, **updates)
    

class BoolExpr(SpecNode):
    @staticmethod
    def _coerce_bool_expr(value):
        if isinstance(value, BoolExpr):
            return value
        raise TypeError(f"Expected BoolExpr, got {type(value).__name__}")

    
    def __bool__(self) -> bool:
        raise TypeError("Spec BoolExpr cannot be used as a Python bool")
    
    def __invert__(self) -> "BoolExpr":
        return Not(self)
    
    def __and__(self, other: "BoolExpr") -> "BoolExpr":
        return self.and_(other)
    
    def __rand__(self, other: "BoolExpr") -> "BoolExpr":
        return And(other, self)
    
    def __or__(self, other: "BoolExpr") -> "BoolExpr":
        return self.or_(other)
    
    def __ror__(self, other: "BoolExpr") -> "BoolExpr":
        return Or(other, self)
    
    def eq(self, other: "BoolExpr") -> "BoolExpr":
        return BoolEq(self, other)

    # this should be BoolNe
    def ne(self, other: "BoolExpr") -> "BoolExpr":
        return BoolEq(~self, other)
    
    def or_(self, other: "BoolExpr") -> "BoolExpr":
        return Or(self, other)
    
    def and_(self, other: "BoolExpr") -> "BoolExpr":
        return And(self, other)

    def implies(self, other: "BoolExpr") -> "BoolExpr":
        return (~self) | other


@dataclass(frozen=True)
class RealVar(RealExpr):
    name: str
    
    def to_egglog(self):
        return Math.Var(self.name)
    
    def to_z3(self, env):
        key = ("real", self.name)
        if key not in env:
            env[key] = z3.Real(self.name)
        return env[key]
    
    def to_dreal(self, env):
        key = ("real", self.name)
        if key not in env:
            env[key] = dreal.Variable(self.name)
        return env[key]
    
    def __str__(self):
        return f"real({self.name})"


@dataclass(frozen=True)
class BoolVar(BoolExpr):
    name: str
    
    def to_egglog(self):
        return MathBool.Var(self.name)
    
    def to_z3(self, env):
        key = ("bool", self.name)
        if key not in env:
            env[key] = z3.Bool(self.name)
        return env[key]
    
    def to_dreal(self, env):
        key = ("bool", self.name)
        if key not in env:
            env[key] = dreal.Variable(self.name, dreal.Variable.Bool)
        return env[key]
    
    def __str__(self):
        return f"bool({self.name})"


@dataclass(frozen=True)
class RealLit(RealExpr):
    value: float | int

    def __post_init__(self):
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("non-finite RealLit is not supported")
    
    def _as_fraction(self) -> Fraction:
        if isinstance(self.value, float):
            if not math.isfinite(self.value):
                raise ValueError("non-finite RealLit is not supported")
            return Fraction(self.value)
        return Fraction(self.value)
    
    def to_egglog(self):
        ratio = self._as_fraction()
        return _fraction_to_egglog(ratio)
    
    def to_z3(self, env):
        ratio = self._as_fraction()
        return z3.RealVal(f"{ratio.numerator}/{ratio.denominator}")
    
    def to_dreal(self, env):
        return dreal.Expression(self.value)
    
    def __str__(self):
        return str(self.value)


@dataclass(frozen=True)
class BoolLit(BoolExpr):
    value: bool
    
    def to_egglog(self):
        return MathBool.True_() if self.value else MathBool.False_()
    
    def to_z3(self, env):
        return z3.BoolVal(self.value)
    
    def to_dreal(self, env):
        return dreal.Formula.TRUE() if self.value else dreal.Formula.FALSE()
    
    def __str__(self):
        return "true" if self.value else "false"


@dataclass(frozen=True)
class Add(RealExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.Add(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return self.lhs.to_z3(env=env) + self.rhs.to_z3(env=env)
    
    def to_dreal(self, env):
        return self.lhs.to_dreal(env=env) + self.rhs.to_dreal(env=env)
    
    def fold(self):
        return lambda lhs, rhs: lhs + rhs
    
    def __str__(self):
        return f"({self.lhs} + {self.rhs})"


@dataclass(frozen=True)
class Sub(RealExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.Add(self.lhs.to_egglog(), Math.Neg(self.rhs.to_egglog()))
    
    def to_z3(self, env):
        return self.lhs.to_z3(env=env) - self.rhs.to_z3(env=env)

    def to_dreal(self, env):
        return self.lhs.to_dreal(env=env) - self.rhs.to_dreal(env=env)

    def fold(self):
        return lambda lhs, rhs: lhs - rhs

    def __str__(self):
        return f"({self.lhs} - {self.rhs})"


@dataclass(frozen=True)
class Mul(RealExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.Mul(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return self.lhs.to_z3(env=env) * self.rhs.to_z3(env=env)

    def to_dreal(self, env):
        return self.lhs.to_dreal(env=env) * self.rhs.to_dreal(env=env)

    def fold(self):
        return lambda lhs, rhs: lhs * rhs

    def __str__(self):
        return f"({self.lhs} * {self.rhs})"


@dataclass(frozen=True)
class Neg(RealExpr):
    value: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.value)
    
    def to_egglog(self):
        return Math.Neg(self.value.to_egglog())
    
    def to_z3(self, env):
        return -self.value.to_z3(env=env)

    def to_dreal(self, env):
        return -self.value.to_dreal(env=env)

    def fold(self):
        return lambda value: -value

    def __str__(self):
        return f"(-{self.value})"


@dataclass(frozen=True)
class Abs(RealExpr):
    value: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.value)
    
    def to_egglog(self):
        return Math.Abs(self.value.to_egglog())
    
    def to_z3(self, env):
        return z3.Abs(self.value.to_z3(env=env))

    def to_dreal(self, env):
        return abs(self.value.to_dreal(env=env))

    def fold(self):
        return lambda value: abs(value)

    def __str__(self):
        return f"abs({self.value})"


@dataclass(frozen=True)
class Pow(RealExpr):
    base: RealExpr
    exponent: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.base)
        RealExpr._coerce_real_expr(self.exponent)

    def to_egglog(self):
        return Math.Pow(self.base.to_egglog(), self.exponent.to_egglog())

    def to_z3(self, env):
        return self.base.to_z3(env=env) ** self.exponent.to_z3(env=env)

    def to_dreal(self, env):
        return self.base.to_dreal(env=env) ** self.exponent.to_dreal(env=env)

    def fold(self):
        return _folded_pow_value

    def __str__(self):
        return f"({self.base} ** {self.exponent})"


@dataclass(frozen=True)
class Max(RealExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.Max(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        lhs = self.lhs.to_z3(env=env)
        rhs = self.rhs.to_z3(env=env)
        return z3.If(lhs >= rhs, lhs, rhs)

    def to_dreal(self, env):
        return dreal.Max(self.lhs.to_dreal(env=env), self.rhs.to_dreal(env=env))

    def fold(self):
        return max

    def __str__(self):
        return f"max({self.lhs}, {self.rhs})"


@dataclass(frozen=True)
class Min(RealExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.Min(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        lhs = self.lhs.to_z3(env=env)
        rhs = self.rhs.to_z3(env=env)
        return z3.If(lhs <= rhs, lhs, rhs)

    def to_dreal(self, env):
        return dreal.Min(self.lhs.to_dreal(env=env), self.rhs.to_dreal(env=env))

    def fold(self):
        return min

    def __str__(self):
        return f"min({self.lhs}, {self.rhs})"


@dataclass(frozen=True)
class If(RealExpr):
    cond: BoolExpr
    on_true: RealExpr
    on_false: RealExpr

    def __new__(cls, cond, on_true, on_false):
        BoolExpr._coerce_bool_expr(cond)
        if isinstance(on_true, FPExpr) or isinstance(on_false, FPExpr):
            raise TypeError(
                "If does not support FPExpr branches; use exhaustive Cases instead"
            )

        true_is_bool = isinstance(on_true, BoolExpr)
        false_is_bool = isinstance(on_false, BoolExpr)
        if true_is_bool or false_is_bool:
            if not (true_is_bool and false_is_bool):
                raise TypeError(
                    "If branches must both be BoolExpr or RealExpr, got "
                    f"{type(on_true).__name__} and "
                    f"{type(on_false).__name__}"
                )
            return (cond & on_true) | ((~cond) & on_false)
        return super().__new__(cls)

    def __post_init__(self):
        BoolExpr._coerce_bool_expr(self.cond)
        RealExpr._coerce_real_expr(self.on_true)
        RealExpr._coerce_real_expr(self.on_false)
    
    def to_egglog(self):
        return Math.If(
            self.cond.to_egglog(),
            self.on_true.to_egglog(),
            self.on_false.to_egglog(),
        )
    
    def to_z3(self, env):
        return z3.If(
            self.cond.to_z3(env=env),
            self.on_true.to_z3(env=env),
            self.on_false.to_z3(env=env)
        )

    def to_dreal(self, env):
        return dreal.if_then_else(
            self.cond.to_dreal(env=env),
            self.on_true.to_dreal(env=env),
            self.on_false.to_dreal(env=env),
        )

    def fold(self):
        return lambda cond, on_true, on_false: on_true if cond else on_false

    def __str__(self):
        return f"(if {self.cond} then {self.on_true} else {self.on_false})"


_CaseValue = BoolExpr | RealExpr | FPExpr


def _coerce_case_value(value: object) -> _CaseValue:
    if isinstance(value, (BoolExpr, RealExpr, FPExpr)):
        return value
    raise TypeError(
        "Cases values must be BoolExpr, RealExpr, or FPExpr, got "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True)
class _CaseEntry:
    condition: BoolExpr
    value: _CaseValue


@dataclass(frozen=True)
class _CasePartition:
    entries: tuple[_CaseEntry, ...]
    value: _CaseValue


def case(condition: BoolExpr, value: _CaseValue) -> _CaseEntry:
    """Create an ordered conditional entry for :func:`Cases`."""
    BoolExpr._coerce_bool_expr(condition)
    return _CaseEntry(condition, _coerce_case_value(value))


def _select_case_value(
    condition: BoolExpr,
    on_true: _CaseValue,
    on_false: _CaseValue,
) -> _CaseValue:
    true_is_fp = isinstance(on_true, FPExpr)
    false_is_fp = isinstance(on_false, FPExpr)
    if true_is_fp or false_is_fp:
        if not (true_is_fp and false_is_fp):
            raise TypeError(
                "Cases branches must all be matching FPExpr values, or all "
                "scalar expressions; got "
                f"{type(on_true).__name__} and {type(on_false).__name__}"
            )
        if type(on_true) is not type(on_false):
            raise TypeError(
                "Cases FP branches must have the same type, got "
                f"{type(on_true).__name__} and {type(on_false).__name__}"
            )
        return type(on_true)._select(condition, on_true, on_false)

    return If(condition, on_true, on_false)


def Cases(*entries: _CaseEntry, ctx) -> _CaseValue:
    """Build an ordered, exhaustive scalar or floating-point expression."""
    for entry in entries:
        if not isinstance(entry, _CaseEntry):
            raise TypeError(
                "Cases entries must be created by case(), got "
                f"{type(entry).__name__}"
            )

    if not entries:
        raise ValueError("Cases requires at least one case() entry")
    if not hasattr(ctx, "require"):
        raise TypeError("Cases ctx must be a SpecContext")

    coverage = entries[0].condition
    for entry in entries[1:]:
        coverage = coverage | entry.condition

    result = entries[-1].value
    for entry in reversed(entries[:-1]):
        result = _select_case_value(entry.condition, entry.value, result)
    ctx.require(coverage)
    ctx.case_partitions.append(_CasePartition(tuple(entries), result))
    return result


@dataclass(frozen=True)
class Eq(BoolExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.Eq(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return self.lhs.to_z3(env=env) == self.rhs.to_z3(env=env)

    def to_dreal(self, env):
        return self.lhs.to_dreal(env=env) == self.rhs.to_dreal(env=env)

    def fold(self):
        return lambda lhs, rhs: lhs == rhs

    def __str__(self):
        return f"({self.lhs} == {self.rhs})"


@dataclass(frozen=True)
class NotEq(BoolExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.NotEq(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return self.lhs.to_z3(env=env) != self.rhs.to_z3(env=env)

    def to_dreal(self, env):
        return self.lhs.to_dreal(env=env) != self.rhs.to_dreal(env=env)

    def fold(self):
        return lambda lhs, rhs: lhs != rhs

    def __str__(self):
        return f"({self.lhs} != {self.rhs})"


@dataclass(frozen=True)
class Lt(BoolExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.Lt(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return self.lhs.to_z3(env=env) < self.rhs.to_z3(env=env)

    def to_dreal(self, env):
        return self.lhs.to_dreal(env=env) < self.rhs.to_dreal(env=env)

    def fold(self):
        return lambda lhs, rhs: lhs < rhs

    def __str__(self):
        return f"({self.lhs} < {self.rhs})"


@dataclass(frozen=True)
class Le(BoolExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.Le(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return self.lhs.to_z3(env=env) <= self.rhs.to_z3(env=env)

    def to_dreal(self, env):
        return self.lhs.to_dreal(env=env) <= self.rhs.to_dreal(env=env)

    def fold(self):
        return lambda lhs, rhs: lhs <= rhs

    def __str__(self):
        return f"({self.lhs} <= {self.rhs})"


@dataclass(frozen=True)
class Gt(BoolExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.Gt(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return self.lhs.to_z3(env=env) > self.rhs.to_z3(env=env)

    def to_dreal(self, env):
        return self.lhs.to_dreal(env=env) > self.rhs.to_dreal(env=env)

    def fold(self):
        return lambda lhs, rhs: lhs > rhs

    def __str__(self):
        return f"({self.lhs} > {self.rhs})"


@dataclass(frozen=True)
class Ge(BoolExpr):
    lhs: RealExpr
    rhs: RealExpr

    def __post_init__(self):
        RealExpr._coerce_real_expr(self.lhs)
        RealExpr._coerce_real_expr(self.rhs)
    
    def to_egglog(self):
        return Math.Ge(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return self.lhs.to_z3(env=env) >= self.rhs.to_z3(env=env)

    def to_dreal(self, env):
        return self.lhs.to_dreal(env=env) >= self.rhs.to_dreal(env=env)

    def fold(self):
        return lambda lhs, rhs: lhs >= rhs

    def __str__(self):
        return f"({self.lhs} >= {self.rhs})"


@dataclass(frozen=True)
class BoolEq(BoolExpr):
    lhs: BoolExpr
    rhs: BoolExpr

    def __post_init__(self):
        BoolExpr._coerce_bool_expr(self.lhs)
        BoolExpr._coerce_bool_expr(self.rhs)
    
    def to_egglog(self):
        return MathBool.Eq(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return self.lhs.to_z3(env=env) == self.rhs.to_z3(env=env)

    def to_dreal(self, env):
        lhs = self.lhs.to_dreal(env=env)
        rhs = self.rhs.to_dreal(env=env)
        return dreal.And(dreal.Or(dreal.Not(lhs), rhs), dreal.Or(dreal.Not(rhs), lhs))

    def fold(self):
        return lambda lhs, rhs: lhs == rhs

    def __str__(self):
        return f"({self.lhs} == {self.rhs})"

    
@dataclass(frozen=True)
class Not(BoolExpr):
    value: BoolExpr

    def __post_init__(self):
        BoolExpr._coerce_bool_expr(self.value)
    
    def to_egglog(self):
        return MathBool.Not(self.value.to_egglog())
    
    def to_z3(self, env):
        return z3.Not(self.value.to_z3(env=env))

    def to_dreal(self, env):
        return dreal.Not(self.value.to_dreal(env=env))

    def fold(self):
        return lambda value: not value

    def __str__(self):
        return f"(not {self.value})"

    
@dataclass(frozen=True)
class Or(BoolExpr):
    lhs: BoolExpr
    rhs: BoolExpr

    def __post_init__(self):
        BoolExpr._coerce_bool_expr(self.lhs)
        BoolExpr._coerce_bool_expr(self.rhs)
        
    def to_egglog(self):
        return MathBool.Or(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return z3.Or(self.lhs.to_z3(env=env), self.rhs.to_z3(env=env))

    def to_dreal(self, env):
        return dreal.Or(self.lhs.to_dreal(env=env), self.rhs.to_dreal(env=env))

    def fold(self):
        return lambda lhs, rhs: lhs or rhs

    def __str__(self):
        return f"({self.lhs} or {self.rhs})"


@dataclass(frozen=True)
class And(BoolExpr):
    lhs: BoolExpr
    rhs: BoolExpr

    def __post_init__(self):
        BoolExpr._coerce_bool_expr(self.lhs)
        BoolExpr._coerce_bool_expr(self.rhs)
    
    def to_egglog(self):
        return MathBool.And(self.lhs.to_egglog(), self.rhs.to_egglog())
    
    def to_z3(self, env):
        return z3.And(self.lhs.to_z3(env=env), self.rhs.to_z3(env=env))

    def to_dreal(self, env):
        return dreal.And(self.lhs.to_dreal(env=env), self.rhs.to_dreal(env=env))

    def fold(self):
        return lambda lhs, rhs: lhs and rhs

    def __str__(self):
        return f"({self.lhs} and {self.rhs})"


def _folded_pow_value(base: float | int, exponent: float | int):
    try:
        value = base ** exponent
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    if isinstance(value, complex):
        return None
    return value


_C_LONG_MIN = -sys.maxsize - 1
_C_LONG_MAX = sys.maxsize


def _fits_c_long(value: int) -> bool:
    return _C_LONG_MIN <= value <= _C_LONG_MAX


def _fraction_fits_c_long(value: Fraction) -> bool:
    return _fits_c_long(value.numerator) and _fits_c_long(value.denominator)


def _real_lit_fits_c_long(value: float | int) -> bool:
    try:
        ratio = Fraction(value)
    except (OverflowError, ValueError):
        return False
    return _fraction_fits_c_long(ratio)


def _math_num(numerator: int, denominator: int = 1):
    return Math.Num(BigRat(numerator, denominator))


def _remove_factor_of_two(value: int) -> tuple[int, int]:
    shift = 0
    while value != 0 and value % 2 == 0:
        value //= 2
        shift += 1
    return value, shift


def _fraction_to_egglog(value: Fraction):
    # Keep integral literals compact. For non-integral binary rationals,
    # preserve the power-of-two scale so exponent rewrites can see it.
    if value.denominator == 1 and _fraction_fits_c_long(value):
        return _math_num(value.numerator, value.denominator)

    sign = -1 if value.numerator < 0 else 1
    numerator_odd, numerator_shift = _remove_factor_of_two(abs(value.numerator))
    denominator_odd, denominator_shift = _remove_factor_of_two(value.denominator)
    coefficient = Fraction(sign * numerator_odd, denominator_odd)
    exponent = numerator_shift - denominator_shift

    if not _fraction_fits_c_long(coefficient) or not _fits_c_long(exponent):
        raise OverflowError(f"RealLit {value} cannot be represented with C-long-sized egglog literals")

    if exponent == 0:
        return _math_num(coefficient.numerator, coefficient.denominator)

    scale = Math.Pow(_math_num(2), _math_num(exponent))
    if coefficient == 1:
        return scale
    if coefficient == -1:
        return Math.Neg(scale)
    return Math.Mul(_math_num(coefficient.numerator, coefficient.denominator), scale)


def _can_constant_fold_literal(output_type, value: float | int | bool) -> bool:
    return output_type is not RealLit or _real_lit_fits_c_long(value)


# Substitute exact learned facts, then rebuild and constant-fold.
def substitute_literals(
    node: "SpecNode",
    replacements: dict[SpecNode, SpecNode],
    _seen: set[SpecNode] | None = None,
) -> "SpecNode":
    if _seen is None:  # Avoid loops
        _seen = set()

    replacement = replacements.get(node)
    if replacement is not None:
        if node in _seen:
            return node
        return substitute_literals(replacement, replacements, _seen | {node})

    args = children(node)
    if args == ():
        return node

    substituted_args = tuple(
        substitute_literals(arg, replacements, _seen) for arg in args
    )
    rebuilt = (
        node
        if all(old is new for old, new in zip(args, substituted_args))
        else type(node)(*substituted_args)
    )
    return rebuilt.constant_fold()


def _shortcut_fold(
    node: SpecNode,
    folded_args: tuple[SpecNode, ...],
) -> SpecNode | None:
    if isinstance(node, Eq):
        lhs, rhs = folded_args
        # x == x -> True
        if identical_nodes(lhs, rhs):
            return BoolLit(True)
        return None

    if isinstance(node, BoolEq):
        lhs, rhs = folded_args
        # x == x -> True
        if identical_nodes(lhs, rhs):
            return BoolLit(True)
        # ~x == x -> False
        if _are_complements(lhs, rhs):
            return BoolLit(False)
        # x == True -> x
        # x == False -> ~x
        if isinstance(lhs, BoolLit):
            return rhs if lhs.value else Not(rhs)
        # True == x -> x
        # False == x -> ~x
        if isinstance(rhs, BoolLit):
            return lhs if rhs.value else Not(lhs)
        return None

    if isinstance(node, NotEq):
        lhs, rhs = folded_args
        # x != x -> False
        if identical_nodes(lhs, rhs):
            return BoolLit(False)
        return None

    if isinstance(node, Not):
        value, = folded_args
        # ~(~x) -> x
        if isinstance(value, Not):
            return value.value
        if isinstance(value, BoolLit):
            return BoolLit(not value.value)
        return None
    
    if isinstance(node, If):
        cond, on_true, on_false = folded_args
        # if True then x else y -> x
        # if False then x else y -> y
        if isinstance(cond, BoolLit):
            return on_true if cond.value else on_false
        # If x then y else y -> y
        if identical_nodes(on_true, on_false):
            return on_true
        return None

    if isinstance(node, Mul):
        lhs, rhs = folded_args
        # x * 0 -> 0
        if isinstance(lhs, RealLit) and lhs.value == 0:
            return lhs
        # 0 * x -> 0
        if isinstance(rhs, RealLit) and rhs.value == 0:
            return rhs
        # 1 * x -> x
        if isinstance(lhs, RealLit) and lhs.value == 1:
            return rhs
        # x * 1 -> x
        if isinstance(rhs, RealLit) and rhs.value == 1:
            return lhs
        return None

    if isinstance(node, Add):
        lhs, rhs = folded_args
        # x + 0 -> x
        if isinstance(lhs, RealLit) and lhs.value == 0:
            return rhs
        # 0 + x -> x
        if isinstance(rhs, RealLit) and rhs.value == 0:
            return lhs
        return None

    if isinstance(node, Sub):
        lhs, rhs = folded_args
        # x - x -> 0
        if identical_nodes(lhs, rhs):
            return RealLit(0)
        # x - 0 -> x
        if isinstance(rhs, RealLit) and rhs.value == 0:
            return lhs
        # 0 - x -> -x
        if isinstance(lhs, RealLit) and lhs.value == 0:
            return Neg(rhs).constant_fold()
        return None

    if isinstance(node, And):
        lhs, rhs = folded_args
        # x and x -> x
        if identical_nodes(lhs, rhs):
            return lhs
        # x and ~x -> False
        if _are_complements(lhs, rhs):
            return BoolLit(False)
        # x and True -> x
        # x and False -> False
        if isinstance(lhs, BoolLit):
            return rhs if lhs.value else BoolLit(False)
        # True and x -> x
        # False and x -> False
        if isinstance(rhs, BoolLit):
            return lhs if rhs.value else BoolLit(False)
        return None

    if isinstance(node, Or):
        lhs, rhs = folded_args
        # x or x -> x
        if identical_nodes(lhs, rhs):
            return lhs
        # x or ~x -> True
        if _are_complements(lhs, rhs):
            return BoolLit(True)
        # x or True -> True
        # x or False -> x
        if isinstance(lhs, BoolLit):
            return BoolLit(True) if lhs.value else rhs
        if isinstance(rhs, BoolLit):
            return BoolLit(True) if rhs.value else lhs
        # (common and x) or (common and y) -> common and (x or y)
        factored = _factor_common_conjunction(lhs, rhs)
        if factored is not None:
            return factored
        return None

    return None


def _are_complements(lhs: SpecNode, rhs: SpecNode) -> bool:
    return (
        isinstance(lhs, Not)
        and identical_nodes(lhs.value, rhs)
    ) or (
        isinstance(rhs, Not)
        and identical_nodes(rhs.value, lhs)
    )


def _factor_common_conjunction(lhs: BoolExpr, rhs: BoolExpr) -> BoolExpr | None:
    if not isinstance(lhs, And) or not isinstance(rhs, And):
        return None

    lhs_pairs = ((lhs.lhs, lhs.rhs), (lhs.rhs, lhs.lhs))
    rhs_pairs = ((rhs.lhs, rhs.rhs), (rhs.rhs, rhs.lhs))
    for lhs_common, lhs_rest in lhs_pairs:
        for rhs_common, rhs_rest in rhs_pairs:
            if identical_nodes(lhs_common, rhs_common):
                return And(lhs_common, Or(lhs_rest, rhs_rest))

    return None


def _negate_bool(expr: SpecNode) -> BoolExpr:
    if isinstance(expr, BoolLit):
        return BoolLit(not expr.value)
    if isinstance(expr, Not):
        return expr.value
    if isinstance(expr, BoolExpr):
        return Not(expr)
    raise TypeError(f"Expected BoolExpr, got {type(expr).__name__}")


def _literal_type(node: SpecNode):
    if isinstance(node, BoolExpr):
        return BoolLit
    if isinstance(node, RealExpr):
        return RealLit
    raise TypeError(f"Unsupported node type: {type(node).__name__}")


def children(node: SpecNode) -> tuple[SpecNode, ...]:
    if not isinstance(node, SpecNode) or not is_dataclass(node):
        raise TypeError(f"Unsupported node type: {type(node).__name__}")
    return tuple(
        value
        for field in fields(node)
        if isinstance(value := getattr(node, field.name), SpecNode)
    )


def variables(node: SpecNode) -> set[RealVar | BoolVar]:
    if isinstance(node, (RealVar, BoolVar)):
        return {node}
    vars_ = set()
    for child in children(node):
        vars_.update(variables(child))
    return vars_


def _identical_values(lhs: object, rhs: object) -> bool:
    if isinstance(lhs, SpecNode):
        return isinstance(rhs, SpecNode) and identical_nodes(lhs, rhs)
    if isinstance(lhs, tuple):
        return (
            isinstance(rhs, tuple)
            and len(lhs) == len(rhs)
            and all(_identical_values(l_item, r_item) for l_item, r_item in zip(lhs, rhs))
        )
    return lhs == rhs


def identical_nodes(lhs: SpecNode, rhs: object) -> bool:
    if type(lhs) is not type(rhs):
        return False
    if not is_dataclass(lhs):
        raise TypeError(f"Unsupported node type: {type(lhs).__name__}")
    return all(
        _identical_values(getattr(lhs, field.name), getattr(rhs, field.name))
        for field in fields(lhs)
    )
