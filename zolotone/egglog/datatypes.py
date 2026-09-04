from __future__ import annotations

from egglog import *

class MathBool(Expr):
    @method(egg_fn="BoolNot")
    @classmethod
    def Not(cls, value: MathBool) -> MathBool: ...

    @method(egg_fn="BoolEq")
    @classmethod
    def Eq(cls, lhs: MathBool, rhs: MathBool) -> MathBool: ...

    @method(egg_fn="Or")
    @classmethod
    def Or(cls, lhs: MathBool, rhs: MathBool) -> MathBool: ...

    @method(egg_fn="And")
    @classmethod
    def And(cls, lhs: MathBool, rhs: MathBool) -> MathBool: ...
    
    @method(egg_fn="BoolVar")
    @classmethod
    def Var(cls, name: StringLike) -> MathBool: ...
    
    @method(egg_fn="BoolTrue")
    @classmethod
    def True_(cls) -> MathBool: ...
    
    @method(egg_fn="BoolFalse")
    @classmethod
    def False_(cls) -> MathBool: ...

class Math(Expr):
    @method(egg_fn="Num")
    @classmethod
    def Num(cls, value: BigRatLike) -> Math: ...
    
    @method(egg_fn="Var")
    @classmethod
    def Var(cls, name: StringLike) -> Math: ...
    
    @method(egg_fn="Add")
    @classmethod
    def Add(cls, lhs: Math, rhs: Math) -> Math: ...
    
    @method(egg_fn="Mul")
    @classmethod
    def Mul(cls, lhs: Math, rhs: Math) -> Math: ...
    
    @method(egg_fn="Pow")
    @classmethod
    def Pow(cls, base: Math, exponent: Math) -> Math: ...
    
    @method(egg_fn="Neg")
    @classmethod
    def Neg(cls, expr: Math) -> Math: ...
    
    @method(egg_fn="Abs")
    @classmethod
    def Abs(cls, expr: Math) -> Math: ...
    
    @method(egg_fn="Max")
    @classmethod
    def Max(cls, lhs: Math, rhs: Math) -> Math: ...
    
    @method(egg_fn="Min")
    @classmethod
    def Min(cls, lhs: Math, rhs: Math) -> Math: ...
    
    @method(egg_fn="Lt")
    @classmethod
    def Lt(cls, lhs: Math, rhs: Math) -> MathBool: ...
    
    @method(egg_fn="Le")
    @classmethod
    def Le(cls, lhs: Math, rhs: Math) -> MathBool: ...
    
    @method(egg_fn="Gt")
    @classmethod
    def Gt(cls, lhs: Math, rhs: Math) -> MathBool: ...
    
    @method(egg_fn="Ge")
    @classmethod
    def Ge(cls, lhs: Math, rhs: Math) -> MathBool: ...
    
    @method(egg_fn="NotEq")
    @classmethod
    def NotEq(cls, lhs: Math, rhs: Math) -> MathBool: ...
    
    @method(egg_fn="Eq")
    @classmethod
    def Eq(cls, lhs: Math, rhs: Math) -> MathBool: ...
    
    @method(egg_fn="If")
    @classmethod
    def If(cls, cnd: MathBool, tru: Math, fls: Math) -> Math: ...
