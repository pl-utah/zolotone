from __future__ import annotations
from pprint import pprint

from egglog import *

from .datatypes import Math, MathBool


def rewrite_rules():
    from ..spec.spec_ast import RealLit, RealVar, BoolLit, BoolVar, If, Pow
    
    a = RealVar("a")
    b = RealVar("b")
    c = RealVar("c")
    d = RealVar("d")
    x = RealVar("x")
    
    bool_var = BoolVar("p")
    bool_var_q = BoolVar("q")
    bool_var_r = BoolVar("r")
    
    zero = RealLit(0)
    one = RealLit(1)
    minus_one = RealLit(-1)
    two = RealLit(2)
    
    true = BoolLit(True)
    false = BoolLit(False)
    
    return [
        # Associativity
        ("assoc_1", (a + (b + c)).eq((a + b) + c)),
        ("assoc_2", ((a + b) + c).eq(a + (b + c))),
        ("assoc_3", (a * (b * c)).eq((a * b) * c)),
        ("assoc_4", ((a * b) * c).eq(a * (b * c))),
        
        # Commutativity
        ("comm_1", (a + b).eq(b + a)),
        ("comm_2", (a * b).eq(b * a)),
        
        ("dist_1", ((a * b) + (a * c)).eq(a * (b + c))),
        ("dist_2", ((b * a) + (c * a)).eq(a * (b + c))),
        ("dist_3", (a * (b + c)).eq((a * b) + (a * c))),
        ("dist_4", (a * (b + c)).eq((b * a) + (c * a))),
        
        # General pow rules
        ("pow_1", (a ** zero).eq(one)),
        ("pow_2", (a ** one).eq(a)),
        
        # Rules with Pow(2, x)
        ("exp2_2", one.eq(two ** zero)),
        ("exp2_4", two.eq(two ** one)),
        ("exp2_5", (two ** (a + b)).eq((two ** a) * (two ** b))),  # Sound, but different bases may not be supported
        ("exp2_6", ((two ** a) * (two ** b)).eq(two ** (a + b))),
        ("exp2_7", ((two ** x) * (two ** (-x))).eq(one)),          # Sound for non-negative base
        
        # Rules with (-1) ** x
        ("pow_neg_1", one.eq(minus_one ** zero)),
        ("pow_neg_2", minus_one.eq(minus_one ** one)),
        
        # Rules with Pow(x, 2)
        ("square_1", (a ** two).eq(a * a)),
        ("square_2", (a * a).eq(a ** two)),
        
         # Rules with negation of addition
        ("neg_1", (-(a + b)).eq((-a) + (-b))),
        ("neg_2", ((-a) + (-b)).eq(-(a + b))),
        ("neg_3", (-(a + (-b))).eq((-a) + b)),
        ("neg_4", ((-a) + b).eq(-(a + (-b)))),
        
        # Rules with negation of multiplication
        ("neg_mul_1", (-(a * b)).eq((-a) * b)),
        ("neg_mul_2", (-(a * b)).eq(a * (-b))),
        ("neg_mul_3", ((-a) * (-b)).eq(a * b)),
        ("neg_mul_4", (-(a * (-b))).eq(a * b)),
        
        # Rules with max operations
        ("max_1", a.max(b).eq(b.max(a))),
        ("max_2", (a.max(b).max(c)).eq(a.max(c).max(b))),
        ("max_3", (a.max(a)).eq(a)),
        
        # Negation/constants
        ("const_1", (-(-x)).eq(x)),
        ("const_2", (x + (-x)).eq(zero)),
        ("const_3", ((-x) + x).eq(zero)),
        ("const_4", (x * one).eq(x)),
        ("const_5", (one * x).eq(x)),
        ("const_6", (x * zero).eq(zero)),
        ("const_7", (zero * x).eq(zero)),
        ("const_8", (x + zero).eq(x)),
        ("const_9", (zero + x).eq(x)),
        ("const_10", (- zero).eq(zero)),
        ("const_11", zero.eq(-zero)),
        
        # Equality
        ("eq_1", (a.eq(a)).eq(true)),
        ("eq_2", (bool_var.eq(bool_var)).eq(true)),
        
        # Boolean simplification
        # Constants
        ("bool_not_1", (~true).eq(false)),
        ("bool_not_2", (~false).eq(true)),
        ("bool_and_1", (bool_var & true).eq(bool_var)),
        ("bool_and_2", (true & bool_var).eq(bool_var)),
        ("bool_and_3", (bool_var & false).eq(false)),
        ("bool_and_4", (false & bool_var).eq(false)),
        
        ("bool_not_3", (~~bool_var).eq(bool_var)),
        ("bool_and_5", (bool_var & bool_var).eq(bool_var)),
        ("bool_and_6", (bool_var & ~bool_var).eq(false)),
        ("bool_and_7", ((~bool_var) & bool_var).eq(false)),
        
        ("bool_and_8", (bool_var & bool_var_q).eq(bool_var_q & bool_var)),
        ("bool_and_9", (bool_var & (bool_var_q & bool_var_r)).eq((bool_var & bool_var_q) & bool_var_r)),
        ("bool_and_10", ((bool_var & bool_var_q) & bool_var_r).eq(bool_var & (bool_var_q & bool_var_r))),
        
        ("bool_or_1", (bool_var | false).eq(bool_var)),
        ("bool_or_2", (false | bool_var).eq(bool_var)),
        ("bool_or_3", (bool_var | true).eq(true)),
        ("bool_or_4", (true | bool_var).eq(true)),
        
        ("bool_or_5", (bool_var | bool_var).eq(bool_var)),
        ("bool_or_6", (bool_var | ~bool_var).eq(true)),
        ("bool_or_7", ((~bool_var) | bool_var).eq(true)),
        
        ("bool_or_8", (bool_var | bool_var_q).eq(bool_var_q | bool_var)),
        ("bool_or_9", (bool_var | (bool_var_q | bool_var_r)).eq((bool_var | bool_var_q) | bool_var_r)),
        ("bool_or_10", ((bool_var | bool_var_q) | bool_var_r).eq(bool_var | (bool_var_q | bool_var_r))),
        
        ("bool_eq_1", (bool_var.eq(true)).eq(bool_var)),
        ("bool_eq_2", (true.eq(bool_var)).eq(bool_var)),
        ("bool_eq_3", (bool_var.eq(false)).eq(~bool_var)),
        ("bool_eq_4", (false.eq(bool_var)).eq(~bool_var)),
        ("bool_eq_5", ((~bool_var).eq(true)).eq(~bool_var)),
        ("bool_eq_6", ((~bool_var).eq(false)).eq(bool_var)),
        ("bool_eq_7",  ((bool_var & bool_var_q) | ((~bool_var) & (~bool_var_q))).eq(bool_var.eq(bool_var_q))),
        ("bool_eq_15", (~(bool_var.eq(bool_var_q))).eq((bool_var & ~bool_var_q) | ((~bool_var) & bool_var_q))),
        
        ("bool_demorgan_1", (~(bool_var & bool_var_q)).eq((~bool_var) | (~bool_var_q))),
        ("bool_demorgan_2", ((~bool_var) | (~bool_var_q)).eq(~(bool_var & bool_var_q))),
        ("bool_demorgan_3", (~(bool_var | bool_var_q)).eq((~bool_var) & (~bool_var_q))),
        ("bool_demorgan_4", ((~bool_var) & (~bool_var_q)).eq(~(bool_var | bool_var_q))),
        
        ("bool_absorb_1", (bool_var | (bool_var & bool_var_q)).eq(bool_var)),
        ("bool_absorb_2", (bool_var & (bool_var | bool_var_q)).eq(bool_var)),
        
        ("bool_imp_1", ((~bool_var) | bool_var_q).eq((bool_var & ~bool_var_q).eq(false))),
        ("bool_imp_2", ((bool_var & ~bool_var_q).eq(false)).eq((~bool_var) | bool_var_q)),
        ("bool_imp_3", (bool_var & ((~bool_var) | bool_var_q)).eq(bool_var & bool_var_q)),
        ("bool_imp_4", (((~bool_var) | bool_var_q) & bool_var).eq(bool_var & bool_var_q)),
        ("bool_case_1", ((bool_var | bool_var_q) & ~bool_var).eq(bool_var_q & ~bool_var)),
        ("bool_case_2", ((bool_var | bool_var_q) & ~bool_var_q).eq(bool_var & ~bool_var_q)),
        
        # If statements
        ("if_1", If(true, a, b).eq(a)),
        ("if_2", If(false, a, b).eq(b)),
        ("if_3", If(bool_var, a, a).eq(a)),
        ("if_4", (one - If(bool_var, one, zero)).eq(If(~bool_var, one, zero))),
        ("if_5", (one - If(bool_var, one, zero)).eq(If(bool_var, zero, one))),
        ("if_6", (If(bool_var, a, b).eq(a)).eq(bool_var | b.eq(a))),
        ("if_7", (If(bool_var, a, b).eq(b)).eq((~bool_var) | a.eq(b))),
        ("if_8", (If(bool_var, a, b).ne(a)).eq((~bool_var) & b.ne(a))),
        ("if_9", (If(bool_var, a, b).ne(b)).eq(bool_var & a.ne(b))),
    ]


def constant_rules(fold_base_two_powers: bool = True):
    m, n = vars_("m n", BigRat)
    a, b = vars_("a b", Math)
    cond = var("cond", MathBool)
    tru = var("tru", Math)
    fls = var("fls", Math)
    zero = Math.Num(BigRat(0, 1))
    one = Math.Num(BigRat(1, 1))
    minus_one = Math.Num(BigRat(-1, 1))
    two = Math.Num(BigRat(2, 1))

    a_is_bit = MathBool.Or(Math.Eq(a, zero), Math.Eq(a, one))
    b_is_bit = MathBool.Or(Math.Eq(b, zero), Math.Eq(b, one))
    bit_result = var("bit_result", Math)
    bit_result_is_bit = MathBool.Or(
        Math.Eq(bit_result, zero),
        Math.Eq(bit_result, one),
    )

    and_conditional = Math.If(
        MathBool.And(Math.Eq(a, one), Math.Eq(b, one)),
        one,
        zero,
    )
    and_product = Math.Mul(a, b)
    and_min = Math.Min(a, b)

    or_conditional = Math.If(
        MathBool.Or(Math.Eq(a, one), Math.Eq(b, one)),
        one,
        zero,
    )
    or_polynomial = Math.Add(
        Math.Add(a, b),
        Math.Neg(Math.Mul(a, b)),
    )
    or_max = Math.Max(a, b)

    xor_conditional = Math.If(Math.NotEq(a, b), one, zero)
    xor_max_product = Math.Add(
        Math.Max(a, b),
        Math.Neg(Math.Mul(a, b)),
    )
    xor_polynomial = Math.Add(
        Math.Add(a, b),
        Math.Neg(Math.Mul(Math.Mul(two, a), b)),
    )

    neg_conditional = Math.If(Math.Eq(a, one), zero, one)
    neg_difference = Math.Add(one, Math.Neg(a))
    neg_zero_conditional = Math.If(Math.Eq(a, zero), one, zero)
    neg_not_zero_conditional = Math.If(Math.NotEq(a, zero), zero, one)
    neg_not_one_conditional = Math.If(Math.NotEq(a, one), one, zero)

    def sign_multiplier(value):
        return Math.If(Math.Eq(value, one), minus_one, one)

    rules = [
        # Equivalent bit-operator representations share e-classes only when
        # their real-valued operands are known to be bits. Requiring the
        # canonical conditional to exist avoids generating every operator for
        # every bit-valued expression.
        rule(
            eq(bit_result).to(and_conditional),
            eq(a_is_bit).to(MathBool.True_()),
            eq(b_is_bit).to(MathBool.True_()),
        ).then(
            union(bit_result).with_(and_product),
            union(bit_result).with_(and_min),
            union(bit_result_is_bit).with_(MathBool.True_()),
        ),
        rule(
            eq(bit_result).to(or_conditional),
            eq(a_is_bit).to(MathBool.True_()),
            eq(b_is_bit).to(MathBool.True_()),
        ).then(
            union(bit_result).with_(or_polynomial),
            union(bit_result).with_(or_max),
            union(bit_result_is_bit).with_(MathBool.True_()),
        ),
        rule(
            eq(bit_result).to(xor_conditional),
            eq(a_is_bit).to(MathBool.True_()),
            eq(b_is_bit).to(MathBool.True_()),
        ).then(
            union(bit_result).with_(xor_max_product),
            union(bit_result).with_(xor_polynomial),
            union(bit_result_is_bit).with_(MathBool.True_()),
            union(sign_multiplier(bit_result)).with_(
                Math.Mul(sign_multiplier(a), sign_multiplier(b))
            ),
        ),
        rule(
            eq(bit_result).to(neg_conditional),
            eq(a_is_bit).to(MathBool.True_()),
        ).then(
            union(bit_result).with_(neg_difference),
            union(bit_result).with_(neg_zero_conditional),
            union(bit_result).with_(neg_not_zero_conditional),
            union(bit_result).with_(neg_not_one_conditional),
            union(bit_result_is_bit).with_(MathBool.True_()),
        ),
        # Reflect a proven object-language equality into egglog's native
        # equality so conditional assumptions can merge their operands.
        rule(eq(Math.Eq(a, b)).to(MathBool.True_())).then(union(a).with_(b)),
        # boolean
        rewrite(Math.Eq(Math.Num(m), Math.Num(n))).to(MathBool.True_(), eq(m).to(n)),
        rewrite(Math.Eq(Math.Num(m), Math.Num(n))).to(MathBool.False_(), ne(m).to(n)),
        rewrite(Math.NotEq(Math.Num(m), Math.Num(n))).to(MathBool.True_(), ne(m).to(n)),
        rewrite(Math.NotEq(Math.Num(m), Math.Num(n))).to(MathBool.False_(), eq(m).to(n)),
        # arithmetic
        rewrite(Math.Num(m)).to(Math.Neg(Math.Num(-m))),
        rewrite(Math.Add(Math.Num(m), Math.Num(n))).to(Math.Num(m + n)),
        rewrite(Math.Neg(Math.Num(m))).to(Math.Num(-m)),
        rewrite(Math.Mul(Math.Num(m), Math.Num(n))).to(Math.Num(m * n)),
        rewrite(Math.Pow(Math.Num(m), Math.Num(BigRat(2, 1)))).to(Math.Num(m * m)),
        rewrite(Math.Pow(Math.Num(BigRat(-1, 1)), Math.Num(m))).to(Math.Num(BigRat(-1, 1) ** m), eq(m.denom).to(1)),
    ]
    if fold_base_two_powers:
        # Proof e-graphs disable this fold so dyadic scales remain visible to
        # symbolic exponent-combination rules.
        rules.append(
            rewrite(
                Math.Pow(Math.Num(BigRat(2, 1)), Math.Num(m))
            ).to(
                Math.Num(BigRat(2, 1) ** m),
                eq(m.denom).to(1),
            )
        )
    return rules


# This lowering is particularly for rules, because we need to walk expressions and replace Var with var(name, Math) - rewrite syntax
def _lower_expr(node: SpecNode) -> Expr:
    from ..spec.spec_ast import (
        Abs,
        Add,
        And,
        BoolEq,
        BoolLit,
        BoolVar,
        Eq,
        Ge,
        Gt,
        If,
        Le,
        Lt,
        Max,
        Min,
        Mul,
        Neg,
        Not,
        NotEq,
        Or,
        Pow,
        RealLit,
        RealVar,
        Sub,
    )
    
    if isinstance(node, RealVar):
        return var(node.name, Math)
    if isinstance(node, RealLit):
        return node.to_egglog()
    if isinstance(node, Add):
        return Math.Add(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, Sub):
        return Math.Add(_lower_expr(node.lhs), Math.Neg(_lower_expr(node.rhs)))
    if isinstance(node, Mul):
        return Math.Mul(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, Neg):
        return Math.Neg(_lower_expr(node.value))
    if isinstance(node, Abs):
        return Math.Abs(_lower_expr(node.value))
    if isinstance(node, Pow):
        return Math.Pow(_lower_expr(node.base), _lower_expr(node.exponent))
    if isinstance(node, Max):
        return Math.Max(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, Min):
        return Math.Min(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, If):
        return Math.If(
            _lower_expr(node.cond),
            _lower_expr(node.on_true),
            _lower_expr(node.on_false),
        )
    if isinstance(node, BoolLit):
        return node.to_egglog()
    if isinstance(node, BoolVar):
        return var(node.name, MathBool)
    if isinstance(node, Eq):
        return Math.Eq(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, NotEq):
        return Math.NotEq(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, Lt):
        return Math.Lt(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, Le):
        return Math.Le(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, Gt):
        return Math.Gt(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, Ge):
        return Math.Ge(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, Not):
        return MathBool.Not(_lower_expr(node.value))
    if isinstance(node, BoolEq):
        return MathBool.Eq(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, And):
        return MathBool.And(_lower_expr(node.lhs), _lower_expr(node.rhs))
    if isinstance(node, Or):
        return MathBool.Or(_lower_expr(node.lhs), _lower_expr(node.rhs))
    raise TypeError(f"Unsupported expression node: {type(node).__name__}")


def lower_rule(rule: Eq):
    from ..spec.spec_ast import Eq, RealExpr, children
    
    lhs, rhs = children(rule)
    lowered_lhs = _lower_expr(lhs)
    lowered_rhs = _lower_expr(rhs)
    return rewrite(lowered_lhs).to(lowered_rhs)


def lower_rules(rules):
    from ..spec.spec_ast import Eq, BoolEq
    
    lowered_rules = []
    for name, rule in rules:
        if not isinstance(rule, Eq) and not isinstance(rule, BoolEq):
            raise TypeError(f"Expected Eq or BoolEq rule for {name}, got {type(rule).__name__}")
        lowered_rules.append(lower_rule(rule))
    return lowered_rules


def check_rules(rules, z3_timeout_ms: int = 10000):
    from ..spec.spec_ast import Eq, BoolEq, children
    from ..spec.spec_context import SpecContext
    from ..smt import z3_check_eq, dreal_check_eq
    
    results = {}
    for name, rule in rules:
        if not isinstance(rule, Eq) and not isinstance(rule, BoolEq):
            raise TypeError(f"Expected Eq or BoolEq rule for {name}, got {type(rule).__name__}")
        lhs, rhs = children(rule)
        ctx = SpecContext(name)
        ctx.check(lhs.eq(rhs))
        report_z3 = z3_check_eq(ctx, timeout_ms=z3_timeout_ms)
        report_dreal = dreal_check_eq(ctx, precision=0.001)
        results[name] = {
            "z3_status": report_z3["status"],
            "dreal_status": report_dreal["status"],
        }
    return results


def load_rules(
    egraph: EGraph,
    fold_base_two_powers: bool = True,
) -> None:
    rewrites = rewrite_rules()
    constants = constant_rules(
        fold_base_two_powers=fold_base_two_powers,
    )

    rules = lower_rules(rewrites) + constants
    egraph.register(*rules)
