from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
import os
import sys
from typing import Any, Iterable, Sequence

from .spec.spec_ast import *

RivalIR = dict[str, Any]

__all__ = [
    "DEFAULT_MAX_RECTS",
    "MAX_RECTS_ENV",
    "RivalRectLimitExceeded",
    "rival_feasibility_check",
    "rival_trim_context",
]


MAX_RECTS_ENV = "ZOLOTONE_RIVAL_MAX_RECTS"
DEFAULT_MAX_RECTS = 50_000


class RivalRectLimitExceeded(RuntimeError):
    def __init__(self, rect_count: int, max_rects: int):
        super().__init__(
            f"Rival rectangle limit exceeded: {rect_count} candidates "
            f"for a limit of {max_rects}"
        )
        self.rect_count = rect_count
        self.max_rects = max_rects


def resolve_max_rects(max_rects: int | None) -> int:
    if max_rects is None:
        configured_max = os.environ.get(MAX_RECTS_ENV)
        if configured_max is None:
            return DEFAULT_MAX_RECTS
        try:
            max_rects = int(configured_max)
        except ValueError as exc:
            raise ValueError(
                f"{MAX_RECTS_ENV} must be a positive integer"
            ) from exc
    if isinstance(max_rects, bool) or not isinstance(max_rects, int):
        raise TypeError("max_rects must be an integer or None")
    if max_rects < 1:
        raise ValueError("max_rects must be at least 1")
    return max_rects


def _check_rect_limit(rect_count: int, max_rects: int) -> None:
    if rect_count > max_rects:
        raise RivalRectLimitExceeded(rect_count, max_rects)


@dataclass(frozen=True)
class RivalAnalysis:
    status: tuple[bool, bool]
    hints: Any


@dataclass(frozen=True)
class RivalExprSearch:
    machine: RivalMachine
    split_indexes: tuple[int, ...]


@dataclass(frozen=True)
class _RivalRectDomain:
    var_indexes: dict[str, int]
    bool_var_names: frozenset[str]
    base_rect: tuple[tuple[float, float], ...]

    @classmethod
    def build(
        cls,
        free_vars: Sequence[str],
        bool_var_names: Iterable[str],
    ) -> "_RivalRectDomain":
        indexes = {name: index for index, name in enumerate(free_vars)}
        bool_names = frozenset(bool_var_names)
        base_rect = tuple(
            (0.0, 1.0)
            if name in bool_names
            else (-math.inf, math.inf)
            for name in free_vars
        )
        return cls(indexes, bool_names, base_rect)

    def new_rect(self) -> list[tuple[float, float]]:
        return list(self.base_rect)


class RivalMachine:
    def __init__(self, raw_machine: Any):
        self._raw_machine = raw_machine

    def apply_with_hints(
        self,
        rect: Sequence[tuple[float, float]],
        hints: Any | None = None,
    ) -> RivalAnalysis:
        status, next_hints = self._raw_machine.apply_with_hints(rect, hints)
        return RivalAnalysis(
            status=(bool(status[0]), bool(status[1])),
            hints=next_hints,
        )


def build_machine(
    exprs: Sequence[SpecNode],
    free_vars: Sequence[str],
) -> RivalMachine:
    expr_list = list(exprs)
    var_list = _validate_free_vars(free_vars)
    _validate_referenced_vars(expr_list, var_list)
    native = _load_native_module()
    translated_exprs = [
        _append_assert(_and_exprs(to_rival_ir(expr) for expr in expr_list))
    ]
    raw_machine = native.build_machine(translated_exprs, var_list)
    return RivalMachine(raw_machine)


def collect_free_vars(exprs: Iterable[SpecNode]) -> list[str]:
    found: set[str] = set()

    def visit(node: SpecNode) -> None:
        if isinstance(node, (RealVar, BoolVar)):
            found.add(node.name)
        for child in children(node):
            visit(child)

    for expr in exprs:
        if not isinstance(expr, SpecNode):
            raise TypeError(f"Expected SpecNode, got {type(expr).__name__}")
        visit(expr)
    return sorted(found)


def _collect_bool_var_names(exprs: Iterable[SpecNode]) -> set[str]:
    bool_names: set[str] = set()
    real_names: set[str] = set()
    for expr in exprs:
        for var in variables(expr):
            if isinstance(var, BoolVar):
                bool_names.add(var.name)
            elif isinstance(var, RealVar):
                real_names.add(var.name)

    ambiguous = sorted(bool_names & real_names)
    if ambiguous:
        raise ValueError(
            "Rival variables cannot be both Boolean and real: "
            f"{ambiguous}"
        )
    return bool_names


def get_rival_rects(
    assumes: Sequence[BoolExpr],
    free_vars: Sequence[str],
    bool_var_names: Iterable[str] | None = None,
    max_rects: int | None = None,
) -> list[list[tuple[float, float]]]:
    resolved_max_rects = resolve_max_rects(max_rects)
    resolved_bool_var_names = _collect_bool_var_names(assumes) if bool_var_names is None else set(bool_var_names)
    rects, _ = _get_rival_rects_and_contributors(
        assumes,
        free_vars,
        resolved_bool_var_names,
        resolved_max_rects,
    )
    return rects


def _get_rival_rects_and_contributors(
    assumes: Sequence[BoolExpr],
    free_vars: Sequence[str],
    bool_var_names: set[str],
    max_rects: int,
) -> tuple[list[list[tuple[float, float]]], list[bool]]:
    domain = _RivalRectDomain.build(free_vars, bool_var_names)
    rects = [domain.new_rect()]
    contributors = [False] * len(assumes)
    for index, assume in enumerate(assumes):
        alternatives = _rival_rect_alternatives(
            assume,
            domain,
            max_rects,
        )
        if alternatives is None:
            continue
        contributors[index] = True
        rects = _intersect_rival_rect_sets(
            rects,
            alternatives,
            max_rects,
        )
        if not rects:
            break
    return rects, contributors


def rival_feasibility_check(
    ctx: "SpecContext",
    max_depth: int = 1,
    checks=False,
    max_rects: int | None = None,
):
    max_depth = int(max_depth)
    exprs = ctx.assumes + ctx.checks if checks else ctx.assumes
    free_vars = collect_free_vars(exprs)
    bool_var_names = _collect_bool_var_names(exprs)
    
    if not exprs:
        return "feasible"
    
    try:
        rects = get_rival_rects(
            ctx.assumes,
            free_vars,
            bool_var_names,
            max_rects=max_rects,
        )
    except RivalRectLimitExceeded:
        return "unknown"
    if not rects:
        return "not feasible"

    # expr_searches:
    #     x+y>2 | {x, y}
    #     x+1<1 | {x}
    combined_machine = build_machine(exprs, free_vars)
    expr_searches = _build_rival_expr_searches(exprs, free_vars)
    may_be_feasible = False

    for rect in rects:
        is_feasible, is_maybe = _rival_feasibility_dfs(
            combined_machine,
            expr_searches,
            rect,
            expr_index=0,
            depth=0,
            max_depth=max_depth,
            combined_hints=None,
            expr_hints=None,
        )
        if is_feasible:
            return "feasible"
        may_be_feasible = may_be_feasible or is_maybe

    return "unknown" if may_be_feasible else "not feasible"


def _rewrite_proven_expressions(
    nodes: Sequence[SpecNode],
    *,
    free_vars: Sequence[str],
    rects: Sequence[Sequence[tuple[float, float]]],
) -> list[SpecNode]:
    proof_cache: dict[BoolExpr, bool | None] = {}

    def unit_sign_magnitude(value: RealExpr) -> RealExpr | None:
        """Return the other factor when ``value`` multiplies by a unit sign."""

        if not isinstance(value, Mul):
            return None

        def is_unit_sign(expr: RealExpr) -> bool:
            if isinstance(expr, RealLit):
                return expr.value in {-1, 1}
            if not isinstance(expr, If):
                return False
            branches = (expr.on_true, expr.on_false)
            return all(
                isinstance(branch, RealLit) and branch.value in {-1, 1}
                for branch in branches
            )

        if is_unit_sign(value.lhs):
            return value.rhs
        if is_unit_sign(value.rhs):
            return value.lhs
        return None

    def prove(predicate: BoolExpr) -> bool | None:
        predicate = predicate.constant_fold()
        if isinstance(predicate, BoolLit):
            return predicate.value
        if not rects:
            return None

        cached = proof_cache.get(predicate)
        if cached is not None or predicate in proof_cache:
            return cached

        true_machine = build_machine([predicate], free_vars)
        true_statuses = [
            true_machine.apply_with_hints(rect, None).status
            for rect in rects
        ]
        if all(status == (False, False) for status in true_statuses):
            result = True
        elif all(status == (True, True) for status in true_statuses):
            # An asserted predicate that always errors may be either false or
            # undefined. Prove falsity by successfully asserting its negation
            # instead of interpreting an error status as Boolean false.
            negated = (~predicate).constant_fold()
            if isinstance(negated, BoolLit):
                result = not negated.value
            else:
                false_machine = build_machine([negated], free_vars)
                false_statuses = [false_machine.apply_with_hints(rect, None).status for rect in rects]
                result = False if all(status == (False, False) for status in false_statuses) else None
        else:
            result = None
        proof_cache[predicate] = result
        return result

    def rewrite(node: SpecNode, *, top_level: bool = False) -> SpecNode:
        old_children = children(node)
        new_children = tuple(rewrite(child) for child in old_children)
        rewritten = (
            node
            if all(old is new for old, new in zip(old_children, new_children))
            else type(node)(*new_children)
        ).constant_fold()

        if isinstance(rewritten, BoolLit):
            return rewritten

        if isinstance(rewritten, (Eq, NotEq, Lt, Le, Gt, Ge, BoolEq)):
            truth = prove(rewritten)
            return BoolLit(truth) if truth is not None else rewritten

        if isinstance(rewritten, If):
            truth = prove(rewritten.cond)
            if truth is True:
                return rewritten.on_true
            if truth is False:
                return rewritten.on_false
            return rewritten

        if isinstance(rewritten, Abs):
            magnitude = unit_sign_magnitude(rewritten.value)
            if magnitude is not None:
                # abs((+/-1) * x) == abs(x). Re-enter the rewrite so the
                # existing sign proof below can also reduce abs(x) to x.
                return rewrite(Abs(magnitude))

            zero = RealLit(0)
            nonnegative = prove(rewritten.value >= zero)
            if nonnegative is True:
                return rewritten.value
            if nonnegative is False:
                return (-rewritten.value).constant_fold()

            nonpositive = prove(rewritten.value <= zero)
            if nonpositive is True:
                return (-rewritten.value).constant_fold()
            if nonpositive is False:
                return rewritten.value
            return rewritten

        if isinstance(rewritten, (Max, Min)):
            lhs_is_at_least_rhs = prove(rewritten.lhs >= rewritten.rhs)
            if lhs_is_at_least_rhs is not None:
                if isinstance(rewritten, Max):
                    return rewritten.lhs if lhs_is_at_least_rhs else rewritten.rhs
                return rewritten.rhs if lhs_is_at_least_rhs else rewritten.lhs

            lhs_is_at_most_rhs = prove(rewritten.lhs <= rewritten.rhs)
            if lhs_is_at_most_rhs is None:
                return rewritten
            if isinstance(rewritten, Max):
                return rewritten.rhs if lhs_is_at_most_rhs else rewritten.lhs
            return rewritten.lhs if lhs_is_at_most_rhs else rewritten.rhs

        if top_level and isinstance(rewritten, BoolExpr):
            truth = prove(rewritten)
            return BoolLit(truth) if truth is not None else rewritten

        return rewritten

    return [rewrite(node, top_level=True) for node in nodes]


# Preserve assumptions used to construct the rectangular domain. Rewrite all
# other assumptions and checks only when every applicable rectangle agrees,
# then drop expressions that are certainly true.
def rival_trim_context(
    ctx: "SpecContext",
    max_rects: int | None = None,
) -> "SpecContext":
    exprs = ctx.assumes + ctx.checks
    free_vars = collect_free_vars(exprs)
    bool_var_names = _collect_bool_var_names(exprs)
    resolved_max_rects = resolve_max_rects(max_rects)
    try:
        assumption_rects, assumption_contributes_to_rect = (
            _get_rival_rects_and_contributors(
                ctx.assumes,
                free_vars,
                bool_var_names,
                resolved_max_rects,
            )
        )
    except RivalRectLimitExceeded:
        return ctx
    # Assumes that do not store pure facts and can be simplified
    rewritable_assumes = [
        assume
        for assume, contributes in zip(
            ctx.assumes,
            assumption_contributes_to_rect,
        )
        if not contributes
    ]
    rewritten_assumes = iter(
        _rewrite_proven_expressions(
            rewritable_assumes,
            free_vars=free_vars,
            rects=assumption_rects,
        )
    )
    rewritten_ctx = ctx.copy(
        assumes=[
            assume if contributes else next(rewritten_assumes)
            for assume, contributes in zip(
                ctx.assumes,
                assumption_contributes_to_rect,
            )
        ],
        checks=_rewrite_proven_expressions(
            ctx.checks,
            free_vars=free_vars,
            rects=assumption_rects,
        ),
    )

    return rewritten_ctx.copy(
        assumes=[
            assume
            for assume in rewritten_ctx.assumes
            if not identical_nodes(assume, BoolLit(True))
        ],
        checks=[
            check
            for check in rewritten_ctx.checks
            if not identical_nodes(check, BoolLit(True))
        ],
    )


def _build_rival_expr_searches(
    exprs: Sequence[SpecNode],
    free_vars: Sequence[str],
) -> list[RivalExprSearch]:
    free_var_indexes = {name: index for index, name in enumerate(free_vars)}
    searches: list[RivalExprSearch] = []
    for expr in exprs:
        # free_vars = ["x", "y", "z"]
        # assume x >= 0      # split_indexes = {0}
        # assume z >= 0      # split_indexes = {2}
        # check  x == y      # split_indexes = {0, 1}
        split_indexes = tuple(sorted({
            free_var_indexes[name]
            for name in collect_free_vars([expr])
            if name in free_var_indexes
        }))
        searches.append(
            RivalExprSearch(
                machine=build_machine([expr], free_vars),
                split_indexes=split_indexes,
            )
        )
    return searches

def _rival_feasibility_dfs(
    combined_machine: RivalMachine,
    expr_searches: Sequence[RivalExprSearch],
    rect: Sequence[tuple[float, float]],
    expr_index: int,
    depth: int,
    max_depth: int,
    combined_hints: Any | None,
    expr_hints: Any | None,
) -> tuple[bool, bool]:
    combined_analysis = combined_machine.apply_with_hints(rect, combined_hints)
    combined_status = combined_analysis.status

    # Not Feasible
    if combined_status == (True, True):
        return False, False

    # Feasible
    if combined_status == (False, False):
        return True, False

    if combined_status != (False, True):
        raise ValueError(f"Unexpected Rival status: {combined_status!r}")

    # The whole expressions is still maybe feasible
    for index in range(expr_index, len(expr_searches)):
        expr_search = expr_searches[index]  # expr[i]
        hints = expr_hints if index == expr_index else None
        analysis = expr_search.machine.apply_with_hints(rect, hints)
        status = analysis.status

        # Rect is maybe feasible on expr[i]
        if status == (False, True):
            # Split rect and try again
            if depth < max_depth:
                children = _subdivide_rival_rect(rect, expr_search.split_indexes)
                if children:
                    may_be_feasible = False
                    for child in children:
                        is_feasible, is_maybe = _rival_feasibility_dfs(
                            combined_machine,
                            expr_searches,
                            child,
                            expr_index=index,
                            depth=depth + 1,
                            max_depth=max_depth,
                            combined_hints=combined_analysis.hints,
                            expr_hints=analysis.hints,
                        )
                        if is_feasible:
                            return True, False
                        may_be_feasible = may_be_feasible or is_maybe
                    return False, may_be_feasible
            # Cannot split anymore
            return False, True

        # Rect is not feasible on expr[i]
        if status == (True, True):
            return False, False

        # Rect is feasible on expr[i]
        if status == (False, False):
            continue

        raise ValueError(f"Unexpected Rival status: {status!r}")

    # Combined machine was "maybe" but every individual expression passed being feasible - overall, feasible
    return True, False


def _subdivide_rival_rect(
    rect: Sequence[tuple[float, float]],
    split_indexes: Iterable[int] | None = None,
) -> list[list[tuple[float, float]]]:
    allowed_indexes = None if split_indexes is None else set(split_indexes)
    dimension_options: list[list[tuple[float, float]]] = []
    split_any_dimension = False
    for index, interval in enumerate(rect):
        if allowed_indexes is not None and index not in allowed_indexes:
            dimension_options.append([interval])
            continue
        
        split = _split_rival_interval(interval)
        if split is None:
            dimension_options.append([interval])
        else:
            split_any_dimension = True
            dimension_options.append(split)
    
    if not split_any_dimension:
        return []
    
    return [list(child) for child in product(*dimension_options)]


def _split_rival_interval(
    interval: tuple[float, float],
) -> list[tuple[float, float]] | None:
    lower, upper = interval
    if lower > upper:
        return None
    
    effective_lower = -sys.float_info.max if lower == -math.inf else lower
    effective_upper = sys.float_info.max if upper == math.inf else upper
    midpoint = (effective_lower / 2.0) + (effective_upper / 2.0)
    right_lower = math.nextafter(midpoint, math.inf)
    
    left = (lower, midpoint)
    right = (right_lower, upper)
    if (
        lower <= midpoint
        and right_lower <= upper
        and left != interval
        and right != interval
    ):
        return [left, right]
    return None


def _rival_rect_alternatives(
    expr: BoolExpr,
    domain: _RivalRectDomain,
    max_rects: int,
) -> list[list[tuple[float, float]]] | None:
    if isinstance(expr, BoolLit):
        return None if expr.value else []

    expr_vars = variables(expr)
    if (
        expr_vars
        and all(isinstance(var, BoolVar) for var in expr_vars)
        and all(var.name in domain.bool_var_names for var in expr_vars)
    ):
        return _rival_boolean_rect_alternatives(
            expr,
            domain,
            max_rects,
        )

    if isinstance(expr, And):
        lhs = _rival_rect_alternatives(expr.lhs, domain, max_rects)
        rhs = _rival_rect_alternatives(expr.rhs, domain, max_rects)
        if lhs == [] or rhs == []:
            return []
        if lhs is None:
            return rhs
        if rhs is None:
            return lhs
        return _intersect_rival_rect_sets(lhs, rhs, max_rects)

    if isinstance(expr, Or):
        lhs = _rival_rect_alternatives(expr.lhs, domain, max_rects)
        rhs = _rival_rect_alternatives(expr.rhs, domain, max_rects)
        if lhs is None or rhs is None:
            return None
        _check_rect_limit(len(lhs) + len(rhs), max_rects)
        return lhs + rhs

    return _rival_comparison_rect(expr, domain)


def _rival_boolean_rect_alternatives(
    expr: BoolExpr,
    domain: _RivalRectDomain,
    max_rects: int,
) -> list[list[tuple[float, float]]] | None:
    bool_vars = sorted(
        (var for var in variables(expr) if isinstance(var, BoolVar)),
        key=lambda var: var.name,
    )
    if any(var.name not in domain.var_indexes for var in bool_vars):
        return None

    alternatives: list[list[tuple[float, float]]] = []
    for values in product((False, True), repeat=len(bool_vars)):
        replacements = {
            var: BoolLit(value)
            for var, value in zip(bool_vars, values)
        }
        folded = substitute_literals(expr, replacements).constant_fold()
        if not isinstance(folded, BoolLit):
            return None
        if not folded.value:
            continue

        rect = domain.new_rect()
        for var, value in zip(bool_vars, values):
            point = 1.0 if value else 0.0
            rect[domain.var_indexes[var.name]] = (point, point)
        alternatives.append(rect)
        _check_rect_limit(len(alternatives), max_rects)
    return alternatives


def _rival_comparison_rect(
    expr: BoolExpr,
    domain: _RivalRectDomain,
) -> list[list[tuple[float, float]]] | None:
    if not isinstance(expr, (Eq, Lt, Le, Gt, Ge)):
        return None

    var: RealVar
    literal: RealLit
    var_on_lhs: bool
    if isinstance(expr.lhs, RealVar) and isinstance(expr.rhs, RealLit):
        var = expr.lhs
        literal = expr.rhs
        var_on_lhs = True
    elif isinstance(expr.lhs, RealLit) and isinstance(expr.rhs, RealVar):
        var = expr.rhs
        literal = expr.lhs
        var_on_lhs = False
    else:
        return None

    literal_enclosure = _rival_literal_enclosure(literal)
    if literal_enclosure is None:
        return None
    literal_lower, literal_upper = literal_enclosure

    var_index = domain.var_indexes.get(var.name)
    if var_index is None:
        return [domain.new_rect()]

    lower = -math.inf
    upper = math.inf
    if isinstance(expr, Eq):
        lower = literal_lower
        upper = literal_upper
    elif isinstance(expr, (Gt, Ge)):
        if var_on_lhs:
            lower = literal_lower
        else:
            upper = literal_upper
    elif isinstance(expr, (Lt, Le)):
        if var_on_lhs:
            upper = literal_upper
        else:
            lower = literal_lower

    rect = domain.new_rect()
    rect[var_index] = (lower, upper)
    return [] if lower > upper else [rect]


def _rival_literal_enclosure(
    literal: RealLit,
) -> tuple[float, float] | None:
    try:
        exact = Fraction(literal.value)
        value = float(exact)
    except (OverflowError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(value):
        return None

    lower = value
    upper = value
    represented = Fraction.from_float(value)
    if represented > exact:
        lower = math.nextafter(value, -math.inf)
    elif represented < exact:
        upper = math.nextafter(value, math.inf)
    return lower, upper


def _intersect_rival_rect_sets(
    lhs_rects: list[list[tuple[float, float]]],
    rhs_rects: list[list[tuple[float, float]]],
    max_rects: int,
) -> list[list[tuple[float, float]]]:
    _check_rect_limit(len(lhs_rects) * len(rhs_rects), max_rects)
    intersections: list[list[tuple[float, float]]] = []
    for lhs in lhs_rects:
        for rhs in rhs_rects:
            intersection = _intersect_rival_rects(lhs, rhs)
            if intersection is not None:
                intersections.append(intersection)
                _check_rect_limit(len(intersections), max_rects)
    return intersections


def _intersect_rival_rects(
    lhs: list[tuple[float, float]],
    rhs: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    rect: list[tuple[float, float]] = []
    for (lhs_lower, lhs_upper), (rhs_lower, rhs_upper) in zip(lhs, rhs):
        lower = max(lhs_lower, rhs_lower)
        upper = min(lhs_upper, rhs_upper)
        if lower > upper:
            return None
        rect.append((lower, upper))
    return rect


def to_rival_ir(node: SpecNode) -> RivalIR:
    if isinstance(node, (RealVar, BoolVar)):
        return {"op": "var", "name": node.name}
    if isinstance(node, RealLit):
        ratio = _real_literal_ratio(node.value)
        return {
            "op": "real_lit",
            "num": str(ratio.numerator),
            "den": str(ratio.denominator),
        }
    if isinstance(node, BoolLit):
        return {"op": "bool_lit", "value": bool(node.value)}
    
    if isinstance(node, Add):
        return _binary("add", node.lhs, node.rhs)
    if isinstance(node, Sub):
        return _binary("sub", node.lhs, node.rhs)
    if isinstance(node, Mul):
        return _binary("mul", node.lhs, node.rhs)
    if isinstance(node, Neg):
        return _unary("neg", node.value)
    if isinstance(node, Abs):
        return _unary("abs", node.value)
    if isinstance(node, Pow):
        return _binary("pow", node.base, node.exponent)
    if isinstance(node, Max):
        return _binary("max", node.lhs, node.rhs)
    if isinstance(node, Min):
        return _binary("min", node.lhs, node.rhs)
    if isinstance(node, If):
        return {
            "op": "if",
            "cond": to_rival_ir(node.cond),
            "on_true": to_rival_ir(node.on_true),
            "on_false": to_rival_ir(node.on_false),
        }
    
    if isinstance(node, Eq):
        return _binary("eq", node.lhs, node.rhs)
    if isinstance(node, NotEq):
        return _binary("ne", node.lhs, node.rhs)
    if isinstance(node, Lt):
        return _binary("lt", node.lhs, node.rhs)
    if isinstance(node, Le):
        return _binary("le", node.lhs, node.rhs)
    if isinstance(node, Gt):
        return _binary("gt", node.lhs, node.rhs)
    if isinstance(node, Ge):
        return _binary("ge", node.lhs, node.rhs)
    if isinstance(node, BoolEq):
        return _binary("bool_eq", node.lhs, node.rhs)
    if isinstance(node, Not):
        return _unary("not", node.value)
    if isinstance(node, Or):
        return _binary("or", node.lhs, node.rhs)
    if isinstance(node, And):
        return _binary("and", node.lhs, node.rhs)
    
    raise TypeError(f"Unsupported SpecNode for Rival translation: {type(node).__name__}")


def _binary(op: str, lhs: SpecNode, rhs: SpecNode) -> RivalIR:
    return {"op": op, "lhs": to_rival_ir(lhs), "rhs": to_rival_ir(rhs)}


def _unary(op: str, arg: SpecNode) -> RivalIR:
    return {"op": op, "arg": to_rival_ir(arg)}


def _and_exprs(exprs: Iterable[RivalIR]) -> RivalIR:
    expr_iter = iter(exprs)
    result = next(expr_iter)
    for expr in expr_iter:
        result = {"op": "and", "lhs": result, "rhs": expr}
    return result


def _append_assert(expr: RivalIR) -> RivalIR:
    return {"op": "assert", "arg": expr}


def _real_literal_ratio(value: int | float) -> Fraction:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Rival translation does not support non-finite RealLit values")
    return Fraction(value)


def _validate_free_vars(free_vars: Sequence[str]) -> list[str]:
    result = list(free_vars)
    if any(not isinstance(var, str) for var in result):
        raise TypeError("free_vars must contain only strings")
    if len(set(result)) != len(result):
        raise ValueError("free_vars must be unique")
    return result


def _validate_referenced_vars(exprs: Sequence[SpecNode], free_vars: Sequence[str]) -> None:
    if not exprs:
        raise ValueError("exprs must be non-empty")
    missing = sorted(set(collect_free_vars(exprs)) - set(free_vars))
    if missing:
        raise ValueError(f"free_vars is missing referenced variables: {missing}")


def _load_native_module():
    try:
        from . import _rival3
    except ImportError:
        try:
            import _rival3  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                "Rival3 native bridge is not built. Build it with "
                "`maturin develop -m crates/rival_bridge/Cargo.toml` or an equivalent "
                "PyO3 build command."
            ) from exc
    return _rival3
