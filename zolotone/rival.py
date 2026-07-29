from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
from time import perf_counter
import sys
from typing import Any, Iterable, Sequence

from .spec.spec_ast import *

RivalIR = dict[str, Any]

__all__ = [
    "rival_feasibility_check",
    "rival_trim_context",
]


@dataclass(frozen=True)
class RivalAnalysis:
    status: tuple[bool, bool]
    hints: Any
    converged: bool


@dataclass(frozen=True)
class RivalExprSearch:
    expr: Any
    machine: RivalMachine
    split_indexes: tuple[int, ...]


class RivalMachine:
    def __init__(self, raw_machine: Any):
        self._raw_machine = raw_machine

    def apply_with_hints(
        self,
        rect: Sequence[tuple[float, float]],
        hints: Any | None = None,
    ) -> RivalAnalysis:
        status, next_hints, converged = self._raw_machine.apply_with_hints(rect, hints)
        return RivalAnalysis(
            status=(bool(status[0]), bool(status[1])),
            hints=next_hints,
            converged=bool(converged),
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


def get_rival_rects(
    assumes: Sequence[BoolExpr],
    free_vars: Sequence[str],
) -> list[list[tuple[float, float]]]:
    var_indexes = {name: index for index, name in enumerate(free_vars)}
    rects = [_rival_unbounded_rect(len(free_vars))]
    for assume in assumes:
        alternatives = _rival_rect_alternatives(
            assume,
            var_indexes,
            len(free_vars),
        )
        if alternatives is None:
            continue
        rects = _intersect_rival_rect_sets(rects, alternatives)
        if not rects:
            break
    return rects


def rival_feasibility_check(ctx: "SpecContext", max_depth: int = 1, checks=False):
    max_depth = int(max_depth)
    exprs = ctx.assumes + ctx.checks if checks else ctx.assumes
    free_vars = collect_free_vars(exprs)
    
    if not exprs:
        return "feasible"
    
    rects = get_rival_rects(ctx.assumes, free_vars)
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
    comparison_rects: Sequence[Sequence[tuple[float, float]]],
    numeric_rects: Sequence[Sequence[tuple[float, float]]],
    simplify_comparisons: bool,
) -> list[SpecNode]:
    comparison_cache: dict[BoolExpr, bool | None] = {}
    numeric_cache: dict[BoolExpr, bool | None] = {}

    def prove(
        predicate: BoolExpr,
        rects: Sequence[Sequence[tuple[float, float]]],
        cache: dict[BoolExpr, bool | None],
    ) -> bool | None:
        predicate = predicate.constant_fold()
        if isinstance(predicate, BoolLit):
            return predicate.value
        if not rects:
            return None

        cached = cache.get(predicate)
        if cached is not None or predicate in cache:
            return cached

        machine = build_machine([predicate], free_vars)
        statuses = [
            machine.apply_with_hints(rect, None).status
            for rect in rects
        ]
        if all(status == (False, False) for status in statuses):
            result = True
        elif all(status == (True, True) for status in statuses):
            result = False
        else:
            result = None
        cache[predicate] = result
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

        if (
            simplify_comparisons
            and isinstance(rewritten, (Eq, NotEq, Lt, Le, Gt, Ge, BoolEq))
        ):
            truth = prove(rewritten, comparison_rects, comparison_cache)
            return BoolLit(truth) if truth is not None else rewritten

        if isinstance(rewritten, If):
            truth = prove(rewritten.cond, numeric_rects, numeric_cache)
            if truth is True:
                return rewritten.on_true
            if truth is False:
                return rewritten.on_false
            return rewritten

        if isinstance(rewritten, Abs):
            zero = RealLit(0)
            nonnegative = prove(
                rewritten.value >= zero,
                numeric_rects,
                numeric_cache,
            )
            if nonnegative is True:
                return rewritten.value
            if nonnegative is False:
                return (-rewritten.value).constant_fold()

            nonpositive = prove(
                rewritten.value <= zero,
                numeric_rects,
                numeric_cache,
            )
            if nonpositive is True:
                return (-rewritten.value).constant_fold()
            if nonpositive is False:
                return rewritten.value
            return rewritten

        if isinstance(rewritten, (Max, Min)):
            lhs_is_at_least_rhs = prove(
                rewritten.lhs >= rewritten.rhs,
                numeric_rects,
                numeric_cache,
            )
            if lhs_is_at_least_rhs is not None:
                if isinstance(rewritten, Max):
                    return rewritten.lhs if lhs_is_at_least_rhs else rewritten.rhs
                return rewritten.rhs if lhs_is_at_least_rhs else rewritten.lhs

            lhs_is_at_most_rhs = prove(
                rewritten.lhs <= rewritten.rhs,
                numeric_rects,
                numeric_cache,
            )
            if lhs_is_at_most_rhs is None:
                return rewritten
            if isinstance(rewritten, Max):
                return rewritten.rhs if lhs_is_at_most_rhs else rewritten.lhs
            return rewritten.lhs if lhs_is_at_most_rhs else rewritten.rhs

        if top_level and isinstance(rewritten, BoolExpr):
            truth = prove(rewritten, comparison_rects, comparison_cache)
            return BoolLit(truth) if truth is not None else rewritten

        return rewritten

    return [rewrite(node, top_level=True) for node in nodes]


# First rewrite comparisons and numeric branches only when every applicable
# rectangle agrees. Then drop expressions that are certainly true. Assumptions
# must hold without relying on themselves; checks may use a conservative
# rectangular enclosure of the assumption domain.
def rival_trim_context(ctx: "SpecContext") -> "SpecContext":
    original_exprs = ctx.assumes + ctx.checks
    original_free_vars = collect_free_vars(original_exprs)
    assumption_rects = get_rival_rects(
        ctx.assumes,
        original_free_vars,
    )
    unbounded_rects = [_rival_unbounded_rect(len(original_free_vars))]
    rewritten_ctx = ctx.copy(
        assumes=_rewrite_proven_expressions(
            ctx.assumes,
            free_vars=original_free_vars,
            comparison_rects=unbounded_rects,
            numeric_rects=assumption_rects,
            simplify_comparisons=False,
        ),
        checks=_rewrite_proven_expressions(
            ctx.checks,
            free_vars=original_free_vars,
            comparison_rects=assumption_rects,
            numeric_rects=assumption_rects,
            simplify_comparisons=True,
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
                expr=expr,
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


def _rival_unbounded_rect(dimensions: int) -> list[tuple[float, float]]:
    return [(-math.inf, math.inf) for _ in range(dimensions)]


def _rival_rect_alternatives(
    expr: BoolExpr,
    var_indexes: dict[str, int],
    dimensions: int,
) -> list[list[tuple[float, float]]] | None:
    if isinstance(expr, And):
        lhs = _rival_rect_alternatives(expr.lhs, var_indexes, dimensions)
        rhs = _rival_rect_alternatives(expr.rhs, var_indexes, dimensions)
        if lhs == [] or rhs == []:
            return []
        if lhs is None:
            return rhs
        if rhs is None:
            return lhs
        return _intersect_rival_rect_sets(lhs, rhs)
    
    if isinstance(expr, Or):
        lhs = _rival_rect_alternatives(expr.lhs, var_indexes, dimensions)
        rhs = _rival_rect_alternatives(expr.rhs, var_indexes, dimensions)
        if lhs is None or rhs is None:
            return None
        return lhs + rhs
    
    return _rival_comparison_rect(expr, var_indexes, dimensions)


def _rival_comparison_rect(
    expr: BoolExpr,
    var_indexes: dict[str, int],
    dimensions: int,
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
    
    var_index = var_indexes.get(var.name)
    if var_index is None:
        return [_rival_unbounded_rect(dimensions)]
    
    lower = -math.inf
    upper = math.inf
    if isinstance(expr, Eq):
        lower = literal_lower
        upper = literal_upper
    elif isinstance(expr, Gt):
        if var_on_lhs:
            lower = literal_lower
        else:
            upper = literal_upper
    elif isinstance(expr, Ge):
        if var_on_lhs:
            lower = literal_lower
        else:
            upper = literal_upper
    elif isinstance(expr, Lt):
        if var_on_lhs:
            upper = literal_upper
        else:
            lower = literal_lower
    elif isinstance(expr, Le):
        if var_on_lhs:
            upper = literal_upper
        else:
            lower = literal_lower
    
    rect = _rival_unbounded_rect(dimensions)
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
) -> list[list[tuple[float, float]]]:
    intersections: list[list[tuple[float, float]]] = []
    for lhs in lhs_rects:
        for rhs in rhs_rects:
            intersection = _intersect_rival_rects(lhs, rhs)
            if intersection is not None:
                intersections.append(intersection)
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
