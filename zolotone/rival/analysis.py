"""Range, domain, and feasibility algorithms built on the Rival machine API."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
import os
import sys
from typing import Any, Iterable, Mapping, Sequence

from ..errors import InfeasibleError, ZolotoneError
from ..spec.spec_ast import *
from ..types import Bool, DataType, Q, UQ
from .machine import (
    RivalMachine, build_boolean_evaluator, build_integer_evaluator, build_machine, collect_free_vars, to_rival_ir,
    _and_exprs, _real_literal_ratio,
)

__all__ = [
    "DEFAULT_MAX_RECTS", "MAX_RECTS_ENV", "RivalDomainError",
    "RivalRectLimitExceeded", "get_rival_rects", "rival_domain_errors",
    "analyze_typed_domain", "rival_feasibility_check", "rival_trim_context",
    "rival_prove_result_fits",
]

MAX_RECTS_ENV = "ZOLOTONE_RIVAL_MAX_RECTS"
DEFAULT_MAX_RECTS = 50_000


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

def _build_predicate_machine(exprs: Sequence[SpecNode], free_vars: Sequence[str]) -> RivalMachine:
    assertion = {"op": "assert", "arg": _and_exprs(to_rival_ir(expr) for expr in exprs)}
    return build_machine([assertion], free_vars)


def _build_domain_machine(exprs: Sequence[SpecNode], free_vars: Sequence[str]) -> RivalMachine:
    return build_machine([{"op": "error", "arg": to_rival_ir(expr)} for expr in exprs], free_vars)

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
class RivalDomainError:
    expression_index: int
    status: tuple[bool, bool]
    free_vars: tuple[str, ...]
    rect: tuple[tuple[float, float], ...]


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


def get_rival_rects(
    assumes: Sequence[BoolExpr],
    free_vars: Sequence[str],
    bool_var_names: Iterable[str] | None = None,
    max_rects: int | None = None,
) -> list[list[tuple[float, float]]]:
    resolved_max_rects = resolve_max_rects(max_rects)
    resolved_bool_var_names = _collect_bool_var_names(assumes) if bool_var_names is None else set(bool_var_names)
    try:
        rects, _ = _get_rival_rects_and_contributors(
            assumes,
            free_vars,
            resolved_bool_var_names,
            resolved_max_rects,
        )
    except RivalRectLimitExceeded:
        rects = [
            _RivalRectDomain.build(
                free_vars,
                resolved_bool_var_names,
            ).new_rect()
        ]
    return rects


def analyze_typed_domain(
    output: SpecNode,
    ctx: "SpecContext",
    input_dtypes: Mapping[SpecNode, DataType],
    output_dtype: object = Q,
    *,
    obligations: Sequence[BoolExpr] = (),
) -> tuple[Fraction, Fraction]:
    """Validate a finite typed domain and enclose its output.

    Inputs are decoded from integer codes. Generic Q/UQ outputs use the
    existing 30-fractional-bit grid, without implicit rounding. Unresolved
    points and domain errors raise rather than supplying unproven bounds.
    """
    if output_dtype is Bool:
        output_dtype = Bool()
    generic_output = output_dtype is Q or output_dtype is UQ
    if not all(isinstance(dtype, (Q, UQ, Bool)) for dtype in input_dtypes.values()):
        raise NotImplementedError("spec domain analysis requires exact Q, UQ, or Bool inputs")
    if not generic_output and not isinstance(output_dtype, (Q, UQ, Bool)):
        raise NotImplementedError("spec domain analysis requires Q, UQ, or exact Bool output")

    assumes = BoolLit(True)
    for assume in ctx.assumes:
        assumes = assumes & assume
    checks = BoolLit(True)
    for check in obligations:
        checks = checks & check

    missing = set(collect_free_vars([assumes, checks, output])) - set(collect_free_vars(input_dtypes))
    if missing:
        raise ZolotoneError(f"Typed domain analysis needs input dtypes for: {sorted(missing)}")

    # Materialized subexpressions may also serve as typed inputs (e.g. LUTs).
    occupied = set(collect_free_vars([assumes, checks, output]))
    dtypes_by_name = {}
    labels_by_name = {}
    for source, dtype in input_dtypes.items():
        if isinstance(source, (RealVar, BoolVar)):
            name = source.name
        else:
            name = f"__rival_input_{len(dtypes_by_name)}"
            while name in occupied:
                name += "_"
        occupied.add(name)
        dtypes_by_name[name] = dtype
        labels_by_name[name] = source.name if isinstance(source, (RealVar, BoolVar)) else str(source)
        code = RealVar(name)
        decoded = (
            code.ne(RealLit(0)) if isinstance(dtype, Bool)
            else code * (RealLit(2) ** RealLit(-dtype.frac_bits))
        )
        assumes = substitute_spec_node(assumes, source, decoded)
        checks = substitute_spec_node(checks, source, decoded)
        output = substitute_spec_node(output, source, decoded)

    # Creating Rival Machines
    free_vars = collect_free_vars([assumes, checks, output])
    missing = set(free_vars) - dtypes_by_name.keys()
    if missing:
        raise ZolotoneError(f"Typed domain analysis needs input dtypes for: {sorted(missing)}")
    kwargs = dict(
        free_vars=free_vars,
    )
    # The precondition path lets an error-free false conjunct reject the box.
    assumes_machine = build_integer_evaluator(
        RealLit(0), precondition=assumes, output_total_bits=1, **kwargs,
    )
    checks_machine = build_boolean_evaluator(checks, **kwargs)
    generic_frac_bits = (
        output.frac_bits
        if isinstance(output, Round) and output.frac_bits is not None else 30
    )
    output_frac_bits = generic_frac_bits if generic_output else (
        0 if isinstance(output_dtype, Bool) else output_dtype.frac_bits
    )
    output_scale = Fraction(2) ** output_frac_bits
    if isinstance(output_dtype, Bool):
        output_machine = build_boolean_evaluator(output, **kwargs)
    else:
        output_machine = build_integer_evaluator(
            output * RealLit(output_scale),
            output_total_bits=30 if generic_output else output_dtype.total_bits(), **kwargs,
        )

    # Creating Initial Rectangulars
    initial_box: list[tuple[int, int]] = []
    for name in kwargs["free_vars"]:
        dtype = dtypes_by_name[name]
        width = dtype.total_bits()
        if isinstance(dtype, Q):
            lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
        else:
            lower, upper = 0, (1 << width) - 1
        initial_box.append((lower, upper))

    def split_box(box: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """Bisect the widest integer-code interval; point boxes have no children."""
        axis = max(range(len(box)), key=lambda i: box[i][1] - box[i][0], default=None)
        if axis is None or box[axis][0] == box[axis][1]:
            return []
        lower, upper = box[axis]
        middle = (lower + upper) // 2
        left, right = list(box), list(box)
        left[axis] = (lower, middle)
        right[axis] = (middle + 1, upper)
        return [left, right]

    def format_box(box):
        assignments = []
        intervals = dict(zip(free_vars, box, strict=True))
        # Preserve the input mapping's order, independently of machine order.
        for name in labels_by_name:
            if name not in intervals:
                continue
            lower, upper = intervals[name]
            dtype = dtypes_by_name[name]
            if isinstance(dtype, Bool):
                lower, upper = bool(lower), bool(upper)
            else:
                scale = Fraction(2) ** -dtype.frac_bits
                lower, upper = lower * scale, upper * scale
            value = str(lower) if lower == upper else f"[{lower}, {upper}]"
            assignments.append(f"{labels_by_name[name]}={value}")
        kind = "point" if all(lo == hi for lo, hi in box) else "range"
        return f"input {kind} [{', '.join(assignments) or 'no inputs'}]"

    def evaluate_box(machine, box, is_point, location):
        try:
            if is_point:
                return machine.apply_point([lower for lower, _ in box])
            return machine.apply_range(box)
        except ValueError as exc:
            raise ZolotoneError(
                f"Could not evaluate {location} at {format_box(box)}: {exc}"
            ) from exc

    reachable = False
    output_bounds = None
    pending = [initial_box]
    while pending:
        box = pending.pop()
        is_point = all(lower == upper for lower, upper in box)

        assumes = evaluate_box(assumes_machine, box, is_point, "assumptions")
        if not assumes.precondition_error and assumes.precondition in (False, (False, False)):
            # An error-free rejection excludes this box.
            continue
        if assumes.precondition_total_error:
            raise ZolotoneError(f"Domain error in assumptions at {format_box(box)}")
        if assumes.precondition_error or assumes.precondition not in (True, (True, True)):
            if is_point:
                raise ZolotoneError(f"Unresolved assumptions at {format_box(box)}")
            pending.extend(split_box(box))
            continue

        # Every point in this box satisfies all assumptions.
        reachable = True

        checks = evaluate_box(checks_machine, box, is_point, "checks")
        if checks.output_total_error:
            raise ZolotoneError(f"Domain error in checks at {format_box(box)}")
        if checks.output_error or checks.output not in (0, 1, (0, 0), (1, 1)):
            if is_point:
                raise ZolotoneError(f"Unresolved checks at {format_box(box)}")
            pending.extend(split_box(box))
            continue
        if checks.output in (0, (0, 0)):
            raise InfeasibleError(f"Specification has a check that does not hold at {format_box(box)}")

        output = evaluate_box(output_machine, box, is_point, "output")
        if output.output_total_error:
            raise ZolotoneError(f"Domain error in output at {format_box(box)}")
        if output.output_error or output.output is None:
            if is_point:
                raise ZolotoneError(f"Unresolved output at {format_box(box)}")
            pending.extend(split_box(box))
            continue

        lower, upper = (output.output, output.output) if is_point else output.output
        lower, upper = lower / output_scale, upper / output_scale
        output_bounds = (
            (lower, upper) if output_bounds is None else
            (min(output_bounds[0], lower), max(output_bounds[1], upper))
        )

    if not reachable:
        raise InfeasibleError(f"Specification {ctx.name!r} is unreachable: no input satisfies its assumptions")
    return output_bounds


def rival_prove_result_fits(
    expr: SpecNode,
    output_type: object,
    ctx: "SpecContext",
    input_dtypes: Mapping[SpecNode, DataType],
) -> bool:
    """Prove a result range over finite input codes using one Rival machine.

    Range enclosures discharge boxes; unresolved boxes are partitioned without
    gaps until point evaluation decides them. Evaluation failures raise rather
    than being mistaken for either a proof or an overflow counterexample.
    """
    input_names = set(collect_free_vars(input_dtypes))
    missing = set(collect_free_vars([expr, *ctx.assumes])) - input_names
    if missing:
        raise ZolotoneError(f"Result-fit proof needs input dtypes for: {', '.join(sorted(missing))}")

    if output_type is Bool or isinstance(output_type, Bool) or output_type is Q:
        return True
    if not isinstance(expr, RealExpr):
        raise TypeError("Result-fit proof expects a real expression")
    if output_type is not UQ and not isinstance(output_type, (Q, UQ)):
        raise TypeError(f"Result-fit proof expects a Q or UQ output type, got {output_type!r}")

    # Precondition
    precondition = BoolLit(True)
    for assume in ctx.assumes:
        precondition = precondition & assume

    # Reserve variable names from the input expressions.
    occupied = set(input_names)
    free_vars = []
    input_domains = []
    initial_box = []
    for source, dtype in input_dtypes.items():
        if not (variables(expr) | variables(precondition)) & variables(source):
            continue
        name = f"__rival_input_{len(free_vars)}"
        while name in occupied:
            name += "_"
        occupied.add(name)
        raw = RealVar(name)
        if isinstance(dtype, (Q, UQ)):
            width = dtype.total_bits()
            lower = -(1 << (width - 1)) if isinstance(dtype, Q) else 0
            upper = (1 << (width - isinstance(dtype, Q))) - 1
            replacement = raw * (RealLit(2) ** RealLit(-dtype.frac_bits))
        elif isinstance(dtype, Bool):
            width, lower, upper = 1, 0, 1
            replacement = raw.ne(RealLit(0))
        else:
            raise NotImplementedError(f"Rival result-fit inputs require Q, UQ or Bool, got {dtype!r}")
        expr = substitute_spec_node(expr, source, replacement)
        precondition = substitute_spec_node(precondition, source, replacement)
        free_vars.append(name)
        input_domains.append((source, dtype))
        initial_box.append((lower, upper))
    # Expression keys may cover only some occurrences of their variables.
    missing = set(collect_free_vars([expr, precondition])) - set(free_vars)
    if missing:
        raise ZolotoneError(f"Result-fit proof needs input dtypes for: {', '.join(sorted(missing))}")

    def format_input_point(point):
        assignments = []
        for (source, dtype), code in zip(input_domains, point, strict=True):
            label = source.name if isinstance(source, (RealVar, BoolVar)) else f"({source})"
            value = bool(code) if isinstance(dtype, Bool) else Fraction(code) * Fraction(2) ** -dtype.frac_bits
            assignments.append(f"{label}={value}")
        return "[" + ", ".join(assignments) + "]"

    if output_type is UQ:
        # A generic unsigned annotation asks only about sign, with no output
        # scale. Encode that predicate as an exact integer Boolean indicator.
        integer_expr = If(precondition, If(expr >= RealLit(0), RealLit(1), RealLit(0)), RealLit(1))
        minimum = maximum = 1
        output_bits = 1
    else:
        output_bits = output_type.total_bits()
        minimum = -(1 << (output_bits - 1)) if isinstance(output_type, Q) else 0
        maximum = (1 << (output_bits - isinstance(output_type, Q))) - 1
        if isinstance(expr, Round):
            # Preserve the final rounding node so the integer machine can
            # refine using its rounding mode. Bounds use that node's scale.
            integer_expr = Round(If(precondition, expr.value, RealLit(0)), expr.mode, expr.frac_bits)
            scale = Fraction(2) ** (expr.frac_bits - output_type.frac_bits)
            minimum *= scale
            maximum *= scale
        else:
            integer_expr = If(precondition, expr * RealLit(1 << output_type.frac_bits), RealLit(0))

    machine = build_integer_evaluator(
        integer_expr, precondition, free_vars=free_vars,
        output_total_bits=output_bits,
    )
    pending = [tuple(initial_box)]
    while pending:
        box = pending.pop()
        split_index = max(range(len(box)), key=lambda index: box[index][1] - box[index][0], default=None)
        if split_index is None or box[split_index][0] == box[split_index][1]:
            point = [lower for lower, _ in box]
            try:
                result = machine.apply_point(point)
            except ValueError as exc:
                raise ZolotoneError(f"Could not prove result fit at input point {format_input_point(point)}: {exc}") from exc
            if result.precondition_error:
                raise ZolotoneError(f"Domain error in assumptions at input point {format_input_point(point)}")
            if result.precondition is False:
                continue
            if result.precondition is not True or result.output_error or result.output is None:
                detail = "domain error in result" if result.output_error else "unresolved or non-finite result"
                raise ZolotoneError(f"Could not prove output range at input point {format_input_point(point)}: {detail}")
            if not minimum <= result.output <= maximum:
                return False
            continue

        result = machine.apply_range(box)
        if not result.precondition_error:
            if result.precondition == (False, False):
                continue
            if not result.output_error and result.output is not None:
                lower, upper = result.output
                if minimum <= lower <= upper <= maximum:
                    continue
        lower, upper = box[split_index]
        midpoint = (lower + upper) // 2
        pending.append(box[:split_index] + ((lower, midpoint),) + box[split_index + 1:])
        pending.append(box[:split_index] + ((midpoint + 1, upper),) + box[split_index + 1:])
    return True


def rival_domain_errors(
    exprs: Sequence[SpecNode],
    assumes: Sequence[BoolExpr],
    max_rects: int | None = None,
) -> list[RivalDomainError]:
    """Return every error observed in the first failing rectangle."""
    expression_list = list(exprs)
    if not expression_list:
        return []
    all_exprs = [*assumes, *expression_list]
    free_vars = collect_free_vars(all_exprs)
    bool_var_names = _collect_bool_var_names(all_exprs)
    rects = get_rival_rects(
        assumes,
        free_vars,
        bool_var_names,
        max_rects=max_rects,
    )
    if not rects:
        return []

    machine = _build_domain_machine(expression_list, free_vars)
    for rect in rects:
        statuses = machine.domain_errors(rect)
        if len(statuses) != len(expression_list):
            raise RuntimeError(
                "Rival domain machine returned an unexpected number of outputs"
            )
        findings = [
            RivalDomainError(
                expression_index=expression_index,
                status=status,
                free_vars=tuple(free_vars),
                rect=tuple(rect),
            )
            for expression_index, status in enumerate(statuses)
            if status != (False, False)
        ]
        if findings:
            failing_expressions = [
                expression_list[finding.expression_index]
                for finding in findings
            ]
            return [
                finding
                for finding in findings
                if not _contains_failing_descendant(
                    expression_list[finding.expression_index],
                    failing_expressions,
                )
            ]
    return []


def _contains_failing_descendant(
    expression: SpecNode,
    failing_expressions: Sequence[SpecNode],
) -> bool:
    pending = list(children(expression))
    while pending:
        descendant = pending.pop()
        if any(
            identical_nodes(descendant, failing)
            for failing in failing_expressions
        ):
            return True
        pending.extend(children(descendant))
    return False


def _get_rival_rects_and_contributors(
    assumes: Sequence[BoolExpr],
    free_vars: Sequence[str],
    bool_var_names: set[str],
    max_rects: int,
) -> tuple[list[list[tuple[float, float]]], list[bool]]:
    from ..spec.spec_context import SpecContext, normalize_ctx

    # Keep assumption positions intact so contributors still refer to the
    # caller's source expressions, even when extraction uses normalized ones.
    normalization_ctx = SpecContext("Rival rectangle assumptions")
    normalization_ctx.assumes = list(assumes)
    assumes = normalize_ctx(normalization_ctx).assumes
    domain = _RivalRectDomain.build(free_vars, bool_var_names)
    variable_kind_cache: dict[int, tuple[bool, bool]] = {}
    rects = [domain.new_rect()]
    contributors = [False] * len(assumes)
    for index, assume in enumerate(assumes):
        alternatives = _rival_rect_alternatives(
            assume,
            domain,
            max_rects,
            variable_kind_cache,
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
    
    rects = get_rival_rects(
        ctx.assumes,
        free_vars,
        bool_var_names,
        max_rects=max_rects,
    )
    if not rects:
        return "not feasible"

    # expr_searches:
    #     x+y>2 | {x, y}
    #     x+1<1 | {x}
    combined_machine = _build_predicate_machine(exprs, free_vars)
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
        predicate = predicate._constant_fold()
        if isinstance(predicate, BoolLit):
            return predicate.value
        if not rects:
            return None

        cached = proof_cache.get(predicate)
        if cached is not None or predicate in proof_cache:
            return cached

        true_machine = _build_predicate_machine([predicate], free_vars)
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
            negated = (~predicate)._constant_fold()
            if isinstance(negated, BoolLit):
                result = not negated.value
            else:
                false_machine = _build_predicate_machine([negated], free_vars)
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
            else rebuild_spec_node(node, new_children)
        )._constant_fold()

        if isinstance(rewritten, BoolLit):
            return rewritten

        if isinstance(rewritten, (Eq, NotEq, Lt, Le, Gt, Ge, BoolEq, BoolNe)):
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
                return (-rewritten.value)._constant_fold()

            nonpositive = prove(rewritten.value <= zero)
            if nonpositive is True:
                return (-rewritten.value)._constant_fold()
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


def _rewrite_assumptions_from_prior_rects(
    assumes: Sequence[BoolExpr],
    free_vars: Sequence[str],
    bool_var_names: set[str],
    max_rects: int,
) -> list[BoolExpr]:
    """Simplify each assumption using only earlier rectangular facts."""
    domain = _RivalRectDomain.build(free_vars, bool_var_names)
    variable_kind_cache: dict[int, tuple[bool, bool]] = {}
    rects = [domain.new_rect()]
    rewritten_assumes = []
    has_prior_rect_fact = False
    for assume in assumes:
        if has_prior_rect_fact:
            assume = _rewrite_proven_expressions(
                [assume], free_vars=free_vars, rects=rects
            )[0]
        if identical_nodes(assume, BoolLit(True)):
            continue
        rewritten_assumes.append(assume)
        alternatives = _rival_rect_alternatives(
            assume, domain, max_rects, variable_kind_cache
        )
        if alternatives is None:
            continue
        rects = _intersect_rival_rect_sets(rects, alternatives, max_rects)
        has_prior_rect_fact = True
    return rewritten_assumes


def rival_trim_context(
    ctx: "SpecContext",
    max_rects: int | None = None,
) -> "SpecContext":
    exprs = ctx.assumes + ctx.checks
    free_vars = collect_free_vars(exprs)
    bool_var_names = _collect_bool_var_names(exprs)
    resolved_max_rects = resolve_max_rects(max_rects)
    try:
        rewritten_assumes = _rewrite_assumptions_from_prior_rects(
            ctx.assumes,
            free_vars,
            bool_var_names,
            resolved_max_rects,
        )
        # The reverse pass also removes earlier facts implied by later,
        # stronger facts without an all-pairs implication search.
        rewritten_assumes = list(
            reversed(
                _rewrite_assumptions_from_prior_rects(
                    list(reversed(rewritten_assumes)),
                    free_vars,
                    bool_var_names,
                    resolved_max_rects,
                )
            )
        )
        ctx = ctx.copy(assumes=rewritten_assumes)
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
                machine=_build_predicate_machine([expr], free_vars),
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
    variable_kind_cache: dict[int, tuple[bool, bool]],
) -> list[list[tuple[float, float]]] | None:
    if isinstance(expr, BoolLit):
        return None if expr.value else []

    has_bool_vars, has_real_vars = _variable_kinds(expr, variable_kind_cache)
    if (
        has_bool_vars
        and not has_real_vars
        and all(
            var.name in domain.bool_var_names
            for var in variables(expr)
            if isinstance(var, BoolVar)
        )
    ):
        return _rival_boolean_rect_alternatives(
            expr,
            domain,
            max_rects,
        )

    if isinstance(expr, And):
        lhs = _rival_rect_alternatives(
            expr.lhs, domain, max_rects, variable_kind_cache
        )
        rhs = _rival_rect_alternatives(
            expr.rhs, domain, max_rects, variable_kind_cache
        )
        if lhs == [] or rhs == []:
            return []
        if lhs is None:
            return rhs
        if rhs is None:
            return lhs
        return _intersect_rival_rect_sets(lhs, rhs, max_rects)

    if isinstance(expr, Or):
        lhs = _rival_rect_alternatives(
            expr.lhs, domain, max_rects, variable_kind_cache
        )
        rhs = _rival_rect_alternatives(
            expr.rhs, domain, max_rects, variable_kind_cache
        )
        if lhs is None or rhs is None:
            return None
        _check_rect_limit(len(lhs) + len(rhs), max_rects)
        return lhs + rhs

    return _rival_comparison_rect(expr, domain)


def _variable_kinds(
    expr: SpecNode,
    cache: dict[int, tuple[bool, bool]],
) -> tuple[bool, bool]:
    """Return (has BoolVar, has RealVar), caching by immutable node identity."""
    cache_key = id(expr)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if isinstance(expr, BoolVar):
        result = (True, False)
    elif isinstance(expr, RealVar):
        result = (False, True)
    else:
        has_bool_vars = False
        has_real_vars = False
        for child in children(expr):
            child_has_bool, child_has_real = _variable_kinds(child, cache)
            has_bool_vars = has_bool_vars or child_has_bool
            has_real_vars = has_real_vars or child_has_real
        result = (has_bool_vars, has_real_vars)

    cache[cache_key] = result
    return result


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
        folded = substitute_literals(expr, replacements)._constant_fold()
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
