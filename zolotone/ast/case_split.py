"""Case-driven verification and floating-point classification fallback."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from itertools import product

from ..solver.engine import check_equivalence as _solver_check_equivalence
from ..solver.report import (
    CaseVerificationResult,
    ProofReport,
    VerificationObserver,
)
from ..spec import (
    BoolExpr,
    BoolLit,
    FPExpr,
    SpecContext,
    SpecNode,
    children,
    substitute_literals,
    variables,
)
from ..spec.spec_context import simplify_ctx
from .parallel_verification import run_verification_cases


@dataclass(frozen=True)
class _AdaptiveCase:
    ctx: SpecContext
    side_assumptions: tuple[tuple[BoolExpr, ...], tuple[BoolExpr, ...]]
    vacuous_side: int | None = None


def _append_case_name(name: str, case_label: str) -> str:
    if name.endswith("]") and "[" in name:
        return f"{name[:-1]},{case_label}]"
    return f"{name}[{case_label}]"


def _case_labels(name: str) -> dict[str, str]:
    if not name.endswith("]") or "[" not in name:
        return {}

    labels = {}
    for label in name[name.rfind("[") + 1:-1].split(","):
        key, separator, value = label.partition("=")
        if separator:
            labels[key] = value
    return labels


def _assume_classification_case(
    ctx: SpecContext,
    value_name: str,
    value: FPExpr,
    selected_name: str,
) -> None:
    ctx.name = _append_case_name(ctx.name, f"{value_name}={selected_name}")
    _assume_classification(ctx, value, selected_name)


def _assume_classification(
    ctx: SpecContext,
    value: FPExpr,
    selected_name: str,
) -> None:
    for assumption in _classification_assumptions(ctx, value, selected_name):
        ctx.assume(assumption)


def _classification_assumptions(
    ctx: SpecContext,
    value: FPExpr,
    selected_name: str,
) -> tuple[BoolExpr, ...]:
    return tuple(
        flag.eq(ctx.bool_val(flag_name == selected_name))
        for flag_name, flag in value.classification_flags().items()
    )


def _named_fp_items(value_name: str, value: tp.Any):
    if isinstance(value, FPExpr):
        yield value_name, value
        return
    if isinstance(value, tuple):
        for idx, item in enumerate(value):
            yield from _named_fp_items(f"{value_name}.{idx}", item)


# Unroll classification flags and observables into equalities
def _add_classification_case_checks(
    ctx: SpecContext,
    spec_inner: tp.Any,
    spec_outer: tp.Any,
    labels: dict[str, str],
) -> None:
    def add_checks(inner, outer, output_name):
        inner_is_tuple = isinstance(inner, tuple)
        outer_is_tuple = isinstance(outer, tuple)
        if inner_is_tuple or outer_is_tuple:
            if not (inner_is_tuple and outer_is_tuple):
                raise TypeError(
                    "Spec shape mismatch: one output is a tuple and the other is not"
                )
            if len(inner) != len(outer):
                raise TypeError(
                    f"Spec tuple arity mismatch: {len(inner)} != {len(outer)}"
                )
            for idx, (inner_item, outer_item) in enumerate(zip(inner, outer)):
                add_checks(inner_item, outer_item, f"{output_name}.{idx}")
            return

        inner_is_fp = isinstance(inner, FPExpr)
        outer_is_fp = isinstance(outer, FPExpr)
        if inner_is_fp != outer_is_fp:
            raise TypeError(
                "Spec shape mismatch: one output is FPExpr and the other is not"
            )
        if not inner_is_fp:
            ctx.check(inner.eq(outer))
            return
        if type(inner) is not type(outer):
            raise TypeError("Different FPExprs are provided")

        selected_class = labels[output_name]
        inner_values = (
            tuple(inner.classification_flags().values())
            + inner.observables_for_classification(selected_class)
        )
        outer_values = (
            tuple(outer.classification_flags().values())
            + outer.observables_for_classification(selected_class)
        )
        for inner_value, outer_value in zip(
            inner_values,
            outer_values,
            strict=True,
        ):
            ctx.check(inner_value.eq(outer_value))

    add_checks(spec_inner, spec_outer, "output")


def _walk_value(value: tp.Any) -> tp.Iterator[SpecNode]:
    if isinstance(value, SpecNode):
        yield value
        for child in children(value):
            yield from _walk_value(child)
    elif isinstance(value, tuple):
        for item in value:
            yield from _walk_value(item)


# TODO: case_partitions does not need to be a list for now - a single object
def _direct_case_partition(ctx: SpecContext, output: tp.Any):
    return next(
        (
            partition
            for partition in reversed(ctx.case_partitions)
            if partition.value is output
        ),
        None,
    )


def _effective_case_guards(entries) -> list[BoolExpr]:
    no_prior_match: BoolExpr = BoolLit(True)
    guards = []
    for entry in entries:
        guards.append(no_prior_match & entry.condition)
        no_prior_match = no_prior_match & (~entry.condition)
    return guards


def _variables_in_value(value: tp.Any) -> set[tp.Any]:
    result = set()
    for expression in _walk_value(value):
        result.update(variables(expression))
    return result


def _partition_for(
    side_contexts: tuple[SpecContext, SpecContext],
    outputs: tuple[tp.Any, tp.Any],
    inputs: list[tp.Any],
    preferred_side: int | None,
):
    candidate_sides = (
        (preferred_side,)
        if preferred_side is not None
        else (0, 1)
    )
    shared_variables = _variables_in_value(tuple(inputs))
    for side_index in candidate_sides:
        # Get partition per side
        partition = _direct_case_partition(
            side_contexts[side_index],
            outputs[side_index],
        )
        if partition is None:
            continue
        guards = _effective_case_guards(partition.entries)
        # TODO: probably we do not care about that?
        if any(
            not variables(guard).issubset(shared_variables)
            for guard in guards
        ):
            continue
        return side_index, partition, guards
    return None


def _descendant_input_flag_groups(
    condition: BoolExpr,
    inputs: list[tp.Any],
) -> list[list[tuple[str, str, BoolExpr]]]:
    """Return feasible classification groups for inputs used by condition."""
    descendant_ids = {id(node) for node in _walk_value(condition)}
    seen_flags: set[int] = set()
    groups = []
    for input_index, item in enumerate(inputs):
        for value_name, fp_value in _named_fp_items(f"arg{input_index}", item):
            classification_flags = [
                (flag_name, flag)
                for flag_name, flag in fp_value.classification_flags().items()
                if not isinstance(flag, BoolLit)
            ]
            mentioned_flags = [
                (flag_name, flag)
                for flag_name, flag in classification_flags
                if id(flag) in descendant_ids
            ]
            if not mentioned_flags:
                continue

            mentioned_false = {
                flag: BoolLit(False)
                for _, flag in mentioned_flags
            }
            condition_with_mentioned_flags_false = substitute_literals(
                condition,
                mentioned_false,
            )
            flags_for_refinement = (
                mentioned_flags
                if isinstance(condition_with_mentioned_flags_false, BoolLit)
                and not condition_with_mentioned_flags_false.value
                else classification_flags
            )

            group = []
            for flag_name, flag in flags_for_refinement:
                flag_id = id(flag)
                if flag_id in seen_flags:
                    continue
                seen_flags.add(flag_id)
                group.append((value_name, flag_name, flag))
            if group:
                groups.append(group)
    return groups


def _refinement_flag_assignments(
    condition: BoolExpr,
    inputs: list[tp.Any],
) -> tp.Iterator[tuple[tuple[str, str, BoolExpr, bool], ...]]:
    """Enumerate feasible one-hot assignments to condition input flags."""
    groups = _descendant_input_flag_groups(condition, inputs)
    if not groups:
        return

    group_options = []
    for flags in groups:
        options = [
            tuple((*flag, flag_index == selected_index)
                  for flag_index, flag in enumerate(flags))
            for selected_index in range(len(flags))
        ]
        group_options.append(options)

    for selected_options in product(*group_options):
        assignment = tuple(
            item
            for group_assignment in selected_options
            for item in group_assignment
        )
        replacements = {
            flag: BoolLit(flag_value)
            for _, _, flag, flag_value in assignment
        }
        folded_condition = substitute_literals(condition, replacements)
        if isinstance(folded_condition, BoolLit) and not folded_condition.value:
            continue
        yield assignment


def _output_classes(value: tp.Any) -> tuple[str | None, ...]:
    if isinstance(value, FPExpr):
        return tuple(value.classification_flags())
    return (None,)


def _adaptive_case(
    combined_ctx: SpecContext,
    partition_side: int,
    path_index: int,
    entry,
    guard: BoolExpr,
    other_output: tp.Any,
    output_class: str | None,
    refinements: tuple[tuple[str | None, BoolExpr], ...] = (),
) -> _AdaptiveCase:
    case_ctx = combined_ctx.copy()
    case_ctx.name = _append_case_name(case_ctx.name, f"path={path_index}")
    case_ctx.assume(guard)
    side_assumptions = [[guard], [guard]]

    labels = {}
    if output_class is not None:
        case_ctx.name = _append_case_name(
            case_ctx.name,
            f"output={output_class}",
        )
        labels["output"] = output_class
        output_assumptions = _classification_assumptions(
            case_ctx,
            entry.value,
            output_class,
        )
        for assumption in output_assumptions:
            case_ctx.assume(assumption)
        side_assumptions[partition_side].extend(output_assumptions)

    for label, assumption in refinements:
        if label is not None:
            case_ctx.name = _append_case_name(case_ctx.name, label)
        case_ctx.assume(assumption)
        side_assumptions[0].append(assumption)
        side_assumptions[1].append(assumption)

    _add_classification_case_checks(
        case_ctx,
        entry.value,
        other_output,
        labels,
    )
    return _AdaptiveCase(
        case_ctx,
        (tuple(side_assumptions[0]), tuple(side_assumptions[1])),
        vacuous_side=partition_side if output_class is not None else None,
    )


def _coarse_cases(
    combined_ctx: SpecContext,
    partition_side: int,
    partition,
    guards: list[BoolExpr],
    other_output: tp.Any,
) -> tp.Iterator[_AdaptiveCase]:
    for path_index, (entry, guard) in enumerate(
        zip(partition.entries, guards, strict=True)
    ):
        for output_class in _output_classes(entry.value):
            yield _adaptive_case(
                combined_ctx,
                partition_side,
                path_index,
                entry,
                guard,
                other_output,
                output_class,
            )


def _refined_cases(
    combined_ctx: SpecContext,
    partition_side: int,
    partition,
    guards: list[BoolExpr],
    other_output: tp.Any,
    inputs: list[tp.Any],
    unresolved_cases: set[tuple[int, str | None]],
) -> tp.Iterator[_AdaptiveCase]:
    for path_index, (entry, guard) in enumerate(
        zip(partition.entries, guards, strict=True)
    ):
        for output_class in _output_classes(entry.value):
            if (path_index, output_class) not in unresolved_cases:
                continue
            for assignment in _refinement_flag_assignments(
                entry.condition,
                inputs,
            ):
                refinements = tuple(
                    (
                        f"{value_name}={flag_name}" if flag_value else None,
                        flag.eq(combined_ctx.bool_val(flag_value)),
                    )
                    for value_name, flag_name, flag, flag_value in assignment
                )
                yield _adaptive_case(
                    combined_ctx,
                    partition_side,
                    path_index,
                    entry,
                    guard,
                    other_output,
                    output_class,
                    refinements,
                )


def _side_feasibilities_with_assumptions_match(
    side_contexts: tuple[SpecContext, SpecContext],
    side_assumptions: tuple[tuple[BoolExpr, ...], tuple[BoolExpr, ...]],
) -> tuple[bool, list[ProofReport]]:
    reports = [
        simplify_ctx(
            side_ctx.copy(
                assumes=list(side_ctx.assumes) + list(assumptions),
            )
        )
        for side_ctx, assumptions in zip(
            side_contexts,
            side_assumptions,
            strict=True,
        )
    ]
    statuses = [
        report.get("feasibility_status", "unknown")
        for report in reports
    ]
    if any(status not in {"feasible", "not feasible"} for status in statuses):
        return False, reports
    return statuses[0] == statuses[1], reports


def _verify_adaptive_case(
    case: _AdaptiveCase,
    schedule: list[str | dict[str, tp.Any]],
    side_contexts: tuple[SpecContext, SpecContext],
) -> CaseVerificationResult:
    status, proof_trace = _solver_check_equivalence(
        case.ctx,
        schedule=schedule,
    )
    combined_feasibility = proof_trace[0].get(
        "feasibility_status",
        "unknown",
    )
    side_feasibility_reports = []
    if combined_feasibility == "not feasible":
        case_proved, side_feasibility_reports = (
            _side_feasibilities_with_assumptions_match(
                side_contexts,
                case.side_assumptions,
            )
        )
        if case.vacuous_side is not None:
            selected_status = side_feasibility_reports[
                case.vacuous_side
            ].get("feasibility_status", "unknown")
            if selected_status == "not feasible":
                case_proved = True
    else:
        case_proved = status == "unsat"

    return CaseVerificationResult(
        name=case.ctx.name,
        proved=case_proved,
        status=status,
        feasibility_status=combined_feasibility,
        proof_trace=proof_trace,
        side_feasibility_reports=side_feasibility_reports,
    )


def _run_adaptive_cases(
    combined_ctx: SpecContext,
    side_contexts: tuple[SpecContext, SpecContext],
    outputs: tuple[tp.Any, tp.Any],
    inputs: list[tp.Any],
    schedule: list[str | dict[str, tp.Any]],
    observer: VerificationObserver,
    max_workers: int,
    partition_side: int,
    partition,
    guards: list[BoolExpr],
) -> list[CaseVerificationResult]:
    other_output = outputs[1 - partition_side]
    coarse_results = run_verification_cases(
        _coarse_cases(
            combined_ctx,
            partition_side,
            partition,
            guards,
            other_output,
        ),
        verify_case=_verify_adaptive_case,
        verification_args=(schedule, side_contexts),
        observer=observer,
        max_workers=max_workers,
    )

    def result_key(result: CaseVerificationResult) -> tuple[int, str | None]:
        labels = _case_labels(result["name"])
        return int(labels["path"]), labels.get("output")

    unresolved_cases = {
        result_key(result)
        for result in coarse_results
        if result["status"] == "unknown"
    }

    refined_results = run_verification_cases(
        _refined_cases(
            combined_ctx,
            partition_side,
            partition,
            guards,
            other_output,
            inputs,
            unresolved_cases,
        ),
        verify_case=_verify_adaptive_case,
        verification_args=(schedule, side_contexts),
        observer=observer,
        max_workers=max_workers,
    )
    refined_by_case = {}
    for result in refined_results:
        refined_by_case.setdefault(result_key(result), []).append(result)

    terminal_results = []
    for coarse_result in coarse_results:
        key = result_key(coarse_result)
        if key not in unresolved_cases:
            terminal_results.append(coarse_result)
            continue
        replacements = refined_by_case.get(key)
        if replacements:
            terminal_results.extend(replacements)
        else:
            terminal_results.append(coarse_result)
    return terminal_results


def _split_classification_cases(
    ctx: SpecContext,
    inputs: list[tp.Any],
    spec_inner: tp.Any,
    spec_outer: tp.Any,
) -> tp.Iterator[SpecContext]:
    input_groups = [
        (fp_item[0], (fp_item,))
        for value_idx, value in enumerate(inputs)
        for fp_item in _named_fp_items(f"arg{value_idx}", value)
    ]
    inner_items = list(_named_fp_items("output", spec_inner))
    outer_items = list(_named_fp_items("output", spec_outer))
    if len(inner_items) != len(outer_items):
        raise TypeError("Spec shape mismatch between FPExpr outputs")

    paired_output_items = list(zip(inner_items, outer_items, strict=True))
    for (_, inner_value), (_, outer_value) in paired_output_items:
        if type(inner_value) is not type(outer_value):
            raise TypeError("Different FPExprs are provided")
        if tuple(inner_value.classification_flags()) != tuple(
            outer_value.classification_flags()
        ):
            raise TypeError(
                "Different FPExpr classifications between specification outputs"
            )

    output_groups = [
        (
            inner_name,
            ((inner_name, inner_value), (outer_name, outer_value)),
        )
        for (inner_name, inner_value), (outer_name, outer_value)
        in paired_output_items
    ]
    classification_groups = input_groups + output_groups
    flag_lists = [
        tuple(items[0][1].classification_flags())
        for _, items in classification_groups
    ]

    for selected_flags in product(*flag_lists):
        case_ctx = ctx.copy()
        labels = {}
        for (case_name, items), selected_name in zip(
            classification_groups,
            selected_flags,
            strict=True,
        ):
            case_ctx.name = _append_case_name(
                case_ctx.name,
                f"{case_name}={selected_name}",
            )
            labels[case_name] = selected_name
            for _, value in items:
                _assume_classification(case_ctx, value, selected_name)
        _add_classification_case_checks(
            case_ctx,
            spec_inner,
            spec_outer,
            labels,
        )
        yield case_ctx


def _classify_collected_spec(
    collected_ctx: SpecContext,
    output: tp.Any,
    inputs: list[tp.Any],
    case_labels: dict[str, str],
    output_name: str,
) -> SpecContext:
    ctx = collected_ctx.copy()
    for idx, value in enumerate(inputs):
        for value_name, fp_value in _named_fp_items(f"arg{idx}", value):
            selected_name = case_labels.get(value_name)
            if selected_name is not None:
                _assume_classification_case(
                    ctx,
                    value_name,
                    fp_value,
                    selected_name,
                )
    for case_name, fp_value in _named_fp_items(output_name, output):
        selected_name = case_labels.get(case_name)
        if selected_name is not None:
            _assume_classification_case(
                ctx,
                case_name,
                fp_value,
                selected_name,
            )
    return ctx


def _collect_classified_spec(
    spec,
    base_ctx: SpecContext,
    inputs: list[tp.Any],
    case_labels: dict[str, str],
) -> SpecContext:
    """Collect one standalone spec and apply a requested classification."""
    ctx = base_ctx.copy()
    output = spec.collect(ctx)
    return _classify_collected_spec(
        ctx,
        output,
        inputs=inputs,
        case_labels=case_labels,
        output_name=spec.name,
    )


def _side_feasibilities_match(
    side_contexts: tuple[SpecContext, SpecContext],
    outputs: tuple[tp.Any, tp.Any],
    inputs: list[tp.Any],
    labels: dict[str, str],
) -> tuple[bool, list[ProofReport]]:
    reports = [
        simplify_ctx(
            _classify_collected_spec(
                side_ctx,
                output,
                inputs=inputs,
                case_labels=labels,
                output_name="output",
            )
        )
        for side_ctx, output in zip(side_contexts, outputs, strict=True)
    ]
    statuses = [
        report.get("feasibility_status", "unknown")
        for report in reports
    ]
    if any(status not in {"feasible", "not feasible"} for status in statuses):
        return False, reports
    return statuses[0] == statuses[1], reports


def _verify_classification_case(
    case_ctx: SpecContext,
    schedule: list[str | dict[str, tp.Any]],
    side_contexts: tuple[SpecContext, SpecContext],
    outputs: tuple[tp.Any, tp.Any],
    inputs: list[tp.Any],
) -> CaseVerificationResult:
    labels = _case_labels(case_ctx.name)
    status, proof_trace = _solver_check_equivalence(case_ctx, schedule=schedule)
    combined_feasibility = proof_trace[0].get(
        "feasibility_status",
        "unknown",
    )
    side_feasibility_reports = []
    if combined_feasibility == "not feasible":
        case_proved, side_feasibility_reports = _side_feasibilities_match(
            side_contexts,
            outputs,
            inputs=inputs,
            labels=labels,
        )
    else:
        case_proved = status == "unsat"
    return CaseVerificationResult(
        name=case_ctx.name,
        proved=case_proved,
        status=status,
        feasibility_status=combined_feasibility,
        proof_trace=proof_trace,
        side_feasibility_reports=side_feasibility_reports,
    )


def run_equivalence_cases(
    combined_ctx: SpecContext,
    side_contexts: tuple[SpecContext, SpecContext],
    outputs: tuple[tp.Any, tp.Any],
    inputs: list[tp.Any],
    schedule: list[str | dict[str, tp.Any]],
    observer: VerificationObserver,
    max_workers: int,
    preferred_side: int | None,
) -> list[CaseVerificationResult]:
    """Choose adaptive Cases evaluation or the exhaustive fallback."""
    selected_partition = _partition_for(
        side_contexts,
        outputs,
        inputs,
        preferred_side,
    )
    if selected_partition is not None:
        partition_side, partition, guards = selected_partition
        return _run_adaptive_cases(
            combined_ctx=combined_ctx,
            side_contexts=side_contexts,
            outputs=outputs,
            inputs=inputs,
            schedule=schedule,
            observer=observer,
            max_workers=max_workers,
            partition_side=partition_side,
            partition=partition,
            guards=guards,
        )

    # No partitions are observed - run exhaustively
    cases = _split_classification_cases(
        combined_ctx,
        inputs,
        outputs[0],
        outputs[1],
    )
    return run_verification_cases(
        cases,
        verify_case=_verify_classification_case,
        verification_args=(schedule, side_contexts, outputs, inputs),
        observer=observer,
        max_workers=max_workers,
    )
