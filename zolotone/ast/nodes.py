import random
import typing as tp
from dataclasses import dataclass
from itertools import product

from ..types.runtime import RuntimeType
from ..types.static import StaticType
from ..utils import make_fixed_arguments
from ..solver.engine import check_equivalence as _solver_check_equivalence
from ..solver.report import *
from .node import Node
from .parallel_verification import *
from .proofs import *
from ..spec import *
from ..spec.spec_context import simplify_ctx


CLowering = tp.Callable[[list[str], bool], str]


class _Spec(tp.NamedTuple):
    name: str
    collect: tp.Callable[[SpecContext], tp.Any]
    partition_cases: bool = False


@dataclass(frozen=True)
class _AdaptiveVerificationCase:
    ctx: SpecContext
    side_assumptions: tuple[tuple[BoolExpr, ...], tuple[BoolExpr, ...]]


def _default_equivalence_schedule() -> list[dict[str, tp.Any]]:
    schedule = []
    for _ in range(3):
        schedule.extend(
            [
                {"tool": "simplify"},
                {
                    "tool": "egglog-rewrite",
                    "iterations": 6,
                    "scheduler": {
                        "match_limit": 500_000,
                        "ban_length": 1,
                    },
                },
            ]
        )
    schedule.append({"tool": "z3", "timeout_ms": 10000})
    return schedule


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
    for flag_name, flag in value.classification_flags().items():
        ctx.assume(flag.eq(ctx.bool_val(flag_name == selected_name)))


def _named_fp_items(value_name: str, value: tp.Any):
    if isinstance(value, FPExpr):
        yield value_name, value
        return
    if isinstance(value, tuple):
        for idx, item in enumerate(value):
            yield from _named_fp_items(f"{value_name}.{idx}", item)


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
                raise TypeError("Spec shape mismatch: one output is a tuple and the other is not")
            if len(inner) != len(outer):
                raise TypeError(f"Spec tuple arity mismatch: {len(inner)} != {len(outer)}")

            for idx, (inner_item, outer_item) in enumerate(zip(inner, outer)):
                add_checks(
                    inner_item,
                    outer_item,
                    f"{output_name}.{idx}",
                )
            return

        inner_is_fp = isinstance(inner, FPExpr)
        outer_is_fp = isinstance(outer, FPExpr)
        if inner_is_fp != outer_is_fp:
            raise TypeError("Spec shape mismatch: one output is FPExpr and the other is not")
        # RealExpr/BoolExpr
        if not inner_is_fp:
            ctx.check(inner.eq(outer))
            return
        if type(inner) is not type(outer):
            raise TypeError("Different FPExprs are provided")

        # Simply unroll flags + values for inner/outer FPExpr
        selected_class = labels[output_name]
        inner_flags = (
            tuple(inner.classification_flags().values())
            + inner.observables_for_classification(selected_class)
        )
        outer_flags = (
            tuple(outer.classification_flags().values())
            + outer.observables_for_classification(selected_class)
        )

        for inner_value, outer_value in zip(inner_flags, outer_flags, strict=True):
            ctx.check(inner_value.eq(outer_value))

    add_checks(spec_inner, spec_outer, "output")


def _add_guarded_observable_equality_checks(
    ctx: SpecContext,
    selected: tp.Any,
    other: tp.Any,
) -> None:
    """Compare complete outputs without observing meaningless FP fields."""
    selected_is_fp = isinstance(selected, FPExpr)
    other_is_fp = isinstance(other, FPExpr)
    if selected_is_fp or other_is_fp:
        if not (selected_is_fp and other_is_fp):
            raise TypeError(
                "Spec shape mismatch: one output is FPExpr and the other is not"
            )
        if type(selected) is not type(other):
            raise TypeError("Different FPExprs are provided")

        selected_flags = selected.classification_flags()
        other_flags = other.classification_flags()
        if tuple(selected_flags) != tuple(other_flags):
            raise TypeError(
                "Different FPExpr classifications between specification outputs"
            )
        for classification, selected_flag in selected_flags.items():
            other_flag = other_flags[classification]
            ctx.check(selected_flag.eq(other_flag))
            selected_observables = selected.observables_for_classification(
                classification
            )
            other_observables = other.observables_for_classification(
                classification
            )
            for selected_value, other_value in zip(
                selected_observables,
                other_observables,
                strict=True,
            ):
                ctx.check(
                    (selected_flag & other_flag).implies(
                        selected_value.eq(other_value)
                    )
                )
        return

    selected_is_scalar = isinstance(selected, (BoolExpr, RealExpr))
    other_is_scalar = isinstance(other, (BoolExpr, RealExpr))
    if not (selected_is_scalar and other_is_scalar) or (
        isinstance(selected, BoolExpr) != isinstance(other, BoolExpr)
    ):
        raise TypeError("Unsupported or mismatched specification output shape")
    ctx.check(selected.eq(other))


def _expressions_in_value(value: tp.Any) -> tp.Iterator[SpecNode]:
    if isinstance(value, FPExpr):
        yield from value.decode()
    elif isinstance(value, SpecNode):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            yield from _expressions_in_value(item)


def _variables_in_value(value: tp.Any) -> set[tp.Any]:
    result = set()
    for expression in _expressions_in_value(value):
        result.update(variables(expression))
    return result


def _classification_assumptions(
    value: FPExpr,
    selected_name: str,
) -> tuple[BoolExpr, ...]:
    return tuple(
        flag.eq(BoolLit(flag_name == selected_name))
        for flag_name, flag in value.classification_flags().items()
    )


def _effective_case_guards(entries) -> list[BoolExpr]:
    no_prior_match: BoolExpr = BoolLit(True)
    guards = []
    for entry in entries:
        guards.append(no_prior_match & entry.condition)
        no_prior_match = no_prior_match & (~entry.condition)
    return guards


def _direct_case_partition(ctx: SpecContext, output: tp.Any):
    return next(
        (
            partition
            for partition in reversed(ctx.case_partitions)
            if partition.value is output
        ),
        None,
    )


def _adaptive_partition(
    specs: tuple[_Spec, _Spec],
    contexts: tuple[SpecContext, SpecContext],
    outputs: tuple[tp.Any, tp.Any],
    inputs: list[tp.Any],
):
    preferred = [idx for idx, spec in enumerate(specs) if spec.partition_cases]
    candidates = preferred or [0, 1]
    shared_variables = _variables_in_value(tuple(inputs))
    for side_index in candidates:
        partition = _direct_case_partition(
            contexts[side_index],
            outputs[side_index],
        )
        if partition is None:
            continue
        guards = _effective_case_guards(partition.entries)
        if any(
            not variables(guard).issubset(shared_variables)
            for guard in guards
        ):
            continue
        other_output = outputs[1 - side_index]
        try:
            probe = SpecContext("case-partition-shape-probe")
            for entry in partition.entries:
                _add_guarded_observable_equality_checks(
                    probe,
                    entry.value,
                    other_output,
                )
        except (TypeError, ValueError):
            continue
        return side_index, partition, guards
    return None


def _selection_replacements(
    selections: tuple[tuple[str, FPExpr, str], ...],
) -> dict[SpecNode, SpecNode]:
    replacements = {}
    for _name, value, selected_name in selections:
        for flag_name, flag in value.classification_flags().items():
            # Literals are global leaves, not identities for one value's
            # classification field. Replacing them would rewrite unrelated
            # true/false constants throughout the guard.
            if not isinstance(flag, BoolLit):
                replacements[flag] = BoolLit(flag_name == selected_name)
    return replacements


def _selection_constant_folds_to_contradiction(
    guard: BoolExpr,
    selections: tuple[tuple[str, FPExpr, str], ...],
) -> bool:
    replacements = {}
    for selection in selections:
        next_replacements = _selection_replacements((selection,))
        if any(
            key in replacements
            and not identical_nodes(replacements[key], replacement)
            for key, replacement in next_replacements.items()
        ):
            return True
        replacements.update(next_replacements)
    folded_guard = substitute_literals(guard, replacements).constant_fold()
    if identical_nodes(folded_guard, BoolLit(False)):
        return True
    for _name, value, selected_name in selections:
        for assumption in _classification_assumptions(value, selected_name):
            folded = substitute_literals(assumption, replacements).constant_fold()
            if identical_nodes(folded, BoolLit(False)):
                return True
    return False


def _associated_input_groups(
    inputs: list[tp.Any],
    guard: BoolExpr,
    selected_value: tp.Any,
) -> list[tuple[str, FPExpr]]:
    dependencies = variables(guard) | _variables_in_value(selected_value)
    return [
        (name, value)
        for input_idx, item in enumerate(inputs)
        for name, value in _named_fp_items(f"arg{input_idx}", item)
        if _variables_in_value(value) & dependencies
    ]


def _coarse_adaptive_cases(
    combined_ctx: SpecContext,
    partition,
    guards: list[BoolExpr],
    other_output: tp.Any,
) -> tp.Iterator[_AdaptiveVerificationCase]:
    for path_index, (entry, guard) in enumerate(
        zip(partition.entries, guards, strict=True)
    ):
        case_ctx = combined_ctx.copy()
        case_ctx.name = _append_case_name(case_ctx.name, f"path={path_index}")
        case_ctx.assume(guard)
        _add_guarded_observable_equality_checks(
            case_ctx,
            entry.value,
            other_output,
        )
        side_assumptions = ((guard,), (guard,))
        yield _AdaptiveVerificationCase(case_ctx, side_assumptions)


def _refined_adaptive_cases(
    combined_ctx: SpecContext,
    partition,
    guards: list[BoolExpr],
    other_output: tp.Any,
    inputs: list[tp.Any],
    partition_side: int,
    unresolved_paths: set[int],
) -> tp.Iterator[_AdaptiveVerificationCase]:
    for path_index, (entry, guard) in enumerate(
        zip(partition.entries, guards, strict=True)
    ):
        if path_index not in unresolved_paths:
            continue

        groups = _associated_input_groups(inputs, guard, entry.value)
        if isinstance(entry.value, FPExpr):
            groups.append(("output", entry.value))

        def enumerate_selections(
            group_index: int,
            selected: tuple[tuple[str, FPExpr, str], ...],
            folded_guard: BoolExpr,
            replacements: dict[SpecNode, SpecNode],
        ):
            if group_index == len(groups):
                yield selected
                return
            group_name, value = groups[group_index]
            for selected_name in value.classification_flags():
                selection = (group_name, value, selected_name)
                group_replacements = _selection_replacements((selection,))
                if any(
                    key in replacements
                    and not identical_nodes(replacements[key], replacement)
                    for key, replacement in group_replacements.items()
                ):
                    continue
                next_guard = substitute_literals(
                    folded_guard,
                    group_replacements,
                ).constant_fold()
                if identical_nodes(next_guard, BoolLit(False)):
                    continue
                next_replacements = replacements | group_replacements
                if any(
                    identical_nodes(
                        substitute_literals(
                            assumption,
                            next_replacements,
                        ).constant_fold(),
                        BoolLit(False),
                    )
                    for assumption in _classification_assumptions(
                        value,
                        selected_name,
                    )
                ):
                    continue
                yield from enumerate_selections(
                    group_index + 1,
                    selected + (selection,),
                    next_guard,
                    next_replacements,
                )

        for selections in enumerate_selections(0, (), guard, {}):
            case_ctx = combined_ctx.copy()
            case_ctx.name = _append_case_name(case_ctx.name, f"path={path_index}")
            case_ctx.assume(guard)
            side_assumes = [[guard], [guard]]
            labels = {}
            for group_name, value, selected_name in selections:
                labels[group_name] = selected_name
                _assume_classification_case(
                    case_ctx,
                    group_name,
                    value,
                    selected_name,
                )
                assumptions = _classification_assumptions(value, selected_name)
                if group_name == "output":
                    side_assumes[partition_side].extend(assumptions)
                else:
                    side_assumes[0].extend(assumptions)
                    side_assumes[1].extend(assumptions)

            if isinstance(entry.value, FPExpr):
                _add_classification_case_checks(
                    case_ctx,
                    entry.value,
                    other_output,
                    labels,
                )
            else:
                case_ctx.check(entry.value.eq(other_output))
            yield _AdaptiveVerificationCase(
                case_ctx,
                (tuple(side_assumes[0]), tuple(side_assumes[1])),
            )


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

    # Corresponding outputs describe the same result, so classify them with
    # one shared choice.  Inputs remain independent classification dimensions.
    paired_output_items = list(zip(inner_items, outer_items, strict=True))
    
    for (_, inner_value), (_, outer_value) in paired_output_items:
        if type(inner_value) is not type(outer_value):
            raise TypeError("Different FPExprs are provided")
        if tuple(inner_value.classification_flags()) != tuple(outer_value.classification_flags()):
            raise TypeError("Different FPExpr classifications between specification outputs")

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
        
        _add_classification_case_checks(case_ctx, spec_inner, spec_outer, labels)
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
                _assume_classification_case(ctx, value_name, fp_value, selected_name)

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
    spec: _Spec,
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
        for side_ctx, output in zip(
            side_contexts,
            outputs,
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


def _verify_classification_case(
    case_ctx: SpecContext,
    schedule: list[str | dict[str, tp.Any]],
    side_contexts: tuple[SpecContext, SpecContext],
    outputs: tuple[tp.Any, tp.Any],
    inputs: list[tp.Any],
) -> CaseVerificationResult:
    """Verify one classification case in either this or a worker process."""
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


def _verify_adaptive_case(
    case: _AdaptiveVerificationCase,
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


class _NullVerificationObserver:
    def case_completed(self, result: CaseVerificationResult) -> None:
        del result


def check_equivalence(
    first: _Spec,
    second: _Spec,
    base_ctx: SpecContext,
    inputs: list[tp.Any],
    schedule: list[str | dict[str, tp.Any]] | None = None,
    observer: VerificationObserver | None = None,
    max_workers: int | None = None,
):
    worker_count = resolve_max_workers(max_workers)
    if first.name == second.name:
        raise ValueError("Equivalent specification sides must have distinct names")
    if schedule is None:
        schedule = _default_equivalence_schedule()
    active_observer = (
        observer if observer is not None else StdoutVerificationObserver()
    )

    base_assume_count = len(base_ctx.assumes)
    base_check_count = len(base_ctx.checks)
    base_requirement_count = len(base_ctx.requirements)

    first_ctx = base_ctx.copy()
    first_output = first.collect(first_ctx)

    second_ctx = base_ctx.copy()
    # Collection is isolated, while fresh names remain globally distinct.
    second_ctx._sym_counter = first_ctx._sym_counter
    second_output = second.collect(second_ctx)

    combined_ctx = base_ctx.copy(
        assumes=list(base_ctx.assumes) + first_ctx.assumes[base_assume_count:] + second_ctx.assumes[base_assume_count:],
        checks=list(base_ctx.checks) + first_ctx.checks[base_check_count:] + second_ctx.checks[base_check_count:],
        requirements=list(base_ctx.requirements) + first_ctx.requirements[base_requirement_count:] + second_ctx.requirements[base_requirement_count:],
    )
    combined_ctx._sym_counter = second_ctx._sym_counter

    requirement_report = combined_ctx.validate_requirements()

    adaptive = _adaptive_partition(
        (first, second),
        (first_ctx, second_ctx),
        (first_output, second_output),
        inputs,
    )
    if adaptive is not None:
        partition_side, partition, guards = adaptive
        outputs = (first_output, second_output)
        side_contexts = (first_ctx, second_ctx)
        other_output = outputs[1 - partition_side]
        coarse_results = run_verification_cases(
            _coarse_adaptive_cases(
                combined_ctx,
                partition,
                guards,
                other_output,
            ),
            verify_case=_verify_adaptive_case,
            verification_args=(schedule, side_contexts),
            observer=_NullVerificationObserver(),
            max_workers=worker_count,
        )

        terminal_coarse_results = []
        unresolved_paths = set()
        unresolved_coarse_results = {}
        for path_index, result in enumerate(coarse_results):
            if result["proved"]:
                terminal_coarse_results.append(result)
                active_observer.case_completed(result)
            elif result["status"] == "unknown":
                unresolved_paths.add(path_index)
                unresolved_coarse_results[path_index] = result
            else:
                terminal_coarse_results.append(result)
                active_observer.case_completed(result)

        refined_results = run_verification_cases(
            _refined_adaptive_cases(
                combined_ctx,
                partition,
                guards,
                other_output,
                inputs,
                partition_side,
                unresolved_paths,
            ),
            verify_case=_verify_adaptive_case,
            verification_args=(schedule, side_contexts),
            observer=active_observer,
            max_workers=worker_count,
        )
        refined_paths = {
            int(_case_labels(result["name"])["path"])
            for result in refined_results
        }
        unresolved_terminal_results = [
            result
            for path_index, result in unresolved_coarse_results.items()
            if path_index not in refined_paths
        ]
        for result in unresolved_terminal_results:
            active_observer.case_completed(result)
        case_results = (
            terminal_coarse_results
            + refined_results
            + unresolved_terminal_results
        )
        return CheckResult(
            proved=all(result["proved"] for result in case_results),
            requirement_report=requirement_report,
            cases=case_results,
        )

    cases = _split_classification_cases(
        combined_ctx,
        inputs,
        first_output,
        second_output,
    )

    case_results = run_verification_cases(
        cases,
        verify_case=_verify_classification_case,
        verification_args=(
            schedule,
            (first_ctx, second_ctx),
            (first_output, second_output),
            inputs,
        ),
        observer=active_observer,
        max_workers=worker_count,
    )
    proved = all(result["proved"] for result in case_results)

    return CheckResult(
        proved=proved,
        requirement_report=requirement_report,
        cases=case_results,
    )


def _check_determinism(
    node: "composite | primitive",
    schedule: list[str | dict[str, tp.Any]] | None = None,
    observer: VerificationObserver | None = None,
    max_workers: int | None = None,
):
    base_ctx = SpecContext(f"{node.name}_determinism")
    inputs = [base_ctx.spec_of(arg) for arg in node.inner_args]

    def collect_spec(ctx):
        encoded_inputs = [special_encoding(value, ctx) for value in inputs]
        return node.spec(*encoded_inputs, ctx=ctx)

    first_spec = _Spec("first_spec", collect_spec, partition_cases=True)
    second_spec = _Spec("second_spec", collect_spec)

    result = check_equivalence(
        first_spec,
        second_spec,
        base_ctx=base_ctx,
        inputs=inputs,
        schedule=schedule,
        observer=observer,
        max_workers=max_workers,
    )

    print(f"{node.name} specification {'is' if result['proved'] else 'is not'} deterministic")

    return result


def Composite(
    name: str,
    spec: tp.Callable[..., tp.Any],
    c_inline: bool = False,
    c_lowering: tp.Optional[CLowering] = None,
):
    def wrapper1(impl: tp.Callable[..., Node]):
        def wrapper2(*args):
            return composite(
                spec=spec,
                impl=impl,
                args=args,
                name=name,
                c_inline=c_inline,
                c_lowering=c_lowering,
            )
        return wrapper2
    return wrapper1

class composite(Node):
    def __init__(
        self,
        spec: tp.Callable[..., tp.Any],
        impl: tp.Callable[..., Node],
        args: list[Node],
        name: str,
        c_inline: bool = False,
        c_lowering: tp.Optional[CLowering] = None,
    ):
        self.c_inline = c_inline
        self.c_lowering = c_lowering
        self.ctx = SpecContext(name)
        self.inner_args = [Var(name=f"arg_{i}", sign=x.node_type.copy()) for i, x in enumerate(args)]
        
        recorder = SpecRecorder(self.ctx)
        with record_specs(recorder):
            self.inner_tree = impl(*self.inner_args)
        
        self._validate_components(name)
        
        def impl_(*args):
            for var, arg in zip(self.inner_args, args):
                var.load_val(arg)
            return self.inner_tree.evaluate()
        
        # Signature is obtained from the inner tree
        def sign(*args):
            return self.inner_tree.node_type
        
        sign = make_fixed_arguments(
            sign,
            arg_types=[type(x.node_type) for x in args],
            return_type=type(self.inner_tree.node_type),
        )
        
        super().__init__(
            spec=spec,
            impl=impl_,
            sign=sign,
            args=args,
            name=name,
        )
    
    def check_spec(
        self,
        schedule: list[str | dict[str, tp.Any]] | None = None,
        observer: VerificationObserver | None = None,
        max_workers: int | None = None,
    ):
        base_ctx = self.ctx.copy()
        inputs = [base_ctx.spec_of(arg) for arg in self.inner_args]

        def collect_inner(ctx):
            encoded_inputs = [
                special_encoding(value, ctx)
                for value in inputs
            ]
            for node, value in zip(self.inner_args, encoded_inputs):
                ctx.spec_cache[node] = value
            return ctx.spec_of(self.inner_tree)

        def collect_outer(ctx):
            encoded_inputs = [
                special_encoding(value, ctx)
                for value in inputs
            ]
            return self.spec(*encoded_inputs, ctx=ctx)

        result = check_equivalence(
            _Spec("inner_spec", collect_inner),
            _Spec("outer_spec", collect_outer, partition_cases=True),
            base_ctx=base_ctx,
            inputs=inputs,
            schedule=schedule,
            observer=observer,
            max_workers=max_workers,
        )
        
        print(f"{self.ctx.name} {'has' if result['proved'] else 'has not'} been proved")
        
        return result

    def check_determinism(
        self,
        schedule: list[str | dict[str, tp.Any]] | None = None,
        observer: VerificationObserver | None = None,
        max_workers: int | None = None,
    ):
        return _check_determinism(
            self,
            schedule=schedule,
            observer=observer,
            max_workers=max_workers,
        )
    
    def _validate_components(self, composite_name: str) -> None:
        visited: set[Node] = set()
        
        def visit(node: Node, path: str) -> None:
            if node in visited:
                return
            visited.add(node)
            
            if isinstance(node, (primitive, composite)):
                for idx, arg in enumerate(node.args):
                    visit(arg, f"{path} -> {node.name}.arg[{idx}]")
                return
            
            if isinstance(node, (Var, Const)):
                return
            
            raise TypeError(
                f"Composite {composite_name} must be composed recursively of Primitive/Composite nodes; "
                f"found {type(node).__name__} {node.name!r} at {path}"
            )
        
        visit(self.inner_tree, f"{composite_name}.impl")
    
    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + f"{self.node_type}: {self.name} [Composite]")
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        if depth > 0:
            print(new_prefix + "└── Impl:")
            self.inner_tree.print_tree(new_prefix + "    ", True, depth - 1)
        else:
            for i, arg in enumerate(self.args):
                is_arg_last = i == len(self.args) - 1
                arg.print_tree(new_prefix, is_arg_last, depth)
    
    def __str__(self):
        return f"[Composite] {self.name}: {' -> '.join([str(x) for x in self.args_types])} -> {self.node_type}"
    
    def _fingerprint(self, jittable: bool = False):
        def build():
            direct_cpp_lowering = None
            if self.c_lowering is not None:
                direct_cpp_lowering = self.c_lowering(
                    [f"${idx}" for idx in range(len(self.args))],
                    jittable,
                )
            return (
                type(self).__name__,
                self.name,
                self.node_type._fingerprint(),
                tuple(arg.node_type._fingerprint() for arg in self.inner_args),
                direct_cpp_lowering,
                self.inner_tree._fingerprint(jittable) if direct_cpp_lowering is None else None,
            )
        
        return self._cached_fingerprint(jittable, build)


def Primitive(
    name: str,
    spec: tp.Callable[..., tp.Any],
    c_inline: bool = False,
    c_lowering: tp.Optional[CLowering] = None,
):
    def wrapper1(impl: tp.Callable[..., Node]):
        def wrapper2(*args):
            return primitive(
                spec=spec,
                impl=impl,
                args=args,
                name=name,
                c_inline=c_inline,
                c_lowering=c_lowering,
            )
        return wrapper2
    return wrapper1

class primitive(Node):
    def __init__(
        self,
        spec: tp.Callable[..., tp.Any],
        impl: tp.Callable[..., Node],
        args: list[Node],
        name: str,
        c_inline: bool = False,
        c_lowering: tp.Optional[CLowering] = None,
    ):
        self.c_inline = c_inline
        self.c_lowering = c_lowering
        # Args will preserve runtime values of arguments
        self.inner_args = [Var(name=f"arg_{i}", sign=x.node_type.copy()) for i, x in enumerate(args)]
        
        self.inner_tree = impl(*self.inner_args)
        
        def impl_(*args):
            for var, arg in zip(self.inner_args, args):
                if isinstance(var, Var):
                    var.load_val(arg)
            return self.inner_tree.evaluate()
        
        # Signature is obtained from the inner tree
        def sign(*args):
            return self.inner_tree.node_type
        
        sign = make_fixed_arguments(
            sign,
            arg_types=[type(x.node_type) for x in args],
            return_type=type(self.inner_tree.node_type),
        )
        
        super().__init__(
            spec=spec,
            impl=impl_,
            sign=sign,
            args=args,
            name=name,
        )

    def check_determinism(
        self,
        schedule: list[str | dict[str, tp.Any]] | None = None,
        observer: VerificationObserver | None = None,
        max_workers: int | None = None,
    ):
        return _check_determinism(
            self,
            schedule=schedule,
            observer=observer,
            max_workers=max_workers,
        )
    
    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + f"{self.node_type}: {self.name} [Primitive]")
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        if depth > 0:
            print(new_prefix + "└── Impl:")
            self.inner_tree.print_tree(new_prefix + "    ", True, depth - 1)
        else:
            for i, arg in enumerate(self.args):
                is_arg_last = i == len(self.args) - 1
                arg.print_tree(new_prefix, is_arg_last, depth)
    
    def __str__(self):
        return f"[Primitive] {self.name}: {' -> '.join([str(x) for x in self.args_types])} -> {self.node_type}"
    
    def _fingerprint(self, jittable: bool = False):
        def build():
            direct_cpp_lowering = None
            if self.c_lowering is not None:
                direct_cpp_lowering = self.c_lowering(
                    [f"${idx}" for idx in range(len(self.args))],
                    jittable,
                )
            return (
                type(self).__name__,
                self.name,
                self.node_type._fingerprint(),
                tuple(arg.node_type._fingerprint() for arg in self.inner_args),
                direct_cpp_lowering,
                self.inner_tree._fingerprint(jittable) if direct_cpp_lowering is None else None,
            )
        
        return self._cached_fingerprint(jittable, build)


class Op(Node):
    def __init__(
        self,
        impl: tp.Callable[..., RuntimeType],
        sign: tp.Callable[..., StaticType],
        args: list[Node],
        name: str,
        c_lowering: tp.Optional[CLowering],
    ):
        self.c_lowering = c_lowering
        super().__init__(
            spec=None,
            impl=impl,
            sign=sign,
            args=args,
            name=name,
        )
    
    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + self.__str__())
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, arg in enumerate(self.args):
            is_arg_last = i == len(self.args) - 1
            arg.print_tree(new_prefix, is_arg_last, depth)
    
    def __str__(self):
        return f"{self.node_type}: {self.name} [Op]"
    
    def _fingerprint(self, jittable: bool = False):
        def build():
            lowering_fingerprint = None
            if self.c_lowering is not None:
                lowering_fingerprint = self.c_lowering(
                    [f"${idx}" for idx in range(len(self.args))],
                    jittable,
                )
            return (
                "Op",
                self.name,
                self.node_type._fingerprint(),
                lowering_fingerprint,
                tuple(arg._fingerprint(jittable) for arg in self.args),
            )
        
        return self._cached_fingerprint(jittable, build)


class Const(Node):
    def __init__(
        self,
        val: RuntimeType,
    ):
        self.val = val
        
        def impl():
            return self.val
        
        def spec(ctx):
            return self.val.to_spec(ctx)
        
        def sign() -> StaticType:
            return self.val.static_type()
        
        super().__init__(
            spec=spec,
            impl=impl,
            sign=sign,
            args=[],
            name=str(self.val.to_val()),
        )
        
        self.node_type.runtime_val = self.val.copy()  # Constant folding
    
    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + self.__str__())
    
    def __str__(self):
        return f"{self.node_type}: {self.name if self.name else str(self.val)} [Const]"
    
    def _fingerprint(self, jittable: bool = False):
        return self._cached_fingerprint(
            jittable,
            lambda: ("Const", self.val._fingerprint()),
        )


class Var(Node):
    def __init__(self, name: str, sign: StaticType):
        self.val = None
        
        def impl():
            if self.val is None:
                raise ValueError(f"Variable {self.name} not bound to a value")
            return self.val
        
        def spec(ctx):
            return sign.to_spec(self.name, ctx)
        
        def signature() -> StaticType:
            return sign
        
        super().__init__(
            spec=spec,
            impl=impl,
            sign=signature,
            args=[],
            name=name,
        )
    
    def load_rand(self, rng: tp.Optional[random.Random] = None):
        if rng is None:
            rng = random.Random()
        self.load_val(self.sign().random_runtime_value(rng))
    
    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + f"{self.node_type}: {self.name} [Var]")
    
    def load_val(self, val: RuntimeType):
        if not isinstance(val, RuntimeType):
            raise TypeError(f"Var's val must be a RuntimeType, {val} is provided")
        if val.static_type() != self.sign():
            raise TypeError(f"Var's val does not match signature {self.sign()}, {val.static_type()} is provided")
        self.val = val
    
    def __str__(self):
        return f"{self.node_type}: {self.name} [Var]"
    
    def _fingerprint(self, jittable: bool = False):
        return self._cached_fingerprint(
            jittable,
            lambda: ("Var", self.name, self.node_type._fingerprint()),
        )
    
