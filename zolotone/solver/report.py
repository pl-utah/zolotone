from __future__ import annotations

from collections import Counter
from typing import Any, Final, Literal, Protocol, cast


ProofStatus = Literal["sat", "unsat", "unknown"]
VALID_PROOF_STATUSES: Final = frozenset({"sat", "unsat", "unknown"})
COUNT_FIELDS: Final = (
    "before",
    "after",
    "unchanged",
    "simplified",
    "discharged",
    "added",
)
TOOL_METADATA_FIELDS: Final = (
    "timeout_ms",
    "wall_clock_timeout_s",
    "precision",
    "iterations_used",
    "egraph_size",
    "rule_application_counts",
)


class ProofReport(dict[str, Any]):
    def to_json(self, *, phase: str) -> dict[str, Any]:
        result = {
            "phase": phase,
            "tool": str(self["tool"]),
            "elapsed_s": float(self.get("runtime_s", 0.0)),
            "status": str(self["status"]),
            "feasibility": self.get("feasibility_status") or "unknown",
            "assumes": _constraint_counts(self, "assumes"),
            "checks": _constraint_counts(self, "checks"),
            "context_nodes": {
                "before": int(self.get("context_nodes_before", 0)),
                "after": int(self.get("context_nodes_after", 0)),
            },
        }
        metadata = {
            field: self[field]
            for field in TOOL_METADATA_FIELDS
            if field in self
        }
        if metadata:
            result["metadata"] = metadata
        return result


class CaseVerificationResult(dict[str, Any]):
    def __init__(
        self,
        *,
        name: str,
        proved: bool,
        status: ProofStatus,
        feasibility_status: str,
        proof_trace: list[ProofReport],
        side_feasibility_reports: list[ProofReport],
    ) -> None:
        super().__init__(
            name=name,
            proved=proved,
            status=status,
            feasibility_status=feasibility_status,
            proof_trace=proof_trace,
            side_feasibility_reports=side_feasibility_reports,
        )

    def to_json(self) -> dict[str, Any]:
        tools = [
            report.to_json(phase="proof")
            for report in self["proof_trace"]
        ] + [
            report.to_json(phase="side_feasibility")
            for report in self["side_feasibility_reports"]
        ]
        return {
            "proved": bool(self["proved"]),
            "status": str(self["status"]),
            "feasibility": self["feasibility_status"] or "unknown",
            "elapsed_s": sum(tool["elapsed_s"] for tool in tools),
            "tools": tools,
        }


class VerificationObserver(Protocol):
    """Receives verification artifacts only after they have completed."""

    def proof_trace_completed(
        self,
        *,
        case_name: str,
        status: ProofStatus,
        proof_trace: list[ProofReport],
    ) -> None: ...

    def case_completed(self, result: CaseVerificationResult) -> None: ...


class CheckResult(dict[str, Any]):
    """Canonical verification result with legacy dictionary-style access."""

    def __init__(
        self,
        *,
        proved: bool,
        requirement_report: ProofReport | None,
        cases: list[CaseVerificationResult],
    ) -> None:
        super().__init__(
            proved=proved,
            proof_traces=[case["proof_trace"] for case in cases],
            requirement_report=requirement_report,
            case_results=cases,
        )

    def to_json(self) -> dict[str, Any]:
        setup_tools = []
        if self["requirement_report"] is not None:
            setup_tools.append(
                self["requirement_report"].to_json(
                    phase="requirement_validation"
                )
            )
        cases = {
            case["name"]: case.to_json()
            for case in self["case_results"]
        }
        return {
            "status": "passed" if self["proved"] else "failed",
            "proved": bool(self["proved"]),
            "elapsed_s": sum(
                tool["elapsed_s"] for tool in setup_tools
            ) + sum(case["elapsed_s"] for case in cases.values()),
            "tools": setup_tools,
            "cases": cases,
        }


def _constraint_counts(report: ProofReport, kind: str) -> dict[str, int]:
    return {
        field: int(report.get(f"{field}_{kind}", 0))
        for field in COUNT_FIELDS
    }


def validate_proof_status(status: object) -> ProofStatus:
    if not isinstance(status, str) or status not in VALID_PROOF_STATUSES:
        raise ValueError(
            f"Proof report status must be one of {sorted(VALID_PROOF_STATUSES)}, "
            f"got {status!r}"
        )
    return cast(ProofStatus, status)


def _count_unchanged_items(before: list[str], after: list[str]) -> int:
    return sum((Counter(before) & Counter(after)).values())


def merge_rule_application_counts(*counts_dicts: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counts in counts_dicts:
        for rule, count in counts.items():
            merged[rule] = merged.get(rule, 0) + int(count)
    return merged


def count_context_nodes(ctx: "SpecContext") -> int:
    """Count AST-node occurrences across every expression in a context."""
    # Import lazily because SpecContext imports this module to build reports.
    from ..spec.spec_ast import children

    stack = [
        expression
        for expressions in (ctx.assumes, ctx.checks, ctx.requirements)
        for expression in expressions
    ]
    count = 0
    while stack:
        count += 1
        stack.extend(children(stack.pop()))
    return count


def build_proof_report(
    old_ctx: "SpecContext",
    new_ctx: "SpecContext",
    tool: str,
    runtime_s: float,
    status: str,
    **extra: Any,
) -> ProofReport:
    assert old_ctx.name == new_ctx.name, "Trying to build proof report between two different designs"
    status = validate_proof_status(status)
    name = old_ctx.name

    assumes_before = len(old_ctx.assumes)
    assumes_after = len(new_ctx.assumes)
    checks_before = len(old_ctx.checks)
    checks_after = len(new_ctx.checks)

    added_assumes = max(0, assumes_after - assumes_before)
    added_checks = max(0, checks_after - checks_before)

    old_assumes = [str(assume) for assume in old_ctx.assumes]
    new_assumes = [str(assume) for assume in new_ctx.assumes]
    old_checks = [str(check) for check in old_ctx.checks]
    new_checks = [str(check) for check in new_ctx.checks]

    unchanged_assumes = _count_unchanged_items(old_assumes, new_assumes)
    unchanged_checks = _count_unchanged_items(old_checks, new_checks)

    discharged_checks = max(0, checks_before - checks_after)
    discharged_assumes = max(0, assumes_before - assumes_after)

    simplified_assumes = max(0, assumes_after - unchanged_assumes) - added_assumes
    simplified_checks = max(0, checks_after - unchanged_checks) - added_checks

    report = ProofReport(
        tool=tool,
        name=old_ctx.name if name is None else name,
        old_ctx=old_ctx,
        new_ctx=new_ctx,
        status=status,
        runtime_s=float(runtime_s),
        assumes_before=assumes_before,
        assumes_after=assumes_after,
        checks_before=checks_before,
        checks_after=checks_after,
        context_nodes_before=count_context_nodes(old_ctx),
        context_nodes_after=count_context_nodes(new_ctx),
        unchanged_assumes=unchanged_assumes,
        unchanged_checks=unchanged_checks,
        discharged_checks=discharged_checks,
        discharged_assumes=discharged_assumes,
        simplified_assumes=simplified_assumes,
        simplified_checks=simplified_checks,
        added_assumes=added_assumes,
        added_checks=added_checks,
    )
    assert unchanged_assumes + discharged_assumes + simplified_assumes == assumes_before
    assert unchanged_assumes + simplified_assumes + added_assumes == assumes_after

    assert unchanged_checks + discharged_checks + simplified_checks == checks_before
    assert unchanged_checks + simplified_checks + added_checks == checks_after
    report.update(extra)
    return report
