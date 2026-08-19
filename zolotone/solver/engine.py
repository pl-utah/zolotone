from __future__ import annotations

from typing import Any

from ..spec import SpecContext
from ..spec.spec_context import simplify_ctx
from .report import ProofReport, validate_proof_status
from ..egglog import egglog_rewrite
from ..smt import z3_check_eq, dreal_check_eq


DEFAULT_REWRITE_ITERS = 6
DEFAULT_Z3_TIMEOUT = 10000
DEFAULT_DREAL_PRECISION = 0.001
DEFAULT_EGGLOG_MATCH_LIMIT = 100000
DEFAULT_EGGLOG_BAN_LENGTH = 1

TOOL_FNS = {
    "simplify": simplify_ctx,
    "egglog-rewrite": egglog_rewrite,
    "z3": z3_check_eq,
    "dreal": dreal_check_eq,
}


def _normalize_egglog_scheduler(step: dict[str, Any]) -> dict[str, int | None]:
    scheduler = step.get("scheduler")
    if scheduler is None:
        return None
    if not isinstance(scheduler, dict):
        raise TypeError("Schedule step 'scheduler' must be a dict")

    match_limit = scheduler.get("match_limit", DEFAULT_EGGLOG_MATCH_LIMIT)
    ban_length = scheduler.get("ban_length", DEFAULT_EGGLOG_BAN_LENGTH)

    if match_limit is not None:
        match_limit = int(match_limit)
    if ban_length is not None:
        ban_length = int(ban_length)

    return {
        "match_limit": match_limit,
        "ban_length": ban_length,
    }


def _normalize_schedule(
    schedule: list[str | dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for step in schedule:
        if not isinstance(step, dict):
            raise TypeError("Each schedule step must be a dict")

        if "tool" not in step:
            raise ValueError("Each schedule step must define a 'tool'")

        tool = step["tool"]
        if not isinstance(tool, str):
            raise TypeError("Schedule step 'tool' must be a string")

        if TOOL_FNS.get(tool) is None:
            raise ValueError(
                f"Unknown schedule tool {step['tool']}. Supported aliases: {list(TOOL_FNS.keys())}"
            )

        if tool == "simplify":
            normalized.append({"tool": tool})
        elif tool == "egglog-rewrite":
            normalized.append(
                {
                    "tool": tool,
                    "iterations": int(step.get("iterations", DEFAULT_REWRITE_ITERS)),
                    "scheduler": _normalize_egglog_scheduler(step),
                }
            )
        elif tool == "z3":
            normalized.append(
                {"tool": tool, "timeout_ms": int(step.get("timeout_ms", DEFAULT_Z3_TIMEOUT))}
            )
        elif tool == "dreal":
            normalized.append(
                {"tool": tool, "precision": float(step.get("precision", DEFAULT_DREAL_PRECISION))}
            )

    return normalized


def _run_tool(ctx: SpecContext, step: dict[str, Any]):
    tool = step["tool"]
    tool_fn = TOOL_FNS[tool]
    kwargs = {key: value for key, value in step.items() if key != "tool"}
    return _normalize_tool_reports(tool_fn(ctx, **kwargs))


def _normalize_tool_reports(
    tool_result: ProofReport | list[ProofReport],
) -> list[ProofReport]:
    if isinstance(tool_result, dict):
        reports = [tool_result]
    elif isinstance(tool_result, list):
        reports = tool_result
    else:
        raise TypeError(
            "Each tool must return a ProofReport or a list of ProofReports"
        )

    for report in reports:
        if "new_ctx" not in report:
            raise KeyError("Each tool report must include 'new_ctx'")
        if "status" not in report:
            raise KeyError("Each tool report must include 'status'")
        if "equivalent" in report:
            raise KeyError("Tool reports must use 'status', not 'equivalent'")
        validate_proof_status(report["status"])
    return reports


def check_equivalence(
    ctx: SpecContext,
    schedule: list[str | dict[str, Any]],
):
    # if ctx.name != "bf16x8_dot_fp32_conventional[path=3,output=norm]":
    #      return "unknown", [{'feasibility_status': 'unknown', 'status': 'unknown', 'tool':"simplfiy"}]
    current_tracks: list[list[ProofReport]] = [[]]
    current_ctxs = [ctx.copy()]
    normalized_schedule = _normalize_schedule(schedule=schedule)
    for step in normalized_schedule:
        next_tracks: list[list[ProofReport]] = []
        next_ctxs: list[SpecContext] = []

        for current_ctx, current_track in zip(current_ctxs, current_tracks):
            reports = _run_tool(current_ctx, step)
            for report in reports:
                next_track = current_track + [report]
                status = report["status"]
                if status in {"sat", "unsat"}:
                    return status, next_track
                next_tracks.append(next_track)
                next_ctxs.append(report["new_ctx"])
            

        current_tracks = next_tracks
        current_ctxs = next_ctxs

    if not current_tracks:
        return "unknown", []
    return "unknown", current_tracks[0]
