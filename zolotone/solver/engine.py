from __future__ import annotations

import multiprocessing
import pickle
from multiprocessing.connection import wait
from time import perf_counter
from typing import Any

from ..spec import SpecContext
from ..spec.spec_context import simplify_ctx
from .report import ProofReport, build_proof_report, validate_proof_status
from ..egglog import egglog_rewrite
from ..smt import z3_check_eq, dreal_check_eq


DEFAULT_REWRITE_ITERS = 6
DEFAULT_Z3_TIMEOUT = 10000
DEFAULT_DREAL_PRECISION = 0.001
DEFAULT_EGGLOG_MATCH_LIMIT = 100000
DEFAULT_EGGLOG_BAN_LENGTH = 1
DEFAULT_TOOL_TIMEOUT_S = 60.0
MAX_TOOL_REPORT_BYTES = 8 * 1024 * 1024
HARD_TIMEOUT_TOOLS = frozenset({"z3", "dreal"})


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


def _run_tool_worker(
    pipe,
    tool: str,
    tool_fn,
    ctx: SpecContext,
    kwargs: dict[str, Any],
    max_report_bytes: int,
):
    try:
        reports = _normalize_tool_reports(tool_fn(ctx, **kwargs))
        payload = pickle.dumps(("ok", reports), protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > max_report_bytes:
            raise ValueError(
                f"{tool} report is {len(payload)} bytes; "
                f"maximum is {max_report_bytes} bytes"
            )
        pipe.send_bytes(payload)
    except BaseException as exc:
        pipe.send_bytes(pickle.dumps(("error", repr(exc))))
    finally:
        pipe.close()


def _run_tool(ctx: SpecContext, step: dict[str, Any], timeout=DEFAULT_TOOL_TIMEOUT_S):
    tool = step["tool"]
    tool_fn = TOOL_FNS[tool]
    kwargs = {key: value for key, value in step.items() if key != "tool"}

    if tool not in HARD_TIMEOUT_TOOLS:
        return _normalize_tool_reports(tool_fn(ctx, **kwargs))

    process_ctx = multiprocessing.get_context("spawn")
    parent_pipe, child_pipe = process_ctx.Pipe(duplex=False)
    process = process_ctx.Process(
        target=_run_tool_worker,
        args=(
            child_pipe,
            tool,
            tool_fn,
            ctx,
            kwargs,
            MAX_TOOL_REPORT_BYTES,
        ),
    )

    started_at = perf_counter()
    started = False
    try:
        process.start()
        started = True
        child_pipe.close()
        ready = wait([parent_pipe, process.sentinel], timeout=timeout)

        if not ready:
            process.terminate()
            process.join()
            return [
                build_proof_report(
                    ctx,
                    ctx.copy(),
                    tool=tool,
                    runtime_s=perf_counter() - started_at,
                    status="unknown",
                    wall_clock_timeout_s=timeout,
                    **kwargs,
                )
            ]

        if not parent_pipe.poll():
            process.join()
            raise RuntimeError(f"{tool} worker exited without returning a result")

        status, result = pickle.loads(parent_pipe.recv_bytes())
        process.join()
        if status == "error":
            raise RuntimeError(f"{tool} worker failed: {result}")
        return result
    finally:
        child_pipe.close()
        parent_pipe.close()
        if started:
            if process.is_alive():
                process.terminate()
            process.join()


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
    current_tracks: list[list[ProofReport]] = [[]]
    current_ctxs = [ctx.copy()]

    normalized_schedule = _normalize_schedule(schedule=schedule)
    for step in normalized_schedule:
        next_tracks: list[list[ProofReport]] = []
        next_ctxs: list[SpecContext] = []

        for current_ctx, current_track in zip(current_ctxs, current_tracks):
            reports = _run_tool(current_ctx, step, timeout=DEFAULT_TOOL_TIMEOUT_S)
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
