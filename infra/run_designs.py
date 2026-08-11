from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DESIGN_TIMEOUT_S = 30 # 10 * 60
DEFAULT_REPORT_PATH = Path("reports/run_designs.json")
CHECK_NAMES = ("determinism", "specification")
PROCESS_TERMINATION_GRACE_S = 5
RUNNER_PATH = Path(__file__).resolve()
PROJECT_ROOT = RUNNER_PATH.parent.parent

# Direct execution adds infra/, rather than the repository root, to sys.path.
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))


from examples.CSA import CSA_tree4
from examples.bf16_add import bf16_add
from examples.bf16_mult import bf16_mult
from examples.bf16_relu import bf16_relu
from examples.bf16x8_dot_fp32_conventional import bf16x8_dot_fp32_conventional
from examples.bf16x8_dot_fp32_optimized import bf16x8_dot_fp32_optimized
from examples.converters import (
    CONVERTER_FORMATS,
    CONVERTER_REGISTRY,
    FORMAT_STATIC_TYPES,
)
from examples.fp32_add import fp32_add
from examples.fp32_mult import fp32_mult
from zolotone import BFloat16T, Float32T, Node, QT, Var
from zolotone.solver import CaseVerificationResult


@dataclass(frozen=True)
class DesignCase:
    name: str
    build: Callable[[], Node]


def _build_csa_tree4() -> Node:
    args = [Var(name=f"a_{index}", sign=QT(10, 10)) for index in range(4)]
    return CSA_tree4(*args)


def _build_bf16_add() -> Node:
    return bf16_add(
        Var(name="a", sign=BFloat16T()),
        Var(name="b", sign=BFloat16T()),
    )


def _build_bf16_mult() -> Node:
    return bf16_mult(
        Var(name="a", sign=BFloat16T()),
        Var(name="b", sign=BFloat16T()),
    )


def _build_bf16_relu() -> Node:
    return bf16_relu(Var(name="x", sign=BFloat16T()))


def _build_converter(name: str) -> Node:
    source_name, _ = CONVERTER_FORMATS[name]
    source = Var(name="x", sign=FORMAT_STATIC_TYPES[source_name]())
    return CONVERTER_REGISTRY[name](source)


def _converter_design_case(name: str) -> DesignCase:
    return DesignCase(name, lambda name=name: _build_converter(name))


def _build_fp32_add() -> Node:
    return fp32_add(
        Var(name="a", sign=Float32T()),
        Var(name="b", sign=Float32T()),
    )


def _build_fp32_mult() -> Node:
    return fp32_mult(
        Var(name="a", sign=Float32T()),
        Var(name="b", sign=Float32T()),
    )


def _dot_product_args() -> list[Var]:
    return [
        *[Var(name=f"a_{index}", sign=BFloat16T()) for index in range(4)],
        *[Var(name=f"b_{index}", sign=BFloat16T()) for index in range(4)],
    ]


def _build_conventional_dot_product() -> Node:
    return bf16x8_dot_fp32_conventional(*_dot_product_args())


def _build_optimized_dot_product() -> Node:
    return bf16x8_dot_fp32_optimized(*_dot_product_args())


DESIGNS = (
    DesignCase("CSA_tree4", _build_csa_tree4),
    DesignCase("bf16_add", _build_bf16_add),
    DesignCase("bf16_mult", _build_bf16_mult),
    DesignCase("bf16_relu", _build_bf16_relu),
    *(_converter_design_case(name) for name in CONVERTER_REGISTRY),
    DesignCase("fp32_add", _build_fp32_add),
    DesignCase("fp32_mult", _build_fp32_mult),
    DesignCase("bf16x8_dot_fp32_conventional", _build_conventional_dot_product),
    DesignCase("bf16x8_dot_fp32_optimized", _build_optimized_dot_product),
)


def _find_design(name: str) -> DesignCase:
    return next(design for design in DESIGNS if design.name == name)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class CompletedCaseJournal:
    """Durably records cases after all work for the case has finished."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def case_completed(self, result: CaseVerificationResult) -> None:
        event = {"case_name": result["name"], "result": result.to_json()}
        self._handle.write(json.dumps(event, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def check_design(
    design_case: DesignCase,
    *,
    check_name: str,
    result_path: Path,
    completed_cases_path: Path,
) -> bool:
    started_at = time.perf_counter()
    design = design_case.build()

    print(f"Built {design_case.name}: {design.node_type}", flush=True)
    checks = {
        "determinism": design.check_determinism,
        "specification": design.check_spec,
    }
    observer = CompletedCaseJournal(completed_cases_path)
    print(f"Checking {design_case.name} {check_name}...", flush=True)
    try:
        result = checks[check_name](observer=observer)
    finally:
        observer.close()

    proved = result["proved"]
    _write_json(
        result_path,
        {
            "name": design_case.name,
            "status": "passed" if proved else "failed",
            "elapsed_s": time.perf_counter() - started_at,
            "checks": {check_name: result.to_json()},
        },
    )
    if not proved:
        print(
            f"[FAIL] {design_case.name} {check_name} was not proved",
            file=sys.stderr,
        )
    return proved


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _read_completed_cases(path: Path) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as journal:
        for line in journal:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            cases[event["case_name"]] = event["result"]

    return cases


def _run_design_subprocess(
    name: str,
    check_name: str,
    timeout_s: float,
) -> tuple[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="zolotone-run-design-") as temp_dir:
        result_path = Path(temp_dir) / "result.json"
        completed_cases_path = Path(temp_dir) / "completed_cases.jsonl"
        started_at = time.perf_counter()
        process = subprocess.Popen(
            [
                sys.executable,
                str(RUNNER_PATH),
                "--design",
                name,
                "--check",
                check_name,
                "--result-file",
                str(result_path),
                "--completed-cases-file",
                str(completed_cases_path),
            ],
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_s)
            status = "passed" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            status = "timeout"
        except KeyboardInterrupt:
            _terminate_process_group(process)
            status = "interrupted"
        elapsed_s = time.perf_counter() - started_at
        if result_path.exists():
            design_result = json.loads(result_path.read_text(encoding="utf-8"))
            check_result = design_result["checks"][check_name]
        else:
            check_result = {
                "status": "error",
                "proved": False,
                "cases": _read_completed_cases(completed_cases_path),
            }
        if status in {"timeout", "interrupted"}:
            check_result.update(status=status, proved=False)
        check_result["elapsed_s"] = elapsed_s
        return status, check_result


def _design_status(checks: dict[str, Any]) -> str:
    statuses = [checks[name]["status"] for name in CHECK_NAMES]
    if "timeout" in statuses:
        return "timeout"
    if all(
        checks[name]["status"] == "passed" and checks[name]["proved"]
        for name in CHECK_NAMES
    ):
        return "passed"
    return "failed"


def run_designs(
    designs: Iterable[DesignCase] = DESIGNS,
    *,
    timeout_s: float = DEFAULT_DESIGN_TIMEOUT_S,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> int:
    report_path = Path(report_path)
    report = {
        "started_at": _utc_timestamp(),
        "finished_at": None,
        "timeout_s": float(timeout_s),
        "designs": {},
    }
    _write_json(report_path, report)

    total = 0
    succeeded = 0
    failed = {}

    for design_case in designs:
        total += 1
        design_started_at = time.perf_counter()
        design_result = {"name": design_case.name, "checks": {}}

        for check_name in CHECK_NAMES:
            print(
                f"[RUN] {design_case.name} {check_name} "
                f"(timeout: {timeout_s:g} seconds)",
                flush=True,
            )
            worker_status, check_result = _run_design_subprocess(
                design_case.name,
                check_name,
                timeout_s,
            )
            design_result["checks"][check_name] = check_result

            if worker_status == "timeout":
                print(
                    f"[TIMEOUT] {design_case.name} {check_name} exceeded "
                    f"{timeout_s:g} seconds",
                    file=sys.stderr,
                    flush=True,
                )
            elif worker_status == "interrupted":
                design_result["status"] = "interrupted"
                design_result["elapsed_s"] = (
                    time.perf_counter() - design_started_at
                )
                report["designs"][design_case.name] = design_result
                report["status"] = "interrupted"
                report["finished_at"] = _utc_timestamp()
                _write_json(report_path, report)
                raise KeyboardInterrupt

        status = _design_status(design_result["checks"])
        design_result["status"] = status
        design_result["elapsed_s"] = time.perf_counter() - design_started_at
        report["designs"][design_case.name] = design_result
        _write_json(report_path, report)

        if status == "passed":
            succeeded += 1
            print(f"[PASS] {design_case.name}", flush=True)
        elif status == "timeout":
            failed[design_case.name] = status
            print(f"[FAIL] {design_case.name} (timeout)", file=sys.stderr, flush=True)
        else:
            failed[design_case.name] = status
            print(f"[FAIL] {design_case.name}", file=sys.stderr, flush=True)

    report["status"] = "failed" if failed else "passed"
    report["finished_at"] = _utc_timestamp()
    _write_json(report_path, report)

    print(f"Passed {succeeded}/{total} designs.")
    print(f"JSON report: {report_path}")
    if failed:
        details = ", ".join(
            f"{name} ({status})" for name, status in failed.items()
        )
        print(f"Unsuccessful designs: {details}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and verify example designs")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"JSON report path (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--design",
        choices=[design_case.name for design_case in DESIGNS],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--check",
        choices=CHECK_NAMES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--completed-cases-file",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.design is not None:
        return 0 if check_design(
            _find_design(args.design),
            check_name=args.check,
            result_path=args.result_file,
            completed_cases_path=args.completed_cases_file,
        ) else 1
    return run_designs(report_path=args.report)


if __name__ == "__main__":
    raise SystemExit(main())
