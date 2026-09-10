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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DESIGN_TIMEOUT_S = 5
DEFAULT_REPORT_PATH = Path("reports/run_designs.json")
CHECK_NAMES = ("determinism", "specification")
PROCESS_TERMINATION_GRACE_S = 5
RUNNER_PATH = Path(__file__).resolve()
PROJECT_ROOT = RUNNER_PATH.parent.parent

# Direct execution adds infra/, rather than the repository root, to sys.path.
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))


from examples.CSA import CSA_tree4
from examples.arithmetic.bf16_add import bf16_add
from examples.arithmetic.bf16_mult import bf16_mult
from examples.arithmetic.bf16_relu import bf16_relu
from examples.arithmetic.fp32_add import fp32_add
from examples.arithmetic.fp32_mult import fp32_mult
from examples.arithmetic.ue4m3x2_e2m1x2_add_fp32 import ue4m3x2_e2m1x2_add_fp32
from examples.arithmetic.ue4m3x2_e2m1x2_mult_fp32 import (
    ue4m3x2_e2m1x2_mult_fp32,
)
from examples.converters import (
    CONVERTER_FORMATS,
    CONVERTER_REGISTRY,
    FORMAT_STATIC_TYPES,
)
from examples.dot_product.bf16x8_dot_fp32_conventional import (
    bf16x8_dot_fp32_conventional,
)
from examples.dot_product.bf16x8_dot_fp32_optimized import bf16x8_dot_fp32_optimized
from examples.dot_product.wgmma_fp16_e4m3_e5m2 import wgmma_fp16_e4m3_e5m2
from examples.dot_product.wgmma_fp32_e4m3_e4m3 import wgmma_fp32_e4m3_e4m3
from examples.dot_product.wgmma_fp32_e5m2_e4m3 import wgmma_fp32_e5m2_e4m3
from zolotone import (
    BFloat16T,
    E2M1T,
    E4M3FNT,
    E5M2T,
    Float16T,
    Float32T,
    Node,
    QT,
    UE4M3T,
    Var,
)
from zolotone.ast.parallel_verification import MAX_WORKERS_ENV
from zolotone.solver import CaseVerificationResult


@dataclass(frozen=True)
class DesignCase:
    name: str
    build: Callable[[], Node]
    category: str = "arithmetic"


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
    return DesignCase(
        name,
        lambda name=name: _build_converter(name),
        category="converter",
    )


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


def _build_ue4m3x2_e2m1x2_mult_fp32() -> Node:
    return ue4m3x2_e2m1x2_mult_fp32(
        Var(name="a0", sign=UE4M3T()),
        Var(name="a1", sign=UE4M3T()),
        Var(name="b0", sign=E2M1T()),
        Var(name="b1", sign=E2M1T()),
    )


def _build_ue4m3x2_e2m1x2_add_fp32() -> Node:
    return ue4m3x2_e2m1x2_add_fp32(
        Var(name="scale0", sign=UE4M3T()),
        Var(name="scale1", sign=UE4M3T()),
        Var(name="x0", sign=E2M1T()),
        Var(name="x1", sign=E2M1T()),
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


def _build_wgmma_fp32_e4m3_e4m3() -> Node:
    return wgmma_fp32_e4m3_e4m3(
        *[Var(name=f"a{index}", sign=E4M3FNT()) for index in range(4)],
        *[Var(name=f"b{index}", sign=E4M3FNT()) for index in range(4)],
        Var(name="c", sign=Float32T()),
    )


def _build_wgmma_fp32_e5m2_e4m3() -> Node:
    return wgmma_fp32_e5m2_e4m3(
        *[Var(name=f"a{index}", sign=E5M2T()) for index in range(4)],
        *[Var(name=f"b{index}", sign=E4M3FNT()) for index in range(4)],
        Var(name="c", sign=Float32T()),
    )


def _build_wgmma_fp16_e4m3_e5m2() -> Node:
    return wgmma_fp16_e4m3_e5m2(
        *[Var(name=f"a{index}", sign=E4M3FNT()) for index in range(4)],
        *[Var(name=f"b{index}", sign=E5M2T()) for index in range(4)],
        Var(name="c", sign=Float16T()),
    )


DESIGNS = (
    DesignCase("CSA_tree4", _build_csa_tree4),
    DesignCase("bf16_add", _build_bf16_add),
    DesignCase("bf16_mult", _build_bf16_mult),
    DesignCase("bf16_relu", _build_bf16_relu),
    *(_converter_design_case(name) for name in CONVERTER_REGISTRY),
    DesignCase("fp32_add", _build_fp32_add),
    DesignCase("fp32_mult", _build_fp32_mult),
    DesignCase(
        "ue4m3x2_e2m1x2_add_fp32",
        _build_ue4m3x2_e2m1x2_add_fp32,
    ),
    DesignCase(
        "ue4m3x2_e2m1x2_mult_fp32",
        _build_ue4m3x2_e2m1x2_mult_fp32,
    ),
    DesignCase(
        "bf16x8_dot_fp32_conventional",
        _build_conventional_dot_product,
        category="dot_product",
    ),
    DesignCase(
        "bf16x8_dot_fp32_optimized",
        _build_optimized_dot_product,
        category="dot_product",
    ),
    DesignCase(
        "wgmma_fp32_e4m3_e4m3",
        _build_wgmma_fp32_e4m3_e4m3,
        category="dot_product",
    ),
    DesignCase(
        "wgmma_fp32_e5m2_e4m3",
        _build_wgmma_fp32_e5m2_e4m3,
        category="dot_product",
    ),
    DesignCase(
        "wgmma_fp16_e4m3_e5m2",
        _build_wgmma_fp16_e4m3_e5m2,
        category="dot_product",
    ),
)


def _find_design(name: str) -> DesignCase:
    return next(design for design in DESIGNS if design.name == name)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2) + "\n",
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
        try:
            result = checks[check_name](observer=observer)
        except Exception as exc:
            elapsed_s = time.perf_counter() - started_at
            error_result = {
                "name": design_case.name,
                "status": "error",
                "elapsed_s": elapsed_s,
                "checks": {
                    check_name: {
                        "status": "error",
                        "proved": False,
                        "cases": _read_completed_cases(completed_cases_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                },
            }
            _write_json(result_path, error_result)
            print(
                f"[ERROR] {design_case.name} {check_name}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return False
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
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return

    deadline = time.monotonic() + PROCESS_TERMINATION_GRACE_S
    while _process_group_exists(process_group_id):
        # Reap the leader as soon as it exits, but keep watching its process
        # group: a descendant may outlive the leader after SIGTERM.
        process.poll()
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(0.05, remaining_s))

    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # wait() is safe after poll() has already observed the exit and guarantees
    # that this coordinator reaps its direct child on every cleanup path.
    process.wait()


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(
    process: subprocess.Popen,
    signum: signal.Signals,
) -> None:
    """Signal a design group, even if its original leader has exited."""
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        process.poll()


@contextmanager
def _manage_design_process_signals(process: subprocess.Popen):
    """Keep a detached design group aligned with terminal job control."""
    previous_handlers = {}

    def interrupt_run(_signum, _frame) -> None:
        raise KeyboardInterrupt

    def suspend_run(_signum, _frame) -> None:
        _signal_process_group(process, signal.SIGSTOP)
        os.kill(os.getpid(), signal.SIGSTOP)
        _signal_process_group(process, signal.SIGCONT)

    handlers = {
        signal.SIGTERM: interrupt_run,
        signal.SIGHUP: interrupt_run,
        signal.SIGQUIT: interrupt_run,
        signal.SIGTSTP: suspend_run,
    }
    try:
        for signum, handler in handlers.items():
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


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


def _wait_for_design_process(
    process: subprocess.Popen,
    timeout_s: float,
) -> tuple[str | None, int]:
    """Wait for one check and clean up its complete process group on failure."""
    try:
        with _manage_design_process_signals(process):
            try:
                return None, process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                status = "timeout"
            except KeyboardInterrupt:
                status = "interrupted"
            _terminate_process_group(process)
    except BaseException:
        _terminate_process_group(process)
        raise

    return status, process.returncode


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
        status, returncode = _wait_for_design_process(process, timeout_s)
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
        elif (
            returncode == 0
            and check_result.get("status") == "passed"
            and check_result.get("proved") is True
        ):
            status = "passed"
        else:
            status = "failed"
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
        design_result = {
            "name": design_case.name,
            "category": design_case.category,
            "checks": {},
        }

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
        "--timeout",
        type=float,
        default=DEFAULT_DESIGN_TIMEOUT_S,
        help=(
            "wall-clock timeout in seconds for each determinism/specification "
            f"check (default: {DEFAULT_DESIGN_TIMEOUT_S})"
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help=(
            "maximum parallel classification workers; defaults to "
            f"${MAX_WORKERS_ENV}, then available CPUs"
        ),
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
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be at least one")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_workers is not None:
        os.environ[MAX_WORKERS_ENV] = str(args.max_workers)
    if args.design is not None:
        check_design(
            _find_design(args.design),
            check_name=args.check,
            result_path=args.result_file,
            completed_cases_path=args.completed_cases_file,
        )
        return 0
    run_designs(timeout_s=args.timeout, report_path=args.report)
    return 0

if __name__ == "__main__":
    main()
