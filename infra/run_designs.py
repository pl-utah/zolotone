from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DESIGN_TIMEOUT_S = 20 * 60
DEFAULT_REPORT_PATH = Path("report/run_designs.json")
REPORT_SCHEMA_VERSION = 1
PROCESS_TERMINATION_GRACE_S = 5
CHILD_PROCESS_TERMINATION_GRACE_S = 0.25
PARENT_PID_ENV = "ZOLOTONE_RUN_DESIGNS_PARENT_PID"
PR_SET_PDEATHSIG = 1

def _terminate_own_process_group(signum, _frame) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except ProcessLookupError:
        os._exit(128 + signum)

    time.sleep(CHILD_PROCESS_TERMINATION_GRACE_S)
    os.killpg(os.getpgrp(), signal.SIGKILL)


def _arm_parent_death_signal() -> None:
    raw_parent_pid = os.environ.pop(PARENT_PID_ENV, None)
    if raw_parent_pid is None:
        return
    expected_parent_pid = int(raw_parent_pid)
    if not sys.platform.startswith("linux"):
        return

    signal.signal(signal.SIGTERM, _terminate_own_process_group)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))

    # The parent may have exited between spawning this process and prctl().
    if os.getppid() != expected_parent_pid:
        _terminate_own_process_group(signal.SIGTERM, None)


# Arm child cleanup before importing the substantially larger design modules.
_arm_parent_death_signal()


from examples.CSA import CSA_tree4
from examples.bf16_add import bf16_add
from examples.bf16_mult import bf16_mult
from examples.bf16_relu import bf16_relu
from examples.bf16_to_fp32 import bf16_to_fp32
from examples.bf16x8_dot_fp32_conventional import bf16x8_dot_fp32_conventional
from examples.bf16x8_dot_fp32_optimized import bf16x8_dot_fp32_optimized
from examples.fp32_add import fp32_add
from examples.fp32_mult import fp32_mult
from examples.fp32_to_bf16 import fp32_to_bf16
from zolotone import BFloat16T, Float32T, Node, QT, Var
from zolotone.solver import CheckResult


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


def _build_bf16_to_fp32() -> Node:
    return bf16_to_fp32(Var(name="x", sign=BFloat16T()))


def _build_fp32_to_bf16() -> Node:
    return fp32_to_bf16(Var(name="x", sign=Float32T()))


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
    DesignCase("bf16_to_fp32", _build_bf16_to_fp32),
    DesignCase("fp32_to_bf16", _build_fp32_to_bf16),
    DesignCase("fp32_add", _build_fp32_add),
    DesignCase("fp32_mult", _build_fp32_mult),
    DesignCase("bf16x8_dot_fp32_conventional", _build_conventional_dot_product),
    DesignCase("bf16x8_dot_fp32_optimized", _build_optimized_dot_product),
)


def _find_design(name: str) -> DesignCase:
    for design_case in DESIGNS:
        if design_case.name == name:
            return design_case
    raise ValueError(f"Unknown design: {name}")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _error_details(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _initial_design_result(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "running",
        "elapsed_s": 0.0,
        "checks": {},
    }


def _checkpoint(path: Path | None, result: dict[str, Any]) -> None:
    if path is not None:
        _atomic_write_json(path, result)


def check_design(
    design_case: DesignCase,
    *,
    result_path: Path | None = None,
) -> bool:
    started_at = time.perf_counter()
    design_result = _initial_design_result(design_case.name)
    _checkpoint(result_path, design_result)

    try:
        design = design_case.build()
        if not isinstance(design, Node):
            raise TypeError(
                f"builder returned {type(design).__name__}, expected Node"
            )
    except Exception as exc:
        design_result.update(
            status="failed",
            elapsed_s=time.perf_counter() - started_at,
            error={"phase": "build", **_error_details(exc)},
        )
        _checkpoint(result_path, design_result)
        print(f"[ERROR] Could not build {design_case.name}", file=sys.stderr)
        traceback.print_exc()
        return False

    print(f"Built {design_case.name}: {design.node_type}", flush=True)
    proved = True
    checks = (
        ("determinism", design.check_determinism),
        ("specification", design.check_spec),
    )
    for check_name, check in checks:
        print(f"Checking {design_case.name} {check_name}...", flush=True)
        design_result["checks"][check_name] = {
            "status": "running",
            "proved": False,
        }
        design_result["elapsed_s"] = time.perf_counter() - started_at
        _checkpoint(result_path, design_result)
        try:
            result = check()
            if not isinstance(result, CheckResult):
                raise TypeError(
                    f"{check_name} check returned {type(result).__name__}, "
                    "expected CheckResult"
                )
            serialized_result = result.to_json()
        except Exception as exc:
            proved = False
            design_result["checks"][check_name] = {
                "status": "error",
                "proved": False,
                "error": _error_details(exc),
            }
            print(
                f"[ERROR] {design_case.name} {check_name} check raised an exception",
                file=sys.stderr,
            )
            traceback.print_exc()
        else:
            check_proved = result.get("proved") is True
            proved = proved and check_proved
            design_result["checks"][check_name] = serialized_result
            if not check_proved:
                print(
                    f"[FAIL] {design_case.name} {check_name} was not proved",
                    file=sys.stderr,
                )

        design_result["elapsed_s"] = time.perf_counter() - started_at
        _checkpoint(result_path, design_result)

    design_result.update(
        status="passed" if proved else "failed",
        elapsed_s=time.perf_counter() - started_at,
    )
    _checkpoint(result_path, design_result)
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


def _read_design_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _run_design_subprocess(
    name: str,
    timeout_s: float,
) -> tuple[str, dict[str, Any] | None, float]:
    child_env = os.environ.copy()
    child_env[PARENT_PID_ENV] = str(os.getpid())
    with tempfile.TemporaryDirectory(prefix="zolotone-run-design-") as temp_dir:
        result_path = Path(temp_dir) / "result.json"
        started_at = time.perf_counter()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "infra.run_designs",
                "--design",
                name,
                "--result-file",
                str(result_path),
            ],
            env=child_env,
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
        return status, _read_design_result(result_path), elapsed_s


def _mark_active_check(design_result: dict[str, Any], status: str) -> None:
    for check_result in design_result.get("checks", {}).values():
        if check_result.get("status") == "running":
            check_result["status"] = status


def _new_report(timeout_s: float) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "started_at": _utc_timestamp(),
        "finished_at": None,
        "timeout_s": float(timeout_s),
        "status": "running",
        "designs": {},
    }


def run_designs(
    designs: Iterable[DesignCase] = DESIGNS,
    *,
    timeout_s: float = DEFAULT_DESIGN_TIMEOUT_S,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> int:
    report_path = Path(report_path)
    report = _new_report(timeout_s)
    _atomic_write_json(report_path, report)

    total = 0
    succeeded = 0
    failed = {}

    for design_case in designs:
        total += 1
        print(
            f"[RUN] {design_case.name} "
            f"(timeout: {timeout_s:g} seconds)",
            flush=True,
        )
        call_started_at = time.perf_counter()
        try:
            outcome = _run_design_subprocess(design_case.name, timeout_s)
            if isinstance(outcome, str):
                status = outcome
                design_result = None
                elapsed_s = time.perf_counter() - call_started_at
            else:
                status, design_result, elapsed_s = outcome
        except Exception as exc:
            status = "failed"
            elapsed_s = time.perf_counter() - call_started_at
            design_result = _initial_design_result(design_case.name)
            design_result["error"] = {
                "phase": "subprocess",
                **_error_details(exc),
            }
            print(
                f"[ERROR] Could not start {design_case.name}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()

        if design_result is None:
            design_result = _initial_design_result(design_case.name)
        design_result["name"] = design_case.name
        design_result["status"] = status
        design_result["elapsed_s"] = elapsed_s
        if status in {"timeout", "interrupted"}:
            _mark_active_check(design_result, status)
        report["designs"][design_case.name] = design_result
        _atomic_write_json(report_path, report)

        if status == "passed":
            succeeded += 1
            print(f"[PASS] {design_case.name}", flush=True)
        elif status == "timeout":
            failed[design_case.name] = status
            print(
                f"[TIMEOUT] {design_case.name} exceeded {timeout_s:g} seconds",
                file=sys.stderr,
                flush=True,
            )
        elif status == "interrupted":
            report["status"] = "interrupted"
            report["finished_at"] = _utc_timestamp()
            _atomic_write_json(report_path, report)
            raise KeyboardInterrupt
        else:
            failed[design_case.name] = status
            print(f"[FAIL] {design_case.name}", file=sys.stderr, flush=True)

    report["status"] = "failed" if failed else "passed"
    report["finished_at"] = _utc_timestamp()
    _atomic_write_json(report_path, report)

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
        "--result-file",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.design is not None:
        return 0 if check_design(
            _find_design(args.design),
            result_path=args.result_file,
        ) else 1
    return run_designs(report_path=args.report)


if __name__ == "__main__":
    raise SystemExit(main())
