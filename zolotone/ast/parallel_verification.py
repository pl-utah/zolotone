"""Bounded process-pool execution for independent verification cases."""

from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    ProcessPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Iterable

from ..solver.report import CaseVerificationResult, VerificationObserver
from ..spec import SpecContext


CaseVerifier = Callable[..., CaseVerificationResult]
MAX_WORKERS_ENV = "ZOLOTONE_MAX_WORKERS"


def automatic_max_workers() -> int:
    """Return the CPUs available to this process, with portable fallbacks."""
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            affinity_count = len(get_affinity(0))
        except (OSError, NotImplementedError):
            pass
        else:
            if affinity_count > 0:
                return affinity_count

    return os.cpu_count() or 1


def resolve_max_workers(max_workers: int | None) -> int:
    if max_workers is None:
        configured_workers = os.environ.get(MAX_WORKERS_ENV)
        if configured_workers is None:
            return automatic_max_workers()
        try:
            max_workers = int(configured_workers)
        except ValueError as exc:
            raise ValueError(
                f"{MAX_WORKERS_ENV} must be a positive integer"
            ) from exc
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError("max_workers must be an integer or None")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    return max_workers


def run_verification_cases(
    cases: Iterable[SpecContext],
    verify_case: CaseVerifier,
    verification_args: tuple[Any, ...],
    observer: VerificationObserver,
    max_workers: int,
) -> list[CaseVerificationResult]:
    if max_workers == 1:
        return _run_serially(
            cases,
            verify_case=verify_case,
            verification_args=verification_args,
            observer=observer,
        )
    return _run_in_parallel(
        cases,
        verify_case=verify_case,
        verification_args=verification_args,
        observer=observer,
        max_workers=max_workers,
    )


def _run_serially(
    cases: Iterable[SpecContext],
    verify_case: CaseVerifier,
    verification_args: tuple[Any, ...],
    observer: VerificationObserver,
) -> list[CaseVerificationResult]:
    results = []
    for case_ctx in cases:
        result = verify_case(case_ctx, *verification_args)
        results.append(result)
        observer.case_completed(result)
    return results


def _run_in_parallel(
    cases: Iterable[SpecContext],
    *,
    verify_case: CaseVerifier,
    verification_args: tuple[Any, ...],
    observer: VerificationObserver,
    max_workers: int,
) -> list[CaseVerificationResult]:
    """Run a lazy, bounded case stream and restore generation order."""
    indexed_cases = enumerate(cases)
    # SpecContext objects can be large. Keeping one submitted case per worker
    # preserves full worker utilization without retaining a second full set in
    # the executor's call queue.
    in_flight_limit = max_workers
    results_by_index: dict[int, CaseVerificationResult] = {}
    process_ctx = multiprocessing.get_context("spawn")
    in_flight = {}
    failed_submission = None

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=process_ctx,
        ) as executor:

            def submit_until_full() -> None:
                nonlocal failed_submission
                while len(in_flight) < in_flight_limit:
                    try:
                        index, case_ctx = next(indexed_cases)
                    except StopIteration:
                        return
                    try:
                        future = executor.submit(
                            verify_case,
                            case_ctx,
                            *verification_args,
                        )
                    except BrokenProcessPool:
                        failed_submission = (index, case_ctx)
                        raise
                    in_flight[future] = (index, case_ctx)

            submit_until_full()
            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    index, _ = in_flight[future]
                    result = future.result()
                    del in_flight[future]
                    results_by_index[index] = result
                    observer.case_completed(result)
                submit_until_full()
    except BrokenProcessPool:
        # A killed worker makes the complete executor unusable. Preserve any
        # results which reached the parent, then retry every other submitted
        # case and the remainder of the lazy case stream without a pool. This
        # both bounds recovery memory and keeps completed-case journals useful.
        for future, (index, _) in list(in_flight.items()):
            if not future.done():
                continue
            try:
                result = future.result()
            except (BrokenProcessPool, CancelledError):
                continue
            results_by_index[index] = result
            observer.case_completed(result)

        retry_cases = {
            index: case_ctx
            for _, (index, case_ctx) in in_flight.items()
            if index not in results_by_index
        }
        if failed_submission is not None:
            index, case_ctx = failed_submission
            retry_cases[index] = case_ctx
        in_flight.clear()

        print(
            "Verification worker pool terminated abruptly; "
            "retrying unfinished cases serially.",
            file=sys.stderr,
            flush=True,
        )
        for index in sorted(retry_cases):
            case_ctx = retry_cases.pop(index)
            result = verify_case(case_ctx, *verification_args)
            results_by_index[index] = result
            observer.case_completed(result)
        for index, case_ctx in indexed_cases:
            result = verify_case(case_ctx, *verification_args)
            results_by_index[index] = result
            observer.case_completed(result)

    return [results_by_index[index] for index in range(len(results_by_index))]
