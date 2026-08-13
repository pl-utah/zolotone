"""Bounded process-pool execution for independent verification cases."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Callable, Iterable

from ..solver.report import CaseVerificationResult, VerificationObserver
from ..spec import SpecContext


CaseVerifier = Callable[..., CaseVerificationResult]


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
        return automatic_max_workers()
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError("max_workers must be an integer or None")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    return max_workers


def run_verification_cases(
    cases: Iterable[SpecContext],
    *,
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
    *,
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
    in_flight_limit = 2 * max_workers
    results_by_index: dict[int, CaseVerificationResult] = {}
    process_ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=process_ctx,
    ) as executor:
        in_flight = {}

        def submit_until_full() -> None:
            while len(in_flight) < in_flight_limit:
                try:
                    index, case_ctx = next(indexed_cases)
                except StopIteration:
                    return
                future = executor.submit(
                    verify_case,
                    case_ctx,
                    *verification_args,
                )
                in_flight[future] = index

        submit_until_full()
        while in_flight:
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                index = in_flight.pop(future)
                result = future.result()
                results_by_index[index] = result
                observer.case_completed(result)
            submit_until_full()

    return [results_by_index[index] for index in range(len(results_by_index))]
