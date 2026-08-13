from __future__ import annotations

from typing import Any

from .report import (
    CaseVerificationResult,
    CheckResult,
    ProofReport,
    StdoutVerificationObserver,
    VerificationObserver,
    build_proof_report,
    count_context_nodes,
    merge_rule_application_counts,
    validate_proof_status,
)

__all__ = [
    "CaseVerificationResult",
    "CheckResult",
    "ProofReport",
    "StdoutVerificationObserver",
    "VerificationObserver",
    "build_proof_report",
    "check_equivalence",
    "count_context_nodes",
    "merge_rule_application_counts",
    "validate_proof_status",
]


def check_equivalence(*args: Any, **kwargs: Any):
    from .engine import check_equivalence as _check_equivalence
    return _check_equivalence(*args, **kwargs)
