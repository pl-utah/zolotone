"""Rival machine bindings; higher-level algorithms live in .analysis."""

from .machine import (
    build_boolean_evaluator,
    RivalAnalysis,
    RivalEvaluation,
    RivalIntegerMachine,
    RivalMachine,
    build_integer_evaluator,
    build_machine,
    collect_free_vars,
    to_rival_ir,
)

__all__ = [
    "build_boolean_evaluator",
    "RivalMachine", "RivalIntegerMachine", "RivalAnalysis", "RivalEvaluation",
    "build_machine", "build_integer_evaluator", "collect_free_vars", "to_rival_ir",
]
