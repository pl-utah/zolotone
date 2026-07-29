from __future__ import annotations

from .spec_ast import *
from ..egglog import *
from egglog import rewrite, vars_
from ..solver.report import build_proof_report
from ..rival import rival_feasibility_check, rival_trim_context

import dreal
from time import perf_counter
import z3
import warnings


class SpecContext:
    """Builder API for creating spec programs over the spec AST."""
    
    def __init__(self, name: str):
        self.assumes: list[BoolExpr] = []
        self.checks: list[BoolExpr] = []
        self.requirements: list[BoolExpr] = []
        self._sym_counter = 0
        self.name = name
        self.spec_cache = {}
        self._spec_cache_valid = True

    def __getstate__(self):
        state = self.__dict__.copy()
        state["spec_cache"] = {}
        state["_spec_cache_valid"] = False
        return state
    
    def assume(self, condition: BoolExpr) -> None:
        if not isinstance(condition, BoolExpr):
            raise TypeError(f"SpecContext.assume expects BoolExpr, got {type(condition).__name__}")
        self.assumes.append(condition)
    
    def check(self, condition: BoolExpr) -> None:
        if not isinstance(condition, BoolExpr):
            raise TypeError(f"SpecContext.check expects BoolExpr, got {type(condition).__name__}")
        self.checks.append(condition)
    
    def require(self, condition: BoolExpr) -> None:
        """Register a condition that must hold for the specification to be valid."""
        if not isinstance(condition, BoolExpr):
            raise TypeError(
                "SpecContext.require expects BoolExpr, "
                f"got {type(condition).__name__}"
            )
        self.requirements.append(condition)
    
    def validate_requirements(self, timeout_ms: int = 10000) -> None:
        """Reject a specification unless all requirements are proved."""
        if not self.requirements:
            return
        
        from ..smt import z3_check_eq
        
        validation_ctx = self.copy(checks=list(self.requirements))
        report = z3_check_eq(validation_ctx, timeout_ms=timeout_ms)
        if report["status"] == "unsat":
            return
        
        detail = report.get("supplementary_info")
        message = (
            f"Could not prove specification requirements for {self.name!r}: "
            f"solver returned {report['status']}"
        )
        if detail:
            message = f"{message}. {detail}"
        raise MalformedSpecification(message)
    
    def _context_not_empty(self):
        if len(self.checks) == 0:
            raise RuntimeError("Context does not statements to check")
    
    def to_z3(self):
        self._context_not_empty()
        env = {}
        assume_terms = [assume.to_z3(env=env) for assume in self.assumes] + [z3.BoolVal(True)]  # make sure it is not empty
        check_terms = [check.to_z3(env=env) for check in self.checks]
        return z3.And(z3.And(*assume_terms), z3.Not(z3.And(*check_terms)))
    
    def to_dreal(self):
        self._context_not_empty()
        env: dict[tuple[str, str], dreal.Variable] = {}
        assume_terms = [assume.to_dreal(env) for assume in self.assumes] + [dreal.Formula.TRUE()]  # make sure it is not empty
        check_terms = [check.to_dreal(env) for check in self.checks]
        return dreal.And(dreal.And(*assume_terms), dreal.Not(dreal.And(*check_terms)))
    
    def to_egglog(self, egraph):
        self._context_not_empty()
        for assume in self.assumes:
            if isinstance(assume, Eq) or isinstance(assume, BoolEq):
                egraph.register(union(assume.lhs.to_egglog()).with_(assume.rhs.to_egglog()))
            else:
                egraph.register(union(assume.to_egglog()).with_(MathBool.True_()))
        
        to_check = []
        for check in self.checks:
            if isinstance(check, Eq) or isinstance(check, BoolEq):
                lhs = check.lhs.to_egglog()
                rhs = check.rhs.to_egglog()
                egraph.register(lhs)
                egraph.register(rhs)
                to_check.append(eq(lhs).to(rhs))
            elif isinstance(check, BoolExpr):
                expr = check.to_egglog()
                egraph.register(expr)
                to_check.append(eq(expr).to(MathBool.True_()))
            else:
                raise NotImplementedError(f"Only BoolExpr checks are supported, got {type(check).__name__}")
        return to_check
    
    def _learned_literals_with_anchors(self) -> dict[SpecNode, tuple[RealLit | BoolLit, int]]:
        """Learn literal facts and retain one assumption that justifies each."""

        candidates: dict[SpecNode, tuple[RealLit | BoolLit, int]] = {}
        
        def record(expr: SpecNode, lit: RealLit | BoolLit, anchor: int) -> None:
            existing = candidates.get(expr)
            if existing is None:
                candidates[expr] = (lit, anchor)
            
            # poorly written spec with contradictions
            elif not identical_nodes(existing[0], lit):
                raise PoorSpec(f"Conflicting learned literals for {expr}: {existing[0]} vs {lit}")
        
        for anchor, assume in enumerate(self.assumes):
            learned = self._canonical_learned_assumption(assume)
            if learned is None:
                continue
            record(*learned, anchor)
        
        return candidates

    # Try to learn literal facts from assumes only. Conflicting facts are errors.
    def learned_literals(self) -> dict[SpecNode, RealLit | BoolLit]:
        return {
            expr: value
            for expr, (value, _anchor) in self._learned_literals_with_anchors().items()
        }
    
    # Try to learn non-literal aliases from assumes only. Multiple aliases for
    # one variable are allowed; the remaining assumptions preserve constraints.
    def learned_aliases(self) -> dict[RealVar | BoolVar, SpecNode]:
        aliases: dict[RealVar | BoolVar, SpecNode] = {}
        
        def safe_alias(var, expr, lit_type):
            if isinstance(expr, (lit_type, RealVar, BoolVar)):
                return None
            if var in variables(expr):
                return None
            return var, expr
        
        def from_sides(lhs, rhs, var_type, lit_type):
            if isinstance(lhs, var_type):
                return safe_alias(lhs, rhs, lit_type)
            if isinstance(rhs, var_type):
                return safe_alias(rhs, lhs, lit_type)
            return None
        
        def learned_from(assume):
            assume = assume.constant_fold()
            if isinstance(assume, Eq):
                return from_sides(assume.lhs, assume.rhs, RealVar, RealLit)
            if isinstance(assume, BoolEq):
                return from_sides(assume.lhs, assume.rhs, BoolVar, BoolLit)
            return None
        
        for assume in self.assumes:
            learned = learned_from(assume)
            if learned is None:
                continue
            var, expr = learned
            aliases.setdefault(var, expr)
        return aliases

    @staticmethod
    def _reject_false_assumption(assume: BoolExpr) -> BoolExpr:
        if identical_nodes(assume, BoolLit(False)):
            raise PoorSpec(f"Assumption folds to false: {assume}")
        return assume
    
    # learning facts like: RealExpr == RealLit
    def _canonical_learned_assumption(self, assume: BoolExpr) -> tuple[SpecNode, RealLit | BoolLit] | None:
        assume = self._reject_false_assumption(assume.constant_fold())
        if isinstance(assume, BoolVar):
            return assume, BoolLit(True)
        # ``predicate == false`` folds to ``not predicate``. Preserve the
        # contextual fact even when predicate is a compound BoolExpr.
        if isinstance(assume, Not):
            return assume.value, BoolLit(False)
        if isinstance(assume, Eq):
            rhs_folded = assume.rhs.constant_fold()
            lhs_folded = assume.lhs.constant_fold()
            if (
                isinstance(lhs_folded, RealExpr)
                and not isinstance(lhs_folded, RealLit)
                and isinstance(rhs_folded, RealLit)
            ):
                return lhs_folded, rhs_folded
            if (
                isinstance(rhs_folded, RealExpr)
                and not isinstance(rhs_folded, RealLit)
                and isinstance(lhs_folded, RealLit)
            ):
                return rhs_folded, lhs_folded
        
        elif isinstance(assume, BoolEq):
            rhs_folded = assume.rhs.constant_fold()
            lhs_folded = assume.lhs.constant_fold()
            if (
                isinstance(lhs_folded, BoolExpr)
                and not isinstance(lhs_folded, BoolLit)
                and isinstance(rhs_folded, BoolLit)
            ):
                return lhs_folded, rhs_folded
            if (
                isinstance(rhs_folded, BoolExpr)
                and not isinstance(rhs_folded, BoolLit)
                and isinstance(lhs_folded, BoolLit)
            ):
                return rhs_folded, lhs_folded
        # Any remaining Boolean in the assumptions list is asserted true.
        # Keep this independent of the Eq/BoolEq branches so equalities that
        # do not expose a literal binding are still learned as Boolean facts.
        if isinstance(assume, BoolExpr) and not isinstance(assume, BoolLit):
            return assume, BoolLit(True)
        return None
    
    def simplify(self) -> "SpecContext":
        """Apply context learning and ordinary constant folding to a fixpoint."""
        simplified = self.copy()
        max_iterations = len(simplified.assumes) + len(simplified.checks) + 1
        for _ in range(max_iterations):
            anchored_literals = simplified._learned_literals_with_anchors()
            literal_replacements = {
                expr: value
                for expr, (value, _anchor) in anchored_literals.items()
            }
            alias_replacements = simplified.learned_aliases()
            
            variable_replacements = {
                expr: lit
                for expr, lit in literal_replacements.items()
                if isinstance(expr, (RealVar, BoolVar))  # get rid only of assigned vars
            }
            check_replacements = alias_replacements | literal_replacements
            
            new_assumes = []
            for assume_idx, assume in enumerate(simplified.assumes):
                new_assume = substitute_literals(
                    assume,
                    alias_replacements
                    | variable_replacements
                    | {
                        expr: value
                        for expr, (value, anchor) in anchored_literals.items()
                        if anchor != assume_idx
                        and not isinstance(expr, (RealVar, BoolVar))
                    },
                )
                new_assumes.append(
                    simplified._reject_false_assumption(new_assume)
                )
            new_checks = [
                substitute_literals(check, check_replacements)
                for check in simplified.checks
            ]
            if new_assumes == simplified.assumes and new_checks == simplified.checks:
                break
            simplified.assumes = new_assumes
            simplified.checks = new_checks

        simplified.assumes = [
            assume
            for assume in simplified.assumes
            if not identical_nodes(assume, BoolLit(True))
        ]
        simplified.checks = [
            check
            for check in simplified.checks
            if not identical_nodes(check, BoolLit(True))
        ]
        return simplified
    
    def spec_of(self, node: Node):
        if not self._spec_cache_valid:
            raise RuntimeError(
                "spec_of() is unavailable because spec_cache was discarded "
                "during multiprocessing serialization"
            )
        return node._evaluate_spec(ctx=self, cache=self.spec_cache)
    
    def real_val(self, value: int | float):
        return RealLit(value=value)
    
    def real(self, name: str) -> RealVar:
        return RealVar(name=name)
    
    def fresh_real(self, base: str) -> RealVar:
        return RealVar(self.fresh_name(base))

    def fresh_name(self, base: str):
        name=f"{base}_{self._sym_counter}"
        self._sym_counter += 1
        return name
    
    def bool(self, name: str) -> BoolVar:
        return BoolVar(name=name)
    
    def fresh_bool(self, base: str) -> BoolVar:
        return BoolVar(self.fresh_name(base))
    
    def bool_val(self, value: bool):
        return BoolLit(value=value)
    
    def true(self) -> BoolLit:
        return BoolLit(value=True)
    
    def false(self) -> BoolLit:
        return BoolLit(value=False)
    
    def reset(self) -> None:
        self.assumes.clear()
        self.checks.clear()
        self.requirements.clear()
        self._sym_counter = 0
        self.spec_cache.clear()
        self._spec_cache_valid = True
    
    def snapshot(self):
        return {
            "name": self.name,
            "assume_count": len(self.assumes),
            "check_count": len(self.checks),
            "requirement_count": len(self.requirements),
            "assumes": [str(assume) for assume in self.assumes],
            "checks": [str(check) for check in self.checks],
            "requirements": [str(requirement) for requirement in self.requirements],
            "context": str(self),
        }
    
    def __str__(self) -> str:
        def format_section(title: str, items: list[BoolExpr]) -> list[str]:
            if not items:
                return [f"{title}:", "  <none>"]
            return [f"{title}:"] + [f"  {item}" for item in items]

        lines = [f"SpecContext({self.name})"]
        lines.extend(format_section("Assumes", self.assumes))
        lines.extend(format_section("Checks", self.checks))
        lines.extend(format_section("Requirements", self.requirements))
        return "\n".join(lines)
    
    def copy(self, assumes=None, checks=None, requirements=None):
        if assumes is None:
            # Spec AST nodes are immutable, so a shallow list copy is enough here.
            # Deep-copying rebuilds nodes such as Eq via pickle-style protocols,
            # which breaks for custom __new__ constructors used in the spec AST.
            assumes = list(self.assumes)
        if checks is None:
            checks = list(self.checks)
        if requirements is None:
            requirements = list(self.requirements)
        
        new_ctx = SpecContext(self.name)
        new_ctx.assumes = assumes
        new_ctx.checks = checks
        new_ctx.requirements = requirements
        new_ctx._sym_counter = self._sym_counter
        new_ctx.spec_cache = dict(self.spec_cache)
        new_ctx._spec_cache_valid = self._spec_cache_valid
        return new_ctx


class PoorSpec(ValueError):
    pass


class MalformedSpecification(ValueError):
    pass


def _context_expression_state(ctx: SpecContext):
    return tuple(ctx.assumes), tuple(ctx.checks)


def _simplify_with_rival(ctx: SpecContext) -> SpecContext:
    """Alternate regular and Rival simplification to a structural fixpoint."""
    current = ctx
    seen_states = set()
    max_passes = 32

    for _ in range(max_passes):
        current_state = _context_expression_state(current)
        if current_state in seen_states:
            warnings.warn(f"Simplification cycle detected for {ctx.name!r}", RuntimeWarning)
            break
        seen_states.add(current_state)

        rewritten = rival_trim_context(current.simplify())
        rewritten_state = _context_expression_state(rewritten)
        current = rewritten
        if rewritten_state == current_state:
            break
    else:
        warnings.warn(f"Simplification did not saturate after {max_passes} passes for {ctx.name!r}", RuntimeWarning)

    return current


def simplify_ctx(ctx: SpecContext):
    run_started_at = perf_counter()
    
    try:
        simplified_ctx = _simplify_with_rival(ctx)
    except PoorSpec as exc:
        return build_proof_report(
            ctx,
            ctx.copy(),
            tool="simplify",
            runtime_s=perf_counter() - run_started_at,
            status="sat",
            feasibility_status="not feasible",
            info=exc,
        )
    
    ############### Feasibility ##################
    feasibility_status = None
    if any([identical_nodes(x, BoolLit(False)) for x in simplified_ctx.assumes]):
        feasibility_status = "not feasible"
    else:
        feasibility_status = rival_feasibility_check(simplified_ctx, max_depth=0, checks=False)
    ##############################################
    
    ############## Satisfiability ################
    satisfiability_status = None
    if feasibility_status == "not feasible":
        satisfiability_status = "sat"
    else:
        if any([identical_nodes(x, BoolLit(False)) for x in simplified_ctx.checks]):
             satisfiability_status = "sat"
        else:
            
            if rival_feasibility_check(simplified_ctx, max_depth=0, checks=True) == "not feasible":
                satisfiability_status = "sat"
            else:
                satisfiability_status = "unsat" if len(simplified_ctx.checks) == 0 else "unknown"
    ##############################################
    
    return build_proof_report(
        ctx,
        simplified_ctx,
        tool="simplify",
        runtime_s=perf_counter() - run_started_at,
        status=satisfiability_status,
        feasibility_status=feasibility_status,
    )
