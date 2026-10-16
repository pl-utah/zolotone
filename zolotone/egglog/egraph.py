from time import perf_counter

from egglog import *

from .datatypes import MathBool
from .rules import load_rules
from ..solver.report import build_proof_report, merge_rule_application_counts

####################### PRIVATE ############################


def _create_egraph() -> EGraph:
    egraph = EGraph()
    load_rules(egraph, fold_base_two_powers=False)
    return egraph


def _make_run_schedule(scheduler: dict[str, int | None] | None):
    if scheduler is None:
        return 1
    return run(
        scheduler=back_off(
            match_limit=scheduler.get("match_limit"),
            ban_length=scheduler.get("ban_length"),
        )
    )


def _egglog_check_ctx(ctx: "SpecContext", iterations=6, scheduler=None):
    egraph = _create_egraph()
    
    to_check = ctx.to_egglog(egraph)
    run_schedule = _make_run_schedule(scheduler)
    
    run_started_at = perf_counter()
    
    rule_application_counts: dict[str, int] = {}
    iterations_used = 0
    proved = egraph.check_bool(*to_check)
    
    for _ in range(iterations):
        if proved:
            break
        run_report = egraph.run(run_schedule)
        iterations_used += 1

        proved = egraph.check_bool(*to_check)
        
    run_runtime_s = perf_counter() - run_started_at
    
    return proved, run_runtime_s, egraph, rule_application_counts, iterations_used


def _simplify_expr(expr: "SpecNode", egraph: EGraph):
    from ..spec.spec_utils import from_egglog
    return from_egglog(egraph.extract(expr.to_egglog()))


def _egglog_simplify_ctx(ctx: "SpecContext", egraph: EGraph):
    from ..spec.spec_ast import Eq, BoolEq, BoolExpr
    
    def simplify_check(check: BoolExpr):
        if isinstance(check, Eq) or isinstance(check, BoolEq):
            lhs = check.lhs.to_egglog()
            rhs = check.rhs.to_egglog()
            check_passed = egraph.check_bool(eq(lhs).to(rhs))
        else:
            expr = check.to_egglog()
            check_passed = egraph.check_bool(eq(expr).to(MathBool.True_()))
        if check_passed:
            return None
        return _simplify_expr(check, egraph)
    
    simplified_checks = []
    discharged_checks = []
    
    run_started_at = perf_counter()
    for check in ctx.checks:
        if not isinstance(check, BoolExpr):
            raise NotImplementedError(
                f"Only BoolExpr checks are supported, got {type(check).__name__}"
            )
        simplified = simplify_check(check)
        if simplified is not None:
            simplified_checks.append(simplified)
        else:
            discharged_checks.append(check)
    run_runtime_s = perf_counter() - run_started_at

    simplified_ctx = ctx.copy(assumes=ctx.assumes + discharged_checks, checks=simplified_checks)
    status = "unsat" if len(simplified_checks) == 0 else "unknown"
    return status, run_runtime_s, simplified_ctx


####################### PUBLIC #############################


def egglog_rewrite(ctx: "SpecContext", iterations: int, scheduler=None):
    proved, egglog_runtime_s, egraph, rule_application_counts, iterations_used = _egglog_check_ctx(
        ctx=ctx,
        iterations=iterations,
        scheduler=scheduler,
    )
    status, simplify_runtime_s, simplified_ctx = _egglog_simplify_ctx(
        ctx=ctx,
        egraph=egraph,
    )

    report = build_proof_report(
        ctx,
        simplified_ctx,
        tool="egglog-rewrite",
        runtime_s=egglog_runtime_s + simplify_runtime_s,
        status=status,
        rule_application_counts=rule_application_counts,
        iterations_used=iterations_used,
        egraph_size=sum(sz for _, sz in egraph.all_function_sizes()),
    )
    return report
