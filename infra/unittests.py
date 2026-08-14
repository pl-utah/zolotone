import unittest
import contextlib
import io
import json
import os
import pickle
import random
import signal
import subprocess
import struct
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock, call, patch

import dreal
import math
import z3
from egglog import EGraph

from zolotone import *
from zolotone.ast import nodes as ast_nodes
from zolotone.ast import parallel_verification as parallel_runner
from zolotone.egglog.rules import (
    check_rules,
    constant_rules,
    load_rules,
    rewrite_rules,
)
from zolotone.smt import dreal_check_eq, z3_check_eq
from zolotone.solver import engine as solver_engine
from zolotone.solver.report import (
    CaseVerificationResult,
    CheckResult,
    build_proof_report,
)
from zolotone.rival import (
    RivalAnalysis,
    build_machine,
    collect_free_vars,
    get_rival_rects,
    rival_feasibility_check,
    rival_trim_context,
    to_rival_ir,
)
from zolotone.spec.spec_context import simplify_ctx
from zolotone.spec.spec_utils import from_egglog
from examples.fp32_add import fp32_add
from examples.fp32_mult import fp32_mult
from examples.bf16_add import bf16_add
from examples.bf16_mult import bf16_mult
from examples.bf16_relu import bf16_relu
from examples.converters import (
    CONVERTER_FORMATS,
    CONVERTER_REGISTRY,
    FORMAT_STATIC_TYPES,
    bf16_to_fp32,
    fp16_to_fp32,
    fp32_to_bf16,
    fp32_to_e2m1,
    fp32_to_e4m3fn,
    fp32_to_e5m2,
    fp32_to_e5m2fnuz,
    fp32_to_fp16,
    fp32_to_ue4m3,
    ue4m3_to_fp32,
)
from examples.ue4m3x2_e2m1x2_add_fp32 import (
    spec_ue4m3x2_e2m1x2_add_fp32,
    ue4m3x2_e2m1x2_add_fp32,
)
from examples.ue4m3x2_e2m1x2_mult_fp32 import (
    spec_ue4m3x2_e2m1x2_mult_fp32,
    ue4m3x2_e2m1x2_mult_fp32,
)
from examples.bf16x8_dot_fp32_conventional import (
    bf16x8_dot_fp32_conventional,
    dot_product_spec as bf16x8_dot_fp32_spec,
)
from examples.bf16x8_dot_fp32_optimized import bf16x8_dot_fp32_optimized
from examples.wgmma import WGMMA_REGISTRY
from examples.wgmma_fp16_e4m3_e5m2 import (
    spec_wgmma_fp16_e4m3_e5m2,
    wgmma_fp16_e4m3_e5m2,
)
from examples.wgmma_fp32_e4m3_e4m3 import (
    spec_wgmma_fp32_e4m3_e4m3,
    wgmma_fp32_e4m3_e4m3,
)
from examples.wgmma_fp32_e5m2_e4m3 import (
    spec_wgmma_fp32_e5m2_e4m3,
    wgmma_fp32_e5m2_e4m3,
)

from infra.compile_cpp import jit_compile, nonjit_compile
from infra import make_designs_html
from infra import run_designs as design_runner


def _flat_trace_tool(ctx, timeout_ms):
    del timeout_ms
    report1 = build_proof_report(
        ctx,
        ctx.copy(),
        tool="branch-a",
        runtime_s=0.0,
        status="unknown",
    )
    report2 = build_proof_report(
        ctx,
        ctx.copy(),
        tool="branch-b",
        runtime_s=0.0,
        status="unsat",
    )
    return [report1, report2]


def _slow_tool(_ctx, timeout_ms):
    del timeout_ms
    time.sleep(1)


def _large_report_tool(ctx, timeout_ms):
    del timeout_ms
    return build_proof_report(
        ctx,
        ctx.copy(),
        tool="z3",
        runtime_s=0.0,
        status="unknown",
        supplementary_info=b"x" * 2048,
    )


class TestMakeDesignsHtml(unittest.TestCase):
    def test_renders_expandable_case_tables_for_each_design(self):
        report = {
            "started_at": "2026-08-04T01:02:03Z <start>",
            "finished_at": "2026-08-04T01:03:03Z",
            "designs": {
                "first <design>": {
                    "elapsed_s": 1.25,
                    "checks": {
                        "determinism": {
                            "status": "passed",
                            "proved": True,
                            "elapsed_s": 0.25,
                            "cases": {
                                "det <case>": {
                                    "status": "unsat",
                                    "feasibility": "feasible & checked",
                                    "proved": True,
                                    "elapsed_s": 0.1254,
                                }
                            },
                        },
                        "specification": {
                            "status": "failed",
                            "proved": False,
                            "elapsed_s": 0.75,
                            "cases": {
                                "spec & case": {
                                    "status": "sat",
                                    "feasibility": "unknown",
                                    "proved": False,
                                    "elapsed_s": 0.5,
                                }
                            },
                        },
                    },
                },
                "second": {"elapsed_s": 0.5, "checks": {}},
            },
        }

        html = make_designs_html.build_html(
            report,
            Path("reports/<source>.json"),
        )

        self.assertIn(
            'class="design-toggle" aria-expanded="false" '
            'aria-controls="design-details-0">first &lt;design&gt;</button>',
            html,
        )
        self.assertIn(
            'id="design-details-0" class="design-details" hidden',
            html,
        )
        self.assertIn('aria-controls="design-details-1">second</button>', html)
        self.assertIn('id="design-details-1"', html)
        for heading in (
            "Case name",
            "Status",
            "Feasibility",
            "Proved?",
            "Time spent",
        ):
            self.assertIn(f'<th scope="col">{heading}</th>', html)
        self.assertIn("<h2>Determinism cases</h2>", html)
        self.assertIn("<h2>Specification cases</h2>", html)
        self.assertIn('<th scope="row">det &lt;case&gt;</th>', html)
        self.assertIn('<span class="badge badge-unsat">UNSAT</span>', html)
        self.assertIn("feasible &amp; checked", html)
        self.assertIn('<span class="badge badge-passed">YES</span>', html)
        self.assertIn('<td class="elapsed">0.125 s</td>', html)
        self.assertIn('<th scope="row">spec &amp; case</th>', html)
        self.assertIn('<span class="badge badge-sat">SAT</span>', html)
        self.assertIn('<span class="badge badge-failed">NO</span>', html)
        self.assertIn("reports/&lt;source&gt;.json", html)

        determinism_position = html.index("det &lt;case&gt;")
        specification_position = html.index("spec &amp; case")
        self.assertLess(determinism_position, specification_position)

    def test_disclosures_toggle_independently(self):
        html = make_designs_html.build_html(
            {
                "started_at": "2026-08-04T01:02:03Z",
                "finished_at": "2026-08-04T01:03:03Z",
                "designs": {
                    "first": {"elapsed_s": 0.1, "checks": {}},
                    "second": {"elapsed_s": 0.2, "checks": {}},
                },
            },
            Path("reports/run_designs.json"),
        )

        self.assertEqual(html.count('<button type="button" class="design-toggle"'), 2)
        self.assertIn(
            'document.querySelectorAll(".design-toggle").forEach((button)',
            html,
        )
        self.assertIn(
            'button.setAttribute("aria-expanded", String(!expanded))',
            html,
        )
        self.assertIn("details.hidden = expanded", html)
        self.assertNotIn("querySelectorAll(\".design-details\")", html)

    def test_missing_and_empty_checks_render_case_empty_states(self):
        report = {
            "started_at": "2026-08-04T01:02:03Z",
            "finished_at": "2026-08-04T01:03:03Z",
            "designs": {
                "slow": {
                    "elapsed_s": 2.0,
                    "checks": {
                        "determinism": {
                            "status": "timeout",
                            "proved": False,
                            "elapsed_s": 1.5,
                        }
                    },
                }
            }
        }

        html = make_designs_html.build_html(
            report,
            Path("reports/run_designs.json"),
        )

        self.assertEqual(html.count("No cases reported."), 2)
        self.assertIn('<td class="elapsed">1.500 s</td>', html)
        self.assertIn("<h2>Determinism cases</h2>", html)
        self.assertIn("<h2>Specification cases</h2>", html)

    def test_timeout_displays_only_completed_cases(self):
        report = {
            "started_at": "2026-08-04T01:02:03Z",
            "finished_at": "2026-08-04T01:03:03Z",
            "designs": {
                "partial": {
                    "elapsed_s": 1.5,
                    "checks": {
                        "determinism": {
                            "status": "timeout",
                            "proved": False,
                            "elapsed_s": 1.5,
                            "cases": {
                                "complete": {
                                    "status": "unsat",
                                    "feasibility": "feasible",
                                    "proved": True,
                                    "elapsed_s": 0.25,
                                },
                            },
                        }
                    }
                }
            }
        }

        html = make_designs_html.build_html(
            report,
            Path("reports/run_designs.json"),
        )

        self.assertIn('<th scope="row">complete</th>', html)
        self.assertNotIn("trace only", html)


class TestRunDesigns(unittest.TestCase):
    def test_main_does_not_raise_system_exit_for_unsuccessful_results(self):
        with patch.object(design_runner, "run_designs", return_value=1) as run:
            result = design_runner.main(["--report", "unused.json"])

        self.assertEqual(result, 0)
        run.assert_called_once_with(report_path=Path("unused.json"))

    def test_completed_case_journal_recovers_only_complete_cases(self):
        case_event = {
            "case_name": "complete-case",
            "result": {
                "status": "unsat",
                "proved": True,
                "feasibility": "feasible",
                "elapsed_s": 0.25,
                "tools": [],
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            journal_path = Path(temp_dir) / "completed_cases.jsonl"
            with journal_path.open("w", encoding="utf-8") as journal:
                journal.write(json.dumps(case_event) + "\n")
                journal.write('{"case_name":"truncated"')

            completed = design_runner._read_completed_cases(journal_path)

        self.assertEqual(set(completed), {"complete-case"})
        complete = completed["complete-case"]
        self.assertTrue(complete["proved"])
        self.assertEqual(complete["elapsed_s"], 0.25)

    def test_check_design_journals_only_completed_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_path = Path(temp_dir) / "completed_cases.jsonl"
            result_path = Path(temp_dir) / "result.json"
            with contextlib.redirect_stdout(io.StringIO()):
                proved = design_runner.check_design(
                    design_runner._find_design("CSA_tree4"),
                    check_name="determinism",
                    result_path=result_path,
                    completed_cases_path=journal_path,
                )
            events = [
                json.loads(line)
                for line in journal_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(proved)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["result"]["status"], "unsat")

    def test_report_cli_uses_default_and_accepts_override(self):
        self.assertEqual(
            design_runner.parse_args([]).report,
            design_runner.DEFAULT_REPORT_PATH,
        )
        self.assertEqual(
            design_runner.parse_args(["--report", "custom/result.json"]).report,
            Path("custom/result.json"),
        )
        worker_args = design_runner.parse_args(
            ["--design", "bf16_relu", "--check", "specification"]
        )
        self.assertEqual(worker_args.design, "bf16_relu")
        self.assertEqual(worker_args.check, "specification")

    def test_worker_main_returns_zero_without_running_full_suite(self):
        with (
            patch.object(design_runner, "check_design", return_value=False) as check,
            patch.object(design_runner, "run_designs") as run_all,
        ):
            status = design_runner.main(
                [
                    "--design",
                    "bf16_relu",
                    "--check",
                    "specification",
                    "--result-file",
                    "result.json",
                    "--completed-cases-file",
                    "completed.jsonl",
                ]
            )

        self.assertEqual(status, 0)
        check.assert_called_once()
        run_all.assert_not_called()

    def test_runner_supports_direct_script_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            completed_cases_path = Path(temp_dir) / "completed_cases.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(design_runner.RUNNER_PATH),
                    "--design",
                    "CSA_tree4",
                    "--check",
                    "determinism",
                    "--result-file",
                    str(result_path),
                    "--completed-cases-file",
                    str(completed_cases_path),
                ],
                cwd=temp_dir,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "passed")
        self.assertEqual(set(result["checks"]), {"determinism"})

    def test_curated_registry_builds_every_design(self):
        expected_names = [
            "CSA_tree4",
            "bf16_add",
            "bf16_mult",
            "bf16_relu",
            *CONVERTER_REGISTRY,
            "fp32_add",
            "fp32_mult",
            "ue4m3x2_e2m1x2_add_fp32",
            "ue4m3x2_e2m1x2_mult_fp32",
            "bf16x8_dot_fp32_conventional",
            "bf16x8_dot_fp32_optimized",
            "wgmma_fp32_e4m3_e4m3",
            "wgmma_fp32_e5m2_e4m3",
            "wgmma_fp16_e4m3_e5m2",
        ]

        self.assertEqual(
            [design_case.name for design_case in design_runner.DESIGNS],
            expected_names,
        )
        for design_case in design_runner.DESIGNS:
            with self.subTest(design=design_case.name):
                self.assertIsInstance(design_case.build(), Node)

        csa = design_runner._find_design("CSA_tree4").build()
        self.assertEqual(
            [arg.node_type for arg in csa.inner_args],
            [QT(10, 10)] * 4,
        )

        unsigned_mixed_multiplier = design_runner._find_design(
            "ue4m3x2_e2m1x2_mult_fp32"
        ).build()
        self.assertEqual(
            [arg.node_type for arg in unsigned_mixed_multiplier.inner_args],
            [UE4M3T(), UE4M3T(), E2M1T(), E2M1T()],
        )
        self.assertEqual(unsigned_mixed_multiplier.node_type, Float32T())

        scaled_add = design_runner._find_design(
            "ue4m3x2_e2m1x2_add_fp32"
        ).build()
        self.assertEqual(
            [arg.node_type for arg in scaled_add.inner_args],
            [UE4M3T(), UE4M3T(), E2M1T(), E2M1T()],
        )
        self.assertEqual(scaled_add.node_type, Float32T())

    def test_check_design_runs_the_selected_check(self):
        design = design_runner._build_bf16_relu()
        design.check_determinism = Mock()
        design.check_spec = Mock(
            return_value=CheckResult(
                proved=True,
                requirement_report=None,
                cases=[],
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "design.json"
            completed_cases_path = Path(temp_dir) / "completed_cases.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                proved = design_runner.check_design(
                    design_runner.DesignCase("test_design", lambda: design),
                    check_name="specification",
                    result_path=result_path,
                    completed_cases_path=completed_cases_path,
                )
            design_report = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(proved)
        self.assertEqual(set(design_report["checks"]), {"specification"})
        design.check_determinism.assert_not_called()
        design.check_spec.assert_called_once()

    def test_checks_get_independent_timeouts_and_later_designs_still_run(self):
        designs = (
            design_runner.DesignCase("first", Mock()),
            design_runner.DesignCase("broken", Mock()),
            design_runner.DesignCase("last", Mock()),
        )
        statuses = iter(
            (
                "passed",
                "passed",
                "timeout",
                "passed",
                "failed",
                "failed",
            )
        )
        calls = []

        def run(name, check_name, timeout_s):
            calls.append((name, check_name, timeout_s))
            worker_status = next(statuses)
            check_status = {
                "passed": "passed",
                "failed": "failed",
                "timeout": "timeout",
            }[worker_status]
            check_result = {
                "status": check_status,
                "proved": check_status == "passed",
                "elapsed_s": 0.5,
            }
            return worker_status, check_result

        stdout = io.StringIO()
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            with (
                patch.object(design_runner, "_run_design_subprocess", side_effect=run),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = design_runner.run_designs(
                    designs,
                    report_path=report_path,
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 1)
        self.assertEqual(
            calls,
            [
                ("first", "determinism", design_runner.DEFAULT_DESIGN_TIMEOUT_S),
                ("first", "specification", design_runner.DEFAULT_DESIGN_TIMEOUT_S),
                ("broken", "determinism", design_runner.DEFAULT_DESIGN_TIMEOUT_S),
                ("broken", "specification", design_runner.DEFAULT_DESIGN_TIMEOUT_S),
                ("last", "determinism", design_runner.DEFAULT_DESIGN_TIMEOUT_S),
                ("last", "specification", design_runner.DEFAULT_DESIGN_TIMEOUT_S),
            ],
        )
        self.assertIn("Passed 1/3 designs.", stdout.getvalue())
        self.assertIn(
            f"[TIMEOUT] broken determinism exceeded "
            f"{design_runner.DEFAULT_DESIGN_TIMEOUT_S:g} seconds",
            stderr.getvalue(),
        )
        self.assertIn("last (failed)", stderr.getvalue())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            {name: result["status"] for name, result in report["designs"].items()},
            {"first": "passed", "broken": "timeout", "last": "failed"},
        )
        self.assertEqual(
            report["designs"]["broken"]["checks"]["determinism"][
                "elapsed_s"
            ],
            0.5,
        )

    def test_report_publishes_design_only_after_all_checks_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            observed_designs = []

            def run(_name, check_name, _timeout_s):
                report = json.loads(report_path.read_text(encoding="utf-8"))
                observed_designs.append(dict(report["designs"]))
                return (
                    "passed",
                    {
                        "status": "passed",
                        "proved": True,
                        "elapsed_s": 0.2,
                    },
                )

            with (
                patch.object(design_runner, "_run_design_subprocess", side_effect=run),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = design_runner.run_designs(
                    (design_runner.DesignCase("demo", Mock()),),
                    report_path=report_path,
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(observed_designs, [{}, {}])
        self.assertEqual(report["designs"]["demo"]["status"], "passed")
        self.assertNotIn("running", json.dumps(report))

    def test_json_report_preserves_case_tool_sequence_and_metrics(self):
        ctx = SpecContext("demo[arg0=norm]")
        x = RealVar("x")
        ctx.assume(x.eq(RealLit(1)))
        ctx.check((x + RealLit(1)).eq(RealLit(2)))
        ctx.require(BoolLit(True))
        simplified = ctx.copy(checks=[])

        simplify_report = build_proof_report(
            ctx,
            simplified,
            tool="simplify",
            runtime_s=0.25,
            status="unknown",
            feasibility_status="feasible",
        )
        z3_report = build_proof_report(
            simplified,
            simplified,
            tool="z3",
            runtime_s=0.5,
            status="unsat",
            timeout_ms=1000,
        )
        result = CheckResult(
            proved=True,
            requirement_report=z3_report,
            cases=[
                CaseVerificationResult(
                    name=ctx.name,
                    proved=True,
                    status="unsat",
                    feasibility_status="feasible",
                    proof_trace=[simplify_report, z3_report],
                    side_feasibility_reports=[simplify_report],
                )
            ],
        )

        serialized = result.to_json()
        case = serialized["cases"][ctx.name]

        self.assertEqual(serialized["tools"][0]["phase"], "requirement_validation")
        self.assertEqual(
            [tool["tool"] for tool in case["tools"]],
            ["simplify", "z3", "simplify"],
        )
        self.assertEqual(case["feasibility"], "feasible")
        self.assertEqual(case["tools"][0]["checks"]["discharged"], 1)
        self.assertEqual(case["tools"][0]["context_nodes"], {"before": 9, "after": 4})
        self.assertEqual(case["tools"][1]["metadata"]["timeout_ms"], 1000)
        self.assertEqual(case["tools"][0]["phase"], "proof")
        self.assertEqual(case["tools"][2]["phase"], "side_feasibility")
        self.assertNotIn("old_ctx", json.dumps(serialized))

    def test_specification_runs_after_every_unsuccessful_determinism_outcome(self):
        scenarios = {
            "failed": (
                "failed",
                {"status": "failed", "proved": False},
            ),
            "error": (
                "failed",
                {"status": "error", "proved": False},
            ),
            "timeout": (
                "timeout",
                {"status": "timeout", "proved": False},
            ),
            "build-failure": (
                "failed",
                {
                    "status": "error",
                    "proved": False,
                },
            ),
        }

        for scenario, determinism_outcome in scenarios.items():
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp_dir:
                calls = []

                def run(_name, check_name, _timeout_s):
                    calls.append(check_name)
                    if check_name == "determinism":
                        return determinism_outcome
                    return (
                        "passed",
                        {
                            "status": "passed",
                            "proved": True,
                            "elapsed_s": 0.3,
                        },
                    )

                with (
                    patch.object(design_runner, "_run_design_subprocess", side_effect=run),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    status = design_runner.run_designs(
                        (design_runner.DesignCase("demo", Mock()),),
                        report_path=Path(temp_dir) / "report.json",
                    )

                self.assertEqual(status, 1)
                self.assertEqual(calls, ["determinism", "specification"])

    def test_specification_timeout_and_dual_timeout_make_design_timeout(self):
        for outcomes in (("passed", "timeout"), ("timeout", "timeout")):
            with self.subTest(outcomes=outcomes), tempfile.TemporaryDirectory() as temp_dir:
                worker_statuses = iter(outcomes)

                def run(_name, check_name, _timeout_s):
                    worker_status = next(worker_statuses)
                    check_result = {"proved": worker_status == "passed"}
                    if worker_status == "passed":
                        check_result["status"] = "passed"
                    else:
                        check_result["status"] = "timeout"
                    check_result["elapsed_s"] = 2.0
                    return (
                        worker_status,
                        check_result,
                    )

                with (
                    patch.object(design_runner, "_run_design_subprocess", side_effect=run),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    status = design_runner.run_designs(
                        (design_runner.DesignCase("slow", Mock()),),
                        timeout_s=1.0,
                        report_path=Path(temp_dir) / "report.json",
                    )
                report = json.loads(
                    (Path(temp_dir) / "report.json").read_text(encoding="utf-8")
                )

                self.assertEqual(status, 1)
                self.assertEqual(report["designs"]["slow"]["status"], "timeout")
                self.assertEqual(
                    report["designs"]["slow"]["checks"]["specification"],
                    {"status": "timeout", "proved": False, "elapsed_s": 2.0},
                )

    def test_interruption_stops_before_specification(self):
        calls = []

        def run(_name, check_name, _timeout_s):
            calls.append(check_name)
            return (
                "interrupted",
                {"status": "interrupted", "proved": False, "elapsed_s": 0.1},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            with (
                patch.object(design_runner, "_run_design_subprocess", side_effect=run),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(KeyboardInterrupt),
            ):
                design_runner.run_designs(
                    (design_runner.DesignCase("demo", Mock()),),
                    report_path=report_path,
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(calls, ["determinism"])
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(
            report["designs"]["demo"]["checks"]["determinism"]["status"],
            "interrupted",
        )

    def test_design_elapsed_is_parent_observed_total(self):
        check_times = iter((1.0, 2.0))

        def run(_name, check_name, _timeout_s):
            elapsed_s = next(check_times)
            return (
                "passed",
                {
                    "status": "passed",
                    "proved": True,
                    "elapsed_s": elapsed_s,
                },
            )

        perf_counter_values = iter((10.0, 15.5))
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            with (
                patch.object(design_runner, "_run_design_subprocess", side_effect=run),
                patch.object(
                    design_runner.time,
                    "perf_counter",
                    side_effect=perf_counter_values,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = design_runner.run_designs(
                    (design_runner.DesignCase("demo", Mock()),),
                    report_path=report_path,
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        result = report["designs"]["demo"]
        self.assertEqual(result["elapsed_s"], 5.5)
        self.assertEqual(
            result["checks"]["determinism"]["elapsed_s"],
            1.0,
        )
        self.assertEqual(
            result["checks"]["specification"]["elapsed_s"],
            2.0,
        )

    def test_worker_interrupt_terminates_active_process_group(self):
        process = Mock(pid=1234)
        process.wait.side_effect = KeyboardInterrupt

        with (
            patch.object(design_runner.subprocess, "Popen", return_value=process) as popen,
            patch.object(design_runner, "_terminate_process_group") as terminate,
        ):
            status, _result = design_runner._run_design_subprocess(
                "CSA_tree4",
                "specification",
                17.0,
            )

        self.assertEqual(status, "interrupted")
        terminate.assert_called_once_with(process)
        command = popen.call_args.args[0]
        self.assertIn("--check", command)
        self.assertEqual(command[command.index("--check") + 1], "specification")
        process.wait.assert_called_once_with(timeout=17.0)

    def test_worker_timeout_terminates_active_process_group(self):
        process = Mock(pid=1234)
        process.wait.side_effect = subprocess.TimeoutExpired("worker", 17.0)

        with (
            patch.object(design_runner.subprocess, "Popen", return_value=process),
            patch.object(design_runner, "_terminate_process_group") as terminate,
        ):
            status, _result = design_runner._run_design_subprocess(
                "CSA_tree4",
                "determinism",
                17.0,
            )

        self.assertEqual(status, "timeout")
        terminate.assert_called_once_with(process)
        process.wait.assert_called_once_with(timeout=17.0)

class TestEgglogRewriteRules(unittest.TestCase):
    def test_rewrite_rules_are_sound(self):
        results = check_rules(rewrite_rules(), z3_timeout_ms=10000)
        invalid_rules = {
            name: report
            for name, report in results.items()
            if report["z3_status"] != "unsat"
            and report["dreal_status"] != "unsat"
        }

        self.assertEqual(invalid_rules, {})


class TestConstantFolding(unittest.TestCase):
    def assert_folded_value(self, node, runtime_type, expected_val):
        self.assertIsNotNone(node.node_type.runtime_val)
        self.assertIsInstance(node.node_type.runtime_val, runtime_type)
        self.assertEqual(node.node_type.runtime_val.val, expected_val)

        evaluated = node.evaluate()
        self.assertIsInstance(evaluated, runtime_type)
        self.assertEqual(evaluated.val, expected_val)

    def test_float32_nan_default_is_quiet(self):
        self.assertEqual(Float32.NaN().val, 0x7FC00000)
        self.assertEqual(Float32.NaN(1).val, 0x7F800001)
        
    def test_cpp_lowering_widening_add_sub(self):
        add_x = Var(name="add_x", sign=UQT(32, 0))
        add_y = Var(name="add_y", sign=UQT(32, 0))
        add_design = uq_add(add_x, add_y)

        sub_x = Var(name="sub_x", sign=UQT(32, 0))
        sub_y = Var(name="sub_y", sign=UQT(32, 0))
        sub_design = uq_sub(sub_x, sub_y)

        add_tempdir_jit, add_fn_jit = jit_compile(add_design)
        add_tempdir_no_jit, add_fn_no_jit = nonjit_compile(add_design)
        sub_tempdir_jit, sub_fn_jit = jit_compile(sub_design)
        sub_tempdir_no_jit, sub_fn_no_jit = nonjit_compile(sub_design)

        add_cases = [
            (0xFFFF_FFFF, 0x0000_0001, 0x1_0000_0000),
            (0x8000_0000, 0x8000_0000, 0x1_0000_0000),
        ]
        sub_cases = [
            (0x0000_0000, 0x0000_0001, 0x1_FFFF_FFFF),
            (0x8000_0000, 0x0000_0001, 0x07FFF_FFFF),
        ]

        try:
            for lhs_bits, rhs_bits, expected in add_cases:
                add_x.load_val(UQ(lhs_bits, 32, 0))
                add_y.load_val(UQ(rhs_bits, 32, 0))
                with self.subTest(op="uq_add", lhs=hex(lhs_bits), rhs=hex(rhs_bits)):
                    self.assertEqual(add_design.evaluate().val, expected)
                    self.assertEqual(add_fn_jit(lhs_bits, rhs_bits), expected)
                    self.assertEqual(add_fn_no_jit(lhs_bits, rhs_bits), expected)

            for lhs_bits, rhs_bits, expected in sub_cases:
                sub_x.load_val(UQ(lhs_bits, 32, 0))
                sub_y.load_val(UQ(rhs_bits, 32, 0))
                with self.subTest(op="uq_sub", lhs=hex(lhs_bits), rhs=hex(rhs_bits)):
                    self.assertEqual(sub_design.evaluate().val, expected)
                    self.assertEqual(sub_fn_jit(lhs_bits, rhs_bits), expected)
                    self.assertEqual(sub_fn_no_jit(lhs_bits, rhs_bits), expected)
        finally:
            add_tempdir_jit.cleanup()
            add_tempdir_no_jit.cleanup()
            sub_tempdir_jit.cleanup()
            sub_tempdir_no_jit.cleanup()


    def test_fp32_multiplier_special_cases(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
        design = fp32_mult(x, y)

        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        cases = [
            (Float32.Inf().val, Float32.Zero().val, Float32.NaN().val),
            (Float32.nInf().val, Float32.Zero().val, Float32.NaN().val),
            (Float32.Inf().val, Float32.from_fields(0, 128, 0).val, Float32.Inf().val),
            (Float32.nInf().val, Float32.from_fields(0, 128, 0).val, Float32.nInf().val),
            (Float32.nZero().val, Float32.from_fields(0, 128, 1 << 22).val, Float32.nZero().val),
            (Float32.nZero().val, Float32.from_fields(1, 128, 1 << 22).val, Float32.Zero().val),
            (Float32.NaN().val, Float32.from_fields(0, 128, 0).val, Float32.NaN().val),
        ]

        try:
            for lhs_bits, rhs_bits, expected_bits in cases:
                x.load_val(Float32(lhs_bits))
                y.load_val(Float32(rhs_bits))
                with self.subTest(lhs=hex(lhs_bits), rhs=hex(rhs_bits)):
                    self.assertEqual(design.evaluate().val, expected_bits)
                    self.assertEqual(fn_jit(lhs_bits, rhs_bits), expected_bits)
                    self.assertEqual(fn_no_jit(lhs_bits, rhs_bits), expected_bits)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

    def test_fp32_multiplier_zero_handling(self):
        x = Var(name="x", sign=Float32T())
        y = Var(name="y", sign=Float32T())
        design = fp32_mult(x, y)

        tempdir_jit, fn_jit = jit_compile(design)
        tempdir_no_jit, fn_no_jit = nonjit_compile(design)

        cases = [
            # Zero products use the XOR of the operand signs.
            ("+0 * +0", 0x00000000, 0x00000000, 0x00000000),
            ("+0 * -0", 0x00000000, 0x80000000, 0x80000000),
            ("-0 * +0", 0x80000000, 0x00000000, 0x80000000),
            ("-0 * -0", 0x80000000, 0x80000000, 0x00000000),
            ("+0 * +1", 0x00000000, 0x3f800000, 0x00000000),
            ("+0 * -1", 0x00000000, 0xbf800000, 0x80000000),
            ("-0 * +1", 0x80000000, 0x3f800000, 0x80000000),
            ("-0 * -1", 0x80000000, 0xbf800000, 0x00000000),
            ("+1 * -0", 0x3f800000, 0x80000000, 0x80000000),
            ("-1 * -0", 0xbf800000, 0x80000000, 0x00000000),
            ("+0 * min-subnormal", 0x00000000, 0x00000001, 0x00000000),
            ("-0 * min-subnormal", 0x80000000, 0x00000001, 0x80000000),
            ("-0 * -min-subnormal", 0x80000000, 0x80000001, 0x00000000),
            # Exact and rounded results at the bottom of the subnormal range.
            ("min-subnormal * +1", 0x00000001, 0x3f800000, 0x00000001),
            ("min-subnormal * -1", 0x00000001, 0xbf800000, 0x80000001),
            ("positive half-minimum tie", 0x00000001, 0x3f000000, 0x00000000),
            ("negative half-minimum tie", 0x80000001, 0x3f000000, 0x80000000),
            ("positive above-half minimum", 0x00000001, 0x3f400000, 0x00000001),
            ("negative above-half minimum", 0x80000001, 0x3f400000, 0x80000001),
            ("min-subnormal times 1.5 tie-to-even", 0x00000001, 0x3fc00000, 0x00000002),
            # Results around the subnormal/normal boundary.
            ("min-normal halved", 0x00800000, 0x3f000000, 0x00400000),
            ("min-normal times 2^-23", 0x00800000, 0x34000000, 0x00000001),
            ("min-normal times 2^-24 tie", 0x00800000, 0x33800000, 0x00000000),
            ("negative min-normal times 2^-24 tie", 0x80800000, 0x33800000, 0x80000000),
            ("normal-boundary tie-to-even", 0x00800000, 0x3f7fffff, 0x00800000),
            ("largest-subnormal doubled", 0x007fffff, 0x40000000, 0x00fffffe),
            # Zero times infinity is invalid regardless of either sign or order.
            ("+0 * +inf", 0x00000000, 0x7f800000, 0x7fc00000),
            ("-0 * +inf", 0x80000000, 0x7f800000, 0x7fc00000),
            ("+0 * -inf", 0x00000000, 0xff800000, 0x7fc00000),
            ("-0 * -inf", 0x80000000, 0xff800000, 0x7fc00000),
            ("+inf * -0", 0x7f800000, 0x80000000, 0x7fc00000),
            ("-inf * +0", 0xff800000, 0x00000000, 0x7fc00000),
        ]

        try:
            for name, lhs_bits, rhs_bits, expected_bits in cases:
                x.load_val(Float32(lhs_bits))
                y.load_val(Float32(rhs_bits))
                with self.subTest(name=name, lhs=hex(lhs_bits), rhs=hex(rhs_bits)):
                    self.assertEqual(design.evaluate().val, expected_bits)
                    self.assertEqual(fn_jit(lhs_bits, rhs_bits), expected_bits)
                    self.assertEqual(fn_no_jit(lhs_bits, rhs_bits), expected_bits)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

    def test_basic_add_folds_for_float_bit_types(self):
        cases = [
            ("float32", Float32),
            ("bfloat16", BFloat16),
        ]

        for name, runtime_type in cases:
            with self.subTest(runtime_type=name):
                folded = basic_add(
                    Const(runtime_type(1)),
                    Const(runtime_type(2)),
                    out=Const(runtime_type(0)),
                )

                self.assert_folded_value(folded, runtime_type, 3)

    def test_basic_add_masks_to_output_width(self):
        folded = basic_add(
            Const(UQ(7, 3, 0)),
            Const(UQ(3, 3, 0)),
            out=Const(UQ(0, 3, 0)),
        )

        self.assert_folded_value(folded, UQ, 2)

    def test_comparison_folds_to_bool(self):
        folded = basic_less(
            Const(UQ(1, 2, 0)),
            Const(UQ(2, 2, 0)),
            out=Const(Bool(0)),
        )

        self.assert_folded_value(folded, Bool, 1)

    def test_tuple_get_item_is_constant_folded(self):
        folded = make_Tuple(
            Const(UQ(5, 3, 0)),
            Const(Bool(1)),
        )[0]

        self.assert_folded_value(folded, UQ, 5)

    def test_if_then_else_is_constant_folded(self):
        folded = if_then_else(
            Const(Bool(1)),
            Const(UQ(5, 3, 0)),
            Const(UQ(2, 3, 0)),
        )

        self.assert_folded_value(folded, UQ, 5)

    def test_non_constant_inputs_do_not_fold(self):
        x = Var("x", sign=UQT(3, 0))
        node = basic_add(
            x,
            Const(UQ(1, 3, 0)),
            out=Const(UQ(0, 3, 0)),
        )

        self.assertIsNone(node.node_type.runtime_val)

        x.load_val(UQ(6, 3, 0))
        self.assertEqual(node.evaluate().val, 7)
        self.assertIsNone(node.node_type.runtime_val)

    def test_uq_rshift_jam_sets_sticky_when_shifted_out_bits_are_nonzero(self):
        cases = [
            (8, 1, 4),
            (9, 1, 5),
        ]

        for x_val, amount_val, expected in cases:
            with self.subTest(x=x_val, amount=amount_val):
                folded = uq_rshift_jam(
                    Const(UQ(x_val, 4, 0)),
                    Const(UQ.from_int(amount_val)),
                )
                self.assert_folded_value(folded, UQ, expected)

    def test_q_rshift_jam_is_sign_symmetric_and_handles_minimum(self):
        value = Var(name="value", sign=QT(5, 0))
        amount = Var(name="amount", sign=UQT(4, 0))
        design = q_rshift_jam(value, amount)
        self.assertIs(design.spec, q_rshift_jam_spec)
        self.assertEqual(design.node_type, QT(5, 0))

        tempdir_jit, compiled_jit = jit_compile(design)
        tempdir_no_jit, compiled_no_jit = nonjit_compile(design)
        cases = (
            # input bits, shift, expected bits
            (0b01000, 1, 0b00100),  # exact positive
            (0b01001, 1, 0b00101),  # positive sticky
            (0b11000, 1, 0b11100),  # exact negative
            (0b10111, 1, 0b11011),  # negative sticky: -9 -> -5
            (0b10000, 0, 0b10000),  # minimum value survives shift zero
            (0b10000, 1, 0b11000),  # minimum value shifts to -8
            (0b00100, 15, 0b00001),  # oversized positive shift jams
            (0b11100, 15, 0b11111),  # oversized negative shift jams
            (0b00000, 15, 0b00000),
        )
        try:
            for input_bits, shift, expected_bits in cases:
                value.load_val(Q(input_bits, 5, 0))
                amount.load_val(UQ(shift, 4, 0))
                with self.subTest(input_bits=input_bits, shift=shift):
                    self.assertEqual(design.evaluate().val, expected_bits)
                    self.assertEqual(compiled_jit(input_bits, shift), expected_bits)
                    self.assertEqual(
                        compiled_no_jit(input_bits, shift), expected_bits
                    )
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

    def test_q_resize_exactly_widens_signed_values(self):
        value = Var(name="value", sign=QT(4, 2))
        design = q_resize(value, 6, 5)
        self.assertIs(design.spec, q_resize_spec)
        self.assertEqual(design.node_type, QT(6, 5))

        tempdir_jit, compiled_jit = jit_compile(design)
        tempdir_no_jit, compiled_no_jit = nonjit_compile(design)
        try:
            for input_bits in range(1 << 6):
                expected_bits = (
                    (input_bits - (1 << 6) if input_bits & (1 << 5) else input_bits)
                    << 3
                ) & ((1 << 11) - 1)
                value.load_val(Q(input_bits, 4, 2))
                with self.subTest(input_bits=input_bits):
                    self.assertEqual(design.evaluate().val, expected_bits)
                    self.assertEqual(compiled_jit(input_bits), expected_bits)
                    self.assertEqual(compiled_no_jit(input_bits), expected_bits)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()

        with self.assertRaisesRegex(ValueError, "integer field"):
            q_resize(value, 3, 2)
        with self.assertRaisesRegex(ValueError, "fractional field"):
            q_resize(value, 4, 1)

    def test_conventional_and_optimized_match_on_cancellation_regression(self):
        vals = [43160, 10458, 11062, 10989, 10589, 10469, 11020, 11013]
        a = [Var(name=f"a_{i}", sign=BFloat16T()) for i in range(4)]
        b = [Var(name=f"b_{i}", sign=BFloat16T()) for i in range(4)]

        for i, bits in enumerate(vals[:4]):
            a[i].load_val(BFloat16(bits))
        for i, bits in enumerate(vals[4:]):
            b[i].load_val(BFloat16(bits))

        conventional = bf16x8_dot_fp32_conventional(*a, *b).evaluate()
        optimized = bf16x8_dot_fp32_optimized(*a, *b).evaluate()

        self.assertEqual(conventional.val, 388040612)
        self.assertEqual(conventional, optimized)

    def test_dot_products_preserve_sticky_evidence_during_alignment(self):
        # Exact sum: 1 + 2**-24 + 2**-31. The 2**-31 term is sticky
        # evidence that makes the FP32 result round up rather than tie to even.
        a_values = [0x3F80, 0x3980, 0x3800, 0x0000]
        b_values = [0x3F80, 0x3980, 0x3780, 0x0000]
        a = [Var(name=f"a_{i}", sign=BFloat16T()) for i in range(4)]
        b = [Var(name=f"b_{i}", sign=BFloat16T()) for i in range(4)]

        for variable, bits in zip(a, a_values, strict=True):
            variable.load_val(BFloat16(bits))
        for variable, bits in zip(b, b_values, strict=True):
            variable.load_val(BFloat16(bits))

        designs = {
            "conventional": bf16x8_dot_fp32_conventional(*a, *b),
            "optimized": bf16x8_dot_fp32_optimized(*a, *b),
        }
        for name, design in designs.items():
            with self.subTest(design=name):
                self.assertEqual(design.evaluate().val, 0x3F800001)

    def test_dot_products_handle_subnormals_and_special_values(self):
        a = [Var(name=f"a_{i}", sign=BFloat16T()) for i in range(4)]
        b = [Var(name=f"b_{i}", sign=BFloat16T()) for i in range(4)]
        designs = {
            "conventional": bf16x8_dot_fp32_conventional(*a, *b),
            "optimized": bf16x8_dot_fp32_optimized(*a, *b),
        }

        zero = BFloat16.Zero()
        one = BFloat16.from_fields(sign=0, exponent=127, mantissa=0)
        largest_finite = BFloat16.from_fields(
            sign=0,
            exponent=254,
            mantissa=127,
        )
        smallest_subnormal = BFloat16.from_fields(sign=0, exponent=0, mantissa=1)

        cases = [
            (
                "subnormal",
                [smallest_subnormal, zero, zero, zero],
                [one, zero, zero, zero],
                0x00010000,
            ),
            (
                "zero product does not set the maximum exponent",
                [smallest_subnormal, zero, zero, zero],
                [one, largest_finite, zero, zero],
                0x00010000,
            ),
            (
                "all zero products",
                [zero, zero, zero, zero],
                [zero, zero, zero, zero],
                Float32.Zero().val,
            ),
            (
                "three normal products and one zero product",
                [one, zero, one, one],
                [one, zero, one, one],
                Float32.from_fields(
                    sign=0,
                    exponent=128,
                    mantissa=1 << 22,
                ).val,
            ),
            (
                "positive infinity",
                [BFloat16.Inf(), zero, zero, zero],
                [one, zero, zero, zero],
                Float32.Inf().val,
            ),
            (
                "negative infinity",
                [BFloat16.nInf(), zero, zero, zero],
                [one, zero, zero, zero],
                Float32.nInf().val,
            ),
            (
                "opposing infinities",
                [BFloat16.Inf(), BFloat16.nInf(), zero, zero],
                [one, one, zero, zero],
                Float32.NaN().val,
            ),
            (
                "zero times infinity",
                [zero, zero, zero, zero],
                [BFloat16.Inf(), zero, zero, zero],
                Float32.NaN().val,
            ),
            (
                "NaN input",
                [BFloat16.NaN(), zero, zero, zero],
                [one, zero, zero, zero],
                Float32.NaN().val,
            ),
        ]

        for name, a_values, b_values, expected in cases:
            for design_name, design in designs.items():
                with self.subTest(name=name, design=design_name):
                    for variable, value in zip(a, a_values):
                        variable.load_val(value)
                    for variable, value in zip(b, b_values):
                        variable.load_val(value)
                    self.assertEqual(design.evaluate().val, expected)

            with self.subTest(name=name, design="spec"):
                outer_ctx = SpecContext(f"dot-product-{name}")
                outer_result = bf16x8_dot_fp32_spec(
                    *(value.to_spec(outer_ctx) for value in (*a_values, *b_values)),
                    ctx=outer_ctx,
                ).constant_fold()
                expected_result = Float32(expected).to_spec(outer_ctx)
                if (
                    expected_result.is_inf.constant_fold().value
                    or expected_result.is_nan.constant_fold().value
                ):
                    self.assertEqual(
                        outer_result.classification_flags(),
                        expected_result.classification_flags(),
                    )
                if expected_result.is_inf.constant_fold().value:
                    self.assertEqual(
                        outer_result.sign.constant_fold(),
                        expected_result.sign.constant_fold(),
                    )
                outer_ctx.validate_requirements()


class TestFingerprint(unittest.TestCase):
    def test_runtime_type_fingerprint_depends_on_structure_and_value(self):
        self.assertEqual(UQ(3, 3, 0)._fingerprint(), UQ(3, 3, 0)._fingerprint())
        self.assertNotEqual(UQ(3, 3, 0)._fingerprint(), UQ(4, 3, 0)._fingerprint())
        self.assertNotEqual(UQ(3, 3, 0)._fingerprint(), UQ(3, 4, 0)._fingerprint())

        lhs = Tuple(UQ(5, 3, 0), Bool(1))
        rhs = Tuple(UQ(5, 3, 0), Bool(1))
        different = Tuple(UQ(5, 3, 0), Bool(0))
        self.assertEqual(lhs._fingerprint(), rhs._fingerprint())
        self.assertNotEqual(lhs._fingerprint(), different._fingerprint())

    def test_static_type_fingerprint_includes_runtime_val(self):
        plain = UQT(3, 0)
        same = UQT(3, 0)
        self.assertEqual(plain._fingerprint(), same._fingerprint())

        with_runtime = UQT(3, 0)
        with_runtime.runtime_val = UQ(1, 3, 0)
        same_runtime = UQT(3, 0)
        same_runtime.runtime_val = UQ(1, 3, 0)
        different_runtime = UQT(3, 0)
        different_runtime.runtime_val = UQ(2, 3, 0)

        self.assertEqual(with_runtime._fingerprint(), same_runtime._fingerprint())
        self.assertNotEqual(plain._fingerprint(), with_runtime._fingerprint())
        self.assertNotEqual(with_runtime._fingerprint(), different_runtime._fingerprint())

    def test_equivalent_graphs_have_equal_fingerprints(self):
        x1 = Var("x", sign=UQT(3, 0))
        y1 = Var("y", sign=UQT(3, 0))
        x2 = Var("x", sign=UQT(3, 0))
        y2 = Var("y", sign=UQT(3, 0))

        lhs = basic_add(x1, y1, Const(UQ(0, 3, 0)))
        rhs = basic_add(x2, y2, Const(UQ(0, 3, 0)))

        self.assertEqual(lhs._fingerprint(), rhs._fingerprint())

    def test_graph_fingerprint_changes_with_operation_and_constants(self):
        x = Var("x", sign=UQT(3, 0))
        y = Var("y", sign=UQT(3, 0))

        add_node = basic_add(x, y, Const(UQ(0, 3, 0)))
        sub_node = basic_sub(x, y, Const(UQ(0, 3, 0)))
        different_const = basic_add(x, y, Const(UQ(1, 3, 0)))

        self.assertNotEqual(add_node._fingerprint(), sub_node._fingerprint())
        self.assertNotEqual(add_node._fingerprint(), different_const._fingerprint())

    def test_fingerprint_depends_on_jittable_lowering_when_codegen_differs(self):
        node = make_Tuple(Const(UQ(1, 3, 0)), Const(UQ(2, 3, 0)))

        self.assertNotEqual(node._fingerprint(jittable=True), node._fingerprint(jittable=False))

    def test_fingerprint_is_stable_across_runtime_variable_bindings(self):
        x = Var("x", sign=UQT(3, 0))
        y = Var("y", sign=UQT(3, 0))
        node = basic_add(x, y, Const(UQ(0, 3, 0)))

        before = node._fingerprint()

        x.load_val(UQ(6, 3, 0))
        y.load_val(UQ(1, 3, 0))
        self.assertEqual(node.evaluate().val, 7)

        after = node._fingerprint()
        self.assertEqual(before, after)


class TestPowSpecOp(unittest.TestCase):
    def test_if_constant_fold_prunes_nonliteral_branch(self):
        x = RealVar("x")
        y = RealVar("y")

        self.assertEqual(
            If(BoolLit(False), x + RealLit(1), y + RealLit(2)).constant_fold(),
            Add(y, RealLit(2)),
        )
        self.assertEqual(
            If(BoolLit(True), x + RealLit(1), y + RealLit(2)).constant_fold(),
            Add(x, RealLit(1)),
        )

    def test_boolean_constant_fold_short_circuits_nonliteral_operands(self):
        p = BoolVar("p")

        self.assertEqual(
            And(BoolLit(False), p).constant_fold(),
            BoolLit(False),
        )
        self.assertEqual(
            And(BoolLit(True), p).constant_fold(),
            p,
        )
        self.assertEqual(
            Or(BoolLit(True), p).constant_fold(),
            BoolLit(True),
        )
        self.assertEqual(
            Or(BoolLit(False), p).constant_fold(),
            p,
        )

    def test_mul_by_zero_shortcuts_nonliteral_operand(self):
        x = RealVar("x")

        self.assertEqual(
            Mul(RealLit(0), x + RealLit(1)).constant_fold(),
            RealLit(0),
        )
        self.assertEqual(
            Mul(x + RealLit(1), RealLit(0)).constant_fold(),
            RealLit(0),
        )

    def test_pow_python_sugar_uses_negative_one_base(self):
        self.assertEqual(
            RealLit(-1) ** RealLit(3),
            Pow(RealLit(-1), RealLit(3)),
        )

    def test_pow_python_sugar_keeps_supported_base2_and_square_forms(self):
        self.assertEqual(
            RealLit(2) ** RealLit(3),
            Pow(RealLit(2), RealLit(3)),
        )
        self.assertEqual(
            RealLit(3) ** RealLit(2),
            Pow(RealLit(3), RealLit(2)),
        )

    def test_pow_round_trips_from_egglog(self):
        exprs = [
            Pow(RealLit(-1), RealLit(2)),
            Pow(RealLit(2), RealLit(3)),
            Pow(RealLit(3), RealLit(2)),
        ]

        for expr in exprs:
            with self.subTest(expr=str(expr)):
                self.assertEqual(from_egglog(expr.to_egglog()), expr)

    def test_pow_constant_folds_in_egglog(self):
        cases = [
            (Pow(RealLit(-1), RealLit(3)), RealLit(-1)),
            (Pow(RealLit(2), RealLit(3)), RealLit(8)),
            (Pow(RealLit(3), RealLit(2)), RealLit(9)),
        ]

        for expr, expected in cases:
            with self.subTest(expr=str(expr)):
                egraph = EGraph()
                load_rules(egraph)
                lowered = expr.to_egglog()

                egraph.register(lowered)
                egraph.run(1)

                self.assertEqual(from_egglog(egraph.extract(lowered)), expected)
    def test_minus_one_symbolic_power_lowers_as_plain_power(self):
        s = RealVar("s")
        expr = RealLit(-1) ** s

        z3_expr = expr.to_z3({})
        dreal_expr = expr.to_dreal({})

        self.assertNotEqual(z3_expr.decl().kind(), z3.Z3_OP_ITE)
        self.assertEqual(str(z3_expr), "-1**s")
        self.assertEqual(str(dreal_expr), "pow(-1, s)")
    def test_numeric_equality_constant_folds_in_egglog(self):
        cases = [
            (RealLit(3).eq(RealLit(3)), BoolLit(True)),
            (RealLit(3).ne(RealLit(4)), BoolLit(True)),
        ]

        for expr, expected in cases:
            with self.subTest(expr=str(expr)):
                egraph = EGraph()
                load_rules(egraph)
                lowered = expr.to_egglog()

                egraph.register(lowered)
                egraph.run(1)

                self.assertEqual(from_egglog(egraph.extract(lowered)), expected)

    def test_numeric_equality_constant_folds_in_simplify_mode(self):
        cases = [
            (RealLit(5).eq(RealLit(5)), BoolLit(True)),
            (RealLit(5).ne(RealLit(6)), BoolLit(True)),
        ]

        for expr, expected in cases:
            with self.subTest(expr=str(expr)):
                egraph = EGraph()
                load_rules(egraph)
                lowered = expr.to_egglog()

                egraph.register(lowered)
                egraph.run(1)

                self.assertEqual(from_egglog(egraph.extract(lowered)), expected)

    def test_context_simplify_uses_if_branch_pruning(self):
        ctx = SpecContext("if-pruning")
        x = ctx.real("x")

        ctx.check(If(BoolLit(False), x + RealLit(1), RealLit(3)).eq(RealLit(3)))

        simplified = ctx.simplify()
        self.assertEqual(simplified.checks, [])

    def test_context_simplify_uses_mul_by_zero_shortcut(self):
        ctx = SpecContext("mul-zero")
        x = ctx.real("x")

        ctx.check((RealLit(0) * (x + RealLit(5))).eq(RealLit(0)))

        simplified = ctx.simplify()
        self.assertEqual(simplified.checks, [])


class TestSpecContextLearning(unittest.TestCase):
    def test_real_literal_shortcuts(self):
        ctx = SpecContext("real-literal-shortcuts")

        self.assertEqual(ctx.zero(), RealLit(0))
        self.assertEqual(ctx.one(), RealLit(1))
        self.assertEqual(ctx.two(), RealLit(2))
        self.assertIsInstance(ctx.zero(), RealLit)
        self.assertIsInstance(ctx.one(), RealLit)
        self.assertIsInstance(ctx.two(), RealLit)

    def test_learned_literals_reads_real_var_equalities(self):
        ctx = SpecContext("learn-real")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(ctx.one()))
        ctx.assume(ctx.two().eq(y))

        learned = ctx.learned_literals()
        self.assertEqual(len(learned), 2)
        self.assertEqual(learned[x], RealLit(1))
        self.assertEqual(learned[y], RealLit(2))

    def test_learned_literals_reads_foldable_real_equalities(self):
        ctx = SpecContext("learn-real-foldable")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(ctx.two() + ctx.real_val(3)))
        ctx.assume((ctx.real_val(10) - ctx.real_val(4)).eq(y))

        learned = ctx.learned_literals()
        self.assertEqual(len(learned), 2)
        self.assertEqual(learned[x], RealLit(5))
        self.assertEqual(learned[y], RealLit(6))

    def test_learned_literals_reads_bool_var_equalities(self):
        ctx = SpecContext("learn-bool")
        p = ctx.bool("p")
        q = ctx.bool("q")

        ctx.assume(p.eq(ctx.true()))
        ctx.assume(ctx.false().eq(q))

        learned = ctx.learned_literals()
        self.assertEqual(len(learned), 2)
        self.assertEqual(learned[p], BoolLit(True))
        self.assertEqual(learned[q], BoolLit(False))

    def test_learned_literals_reads_foldable_bool_equalities(self):
        ctx = SpecContext("learn-bool-foldable")
        p = ctx.bool("p")
        q = ctx.bool("q")

        ctx.assume(p.eq(ctx.two().eq(ctx.two())))
        ctx.assume((ctx.two().eq(ctx.real_val(3))).eq(q))

        learned = ctx.learned_literals()
        self.assertEqual(len(learned), 2)
        self.assertEqual(learned[p], BoolLit(True))
        self.assertEqual(learned[q], BoolLit(False))

    def test_learned_literals_reads_non_literal_equalities_as_boolean_facts(self):
        ctx = SpecContext("learn-ignore")
        x = ctx.real("x")
        y = ctx.real("y")
        p = ctx.bool("p")
        q = ctx.bool("q")

        ctx.assume(x.eq(y))
        ctx.assume(p.eq(q))

        self.assertEqual(
            ctx.learned_literals(),
            {
                x.eq(y): BoolLit(True),
                p.eq(q): BoolLit(True),
            },
        )

    def test_learned_literals_raises_on_conflicting_bindings(self):
        ctx = SpecContext("learn-conflict")
        x = ctx.real("x")
        p = ctx.bool("p")

        ctx.assume(x.eq(ctx.zero()))
        ctx.assume(p.eq(ctx.true()))
        ctx.assume(x.eq(ctx.one()))

        with self.assertRaises(ValueError):
            ctx.learned_literals()

    def test_conflicting_assumptions_are_deferred_until_simplification(self):
        ctx = SpecContext("assume-deferred-conflict")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(ctx.zero()))
        before = ctx.copy()
        ctx.assume(x.eq(ctx.one()))

        self.assertEqual(
            ctx.assumes,
            before.assumes + [Eq(x, RealLit(1))],
        )
        self.assertEqual(ctx.checks, before.checks)

        ctx.assume(y.eq(x))
        with self.assertRaises(ValueError):
            ctx.simplify()

        self.assertEqual(
            ctx.assumes,
            [Eq(x, RealLit(0)), Eq(x, RealLit(1)), Eq(y, x)],
        )

    def test_context_fixpoint_simplifies_assumptions_from_learned_literals(self):
        ctx = SpecContext("simplify-assumes")
        and_res = ctx.real("and_res")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(and_res.eq(x * y))
        ctx.assume(x.eq(ctx.zero()))
        ctx.assume(y.eq(ctx.zero()))

        simplified = ctx.simplify()

        self.assertEqual(
            ctx.assumes,
            [
                Eq(and_res, x * y),
                Eq(x, RealLit(0)),
                Eq(y, RealLit(0)),
            ],
        )
        self.assertEqual(
            simplified.assumes,
            [],
        )

    def test_context_fixpoint_inlines_non_literal_aliases(self):
        ctx = SpecContext("simplify-aliases")
        xor_res = ctx.real("xor_res")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(xor_res.eq(x + y))
        ctx.assume((xor_res * ctx.one()).eq(x + y))
        ctx.check((xor_res + ctx.one()).eq((x + y) + ctx.one()))

        simplified = ctx.simplify()

        self.assertNotIn("xor_res", str(simplified))
        self.assertEqual(simplified.assumes, [])
        self.assertEqual(simplified.checks, [])

    def test_context_fixpoint_preserves_duplicate_aliases_as_constraints(self):
        ctx = SpecContext("simplify-duplicate-aliases")
        alias = ctx.real("alias")
        y = ctx.real("y")
        z = ctx.real("z")

        ctx.assume(alias.eq(y + ctx.one()))
        ctx.assume(alias.eq(z + ctx.one()))

        simplified = ctx.simplify()

        self.assertEqual(
            simplified.assumes,
            [Eq(y + RealLit(1), z + RealLit(1))],
        )

    def test_context_fixpoint_keeps_self_referential_constraints(self):
        ctx = SpecContext("simplify-self-reference")
        x = ctx.real("x")

        ctx.assume(x.eq(abs(x)))
        ctx.check(x.eq(abs(x)))

        simplified = ctx.simplify()

        self.assertEqual(simplified.assumes, [Eq(x, Abs(x))])
        self.assertEqual(simplified.checks, [])

    def test_assume_records_alias_loops(self):
        ctx = SpecContext("assume-alias-loop")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(y + ctx.one()))
        ctx.assume(y.eq(x + ctx.one()))

        self.assertEqual(
            ctx.assumes,
            [
                Eq(x, y + RealLit(1)),
                Eq(y, x + RealLit(1)),
            ],
        )

    def test_check_records_alias_loops(self):
        ctx = SpecContext("check-alias-loop")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(y + ctx.one()))
        ctx.check(y.eq(x + ctx.one()))

        self.assertEqual(ctx.checks, [Eq(y, x + RealLit(1))])

    def test_context_fixpoint_substitutes_canonical_learned_assumptions(self):
        ctx = SpecContext("canonical-assumes")
        x = ctx.real("x")
        p = ctx.bool("p")

        ctx.assume(x.eq(ctx.one() + ctx.two()))
        ctx.assume(ctx.real_val(4).eq(x + ctx.one()))
        ctx.assume(p.eq(ctx.two().eq(ctx.two())))

        simplified = ctx.simplify()

        self.assertEqual(
            simplified.assumes,
            [],
        )
        self.assertEqual(simplified.learned_literals(), {})

    def test_context_fixpoint_propagates_through_multiple_rounds(self):
        ctx = SpecContext("simplify-multi-round")
        x = ctx.real("x")
        y = ctx.real("y")
        z = ctx.real("z")

        ctx.assume(x.eq(y + ctx.one()))
        ctx.assume(y.eq(z + ctx.one()))
        ctx.assume(z.eq(ctx.zero()))

        simplified = ctx.simplify()

        self.assertEqual(
            simplified.assumes,
            [],
        )
        self.assertEqual(simplified.learned_literals(), {})

    def test_context_fixpoint_simplifies_checks_from_learned_literals(self):
        ctx = SpecContext("simplify-checks")
        x = ctx.real("x")

        ctx.assume(x.eq(ctx.one()))
        ctx.check((x + ctx.two()).eq(ctx.real_val(3)))

        simplified = ctx.simplify()

        self.assertEqual(ctx.checks, [Eq(x + RealLit(2), RealLit(3))])
        self.assertEqual(simplified.checks, [])

    def test_context_fixpoint_simplifies_bool_assumptions_and_checks(self):
        ctx = SpecContext("simplify-bool")
        p = ctx.bool("p")
        q = ctx.bool("q")

        ctx.assume(p.eq(ctx.true()))
        ctx.assume(q.eq(p))
        ctx.check(q.eq(ctx.true()))

        simplified = ctx.simplify()

        self.assertEqual(ctx.assumes, [BoolEq(p, BoolLit(True)), BoolEq(q, p)])
        self.assertEqual(simplified.assumes, [])
        self.assertEqual(simplified.checks, [])
        self.assertEqual(simplified.learned_literals(), {})

    def test_context_fixpoint_learns_negated_compound_boolean_fact(self):
        ctx = SpecContext("simplify-negated-compound-bool")
        x = ctx.real("x")
        overflow = abs(x) > ctx.one()

        ctx.assume(overflow.eq(ctx.false()))
        ctx.check(overflow.eq(ctx.false()))

        simplified = ctx.simplify()

        self.assertEqual(simplified.assumes, [~overflow])
        self.assertEqual(simplified.checks, [])
        self.assertEqual(
            simplified.learned_literals(),
            {overflow: BoolLit(False)},
        )

    def test_context_fixpoint_learns_positive_compound_boolean_fact(self):
        ctx = SpecContext("simplify-positive-compound-bool")
        x = ctx.real("x")
        in_range = (x >= ctx.real_val(-1)) & (x <= ctx.one())

        ctx.assume(in_range.eq(ctx.true()))
        ctx.check(in_range)

        simplified = ctx.simplify()

        self.assertEqual(simplified.assumes, [in_range])
        self.assertEqual(simplified.checks, [])
        self.assertEqual(
            simplified.learned_literals(),
            {in_range: BoolLit(True)},
        )

    def test_context_fixpoint_simplifies_assumptions_from_other_compound_facts(self):
        ctx = SpecContext("simplify-compound-assumptions")
        x = ctx.real("x")
        y = ctx.real("y")
        positive_x = x > ctx.zero()
        positive_y = y > ctx.zero()

        ctx.assume(positive_x)
        ctx.assume(positive_y)
        ctx.assume(~(positive_x & positive_y))

        with self.assertRaisesRegex(ValueError, "Assumption folds to false"):
            ctx.simplify()

    def test_context_fixpoint_keeps_one_anchor_for_duplicate_compound_facts(self):
        ctx = SpecContext("simplify-duplicate-compound")
        x = ctx.real("x")
        positive = x > ctx.zero()

        ctx.assume(positive)
        ctx.assume(positive)
        ctx.check(positive)

        simplified = ctx.simplify()

        self.assertEqual(simplified.assumes, [positive])
        self.assertEqual(simplified.checks, [])

    def test_context_fixpoint_discharges_asserted_equalities(self):
        ctx = SpecContext("simplify-asserted-equalities")
        x = ctx.real("x")
        y = ctx.real("y")
        p = ctx.bool("p")
        q = ctx.bool("q")

        ctx.assume(x.eq(y))
        ctx.assume(p.eq(q))
        ctx.check(x.eq(y))
        ctx.check(p.eq(q))

        simplified = ctx.simplify()

        self.assertEqual(simplified.assumes, [x.eq(y), p.eq(q)])
        self.assertEqual(simplified.checks, [])

    def test_context_preserves_finite_if_after_alias_substitution(self):
        ctx = SpecContext("simplify-if-alias")
        selected = ctx.fresh_real("selected")
        condition = ctx.bool("condition")

        ctx.assume(selected.eq(If(condition, ctx.one(), ctx.zero())))
        ctx.assume((ctx.one() - selected).eq(ctx.one()))
        ctx.check(condition)

        simplified = ctx.simplify()

        self.assertEqual(
            simplified.assumes,
            [
                (
                    ctx.one()
                    - If(condition, ctx.one(), ctx.zero())
                ).eq(ctx.one())
            ],
        )
        self.assertEqual(simplified.checks, [condition])

    def test_context_fixpoint_accepts_duplicate_equivalent_bindings(self):
        ctx = SpecContext("simplify-duplicate")
        x = ctx.real("x")
        p = ctx.bool("p")

        ctx.assume(x.eq(ctx.one()))
        ctx.assume((ctx.two() - ctx.one()).eq(x))
        ctx.assume(p.eq(ctx.true()))
        ctx.assume(ctx.real_val(3).eq(ctx.real_val(3)).eq(p))

        simplified = ctx.simplify()

        self.assertEqual(
            simplified.assumes,
            [],
        )
        self.assertEqual(simplified.learned_literals(), {})

    def test_context_fixpoint_raises_on_conflicting_bindings(self):
        ctx = SpecContext("simplify-conflict")
        x = ctx.real("x")

        ctx.assume(x.eq(ctx.zero()))
        ctx.assume(x.eq(ctx.one()))

        with self.assertRaises(ValueError):
            ctx.simplify()

    def test_simplify_returns_new_simplified_context(self):
        ctx = SpecContext("simplify-output")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x.eq(y + ctx.one()))
        ctx.assume(y.eq(ctx.two()))
        ctx.check(x.eq(ctx.real_val(3)))

        simplified = ctx.simplify()

        self.assertIsNot(simplified, ctx)
        self.assertEqual(ctx.assumes, [Eq(x, Add(y, RealLit(1))), Eq(y, RealLit(2))])
        self.assertEqual(ctx.checks, [Eq(x, RealLit(3))])
        self.assertEqual(
            simplified.assumes,
            [],
        )
        self.assertEqual(simplified.checks, [])

    def test_simplify_leaves_original_context_unchanged_on_conflict(self):
        ctx = SpecContext("simplify-conflict-output")
        x = ctx.real("x")

        ctx.assume(x.eq(ctx.zero()))
        ctx.assume(x.eq(ctx.one()))

        with self.assertRaises(ValueError):
            ctx.simplify()

        self.assertEqual(ctx.assumes, [Eq(x, RealLit(0)), Eq(x, RealLit(1))])

    def test_simplify_ctx_skips_rival_for_false_assumption(self):
        ctx = SpecContext("infeasible-shortcut")
        ctx.assume(BoolLit(False))

        with (
            patch("zolotone.spec.spec_context.rival_feasibility_check") as feasibility,
            patch("zolotone.spec.spec_context.rival_trim_context") as trim,
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            report = simplify_ctx(ctx)

        feasibility.assert_not_called()
        trim.assert_not_called()
        self.assertEqual(report["feasibility_status"], "not feasible")
        self.assertEqual(report["status"], "sat")

    def test_simplify_ctx_alternates_regular_and_rival_until_saturated(self):
        ctx = SpecContext("alternating-simplification")
        x = ctx.real("x")
        zero = ctx.zero()
        one = ctx.one()
        ctx.assume(x >= zero)
        ctx.assume(abs(x).eq(one))
        ctx.check(x.eq(one))

        report = simplify_ctx(ctx)

        self.assertEqual(report["new_ctx"].assumes, [])
        self.assertEqual(report["new_ctx"].checks, [])
        self.assertEqual(report["status"], "unsat")
        self.assertEqual(ctx.assumes, [x >= zero, abs(x).eq(one)])
        self.assertEqual(ctx.checks, [x.eq(one)])

class TestEgglogFloatLiterals(unittest.TestCase):
    def exact_literal_value(self, expr):
        if isinstance(expr, RealLit):
            return Fraction(expr.value)
        if isinstance(expr, Mul):
            return (
                self.exact_literal_value(expr.lhs)
                * self.exact_literal_value(expr.rhs)
            )
        if isinstance(expr, Neg):
            return -self.exact_literal_value(expr.value)
        if isinstance(expr, Pow):
            base = self.exact_literal_value(expr.base)
            exponent = self.exact_literal_value(expr.exponent)
            self.assertEqual(exponent.denominator, 1)
            return base ** exponent.numerator
        self.fail(f"unexpected literal representation: {expr!r}")

    def assert_expr_has_exact_float_value(self, expr, value):
        self.assertEqual(self.exact_literal_value(expr), Fraction(value))

    def test_dyadic_literals_preserve_explicit_base_two_scale(self):
        cases = (
            (0.125, RealLit(2) ** RealLit(-3)),
            (
                0.375,
                RealLit(3) * (RealLit(2) ** RealLit(-3)),
            ),
            (-0.5, -(RealLit(2) ** RealLit(-1))),
        )

        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(
                    from_egglog(RealLit(value).to_egglog()),
                    expected,
                )

    def test_proof_rules_do_not_fold_dyadic_scale_to_rational(self):
        dyadic = RealLit(0.125).to_egglog()
        rational = Math.Num(BigRat(1, 8))

        proof_egraph = EGraph()
        load_rules(proof_egraph, fold_base_two_powers=False)
        proof_egraph.register(dyadic)
        proof_egraph.run(2)

        folding_egraph = EGraph()
        load_rules(folding_egraph, fold_base_two_powers=True)
        folding_egraph.register(dyadic)
        folding_egraph.run(2)

        self.assertFalse(proof_egraph.check_bool(eq(dyadic).to(rational)))
        self.assertTrue(folding_egraph.check_bool(eq(dyadic).to(rational)))

    def test_xnor_extracts_as_boolean_equality(self):
        p = BoolVar("p")
        q = BoolVar("q")
        xnor = (p & q) | ((~p) & (~q))

        egraph = EGraph()
        load_rules(egraph)
        lowered = xnor.to_egglog()

        egraph.register(lowered)
        egraph.run(1)

        self.assertEqual(from_egglog(egraph.extract(lowered)), p.eq(q))

    def test_real_lit_round_trips_exact_finite_float_values(self):
        values = [
            0.1,
            0.2,
            0.3,
            1.25,
            -2.5,
            1e-6,
            2.0 ** -20,
        ]

        for value in values:
            with self.subTest(value=value):
                round_tripped = from_egglog(RealLit(value).to_egglog())
                self.assert_expr_has_exact_float_value(round_tripped, value)

    def test_fractional_literals_constant_fold_in_egglog_without_losing_float_bits(self):
        expr = RealLit(0.1) + RealLit(0.2)

        egraph = EGraph()
        load_rules(egraph)
        lowered = expr.to_egglog()

        egraph.register(lowered)
        egraph.run(1)

        folded = from_egglog(egraph.extract(lowered)).constant_fold()
        self.assertEqual(folded, RealLit(0.1 + 0.2))
        self.assertEqual(folded.value.hex(), (0.1 + 0.2).hex())

    def test_fractional_pow_round_trips_through_egglog(self):
        expr = Pow(RealLit(2.5), RealLit(0.5))
        round_tripped = from_egglog(expr.to_egglog())

        self.assertIsInstance(round_tripped, Pow)
        self.assert_expr_has_exact_float_value(round_tripped.base, 2.5)
        self.assert_expr_has_exact_float_value(round_tripped.exponent, 0.5)

    def test_non_finite_real_lits_are_rejected_by_egglog_lowering(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RealLit(value).to_egglog()


class TestSpecAstConstantFolding(unittest.TestCase):
    def assert_check_simplifies_to(self, expr, expected):
        ctx = SpecContext("simplify-check-expression")
        ctx.check(expr)

        expected = expected.constant_fold()
        expected_checks = [] if expected == BoolLit(True) else [expected]
        self.assertEqual(ctx.simplify().checks, expected_checks)

    def test_fp_expr_declares_required_abstract_format_operations(self):
        self.assertEqual(
            FPExpr.__abstractmethods__,
            {
                "classification_flags",
                "decode",
                "encode",
                "fresh",
                "is_finite",
                "observables_for_classification",
            },
        )

    def test_fp_expr_builds_shared_exclusive_classification_assumptions(self):
        flags = tuple(
            BoolVar(name)
            for name in ("norm", "sub", "zero", "inf", "nan")
        )
        value = fp32(
            value=RealVar("value"),
            sign=RealVar("sign"),
            exponent=RealVar("exponent"),
            mantissa=RealVar("mantissa"),
            is_norm=flags[0],
            is_sub=flags[1],
            is_zero=flags[2],
            is_inf=flags[3],
            is_nan=flags[4],
        )
        ctx = SpecContext("shared-fp-classification")

        value._assume_exclusive_classification(ctx)

        at_least_one = flags[0]
        for flag in flags[1:]:
            at_least_one = at_least_one | flag
        pairwise_exclusions = [
            (~lhs) | (~rhs)
            for idx, lhs in enumerate(flags)
            for rhs in flags[idx + 1:]
        ]
        self.assertEqual(ctx.assumes, [at_least_one, *pairwise_exclusions])

    def test_fp_expr_constant_fold_is_shared_by_all_formats(self):
        self.assertIs(fp16.constant_fold, FPExpr.constant_fold)
        self.assertIs(bf16.constant_fold, FPExpr.constant_fold)
        self.assertIs(fp32.constant_fold, FPExpr.constant_fold)

        value = fp16(
            value=RealLit(1) + RealLit(2),
            sign=RealLit(0),
            exponent=RealLit(15),
            mantissa=RealLit(0),
            is_norm=BoolLit(True),
            is_sub=BoolLit(False),
            is_zero=BoolLit(False),
            is_inf=BoolLit(False),
            is_nan=BoolLit(False),
        )

        folded = value.constant_fold()

        self.assertIsInstance(folded, fp16)
        self.assertEqual(folded.value, RealLit(3))
        self.assertIs(folded.constant_fold(), folded)

    def test_fp_expr_requires_a_declared_value_field(self):
        with self.assertRaisesRegex(TypeError, "must declare a value"):
            class MissingValueFP(FPExpr):
                pass

    def test_fp_expr_requires_value_to_be_real_expr(self):
        with self.assertRaisesRegex(TypeError, "value must be RealExpr"):
            fp32(
                value=object(),
                sign=RealLit(0),
                exponent=RealLit(0),
                mantissa=RealLit(0),
                is_norm=BoolLit(False),
                is_sub=BoolLit(False),
                is_zero=BoolLit(True),
                is_inf=BoolLit(False),
                is_nan=BoolLit(False),
            )

    def test_constant_fold_method_folds_literal_tree(self):
        expr = (RealLit(2) + RealLit(3)) * RealLit(4)

        self.assertEqual(expr.constant_fold(), RealLit(20))

    def test_constant_fold_collapses_literal_subtrees(self):
        expr = ((RealLit(2) + RealLit(3)) * RealLit(4)).eq(RealLit(20))

        self.assertEqual(expr.constant_fold(), BoolLit(True))

    def test_constant_fold_can_produce_false(self):
        expr = (RealLit(2) + RealLit(3)).eq(RealLit(6))

        self.assertEqual(expr.constant_fold(), BoolLit(False))

    def test_constant_fold_cancels_identical_subtraction(self):
        x = RealVar("x")

        self.assertEqual((x - x).constant_fold(), RealLit(0))

    def test_constant_fold_keeps_symbolic_if_shape(self):
        expr = If(BoolLit(True), RealVar("x"), RealVar("y"))

        self.assertEqual(expr.constant_fold(), RealVar("x"))

    def test_if_keeps_real_branches_as_if(self):
        expr = If(BoolVar("condition"), RealVar("x"), RealVar("y"))

        self.assertIsInstance(expr, If)

    def test_if_lowers_bool_branches_to_boolean_logic(self):
        condition = BoolVar("condition")
        on_true = BoolVar("on_true")
        on_false = BoolVar("on_false")

        expr = If(condition, on_true, on_false)

        self.assertIsInstance(expr, BoolExpr)
        self.assertEqual(
            expr,
            (condition & on_true) | ((~condition) & on_false),
        )

    def test_if_constant_folds_bool_branches(self):
        self.assertEqual(
            If(BoolLit(True), BoolLit(False), BoolLit(True)).constant_fold(),
            BoolLit(False),
        )
        self.assertEqual(
            If(BoolLit(False), BoolLit(False), BoolLit(True)).constant_fold(),
            BoolLit(True),
        )

    def test_implies_lowers_directly_to_existing_boolean_ast(self):
        lhs = BoolVar("lhs")
        rhs = BoolVar("rhs")

        expression = lhs.implies(rhs)

        self.assertEqual(expression, (~lhs) | rhs)
        self.assertEqual(from_egglog(expression.to_egglog()), (~lhs) | rhs)
        self.assertEqual(
            BoolLit(True).implies(BoolLit(False)).constant_fold(),
            BoolLit(False),
        )
        self.assertEqual(
            BoolLit(False).implies(BoolLit(False)).constant_fold(),
            BoolLit(True),
        )

    def test_implies_rejects_non_boolean_operands(self):
        with self.assertRaisesRegex(TypeError, "Expected BoolExpr"):
            BoolLit(True).implies(RealLit(0))

    def test_if_rejects_mixed_bool_and_real_branches(self):
        with self.assertRaisesRegex(TypeError, "If branches"):
            If(BoolVar("condition"), BoolVar("on_true"), RealLit(0))

    def test_if_rejects_fp_branches_and_directs_users_to_cases(self):
        ctx = SpecContext("if-rejects-fp")
        with self.assertRaisesRegex(
            TypeError,
            "If does not support FPExpr branches; use exhaustive Cases",
        ):
            If(BoolVar("condition"), fp32.nan(ctx), fp32.ninf(ctx))

    def test_if_rejects_mixed_real_and_fp_branches(self):
        ctx = SpecContext("if-rejects-mixed-fp")
        with self.assertRaisesRegex(TypeError, "use exhaustive Cases"):
            If(BoolVar("condition"), fp32.nan(ctx), RealLit(0))

    def test_cases_lower_to_ordered_nested_ifs(self):
        first = BoolVar("first")
        second = BoolVar("second")
        ctx = SpecContext("ordered-cases")

        expr = Cases(
            case(first, RealLit(1)),
            case(second, RealLit(2)),
            case(BoolLit(True), RealLit(3)),
            ctx=ctx,
        )

        self.assertEqual(
            expr,
            If(first, RealLit(1), If(second, RealLit(2), RealLit(3))),
        )

    def test_cases_select_the_first_matching_case(self):
        ctx = SpecContext("first-matching-case")
        expr = Cases(
            case(BoolLit(True), RealLit(1)),
            case(BoolLit(True), RealLit(2)),
            ctx=ctx,
        )

        self.assertEqual(expr.constant_fold(), RealLit(1))

    def test_cases_support_fp_values(self):
        ctx = SpecContext("fp-cases")
        expr = Cases(
            case(BoolLit(False), fp32.nan(ctx)),
            case(BoolLit(True), fp32.ninf(ctx)),
            ctx=ctx,
        )

        self.assertIsInstance(expr, fp32)
        self.assertEqual(expr.is_ninf.constant_fold(), BoolLit(True))

    def test_special_encoding_preserves_finite_value(self):
        ctx = SpecContext("finite-special-encoding")
        finite = fp32(
            value=RealLit(3),
            sign=RealLit(0),
            exponent=RealLit(128),
            mantissa=RealLit(0),
            is_norm=BoolLit(True),
            is_sub=BoolLit(False),
            is_zero=BoolLit(False),
            is_inf=BoolLit(False),
            is_nan=BoolLit(False),
        )

        encoded = special_encoding(finite, ctx)

        self.assertEqual(encoded.constant_fold(), finite)

    def test_special_encoding_freshens_nonfinite_value_per_collection(self):
        ctx = SpecContext("nonfinite-special-encoding")
        infinity = fp32(
            value=RealVar("shared_input_special"),
            sign=RealLit(0),
            exponent=RealLit(255),
            mantissa=RealLit(0),
            is_norm=BoolLit(False),
            is_sub=BoolLit(False),
            is_zero=BoolLit(False),
            is_inf=BoolLit(True),
            is_nan=BoolLit(False),
        )

        first = special_encoding(infinity, ctx).constant_fold()
        second = special_encoding(infinity, ctx).constant_fold()

        self.assertIsInstance(first.value, RealVar)
        self.assertIsInstance(second.value, RealVar)
        self.assertNotEqual(first.value.name, second.value.name)
        self.assertNotEqual(first.value.name, "shared_input_special")
        self.assertNotEqual(second.value.name, "shared_input_special")

    def test_cases_support_bool_values(self):
        condition = BoolVar("condition")
        on_true = BoolVar("on_true")
        on_false = BoolVar("on_false")
        ctx = SpecContext("bool-cases")

        expr = Cases(
            case(condition, on_true),
            case(~condition, on_false),
            ctx=ctx,
        )

        self.assertIsInstance(expr, BoolExpr)
        self.assertEqual(
            expr,
            (condition & on_true) | ((~condition) & on_false),
        )

    def test_cases_require_at_least_one_case_and_a_context(self):
        condition = BoolVar("condition")
        value = RealLit(1)
        ctx = SpecContext("invalid-cases")

        with self.assertRaisesRegex(ValueError, "at least one case"):
            Cases(ctx=ctx)
        with self.assertRaisesRegex(TypeError, "missing.*ctx"):
            Cases(case(condition, value))

    def test_cases_validate_entries_conditions_and_values(self):
        ctx = SpecContext("invalid-case-entries")

        with self.assertRaisesRegex(TypeError, "created by case"):
            Cases([case(BoolLit(True), RealLit(1))], ctx=ctx)
        with self.assertRaisesRegex(TypeError, "created by case"):
            Cases(object(), ctx=ctx)
        with self.assertRaisesRegex(TypeError, "Expected BoolExpr"):
            case(RealLit(1), RealLit(2))
        with self.assertRaisesRegex(TypeError, "Cases values"):
            case(BoolLit(True), object())

    def test_cases_reject_mismatched_branch_types(self):
        condition = BoolVar("condition")
        ctx = SpecContext("mismatched-case-types")

        with self.assertRaisesRegex(TypeError, "Cases branches"):
            Cases(
                case(condition, fp32.nan(ctx)),
                case(~condition, RealLit(0)),
                ctx=ctx,
            )

    def test_cases_validate_exhaustive_coverage(self):
        ctx = SpecContext("exhaustive-cases")
        condition = ctx.bool("condition")

        Cases(
            case(condition, RealLit(1)),
            case(~condition, RealLit(2)),
            ctx=ctx,
        )

        report = ctx.validate_requirements(timeout_ms=1000)

        self.assertEqual(report["tool"], "z3")
        self.assertEqual(report["status"], "unsat")
        self.assertGreater(report["context_nodes_before"], 0)

    def test_cases_reject_incomplete_coverage(self):
        ctx = SpecContext("incomplete-cases")
        condition = ctx.bool("condition")

        Cases(
            case(condition, RealLit(1)),
            ctx=ctx,
        )

        with self.assertRaisesRegex(
            MalformedSpecification,
            "Could not prove specification requirements",
        ):
            ctx.validate_requirements(timeout_ms=1000)

    def test_cases_reject_unknown_coverage(self):
        ctx = SpecContext("unknown-case-coverage")
        Cases(
            case(BoolLit(True), RealLit(1)),
            ctx=ctx,
        )

        with (
            patch(
                "zolotone.smt.z3_check_eq",
                return_value={
                    "status": "unknown",
                    "supplementary_info": "timeout",
                },
            ),
            self.assertRaisesRegex(MalformedSpecification, "solver returned unknown"),
        ):
            ctx.validate_requirements(timeout_ms=1000)

    def test_fp32_decoder_returns_named_fields(self):
        ctx = SpecContext("structured-fp32-decode")

        decoded = fp32_decode(Const(Float32.Zero()))

        self.assertEqual(
            tuple(ctx.spec_of(field).constant_fold() for field in decoded),
            tuple(RealLit(value) for value in (0, 0, 0, 0, 0, 1, 0, 0)),
        )
        self.assertIs(decoded.sign, decoded[0])
        self.assertIs(decoded.is_nan, decoded[7])

    def test_fp32_encoder_spec_canonicalizes_exact_zero(self):
        for sign in (0, 1):
            with self.subTest(sign=sign):
                ctx = SpecContext(f"fp32-encode-zero-sign-{sign}")
                encoded = fp32_encode_spec(
                    RealLit(sign),
                    RealLit(0),
                    RealLit(0),
                    ctx,
                )
                ctx.check(encoded.is_pzero)

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_fp32_encode_constructs_result_without_calling_fresh(self):
        ctx = SpecContext("direct-fp32-encode")

        with patch.object(
            fp32,
            "fresh",
            side_effect=AssertionError("encode must construct fp32 directly"),
        ):
            encoded = fp32.encode(RealLit(1), ctx)

        self.assertIsInstance(encoded, fp32)
        self.assertNotIsInstance(encoded.is_norm, BoolVar)
        self.assertIsInstance(encoded.exponent, RealVar)
        self.assertIsInstance(encoded.mantissa, RealVar)

    def test_fp32_encode_design_canonicalizes_exact_zero(self):
        for sign in (0, 1):
            with self.subTest(sign=sign):
                design = fp32_encode(
                    Const(UQ(sign, 1, 0)),
                    Const(Q.from_int(127)),
                    Const(UQ.from_float(0, 1, Float32.mantissa_bits)),
                )
                self.assertEqual(design.evaluate(), Float32.Zero())

    def test_fp32_encode_spec_rounds_negative_underflow_to_negative_zero(self):
        # 2^-150 is exactly halfway between zero and the smallest binary32
        # subnormal. RNE selects the even encoding, zero, and keeps the sign.
        for name, value in (
            ("below-midpoint", -(2.0 ** -151)),
            ("midpoint", -(2.0 ** -150)),
        ):
            with self.subTest(name=name):
                ctx = SpecContext(f"fp32-encode-{name}")
                encoded = fp32.encode(RealLit(value), ctx)
                ctx.check(encoded.is_nzero)

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_fp32_encode_design_preserves_negative_underflow_sign(self):
        sign = Const(UQ(1, 1, 0))
        exponent = Const(Q.from_int(-22))

        cases = (
            (0.25, Float32.nZero()),
            (0.5, Float32.nZero()),
            (0.75, Float32.from_fields(1, 0, 1)),
        )
        for mantissa, expected in cases:
            with self.subTest(mantissa=mantissa):
                design = fp32_encode(
                    sign,
                    exponent,
                    Const(UQ.from_float(mantissa, 1, Float32.mantissa_bits)),
                )
                self.assertEqual(design.evaluate(), expected)

    def test_shift_if_subnormal_accepts_configurable_extra_bits(self):
        mantissa = Const(UQ.from_float(1.5, 1, 3))
        exponent = Const(Q.from_int(1))

        default_mantissa, _ = shift_if_subnormal(mantissa, exponent)
        custom_mantissa, _ = shift_if_subnormal(
            mantissa,
            exponent,
            subnormal_extra_bits=5,
        )

        self.assertEqual(default_mantissa.node_type, UQT(1, 6))
        self.assertEqual(custom_mantissa.node_type, UQT(1, 8))
        self.assertEqual(default_mantissa.evaluate().to_val(), 1.5)
        self.assertEqual(custom_mantissa.evaluate().to_val(), 1.5)

        with self.assertRaisesRegex(TypeError, "must be an int"):
            shift_if_subnormal(mantissa, exponent, subnormal_extra_bits=1.5)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            shift_if_subnormal(mantissa, exponent, subnormal_extra_bits=-1)

    def test_fp32_multiplier_spec_zero_handling(self):
        from examples.fp32_mult import spec_fp32_mult

        cases = (
            ("+0 * +0", 0x00000000, 0x00000000, "is_pzero"),
            ("+0 * -0", 0x00000000, 0x80000000, "is_nzero"),
            ("-0 * +0", 0x80000000, 0x00000000, "is_nzero"),
            ("-0 * -0", 0x80000000, 0x80000000, "is_pzero"),
            ("+0 * -1", 0x00000000, 0xbf800000, "is_nzero"),
            ("-0 * +1", 0x80000000, 0x3f800000, "is_nzero"),
            ("-1 * +0", 0xbf800000, 0x00000000, "is_nzero"),
            ("+1 * -0", 0x3f800000, 0x80000000, "is_nzero"),
            ("+0 * +inf", 0x00000000, 0x7f800000, "is_nan"),
            ("-0 * -inf", 0x80000000, 0xff800000, "is_nan"),
            ("+inf * -0", 0x7f800000, 0x80000000, "is_nan"),
            ("-inf * +0", 0xff800000, 0x00000000, "is_nan"),
        )

        for name, lhs_bits, rhs_bits, predicate in cases:
            with self.subTest(name=name):
                ctx = SpecContext(f"fp32-mult-{name}")
                result = spec_fp32_mult(
                    Float32(lhs_bits).to_spec(ctx),
                    Float32(rhs_bits).to_spec(ctx),
                    ctx,
                )
                ctx.check(getattr(result, predicate))

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_fp32_multiplier_spec_preserves_underflow_zero_sign(self):
        from examples.fp32_mult import spec_fp32_mult

        cases = (
            ("positive half-minimum tie", 0x00000001, 0x3f000000, "is_pzero"),
            ("negative half-minimum tie", 0x80000001, 0x3f000000, "is_nzero"),
            ("negative min-normal underflow", 0x80800000, 0x33800000, "is_nzero"),
            ("two negatives underflow", 0x80800000, 0xb3800000, "is_pzero"),
        )

        for name, lhs_bits, rhs_bits, predicate in cases:
            with self.subTest(name=name):
                ctx = SpecContext(f"fp32-mult-{name}")
                result = spec_fp32_mult(
                    Float32(lhs_bits).to_spec(ctx),
                    Float32(rhs_bits).to_spec(ctx),
                    ctx,
                )
                ctx.check(getattr(result, predicate))

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_fp32_adder_spec_preserves_single_infinity(self):
        from examples.fp32_add import spec_fp32_add

        ctx = SpecContext("fp32-adder-single-infinity")
        cases = (
            (fp32.inf(ctx), fp32.zero(ctx), "is_pinf"),
            (fp32.zero(ctx), fp32.inf(ctx), "is_pinf"),
            (fp32.ninf(ctx), fp32.zero(ctx), "is_ninf"),
            (fp32.zero(ctx), fp32.ninf(ctx), "is_ninf"),
        )

        for lhs, rhs, expected_predicate in cases:
            with self.subTest(lhs=lhs, rhs=rhs):
                result = spec_fp32_add(lhs, rhs, ctx)
                ctx.validate_requirements(timeout_ms=1000)
                self.assertEqual(
                    getattr(result, expected_predicate).constant_fold(),
                    BoolLit(True),
                )

    def test_fp32_adder_cases_reject_missing_nan_branch(self):
        ctx = SpecContext("fp32-adder-missing-nan-case")
        x = fp32.inf(ctx)
        y = fp32.ninf(ctx)
        nan_case = (
            x.is_nan
            | y.is_nan
            | (x.is_pinf & y.is_ninf)
            | (x.is_ninf & y.is_pinf)
        )
        neg_inf_case = (x.is_ninf | y.is_ninf) & (~nan_case)
        pos_inf_case = (x.is_pinf | y.is_pinf) & (~nan_case)

        Cases(
            case(neg_inf_case, fp32.ninf(ctx)),
            case(pos_inf_case, fp32.inf(ctx)),
            case(x.is_finite & y.is_finite, fp32.zero(ctx)),
            ctx=ctx,
        )

        with self.assertRaises(MalformedSpecification):
            ctx.validate_requirements(timeout_ms=1000)

    def test_constant_fold_partially_rebuilds_symbolic_real_expr(self):
        x = RealVar("x")
        expr = x + (RealLit(2) + RealLit(3))

        self.assertEqual(expr.constant_fold(), Add(x, RealLit(5)))

    def test_constant_fold_partially_rebuilds_symbolic_bool_expr(self):
        p = BoolVar("p")
        expr = p | RealLit(2).eq(RealLit(2))

        self.assertEqual(expr.constant_fold(), BoolLit(True))

    def test_constant_fold_folds_boolean_operator_trees(self):
        expr = RealLit(2).eq(RealLit(2)) & ~BoolLit(False)

        self.assertEqual(expr.constant_fold(), BoolLit(True))

    def test_constant_fold_factors_complementary_boolean_partitions(self):
        p = BoolVar("p")
        q = BoolVar("q")
        r = BoolVar("r")

        cases = (
            ((p & q) | (p & (~q)), p),
            ((q & p) | ((~q) & p), p),
            (((p & q) & r) | ((p & (~q)) & r), r & p),
        )

        for expr, expected in cases:
            with self.subTest(expr=str(expr)):
                self.assertEqual(expr.constant_fold(), expected.constant_fold())

    def test_context_simplify_leaves_finite_if_equality_unchanged(self):
        select = BoolVar("select")
        indicator_complement = RealLit(1) - If(
            select,
            RealLit(1),
            RealLit(0),
        )
        equality = indicator_complement.eq(RealLit(1))

        self.assert_check_simplifies_to(equality, equality)

    def test_constant_fold_leaves_finite_symbolic_equality_unchanged(self):
        equality = RealVar("x").eq(RealVar("y"))

        self.assertIs(equality.constant_fold(), equality)

    def test_constant_fold_leaves_unsupported_literal_pow_unchanged(self):
        expr = Pow(RealLit(-2), RealLit(0.5))

        self.assertEqual(expr.constant_fold(), expr)

    def test_non_finite_real_literals_are_rejected_at_construction(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-finite RealLit"):
                    RealLit(value)

    def test_float32_value_selects_finite_formula_or_fresh_special(self):
        ctx = SpecContext("float32-finite-value")
        value = fp32.fresh("x", ctx)

        self.assertIsInstance(value.value, If)
        self.assertEqual(value.value.cond, value.is_norm)
        subnormal_case = value.value.on_false
        self.assertIsInstance(subnormal_case, If)
        self.assertEqual(subnormal_case.cond, value.is_sub)
        zero_case = subnormal_case.on_false
        self.assertIsInstance(zero_case, If)
        self.assertEqual(zero_case.cond, value.is_zero)
        self.assertEqual(zero_case.on_true, RealLit(0))
        self.assertIsInstance(zero_case.on_false, RealVar)
        self.assertTrue(zero_case.on_false.name.startswith("special_"))

    def test_fp32_uses_explicit_non_finite_constructors(self):
        ctx = SpecContext("explicit-fp32-specials")
        cases = (
            ("nan", fp32.nan(ctx), "is_nan"),
            ("inf", fp32.inf(ctx), "is_pinf"),
            ("ninf", fp32.ninf(ctx), "is_ninf"),
        )

        special_names = set()
        for name, encoded, expected_predicate in cases:
            with self.subTest(value=name):
                self.assertIsInstance(encoded.value, RealVar)
                self.assertTrue(encoded.value.name.startswith("special_"))
                special_names.add(encoded.value.name)
                self.assertEqual(
                    getattr(encoded, expected_predicate).constant_fold(),
                    BoolLit(True),
                )
        self.assertEqual(len(special_names), len(cases))

    def test_encode_fp32_infers_infinity_sign_from_value(self):
        cases = (
            ("positive", RealLit(1e300), "is_pinf"),
            ("negative", RealLit(-1e300), "is_ninf"),
        )

        for name, value, expected_predicate in cases:
            with self.subTest(value=name):
                ctx = SpecContext(f"encode-{name}-infinity")
                encoded = fp32.encode(value=value, ctx=ctx)
                ctx.check(getattr(encoded, expected_predicate))

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_float32_classification_observes_only_relevant_signs(self):
        def make_value(*, sign, norm=False, sub=False, zero=False, inf=False, nan=False):
            return fp32(
                value=RealVar("value"),
                sign=RealLit(sign),
                exponent=RealLit(0),
                mantissa=RealLit(0),
                is_norm=BoolLit(norm),
                is_sub=BoolLit(sub),
                is_zero=BoolLit(zero),
                is_inf=BoolLit(inf),
                is_nan=BoolLit(nan),
            )

        pinf = make_value(sign=0, inf=True).observables_for_classification("inf")
        ninf = make_value(sign=1, inf=True).observables_for_classification("inf")
        pzero = make_value(sign=0, zero=True).observables_for_classification("zero")
        nzero = make_value(sign=1, zero=True).observables_for_classification("zero")
        pnan = make_value(sign=0, nan=True).observables_for_classification("nan")
        nnan = make_value(sign=1, nan=True).observables_for_classification("nan")

        self.assertNotEqual(pinf[0].constant_fold(), ninf[0].constant_fold())
        self.assertNotEqual(pzero[0].constant_fold(), nzero[0].constant_fold())
        self.assertEqual(pnan[0].constant_fold(), nnan[0].constant_fold())

    def test_fp32_known_classification_exposes_only_relevant_fields(self):
        value = fp32(
            value=RealVar("value"),
            sign=RealVar("sign"),
            exponent=RealVar("exponent"),
            mantissa=RealVar("mantissa"),
            is_norm=BoolVar("is_norm"),
            is_sub=BoolVar("is_sub"),
            is_zero=BoolVar("is_zero"),
            is_inf=BoolVar("is_inf"),
            is_nan=BoolVar("is_nan"),
        )

        self.assertEqual(value.observables_for_classification("norm"), (value.value,))
        self.assertEqual(value.observables_for_classification("sub"), (value.value,))
        self.assertEqual(value.observables_for_classification("zero"), (value.sign,))
        self.assertEqual(value.observables_for_classification("inf"), (value.sign,))
        self.assertEqual(
            value.observables_for_classification("nan"),
            (BoolLit(True),),
        )
        with self.assertRaisesRegex(ValueError, "Unknown fp32 classification"):
            value.observables_for_classification("finite")

    def test_nested_fp32_outputs_are_split_and_lowered_to_scalar_queries(self):
        ctx = SpecContext("nested-fp32")
        inner = fp32.zero(ctx)
        outer = fp32.zero(ctx)

        cases = ast_nodes._split_classification_cases(
            ctx,
            [],
            ((inner,), RealLit(7)),
            ((outer,), RealLit(7)),
        )
        case = next(
            case
            for case in cases
            if ast_nodes._case_labels(case.name)
            == {
                "output.0.0": "zero",
            }
        )

        inner_query = tuple(inner.classification_flags().values()) + (inner.sign,)
        outer_query = tuple(outer.classification_flags().values()) + (outer.sign,)
        self.assertEqual(
            case.checks,
            [
                *(lhs.eq(rhs) for lhs, rhs in zip(inner_query, outer_query)),
                RealLit(7).eq(RealLit(7)),
            ],
        )
        labels = ast_nodes._case_labels(case.name)
        self.assertEqual(labels, {"output.0.0": "zero"})
        status, _ = solver_engine.check_equivalence(
            case,
            schedule=[{"tool": "simplify"}],
        )
        self.assertEqual(status, "unsat")

    def test_classification_cases_are_constructed_lazily(self):
        ctx = SpecContext("lazy-classification-cases")
        inner = fp32.zero(ctx)
        outer = fp32.zero(ctx)

        with patch.object(ctx, "copy", wraps=ctx.copy) as copy_ctx:
            cases = ast_nodes._split_classification_cases(
                ctx,
                [],
                inner,
                outer,
            )
            self.assertEqual(copy_ctx.call_count, 0)

            next(cases)
            self.assertEqual(copy_ctx.call_count, 1)

            next(cases)
            self.assertEqual(copy_ctx.call_count, 2)

    def test_corresponding_fp32_outputs_share_classification_cases(self):
        ctx = SpecContext("paired-output-classification-cases")
        inner = fp32.fresh("inner", ctx)
        outer = fp32.fresh("outer", ctx)

        cases = list(ast_nodes._split_classification_cases(
            ctx,
            [],
            inner,
            outer,
        ))

        self.assertEqual(len(cases), len(inner.classification_flags()))
        for split_ctx in cases:
            labels = ast_nodes._case_labels(split_ctx.name)
            self.assertEqual(set(labels), {"output"})


class TestFloat16Spec(unittest.TestCase):
    def test_float16_runtime_layout_values_and_specials(self):
        self.assertEqual(Float16.mantissa_bits, 10)
        self.assertEqual(Float16.exponent_bits, 5)
        self.assertEqual(Float16.exponent_bias, 15)
        self.assertEqual(Float16.Zero().val, 0x0000)
        self.assertEqual(Float16.nZero().val, 0x8000)
        self.assertEqual(Float16.Inf().val, 0x7C00)
        self.assertEqual(Float16.nInf().val, 0xFC00)
        self.assertEqual(Float16.NaN().val, 0x7E00)
        self.assertEqual(Float16.NaN(1).val, 0x7C01)

        largest_negative = Float16.from_fields(
            sign=1,
            exponent=30,
            mantissa=1023,
        )
        self.assertEqual(largest_negative.val, 0xFBFF)
        self.assertEqual(largest_negative.sign, 1)
        self.assertEqual(largest_negative.exponent, 30)
        self.assertEqual(largest_negative.mantissa, 1023)
        self.assertEqual(largest_negative.copy(), largest_negative)
        self.assertEqual(largest_negative.static_type(), Float16T())

        self.assertEqual(Float16(0x0001).to_val(), 2 ** -24)
        self.assertEqual(Float16(0x03FF).to_val(), 1023 * (2 ** -24))
        self.assertEqual(Float16(0x0400).to_val(), 2 ** -14)
        self.assertEqual(Float16(0x3C00).to_val(), 1.0)
        self.assertEqual(Float16(0x7BFF).to_val(), 65504.0)
        self.assertTrue(math.isinf(Float16.Inf().to_val()))
        self.assertTrue(math.isnan(Float16.NaN().to_val()))
        self.assertLess(math.copysign(1.0, Float16.nZero().to_val()), 0)

    def test_float16_runtime_rejects_invalid_encodings(self):
        with self.assertRaisesRegex(TypeError, "packed bits as int"):
            Float16(1.0)
        with self.assertRaisesRegex(ValueError, "packed bits must fit"):
            Float16(1 << 16)
        with self.assertRaisesRegex(ValueError, "sign must be 0 or 1"):
            Float16.from_fields(2, 0, 0)
        with self.assertRaisesRegex(ValueError, "exponent out of range"):
            Float16.from_fields(0, 32, 0)
        with self.assertRaisesRegex(ValueError, "mantissa out of range"):
            Float16.from_fields(0, 0, 1024)
        with self.assertRaisesRegex(TypeError, "NaN payload must be int"):
            Float16.NaN("quiet")
        with self.assertRaisesRegex(ValueError, "non-zero"):
            Float16.NaN(0)

    def test_fp16_format_and_explicit_classifications(self):
        self.assertEqual(fp16.exponent_bits, 5)
        self.assertEqual(fp16.mantissa_bits, 10)
        self.assertEqual(fp16.exponent_bias, 15)

        ctx = SpecContext("explicit-fp16-values")
        cases = (
            (fp16.nan(ctx), "is_nan"),
            (fp16.inf(ctx), "is_pinf"),
            (fp16.ninf(ctx), "is_ninf"),
            (fp16.zero(ctx), "is_pzero"),
            (fp16.nzero(ctx), "is_nzero"),
        )
        special_names = set()
        for value, predicate in cases:
            with self.subTest(predicate=predicate):
                self.assertEqual(
                    getattr(value, predicate).constant_fold(),
                    BoolLit(True),
                )
                if predicate in {"is_nan", "is_pinf", "is_ninf"}:
                    self.assertIsInstance(value.value, RealVar)
                    special_names.add(value.value.name)
                else:
                    self.assertEqual(value.value, RealLit(0))
        self.assertEqual(len(special_names), 3)

    def test_float16_runtime_and_static_types_produce_structured_specs(self):
        runtime_cases = (
            (Float16.Zero(), "is_pzero"),
            (Float16.nZero(), "is_nzero"),
            (Float16(0x0001), "is_sub"),
            (Float16(0x3C00), "is_norm"),
            (Float16.Inf(), "is_pinf"),
            (Float16.nInf(), "is_ninf"),
            (Float16.NaN(), "is_nan"),
        )
        for runtime_value, predicate in runtime_cases:
            with self.subTest(bits=runtime_value.val):
                ctx = SpecContext("runtime-fp16")
                value = runtime_value.to_spec(ctx)
                self.assertIsInstance(value, fp16)
                self.assertEqual(
                    getattr(value, predicate).constant_fold(),
                    BoolLit(True),
                )

        ctx = SpecContext("static-fp16")
        self.assertIsInstance(Float16T().to_spec("input", ctx), fp16)
        rng = random.Random(1)
        self.assertIsInstance(Float16T().random_runtime_value(rng), Float16)

    def test_fp16_encode_classifies_representative_boundaries(self):
        cases = (
            ("positive-zero", 0.0, "is_pzero"),
            ("negative-underflow", -(2 ** -26), "is_nzero"),
            ("smallest-subnormal", 2 ** -24, "is_sub"),
            ("smallest-normal", 2 ** -14, "is_norm"),
            ("greatest-normal", 65504.0, "is_norm"),
            ("positive-overflow", 70000.0, "is_pinf"),
            ("negative-overflow", -70000.0, "is_ninf"),
        )

        for name, real_value, predicate in cases:
            with self.subTest(value=name):
                ctx = SpecContext(f"fp16-encode-{name}")
                encoded = fp16.encode(RealLit(real_value), ctx)
                ctx.check(getattr(encoded, predicate))

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

        with self.assertRaisesRegex(TypeError, "fp16.encode value must be RealExpr"):
            fp16.encode(1.0, SpecContext("invalid-fp16-encode"))

    def test_fp16_decode_names_and_classifies_fields(self):
        cases = (
            (Float16.Zero(), (0, 0, 0, 0, 0, 1, 0, 0)),
            (Float16.nZero(), (1, 0, 0, 0, 0, 1, 0, 0)),
            (Float16(0x0001), (0, 0, 1, 0, 1, 0, 0, 0)),
            (Float16(0x3C00), (0, 15, 0, 1, 0, 0, 0, 0)),
            (Float16.Inf(), (0, 31, 0, 0, 0, 0, 1, 0)),
            (Float16.NaN(), (0, 31, 512, 0, 0, 0, 0, 1)),
        )

        for runtime_value, expected in cases:
            with self.subTest(bits=runtime_value.val):
                decoded = fp16_decode(Const(runtime_value))
                self.assertEqual(
                    tuple(field.evaluate().val for field in decoded),
                    expected,
                )
                self.assertIs(decoded.sign, decoded[0])
                self.assertIs(decoded.is_nan, decoded[7])

    def test_fp16_pack_round_trips_in_python_and_cpp(self):
        def identity_spec(x, ctx):
            del ctx
            return x

        @Composite(name="fp16_pack_decode_roundtrip", spec=identity_spec)
        def fp16_pack_decode_roundtrip(x):
            decoded = fp16_decode(x)
            return fp16_pack(decoded.sign, decoded.exponent, decoded.mantissa)

        value = Var(name="value", sign=Float16T())
        design = fp16_pack_decode_roundtrip(value)
        tempdir, compiled = jit_compile(design)
        cases = (0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x7C00, 0x7E00, 0xFFFF)
        try:
            for bits in cases:
                with self.subTest(bits=bits):
                    value.load_val(Float16(bits))
                    self.assertEqual(design.evaluate().val, bits)
                    self.assertEqual(compiled(bits), bits)
        finally:
            tempdir.cleanup()

    def test_fp16_observables_match_other_ieee_formats(self):
        value = fp16(
            value=RealVar("value"),
            sign=RealVar("sign"),
            exponent=RealVar("exponent"),
            mantissa=RealVar("mantissa"),
            is_norm=BoolVar("is_norm"),
            is_sub=BoolVar("is_sub"),
            is_zero=BoolVar("is_zero"),
            is_inf=BoolVar("is_inf"),
            is_nan=BoolVar("is_nan"),
        )

        self.assertEqual(value.observables_for_classification("norm"), (value.value,))
        self.assertEqual(value.observables_for_classification("sub"), (value.value,))
        self.assertEqual(value.observables_for_classification("zero"), (value.sign,))
        self.assertEqual(value.observables_for_classification("inf"), (value.sign,))
        self.assertEqual(value.observables_for_classification("nan"), (BoolLit(True),))
        with self.assertRaisesRegex(ValueError, "Unknown fp16 classification"):
            value.observables_for_classification("finite")


class TestE4M3FNSpec(unittest.TestCase):
    def test_runtime_layout_values_and_validation(self):
        self.assertEqual(E4M3FN.Zero().val, 0x00)
        self.assertEqual(E4M3FN.nZero().val, 0x80)
        self.assertEqual(E4M3FN.NaN().val, 0x7F)
        self.assertEqual(E4M3FN(0x01).to_val(), 2 ** -9)
        self.assertEqual(E4M3FN(0x07).to_val(), 7 * 2 ** -9)
        self.assertEqual(E4M3FN(0x08).to_val(), 2 ** -6)
        self.assertEqual(E4M3FN(0x7E).to_val(), 448.0)
        self.assertEqual(E4M3FN(0xFE).to_val(), -448.0)
        self.assertTrue(math.isnan(E4M3FN(0x7F).to_val()))
        self.assertTrue(math.isnan(E4M3FN(0xFF).to_val()))

        value = E4M3FN.from_fields(1, 15, 6)
        self.assertEqual((value.sign, value.exponent, value.mantissa), (1, 15, 6))
        self.assertEqual(value.copy(), value)
        self.assertEqual(value.static_type(), E4M3FNT())
        self.assertEqual(value.total_bits(), 8)

        for invalid in (-1, 256, 1.5, "0"):
            with self.subTest(packed=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    E4M3FN(invalid)
        for fields in ((2, 0, 0), (0, -1, 0), (0, 16, 0), (0, 0, -1), (0, 0, 8)):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    E4M3FN.from_fields(*fields)

    def test_exhaustive_decode_classification_and_values(self):
        counts = {"norm": 0, "sub": 0, "zero": 0, "nan": 0}
        names = tuple(counts)
        for bits in range(256):
            runtime = E4M3FN(bits)
            decoded = e4m3fn_decode(Const(runtime))
            fields = tuple(node.evaluate().val for node in decoded)
            self.assertEqual(fields[:3], (bits >> 7, (bits >> 3) & 15, bits & 7))
            self.assertEqual(sum(fields[3:]), 1)
            counts[names[fields[3:].index(1)]] += 1

            spec = runtime.to_spec(SpecContext(f"e4m3fn-{bits}"))
            expected = spec.classification_flags()[names[fields[3:].index(1)]]
            self.assertEqual(expected.constant_fold(), BoolLit(True))

        self.assertEqual(counts, {"norm": 238, "sub": 14, "zero": 2, "nan": 2})

    def test_pack_decode_round_trip_in_python_and_jit_cpp(self):
        @Composite(name="e4m3fn_pack_decode_roundtrip", spec=lambda x, ctx: x)
        def roundtrip(x):
            decoded = e4m3fn_decode(x)
            return e4m3fn_pack(decoded.sign, decoded.exponent, decoded.mantissa)

        value = Var("value", sign=E4M3FNT())
        design = roundtrip(value)
        tempdir, compiled = jit_compile(design)
        try:
            for bits in range(256):
                value.load_val(E4M3FN(bits))
                self.assertEqual(design.evaluate().val, bits)
                self.assertEqual(compiled(bits), bits)
        finally:
            tempdir.cleanup()

    def test_symbolic_shape_classification_and_observables(self):
        ctx = SpecContext("fresh-e4m3fn")
        value = e4m3fn.fresh("x", ctx)
        self.assertEqual(tuple(value.classification_flags()), ("norm", "sub", "zero", "nan"))
        self.assertNotIn("inf", value.classification_flags())
        self.assertEqual(value.observables_for_classification("norm"), (value.value,))
        self.assertEqual(value.observables_for_classification("sub"), (value.value,))
        self.assertEqual(value.observables_for_classification("zero"), (value.sign,))
        self.assertEqual(value.observables_for_classification("nan"), (BoolLit(True),))
        self.assertIsInstance(E4M3FNT().to_spec("input", ctx), e4m3fn)
        self.assertIsInstance(E4M3FNT().random_runtime_value(random.Random(1)), E4M3FN)

        self.assertEqual(e4m3fn.zero(ctx).is_pzero.constant_fold(), BoolLit(True))
        self.assertEqual(e4m3fn.nzero(ctx).is_nzero.constant_fold(), BoolLit(True))
        with self.assertRaisesRegex(ValueError, "Unknown e4m3fn classification"):
            value.observables_for_classification("inf")

    def test_symbolic_encode_boundaries(self):
        cases = (
            (0.0, "is_pzero"),
            (2 ** -10, "is_pzero"),
            (2 ** -9, "is_sub"),
            (2 ** -6, "is_norm"),
            (448.0, "is_norm"),
            (1000.0, "is_norm"),
            (-1000.0, "is_norm"),
        )
        for real_value, predicate in cases:
            with self.subTest(real_value=real_value):
                ctx = SpecContext("encode-e4m3fn")
                encoded = e4m3fn.encode(RealLit(real_value), ctx)
                ctx.check(getattr(encoded, predicate))
                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)
                self.assertEqual(report["status"], "unsat", report)

    def test_bit_level_encoder_rne_canonical_zero_and_saturation(self):
        sign = Var("sign", UQT(1, 0))
        exponent = Var("exponent", QT(8, 0))
        mantissa = Var("mantissa", UQT(12, 12))
        design = e4m3fn_encode(sign, exponent, mantissa)
        tempdir, compiled = jit_compile(design)

        def q8(value):
            return Q(value & 0xFF, 8, 0)

        cases = (
            (0, 7, 0.0, 0x00),
            (1, 7, 0.0, 0x00),
            (0, 7, 1.0, 0x38),
            (1, 7, 1.0, 0xB8),
            (0, -1, 0.25, 0x00),  # half of the minimum subnormal
            (1, -1, 0.25, 0x80),  # negative nonzero underflow retains its sign
            (0, -1, 0.75, 0x02),  # 1.5 subnormal units ties to even
            (0, -1, 1.25, 0x02),  # 2.5 subnormal units ties to even
            (0, -1, 3.75, 0x08),  # largest-sub/min-normal midpoint
            (0, 7, 1.0625, 0x38),  # normal tie rounds to even mantissa 0
            (0, 7, 1.1875, 0x3A),  # normal tie rounds to even mantissa 2
            (0, 7, 1.9375, 0x40),  # rounding carry increments exponent
            (0, 15, 1.75, 0x7E),
            (0, 15, 1.875, 0x7E),  # reserved NaN result saturates
            (0, 20, 1.0, 0x7E),
            (1, 20, 1.0, 0xFE),
        )
        try:
            for sign_value, exponent_value, mantissa_value, expected in cases:
                with self.subTest(
                    sign=sign_value,
                    exponent=exponent_value,
                    mantissa=mantissa_value,
                ):
                    sign_runtime = UQ(sign_value, 1, 0)
                    exponent_runtime = q8(exponent_value)
                    mantissa_runtime = UQ.from_float(mantissa_value, 12, 12)
                    sign.load_val(sign_runtime)
                    exponent.load_val(exponent_runtime)
                    mantissa.load_val(mantissa_runtime)
                    self.assertEqual(design.evaluate().val, expected)
                    self.assertEqual(
                        compiled(
                            sign_runtime.val,
                            exponent_runtime.val,
                            mantissa_runtime.val,
                        ),
                        expected,
                    )
        finally:
            tempdir.cleanup()

    def test_encoder_recreates_every_finite_encoding(self):
        sign = Var("finite_sign", UQT(1, 0))
        exponent = Var("finite_exponent", QT(8, 0))
        mantissa = Var("finite_mantissa", UQT(2, 12))
        design = e4m3fn_encode(sign, exponent, mantissa)
        tempdir, compiled = jit_compile(design)
        try:
            for bits in range(256):
                runtime = E4M3FN(bits)
                if runtime.is_nan:
                    continue
                if runtime.exponent == 0:
                    exponent_value = 1
                    mantissa_value = runtime.mantissa / 8
                else:
                    exponent_value = runtime.exponent
                    mantissa_value = 1 + runtime.mantissa / 8

                sign_value = UQ(runtime.sign, 1, 0)
                exponent_value = Q(exponent_value, 8, 0)
                mantissa_value = UQ.from_float(mantissa_value, 2, 12)
                sign.load_val(sign_value)
                exponent.load_val(exponent_value)
                mantissa.load_val(mantissa_value)
                expected = 0x00 if bits == E4M3FN.nZero().val else bits
                self.assertEqual(design.evaluate().val, expected)
                self.assertEqual(
                    compiled(sign_value.val, exponent_value.val, mantissa_value.val),
                    expected,
                )
        finally:
            tempdir.cleanup()

    def test_saturating_encoder_determinism_and_specification_proofs(self):
        def spec(ctx):
            return e4m3fn.encode(ctx.two() ** ctx.real_val(13), ctx)

        @Composite(name="e4m3fn_encode_saturation_proof", spec=spec)
        def saturation_design():
            return e4m3fn_encode(
                Const(UQ(0, 1, 0)),
                Const(Q.from_int(20)),
                Const(UQ.from_int(1)),
            )

        design = saturation_design()
        schedule = [
            {"tool": "simplify"},
            {"tool": "z3", "timeout_ms": 5000},
        ]
        self.assertTrue(design.check_determinism(schedule=schedule)["proved"])
        self.assertTrue(design.check_spec(schedule=schedule)["proved"])


class TestUE4M3Spec(unittest.TestCase):
    def test_runtime_layout_values_and_validation(self):
        self.assertEqual(UE4M3.Zero().val, 0x00)
        self.assertEqual(UE4M3.NaN().val, 0x7F)
        self.assertEqual(UE4M3(0x01).to_val(), 2 ** -9)
        self.assertEqual(UE4M3(0x07).to_val(), 7 * 2 ** -9)
        self.assertEqual(UE4M3(0x08).to_val(), 2 ** -6)
        self.assertEqual(UE4M3(0x38).to_val(), 1.0)
        self.assertEqual(UE4M3(0x7E).to_val(), 448.0)
        self.assertTrue(math.isnan(UE4M3(0x7F).to_val()))

        value = UE4M3.from_fields(15, 6)
        self.assertEqual((value.exponent, value.mantissa), (15, 6))
        self.assertEqual(value.copy(), value)
        self.assertEqual(value.static_type(), UE4M3T())
        self.assertEqual(value.total_bits(), 8)

        for invalid in (-1, 128, 255, 1.5, "0"):
            with self.subTest(packed=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    UE4M3(invalid)
        for fields in ((-1, 0), (16, 0), (0, -1), (0, 8)):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    UE4M3.from_fields(*fields)

    def test_exhaustive_decode_classification_values_and_pack_round_trip(self):
        @Composite(name="ue4m3_pack_decode_roundtrip", spec=lambda x, ctx: x)
        def roundtrip(x):
            decoded = ue4m3_decode(x)
            return ue4m3_pack(decoded.exponent, decoded.mantissa)

        value = Var("value", sign=UE4M3T())
        design = roundtrip(value)
        tempdir, compiled = jit_compile(design)
        counts = {"norm": 0, "sub": 0, "zero": 0, "nan": 0}
        names = tuple(counts)
        try:
            for bits in range(128):
                runtime = UE4M3(bits)
                decoded = ue4m3_decode(Const(runtime))
                fields = tuple(node.evaluate().val for node in decoded)
                self.assertEqual(fields[:2], ((bits >> 3) & 15, bits & 7))
                self.assertEqual(sum(fields[2:]), 1)
                classification = names[fields[2:].index(1)]
                counts[classification] += 1

                spec = runtime.to_spec(SpecContext(f"ue4m3-{bits}"))
                self.assertEqual(
                    spec.classification_flags()[classification].constant_fold(),
                    BoolLit(True),
                )
                value.load_val(runtime)
                self.assertEqual(design.evaluate().val, bits)
                self.assertEqual(compiled(bits), bits)
        finally:
            tempdir.cleanup()

        self.assertEqual(counts, {"norm": 119, "sub": 7, "zero": 1, "nan": 1})

    def test_symbolic_shape_and_magnitude_encoding(self):
        ctx = SpecContext("fresh-ue4m3")
        value = ue4m3.fresh("x", ctx)
        self.assertEqual(
            tuple(value.classification_flags()),
            ("norm", "sub", "zero", "nan"),
        )
        self.assertEqual(value.observables_for_classification("zero"), (BoolLit(True),))
        self.assertIsInstance(UE4M3T().to_spec("input", ctx), ue4m3)
        self.assertIsInstance(UE4M3T().random_runtime_value(random.Random(1)), UE4M3)

        positive_ctx = SpecContext("positive-ue4m3")
        positive = ue4m3.encode(RealLit(1), positive_ctx)
        negative_ctx = SpecContext("negative-ue4m3")
        negative = ue4m3.encode(RealLit(-1), negative_ctx)
        self.assertEqual(positive.value.constant_fold(), RealLit(1))
        self.assertEqual(negative.value.constant_fold(), RealLit(1))

    def test_bit_level_encoder_rne_zero_and_saturation(self):
        exponent = Var("exponent", QT(8, 0))
        mantissa = Var("mantissa", UQT(12, 12))
        design = ue4m3_encode(exponent, mantissa)
        tempdir, compiled = jit_compile(design)

        cases = (
            (7, 0.0, 0x00),
            (7, 1.0, 0x38),
            (-1, 0.25, 0x00),
            (-1, 0.75, 0x02),
            (-1, 1.25, 0x02),
            (-1, 3.75, 0x08),
            (7, 1.0625, 0x38),
            (7, 1.1875, 0x3A),
            (7, 1.9375, 0x40),
            (15, 1.75, 0x7E),
            (15, 1.875, 0x7E),
            (20, 1.0, 0x7E),
        )
        try:
            for exponent_value, mantissa_value, expected in cases:
                exponent_runtime = Q(exponent_value & 0xFF, 8, 0)
                mantissa_runtime = UQ.from_float(mantissa_value, 12, 12)
                exponent.load_val(exponent_runtime)
                mantissa.load_val(mantissa_runtime)
                with self.subTest(exponent=exponent_value, mantissa=mantissa_value):
                    self.assertEqual(design.evaluate().val, expected)
                    self.assertEqual(
                        compiled(exponent_runtime.val, mantissa_runtime.val),
                        expected,
                    )
        finally:
            tempdir.cleanup()

    def test_fp32_converter_pins_rne_magnitude_and_special_values(self):
        source = Var("source", Float32T())
        design = fp32_to_ue4m3(source)
        tempdir, compiled = jit_compile(design)
        cases = (
            (0x00000000, 0x00),
            (0x80000000, 0x00),
            (0x3A800000, 0x00),
            (0x3B000000, 0x01),
            (0x3B400000, 0x02),
            (0x3BA00000, 0x02),
            (0x3C700000, 0x08),
            (0x3F880000, 0x38),
            (0x3F980000, 0x3A),
            (0xBF980000, 0x3A),
            (0x43E00000, 0x7E),
            (0x44000000, 0x7E),
            (0x7F800000, 0x7E),
            (0xFF800000, 0x7E),
            (0x7FC00000, 0x7F),
        )
        try:
            for source_bits, expected in cases:
                source.load_val(Float32(source_bits))
                with self.subTest(source=hex(source_bits)):
                    self.assertEqual(design.evaluate().val, expected)
                    self.assertEqual(compiled(source_bits), expected)
        finally:
            tempdir.cleanup()


class TestE5M2Spec(unittest.TestCase):
    def test_runtime_constants_constructors_validation_copy_and_static_type(self):
        self.assertEqual(E5M2.exponent_bias, 15)
        self.assertEqual(E5M2.min_subnormal, 2 ** -16)
        self.assertEqual(E5M2.min_normal, 2 ** -14)
        self.assertEqual(E5M2.max_finite, 57344.0)
        self.assertEqual(E5M2.Zero().val, 0x00)
        self.assertEqual(E5M2.nZero().val, 0x80)
        self.assertEqual(E5M2.Inf().val, 0x7C)
        self.assertEqual(E5M2.nInf().val, 0xFC)
        self.assertEqual(E5M2.NaN().val, 0x7E)
        self.assertEqual(
            [E5M2.NaN(payload).val for payload in range(1, 4)],
            [0x7D, 0x7E, 0x7F],
        )
        self.assertEqual(E5M2(0x01).to_val(), 2 ** -16)
        self.assertEqual(E5M2(0x03).to_val(), 3 * 2 ** -16)
        self.assertEqual(E5M2(0x04).to_val(), 2 ** -14)
        self.assertEqual(E5M2(0x7B).to_val(), 57344.0)
        self.assertEqual(E5M2(0xFB).to_val(), -57344.0)
        self.assertEqual(E5M2(0x7C).to_val(), float("inf"))
        self.assertEqual(E5M2(0xFC).to_val(), float("-inf"))
        for bits in (0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF):
            self.assertTrue(math.isnan(E5M2(bits).to_val()))

        value = E5M2.from_fields(1, 30, 3)
        self.assertEqual((value.sign, value.exponent, value.mantissa), (1, 30, 3))
        self.assertEqual(value.copy(), value)
        self.assertEqual(value.static_type(), E5M2T())
        self.assertEqual(value.total_bits(), 8)
        generator, next_shared = E5M2.random_generator(
            seed=1, shared_exponent_bits=2
        )
        self.assertIsInstance(generator(), E5M2)
        self.assertEqual(next_shared() & 0x07, 0)

        for invalid in (-1, 256, 1.5, "0"):
            with self.subTest(packed=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                E5M2(invalid)
        for fields in (
            (2, 0, 0),
            (0, -1, 0),
            (0, 32, 0),
            (0, 0, -1),
            (0, 0, 4),
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                E5M2.from_fields(*fields)
        for payload in (0, 4, -1, 1.5, "2"):
            with self.subTest(payload=payload), self.assertRaises(
                (TypeError, ValueError)
            ):
                E5M2.NaN(payload)

    def test_exhaustive_classification_and_pack_decode_round_trip(self):
        counts = {"norm": 0, "sub": 0, "zero": 0, "inf": 0, "nan": 0}
        names = tuple(counts)

        @Composite(name="e5m2_pack_decode_roundtrip", spec=lambda x, ctx: x)
        def roundtrip(x):
            decoded = e5m2_decode(x)
            return e5m2_pack(decoded.sign, decoded.exponent, decoded.mantissa)

        value = Var("e5m2_value", sign=E5M2T())
        design = roundtrip(value)
        tempdir, compiled = jit_compile(design)
        try:
            for bits in range(256):
                runtime = E5M2(bits)
                decoded = e5m2_decode(Const(runtime))
                fields = tuple(node.evaluate().val for node in decoded)
                self.assertEqual(
                    fields[:3], (bits >> 7, (bits >> 2) & 31, bits & 3)
                )
                self.assertEqual(sum(fields[3:]), 1)
                classification = names[fields[3:].index(1)]
                counts[classification] += 1
                spec = runtime.to_spec(SpecContext(f"e5m2-{bits}"))
                self.assertEqual(
                    spec.classification_flags()[classification].constant_fold(),
                    BoolLit(True),
                )
                value.load_val(runtime)
                self.assertEqual(design.evaluate().val, bits)
                self.assertEqual(compiled(bits), bits)
        finally:
            tempdir.cleanup()
        self.assertEqual(
            counts,
            {"norm": 240, "sub": 6, "zero": 2, "inf": 2, "nan": 6},
        )

    def test_symbolic_shape_constructors_and_observables(self):
        ctx = SpecContext("fresh-e5m2")
        value = e5m2.fresh("x", ctx)
        self.assertEqual(
            tuple(value.classification_flags()),
            ("norm", "sub", "zero", "inf", "nan"),
        )
        self.assertEqual(value.observables_for_classification("norm"), (value.value,))
        self.assertEqual(value.observables_for_classification("sub"), (value.value,))
        self.assertEqual(value.observables_for_classification("zero"), (value.sign,))
        self.assertEqual(value.observables_for_classification("inf"), (value.sign,))
        self.assertEqual(value.observables_for_classification("nan"), (BoolLit(True),))
        with self.assertRaisesRegex(ValueError, "Unknown e5m2 classification"):
            value.observables_for_classification("finite")
        self.assertIsInstance(E5M2T().to_spec("input", ctx), e5m2)
        self.assertIsInstance(E5M2T().random_runtime_value(random.Random(1)), E5M2)
        self.assertEqual(e5m2.zero(ctx).is_pzero.constant_fold(), BoolLit(True))
        self.assertEqual(e5m2.nzero(ctx).is_nzero.constant_fold(), BoolLit(True))
        self.assertEqual(e5m2.inf(ctx).is_pinf.constant_fold(), BoolLit(True))
        self.assertEqual(e5m2.ninf(ctx).is_ninf.constant_fold(), BoolLit(True))
        self.assertEqual(e5m2.nan(ctx).is_nan.constant_fold(), BoolLit(True))

    def test_encoder_rne_canonical_zero_infinity_and_all_finite_encodings(self):
        sign = Var("e5m2_sign", UQT(1, 0))
        exponent = Var("e5m2_exponent", QT(8, 0))
        mantissa = Var("e5m2_mantissa", UQT(2, 12))
        design = e5m2_encode(sign, exponent, mantissa)
        tempdir, compiled = jit_compile(design)

        def run(sign_value, exponent_value, mantissa_value):
            sign_runtime = UQ(sign_value, 1, 0)
            exponent_runtime = Q(exponent_value & 0xFF, 8, 0)
            mantissa_runtime = UQ.from_float(mantissa_value, 2, 12)
            sign.load_val(sign_runtime)
            exponent.load_val(exponent_runtime)
            mantissa.load_val(mantissa_runtime)
            return (
                design.evaluate().val,
                compiled(sign_runtime.val, exponent_runtime.val, mantissa_runtime.val),
            )

        cases = (
            (0, 15, 0.0, 0x00),
            (1, 15, 0.0, 0x00),
            (0, -1, 0.5, 0x00),
            (1, -1, 0.5, 0x80),
            (0, -1, 1.5, 0x02),
            (0, -1, 2.5, 0x02),
            (0, -1, 3.5, 0x04),
            (0, 15, 1.125, 0x3C),
            (0, 15, 1.375, 0x3E),
            (0, 15, 1.875, 0x40),
            (0, 30, 1.75, 0x7B),
            (0, 30, 1.875, 0x7C),
            (0, 40, 1.0, 0x7C),
            (1, 40, 1.0, 0xFC),
        )
        try:
            for sign_value, exponent_value, mantissa_value, expected in cases:
                with self.subTest(
                    sign=sign_value,
                    exponent=exponent_value,
                    mantissa=mantissa_value,
                ):
                    self.assertEqual(
                        run(sign_value, exponent_value, mantissa_value),
                        (expected, expected),
                    )

            for bits in range(256):
                runtime = E5M2(bits)
                if runtime.is_inf or runtime.is_nan:
                    continue
                exponent_value = 1 if runtime.exponent == 0 else runtime.exponent
                mantissa_value = (
                    runtime.mantissa / 4
                    if runtime.exponent == 0
                    else 1 + runtime.mantissa / 4
                )
                expected = 0x00 if bits == E5M2.nZero().val else bits
                self.assertEqual(
                    run(runtime.sign, exponent_value, mantissa_value),
                    (expected, expected),
                )
        finally:
            tempdir.cleanup()

    def test_overflow_encoder_determinism_and_specification_proofs(self):
        def spec(ctx):
            return e5m2.encode(ctx.two() ** ctx.real_val(24), ctx)

        @Composite(name="e5m2_encode_overflow_proof", spec=spec)
        def overflow_design():
            return e5m2_encode(
                Const(UQ(0, 1, 0)),
                Const(Q.from_int(39)),
                Const(UQ.from_int(1)),
            )

        design = overflow_design()
        schedule = [{"tool": "simplify"}, {"tool": "z3", "timeout_ms": 5000}]
        self.assertTrue(design.check_determinism(schedule=schedule)["proved"])
        self.assertTrue(design.check_spec(schedule=schedule)["proved"])


class TestE5M2FNUZSpec(unittest.TestCase):
    def test_runtime_layout_values_validation_copy_and_static_type(self):
        self.assertEqual(E5M2FNUZ.Zero().val, 0x00)
        self.assertEqual(E5M2FNUZ.NaN().val, 0x80)
        self.assertEqual(E5M2FNUZ.exponent_bias, 16)
        self.assertEqual(E5M2FNUZ(0x01).to_val(), 2 ** -17)
        self.assertEqual(E5M2FNUZ(0x03).to_val(), 3 * 2 ** -17)
        self.assertEqual(E5M2FNUZ(0x04).to_val(), 2 ** -15)
        self.assertEqual(E5M2FNUZ(0x7F).to_val(), 57344.0)
        self.assertEqual(E5M2FNUZ(0xFF).to_val(), -57344.0)
        self.assertTrue(math.isnan(E5M2FNUZ(0x80).to_val()))

        value = E5M2FNUZ.from_fields(1, 31, 3)
        self.assertEqual((value.sign, value.exponent, value.mantissa), (1, 31, 3))
        self.assertEqual(value.copy(), value)
        self.assertEqual(value.static_type(), E5M2FNUZT())
        self.assertEqual(value.total_bits(), 8)
        generator, next_shared = E5M2FNUZ.random_generator(
            seed=1, shared_exponent_bits=2
        )
        self.assertIsInstance(generator(), E5M2FNUZ)
        self.assertEqual(next_shared() & 0x07, 0)

        for invalid in (-1, 256, 1.5, "0"):
            with self.subTest(packed=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    E5M2FNUZ(invalid)
        for fields in (
            (2, 0, 0),
            (0, -1, 0),
            (0, 32, 0),
            (0, 0, -1),
            (0, 0, 4),
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                E5M2FNUZ.from_fields(*fields)

    def test_exhaustive_classification_and_pack_decode_round_trip(self):
        counts = {"norm": 0, "sub": 0, "zero": 0, "nan": 0}
        names = tuple(counts)

        @Composite(name="e5m2fnuz_pack_decode_roundtrip", spec=lambda x, ctx: x)
        def roundtrip(x):
            decoded = e5m2fnuz_decode(x)
            return e5m2fnuz_pack(decoded.sign, decoded.exponent, decoded.mantissa)

        value = Var("e5m2fnuz_value", sign=E5M2FNUZT())
        design = roundtrip(value)
        tempdir, compiled = jit_compile(design)
        try:
            for bits in range(256):
                runtime = E5M2FNUZ(bits)
                decoded = e5m2fnuz_decode(Const(runtime))
                fields = tuple(node.evaluate().val for node in decoded)
                self.assertEqual(fields[:3], (bits >> 7, (bits >> 2) & 31, bits & 3))
                self.assertEqual(sum(fields[3:]), 1)
                classification = names[fields[3:].index(1)]
                counts[classification] += 1
                spec = runtime.to_spec(SpecContext(f"e5m2fnuz-{bits}"))
                self.assertEqual(
                    spec.classification_flags()[classification].constant_fold(),
                    BoolLit(True),
                )
                value.load_val(runtime)
                self.assertEqual(design.evaluate().val, bits)
                self.assertEqual(compiled(bits), bits)
        finally:
            tempdir.cleanup()
        self.assertEqual(counts, {"norm": 248, "sub": 6, "zero": 1, "nan": 1})

    def test_symbolic_shape_constructors_and_boundaries(self):
        ctx = SpecContext("fresh-e5m2fnuz")
        value = e5m2fnuz.fresh("x", ctx)
        self.assertEqual(
            tuple(value.classification_flags()), ("norm", "sub", "zero", "nan")
        )
        self.assertEqual(value.observables_for_classification("zero"), (BoolLit(True),))
        self.assertEqual(value.observables_for_classification("nan"), (BoolLit(True),))
        self.assertIsInstance(E5M2FNUZT().to_spec("input", ctx), e5m2fnuz)
        self.assertIsInstance(
            E5M2FNUZT().random_runtime_value(random.Random(1)), E5M2FNUZ
        )
        self.assertEqual(e5m2fnuz.zero(ctx).is_pzero.constant_fold(), BoolLit(True))
        self.assertEqual(e5m2fnuz.nan(ctx).is_nan.constant_fold(), BoolLit(True))
        self.assertFalse(hasattr(e5m2fnuz, "nzero"))
        self.assertFalse(hasattr(e5m2fnuz, "inf"))

        for real_value, predicate in (
            (0.0, "is_pzero"),
            (2 ** -18, "is_pzero"),
            (2 ** -17, "is_sub"),
            (2 ** -15, "is_norm"),
            (57344.0, "is_norm"),
            (100000.0, "is_norm"),
            (-100000.0, "is_norm"),
        ):
            boundary_ctx = SpecContext("encode-e5m2fnuz")
            encoded = e5m2fnuz.encode(RealLit(real_value), boundary_ctx)
            boundary_ctx.check(getattr(encoded, predicate))
            report = simplify_ctx(boundary_ctx)
            if report["status"] == "unknown":
                report = z3_check_eq(report["new_ctx"], timeout_ms=1000)
            self.assertEqual(report["status"], "unsat", report)

    def test_encoder_rne_unsigned_zero_saturation_and_all_finite_encodings(self):
        sign = Var("e5_sign", UQT(1, 0))
        exponent = Var("e5_exponent", QT(8, 0))
        mantissa = Var("e5_mantissa", UQT(2, 12))
        design = e5m2fnuz_encode(sign, exponent, mantissa)
        tempdir, compiled = jit_compile(design)

        def run(sign_value, exponent_value, mantissa_value):
            sign_runtime = UQ(sign_value, 1, 0)
            exponent_runtime = Q(exponent_value & 0xFF, 8, 0)
            mantissa_runtime = UQ.from_float(mantissa_value, 2, 12)
            sign.load_val(sign_runtime)
            exponent.load_val(exponent_runtime)
            mantissa.load_val(mantissa_runtime)
            return (
                design.evaluate().val,
                compiled(sign_runtime.val, exponent_runtime.val, mantissa_runtime.val),
            )

        try:
            cases = (
                (0, 16, 0.0, 0x00),
                (1, 16, 0.0, 0x00),
                (0, 1, 0.125, 0x00),
                (1, 1, 0.125, 0x00),
                (0, 1, 0.375, 0x02),
                (0, 1, 0.625, 0x02),
                (0, 1, 0.875, 0x04),
                (0, 16, 1.125, 0x40),
                (0, 16, 1.375, 0x42),
                (0, 16, 1.875, 0x44),
                (0, 31, 1.75, 0x7F),
                (1, 40, 1.0, 0xFF),
            )
            for sign_value, exponent_value, mantissa_value, expected in cases:
                with self.subTest(
                    sign=sign_value, exponent=exponent_value, mantissa=mantissa_value
                ):
                    self.assertEqual(
                        run(sign_value, exponent_value, mantissa_value),
                        (expected, expected),
                    )

            for bits in range(256):
                runtime = E5M2FNUZ(bits)
                if runtime.is_nan:
                    continue
                exponent_value = 1 if runtime.exponent == 0 else runtime.exponent
                mantissa_value = (
                    runtime.mantissa / 4
                    if runtime.exponent == 0
                    else 1 + runtime.mantissa / 4
                )
                self.assertEqual(
                    run(runtime.sign, exponent_value, mantissa_value), (bits, bits)
                )
        finally:
            tempdir.cleanup()

    def test_saturating_encoder_determinism_and_specification_proofs(self):
        def spec(ctx):
            return e5m2fnuz.encode(ctx.two() ** ctx.real_val(24), ctx)

        @Composite(name="e5m2fnuz_encode_saturation_proof", spec=spec)
        def saturation_design():
            return e5m2fnuz_encode(
                Const(UQ(0, 1, 0)),
                Const(Q.from_int(40)),
                Const(UQ.from_int(1)),
            )

        design = saturation_design()
        schedule = [{"tool": "simplify"}, {"tool": "z3", "timeout_ms": 5000}]
        self.assertTrue(design.check_determinism(schedule=schedule)["proved"])
        self.assertTrue(design.check_spec(schedule=schedule)["proved"])


class TestE2M1Spec(unittest.TestCase):
    def test_runtime_layout_values_validation_and_static_type(self):
        self.assertEqual(E2M1.Zero().val, 0x0)
        self.assertEqual(E2M1.nZero().val, 0x8)
        self.assertEqual(E2M1(0x1).to_val(), 0.5)
        self.assertEqual(E2M1(0x2).to_val(), 1.0)
        self.assertEqual(E2M1(0x7).to_val(), 6.0)
        self.assertEqual(E2M1(0xF).to_val(), -6.0)
        value = E2M1.from_fields(1, 3, 1)
        self.assertEqual((value.sign, value.exponent, value.mantissa), (1, 3, 1))
        self.assertEqual(value.copy(), value)
        self.assertEqual(value.static_type(), E2M1T())
        self.assertEqual(value.total_bits(), 4)
        self.assertFalse(hasattr(E2M1, "NaN"))
        self.assertFalse(hasattr(E2M1, "Inf"))
        generator, next_shared = E2M1.random_generator(seed=1, shared_exponent_bits=1)
        self.assertIsInstance(generator(), E2M1)
        self.assertEqual(next_shared() & 0x01, 0)

        for invalid in (-1, 16, 1.5, "0"):
            with self.subTest(packed=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    E2M1(invalid)
        for fields in ((2, 0, 0), (0, -1, 0), (0, 4, 0), (0, 0, -1), (0, 0, 2)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                E2M1.from_fields(*fields)

    def test_exhaustive_classification_and_pack_decode_round_trip(self):
        counts = {"norm": 0, "sub": 0, "zero": 0}
        names = tuple(counts)

        @Composite(name="e2m1_pack_decode_roundtrip", spec=lambda x, ctx: x)
        def roundtrip(x):
            decoded = e2m1_decode(x)
            return e2m1_pack(decoded.sign, decoded.exponent, decoded.mantissa)

        value = Var("e2m1_value", sign=E2M1T())
        design = roundtrip(value)
        tempdir, compiled = jit_compile(design)
        try:
            for bits in range(16):
                runtime = E2M1(bits)
                decoded = e2m1_decode(Const(runtime))
                fields = tuple(node.evaluate().val for node in decoded)
                self.assertEqual(fields[:3], (bits >> 3, (bits >> 1) & 3, bits & 1))
                self.assertEqual(sum(fields[3:]), 1)
                classification = names[fields[3:].index(1)]
                counts[classification] += 1
                spec = runtime.to_spec(SpecContext(f"e2m1-{bits}"))
                self.assertEqual(
                    spec.classification_flags()[classification].constant_fold(),
                    BoolLit(True),
                )
                value.load_val(runtime)
                self.assertEqual(design.evaluate().val, bits)
                self.assertEqual(compiled(bits), bits)
        finally:
            tempdir.cleanup()
        self.assertEqual(counts, {"norm": 12, "sub": 2, "zero": 2})

    def test_symbolic_shape_and_zero_constructors(self):
        ctx = SpecContext("fresh-e2m1")
        value = e2m1.fresh("x", ctx)
        self.assertEqual(tuple(value.classification_flags()), ("norm", "sub", "zero"))
        self.assertEqual(value.observables_for_classification("zero"), (value.sign,))
        self.assertIsInstance(E2M1T().to_spec("input", ctx), e2m1)
        self.assertIsInstance(E2M1T().random_runtime_value(random.Random(1)), E2M1)
        self.assertEqual(e2m1.zero(ctx).is_pzero.constant_fold(), BoolLit(True))
        self.assertEqual(e2m1.nzero(ctx).is_nzero.constant_fold(), BoolLit(True))
        self.assertFalse(hasattr(e2m1, "nan"))
        self.assertFalse(hasattr(e2m1, "inf"))

        for real_value, predicate in (
            (0.0, "is_pzero"),
            (-0.25, "is_nzero"),
            (0.5, "is_sub"),
            (1.0, "is_norm"),
            (6.0, "is_norm"),
            (10.0, "is_norm"),
            (-10.0, "is_norm"),
        ):
            boundary_ctx = SpecContext("encode-e2m1")
            encoded = e2m1.encode(RealLit(real_value), boundary_ctx)
            boundary_ctx.check(getattr(encoded, predicate))
            report = simplify_ctx(boundary_ctx)
            if report["status"] == "unknown":
                report = z3_check_eq(report["new_ctx"], timeout_ms=1000)
            self.assertEqual(report["status"], "unsat", report)

    def test_encoder_rne_canonical_zero_saturation_and_all_encodings(self):
        sign = Var("e2_sign", UQT(1, 0))
        exponent = Var("e2_exponent", QT(8, 0))
        mantissa = Var("e2_mantissa", UQT(2, 12))
        design = e2m1_encode(sign, exponent, mantissa)
        tempdir, compiled = jit_compile(design)

        def run(sign_value, exponent_value, mantissa_value):
            sign_runtime = UQ(sign_value, 1, 0)
            exponent_runtime = Q(exponent_value & 0xFF, 8, 0)
            mantissa_runtime = UQ.from_float(mantissa_value, 2, 12)
            sign.load_val(sign_runtime)
            exponent.load_val(exponent_runtime)
            mantissa.load_val(mantissa_runtime)
            return (
                design.evaluate().val,
                compiled(sign_runtime.val, exponent_runtime.val, mantissa_runtime.val),
            )

        try:
            cases = (
                (0, 1, 0.0, 0x0),
                (1, 1, 0.0, 0x0),
                (0, 1, 0.25, 0x0),
                (1, 1, 0.25, 0x8),
                (0, 1, 0.75, 0x2),
                (1, 1, 0.75, 0xA),
                (0, 1, 1.25, 0x2),
                (0, 1, 1.75, 0x4),
                (0, 3, 1.25, 0x6),
                (0, 3, 1.5, 0x7),
                (1, 5, 1.0, 0xF),
            )
            for sign_value, exponent_value, mantissa_value, expected in cases:
                with self.subTest(
                    sign=sign_value, exponent=exponent_value, mantissa=mantissa_value
                ):
                    self.assertEqual(
                        run(sign_value, exponent_value, mantissa_value),
                        (expected, expected),
                    )

            for bits in range(16):
                runtime = E2M1(bits)
                exponent_value = 1 if runtime.exponent == 0 else runtime.exponent
                mantissa_value = (
                    runtime.mantissa / 2
                    if runtime.exponent == 0
                    else 1 + runtime.mantissa / 2
                )
                expected = 0x0 if bits == E2M1.nZero().val else bits
                self.assertEqual(
                    run(runtime.sign, exponent_value, mantissa_value),
                    (expected, expected),
                )
        finally:
            tempdir.cleanup()

    def test_saturating_encoder_determinism_and_specification_proofs(self):
        def spec(ctx):
            return e2m1.encode(ctx.real_val(16), ctx)

        @Composite(name="e2m1_encode_saturation_proof", spec=spec)
        def saturation_design():
            return e2m1_encode(
                Const(UQ(0, 1, 0)),
                Const(Q.from_int(5)),
                Const(UQ.from_int(1)),
            )

        design = saturation_design()
        schedule = [{"tool": "simplify"}, {"tool": "z3", "timeout_ms": 5000}]
        self.assertTrue(design.check_determinism(schedule=schedule)["proved"])
        self.assertTrue(design.check_spec(schedule=schedule)["proved"])


class TestBFloat16Spec(unittest.TestCase):
    def test_bf16_value_selects_finite_formula_or_fresh_special(self):
        ctx = SpecContext("bf16-finite-value")
        value = bf16.fresh("x", ctx)

        self.assertIsInstance(value.value, If)
        self.assertEqual(value.value.cond, value.is_norm)
        subnormal_case = value.value.on_false
        self.assertIsInstance(subnormal_case, If)
        self.assertEqual(subnormal_case.cond, value.is_sub)
        zero_case = subnormal_case.on_false
        self.assertIsInstance(zero_case, If)
        self.assertEqual(zero_case.cond, value.is_zero)
        self.assertEqual(zero_case.on_true, RealLit(0))
        self.assertIsInstance(zero_case.on_false, RealVar)
        self.assertTrue(zero_case.on_false.name.startswith("special_"))

    def test_bf16_static_inputs_use_structured_spec_values(self):
        ctx = SpecContext("structured-bf16-input")

        value = BFloat16T().to_spec("input", ctx)

        self.assertIsInstance(value, bf16)

    def test_bf16_decoder_delegates_to_structured_decode(self):
        ctx = SpecContext("structured-bf16-decode")

        decoded = bf16_decode(Const(BFloat16.Zero()))

        self.assertEqual(
            tuple(ctx.spec_of(field).constant_fold() for field in decoded),
            tuple(RealLit(value) for value in (0, 0, 0, 0, 0, 1, 0, 0)),
        )
        self.assertIs(decoded.sign, decoded[0])
        self.assertIs(decoded.is_nan, decoded[7])

    def test_bf16_format_and_explicit_non_finite_constructors(self):
        self.assertEqual(bf16.exponent_bits, 8)
        self.assertEqual(bf16.mantissa_bits, 7)
        self.assertEqual(bf16.exponent_bias, 127)

        ctx = SpecContext("explicit-bf16-specials")
        cases = (
            (bf16.nan(ctx), "is_nan"),
            (bf16.inf(ctx), "is_pinf"),
            (bf16.ninf(ctx), "is_ninf"),
            (bf16.zero(ctx), "is_pzero"),
            (bf16.nzero(ctx), "is_nzero"),
        )
        special_names = set()
        for value, predicate in cases:
            with self.subTest(predicate=predicate):
                if predicate in {"is_nan", "is_pinf", "is_ninf"}:
                    self.assertIsInstance(value.value, RealVar)
                    self.assertTrue(value.value.name.startswith("special_"))
                    special_names.add(value.value.name)
                else:
                    self.assertEqual(value.value, RealLit(0))
                self.assertEqual(
                    getattr(value, predicate).constant_fold(),
                    BoolLit(True),
                )
        self.assertEqual(len(special_names), 3)

    def test_bf16_encode_classifies_representative_values(self):
        greatest_normal = (2 - 2 ** -bf16.mantissa_bits) * 2 ** 127
        cases = (
            ("positive-zero", 0.0, "is_pzero"),
            ("negative-underflow", -(2 ** -134), "is_nzero"),
            ("smallest-subnormal", 2 ** -133, "is_sub"),
            ("smallest-normal", 2 ** -126, "is_norm"),
            ("greatest-normal", greatest_normal, "is_norm"),
            ("positive-overflow", 4e38, "is_pinf"),
            ("negative-overflow", -4e38, "is_ninf"),
        )

        for name, real_value, predicate in cases:
            with self.subTest(value=name):
                ctx = SpecContext(f"bf16-encode-{name}")
                encoded = bf16.encode(RealLit(real_value), ctx)
                ctx.check(getattr(encoded, predicate))

                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(report["status"], "unsat", report)

    def test_bf16_known_classification_exposes_only_observable_fields(self):
        value = bf16(
            value=RealVar("value"),
            sign=RealVar("sign"),
            exponent=RealVar("exponent"),
            mantissa=RealVar("mantissa"),
            is_norm=BoolVar("is_norm"),
            is_sub=BoolVar("is_sub"),
            is_zero=BoolVar("is_zero"),
            is_inf=BoolVar("is_inf"),
            is_nan=BoolVar("is_nan"),
        )

        self.assertEqual(value.observables_for_classification("norm"), (value.value,))
        self.assertEqual(value.observables_for_classification("sub"), (value.value,))
        self.assertEqual(value.observables_for_classification("zero"), (value.sign,))
        self.assertEqual(value.observables_for_classification("inf"), (value.sign,))
        self.assertEqual(
            value.observables_for_classification("nan"),
            (BoolLit(True),),
        )
        with self.assertRaisesRegex(ValueError, "Unknown bf16 classification"):
            value.observables_for_classification("finite")

    def test_bf16_encode_requires_real_expression(self):
        with self.assertRaisesRegex(TypeError, "bf16.encode value must be RealExpr"):
            bf16.encode(1.0, SpecContext("bf16-invalid-encode"))

    def test_bf16_encode_constructs_result_without_calling_fresh(self):
        ctx = SpecContext("direct-bf16-encode")

        with patch.object(
            bf16,
            "fresh",
            side_effect=AssertionError("encode must construct bf16 directly"),
        ):
            encoded = bf16.encode(RealLit(1), ctx)

        self.assertIsInstance(encoded, bf16)
        self.assertNotIsInstance(encoded.is_norm, BoolVar)
        self.assertIsInstance(encoded.exponent, RealVar)
        self.assertIsInstance(encoded.mantissa, RealVar)


class TestFloatFormatConversions(unittest.TestCase):
    def test_registry_contains_only_fp32_round_trips(self):
        formats = {
            "bf16",
            "fp16",
            "e5m2",
            "e5m2fnuz",
            "e4m3fn",
            "ue4m3",
            "e2m1",
        }
        expected = {
            name
            for format_name in formats
            for name in (
                f"{format_name}_to_fp32",
                f"fp32_to_{format_name}",
            )
        }

        self.assertEqual(len(CONVERTER_REGISTRY), 14)
        self.assertEqual(set(CONVERTER_REGISTRY), expected)
        self.assertEqual(set(CONVERTER_FORMATS), expected)

        for name, conversion in CONVERTER_REGISTRY.items():
            source_name, target_name = CONVERTER_FORMATS[name]
            with self.subTest(conversion=name):
                source = Var(name="source", sign=FORMAT_STATIC_TYPES[source_name]())
                design = conversion(source)
                self.assertEqual(design.name, name)
                self.assertEqual(design.node_type, FORMAT_STATIC_TYPES[target_name]())

    def test_small_formats_convert_exhaustively_to_fp32(self):
        runtime_types = {
            "e5m2": E5M2,
            "e5m2fnuz": E5M2FNUZ,
            "e4m3fn": E4M3FN,
            "ue4m3": UE4M3,
            "e2m1": E2M1,
        }

        for source_name, runtime_type in runtime_types.items():
            conversion = CONVERTER_REGISTRY[f"{source_name}_to_fp32"]
            source = Var(name="source", sign=FORMAT_STATIC_TYPES[source_name]())
            design = conversion(source)
            tempdir, compiled = jit_compile(design)
            try:
                encoding_count = 128 if runtime_type is UE4M3 else 1 << runtime_type(0).total_bits()
                for source_bits in range(encoding_count):
                    value = runtime_type(source_bits)
                    if math.isnan(value.to_val()):
                        expected = Float32.NaN().val
                    elif math.isinf(value.to_val()):
                        expected = (
                            Float32.nInf().val if value.sign else Float32.Inf().val
                        )
                    else:
                        expected = struct.unpack(
                            ">I",
                            struct.pack(">f", value.to_val()),
                        )[0]

                    source.load_val(value)
                    with self.subTest(
                        conversion=conversion.__name__,
                        source_bits=hex(source_bits),
                    ):
                        self.assertEqual(design.evaluate().val, expected)
                        self.assertEqual(compiled(source_bits), expected)
            finally:
                tempdir.cleanup()

    def _assert_python_and_cpp(self, conversion, runtime_type, cases):
        source = Var(name="source", sign=runtime_type(0).static_type())
        design = conversion(source)
        tempdir, compiled = jit_compile(design)
        try:
            for name, source_bits, expected_bits in cases:
                with self.subTest(conversion=conversion.__name__, name=name):
                    source.load_val(runtime_type(source_bits))
                    self.assertEqual(design.evaluate().val, expected_bits)
                    self.assertEqual(compiled(source_bits), expected_bits)
        finally:
            tempdir.cleanup()

    def test_fp16_to_fp32_is_exact(self):
        self._assert_python_and_cpp(
            fp16_to_fp32,
            Float16,
            (
                ("positive-zero", 0x0000, 0x00000000),
                ("negative-zero", 0x8000, 0x80000000),
                ("smallest-subnormal", 0x0001, 0x33800000),
                ("largest-subnormal", 0x03FF, 0x387FC000),
                ("one", 0x3C00, 0x3F800000),
                ("largest-finite", 0x7BFF, 0x477FE000),
                ("positive-infinity", 0x7C00, 0x7F800000),
                ("negative-infinity", 0xFC00, 0xFF800000),
                ("nan", 0x7E00, 0x7FC00000),
            ),
        )

    def test_bf16_to_fp32_is_exact(self):
        self._assert_python_and_cpp(
            bf16_to_fp32,
            BFloat16,
            (
                ("positive-zero", 0x0000, 0x00000000),
                ("negative-zero", 0x8000, 0x80000000),
                ("smallest-subnormal", 0x0001, 0x00010000),
                ("largest-subnormal", 0x007F, 0x007F0000),
                ("one", 0x3F80, 0x3F800000),
                ("largest-finite", 0x7F7F, 0x7F7F0000),
                ("positive-infinity", 0x7F80, 0x7F800000),
                ("negative-infinity", 0xFF80, 0xFF800000),
                ("nan", 0x7FC0, 0x7FC00000),
            ),
        )

    def test_fp32_to_fp16_rounds_and_classifies(self):
        self._assert_python_and_cpp(
            fp32_to_fp16,
            Float32,
            (
                ("positive-zero", 0x00000000, 0x0000),
                ("negative-zero", 0x80000000, 0x8000),
                ("half-minimum-tie", 0x33000000, 0x0000),
                ("smallest-subnormal", 0x33800000, 0x0001),
                ("smallest-normal", 0x38800000, 0x0400),
                ("tie-to-even-down", 0x3F801000, 0x3C00),
                ("tie-to-even-up", 0x3F803000, 0x3C02),
                ("largest-finite", 0x477FE000, 0x7BFF),
                ("overflow", 0x47800000, 0x7C00),
                ("positive-infinity", 0x7F800000, 0x7C00),
                ("negative-infinity", 0xFF800000, 0xFC00),
                ("nan", 0x7FC00000, 0x7E00),
            ),
        )

    def test_fp32_to_bf16_rounds_to_nearest_even(self):
        self._assert_python_and_cpp(
            fp32_to_bf16,
            Float32,
            (
                ("positive-zero", 0x00000000, 0x0000),
                ("negative-zero", 0x80000000, 0x8000),
                ("tie-to-even-down", 0x3F808000, 0x3F80),
                ("tie-to-even-up", 0x3F818000, 0x3F82),
                ("positive-infinity", 0x7F800000, 0x7F80),
                ("negative-infinity", 0xFF800000, 0xFF80),
                ("nan", 0x7FC00000, 0x7FC0),
            ),
        )

    def test_fp32_to_small_formats_handles_special_values(self):
        cases = (
            (
                fp32_to_e5m2,
                (
                    (0x00000000, 0x00),
                    (0x80000000, 0x80),
                    (0x3F800000, 0x3C),
                    (0x7F800000, 0x7C),
                    (0xFF800000, 0xFC),
                    (0x7FC00000, 0x7E),
                ),
            ),
            (
                fp32_to_e5m2fnuz,
                (
                    (0x00000000, 0x00),
                    (0x80000000, 0x00),
                    (0x3F800000, 0x40),
                    (0x7F800000, 0x7F),
                    (0xFF800000, 0xFF),
                    (0x7FC00000, 0x80),
                ),
            ),
            (
                fp32_to_e4m3fn,
                (
                    (0x00000000, 0x00),
                    (0x80000000, 0x80),
                    (0x3F800000, 0x38),
                    (0x7F800000, 0x7E),
                    (0xFF800000, 0xFE),
                    (0x7FC00000, 0x7F),
                ),
            ),
            (
                fp32_to_e2m1,
                (
                    (0x00000000, 0x0),
                    (0x80000000, 0x8),
                    (0x3E800000, 0x0),
                    (0x3F400000, 0x2),
                    (0x7F800000, 0x7),
                    (0xFF800000, 0xF),
                    (0x7FC00000, 0x7),
                ),
            ),
            (
                fp32_to_ue4m3,
                (
                    (0x00000000, 0x00),
                    (0x80000000, 0x00),
                    (0x3F800000, 0x38),
                    (0xBF800000, 0x38),
                    (0x7F800000, 0x7E),
                    (0xFF800000, 0x7E),
                    (0x7FC00000, 0x7F),
                ),
            ),
        )
        for conversion, raw_cases in cases:
            self._assert_python_and_cpp(
                conversion,
                Float32,
                tuple(
                    (hex(source_bits), source_bits, expected_bits)
                    for source_bits, expected_bits in raw_cases
                ),
            )


class TestBFloat16Add(unittest.TestCase):
    @staticmethod
    def _reference_bits(lhs: BFloat16, rhs: BFloat16) -> int:
        result = lhs.to_val() + rhs.to_val()
        if math.isnan(result):
            return BFloat16.NaN().val
        if math.isinf(result):
            return BFloat16.nInf().val if result < 0 else BFloat16.Inf().val

        try:
            fp32_bits = struct.unpack(">I", struct.pack(">f", result))[0]
        except OverflowError:
            return BFloat16.nInf().val if result < 0 else BFloat16.Inf().val

        rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1)
        return ((fp32_bits + rounding_bias) >> 16) & 0xFFFF

    def _make_design(self):
        lhs = Var(name="lhs", sign=BFloat16T())
        rhs = Var(name="rhs", sign=BFloat16T())
        return lhs, rhs, bf16_add(lhs, rhs)

    def test_bf16_add_specifications_can_be_chained(self):
        lhs = Var(name="lhs", sign=BFloat16T())
        rhs = Var(name="rhs", sign=BFloat16T())
        inner = bf16_add(lhs, rhs)
        outer = bf16_add(inner, lhs)
        ctx = SpecContext("chained-bf16-add")

        output = ctx.spec_of(outer)

        self.assertIsInstance(output, bf16)
        ctx.validate_requirements(timeout_ms=1000)

    def test_bf16_add_handles_rounding_subnormals_and_special_values(self):
        one = BFloat16.from_fields(sign=0, exponent=127, mantissa=0)
        negative_one = BFloat16.from_fields(sign=1, exponent=127, mantissa=0)
        half_ulp_at_one = BFloat16.from_fields(sign=0, exponent=119, mantissa=0)
        smallest_subnormal = BFloat16.from_fields(sign=0, exponent=0, mantissa=1)
        max_finite = BFloat16.from_fields(sign=0, exponent=254, mantissa=127)

        cases = (
            ("one-plus-one", one, one, BFloat16.from_fields(0, 0, 128)),
            ("cancellation", one, negative_one, BFloat16.Zero()),
            ("ties-to-even", one, half_ulp_at_one, one),
            (
                "subnormal-add",
                smallest_subnormal,
                smallest_subnormal,
                BFloat16.from_fields(0, 2, 0),
            ),
            ("overflow", max_finite, max_finite, BFloat16.Inf()),
            ("positive-infinity", BFloat16.Inf(), one, BFloat16.Inf()),
            ("negative-infinity", BFloat16.nInf(), one, BFloat16.nInf()),
            (
                "opposite-infinities",
                BFloat16.Inf(),
                BFloat16.nInf(),
                BFloat16.NaN(),
            ),
            ("nan", BFloat16.NaN(), one, BFloat16.NaN()),
            ("negative-zero", BFloat16.nZero(), BFloat16.nZero(), BFloat16.nZero()),
        )

        lhs, rhs, design = self._make_design()
        for name, lhs_value, rhs_value, expected in cases:
            with self.subTest(case=name):
                lhs.load_val(lhs_value)
                rhs.load_val(rhs_value)
                actual = design.evaluate()
                if math.isnan(expected.to_val()):
                    self.assertTrue(math.isnan(actual.to_val()))
                else:
                    self.assertEqual(actual, expected)

    def test_bf16_add_matches_rne_reference_for_random_finite_inputs(self):
        lhs, rhs, design = self._make_design()
        rng = random.Random(0)

        for _ in range(500):
            lhs_value = BFloat16.from_fields(
                sign=rng.getrandbits(1),
                exponent=rng.randrange(BFloat16.inf_code),
                mantissa=rng.getrandbits(BFloat16.mantissa_bits),
            )
            rhs_value = BFloat16.from_fields(
                sign=rng.getrandbits(1),
                exponent=rng.randrange(BFloat16.inf_code),
                mantissa=rng.getrandbits(BFloat16.mantissa_bits),
            )
            lhs.load_val(lhs_value)
            rhs.load_val(rhs_value)

            with self.subTest(lhs=lhs_value.val, rhs=rhs_value.val):
                self.assertEqual(
                    design.evaluate().val,
                    self._reference_bits(lhs_value, rhs_value),
                )

    def test_bf16_add_jit_matches_python_evaluation(self):
        lhs, rhs, design = self._make_design()
        tempdir, compiled = jit_compile(design)
        rng = random.Random(1)
        try:
            for _ in range(100):
                lhs_bits = rng.getrandbits(16)
                rhs_bits = rng.getrandbits(16)
                lhs.load_val(BFloat16(lhs_bits))
                rhs.load_val(BFloat16(rhs_bits))
                self.assertEqual(
                    compiled(lhs_bits, rhs_bits),
                    design.evaluate().val,
                )
        finally:
            tempdir.cleanup()


class TestBFloat16Mult(unittest.TestCase):
    @staticmethod
    def _reference_bits(lhs: BFloat16, rhs: BFloat16) -> int:
        result = lhs.to_val() * rhs.to_val()
        if math.isnan(result):
            return BFloat16.NaN().val
        if math.isinf(result):
            return BFloat16.nInf().val if result < 0 else BFloat16.Inf().val

        try:
            fp32_bits = struct.unpack(">I", struct.pack(">f", result))[0]
        except OverflowError:
            return BFloat16.nInf().val if result < 0 else BFloat16.Inf().val

        rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1)
        return ((fp32_bits + rounding_bias) >> 16) & 0xFFFF

    def _make_design(self):
        lhs = Var(name="lhs", sign=BFloat16T())
        rhs = Var(name="rhs", sign=BFloat16T())
        return lhs, rhs, bf16_mult(lhs, rhs)

    def test_bf16_mult_specifications_can_be_chained(self):
        lhs = Var(name="lhs", sign=BFloat16T())
        rhs = Var(name="rhs", sign=BFloat16T())
        inner = bf16_mult(lhs, rhs)
        outer = bf16_mult(inner, lhs)
        ctx = SpecContext("chained-bf16-mult")

        output = ctx.spec_of(outer)

        self.assertIsInstance(output, bf16)
        ctx.validate_requirements(timeout_ms=1000)

    def test_bf16_mult_handles_rounding_subnormals_and_special_values(self):
        cases = (
            ("one", 0x3F80, 0x3F80, 0x3F80),
            ("negative", 0xBF80, 0x4000, 0xC000),
            ("rounded-product", 0x3FC0, 0x3FC0, 0x4010),
            ("min-subnormal-identity", 0x0001, 0x3F80, 0x0001),
            ("positive-half-minimum-tie", 0x0001, 0x3F00, 0x0000),
            ("negative-half-minimum-tie", 0x8001, 0x3F00, 0x8000),
            ("subnormal-tie-to-even", 0x0001, 0x3FC0, 0x0002),
            ("min-normal-halved", 0x0080, 0x3F00, 0x0040),
            ("largest-subnormal-doubled", 0x007F, 0x4000, 0x00FE),
            ("overflow", 0x7F7F, 0x4000, 0x7F80),
            ("negative-zero", 0x8000, 0x3F80, 0x8000),
            ("zero-times-infinity", 0x0000, 0x7F80, 0x7FC0),
            ("negative-infinity", 0xFF80, 0x4000, 0xFF80),
            ("nan", 0x7FC0, 0x3F80, 0x7FC0),
        )

        lhs, rhs, design = self._make_design()
        for name, lhs_bits, rhs_bits, expected_bits in cases:
            with self.subTest(case=name):
                lhs.load_val(BFloat16(lhs_bits))
                rhs.load_val(BFloat16(rhs_bits))
                self.assertEqual(design.evaluate().val, expected_bits)

    def test_bf16_mult_matches_rne_reference_for_random_finite_inputs(self):
        lhs, rhs, design = self._make_design()
        rng = random.Random(0)

        for _ in range(500):
            lhs_value = BFloat16.from_fields(
                sign=rng.getrandbits(1),
                exponent=rng.randrange(BFloat16.inf_code),
                mantissa=rng.getrandbits(BFloat16.mantissa_bits),
            )
            rhs_value = BFloat16.from_fields(
                sign=rng.getrandbits(1),
                exponent=rng.randrange(BFloat16.inf_code),
                mantissa=rng.getrandbits(BFloat16.mantissa_bits),
            )
            lhs.load_val(lhs_value)
            rhs.load_val(rhs_value)

            with self.subTest(lhs=lhs_value.val, rhs=rhs_value.val):
                self.assertEqual(
                    design.evaluate().val,
                    self._reference_bits(lhs_value, rhs_value),
                )

    def test_bf16_mult_cpp_lowering_matches_python_evaluation(self):
        lhs, rhs, design = self._make_design()
        tempdir_jit, compiled_jit = jit_compile(design)
        tempdir_no_jit, compiled_no_jit = nonjit_compile(design)
        rng = random.Random(1)
        try:
            for _ in range(100):
                lhs_bits = rng.getrandbits(16)
                rhs_bits = rng.getrandbits(16)
                lhs.load_val(BFloat16(lhs_bits))
                rhs.load_val(BFloat16(rhs_bits))
                expected = design.evaluate().val
                self.assertEqual(compiled_jit(lhs_bits, rhs_bits), expected)
                self.assertEqual(compiled_no_jit(lhs_bits, rhs_bits), expected)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()


class TestReducedWGMMA(unittest.TestCase):
    CONFIGURATIONS = {
        "wgmma_fp32_e4m3_e4m3": (
            wgmma_fp32_e4m3_e4m3,
            [E4M3FN] * 8,
            [E4M3FNT()] * 8 + [Float32T()],
            Float32,
            34,
        ),
        "wgmma_fp32_e5m2_e4m3": (
            wgmma_fp32_e5m2_e4m3,
            [E5M2] * 4 + [E4M3FN] * 4,
            [E5M2T()] * 4 + [E4M3FNT()] * 4 + [Float32T()],
            Float32,
            48,
        ),
        "wgmma_fp16_e4m3_e5m2": (
            wgmma_fp16_e4m3_e5m2,
            [E4M3FN] * 4 + [E5M2] * 4,
            [E4M3FNT()] * 4 + [E5M2T()] * 4 + [Float16T()],
            Float16,
            48,
        ),
    }

    @staticmethod
    def _make_design(function, static_types):
        variables = [
            Var(name=f"x{index}", sign=type_.copy())
            for index, type_ in enumerate(static_types)
        ]
        return variables, function(*variables)

    @staticmethod
    def _finite_fraction(value):
        if isinstance(value, E4M3FN):
            mantissa_bits, bias = 3, 7
        elif isinstance(value, E5M2):
            mantissa_bits, bias = 2, 15
        elif isinstance(value, Float32):
            mantissa_bits, bias = 23, 127
        else:
            mantissa_bits, bias = 10, 15

        if value.exponent == 0:
            significand = value.mantissa
            exponent = 1 - bias - mantissa_bits
        else:
            significand = (1 << mantissa_bits) + value.mantissa
            exponent = value.exponent - bias - mantissa_bits
        scale = (
            Fraction(1 << exponent)
            if exponent >= 0
            else Fraction(1, 1 << -exponent)
        )
        result = significand * scale
        return -result if value.sign else result

    @staticmethod
    def _round_even(value):
        quotient, remainder = divmod(value.numerator, value.denominator)
        if 2 * remainder > value.denominator or (
            2 * remainder == value.denominator and quotient & 1
        ):
            quotient += 1
        return quotient

    @classmethod
    def _encode_fraction(cls, value, destination):
        sign = int(value < 0)
        magnitude = abs(value)
        if magnitude == 0:
            return destination.Zero().val

        mantissa_bits = destination.mantissa_bits
        minimum_quantum_exponent = (
            1 - destination.exponent_bias - mantissa_bits
        )
        if minimum_quantum_exponent < 0:
            subnormal_units = magnitude * (1 << -minimum_quantum_exponent)
        else:
            subnormal_units = magnitude / (1 << minimum_quantum_exponent)
        significand = cls._round_even(subnormal_units)
        if significand < (1 << mantissa_bits):
            return destination.from_fields(sign, 0, significand).val

        exponent = (
            magnitude.numerator.bit_length()
            - magnitude.denominator.bit_length()
        )
        power = (
            Fraction(1 << exponent)
            if exponent >= 0
            else Fraction(1, 1 << -exponent)
        )
        if power > magnitude:
            exponent -= 1
        shift = exponent - mantissa_bits
        scaled = (
            magnitude / (1 << shift)
            if shift >= 0
            else magnitude * (1 << -shift)
        )
        significand = cls._round_even(scaled)
        if significand == (1 << (mantissa_bits + 1)):
            significand >>= 1
            exponent += 1

        biased_exponent = exponent + destination.exponent_bias
        if biased_exponent >= destination.inf_code:
            return destination.nInf().val if sign else destination.Inf().val
        return destination.from_fields(
            sign,
            biased_exponent,
            significand - (1 << mantissa_bits),
        ).val

    @classmethod
    def _finite_reference(cls, values, destination):
        exact = cls._finite_fraction(values[-1])
        exact += sum(
            (
                cls._finite_fraction(values[index])
                * cls._finite_fraction(values[index + 4])
                for index in range(4)
            ),
            Fraction(),
        )
        return cls._encode_fraction(exact, destination)

    @staticmethod
    def _load(variables, values):
        for variable, value in zip(variables, values):
            variable.load_val(value)

    def test_public_registry_signatures_and_topology(self):
        self.assertEqual(set(WGMMA_REGISTRY), set(self.CONFIGURATIONS))
        for name, (function, _runtime_types, static_types, _destination, width) in (
            self.CONFIGURATIONS.items()
        ):
            variables, design = self._make_design(function, static_types)
            with self.subTest(model=name):
                self.assertEqual(
                    [variable.node_type for variable in variables], static_types
                )
                self.assertEqual(design.node_type, static_types[-1])

                visited = set()
                nodes = []

                def visit(node):
                    if node in visited:
                        return
                    visited.add(node)
                    nodes.append(node)
                    for argument in getattr(node, "args", ()):
                        visit(argument)

                visit(design.inner_tree)
                names = [node.name for node in nodes if hasattr(node, "name")]
                self.assertNotIn("CSA_tree4", names)
                self.assertIn("q_add", names)
                self.assertIn("uq_max", names)
                exact_product_resizes = [
                    node
                    for node in nodes
                    if getattr(node, "name", None) == "uq_resize"
                    and isinstance(node.node_type, UQT)
                    and node.node_type.int_bits == 2
                    and node.node_type.frac_bits == width
                ]
                self.assertEqual(len(exact_product_resizes), 4)

    def test_golden_specs_return_destination_types(self):
        configurations = (
            (
                spec_wgmma_fp32_e4m3_e4m3,
                [E4M3FNT()] * 8 + [Float32T()],
                fp32,
            ),
            (
                spec_wgmma_fp32_e5m2_e4m3,
                [E5M2T()] * 4 + [E4M3FNT()] * 4 + [Float32T()],
                fp32,
            ),
            (
                spec_wgmma_fp16_e4m3_e5m2,
                [E4M3FNT()] * 4 + [E5M2T()] * 4 + [Float16T()],
                fp16,
            ),
        )
        for index, (spec, types, expected_type) in enumerate(configurations):
            ctx = SpecContext(f"wgmma-spec-type-{index}")
            values = [
                type_.to_spec(f"x{position}", ctx)
                for position, type_ in enumerate(types)
            ]
            self.assertIsInstance(spec(*values, ctx), expected_type)

    def test_finite_subnormal_cancellation_and_passthrough_cases(self):
        one = {E4M3FN: 0x38, E5M2: 0x3C}
        for name, (function, runtime_types, static_types, destination, _width) in (
            self.CONFIGURATIONS.items()
        ):
            variables, design = self._make_design(function, static_types)
            destination_zero = destination.Zero()
            cases = []
            cases.append([runtime_type(one[runtime_type]) for runtime_type in runtime_types] + [destination_zero])
            cases.append(
                [
                    runtime_type(one[runtime_type] | (0x80 if index == 0 else 0))
                    for index, runtime_type in enumerate(runtime_types)
                ]
                + [destination_zero]
            )
            # Every FP8 lane position participates correctly when subnormal.
            for position in range(8):
                cases.append(
                    [
                        runtime_type(0x01 if index == position else one[runtime_type])
                        for index, runtime_type in enumerate(runtime_types)
                    ]
                    + [destination_zero]
                )
            # Exact product cancellation must leave C unchanged.
            cancel = [runtime_type(one[runtime_type]) for runtime_type in runtime_types]
            cancel[0] = runtime_types[0](one[runtime_types[0]] | 0x80)
            cancel[1] = runtime_types[1](one[runtime_types[1]] | 0x80)
            c_value = destination.from_fields(0, destination.exponent_bias - 3, 1)
            cases.append(cancel + [c_value])

            for values in cases:
                self._load(variables, values)
                with self.subTest(model=name, inputs=[value.val for value in values]):
                    self.assertEqual(
                        design.evaluate().val,
                        self._finite_reference(values, destination),
                    )

    def test_nan_infinity_zero_product_and_overflow(self):
        one4 = E4M3FN(0x38)
        one5 = E5M2(0x3C)

        variables, mixed = self._make_design(
            wgmma_fp32_e5m2_e4m3,
            [E5M2T()] * 4 + [E4M3FNT()] * 4 + [Float32T()],
        )
        cases = (
            ([E5M2.NaN(), one5, one5, one5] + [one4] * 4 + [Float32.Zero()], Float32.NaN()),
            ([E5M2.Inf(), one5, one5, one5] + [E4M3FN.Zero(), one4, one4, one4] + [Float32.Zero()], Float32.NaN()),
            ([E5M2.Inf(), E5M2.nInf(), one5, one5] + [one4] * 4 + [Float32.Zero()], Float32.NaN()),
            ([E5M2.Inf(), one5, one5, one5] + [one4] * 4 + [Float32.nInf()], Float32.NaN()),
            ([E5M2.nInf(), one5, one5, one5] + [one4] * 4 + [Float32.nInf()], Float32.nInf()),
            ([one5] * 4 + [one4] * 4 + [Float32.NaN()], Float32.NaN()),
            ([one5] * 4 + [one4] * 4 + [Float32.Inf()], Float32.Inf()),
        )
        for values, expected in cases:
            self._load(variables, values)
            self.assertEqual(mixed.evaluate().val, expected.val)

        variables, half = self._make_design(
            wgmma_fp16_e4m3_e5m2,
            [E4M3FNT()] * 4 + [E5M2T()] * 4 + [Float16T()],
        )
        overflow_values = [E4M3FN(0x7E)] * 4 + [E5M2(0x7B)] * 4 + [Float16.Zero()]
        self._load(variables, overflow_values)
        self.assertEqual(half.evaluate().val, Float16.Inf().val)

        nan_accumulator_values = [one4] * 4 + [one5] * 4 + [Float16.NaN()]
        self._load(variables, nan_accumulator_values)
        self.assertEqual(half.evaluate().val, Float16.NaN().val)

    def test_accumulation_is_fused_instead_of_staged(self):
        configurations = (
            (
                wgmma_fp16_e4m3_e5m2,
                [E4M3FNT()] * 4 + [E5M2T()] * 4 + [Float16T()],
                [
                    E4M3FN(0x7B),
                    E4M3FN(0x9A),
                    E4M3FN(0x54),
                    E4M3FN(0x69),
                    E5M2(0x3C),
                    E5M2(0xD2),
                    E5M2(0xC9),
                    E5M2(0xAC),
                    Float16(0xAFD0),
                ],
                0x5B39,
                0x5B3A,
            ),
            (
                wgmma_fp32_e5m2_e4m3,
                [E5M2T()] * 4 + [E4M3FNT()] * 4 + [Float32T()],
                [
                    E5M2(0x35),
                    E5M2(0xE5),
                    E5M2(0x88),
                    E5M2(0xC9),
                    E4M3FN(0x79),
                    E4M3FN(0x2A),
                    E4M3FN(0xB6),
                    E4M3FN(0x39),
                    Float32(0xCF39B00F),
                ],
                0xCF39B010,
                0xCF39B011,
            ),
        )
        for function, types, values, fused, staged in configurations:
            variables, design = self._make_design(function, types)
            self._load(variables, values)
            self.assertNotEqual(fused, staged)
            self.assertEqual(design.evaluate().val, fused)

    def test_random_finite_python_jit_and_nonjit_match_exact_reference(self):
        rng = random.Random(314159)
        for name, (function, runtime_types, static_types, destination, _width) in (
            self.CONFIGURATIONS.items()
        ):
            variables, design = self._make_design(function, static_types)
            tempdir_jit, compiled_jit = jit_compile(design)
            tempdir_no_jit, compiled_no_jit = nonjit_compile(design)
            try:
                for _ in range(24):
                    values = []
                    for runtime_type in [*runtime_types, destination]:
                        bits = 32 if runtime_type is Float32 else 16 if runtime_type is Float16 else 8
                        while True:
                            value = runtime_type(rng.getrandbits(bits))
                            exponent_is_special = (
                                hasattr(runtime_type, "inf_code")
                                and value.exponent == runtime_type.inf_code
                            )
                            e4m3_nan = isinstance(value, E4M3FN) and value.is_nan
                            if not exponent_is_special and not e4m3_nan:
                                break
                        values.append(value)
                    expected = self._finite_reference(values, destination)
                    packed = [value.val for value in values]
                    self._load(variables, values)
                    with self.subTest(model=name, inputs=packed):
                        self.assertEqual(design.evaluate().val, expected)
                        self.assertEqual(compiled_jit(*packed), expected)
                        self.assertEqual(compiled_no_jit(*packed), expected)
            finally:
                tempdir_jit.cleanup()
                tempdir_no_jit.cleanup()


class TestUE4M3x2E2M1x2AddFP32(unittest.TestCase):
    @staticmethod
    def _reference_bits(scale0, scale1, x0, x1):
        if scale0.is_nan or scale1.is_nan:
            return Float32.NaN().val

        value = scale0.to_val() * x0.to_val() + scale1.to_val() * x1.to_val()
        return struct.unpack(">I", struct.pack(">f", value))[0]

    @staticmethod
    def _cases():
        return (
            ("ordinary-normal", 0x38, 0x40, 0x2, 0x3, 0x40800000),
            ("ue4m3-subnormal", 0x01, 0x00, 0x1, 0x2, 0x3A800000),
            ("e2m1-subnormal", 0x38, 0x38, 0x1, 0x2, 0x3FC00000),
            ("unequal-scales", 0x7E, 0x01, 0x7, 0x1, 0x45280004),
            ("exact-cancellation", 0x40, 0x40, 0x3, 0xB, 0x00000000),
            ("both-negative-zero", 0x00, 0x38, 0xF, 0x8, 0x80000000),
            ("mixed-sign-zeros", 0x00, 0x38, 0xF, 0x0, 0x00000000),
            ("both-positive-zero", 0x00, 0x38, 0x7, 0x0, 0x00000000),
            ("maximum-positive", 0x7E, 0x7E, 0x7, 0x7, 0x45A80000),
            ("maximum-negative", 0x7E, 0x7E, 0xF, 0xF, 0xC5A80000),
            ("nan0-precedes-negative-zero", 0x7F, 0x00, 0xF, 0x8, 0x7FC00000),
            ("nan1-precedes-cancellation", 0x38, 0x7F, 0x2, 0xA, 0x7FC00000),
        )

    def _make_design(self):
        scale0 = Var(name="scale0", sign=UE4M3T())
        scale1 = Var(name="scale1", sign=UE4M3T())
        x0 = Var(name="x0", sign=E2M1T())
        x1 = Var(name="x1", sign=E2M1T())
        return (scale0, scale1, x0, x1), ue4m3x2_e2m1x2_add_fp32(
            scale0, scale1, x0, x1
        )

    @staticmethod
    def _load(variables, bits):
        variables[0].load_val(UE4M3(bits[0]))
        variables[1].load_val(UE4M3(bits[1]))
        variables[2].load_val(E2M1(bits[2]))
        variables[3].load_val(E2M1(bits[3]))

    def test_golden_spec_returns_fp32(self):
        ctx = SpecContext("ue4m3x2-e2m1x2-add-fp32-spec")
        values = (
            UE4M3T().to_spec("scale0", ctx),
            UE4M3T().to_spec("scale1", ctx),
            E2M1T().to_spec("x0", ctx),
            E2M1T().to_spec("x1", ctx),
        )
        result = spec_ue4m3x2_e2m1x2_add_fp32(*values, ctx)
        self.assertIsInstance(result, fp32)

    def _assert_spec_case_proves(self, target_name):
        _, design = self._make_design()
        split_classification_cases = ast_nodes._split_classification_cases

        def select_target_case(*args, **kwargs):
            cases = split_classification_cases(*args, **kwargs)
            return [next(case for case in cases if case.name == target_name)]

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_target_case,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = design.check_spec()

        self.assertTrue(result["proved"])
        proof_trace = result["case_results"][0]["proof_trace"]
        self.assertEqual(proof_trace[-1]["status"], "unsat")

    def test_default_schedule_proves_normal_and_subnormal_cases(self):
        self._assert_spec_case_proves(
            "ue4m3x2_e2m1x2_add_fp32["
            "arg0=norm,arg1=norm,arg2=norm,arg3=norm,output=norm]"
        )
        self._assert_spec_case_proves(
            "ue4m3x2_e2m1x2_add_fp32["
            "arg0=sub,arg1=norm,arg2=sub,arg3=norm,output=norm]"
        )

    def test_determinism(self):
        _, design = self._make_design()
        target_name = (
            "ue4m3x2_e2m1x2_add_fp32_determinism["
            "arg0=norm,arg1=norm,arg2=norm,arg3=norm,output=norm]"
        )
        split_classification_cases = ast_nodes._split_classification_cases

        def select_target_case(*args, **kwargs):
            cases = split_classification_cases(*args, **kwargs)
            return [next(case for case in cases if case.name == target_name)]

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_target_case,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = design.check_determinism()
        self.assertTrue(result["proved"])

    def test_curated_exact_results_zeros_and_nan_precedence(self):
        variables, design = self._make_design()
        for name, scale0, scale1, x0, x1, expected in self._cases():
            with self.subTest(case=name):
                self._load(variables, (scale0, scale1, x0, x1))
                self.assertEqual(design.evaluate().val, expected)

    def test_random_inputs_match_exact_fused_reference(self):
        variables, design = self._make_design()
        rng = random.Random(20260814)
        for _ in range(500):
            values = (
                UE4M3(rng.randrange(0x80)),
                UE4M3(rng.randrange(0x80)),
                E2M1(rng.randrange(16)),
                E2M1(rng.randrange(16)),
            )
            for variable, value in zip(variables, values):
                variable.load_val(value)
            self.assertEqual(design.evaluate().val, self._reference_bits(*values))

    def test_jit_and_nonjit_cpp_match_reference(self):
        variables, design = self._make_design()
        tempdir_jit, compiled_jit = jit_compile(design)
        tempdir_no_jit, compiled_no_jit = nonjit_compile(design)
        rng = random.Random(20260815)
        inputs = [case[1:5] for case in self._cases()]
        inputs.extend(
            (
                rng.randrange(0x80),
                rng.randrange(0x80),
                rng.randrange(16),
                rng.randrange(16),
            )
            for _ in range(100)
        )
        try:
            for bits in inputs:
                values = (
                    UE4M3(bits[0]),
                    UE4M3(bits[1]),
                    E2M1(bits[2]),
                    E2M1(bits[3]),
                )
                expected = self._reference_bits(*values)
                self._load(variables, bits)
                with self.subTest(inputs=bits):
                    self.assertEqual(design.evaluate().val, expected)
                    self.assertEqual(compiled_jit(*bits), expected)
                    self.assertEqual(compiled_no_jit(*bits), expected)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()


class TestUE4M3x2E2M1x2MultFP32(unittest.TestCase):
    @staticmethod
    def _reference_bits(a0, a1, b0, b1):
        if a0.is_nan or a1.is_nan:
            return Float32.NaN().val

        sign = b0.sign ^ b1.sign
        values = (a0, a1, b0, b1)
        if any(value.exponent == 0 and value.mantissa == 0 for value in values):
            return Float32.nZero().val if sign else Float32.Zero().val

        product = a0.to_val() * a1.to_val() * b0.to_val() * b1.to_val()
        return struct.unpack(">I", struct.pack(">f", product))[0]

    @staticmethod
    def _cases():
        return (
            ("identity", 0x38, 0x38, 0x2, 0x2, 0x3F800000),
            ("ordinary-negative", 0x38, 0x40, 0x3, 0x9, 0xBFC00000),
            ("even-negative-parity", 0x38, 0x40, 0xB, 0x9, 0x3FC00000),
            ("subnormal-and-normal", 0x03, 0x40, 0x1, 0x4, 0x3C400000),
            ("minimum-nonzero", 0x01, 0x01, 0x1, 0x1, 0x35800000),
            ("maximum", 0x7E, 0x7E, 0x7, 0x7, 0x4ADC8000),
            ("a0-zero-negative", 0x00, 0x38, 0xA, 0x2, 0x80000000),
            ("a1-zero-negative", 0x38, 0x00, 0xA, 0x2, 0x80000000),
            ("b0-negative-zero", 0x38, 0x38, 0x8, 0x2, 0x80000000),
            ("b1-negative-zero", 0x38, 0x38, 0x2, 0x8, 0x80000000),
            ("even-negative-zero", 0x00, 0x38, 0xA, 0xA, 0x00000000),
            ("nan-precedes-zero", 0x7F, 0x00, 0x8, 0x2, 0x7FC00000),
        )

    def _make_design(self):
        a0 = Var(name="a0", sign=UE4M3T())
        a1 = Var(name="a1", sign=UE4M3T())
        b0 = Var(name="b0", sign=E2M1T())
        b1 = Var(name="b1", sign=E2M1T())
        return (a0, a1, b0, b1), ue4m3x2_e2m1x2_mult_fp32(
            a0, a1, b0, b1
        )

    @staticmethod
    def _load(variables, bits):
        variables[0].load_val(UE4M3(bits[0]))
        variables[1].load_val(UE4M3(bits[1]))
        variables[2].load_val(E2M1(bits[2]))
        variables[3].load_val(E2M1(bits[3]))

    def test_golden_spec_returns_fp32(self):
        ctx = SpecContext("ue4m3x2-e2m1x2-mult-fp32-spec")
        values = (
            UE4M3T().to_spec("a0", ctx),
            UE4M3T().to_spec("a1", ctx),
            E2M1T().to_spec("b0", ctx),
            E2M1T().to_spec("b1", ctx),
        )
        result = spec_ue4m3x2_e2m1x2_mult_fp32(*values, ctx)
        self.assertIsInstance(result, fp32)

    def _assert_spec_case_proves(self, target_name):
        _, design = self._make_design()
        split_classification_cases = ast_nodes._split_classification_cases

        def select_target_case(*args, **kwargs):
            cases = split_classification_cases(*args, **kwargs)
            return [next(case for case in cases if case.name == target_name)]

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_target_case,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = design.check_spec()

        self.assertTrue(result["proved"])
        self.assertEqual(result["case_results"][0]["proof_trace"][-1]["status"], "unsat")

    def test_default_schedule_proves_normal_and_ue4m3_subnormal_cases(self):
        self._assert_spec_case_proves(
            "ue4m3x2_e2m1x2_mult_fp32["
            "arg0=norm,arg1=norm,arg2=norm,arg3=norm,output=norm]"
        )
        self._assert_spec_case_proves(
            "ue4m3x2_e2m1x2_mult_fp32["
            "arg0=norm,arg1=sub,arg2=norm,arg3=norm,output=norm]"
        )

    def test_determinism(self):
        _, design = self._make_design()
        target_name = (
            "ue4m3x2_e2m1x2_mult_fp32_determinism["
            "arg0=norm,arg1=norm,arg2=norm,arg3=norm,output=norm]"
        )
        split_classification_cases = ast_nodes._split_classification_cases

        def select_target_case(*args, **kwargs):
            cases = split_classification_cases(*args, **kwargs)
            return [next(case for case in cases if case.name == target_name)]

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_target_case,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = design.check_determinism()
        self.assertTrue(result["proved"])

    def test_exact_boundaries_signs_zeros_and_nan_precedence(self):
        variables, design = self._make_design()
        for name, a0, a1, b0, b1, expected in self._cases():
            with self.subTest(case=name):
                self._load(variables, (a0, a1, b0, b1))
                self.assertEqual(design.evaluate().val, expected)

    def test_random_finite_inputs_match_host_fp32_bits(self):
        variables, design = self._make_design()
        finite_ue4m3 = list(range(0x7F))
        rng = random.Random(2)
        for _ in range(500):
            values = (
                UE4M3(rng.choice(finite_ue4m3)),
                UE4M3(rng.choice(finite_ue4m3)),
                E2M1(rng.randrange(16)),
                E2M1(rng.randrange(16)),
            )
            for variable, value in zip(variables, values):
                variable.load_val(value)
            self.assertEqual(design.evaluate().val, self._reference_bits(*values))

    def test_jit_and_nonjit_cpp_match_reference(self):
        variables, design = self._make_design()
        tempdir_jit, compiled_jit = jit_compile(design)
        tempdir_no_jit, compiled_no_jit = nonjit_compile(design)
        rng = random.Random(3)
        inputs = [case[1:5] for case in self._cases()]
        inputs.extend(
            (
                rng.randrange(0x7F),
                rng.randrange(0x7F),
                rng.randrange(16),
                rng.randrange(16),
            )
            for _ in range(100)
        )
        try:
            for bits in inputs:
                values = (
                    UE4M3(bits[0]),
                    UE4M3(bits[1]),
                    E2M1(bits[2]),
                    E2M1(bits[3]),
                )
                expected = self._reference_bits(*values)
                self._load(variables, bits)
                with self.subTest(inputs=bits):
                    self.assertEqual(design.evaluate().val, expected)
                    self.assertEqual(compiled_jit(*bits), expected)
                    self.assertEqual(compiled_no_jit(*bits), expected)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()


class TestBFloat16ReLU(unittest.TestCase):
    @staticmethod
    def _reference_bits(value: BFloat16) -> int:
        if value.exponent == BFloat16.nan_code and value.mantissa != 0:
            return BFloat16.NaN().val
        if value.sign:
            return BFloat16.Zero().val
        return value.val

    def _make_design(self):
        value = Var(name="value", sign=BFloat16T())
        return value, bf16_relu(value)

    def test_bf16_relu_specifications_can_be_chained(self):
        value = Var(name="value", sign=BFloat16T())
        inner = bf16_relu(value)
        outer = bf16_relu(inner)
        ctx = SpecContext("chained-bf16-relu")

        output = ctx.spec_of(outer)

        self.assertIsInstance(output, bf16)
        ctx.validate_requirements(timeout_ms=1000)

    def test_bf16_relu_handles_finite_and_special_values(self):
        cases = (
            ("positive-normal", 0x3FC0, 0x3FC0),
            ("negative-normal", 0xBFC0, 0x0000),
            ("positive-subnormal", 0x0001, 0x0001),
            ("negative-subnormal", 0x8001, 0x0000),
            ("positive-zero", 0x0000, 0x0000),
            ("negative-zero", 0x8000, 0x0000),
            ("positive-infinity", 0x7F80, 0x7F80),
            ("negative-infinity", 0xFF80, 0x0000),
            ("positive-nan", 0x7FC1, BFloat16.NaN().val),
            ("negative-nan", 0xFFC1, BFloat16.NaN().val),
        )

        value, design = self._make_design()
        for name, input_bits, expected_bits in cases:
            with self.subTest(case=name):
                value.load_val(BFloat16(input_bits))
                self.assertEqual(design.evaluate().val, expected_bits)

    def test_bf16_relu_cpp_lowering_matches_reference_for_all_inputs(self):
        value, design = self._make_design()
        tempdir_jit, compiled_jit = jit_compile(design)
        tempdir_no_jit, compiled_no_jit = nonjit_compile(design)
        try:
            for input_bits in range(1 << 16):
                expected = self._reference_bits(BFloat16(input_bits))
                self.assertEqual(compiled_jit(input_bits), expected)
                self.assertEqual(compiled_no_jit(input_bits), expected)
        finally:
            tempdir_jit.cleanup()
            tempdir_no_jit.cleanup()


class TestRivalTranslation(unittest.TestCase):
    def test_real_expression_translates_to_rival_ir(self):
        x = RealVar("x")
        y = RealVar("y")
        expr = If(x < y, abs(x - RealLit(1)), RealLit(2) ** y)

        self.assertEqual(
            to_rival_ir(expr),
            {
                "op": "if",
                "cond": {
                    "op": "lt",
                    "lhs": {"op": "var", "name": "x"},
                    "rhs": {"op": "var", "name": "y"},
                },
                "on_true": {
                    "op": "abs",
                    "arg": {
                        "op": "sub",
                        "lhs": {"op": "var", "name": "x"},
                        "rhs": {"op": "real_lit", "num": "1", "den": "1"},
                    },
                },
                "on_false": {
                    "op": "pow",
                    "lhs": {"op": "real_lit", "num": "2", "den": "1"},
                    "rhs": {"op": "var", "name": "y"},
                },
            },
        )

    def test_bool_expression_translates_to_rival_ir(self):
        p = BoolVar("p")
        q = BoolVar("q")

        self.assertEqual(
            to_rival_ir((p & ~q).eq(BoolLit(True))),
            {
                "op": "bool_eq",
                "lhs": {
                    "op": "and",
                    "lhs": {"op": "var", "name": "p"},
                    "rhs": {"op": "not", "arg": {"op": "var", "name": "q"}},
                },
                "rhs": {"op": "bool_lit", "value": True},
            },
        )

    def test_float_literal_translates_exact_fraction(self):
        self.assertEqual(
            to_rival_ir(RealLit(0.5)),
            {"op": "real_lit", "num": "1", "den": "2"},
        )

    def test_collect_free_vars_is_sorted(self):
        exprs = [RealVar("z") + RealVar("a"), BoolVar("flag")]

        self.assertEqual(collect_free_vars(exprs), ["a", "flag", "z"])

    def test_build_machine_combines_exprs_into_single_asserted_and(self):
        x = RealVar("x")
        y = RealVar("y")
        raw_machine = object()
        native = Mock()
        native.build_machine.return_value = raw_machine

        with patch("zolotone.rival._load_native_module", return_value=native):
            machine = build_machine(
                [
                    x >= RealLit(0),
                    y <= RealLit(1),
                    x.eq(y),
                ],
                ["x", "y"],
            )

        self.assertIs(machine._raw_machine, raw_machine)
        native.build_machine.assert_called_once_with(
            [
                {
                    "op": "assert",
                    "arg": {
                        "op": "and",
                        "lhs": {
                            "op": "and",
                            "lhs": {
                                "op": "ge",
                                "lhs": {"op": "var", "name": "x"},
                                "rhs": {"op": "real_lit", "num": "0", "den": "1"},
                            },
                            "rhs": {
                                "op": "le",
                                "lhs": {"op": "var", "name": "y"},
                                "rhs": {"op": "real_lit", "num": "1", "den": "1"},
                            },
                        },
                        "rhs": {
                            "op": "eq",
                            "lhs": {"op": "var", "name": "x"},
                            "rhs": {"op": "var", "name": "y"},
                        },
                    },
                }
            ],
            ["x", "y"],
        )

    def test_rival_rects_default_to_unbounded(self):
        ctx = SpecContext("rival-rects-default")

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x", "y"]),
            [[(-math.inf, math.inf), (-math.inf, math.inf)]],
        )

    def test_rival_rects_use_boolean_point_domains(self):
        true_ctx = SpecContext("rival-rects-bool-true")
        true_predicate = true_ctx.bool("predicate")
        true_ctx.assume(true_predicate)

        false_ctx = SpecContext("rival-rects-bool-false")
        false_predicate = false_ctx.bool("predicate")
        false_ctx.assume(~false_predicate)

        self.assertEqual(
            get_rival_rects([], ["predicate"], ["predicate"]),
            [[(0.0, 1.0)]],
        )
        self.assertEqual(
            get_rival_rects(true_ctx.assumes, ["predicate"]),
            [[(1.0, 1.0)]],
        )
        self.assertEqual(
            get_rival_rects(false_ctx.assumes, ["predicate"]),
            [[(0.0, 0.0)]],
        )

    def test_rival_rects_enumerate_boolean_equalities(self):
        ctx = SpecContext("rival-rects-bool-equality")
        p = ctx.bool("p")
        q = ctx.bool("q")
        ctx.assume(p.eq(q))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["p", "q"]),
            [
                [(0.0, 0.0), (0.0, 0.0)],
                [(1.0, 1.0), (1.0, 1.0)],
            ],
        )

    def test_rival_rects_combine_boolean_and_real_bounds(self):
        ctx = SpecContext("rival-rects-bool-real")
        predicate = ctx.bool("predicate")
        x = ctx.real("x")
        ctx.assume(predicate & (x >= ctx.zero()))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["predicate", "x"]),
            [[(1.0, 1.0), (0.0, math.inf)]],
        )

    def test_rival_rects_extract_closed_bounds(self):
        ctx = SpecContext("rival-rects-closed")
        x = ctx.real("x")

        ctx.assume((x >= ctx.one()) & (x <= ctx.real_val(254)))

        self.assertEqual(get_rival_rects(ctx.assumes, ["x"]), [[(1.0, 254.0)]])

    def test_rival_rects_extract_bounds_with_literal_on_lhs(self):
        ctx = SpecContext("rival-rects-reversed")
        x = ctx.real("x")

        ctx.assume((ctx.one() <= x) & (ctx.real_val(254) >= x))

        self.assertEqual(get_rival_rects(ctx.assumes, ["x"]), [[(1.0, 254.0)]])

    def test_rival_rects_conservatively_close_strict_bounds(self):
        ctx = SpecContext("rival-rects-strict")
        x = ctx.real("x")

        ctx.assume((x > ctx.one()) & (x < ctx.real_val(254)))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x"]),
            [[(1.0, 254.0)]],
        )

    def test_rival_rects_conservatively_close_reversed_strict_bounds(self):
        ctx = SpecContext("rival-rects-reversed-strict")
        x = ctx.real("x")

        ctx.assume((ctx.one() < x) & (ctx.real_val(254) > x))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x"]),
            [[(1.0, 254.0)]],
        )

    def test_rival_rects_enclose_nonrepresentable_literals(self):
        ctx = SpecContext("rival-rects-rounded-literal")
        x = ctx.real("x")
        exact = (1 << 53) + 1

        ctx.assume(x.eq(ctx.real_val(exact)))

        rounded_down = float(1 << 53)
        rounded_up = math.nextafter(rounded_down, math.inf)
        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x"]),
            [[(rounded_down, rounded_up)]],
        )

    def test_rival_rects_extract_point_disjunction(self):
        ctx = SpecContext("rival-rects-point-or")
        sign = ctx.real("sign")

        ctx.assume(sign.eq(ctx.zero()) | ctx.one().eq(sign))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["sign"]),
            [[(0.0, 0.0)], [(1.0, 1.0)]],
        )

    def test_rival_rects_cartesian_product_for_independent_disjunctions(self):
        ctx = SpecContext("rival-rects-cartesian-or")
        sign = ctx.real("sign")
        exponent = ctx.real("exponent")

        ctx.assume(sign.eq(ctx.zero()) | sign.eq(ctx.one()))
        ctx.assume(exponent.eq(ctx.zero()) | exponent.eq(ctx.real_val(255)))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["sign", "exponent"]),
            [
                [(0.0, 0.0), (0.0, 0.0)],
                [(0.0, 0.0), (255.0, 255.0)],
                [(1.0, 1.0), (0.0, 0.0)],
                [(1.0, 1.0), (255.0, 255.0)],
            ],
        )

    def test_rival_rects_preserve_free_var_order(self):
        ctx = SpecContext("rival-rects-order")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume(x >= ctx.one())
        ctx.assume(y <= ctx.two())

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["y", "x", "z"]),
            [[(-math.inf, 2.0), (1.0, math.inf), (-math.inf, math.inf)]],
        )

    def test_rival_rects_drop_conflicting_bounds(self):
        ctx = SpecContext("rival-rects-conflict")
        x = ctx.real("x")

        ctx.assume(x >= ctx.two())
        ctx.assume(x <= ctx.one())

        self.assertEqual(get_rival_rects(ctx.assumes, ["x"]), [])

    def test_rival_rects_ignore_unsupported_assumptions(self):
        ctx = SpecContext("rival-rects-unsupported")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume((x + y).eq(ctx.one()))
        ctx.assume(x >= ctx.zero())

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x", "y"]),
            [[(0.0, math.inf), (-math.inf, math.inf)]],
        )

    def test_rival_rects_ignore_or_when_any_branch_is_unsupported(self):
        ctx = SpecContext("rival-rects-unsupported-or")
        x = ctx.real("x")
        y = ctx.real("y")

        ctx.assume((x >= ctx.zero()) | (x + y).eq(ctx.one()))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["x", "y"]),
            [[(-math.inf, math.inf), (-math.inf, math.inf)]],
        )

    def test_rival_rects_expand_independent_disjunctions_without_coalescing(self):
        ctx = SpecContext("rival-rects-no-coalesce")
        sign = ctx.real("sign")
        exponent = ctx.real("exponent")

        ctx.assume(sign.eq(ctx.zero()) | sign.eq(ctx.one()))
        ctx.assume(exponent.eq(ctx.zero()) | exponent.eq(ctx.real_val(255)))

        self.assertEqual(
            get_rival_rects(ctx.assumes, ["sign", "exponent"]),
            [
                [(0.0, 0.0), (0.0, 0.0)],
                [(0.0, 0.0), (255.0, 255.0)],
                [(1.0, 1.0), (0.0, 0.0)],
                [(1.0, 1.0), (255.0, 255.0)],
            ],
        )

    def test_rival_feasibility_splits_only_variables_from_maybe_exprs(self):
        ctx = SpecContext("rival-guided-split")
        x = ctx.real("x")
        y = ctx.real("y")
        z = ctx.real("z")
        good_x = x >= ctx.zero()
        good_z = z >= ctx.zero()
        maybe = x.eq(y)
        ctx.assume(good_x)
        ctx.assume(good_z)
        ctx.check(maybe)

        root = [(0.0, math.inf), (-math.inf, math.inf), (0.0, math.inf)]
        x_left = (0.0, sys.float_info.max / 2.0)
        x_right = (math.nextafter(sys.float_info.max / 2.0, math.inf), math.inf)
        y_left = (-math.inf, 0.0)
        y_right = (math.nextafter(0.0, math.inf), math.inf)
        combined_calls = []

        def build(exprs, free_vars):
            self.assertEqual(free_vars, ["x", "y", "z"])
            machine = Mock()
            if exprs == ctx.assumes + ctx.checks:
                def apply(rect, hints=None):
                    combined_calls.append((rect, hints))
                    if rect == root:
                        return RivalAnalysis(
                            status=(False, True),
                            hints="root-hints",
                        )
                    return RivalAnalysis(status=(True, True), hints=None)

                machine.apply_with_hints.side_effect = apply
            elif exprs == [maybe]:
                machine.apply_with_hints.return_value = RivalAnalysis(
                    status=(False, True),
                    hints=None,
                )
            else:
                machine.apply_with_hints.return_value = RivalAnalysis(
                    status=(False, False),
                    hints=None,
                )
            return machine

        with patch("zolotone.rival.build_machine", side_effect=build):
            status = rival_feasibility_check(ctx, max_depth=1, checks=True)

        self.assertEqual(status, "not feasible")
        self.assertEqual(
            combined_calls,
            [
                (root, None),
                ([x_left, y_left, (0.0, math.inf)], "root-hints"),
                ([x_left, y_right, (0.0, math.inf)], "root-hints"),
                ([x_right, y_left, (0.0, math.inf)], "root-hints"),
                ([x_right, y_right, (0.0, math.inf)], "root-hints"),
            ],
        )

    def test_rival_feasibility_returns_first_clean_rect(self):
        ctx = SpecContext("rival-feasible-rect")
        x = ctx.real("x")
        ctx.assume(x.eq(ctx.zero()) | x.eq(ctx.one()))
        ctx.check(x.eq(ctx.zero()))

        clean_rect = [(0.0, 0.0)]
        bad_rect = [(1.0, 1.0)]
        combined_calls = []

        def build(exprs, free_vars):
            self.assertEqual(free_vars, ["x"])
            machine = Mock()
            if exprs == ctx.assumes + ctx.checks:
                def apply(rect, hints=None):
                    combined_calls.append((rect, hints))
                    if rect == clean_rect:
                        return RivalAnalysis(
                            status=(False, False),
                            hints=None,
                        )
                    if rect == bad_rect:
                        return RivalAnalysis(
                            status=(True, True),
                            hints=None,
                        )
                    raise AssertionError(f"unexpected rect {rect}")

                machine.apply_with_hints.side_effect = apply
            else:
                machine.apply_with_hints.return_value = RivalAnalysis(
                    status=(False, False),
                    hints=None,
                )
            return machine

        with (
            patch("zolotone.rival.get_rival_rects", return_value=[clean_rect, bad_rect]),
            patch("zolotone.rival.build_machine", side_effect=build),
        ):
            status = rival_feasibility_check(ctx, max_depth=1, checks=True)

        self.assertEqual(status, "feasible")
        self.assertEqual(combined_calls, [(clean_rect, None)])

    def test_rival_trim_context_preserves_rect_assumptions_for_checks(self):
        ctx = SpecContext("rival-trim-assumption-rects")
        x = ctx.real("x")
        bounded = x >= ctx.zero()
        ctx.assume(bounded)
        ctx.check(bounded)

        assumption_rect = [(0.0, math.inf)]
        seen_rects = []

        def build(exprs, free_vars):
            self.assertEqual(free_vars, ["x"])
            self.assertEqual(exprs, [bounded])
            machine = Mock()

            def apply(rect, hints=None):
                self.assertIsNone(hints)
                seen_rects.append(rect)
                return RivalAnalysis(
                    status=(False, False)
                    if rect == assumption_rect
                    else (False, True),
                    hints=None,
                )

            machine.apply_with_hints.side_effect = apply
            return machine

        with patch("zolotone.rival.build_machine", side_effect=build):
            trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.assumes, [bounded])
        self.assertEqual(trimmed.checks, [])
        self.assertEqual(seen_rects, [assumption_rect])

    def test_rival_trim_context_only_rewrites_non_rect_assumptions(self):
        ctx = SpecContext("rival-trim-preserve-rect-assumption")
        x = ctx.real("x")
        zero = ctx.zero()
        redundant = abs(x).eq(x)
        contributing = (x >= zero) & redundant
        ctx.assume(contributing)
        ctx.assume(redundant)

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.assumes, [contributing])

    def test_rival_trim_context_keeps_maybe_exprs(self):
        ctx = SpecContext("rival-trim-maybe")
        x = ctx.real("x")
        y = ctx.real("y")
        assume = x >= ctx.zero()
        check = x.eq(y)
        ctx.assume(assume)
        ctx.check(check)

        def build(exprs, free_vars):
            self.assertEqual(free_vars, ["x", "y"])
            machine = Mock()
            machine.apply_with_hints.return_value = RivalAnalysis(
                status=(False, True),
                hints=None,
            )
            return machine

        with patch("zolotone.rival.build_machine", side_effect=build):
            trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.assumes, [assume])
        self.assertEqual(trimmed.checks, [check])

    def test_rival_trim_context_keeps_unconstrained_bool_check(self):
        ctx = SpecContext("rival-trim-bool-check")
        predicate = ctx.bool("predicate")
        ctx.check(predicate)

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.checks, [predicate])

    def test_rival_trim_context_uses_boolean_assumption_rect(self):
        ctx = SpecContext("rival-trim-bool-assumption")
        predicate = ctx.bool("predicate")
        ctx.assume(predicate)
        ctx.check(predicate)

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.assumes, [predicate])
        self.assertEqual(trimmed.checks, [])

    def test_rival_trim_context_proves_boolean_tautology(self):
        ctx = SpecContext("rival-trim-bool-tautology")
        predicate = ctx.bool("predicate")
        ctx.check(predicate | ~predicate)

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.checks, [])

    def test_rival_trim_context_keeps_if_with_unconstrained_bool_condition(self):
        ctx = SpecContext("rival-trim-bool-if")
        predicate = ctx.bool("predicate")
        one = ctx.one()
        check = If(predicate, one, ctx.two()).eq(one)
        ctx.check(check)

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.checks, [check])

    def test_rival_trim_context_does_not_treat_undefined_predicate_as_false(self):
        ctx = SpecContext("rival-trim-undefined-predicate")
        x = ctx.real("x")
        zero = ctx.zero()
        one = ctx.one()
        two = ctx.two()
        undefined = (x ** ctx.real_val(-1)) > zero
        checks = [
            ~undefined,
            If(undefined, one, two).eq(two),
            abs(x ** ctx.real_val(-1)).eq(one),
            (x ** ctx.real_val(-1)).max(one).eq(one),
            (x ** ctx.real_val(-1)).min(one).eq(one),
        ]
        ctx.assume(x.eq(zero))
        for check in checks:
            ctx.check(check)

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.checks, checks)
        self.assertEqual(
            rival_feasibility_check(trimmed, max_depth=0, checks=True),
            "not feasible",
        )
        self.assertEqual(simplify_ctx(ctx)["status"], "sat")

    def test_rival_trim_context_requires_every_assumption_rect(self):
        ctx = SpecContext("rival-trim-all-rects")
        sign = ctx.real("sign")
        bit_domain = sign.eq(ctx.zero()) | sign.eq(ctx.one())
        check = sign.eq(ctx.zero())
        ctx.assume(bit_domain)
        ctx.check(check)

        def build(exprs, free_vars):
            self.assertEqual(free_vars, ["sign"])
            machine = Mock()
            if exprs == [bit_domain]:
                machine.apply_with_hints.return_value = RivalAnalysis(
                    status=(False, True),
                    hints=None,
                )
            else:
                self.assertEqual(exprs, [check])

                def apply(rect, hints=None):
                    self.assertIsNone(hints)
                    return RivalAnalysis(
                        status=(False, False)
                        if rect == [(0.0, 0.0)]
                        else (True, True),
                        hints=None,
                    )

                machine.apply_with_hints.side_effect = apply
            return machine

        with patch("zolotone.rival.build_machine", side_effect=build):
            trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.assumes, [bit_domain])
        self.assertEqual(trimmed.checks, [check])

    def test_rival_trim_context_rewrites_extrema_from_assumption_bounds(self):
        ctx = SpecContext("rival-trim-bounded-extrema")
        x = ctx.real("x")
        one = ctx.one()
        ctx.assume(x >= one)
        ctx.check(x.max(one).eq(x))
        ctx.check(one.max(x).eq(x))
        ctx.check(x.min(one).eq(one))
        ctx.check(one.min(x).eq(one))

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.assumes, [x >= one])
        self.assertEqual(trimmed.checks, [])

    def test_rival_trim_context_requires_bound_in_every_disjunct(self):
        ctx = SpecContext("rival-trim-disjunctive-bounds")
        x = ctx.real("x")
        one = ctx.one()
        ctx.assume((x >= one) | (x <= -one))
        unresolved = x.max(one).eq(x)
        ctx.check(unresolved)

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.checks, [unresolved])

    def test_rival_trim_context_uses_bound_shared_by_every_disjunct(self):
        ctx = SpecContext("rival-trim-shared-disjunctive-bound")
        x = ctx.real("x")
        one = ctx.one()
        two = ctx.two()
        ctx.assume((x >= one) | (x >= two))
        ctx.check(x.max(one).eq(x))

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.checks, [])

    def test_rival_trim_context_folds_comparisons_from_assumption_bounds(self):
        ctx = SpecContext("rival-trim-bounded-comparisons")
        x = ctx.real("x")
        y = ctx.real("y")
        zero = ctx.zero()
        one = ctx.one()
        five = ctx.real_val(5)
        eight = ctx.real_val(8)
        ctx.assume((x >= one) & (x <= five))
        ctx.assume(y >= one)
        ctx.check(x >= zero)
        ctx.check(x <= five)
        ctx.check(x > zero)
        ctx.check(x.eq(eight))
        ctx.check(x.ne(eight))
        ctx.check((x + y) >= (one + one))
        ctx.check(x < zero)

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.checks, [BoolLit(False), BoolLit(False)])

    def test_rival_trim_context_does_not_discharge_bound_assumption_itself(self):
        ctx = SpecContext("rival-trim-retained-bound")
        x = ctx.real("x")
        one = ctx.one()
        bound = x >= one
        ctx.assume(bound)

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.assumes, [bound])

    def test_rival_trim_context_simplifies_abs_from_known_sign(self):
        nonnegative_ctx = SpecContext("rival-trim-nonnegative-abs")
        x = nonnegative_ctx.real("x")
        zero = nonnegative_ctx.zero()
        nonnegative_ctx.assume(x >= zero)
        nonnegative_ctx.check(abs(x).eq(x))

        nonpositive_ctx = SpecContext("rival-trim-nonpositive-abs")
        y = nonpositive_ctx.real("y")
        nonpositive_ctx.assume(y <= zero)
        nonpositive_ctx.check(abs(y).eq(-y))

        self.assertEqual(rival_trim_context(nonnegative_ctx).checks, [])
        self.assertEqual(rival_trim_context(nonpositive_ctx).checks, [])

    def test_rival_trim_context_selects_if_branch_from_assumptions(self):
        ctx = SpecContext("rival-trim-bounded-if")
        x = ctx.real("x")
        zero = ctx.zero()
        one = ctx.one()
        two = ctx.two()
        ctx.assume(x >= zero)
        ctx.check(If(x >= zero, one, two).eq(one))

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.checks, [])

    def test_rival_trim_context_keeps_abs_and_if_across_mixed_disjuncts(self):
        ctx = SpecContext("rival-trim-mixed-sign")
        x = ctx.real("x")
        one = ctx.one()
        two = ctx.two()
        sign_domain = (x >= one) | (x <= -one)
        abs_check = abs(x).eq(x)
        if_check = If(x >= one, one, two).eq(one)
        ctx.assume(sign_domain)
        ctx.check(abs_check)
        ctx.check(if_check)

        trimmed = rival_trim_context(ctx)

        self.assertEqual(trimmed.checks, [abs_check, if_check])

class TestStdoutVerificationObserver(unittest.TestCase):
    def test_default_schedule_restarts_rewrite_pipeline_three_times(self):
        schedule = ast_nodes._default_equivalence_schedule()

        self.assertEqual(
            [step["tool"] for step in schedule],
            [
                "simplify",
                "egglog-rewrite",
                "simplify",
                "egglog-rewrite",
                "simplify",
                "egglog-rewrite",
                "z3",
            ],
        )
        for rewrite_step in schedule[1:6:2]:
            self.assertEqual(rewrite_step["iterations"], 6)
            self.assertEqual(
                rewrite_step["scheduler"],
                {"match_limit": 500_000, "ban_length": 1},
            )

    def test_prints_completed_case_with_decisive_tool(self):
        ctx = SpecContext("demo[arg0=norm,output=norm]")
        simplified = ctx.copy()
        simplify_report = build_proof_report(
            ctx,
            simplified,
            tool="simplify",
            runtime_s=0.25,
            status="unknown",
            feasibility_status="feasible",
        )
        z3_report = build_proof_report(
            simplified,
            simplified,
            tool="z3",
            runtime_s=0.5,
            status="unsat",
        )
        result = CaseVerificationResult(
            name=ctx.name,
            proved=True,
            status="unsat",
            feasibility_status="feasible",
            proof_trace=[simplify_report, z3_report],
            side_feasibility_reports=[],
        )
        output = io.StringIO()

        ast_nodes.StdoutVerificationObserver(output).case_completed(result)

        rendered = output.getvalue()
        self.assertIn("Verification cases:\n", rendered)
        self.assertIn(
            "PROVED  STATUS   FEASIBILITY     TOOL                          TIME  CASE",
            rendered,
        )
        self.assertIn(
            "yes     unsat    feasible        z3                          0.750s  "
            "demo[arg0=norm,output=norm]",
            rendered,
        )

    def test_check_equivalence_uses_stdout_observer_by_default(self):
        ctx = SpecContext("default-observer")
        x = ctx.real("x")
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = ast_nodes.check_equivalence(
                ast_nodes._Spec("first", lambda _ctx: x),
                ast_nodes._Spec("second", lambda _ctx: x),
                base_ctx=ctx,
                inputs=[],
                schedule=[{"tool": "simplify"}],
            )

        self.assertTrue(result["proved"])
        self.assertIn(
            "yes     unsat    feasible        simplify",
            output.getvalue(),
        )
        self.assertIn("default-observer", output.getvalue())

    def test_explicit_observer_replaces_stdout_default(self):
        class CollectingObserver:
            def __init__(self):
                self.results = []

            def case_completed(self, result):
                self.results.append(result)

        ctx = SpecContext("explicit-observer")
        x = ctx.real("x")
        observer = CollectingObserver()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            ast_nodes.check_equivalence(
                ast_nodes._Spec("first", lambda _ctx: x),
                ast_nodes._Spec("second", lambda _ctx: x),
                base_ctx=ctx,
                inputs=[],
                schedule=[{"tool": "simplify"}],
                observer=observer,
            )

        self.assertEqual(len(observer.results), 1)
        self.assertEqual(output.getvalue(), "")


class TestParallelClassificationVerification(unittest.TestCase):
    @staticmethod
    def _fp_identity_result(max_workers, observer=None):
        ctx = SpecContext("parallel-fp-identity")
        value = fp32.fresh("value", ctx)
        return ast_nodes.check_equivalence(
            ast_nodes._Spec("first", lambda _ctx: value),
            ast_nodes._Spec("second", lambda _ctx: value),
            base_ctx=ctx,
            inputs=[],
            schedule=[{"tool": "simplify"}],
            observer=observer,
            max_workers=max_workers,
        )

    @staticmethod
    def _case_summary(result):
        return [
            (
                case_result["name"],
                case_result["proved"],
                case_result["status"],
                case_result["feasibility_status"],
                [report["tool"] for report in case_result["proof_trace"]],
                [
                    report["feasibility_status"]
                    for report in case_result["side_feasibility_reports"]
                ],
            )
            for case_result in result["case_results"]
        ]

    def test_parallel_matches_serial_order_and_observes_in_parent(self):
        class ParentObserver:
            def __init__(self):
                self.calls = []

            def case_completed(self, result):
                self.calls.append((os.getpid(), result["name"]))

        observer = ParentObserver()
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            serial = self._fp_identity_result(max_workers=1)
            parallel = self._fp_identity_result(
                max_workers=2,
                observer=observer,
            )

        self.assertTrue(serial["proved"])
        self.assertEqual(
            self._case_summary(parallel),
            self._case_summary(serial),
        )
        self.assertEqual(len(observer.calls), len(parallel["case_results"]))
        self.assertEqual(
            {case_name for _, case_name in observer.calls},
            {case["name"] for case in parallel["case_results"]},
        )
        self.assertEqual({pid for pid, _ in observer.calls}, {os.getpid()})

    def test_parallel_worker_can_launch_nested_z3_process(self):
        def run(max_workers):
            ctx = SpecContext("parallel-nested-z3")
            value = ctx.real("value")
            return ast_nodes.check_equivalence(
                ast_nodes._Spec("first", lambda _ctx: value),
                ast_nodes._Spec("second", lambda _ctx: value + ctx.one()),
                base_ctx=ctx,
                inputs=[],
                schedule=[{"tool": "z3", "timeout_ms": 1000}],
                max_workers=max_workers,
            )

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            serial = run(1)
            parallel = run(2)

        self.assertFalse(parallel["proved"])
        self.assertEqual(self._case_summary(parallel), self._case_summary(serial))
        self.assertEqual(parallel["proof_traces"][0][-1]["tool"], "z3")
        self.assertEqual(parallel["proof_traces"][0][-1]["status"], "sat")

    def test_serial_and_parallel_match_for_side_feasibility_cases(self):
        def run(max_workers, second_is_feasible):
            ctx = SpecContext("parallel-side-feasibility")
            value = ctx.real("value")

            def collect_zero(side_ctx):
                side_ctx.assume(value.eq(side_ctx.zero()))
                return value

            def collect_second(side_ctx):
                if second_is_feasible:
                    side_ctx.assume(value.eq(side_ctx.one()))
                else:
                    side_ctx.assume(side_ctx.false())
                return value

            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                return ast_nodes.check_equivalence(
                    ast_nodes._Spec("zero", collect_zero),
                    ast_nodes._Spec("second", collect_second),
                    base_ctx=ctx,
                    inputs=[],
                    schedule=[{"tool": "simplify"}],
                    max_workers=max_workers,
                )

        for second_is_feasible in (True, False):
            with self.subTest(second_is_feasible=second_is_feasible):
                serial = run(1, second_is_feasible)
                parallel = run(2, second_is_feasible)
                self.assertEqual(serial["proved"], parallel["proved"])
                self.assertEqual(
                    self._case_summary(serial),
                    self._case_summary(parallel),
                )

    def test_automatic_worker_count_uses_affinity_and_fallbacks(self):
        with (
            patch.object(
                parallel_runner.os,
                "sched_getaffinity",
                return_value={2, 4},
                create=True,
            ),
            patch.object(parallel_runner.os, "cpu_count") as cpu_count,
        ):
            self.assertEqual(parallel_runner.automatic_max_workers(), 2)
            cpu_count.assert_not_called()

        with (
            patch.object(
                parallel_runner.os,
                "sched_getaffinity",
                side_effect=OSError,
                create=True,
            ),
            patch.object(parallel_runner.os, "cpu_count", return_value=7),
        ):
            self.assertEqual(parallel_runner.automatic_max_workers(), 7)

        with (
            patch.object(
                parallel_runner.os,
                "sched_getaffinity",
                return_value=set(),
                create=True,
            ),
            patch.object(parallel_runner.os, "cpu_count", return_value=None),
        ):
            self.assertEqual(parallel_runner.automatic_max_workers(), 1)

    def test_invalid_worker_counts_are_rejected(self):
        for invalid in (True, 1.5, "2", []):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    parallel_runner.resolve_max_workers(invalid)
        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parallel_runner.resolve_max_workers(invalid)

    def test_parallel_submission_keeps_only_two_cases_per_worker_in_flight(self):
        generated_count = 0
        completed_count = 0
        generation_leads = []

        def cases():
            nonlocal generated_count
            for index in range(11):
                generated_count += 1
                yield SpecContext(f"bounded-{index}")

        class ImmediateFuture:
            def __init__(self, case_ctx):
                self.case_ctx = case_ctx

            def result(self):
                return CaseVerificationResult(
                    name=self.case_ctx.name,
                    proved=True,
                    status="unsat",
                    feasibility_status="feasible",
                    proof_trace=[],
                    side_feasibility_reports=[],
                )

        class FakeExecutor:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def submit(self, _fn, case_ctx, *_args):
                return ImmediateFuture(case_ctx)

        def complete_one(futures, return_when):
            nonlocal completed_count
            self.assertEqual(return_when, parallel_runner.FIRST_COMPLETED)
            generation_leads.append(generated_count - completed_count)
            completed_count += 1
            return {next(iter(futures))}, set(futures)

        with (
            patch.object(parallel_runner, "ProcessPoolExecutor", FakeExecutor),
            patch.object(parallel_runner, "wait", side_effect=complete_one),
        ):
            results = parallel_runner._run_in_parallel(
                cases(),
                verify_case=lambda case_ctx: case_ctx,
                verification_args=(),
                observer=Mock(),
                max_workers=2,
            )

        self.assertEqual(
            [result["name"] for result in results],
            [f"bounded-{index}" for index in range(11)],
        )
        self.assertLessEqual(max(generation_leads), 4)

    def test_parallel_worker_exception_propagates(self):
        ctx = SpecContext("parallel-worker-error")
        value = ctx.real("value")
        with self.assertRaisesRegex(ValueError, "Unknown schedule tool"):
            ast_nodes.check_equivalence(
                ast_nodes._Spec("first", lambda _ctx: value),
                ast_nodes._Spec("second", lambda _ctx: value),
                base_ctx=ctx,
                inputs=[],
                schedule=[{"tool": "not-a-tool"}],
                observer=Mock(),
                max_workers=2,
            )


class TestSpecificationDeterminism(unittest.TestCase):
    def test_check_spec_rejects_non_exhaustive_cases_before_equivalence(self):
        def malformed_spec(x, ctx):
            return Cases(
                case(BoolLit(False), x),
                ctx=ctx,
            )

        @Composite(name="non_exhaustive_cases", spec=malformed_spec)
        def non_exhaustive_cases(x):
            return x

        node = non_exhaustive_cases(Var(name="x", sign=UQT(2, 0)))
        with self.assertRaises(MalformedSpecification):
            node.check_spec(schedule=[{"tool": "simplify"}])

    def test_deterministic_primitive_uses_same_inputs_for_both_spec_runs(self):
        seen_inputs = []

        def deterministic_spec(x, ctx):
            seen_inputs.append(x)
            out = ctx.fresh_real("out")
            ctx.assume(out.eq(x + ctx.one()))
            return out

        @Primitive(name="deterministic_primitive", spec=deterministic_spec)
        def deterministic_primitive(x):
            return x.copy()

        node = deterministic_primitive(Var(name="x", sign=UQT(2, 0)))
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = node.check_determinism(
                schedule=[
                    {"tool": "simplify"},
                    {"tool": "z3", "timeout_ms": 1000},
                ]
            )

        self.assertTrue(result["proved"])
        self.assertEqual(len(result["proof_traces"]), 1)
        self.assertEqual(len(seen_inputs), 2)
        self.assertIs(seen_inputs[0], seen_inputs[1])

    def test_fp_inputs_get_independent_special_encoding_per_spec_run(self):
        seen_inputs = []

        def identity_spec(x, ctx):
            del ctx
            seen_inputs.append(x)
            return x

        @Primitive(name="fp_special_encoding", spec=identity_spec)
        def fp_special_encoding(x):
            return x.copy()

        node = fp_special_encoding(Var(name="x", sign=Float32T()))
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            node.check_determinism(schedule=[{"tool": "simplify"}])

        self.assertEqual(len(seen_inputs), 2)
        first, second = seen_inputs
        self.assertIsNot(first, second)
        self.assertEqual(
            first.classification_flags(),
            second.classification_flags(),
        )
        self.assertIsInstance(first.value, If)
        self.assertIsInstance(second.value, If)
        self.assertNotEqual(first.value.on_false, second.value.on_false)

    def test_check_spec_encodes_inner_and_outer_fp_inputs_independently(self):
        inner_inputs = []
        outer_inputs = []

        def inner_spec(x, ctx):
            del ctx
            inner_inputs.append(x)
            return x

        @Primitive(name="inner_fp_identity", spec=inner_spec)
        def inner_fp_identity(x):
            return x.copy()

        def outer_spec(x, ctx):
            del ctx
            outer_inputs.append(x)
            return x

        @Composite(name="outer_fp_identity", spec=outer_spec)
        def outer_fp_identity(x):
            return inner_fp_identity(x)

        node = outer_fp_identity(Var(name="x", sign=Float32T()))
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            node.check_spec(schedule=[{"tool": "simplify"}])

        self.assertEqual(len(inner_inputs), 1)
        self.assertEqual(len(outer_inputs), 1)
        inner = inner_inputs[0]
        outer = outer_inputs[0]
        self.assertEqual(
            inner.classification_flags(),
            outer.classification_flags(),
        )
        self.assertIsInstance(inner.value, If)
        self.assertIsInstance(outer.value, If)
        self.assertNotEqual(inner.value.on_false, outer.value.on_false)

    def test_underconstrained_primitive_is_not_deterministic(self):
        def nondeterministic_spec(_x, ctx):
            out = ctx.fresh_real("out")
            zero = ctx.zero()
            one = ctx.one()
            ctx.assume(out.eq(zero) | out.eq(one))
            return out

        @Primitive(name="nondeterministic_primitive", spec=nondeterministic_spec)
        def nondeterministic_primitive(x):
            return x.copy()

        node = nondeterministic_primitive(Var(name="x", sign=UQT(2, 0)))
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = node.check_determinism(
                schedule=[
                    {"tool": "simplify"},
                    {"tool": "z3", "timeout_ms": 1000},
                ]
            )

        self.assertFalse(result["proved"])
        self.assertEqual(result["proof_traces"][-1][-1]["status"], "sat")

    def test_combined_infeasibility_is_discharged_without_recollection(self):
        base_ctx = SpecContext("combined_infeasibility")
        x = base_ctx.real("x")
        collect_counts = {"zero": 0, "one": 0}

        def collect_zero(ctx):
            collect_counts["zero"] += 1
            ctx.assume(x.eq(ctx.zero()))
            return x

        def collect_one(ctx):
            collect_counts["one"] += 1
            ctx.assume(x.eq(ctx.one()))
            return x

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = ast_nodes.check_equivalence(
                ast_nodes._Spec("zero_spec", collect_zero),
                ast_nodes._Spec("one_spec", collect_one),
                base_ctx=base_ctx,
                inputs=[],
                schedule=[{"tool": "simplify"}],
            )

        self.assertTrue(result["proved"])
        self.assertEqual(collect_counts, {"zero": 1, "one": 1})
        self.assertEqual(len(result["case_results"]), 1)
        case_result = result["case_results"][0]
        self.assertEqual(case_result["feasibility_status"], "not feasible")
        self.assertEqual(len(case_result["side_feasibility_reports"]), 2)
        self.assertEqual(
            [
                report["feasibility_status"]
                for report in case_result["side_feasibility_reports"]
            ],
            ["feasible", "feasible"],
        )

    def test_combined_infeasibility_with_matching_infeasible_sides_is_proved(self):
        base_ctx = SpecContext("matching_infeasible_sides")
        x = base_ctx.real("x")

        def collect_infeasible(ctx):
            ctx.assume(ctx.false())
            return x

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = ast_nodes.check_equivalence(
                ast_nodes._Spec("first_spec", collect_infeasible),
                ast_nodes._Spec("second_spec", collect_infeasible),
                base_ctx=base_ctx,
                inputs=[],
                schedule=[{"tool": "simplify"}],
            )

        self.assertTrue(result["proved"])

    def test_side_feasibility_mismatch_is_not_proved(self):
        base_ctx = SpecContext("side_feasibility_mismatch")
        x = base_ctx.real("x")

        def collect_feasible(_ctx):
            return x

        def collect_infeasible(ctx):
            ctx.assume(ctx.false())
            return x

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = ast_nodes.check_equivalence(
                ast_nodes._Spec("feasible_spec", collect_feasible),
                ast_nodes._Spec("infeasible_spec", collect_infeasible),
                base_ctx=base_ctx,
                inputs=[],
                schedule=[{"tool": "simplify"}],
            )

        self.assertFalse(result["proved"])

    def test_unknown_side_feasibility_is_not_proved(self):
        base_ctx = SpecContext("unknown_side_feasibility")
        x = base_ctx.real("x")

        def collect_zero(ctx):
            ctx.assume(x.eq(ctx.zero()))
            return x

        def collect_one(ctx):
            ctx.assume(x.eq(ctx.one()))
            return x

        with (
            patch.object(
                ast_nodes,
                "simplify_ctx",
                return_value={"feasibility_status": "unknown"},
            ),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            result = ast_nodes.check_equivalence(
                ast_nodes._Spec("zero_spec", collect_zero),
                ast_nodes._Spec("one_spec", collect_one),
                base_ctx=base_ctx,
                inputs=[],
                schedule=[{"tool": "simplify"}],
                max_workers=1,
            )

        self.assertFalse(result["proved"])

    def test_composite_determinism_ignores_inner_proof_context(self):
        def identity_spec(x, ctx):
            del ctx
            return x

        @Composite(name="deterministic_composite", spec=identity_spec)
        def deterministic_composite(x):
            with context() as ctx:
                ctx.check(ctx.false())
            return x.copy()

        node = deterministic_composite(Var(name="x", sign=UQT(2, 0)))
        self.assertEqual(len(node.ctx.checks), 1)

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = node.check_determinism(schedule=[{"tool": "simplify"}])

        self.assertTrue(result["proved"])

    def test_nested_tuple_outputs_are_compared_recursively(self):
        def identity_spec(x, ctx):
            del ctx
            return x

        @Primitive(name="tuple_identity", spec=identity_spec)
        def tuple_identity(x):
            return x.copy()

        node = tuple_identity(
            Var(
                name="x",
                sign=TupleT(
                    UQT(2, 0),
                    TupleT(BoolT(), UQT(1, 1)),
                ),
            )
        )
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = node.check_determinism(schedule=[{"tool": "simplify"}])

        self.assertTrue(result["proved"])

    def test_output_shape_mismatch_raises_type_error(self):
        call_count = 0

        def unstable_shape_spec(x, ctx):
            del ctx
            nonlocal call_count
            call_count += 1
            return x if call_count == 1 else (x, x)

        @Primitive(name="unstable_shape", spec=unstable_shape_spec)
        def unstable_shape(x):
            return x.copy()

        node = unstable_shape(Var(name="x", sign=UQT(2, 0)))
        with self.assertRaisesRegex(TypeError, "Spec shape mismatch"):
            node.check_determinism(schedule=[{"tool": "simplify"}])

    def test_unobservable_nan_fields_do_not_make_spec_nondeterministic(self):
        def nan_spec(ctx):
            out = fp32.fresh("out", ctx)
            ctx.assume(out.is_nan.eq(ctx.true()))
            return out

        @Primitive(name="nan_primitive", spec=nan_spec)
        def nan_primitive():
            return Const(Float32.NaN())

        node = nan_primitive()
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = node.check_determinism(schedule=[{"tool": "simplify"}])

        self.assertTrue(result["proved"])
        self.assertEqual(len(result["proof_traces"]), 5)


class TestSolverApis(unittest.TestCase):
    def test_fp32_multiplier_check_spec_proves_zero_input_cases(self):
        multiplier = fp32_mult(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        zero_input_pairs = {
            ("zero", "zero"),
            ("zero", "norm"),
            ("norm", "zero"),
            ("zero", "sub"),
            ("sub", "zero"),
            ("zero", "inf"),
            ("inf", "zero"),
            ("zero", "nan"),
            ("nan", "zero"),
        }
        split_classification_cases = ast_nodes._split_classification_cases

        def select_zero_input_cases(
            ctx,
            inputs,
            spec_inner,
            spec_outer,
        ):
            cases = split_classification_cases(
                ctx,
                inputs,
                spec_inner,
                spec_outer,
            )
            selected = [
                case
                for case in cases
                if (
                    ast_nodes._case_labels(case.name)["arg0"],
                    ast_nodes._case_labels(case.name)["arg1"],
                ) in zero_input_pairs
            ]
            self.assertEqual(len(selected), len(zero_input_pairs) * 5)
            return selected

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_zero_input_cases,
            ),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            check_result = multiplier.check_spec(schedule=[{"tool": "simplify"}])

        self.assertTrue(check_result["proved"])
        self.assertEqual(
            len(check_result["proof_traces"]),
            len(zero_input_pairs) * 5,
        )

    def _assert_dot_product_check_spec_with_two_zero_inputs(
        self,
        design_fn,
        *,
        expect_egglog,
    ):
        zero = BFloat16.Zero()
        one = BFloat16.from_fields(sign=0, exponent=127, mantissa=0)
        largest_finite = BFloat16.from_fields(
            sign=0,
            exponent=254,
            mantissa=127,
        )
        small_normal = BFloat16.from_fields(
            sign=0,
            exponent=BFloat16.exponent_bias - 40,
            mantissa=0,
        )
        # The two small products sum to 2**-39. Before zero-product
        # exponents were masked, zero * largest_finite selected the maximum
        # exponent and shifted both real products completely away.
        input_values = (
            small_normal,
            zero,
            one,
            small_normal,
            one,
            largest_finite,
            zero,
            one,
        )
        self.assertEqual(sum(value.val == zero.val for value in input_values), 2)

        design = design_fn(*(
            Var(name=f"arg_{idx}", sign=BFloat16T())
            for idx in range(len(input_values))
        ))
        input_classifications = (
            "norm",
            "zero",
            "norm",
            "norm",
            "norm",
            "norm",
            "zero",
            "norm",
        )

        def select_normal_result_case(
            ctx,
            inputs,
            spec_inner,
            spec_outer,
        ):
            case_ctx = ctx.copy()
            named_inputs = [
                (f"arg{idx}", value)
                for idx, value in enumerate(inputs)
            ]
            labels = {
                name: classification
                for (name, _), classification in zip(
                    named_inputs,
                    input_classifications,
                    strict=True,
                )
            }
            labels["output"] = "norm"

            for symbolic, concrete in zip(inputs, input_values, strict=True):
                concrete_spec = concrete.to_spec(case_ctx)
                for symbolic_field, concrete_field in zip(
                    symbolic.decode()[1:],
                    concrete_spec.decode()[1:],
                    strict=True,
                ):
                    case_ctx.assume(symbolic_field.eq(concrete_field))

            for name, value in named_inputs:
                ast_nodes._assume_classification_case(
                    case_ctx,
                    name,
                    value,
                    labels[name],
                )
            ast_nodes._assume_classification_case(
                case_ctx,
                "output",
                spec_inner,
                labels["output"],
            )
            ast_nodes._assume_classification(
                case_ctx,
                spec_outer,
                labels["output"],
            )
            expected_labels = ",".join(
                f"arg{idx}={classification}"
                for idx, classification in enumerate(input_classifications)
            )
            self.assertEqual(
                case_ctx.name,
                f"{ctx.name}[{expected_labels},output=norm]",
            )
            ast_nodes._add_classification_case_checks(
                case_ctx,
                spec_inner,
                spec_outer,
                labels,
            )
            return [case_ctx]

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_normal_result_case,
            ),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            check_result = design.check_spec(
                schedule=[
                    {"tool": "simplify"},
                    {
                        "tool": "egglog-rewrite",
                        "iterations": 6,
                        "scheduler": {
                            "match_limit": 500_000,
                            "ban_length": 1,
                        },
                    },
                    {"tool": "simplify"},
                ],
            )

        self.assertTrue(check_result["proved"], check_result)
        self.assertEqual(len(check_result["proof_traces"]), 1)
        proof_trace = check_result["proof_traces"][0]
        self.assertTrue(
            any(
                report["tool"] in {"simplify", "egglog-rewrite"}
                and report["status"] == "unsat"
                for report in proof_trace
            ),
            proof_trace,
        )
        if expect_egglog:
            self.assertTrue(
                any(
                    report["tool"] == "egglog-rewrite"
                    and report["checks_after"] < report["checks_before"]
                    for report in proof_trace
                ),
                proof_trace,
            )

    def test_conventional_check_spec_with_two_zero_inputs(self):
        self._assert_dot_product_check_spec_with_two_zero_inputs(
            bf16x8_dot_fp32_conventional,
            expect_egglog=False,
        )

    def test_optimized_check_spec_with_two_zero_inputs(self):
        self._assert_dot_product_check_spec_with_two_zero_inputs(
            bf16x8_dot_fp32_optimized,
            expect_egglog=True,
        )

    def test_fp32_adder_norm_inf_inf_inf_is_trimmed_before_egglog(self):
        adder = fp32_add(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        ctx = adder.ctx.copy()
        spec_inner = ctx.spec_of(adder.inner_tree)
        inputs = [ctx.spec_of(arg) for arg in adder.inner_args]
        spec_outer = adder.spec(*inputs, ctx=ctx)
        target_name = (
            "fp32_add["
            "arg0=norm,arg1=inf,output=inf]"
        )
        case = next(
            case
            for case in ast_nodes._split_classification_cases(
                ctx,
                inputs,
                spec_inner,
                spec_outer,
            )
            if case.name == target_name
        )

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            status, trace = solver_engine.check_equivalence(
                case,
                schedule=[{"tool": "simplify"}],
            )

        self.assertEqual(status, "unsat")
        self.assertEqual(trace[-1]["new_ctx"].checks, [])


    def test_fp32_adder_inf_inf_norm_side_case_uses_only_real_and_bool_exprs(self):
        adder = fp32_add(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        labels = {
            "arg0": "inf",
            "arg1": "inf",
            "inner_spec": "norm",
            "outer_spec": "norm",
        }

        base_ctx = adder.ctx.copy()
        inputs = [base_ctx.spec_of(arg) for arg in adder.inner_args]
        simplified = ast_nodes._collect_classified_spec(
            ast_nodes._Spec(
                "inner_spec",
                lambda ctx: ctx.spec_of(adder.inner_tree),
            ),
            base_ctx=base_ctx,
            inputs=inputs,
            case_labels=labels,
        ).simplify()

        def uses_only_supported_exprs(node):
            return isinstance(node, (RealExpr, BoolExpr)) and all(
                uses_only_supported_exprs(child) for child in children(node)
            )

        self.assertTrue(all(uses_only_supported_exprs(expr) for expr in simplified.assumes))

    def test_fp32_adder_inf_inf_cannot_have_normal_outer_spec(self):
        adder = fp32_add(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        base_ctx = adder.ctx.copy()
        inputs = [base_ctx.spec_of(arg) for arg in adder.inner_args]
        simplified = ast_nodes._collect_classified_spec(
            ast_nodes._Spec(
                "outer_spec",
                lambda ctx: adder.spec(*inputs, ctx=ctx),
            ),
            base_ctx=base_ctx,
            inputs=inputs,
            case_labels={
                "arg0": "inf",
                "arg1": "inf",
                "outer_spec": "norm",
            },
        ).simplify()

        env = {}
        solver = z3.Solver()
        solver.add(*(assume.to_z3(env) for assume in simplified.assumes))
        self.assertEqual(solver.check(), z3.unsat)


    def test_check_equivalence_with_simplify_schedule_short_circuits(self):
        ctx = SpecContext("simplify-schedule")
        ctx.check((RealLit(1) + RealLit(2)).eq(RealLit(3)))

        status, proof_trace = solver_engine.check_equivalence(
            ctx,
            schedule=[{"tool": "simplify"}],
        )

        self.assertEqual(status, "unsat")
        self.assertEqual(len(proof_trace), 1)
        self.assertEqual(proof_trace[0]["tool"], "simplify")

    def test_check_equivalence_reports_poor_spec_from_simplify_schedule(self):
        ctx = SpecContext("simplify-conflict-schedule")
        x = ctx.real("x")

        ctx.assume(x.eq(ctx.zero()))
        ctx.assume(x.eq(ctx.one()))
        ctx.check(x.eq(ctx.zero()))

        status, proof_trace = solver_engine.check_equivalence(
            ctx,
            schedule=[{"tool": "simplify"}],
        )

        self.assertEqual(status, "sat")
        self.assertEqual(len(proof_trace), 1)
        self.assertEqual(proof_trace[0]["tool"], "simplify")
        self.assertEqual(proof_trace[0]["status"], "sat")
        self.assertIn("Conflicting learned literals", str(proof_trace[0]["info"]))

    def test_check_equivalence_returns_flat_proof_trace(self):
        ctx = SpecContext("flat-trace")
        ctx.check(RealLit(1).eq(RealLit(1)))

        with patch.dict(solver_engine.TOOL_FNS, {"z3": _flat_trace_tool}):
            status, proof_trace = solver_engine.check_equivalence(
                ctx,
                schedule=[{"tool": "z3", "timeout_ms": 1}],
            )

        self.assertEqual(status, "unsat")
        self.assertIsInstance(proof_trace, list)
        self.assertEqual(len(proof_trace), 1)
        self.assertIsInstance(proof_trace[0], dict)
        self.assertEqual(proof_trace[0]["tool"], "branch-b")

    def test_run_tool_has_hard_wall_clock_timeout(self):
        ctx = SpecContext("tool-timeout")

        with patch.dict(solver_engine.TOOL_FNS, {"z3": _slow_tool}):
            reports = solver_engine._run_tool(
                ctx,
                {"tool": "z3", "timeout_ms": 1},
                timeout=0.01,
            )

        self.assertEqual(reports[0]["status"], "unknown")
        self.assertEqual(reports[0]["wall_clock_timeout_s"], 0.01)

    def test_run_tool_reaps_child_when_wait_is_interrupted(self):
        ctx = SpecContext("interrupted-tool")
        parent_pipe = Mock()
        child_pipe = Mock()
        process = Mock(sentinel=object())
        process.is_alive.return_value = True
        process_ctx = Mock()
        process_ctx.Pipe.return_value = (parent_pipe, child_pipe)
        process_ctx.Process.return_value = process

        with (
            patch.object(
                solver_engine.multiprocessing,
                "get_context",
                return_value=process_ctx,
            ),
            patch.object(solver_engine, "wait", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            solver_engine._run_tool(
                ctx,
                {"tool": "z3", "timeout_ms": 1},
            )

        process.terminate.assert_called_once_with()
        process.join.assert_called_once_with()
        parent_pipe.close.assert_called_once_with()

    def test_spec_context_is_pickleable(self):
        ctx = SpecContext("pickle-context")
        x = ctx.real("x")
        ctx.check((x + ctx.one()).eq(ctx.two()))

        restored = pickle.loads(pickle.dumps(ctx))

        self.assertEqual(str(restored), str(ctx))
        for transported_ctx in (restored, restored.copy()):
            with self.assertRaisesRegex(RuntimeError, "spec_cache was discarded"):
                transported_ctx.spec_of(object())

    def test_run_tool_rejects_large_report(self):
        ctx = SpecContext("large-report")

        with (
            patch.dict(solver_engine.TOOL_FNS, {"z3": _large_report_tool}),
            patch.object(solver_engine, "MAX_TOOL_REPORT_BYTES", 1024),
        ):
            with self.assertRaisesRegex(RuntimeError, "maximum is 1024 bytes"):
                solver_engine._run_tool(ctx, {"tool": "z3", "timeout_ms": 1})

    def test_check_equivalence_rejects_rival_feasibility_as_proof_tool(self):
        ctx = SpecContext("rival-schedule")
        ctx.check(RealLit(1).eq(RealLit(1)))

        with self.assertRaises(ValueError) as raised:
            solver_engine.check_equivalence(
                ctx,
                schedule=[
                    {
                        "tool": "rival_feasibility_check",
                        "max_depth": "1",
                    }
                ],
            )

        self.assertIn("Unknown schedule tool rival_feasibility_check", str(raised.exception))

    def test_fp32_adder_norm_sub_zero_zero_is_proved_infeasible(self):
        adder = fp32_add(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        target_name = (
            "fp32_add["
            "arg0=norm,arg1=sub,output=zero]"
        )
        split_classification_cases = ast_nodes._split_classification_cases

        def select_target_case(*args, **kwargs):
            cases = split_classification_cases(*args, **kwargs)
            return [next(case for case in cases if case.name == target_name)]

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_target_case,
            ),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            check_result = adder.check_spec(
                schedule=[{"tool": "simplify"}],
            )

        self.assertTrue(check_result["proved"])
        self.assertEqual(
            check_result["proof_traces"][0][0]["feasibility_status"],
            "not feasible",
        )

    def test_fp32_adder_norm_norm_proves_with_egglog(self):
        adder = fp32_add(
            Var(name="a", sign=Float32T()),
            Var(name="b", sign=Float32T()),
        )
        target_name = "fp32_add[arg0=norm,arg1=norm,output=norm]"
        split_classification_cases = ast_nodes._split_classification_cases

        def select_norm_norm_case(*args, **kwargs):
            cases = split_classification_cases(*args, **kwargs)
            return [next(case for case in cases if case.name == target_name)]

        with (
            patch.object(
                ast_nodes,
                "_split_classification_cases",
                side_effect=select_norm_norm_case,
            ),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            check_result = adder.check_spec(
                schedule=[
                    {"tool": "simplify"},
                    {
                        "tool": "egglog-rewrite",
                        "iterations": 6,
                        "scheduler": {"match_limit": 500_000, "ban_length": 1},
                    },
                ]
            )

        self.assertTrue(check_result["proved"])
        matching_traces = [
            proof_trace
            for proof_trace in check_result["proof_traces"]
            if proof_trace and proof_trace[0]["name"] == target_name
        ]
        self.assertEqual(len(matching_traces), 1)
        self.assertTrue(
            any(
                report["tool"] == "egglog-rewrite" and report["status"] == "unsat"
                for report in matching_traces[0]
            ),
            matching_traces[0],
        )

    def test_z3_check_eq_returns_single_report(self):
        ctx = SpecContext("z3-api")
        ctx.check(RealLit(1).eq(RealLit(1)))

        report = z3_check_eq(ctx, timeout_ms=1000)

        self.assertIsInstance(report, dict)
        self.assertEqual(report["tool"], "z3")
        self.assertEqual(report["status"], "unsat")

    def test_dreal_check_eq_returns_single_report(self):
        ctx = SpecContext("dreal-api")
        ctx.check(RealLit(1).eq(RealLit(1)))

        report = dreal_check_eq(ctx, precision=0.001)

        self.assertIsInstance(report, dict)
        self.assertEqual(report["tool"], "dreal")
        self.assertEqual(report["status"], "unsat")


class TestSignSpecs(unittest.TestCase):
    def test_bit_operator_specs_use_one_canonical_conditional_form(self):
        ctx = SpecContext("bit-operator-canonical")
        x = ctx.real("x")
        y = ctx.real("y")
        zero = ctx.zero()
        one = ctx.one()

        expected = {
            and_spec: If(x.eq(one) & y.eq(one), one, zero),
            or_spec: If(x.eq(one) | y.eq(one), one, zero),
            xor_spec: If(x.ne(y), one, zero),
        }

        for spec, expected_result in expected.items():
            with self.subTest(spec=spec.__name__):
                self.assertEqual(spec(x, y, ctx), expected_result)
        self.assertEqual(neg_spec(x, ctx), If(x.eq(one), zero, one))
        self.assertEqual(ctx.assumes, [])

    def test_egglog_unions_guarded_bit_operator_forms(self):
        ctx = SpecContext("bit-operator-egglog")
        x = ctx.real("x")
        y = ctx.real("y")
        zero = ctx.zero()
        one = ctx.one()
        two = ctx.two()
        and_conditional = If(x.eq(one) & y.eq(one), one, zero)
        or_conditional = If(x.eq(one) | y.eq(one), one, zero)
        xor_conditional = If(x.ne(y), one, zero)
        neg_conditional = If(x.eq(one), zero, one)

        ctx.assume(x.eq(zero) | x.eq(one))
        ctx.assume(y.eq(zero) | y.eq(one))
        ctx.check(and_conditional.eq(x * y))
        ctx.check(and_conditional.eq(x.min(y)))
        ctx.check(or_conditional.eq(x + y - x * y))
        ctx.check(or_conditional.eq(x.max(y)))
        ctx.check(xor_conditional.eq(x.max(y) - x * y))
        ctx.check(xor_conditional.eq(x + y - two * x * y))
        ctx.check(sign_multiplier(ctx, xor_conditional).eq(
            sign_multiplier(ctx, x) * sign_multiplier(ctx, y)
        ))
        ctx.check(neg_conditional.eq(one - x))
        ctx.check(neg_conditional.eq(If(x.eq(zero), one, zero)))
        ctx.check(neg_conditional.eq(If(x.ne(zero), zero, one)))
        ctx.check(neg_conditional.eq(If(x.ne(one), one, zero)))
        for bit_result in (
            and_conditional,
            or_conditional,
            xor_conditional,
            neg_conditional,
        ):
            ctx.check(bit_result.eq(zero) | bit_result.eq(one))

        egraph = EGraph()
        egraph.register(*constant_rules())
        checks = ctx.to_egglog(egraph)
        egraph.run(1)

        self.assertTrue(egraph.check_bool(*checks))

    def test_egglog_propagates_nested_xor_result_as_bit(self):
        ctx = SpecContext("nested-xor-bit-closure")
        a, b, c, d = [ctx.real(name) for name in ("a", "b", "c", "d")]
        zero = ctx.zero()
        one = ctx.one()

        for value in (a, b, c, d):
            ctx.assume(value.eq(zero) | value.eq(one))

        ab = If(a.ne(b), one, zero)
        cd = If(c.ne(d), one, zero)
        nested_xor = If(ab.ne(cd), one, zero)
        ctx.check(nested_xor.eq(zero) | nested_xor.eq(one))
        ctx.check(
            sign_multiplier(ctx, nested_xor).eq(
                (
                    sign_multiplier(ctx, a)
                    * sign_multiplier(ctx, b)
                )
                * (
                    sign_multiplier(ctx, c)
                    * sign_multiplier(ctx, d)
                )
            )
        )

        egraph = EGraph()
        egraph.register(*constant_rules())
        checks = ctx.to_egglog(egraph)
        egraph.run(2)

        self.assertTrue(egraph.check_bool(*checks))

    def test_egglog_does_not_apply_bit_operator_forms_without_guards(self):
        ctx = SpecContext("bit-operator-egglog-unguarded")
        x = ctx.real("x")
        y = ctx.real("y")
        zero = ctx.zero()
        one = ctx.one()
        ctx.check(If(x.eq(one) & y.eq(one), one, zero).eq(x * y))
        ctx.check(If(x.eq(one) | y.eq(one), one, zero).eq(x.max(y)))
        ctx.check(If(x.ne(y), one, zero).eq(x.max(y) - x * y))
        ctx.check(If(x.eq(one), zero, one).eq(one - x))
        ctx.check(If(x.eq(one), zero, one).eq(If(x.eq(zero), one, zero)))
        ctx.check(If(x.eq(one), zero, one).eq(If(x.ne(zero), zero, one)))
        ctx.check(If(x.eq(one), zero, one).eq(If(x.ne(one), one, zero)))

        egraph = EGraph()
        egraph.register(*constant_rules())
        checks = ctx.to_egglog(egraph)
        egraph.run(1)

        self.assertTrue(all(
            not egraph.check_bool(check)
            for check in checks
        ))

    def test_q_signs_xor_spec_matches_constant_sign_combinations(self):
        cases = [
            (-1, -1, 0),
            (-1, 1, 1),
            (1, -1, 1),
            (1, 1, 0),
        ]

        for lhs, rhs, expected in cases:
            with self.subTest(lhs=lhs, rhs=rhs):
                node = q_signs_xor(Const(Q.from_int(lhs)), Const(Q.from_int(rhs)))
                self.assertEqual(node.evaluate().val, expected)

                ctx = SpecContext("q_signs_xor")
                spec = ctx.spec_of(node)
                ctx.check(spec.eq(ctx.real_val(expected)))
                report = simplify_ctx(ctx)
                if report["status"] == "unknown":
                    report = z3_check_eq(report["new_ctx"], timeout_ms=1000)

                self.assertEqual(str(spec), "real(xored_signs_2)")
                self.assertEqual(report["status"], "unsat", report)


if __name__ == "__main__":
    unittest.main()
