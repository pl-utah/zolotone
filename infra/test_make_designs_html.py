from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from infra import make_designs_html


def _design(
    elapsed_s=1.0,
    determinism="passed",
    specification="passed",
    determinism_elapsed_s=0.25,
    specification_elapsed_s=0.75,
):
    checks = {}
    if determinism is not None:
        checks["determinism"] = {
            "status": determinism,
            "elapsed_s": determinism_elapsed_s,
        }
    if specification is not None:
        checks["specification"] = {
            "status": specification,
            "elapsed_s": specification_elapsed_s,
        }
    return {"elapsed_s": elapsed_s, "checks": checks}


class TestMakeDesignsHtml(unittest.TestCase):
    def test_renders_columns_order_formatting_statuses_and_escaping(self):
        names = [
            "passed <design>",
            "failed & design",
            "error",
            "timeout",
            "interrupted",
            "running",
            "not-run",
        ]
        report = {
            "started_at": "2026-08-04T01:02:03Z <start>",
            "finished_at": None,
            "designs": {
                names[0]: _design(
                    1.23456,
                    "passed",
                    "failed",
                    determinism_elapsed_s=0.23456,
                    specification_elapsed_s=0.98765,
                ),
                names[1]: _design(2, "failed", "error"),
                names[2]: _design(3, "error", "timeout"),
                names[3]: _design(4, "timeout", "interrupted"),
                names[4]: _design(5, "interrupted", "running"),
                names[5]: _design(6, "running", "passed"),
                names[6]: _design(7, None, None),
            },
        }

        html = make_designs_html.build_html(
            report,
            Path("report/<source>.json"),
        )

        for heading in (
            "Design",
            "Total time",
            "Determinism status",
            "Solver time spent",
            "Specification status",
            "Solver time spent",
        ):
            self.assertIn(f'<th scope="col">{heading}</th>', html)
        self.assertIn("1.235 s", html)
        self.assertIn("0.235 s", html)
        self.assertIn("0.988 s", html)
        self.assertIn("passed &lt;design&gt;", html)
        self.assertIn("failed &amp; design", html)
        self.assertIn("report/&lt;source&gt;.json", html)
        self.assertIn("2026-08-04T01:02:03Z &lt;start&gt;", html)
        for label in (
            "PASSED",
            "FAILED",
            "ERROR",
            "TIMEOUT",
            "INTERRUPTED",
            "RUNNING",
            "NOT RUN",
        ):
            self.assertIn(f">{label}</span>", html)

        positions = [
            html.index(f'<th scope="row">{name}')
            for name in (
                "passed &lt;design&gt;",
                "failed &amp; design",
                "error",
                "timeout",
                "interrupted",
                "running",
                "not-run",
            )
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("<script", html.lower())
        self.assertNotIn("https://", html.lower())

    def test_uses_design_order_from_json_object(self):
        report = {
            "designs": {
                "extra-first": _design(),
                "fp32_add": _design(),
                "bf16_add": _design(),
                "extra-last": _design(),
            }
        }

        html = make_designs_html.build_html(report, Path("run_designs.json"))

        positions = [
            html.index(f'<th scope="row">{name}</th>')
            for name in ("extra-first", "fp32_add", "bf16_add", "extra-last")
        ]
        self.assertEqual(positions, sorted(positions))

    def test_invalid_elapsed_values_render_as_not_available(self):
        invalid_values = [None, True, "invalid", "nan", "inf", -0.1]
        report = {
            "designs": {
                f"d{index}": _design(value)
                for index, value in enumerate(invalid_values)
            },
        }

        html = make_designs_html.build_html(report, Path("run_designs.json"))

        self.assertEqual(
            html.count('<td class="elapsed">n/a</td>'),
            len(invalid_values),
        )

    def test_missing_or_invalid_check_times_render_as_not_available(self):
        report = {
            "designs": {
                "missing": _design(
                    determinism_elapsed_s=None,
                    specification=None,
                ),
                "invalid": _design(
                    determinism_elapsed_s="bad",
                    specification_elapsed_s="nan",
                ),
            },
        }

        html = make_designs_html.build_html(report, Path("run_designs.json"))

        self.assertEqual(html.count('<td class="elapsed">n/a</td>'), 4)

    def test_cli_reads_and_writes_custom_report_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir)
            (report_dir / "run_designs.json").write_text(
                json.dumps(
                    {
                        "designs": {"demo": _design(0.125)},
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "infra.make_designs_html",
                    "--report-dir",
                    str(report_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            html_path = report_dir / "index.html"
            self.assertTrue(html_path.is_file())
            self.assertIn("demo", html_path.read_text(encoding="utf-8"))

    def test_bad_reports_fail_clearly_without_creating_html(self):
        cases = {
            "missing": None,
            "malformed": "{not json",
            "invalid-top-level": "[]",
            "invalid-designs": '{"designs": []}',
            "invalid-design": '{"designs": {"demo": []}}',
        }

        for name, contents in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                report_dir = Path(temp_dir)
                if contents is not None:
                    (report_dir / "run_designs.json").write_text(
                        contents,
                        encoding="utf-8",
                    )
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    status = make_designs_html.main(
                        ["--report-dir", str(report_dir)]
                    )

                self.assertEqual(status, 1)
                self.assertIn("error:", stderr.getvalue())
                self.assertFalse((report_dir / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
