from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path("reports")
REPORT_FILENAME = "run_designs.json"
HTML_FILENAME = "index.html"
REPORT_FILE_MODE = 0o644
CHECK_NAMES = ("determinism", "specification")
EMPTY_TABLE_ROW = (
    '<tr><td class="empty" colspan="6">No designs in this report.</td></tr>'
)

STATUS_LABELS = {
    "passed": "PASSED",
    "failed": "FAILED",
    "error": "ERROR",
    "timeout": "TIMEOUT",
    "interrupted": "INTERRUPTED",
}
CASE_STATUSES = frozenset({"sat", "unsat", "unknown"})


class ReportError(ValueError):
    """A report cannot be loaded or does not have the expected structure."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML summary from run_designs.json"
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=(
            "Directory containing run_designs.json and receiving index.html "
            f"(default: {DEFAULT_REPORT_DIR})"
        ),
    )
    return parser.parse_args(argv)


def _validate_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ReportError(
            "invalid report structure: top-level value must be an object"
        )

    designs = report.get("designs")
    if not isinstance(designs, dict):
        raise ReportError("invalid report structure: 'designs' must be an object")

    for name, result in designs.items():
        if not isinstance(name, str) or not isinstance(result, dict):
            raise ReportError(
                "invalid report structure: each design must map a name to an object"
            )
        checks = result.get("checks", {})
        if not isinstance(checks, dict):
            raise ReportError(
                "invalid report structure: "
                f"checks for design {name!r} must be an object"
            )
        for check_name in CHECK_NAMES:
            if check_name not in checks:
                continue
            check = checks[check_name]
            if not isinstance(check, dict):
                raise ReportError(
                    "invalid report structure: "
                    f"{check_name} check for design {name!r} must be an object"
                )
            cases = check.get("cases", {})
            if not isinstance(cases, dict):
                raise ReportError(
                    "invalid report structure: "
                    f"cases for {check_name} check of design {name!r} "
                    "must be an object"
                )
            if any(
                not isinstance(case_name, str) or not isinstance(case, dict)
                for case_name, case in cases.items()
            ):
                raise ReportError(
                    "invalid report structure: each case for "
                    f"{check_name} check of design {name!r} must map a name "
                    "to an object"
                )

    return report


def load_report(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except FileNotFoundError as exc:
        raise ReportError(f"report file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(
            f"malformed JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise ReportError(f"could not read report {path}: {exc}") from exc
    return _validate_report(report)


def _format_elapsed(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return "n/a"
    try:
        elapsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    if not math.isfinite(elapsed) or elapsed < 0:
        return "n/a"
    return f"{elapsed:.3f} s"


def _format_check_elapsed(check: Any) -> str:
    if not isinstance(check, dict):
        return "n/a"

    timed_out = check.get("status") == "timeout"
    elapsed_key = "wall_elapsed_s" if timed_out else "elapsed_s"
    elapsed = _format_elapsed(check.get(elapsed_key))
    if timed_out and elapsed != "n/a":
        return f"{elapsed} (wall)"
    return elapsed


def _status_badge(check: Any) -> str:
    if not isinstance(check, dict) or "status" not in check:
        status = "not-run"
        label = "NOT RUN"
    else:
        raw_status = check["status"]
        if isinstance(raw_status, str) and raw_status in STATUS_LABELS:
            status = raw_status
            label = STATUS_LABELS[raw_status]
        else:
            status = "unknown"
            label = "UNKNOWN" if raw_status is None else str(raw_status).upper()
    return f'<span class="badge badge-{status}">{escape(label)}</span>'


def _case_status_badge(case: dict[str, Any]) -> str:
    raw_status = case.get("status")
    status = (
        raw_status
        if isinstance(raw_status, str) and raw_status in CASE_STATUSES
        else "unknown"
    )
    label = "UNKNOWN" if raw_status is None else str(raw_status).upper()
    return f'<span class="badge badge-{status}">{escape(label)}</span>'


def _proved_badge(value: Any) -> str:
    if value is True:
        status, label = "passed", "YES"
    elif value is False:
        status, label = "failed", "NO"
    else:
        status, label = "not-run", "n/a"
    return f'<span class="badge badge-{status}">{label}</span>'


def _elapsed_cell(elapsed: str) -> str:
    return f'<td class="elapsed">{escape(elapsed)}</td>'


def _render_check_cells(check: Any) -> str:
    status_cell = f"<td>{_status_badge(check)}</td>"
    elapsed_cell = _elapsed_cell(_format_check_elapsed(check))
    return status_cell + elapsed_cell


def _render_case_row(name: str, case: dict[str, Any]) -> str:
    case_name = escape(name)
    cells = (
        f'<th scope="row">{case_name}</th>',
        f"<td>{_case_status_badge(case)}</td>",
        f'<td>{_display_text(case.get("feasibility"))}</td>',
        f"<td>{_proved_badge(case.get('proved'))}</td>",
        _elapsed_cell(_format_elapsed(case.get("elapsed_s"))),
    )
    return f"<tr>{''.join(cells)}</tr>"


def _render_cases_table(check_name: str, check: Any) -> str:
    cases = check.get("cases", {}) if isinstance(check, dict) else {}
    rows = "".join(
        _render_case_row(case_name, case)
        for case_name, case in cases.items()
    )
    if not rows:
        rows = '<tr><td class="empty" colspan="5">No cases reported.</td></tr>'

    heading = check_name.capitalize()
    return f"""
              <section class="case-section">
                <h2>{heading} cases</h2>
                <div class="case-table-wrap">
                  <table class="case-table">
                    <thead>
                      <tr>
                        <th scope="col">Case name</th>
                        <th scope="col">Status</th>
                        <th scope="col">Feasibility</th>
                        <th scope="col">Proved?</th>
                        <th scope="col">Time spent</th>
                      </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                  </table>
                </div>
              </section>"""


def _render_design_rows(
    name: str,
    result: dict[str, Any],
    index: int,
) -> str:
    checks = result.get("checks", {})
    details_id = f"design-details-{index}"
    design_button = (
        '<button type="button" class="design-toggle" '
        f'aria-expanded="false" aria-controls="{details_id}">'
        f"{escape(name)}</button>"
    )
    cells = (
        f'<th scope="row">{design_button}</th>',
        _elapsed_cell(_format_elapsed(result.get("elapsed_s"))),
        _render_check_cells(checks.get("determinism")),
        _render_check_cells(checks.get("specification")),
    )
    summary_row = f'<tr class="design-row">{"".join(cells)}</tr>'
    detail_tables = "".join(
        _render_cases_table(check_name, checks.get(check_name))
        for check_name in CHECK_NAMES
    )
    detail_row = (
        f'<tr id="{details_id}" class="design-details" hidden>'
        f'<td colspan="6">{detail_tables}</td></tr>'
    )
    return summary_row + detail_row


def _display_text(value: Any) -> str:
    return escape("n/a" if value is None else str(value))


def build_html(report: dict[str, Any], source_path: Path) -> str:
    report = _validate_report(report)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rows = "".join(
        _render_design_rows(name, result, index)
        for index, (name, result) in enumerate(report["designs"].items())
    ) or EMPTY_TABLE_ROW

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zolotone Report</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 1.5rem;
      color: #222;
      background: #fff;
      font-family: Arial, sans-serif;
    }}
    main {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{ margin-bottom: .5rem; font-size: 1.75rem; }}
    h2 {{ margin: 0 0 .6rem; font-size: 1.1rem; }}
    .metadata {{ margin: 0 0 1.25rem; color: #555; line-height: 1.5; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; min-width: 800px; border-collapse: collapse; }}
    th, td {{ padding: .6rem .75rem; text-align: left; border: 1px solid #ccc; }}
    thead th {{ background: #eee; }}
    .elapsed {{ white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .badge {{
      font-weight: 700;
      white-space: nowrap;
    }}
    .badge-passed {{ color: #16733c; }}
    .badge-failed, .badge-error {{ color: #b42318; }}
    .badge-timeout, .badge-interrupted {{ color: #8a6100; }}
    .badge-unsat {{ color: #16733c; }}
    .badge-sat {{ color: #b42318; }}
    .badge-not-run, .badge-unknown, .empty {{ color: #666; }}
    .empty {{ text-align: center; }}
    .design-toggle {{
      display: inline-flex;
      gap: .4rem;
      align-items: center;
      padding: 0;
      border: 0;
      color: #175ea8;
      background: transparent;
      font: inherit;
      font-weight: 700;
      text-align: left;
      cursor: pointer;
    }}
    .design-toggle::before {{ content: "\\25B8"; display: inline-block; }}
    .design-toggle[aria-expanded="true"]::before {{ transform: rotate(90deg); }}
    .design-toggle:focus-visible {{ outline: 2px solid #175ea8; outline-offset: 3px; }}
    .design-details[hidden] {{ display: none; }}
    .design-details > td {{ padding: 1rem; background: #f8f9fa; }}
    .case-section + .case-section {{ margin-top: 1rem; }}
    .case-table-wrap {{ overflow-x: auto; }}
    .case-table {{ min-width: 640px; background: #fff; }}
    .case-table th, .case-table td {{ padding: .45rem .6rem; }}
  </style>
</head>
<body>
  <main>
    <h1>Design Verification Report</h1>
    <p class="metadata">
      Source: {_display_text(source_path)}<br>
      Report started: {_display_text(report.get("started_at"))}<br>
      Report finished: {_display_text(report.get("finished_at"))}<br>
      Generated at: {_display_text(generated_at)}
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Design</th>
            <th scope="col">Total time</th>
            <th scope="col">Determinism status</th>
            <th scope="col">Solver time spent</th>
            <th scope="col">Specification status</th>
            <th scope="col">Solver time spent</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
  </main>
  <script>
    document.querySelectorAll(".design-toggle").forEach((button) => {{
      button.addEventListener("click", () => {{
        const details = document.getElementById(
          button.getAttribute("aria-controls")
        );
        if (!details) return;
        const expanded = button.getAttribute("aria-expanded") === "true";
        button.setAttribute("aria-expanded", String(!expanded));
        details.hidden = expanded;
      }});
    }});
  </script>
</body>
</html>
"""


def _atomic_write(path: Path, contents: str) -> None:
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
            temporary_file.write(contents)
        os.chmod(temporary_path, REPORT_FILE_MODE)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report_dir / REPORT_FILENAME
    html_path = args.report_dir / HTML_FILENAME
    try:
        report = load_report(report_path)
        html = build_html(report, report_path)
        _atomic_write(html_path, html)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not write {html_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
