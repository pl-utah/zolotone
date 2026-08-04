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


DEFAULT_REPORT_DIR = Path("report")
REPORT_FILENAME = "run_designs.json"
HTML_FILENAME = "index.html"

STATUS_LABELS = {
    "passed": "PASSED",
    "failed": "FAILED",
    "error": "ERROR",
    "timeout": "TIMEOUT",
    "interrupted": "INTERRUPTED",
    "running": "RUNNING",
}


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
        for check_name in ("determinism", "specification"):
            if check_name in checks and not isinstance(checks[check_name], dict):
                raise ReportError(
                    "invalid report structure: "
                    f"{check_name} check for design {name!r} must be an object"
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


def _display_metadata(value: Any) -> str:
    return "n/a" if value is None else str(value)


def build_html(report: dict[str, Any], source_path: Path) -> str:
    report = _validate_report(report)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    designs = report["designs"]
    rows = []
    for name, result in designs.items():
        checks = result.get("checks", {})
        determinism = checks.get("determinism")
        specification = checks.get("specification")
        rows.append(
            "<tr>"
            f'<th scope="row">{escape(name)}</th>'
            f'<td class="elapsed">{escape(_format_elapsed(result.get("elapsed_s")))}</td>'
            f'<td>{_status_badge(determinism)}</td>'
            f'<td class="elapsed">{escape(_format_elapsed(determinism.get("elapsed_s") if determinism else None))}</td>'
            f'<td>{_status_badge(specification)}</td>'
            f'<td class="elapsed">{escape(_format_elapsed(specification.get("elapsed_s") if specification else None))}</td>'
            "</tr>"
        )

    if not rows:
        rows.append(
            '<tr><td class="empty" colspan="6">No designs in this report.</td></tr>'
        )

    source = escape(str(source_path))
    started_at = escape(_display_metadata(report.get("started_at")))
    finished_at = escape(_display_metadata(report.get("finished_at")))
    generated = escape(generated_at)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Design Verification Report</title>
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
    .badge-running {{ color: #175ea8; }}
    .badge-not-run, .badge-unknown, .empty {{ color: #666; }}
    .empty {{ text-align: center; }}
  </style>
</head>
<body>
  <main>
    <h1>Design Verification Report</h1>
    <p class="metadata">
      Source: {source}<br>
      Report started: {started_at}<br>
      Report finished: {finished_at}<br>
      Generated at: {generated}
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
          {''.join(rows)}
        </tbody>
      </table>
    </div>
  </main>
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
