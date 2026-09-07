from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_REPORT_DIR = Path("reports")
REPORT_FILENAME = "run_designs.json"
HTML_FILENAME = "index.html"
CHECK_NAMES = ("determinism", "specification")
EMPTY_TABLE_ROW = (
    '<tr><td class="empty" colspan="6">No designs in this report.</td></tr>'
)

CATEGORY_LABELS = {
    "arithmetic": "Arithmetic",
    "dot_product": "Dot product",
    "converter": "Converter",
}

LATEX_CATEGORY_LABELS = {
    "arithmetic": "Arithmetic",
    "dot_product": "Dot products",
    "converter": "Converters",
    "uncategorized": "Uncategorized",
}

LATEX_CATEGORY_ORDER = (
    "arithmetic",
    "dot_product",
    "converter",
    "uncategorized",
)

STATUS_LABELS = {
    "passed": "PASSED",
    "failed": "FAILED",
    "error": "ERROR",
    "timeout": "TIMEOUT",
    "interrupted": "INTERRUPTED",
}


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


def _format_elapsed(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} s"


def _status_badge(check: dict[str, Any] | None) -> str:
    if check is None:
        status = "not-run"
        label = "NOT RUN"
    else:
        status = check["status"]
        label = STATUS_LABELS[status]
    return f'<span class="badge badge-{status}">{label}</span>'


def _case_status_badge(case: dict[str, Any]) -> str:
    status = case["status"]
    return f'<span class="badge badge-{status}">{status.upper()}</span>'


def _proved_badge(value: bool) -> str:
    if value:
        status, label = "passed", "YES"
    else:
        status, label = "failed", "NO"
    return f'<span class="badge badge-{status}">{label}</span>'


def _elapsed_cell(elapsed: str) -> str:
    return f'<td class="elapsed">{escape(elapsed)}</td>'


def _render_check_cells(check: dict[str, Any] | None) -> str:
    status_cell = f"<td>{_status_badge(check)}</td>"
    elapsed = None if check is None else check["elapsed_s"]
    elapsed_cell = _elapsed_cell(_format_elapsed(elapsed))
    return status_cell + elapsed_cell


def _render_case_row(name: str, case: dict[str, Any]) -> str:
    case_name = escape(name)
    cells = (
        f'<th scope="row">{case_name}</th>',
        f"<td>{_case_status_badge(case)}</td>",
        f'<td>{escape(case["feasibility"])}</td>',
        f"<td>{_proved_badge(case['proved'])}</td>",
        _elapsed_cell(_format_elapsed(case["elapsed_s"])),
    )
    return f"<tr>{''.join(cells)}</tr>"


def _render_cases_table(
    check_name: str,
    check: dict[str, Any] | None,
) -> str:
    cases = {} if check is None else check.get("cases", {})
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
    checks = result["checks"]
    details_id = f"design-details-{index}"
    design_button = (
        '<button type="button" class="design-toggle" '
        f'aria-expanded="false" aria-controls="{details_id}">'
        f"{escape(name)}</button>"
    )
    cells = (
        f'<th scope="row">{design_button}</th>',
        _elapsed_cell(_format_elapsed(result["elapsed_s"])),
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


def _render_designs(designs: dict[str, dict[str, Any]]) -> str:
    sections = []
    previous_category = None
    for index, (name, result) in enumerate(designs.items()):
        category = result.get("category")
        key = category if category in CATEGORY_LABELS else "uncategorized"
        if key != previous_category:
            label = CATEGORY_LABELS.get(key, "Uncategorized")
            sections.append(
                '<tr class="category-row">'
                f'<th scope="colgroup" colspan="6">{label}</th></tr>'
            )
            previous_category = key
        sections.append(_render_design_rows(name, result, index))
    return "".join(sections)


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _format_latex_elapsed(check: dict[str, Any] | None) -> str:
    if check is None or check.get("elapsed_s") is None:
        return r"\todo{...s}"
    return f"{float(check['elapsed_s']):.3f}\\,s"


def build_latex_table(
    report: dict[str, Any],
    expected_designs: list[tuple[str, str]] | None = None,
) -> str:
    designs = {
        name: {"category": category, "checks": {}}
        for name, category in (expected_designs or [])
    }
    designs.update(report["designs"])

    grouped_designs = {category: [] for category in LATEX_CATEGORY_ORDER}
    for name, result in designs.items():
        category = result.get("category")
        key = category if category in CATEGORY_LABELS else "uncategorized"
        grouped_designs[key].append((name, result))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        "Design",
        r"& \makecell{Equivalence\\check (s)}",
        r"& \makecell{Determinism\\check (s)} \\",
        r"\midrule",
    ]

    category_written = False
    for category in LATEX_CATEGORY_ORDER:
        designs = grouped_designs[category]
        if not designs:
            continue
        if category_written:
            lines.append(r"\midrule")
        category_written = True
        label = LATEX_CATEGORY_LABELS[category]
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{label}}}}} \\")
        for name, result in sorted(designs):
            checks = result.get("checks", {})
            equivalence = _format_latex_elapsed(checks.get("specification"))
            determinism = _format_latex_elapsed(checks.get("determinism"))
            lines.extend(
                (
                    _escape_latex(name),
                    f"    & {equivalence} & {determinism} \\\\",
                )
            )

    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{\tool's runtime spent on verification time for the circuit designs.}",
            r"\label{tab:verification-time}",
            r"\end{table}",
        )
    )
    return "\n".join(lines) + "\n"


def _registered_designs() -> list[tuple[str, str]]:
    from infra.run_designs import DESIGNS

    return [(design.name, design.category) for design in DESIGNS]


def build_html(report: dict[str, Any], source_path: Path) -> str:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rows = _render_designs(report["designs"]) or EMPTY_TABLE_ROW

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
    .category-row th {{
      padding: .5rem .75rem;
      color: #334155;
      background: #e8eef5;
      font-size: .85rem;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
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
      Source: {escape(str(source_path))}<br>
      Report started: {escape(report["started_at"])}<br>
      Generated at: {generated_at}
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Design</th>
            <th scope="col">Total time</th>
            <th scope="col">Determinism status</th>
            <th scope="col">Time spent</th>
            <th scope="col">Specification status</th>
            <th scope="col">Time spent</th>
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
        const expanded = button.getAttribute("aria-expanded") === "true";
        button.setAttribute("aria-expanded", String(!expanded));
        details.hidden = expanded;
      }});
    }});
  </script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report_dir / REPORT_FILENAME
    html_path = args.report_dir / HTML_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    html = build_html(report, report_path)
    html_path.write_text(html, encoding="utf-8")

    print(f"Generated {html_path}", file=sys.stderr)
    print(
        build_latex_table(report, expected_designs=_registered_designs()),
        end="",
    )
    return 0


if __name__ == "__main__":
    main()
