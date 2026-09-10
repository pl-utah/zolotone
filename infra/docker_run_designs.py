"""Docker coordinator for verification and HTML report generation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

COORDINATOR_PATH = Path(__file__).resolve()
PROJECT_ROOT = COORDINATOR_PATH.parent.parent

# Direct execution adds infra/, rather than the repository root, to sys.path.
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from infra import make_designs_html, run_designs


DEFAULT_REPORT_DIR = Path("/reports")
DEFAULT_TIMEOUT_S = "600"


def _runner_args(argv: list[str], environ: dict[str, str]) -> list[str]:
    report_dir = Path(environ.get("REPORT_DIR", str(DEFAULT_REPORT_DIR)))
    args = [
        "--report",
        str(report_dir / make_designs_html.REPORT_FILENAME),
        "--timeout",
        environ.get("DESIGN_TIMEOUT_S", DEFAULT_TIMEOUT_S),
    ]
    max_workers = environ.get("DESIGN_MAX_WORKERS")
    if max_workers:
        args.extend(("--max-workers", max_workers))
    # argparse uses the last occurrence, so explicit container arguments take
    # precedence over environment-derived defaults.
    return [*args, *argv]


def _generate_html(report_path: Path, report_dir: Path) -> Path:
    html_path = report_dir / make_designs_html.HTML_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    html = make_designs_html.build_html(report, report_path)
    html_path.write_text(html, encoding="utf-8")
    print(f"Generated {html_path}")
    return html_path


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    report_dir = Path(os.environ.get("REPORT_DIR", str(DEFAULT_REPORT_DIR)))
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        runner_args = _runner_args(argv, os.environ)
        configured_args = run_designs.parse_args(runner_args)
    except (OSError, SystemExit) as exc:
        print(f"Failed to configure design verification: {exc}", file=sys.stderr)
        return 1

    interrupted = False
    try:
        run_designs.main(runner_args)
    except KeyboardInterrupt:
        interrupted = True
    except BaseException as exc:
        print(
            f"Design verification infrastructure failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    report_path = configured_args.report
    if not report_path.is_file():
        print(f"JSON report was not created at {report_path}", file=sys.stderr)
        return 1

    try:
        html_path = _generate_html(report_path, report_dir)
    except BaseException as exc:
        print(
            f"Failed to generate the HTML report: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Reports available in {report_dir}:")
    print(f"  {report_path}")
    print(f"  {html_path}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
