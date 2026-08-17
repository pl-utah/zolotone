#!/usr/bin/env bash
set -uo pipefail

report_dir="${REPORT_DIR:-/reports}"
timeout_s="${DESIGN_TIMEOUT_S:-600}"
report_path="${report_dir}/run_designs.json"
runner_args=(
    --report "${report_path}"
    --timeout "${timeout_s}"
)

if [[ -n "${DESIGN_MAX_WORKERS:-}" ]]; then
    runner_args+=(--max-workers "${DESIGN_MAX_WORKERS}")
fi

mkdir -p "${report_dir}"

run_status=0
python /app/infra/run_designs.py "${runner_args[@]}" "$@" || run_status=$?

if [[ ! -f "${report_path}" ]]; then
    echo "JSON report was not created at ${report_path}" >&2
    exit 1
fi

if ! python /app/infra/make_designs_html.py --report-dir "${report_dir}"; then
    echo "Failed to generate the HTML report" >&2
    exit 1
fi

echo "Reports available in ${report_dir}:"
echo "  ${report_path}"
echo "  ${report_dir}/index.html"

exit "${run_status}"
