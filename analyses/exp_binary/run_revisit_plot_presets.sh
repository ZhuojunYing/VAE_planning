#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash analyses/exp_binary/run_revisit_plot_presets.sh [extra_plot_args...]

Runs analyses/exp_binary/plot_revisit_policy_diagnostics.R for every row in
analyses/exp_binary/revisit_plot_presets.csv. For each preset, it runs both:
  1. the full plot set
  2. the selected/core plot set using --selected-plots-only

Extra arguments are forwarded to both plotting calls. For example:
  bash analyses/exp_binary/run_revisit_plot_presets.sh --min-samples 25

Environment overrides:
  PRESET_FILE=analyses/exp_binary/revisit_plot_presets.csv
  PLOT_SCRIPT=analyses/exp_binary/plot_revisit_policy_diagnostics.R
  RSCRIPT_BIN=Rscript
  RUN_MODE=both        # one of: both, full, selected
  DRY_RUN=1            # print commands without running them

Examples:
  DRY_RUN=1 bash analyses/exp_binary/run_revisit_plot_presets.sh --min-samples 25
  bash analyses/exp_binary/run_revisit_plot_presets.sh --min-samples 25
  RUN_MODE=selected bash analyses/exp_binary/run_revisit_plot_presets.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PRESET_FILE="${PRESET_FILE:-analyses/exp_binary/revisit_plot_presets.csv}"
PLOT_SCRIPT="${PLOT_SCRIPT:-analyses/exp_binary/plot_revisit_policy_diagnostics.R}"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
RUN_MODE="${RUN_MODE:-both}"
EXTRA_ARGS=("$@")

case "${RUN_MODE}" in
  both|full|selected) ;;
  *)
    echo "RUN_MODE must be one of: both, full, selected. Got: ${RUN_MODE}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${PRESET_FILE}" ]]; then
  echo "Preset file not found: ${PRESET_FILE}" >&2
  exit 1
fi
if [[ ! -f "${PLOT_SCRIPT}" ]]; then
  echo "Plot script not found: ${PLOT_SCRIPT}" >&2
  exit 1
fi

mapfile -t PRESETS < <(
  "${RSCRIPT_BIN}" -e '
    args <- commandArgs(trailingOnly = TRUE)
    path <- args[[1]]
    dat <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
    required <- c("tree", "vary")
    missing <- setdiff(required, names(dat))
    if (length(missing) > 0) {
      stop(sprintf("Preset file is missing column(s): %s", paste(missing, collapse = ", ")))
    }
    dat <- dat[nzchar(trimws(dat$tree)) & nzchar(trimws(dat$vary)), , drop = FALSE]
    for (i in seq_len(nrow(dat))) {
      cat(trimws(dat$tree[[i]]), "\t", trimws(dat$vary[[i]]), "\n", sep = "")
    }
  ' "${PRESET_FILE}"
)

if [[ "${#PRESETS[@]}" -eq 0 ]]; then
  echo "No presets found in ${PRESET_FILE}" >&2
  exit 1
fi

declare -a MODES
case "${RUN_MODE}" in
  both) MODES=(full selected) ;;
  full) MODES=(full) ;;
  selected) MODES=(selected) ;;
esac

total_runs=$(( ${#PRESETS[@]} * ${#MODES[@]} ))
echo "Running ${total_runs} revisit plotting command(s)."
echo "Preset file: ${PRESET_FILE}"
echo "Plot script: ${PLOT_SCRIPT}"
echo "Mode: ${RUN_MODE}"
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  echo "Extra plot args: ${EXTRA_ARGS[*]}"
fi

run_index=0
for preset in "${PRESETS[@]}"; do
  IFS=$'\t' read -r tree vary <<< "${preset}"
  for mode in "${MODES[@]}"; do
    run_index=$((run_index + 1))
    cmd=("${RSCRIPT_BIN}" "${PLOT_SCRIPT}" "${tree}" "${vary}")
    if [[ "${mode}" == "selected" ]]; then
      cmd+=("--selected-plots-only")
    fi
    cmd+=("${EXTRA_ARGS[@]}")
    echo "[${run_index}/${total_runs}] ${cmd[*]}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      "${cmd[@]}"
    fi
  done
done
