#!/usr/bin/env bash
set -euo pipefail

TASKS="${1:-all}"
INPUT_TYPE="${2:-uniform}"
BREAKPOINT_DIR="${3:-analyses/exp_binary/results/exact_time_cost_breakpoints}"
EXACT_DIR="${4:-analyses/exp_binary/results/exact_time_cost}"
OUTPUT_PREFIX="${5:-exact_time_cost}"
SAMPLE_COST_COUNT="${6:-10}"

GENERATOR="analyses/exp_binary/generate_exact_time_cost_breakpoints.py"
SOLVER="analyses/exp_binary/exact_time_cost_solution.py"
TMP_DIR="${EXACT_DIR}/.breakpoint_task_runs"

mkdir -p "${BREAKPOINT_DIR}" "${EXACT_DIR}" "${TMP_DIR}"

python "${GENERATOR}" \
  --task ${TASKS} \
  --input-type "${INPUT_TYPE}" \
  --output-dir "${BREAKPOINT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --sample-cost-count "${SAMPLE_COST_COUNT}"

COST_CSV="${BREAKPOINT_DIR}/${OUTPUT_PREFIX}_representative_costs.csv"

mapfile -t TASK_LIST < <(python - "${COST_CSV}" <<'PY'
import csv
import sys

seen = []
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        task = row["task"]
        if task not in seen:
            seen.append(task)
print("\n".join(seen))
PY
)

for task in "${TASK_LIST[@]}"; do
  costs="$(python - "${COST_CSV}" "${task}" <<'PY'
import csv
import sys

costs = []
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        if row["task"] == sys.argv[2]:
            costs.append(row["representative_time_cost"])
print(",".join(costs))
PY
)"
  echo "Running ${task} at costs: ${costs}"
  python "${SOLVER}" \
    --task "${task}" \
    --input-type "${INPUT_TYPE}" \
    --time-costs "${costs}" \
    --output-dir "${TMP_DIR}" \
    --output-prefix "${OUTPUT_PREFIX}_${task}"
done

combine_csvs() {
  local suffix="$1"
  local output_path="${EXACT_DIR}/${OUTPUT_PREFIX}_${suffix}.csv"
  python - "${output_path}" "${TMP_DIR}" "${OUTPUT_PREFIX}" "${suffix}" "${TASK_LIST[@]}" <<'PY'
import csv
import os
import sys

output_path, tmp_dir, prefix, suffix, *tasks = sys.argv[1:]
rows = []
fieldnames = []
for task in tasks:
    path = os.path.join(tmp_dir, f"{prefix}_{task}_{suffix}.csv")
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for name in reader.fieldnames or []:
            if name not in fieldnames:
                fieldnames.append(name)
        rows.extend(reader)

with open(output_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

combine_csvs "states"
combine_csvs "actions"
combine_csvs "occupancy"
combine_csvs "summary"

echo "Wrote combined exact outputs to ${EXACT_DIR}/${OUTPUT_PREFIX}_{states,actions,occupancy,summary}.csv"
