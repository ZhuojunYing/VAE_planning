#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash analyses/exp_binary/run_bayesian_revisit_2node_grid.sh \
    [time_costs] [sigma_list] [seed_spec] [max_observations] [grid_size] \
    [quadrature_order] [simulate_trials] [outdir] [extra_solver_args...]

Arguments:
  time_costs          Comma-separated costs solved within each job.
  sigma_list          Comma-separated observation-noise SDs.
  seed_spec           Either start:end, comma-list, or one seed.
  max_observations    Maximum observations before forced stop.
  grid_size           Sum-statistic interpolation grid size.
  quadrature_order    Gauss-Hermite quadrature order.
  simulate_trials     Number of optimal-policy simulation trials per job.
  outdir              Output directory for Bayesian optimal CSVs.

Defaults:
  time_costs          0.02,0.04,0.08,0.1
  sigma_list          0,0.5,1.0
  seed_spec           1:3
  max_observations    10
  grid_size           161
  quadrature_order    21
  simulate_trials     2000
  outdir              analyses/exp_binary/results/bayesian_revisit_2node

Examples:
  bash analyses/exp_binary/run_bayesian_revisit_2node_grid.sh \
    "0.02,0.04,0.08,0.1" "0,0.5,1.0" "1:3" 10 161 21 2000

  bash analyses/exp_binary/run_bayesian_revisit_2node_grid.sh \
    "0.02,0.04" "1.5,2,3" "1,4,7" 10 201 31 5000 \
    analyses/exp_binary/results/bayesian_revisit_2node --grid-sigma-bound 8

Optional environment overrides for sbatch:
  SLURM_PARTITION=general
  SLURM_TIME=08:00:00
  SLURM_CPUS_PER_TASK=1
  SLURM_MEM=16G
  DRY_RUN=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TIME_COSTS="${1:-0.02,0.04,0.08,0.1}"
SIGMA_LIST="${2:-0,0.5,1.0}"
SEED_SPEC="${3:-1:3}"
MAX_OBSERVATIONS="${4:-10}"
GRID_SIZE="${5:-161}"
QUADRATURE_ORDER="${6:-21}"
SIMULATE_TRIALS="${7:-2000}"
OUTDIR="${8:-analyses/exp_binary/results/bayesian_revisit_2node}"

if (($# >= 8)); then
  shift 8
else
  shift $#
fi
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_bayesian_revisit_2node.sh"

if [[ ! -f "${SUBMIT_SCRIPT}" ]]; then
  echo "Could not find ${SUBMIT_SCRIPT}" >&2
  exit 1
fi

split_csv() {
  local value="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${value}"
}

parse_seed_spec() {
  local spec="$1"
  local -n out_ref="$2"
  out_ref=()
  if [[ "${spec}" == *:* ]]; then
    local start="${spec%%:*}"
    local end="${spec##*:}"
    if [[ -z "${start}" || -z "${end}" ]]; then
      echo "Invalid seed range: ${spec}" >&2
      exit 1
    fi
    if ((start <= end)); then
      for ((seed = start; seed <= end; seed++)); do
        out_ref+=("${seed}")
      done
    else
      for ((seed = start; seed >= end; seed--)); do
        out_ref+=("${seed}")
      done
    fi
  elif [[ "${spec}" == *,* ]]; then
    split_csv "${spec}" out_ref
  else
    out_ref=("${spec}")
  fi
}

safe_label() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+}"
  value="${value//,/}"
  echo "${value}"
}

mkdir -p logs "${OUTDIR}"

split_csv "${SIGMA_LIST}" SIGMAS
parse_seed_spec "${SEED_SPEC}" SEEDS

PARTITION="${SLURM_PARTITION:-general}"
TIME_LIMIT="${SLURM_TIME:-08:00:00}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
MEMORY="${SLURM_MEM:-16G}"

job_count=$(( ${#SIGMAS[@]} * ${#SEEDS[@]} ))
echo "Submitting ${job_count} Bayesian revisit 2-node job(s)."
echo "Time costs: ${TIME_COSTS}"
echo "Sigmas: ${SIGMAS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Max observations: ${MAX_OBSERVATIONS}"
echo "Grid size: ${GRID_SIZE}"
echo "Quadrature order: ${QUADRATURE_ORDER}"
echo "Simulate trials: ${SIMULATE_TRIALS}"
echo "Outdir: ${OUTDIR}"
echo "Slurm: partition=${PARTITION} time=${TIME_LIMIT} cpus=${CPUS_PER_TASK} mem=${MEMORY}"

job_index=0
for sigma in "${SIGMAS[@]}"; do
  sigma_trimmed="$(echo "${sigma}" | xargs)"
  [[ -z "${sigma_trimmed}" ]] && continue
  for seed in "${SEEDS[@]}"; do
    seed_trimmed="$(echo "${seed}" | xargs)"
    [[ -z "${seed_trimmed}" ]] && continue
    job_index=$((job_index + 1))
    job_name="bayes_r2_s$(safe_label "${sigma_trimmed}")_seed${seed_trimmed}"
    cmd=(
      sbatch
      -p "${PARTITION}"
      --time "${TIME_LIMIT}"
      --cpus-per-task "${CPUS_PER_TASK}"
      --mem "${MEMORY}"
      --job-name "${job_name}"
      "${SUBMIT_SCRIPT}"
      "${TIME_COSTS}"
      "${sigma_trimmed}"
      "${MAX_OBSERVATIONS}"
      "${GRID_SIZE}"
      "${QUADRATURE_ORDER}"
      "${SIMULATE_TRIALS}"
      "${OUTDIR}"
      "${seed_trimmed}"
      "${EXTRA_ARGS[@]}"
    )
    echo "[${job_index}/${job_count}] ${cmd[*]}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      "${cmd[@]}"
    fi
  done
done
