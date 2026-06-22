#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash analyses/exp_binary/run_bayesian_revisit_2node_grid_local.sh \
    [time_costs] [sigma_list] [seed_spec] [max_observations] [grid_size] \
    [quadrature_order] [simulate_trials] [outdir] [extra_solver_args...]

Runs the Bayesian revisit 2-node grid locally, sequentially, without Slurm.
The argument order matches run_bayesian_revisit_2node_grid.sh.

Arguments:
  time_costs          Comma-separated costs solved within each run.
  sigma_list          Comma-separated observation-noise SDs.
  seed_spec           Either start:end, comma-list, or one seed.
  max_observations    Maximum observations before forced stop.
  grid_size           Sum-statistic interpolation grid size.
  quadrature_order    Gauss-Hermite quadrature order.
  simulate_trials     Number of optimal-policy simulation trials per run.
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
  bash analyses/exp_binary/run_bayesian_revisit_2node_grid_local.sh \
    "0.02,0.04,0.08,0.1" "0,0.5,1.0" "1:3" 10 161 21 2000

  bash analyses/exp_binary/run_bayesian_revisit_2node_grid_local.sh \
    "0.02,0.04" "1.5,2,3" "1,4,7" 20 201 31 5000 \
    analyses/exp_binary/results/bayesian_revisit_2node --grid-sigma-bound 8

Optional environment overrides:
  PYTHON=python
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p "${OUTDIR}"

if [[ -f "vae_env/bin/activate" ]]; then
  source vae_env/bin/activate
fi

PYTHON_BIN="${PYTHON:-python}"

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

split_csv "${SIGMA_LIST}" SIGMAS
parse_seed_spec "${SEED_SPEC}" SEEDS

run_count=$(( ${#SIGMAS[@]} * ${#SEEDS[@]} ))
echo "Running ${run_count} Bayesian revisit 2-node run(s) locally in sequence."
echo "Time costs: ${TIME_COSTS}"
echo "Sigmas: ${SIGMAS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Max observations: ${MAX_OBSERVATIONS}"
echo "Grid size: ${GRID_SIZE}"
echo "Quadrature order: ${QUADRATURE_ORDER}"
echo "Simulate trials: ${SIMULATE_TRIALS}"
echo "Outdir: ${OUTDIR}"
echo "Python: ${PYTHON_BIN}"

run_index=0
for sigma in "${SIGMAS[@]}"; do
  sigma_trimmed="$(echo "${sigma}" | xargs)"
  [[ -z "${sigma_trimmed}" ]] && continue
  for seed in "${SEEDS[@]}"; do
    seed_trimmed="$(echo "${seed}" | xargs)"
    [[ -z "${seed_trimmed}" ]] && continue
    run_index=$((run_index + 1))
    cmd=(
      "${PYTHON_BIN}"
      analyses/exp_binary/bayesian_optimal_revisit_2node.py
      --time-costs "${TIME_COSTS}"
      --sigma "${sigma_trimmed}"
      --max-observations "${MAX_OBSERVATIONS}"
      --grid-size "${GRID_SIZE}"
      --quadrature-order "${QUADRATURE_ORDER}"
      --simulate-trials "${SIMULATE_TRIALS}"
      --outdir "${OUTDIR}"
      --seed "${seed_trimmed}"
      "${EXTRA_ARGS[@]}"
    )
    echo "[${run_index}/${run_count}] ${cmd[*]}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      "${cmd[@]}"
    fi
  done
done
