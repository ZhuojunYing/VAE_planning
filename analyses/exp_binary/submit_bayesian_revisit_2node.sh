#!/usr/bin/env bash
#SBATCH --job-name=bayes_revisit2
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/bayes_revisit2-%j.out
#SBATCH --error=logs/bayes_revisit2-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch analyses/exp_binary/submit_bayesian_revisit_2node.sh \
    [time_costs] [sigma] [max_observations] [grid_size] [quadrature_order] \
    [simulate_trials] [outdir] [seed] [extra_solver_args...]

Defaults:
  time_costs          0.02,0.04,0.08,0.1
  sigma               1.0
  max_observations    10
  grid_size           161
  quadrature_order    21
  simulate_trials     0
  outdir              analyses/exp_binary/results/bayesian_revisit_2node
  seed                1

Example:
  sbatch analyses/exp_binary/submit_bayesian_revisit_2node.sh \
    "0.02,0.04,0.08,0.1" 1.0 10 161 21 2000 \
    analyses/exp_binary/results/bayesian_revisit_2node 1

Extra solver args can include:
  --grid-sigma-bound 8
  --min-observations-before-stop 1
  --tie-mode first
  --no-normalize-reward
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

TIME_COSTS="${1:-0.02,0.04,0.08,0.1}"
SIGMA="${2:-1.0}"
MAX_OBSERVATIONS="${3:-10}"
GRID_SIZE="${4:-161}"
QUADRATURE_ORDER="${5:-21}"
SIMULATE_TRIALS="${6:-0}"
OUTDIR="${7:-analyses/exp_binary/results/bayesian_revisit_2node}"
SEED="${8:-1}"

if (($# >= 8)); then
  shift 8
else
  shift $#
fi
EXTRA_ARGS=("$@")

if [ -f "vae_env/bin/activate" ]; then
  source vae_env/bin/activate
fi

echo "Running Bayesian optimal revisit 2-node DP"
echo "Time costs: ${TIME_COSTS}"
echo "Sigma: ${SIGMA}"
echo "Max observations: ${MAX_OBSERVATIONS}"
echo "Grid size: ${GRID_SIZE}"
echo "Quadrature order: ${QUADRATURE_ORDER}"
echo "Simulate trials: ${SIMULATE_TRIALS}"
echo "Outdir: ${OUTDIR}"
echo "Seed: ${SEED}"
echo "Extra args: ${EXTRA_ARGS[*]:-}"
echo "Host: $(hostname)"
echo "Started: $(date)"

python analyses/exp_binary/bayesian_optimal_revisit_2node.py \
  --time-costs "${TIME_COSTS}" \
  --sigma "${SIGMA}" \
  --max-observations "${MAX_OBSERVATIONS}" \
  --grid-size "${GRID_SIZE}" \
  --quadrature-order "${QUADRATURE_ORDER}" \
  --simulate-trials "${SIMULATE_TRIALS}" \
  --outdir "${OUTDIR}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo "Finished: $(date)"
