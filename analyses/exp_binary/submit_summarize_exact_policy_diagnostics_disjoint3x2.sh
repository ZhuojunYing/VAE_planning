#!/usr/bin/env bash
#SBATCH --job-name=exact_diag_d3x2
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/exact_diag_d3x2-%j.out
#SBATCH --error=logs/exact_diag_d3x2-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

TASK="${1:-disjoint3x2}"
EXACT_DIR="${2:-analyses/exp_binary/results/exact_time_cost}"
ZERO_EXACT_DIR="${3:-analyses/exp_binary/results/exact_time_cost_zero}"
TIME_COSTS="${4:-0.00209396675828,0.0210510487933,0.0420040541888,0.0782545291379,0.103839811543,0.121501294259,0.152717337473,0.172982140391,0.206613347125,0.47691401649}"

if [ -f "vae_env/bin/activate" ]; then
  source vae_env/bin/activate
fi

echo "Running exact-policy diagnostic summary"
echo "Task: ${TASK}"
echo "Exact dir: ${EXACT_DIR}"
echo "Zero exact dir: ${ZERO_EXACT_DIR}"
echo "Time costs: ${TIME_COSTS}"
echo "Host: $(hostname)"
echo "Started: $(date)"

python analyses/exp_binary/summarize_exact_policy_diagnostics.py \
  --task "${TASK}" \
  --exact-dir "${EXACT_DIR}" \
  --zero-exact-dir "${ZERO_EXACT_DIR}" \
  --time-costs "${TIME_COSTS}"

echo "Finished: $(date)"
