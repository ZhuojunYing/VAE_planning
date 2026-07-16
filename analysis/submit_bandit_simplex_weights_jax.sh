#!/usr/bin/env bash
#SBATCH --job-name=bandit_simplex
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --output=logs/bandit_simplex-%j.out
#SBATCH --error=logs/bandit_simplex-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch analysis/submit_bandit_simplex_weights_jax.sh [analysis args...]

Examples:
  sbatch analysis/submit_bandit_simplex_weights_jax.sh default \
    --vary-memory-lambda-values "10,20,80" \
    --vary-opportunity-values "0.06,0.2,0.4" \
    --sigmas "0,0.5,1.0,2.0" \
    --seeds "1,2,3" \
    --rnn-units 16 \
    --latent-dim 2 \
    --max-observations-before-stop 10 \
    --n-trials 2000 \
    --rollout-mode round_robin

  sbatch --time=04:00:00 --mem=16G \
    analysis/submit_bandit_simplex_weights_jax.sh bandit3 \
    --tree-size 3 \
    --vary-memory-lambda-values "20,60,100" \
    --vary-opportunity-values "0.04,0.1,0.2" \
    --sigmas "0,0.5,1.0,2.0" \
    --rnn-units 32 \
    --latent-dim 16 \
    --max-observations-before-stop 10

If no analysis args are provided, this wrapper runs:
  analysis/plot_bandit_simplex_weights_jax.py default
using the Python script's defaults.

Optional environment overrides:
  PYTHON_BIN=vae_env/bin/python
  MPLCONFIGDIR=/tmp/matplotlib
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-vae_env/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if (($# == 0)); then
  set -- default
fi

echo "Running bandit/revisit simplex-weight analysis"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Arguments: $*"

"${PYTHON_BIN}" analysis/plot_bandit_simplex_weights_jax.py "$@"

echo "Finished: $(date)"
