#!/usr/bin/env bash
#SBATCH --job-name=sample_pair_kl
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/sample_pair_kl-%j.out
#SBATCH --error=logs/sample_pair_kl-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch analysis/submit_sample_set_pairwise_last_paid_kl_jax.sh [analysis args...]

Example:
  sbatch analysis/submit_sample_set_pairwise_last_paid_kl_jax.sh bandit3 \
    --vary-beta-values "20,60,100" \
    --vary-opportunity-values "0.04,0.1,0.2" \
    --sigmas "0,0.5,1.0,2.0" \
    --seeds "1,2,3" \
    --n-sample-sets 50 \
    --n-reward-combinations 100 \
    --min-samples 2

If no analysis args are provided, the example arguments above are used.

Optional environment overrides:
  PYTHON_BIN=vae_env/bin/python
  JAX_PLATFORM_NAME=cpu
  XLA_PYTHON_CLIENT_PREALLOCATE=false

SLURM resources can be overridden at submission time, for example:
  sbatch --time=12:00:00 --mem=32G analysis/submit_sample_set_pairwise_last_paid_kl_jax.sh ...
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

export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

if (($# == 0)); then
  set -- \
    bandit3 \
    --vary-beta-values "20,60,100" \
    --vary-opportunity-values "0.04,0.1,0.2" \
    --sigmas "0,0.5,1.0,2.0" \
    --seeds "1,2,3" \
    --n-sample-sets 50 \
    --n-reward-combinations 100 \
    --min-samples 2
fi

echo "Running sample-set pairwise last-paid KL analysis"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "JAX_PLATFORM_NAME: ${JAX_PLATFORM_NAME}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Arguments: $*"

"${PYTHON_BIN}" analysis/sample_set_pairwise_last_paid_kl_jax.py "$@"

echo "Finished: $(date)"
