#!/usr/bin/env bash
#SBATCH --job-name=evi_optimal
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/evi_optimal-%j.out
#SBATCH --error=logs/evi_optimal-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch model_jax/submit_evidence_accumulation_optimal.sh [optimal args...]

This wrapper runs:
  vae_env/bin/python model_jax/evidence_accumulation_optimal.py [optimal args...]

Example:
  sbatch model_jax/submit_evidence_accumulation_optimal.sh \
    --opportunity-costs "0.001,0.005,0.01" \
    --coherence-values "0,0.05,0.1,0.2,0.4,0.8" \
    --observation-noise-std "0.1,0.5,1.0" \
    --seeds "4,5,6" \
    --num-trials 10000 \
    --max-observations-before-stop 10 \
    --correct-reward 5 \
    --sim-dir outputs/jax_simulations_evi_optimal \
    --policy-dir outputs/evidence_optimal_policies \
    --reuse-policy

If no args are provided, the Python script's own defaults are used.

Optional environment overrides:
  PYTHON_BIN=vae_env/bin/python
  JAX_PLATFORM_NAME=cpu
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  MPLCONFIGDIR=/tmp/matplotlib

SLURM resources can be overridden at submission time, for example:
  sbatch --time=24:00:00 --mem=32G model_jax/submit_evidence_accumulation_optimal.sh ...
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

echo "Running Bayes-optimal evidence accumulation"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "JAX_PLATFORM_NAME: ${JAX_PLATFORM_NAME}"
echo "MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Arguments: $*"

"${PYTHON_BIN}" model_jax/evidence_accumulation_optimal.py "$@"

echo "Finished: $(date)"
