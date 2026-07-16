#!/usr/bin/env bash
#SBATCH --job-name=evi_fixed
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/evi_fixed-%j.out
#SBATCH --error=logs/evi_fixed-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch model_jax/submit_evidence_accumulation_fixed_duration.sh [fixed-duration args...]

This wrapper runs:
  vae_env/bin/python model_jax/evidence_accumulation_fixed_duration.py [fixed-duration args...]

Positional args start with:
  loss_scale alpha beta model_dir epochs input_type seed tree_size eval_mode tree_type opportunity_cost ...

Memory KL is weighted directly by beta unless --memory-lambda is supplied.
The old 1/beta memory weighting convention is not used for new runs.

Example:
  sbatch model_jax/submit_evidence_accumulation_fixed_duration.sh \
    100.0 0.0 "0.00001,0.0001,0.001,0.01" outputs/jax_models_evi 120 evidence 4 2 sim evidence "0.0" lstm vae 16 1 \
    --seeds "4,5,6" \
    --sim-dir outputs/jax_simulations_evi_fixed_duration \
    --coherence-values "0,0.05,0.1,0.2,0.4,0.8" \
    --observation-noise-std "0.1,0.5,1.0" \
    --n-sim-trials 10000 \
    --max-observations-before-stop 10 \
    --correct-reward 5 \
    --pay-kl-on-stop \
    --observer-only \
    --checkpoints final

If no args are provided, the Python script's own defaults are used.

Optional environment overrides:
  PYTHON_BIN=vae_env/bin/python
  JAX_PLATFORM_NAME=cpu
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  MPLCONFIGDIR=/tmp/matplotlib
  EXPANSION_RETURN_TARGET=sampled_lambda
  SAMPLED_LAMBDA_CRITIC=q|value
  MEMORY_LAMBDA=<override paid-KL coefficient>
  CHOICE_AT_END_ONLY=1  # equivalent to passing --observer-only

SLURM resources can be overridden at submission time, for example:
  sbatch --time=01:00:00 --mem=32G model_jax/submit_evidence_accumulation_fixed_duration.sh ...
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

echo "Running fixed-duration evidence accumulation evaluation"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "JAX_PLATFORM_NAME: ${JAX_PLATFORM_NAME}"
echo "MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "MEMORY_LAMBDA: ${MEMORY_LAMBDA:-<default: beta positional value>}"
echo "CHOICE_AT_END_ONLY: ${CHOICE_AT_END_ONLY:-<default: off unless --observer-only is passed>}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Arguments: $*"

"${PYTHON_BIN}" model_jax/evidence_accumulation_fixed_duration.py "$@"

echo "Finished: $(date)"
