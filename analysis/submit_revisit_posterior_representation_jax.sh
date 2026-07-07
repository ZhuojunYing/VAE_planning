#!/usr/bin/env bash
#SBATCH --job-name=revisit_h2_probe
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/revisit_h2_probe-%j.out
#SBATCH --error=logs/revisit_h2_probe-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch analysis/submit_revisit_posterior_representation_jax.sh [analysis args...]

Example:
  sbatch analysis/submit_revisit_posterior_representation_jax.sh default \
    --vary-beta-values "10,20,80" \
    --vary-opportunity-values "0.06,0.2,0.4" \
    --sigmas "0,0.5,1.0,2.0" \
    --probe-n-states 4000 \
    --max-history-length 10 \
    --probe-decoder-types deep \
    --deep-probe-hidden-dims "64,32" \
    --deep-probe-epochs 300 \
    --h2-extra-tests terminal_history_value_error \
    --terminal-decoder-target-mode all \
    --terminal-decoder-representation-mode sample \
    --max-analysis-states-per-combo 2000 \
    --full-h2-plots

If no analysis args are provided, the example arguments above are used.

Optional environment overrides:
  PYTHON_BIN=vae_env/bin/python
  JAX_PLATFORM_NAME=cpu
  XLA_PYTHON_CLIENT_PREALLOCATE=false

SLURM resources can be overridden at submission time, for example:
  sbatch --time=12:00:00 --mem=32G analysis/submit_revisit_posterior_representation_jax.sh ...
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
    default \
    --vary-beta-values "10,20,80" \
    --vary-opportunity-values "0.06,0.2,0.4" \
    --sigmas "0,0.5,1.0,2.0" \
    --probe-n-states 4000 \
    --max-history-length 10 \
    --probe-decoder-types deep \
    --deep-probe-hidden-dims "64,32" \
    --deep-probe-epochs 300 \
    --h2-extra-tests terminal_history_value_error \
    --terminal-decoder-target-mode all \
    --terminal-decoder-representation-mode sample \
    --max-analysis-states-per-combo 2000 \
    --full-h2-plots
fi

echo "Running revisit hypothesis-2 posterior-representation analysis"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "JAX_PLATFORM_NAME: ${JAX_PLATFORM_NAME}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Arguments: $*"

"${PYTHON_BIN}" analysis/test_revisit_posterior_representation_jax.py "$@"

echo "Finished: $(date)"
