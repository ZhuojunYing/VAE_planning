#!/usr/bin/env bash
#SBATCH --job-name=revisit_gauss_pga
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/revisit_gaussian_pga-%j.out
#SBATCH --error=logs/revisit_gaussian_pga-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch analysis/submit_revisit_latent_density_gaussian_pga_jax.sh [analysis args...]

Example:
  sbatch analysis/submit_revisit_latent_density_gaussian_pga_jax.sh \
    --alphas 0.0 \
    --betas 1000.0 \
    --lambdas 100.0 \
    --opportunity-costs 0.02 0.1 \
    --sigmas 0 0.5 1.0 \
    --seeds 1 \
    --rnn-dims 16 \
    --latent-dims 2 \
    --n-trials 2000 \
    --tree-size 2 \
    --tree-type default \
    --input-type uniform \
    --expansion-decision-version lstm \
    --model-variant vae \
    --max-observations-before-stop 10 \
    --checkpoint-root outputs/jax_models \
    --outdir analysis_outputs/revisit_latent_density_gaussian_pga \
    --reduction-method both \
    --pga-fit-scope pooled

If no analysis args are provided, the example arguments above are used.

Optional environment overrides:
  PYTHON_BIN=vae_env/bin/python
  JAX_PLATFORM_NAME=cpu
  XLA_PYTHON_CLIENT_PREALLOCATE=false

SLURM resources can be overridden at submission time, for example:
  sbatch --time=12:00:00 --mem=32G analysis/submit_revisit_latent_density_gaussian_pga_jax.sh ...
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
    --alphas 0.0 \
    --betas 1000.0 \
    --lambdas 100.0 \
    --opportunity-costs 0.02 0.1 \
    --sigmas 0 0.5 1.0 \
    --seeds 1 \
    --rnn-dims 16 \
    --latent-dims 2 \
    --n-trials 2000 \
    --tree-size 2 \
    --tree-type default \
    --input-type uniform \
    --expansion-decision-version lstm \
    --model-variant vae \
    --max-observations-before-stop 10 \
    --checkpoint-root outputs/jax_models \
    --outdir analysis_outputs/revisit_latent_density_gaussian_pga \
    --reduction-method both \
    --pga-fit-scope pooled
fi

echo "Running revisit Gaussian PGA/MDS latent analysis"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "JAX_PLATFORM_NAME: ${JAX_PLATFORM_NAME}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Arguments: $*"

"${PYTHON_BIN}" analysis/plot_revisit_latent_density_gaussian_pga_jax.py "$@"

echo "Finished: $(date)"
