#!/usr/bin/env bash
#SBATCH --job-name=sample_latent_traj
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --output=logs/sample_latent_traj-%j.out
#SBATCH --error=logs/sample_latent_traj-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch analysis/submit_sample_set_latent_trajectory_jax.sh [analysis args...]

Examples:
  sbatch analysis/submit_sample_set_latent_trajectory_jax.sh --min-samples-per-dot 20

  sbatch analysis/submit_sample_set_latent_trajectory_jax.sh default \
    --vary-beta-values "10,20,80" \
    --vary-opportunity-values "0.06,0.2,0.4" \
    --sigmas "0,0.5,1.0,2.0" \
    --seeds "1,2,3" \
    --n-sample-sets 50 \
    --force-first-observe-node 1 \
    --min-samples-per-dot 20

  sbatch analysis/submit_sample_set_latent_trajectory_jax.sh default \
    --plot-only \
    --plot-seeds "1,3" \
    --min-samples-per-dot 20

  sbatch analysis/submit_sample_set_latent_trajectory_jax.sh evidence \
    --loss-scale 100.0 \
    --vary-beta-values "0.00001,0.0001,0.001,0.01" \
    --vary-opportunity-values "0.001,0.005,0.01" \
    --sigmas "0.1,0.5,1.0" \
    --seeds "4,5,6" \
    --n-sample-sets 50 \
    --pay-kl-on-stop \
    --correct-reward 5 \
    --min-samples-per-dot 20

For evidence runs, --loss-scale is the task/action/critic loss scale. The beta
values are direct memory-KL coefficients unless --memory-lambda is supplied to
the underlying evidence model tooling.

  sbatch analysis/submit_sample_set_latent_trajectory_jax.sh evidence \
    --plot-only \
    --pay-kl-on-stop \
    --min-samples-per-dot 20

If no analysis args are provided, this wrapper runs:
  analysis/sample_set_latent_trajectory_jax.py --min-samples-per-dot 20
using the Python script's built-in defaults.

To use evidence-accumulation defaults when no analysis args are provided:
  SAMPLE_LATENT_TRAJ_DEFAULT=evidence sbatch analysis/submit_sample_set_latent_trajectory_jax.sh

Optional environment overrides:
  PYTHON_BIN=vae_env/bin/python
  JAX_PLATFORM_NAME=cpu
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  SAMPLE_LATENT_TRAJ_DEFAULT=revisit|evidence
  MEMORY_LAMBDA=<optional paid-KL coefficient override for evidence helpers>

SLURM resources can be overridden at submission time, for example:
  sbatch --time=04:00:00 --mem=16G analysis/submit_sample_set_latent_trajectory_jax.sh --plot-only ...
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

if (($# == 0)); then
  case "${SAMPLE_LATENT_TRAJ_DEFAULT:-revisit}" in
    evidence)
      set -- \
        evidence \
        --loss-scale 100.0 \
        --vary-beta-values "0.00001,0.0001,0.001,0.01" \
        --vary-opportunity-values "0.001,0.005,0.01" \
        --sigmas "0.1,0.5,1.0" \
        --seeds "4,5,6" \
        --n-sample-sets 50 \
        --pay-kl-on-stop \
        --correct-reward 5 \
        --min-samples-per-dot 20
      ;;
    revisit)
      set -- --min-samples-per-dot 20
      ;;
    *)
      echo "Unknown SAMPLE_LATENT_TRAJ_DEFAULT=${SAMPLE_LATENT_TRAJ_DEFAULT}. Expected revisit or evidence." >&2
      exit 1
      ;;
  esac
fi

echo "Running sample-set latent trajectory analysis"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "JAX_PLATFORM_NAME: ${JAX_PLATFORM_NAME}"
echo "MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "MEMORY_LAMBDA: ${MEMORY_LAMBDA:-<default: beta value in evidence helpers>}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Arguments: $*"

"${PYTHON_BIN}" analysis/sample_set_latent_trajectory_jax.py "$@"

echo "Finished: $(date)"
