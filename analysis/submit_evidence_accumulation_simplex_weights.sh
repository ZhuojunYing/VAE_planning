#!/usr/bin/env bash
#SBATCH --job-name=evi_simplex
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --output=logs/evi_simplex-%j.out
#SBATCH --error=logs/evi_simplex-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch analysis/submit_evidence_accumulation_simplex_weights.sh [plot args...]

Examples:
  sbatch analysis/submit_evidence_accumulation_simplex_weights.sh evidence \
    --observer-only \
    --simple-fixed-obsstd 1 \
    --simple-coherence-values "0.05,0.2,0.4,0.8" \
    --min-trials-per-fit 50 \
    --num-cv-folds 0

  sbatch --time=04:00:00 --mem=16G \
    analysis/submit_evidence_accumulation_simplex_weights.sh evidence \
    --observer-only \
    --simple-fixed-obsstd 0.5 \
    --simple-coherence-values "0.05,0.2,0.4,0.8" \
    --z-mu-simplex-dims "0"

If no plot args are provided, this wrapper runs:
  analysis/plot_evidence_accumulation_simplex_weights.py evidence
using the Python script's preset defaults.

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
  set -- evidence
fi

echo "Running evidence-accumulation simplex-weight analysis"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Arguments: $*"

"${PYTHON_BIN}" analysis/plot_evidence_accumulation_simplex_weights.py "$@"

echo "Finished: $(date)"
