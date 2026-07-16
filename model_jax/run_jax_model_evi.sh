#!/bin/bash

set -euo pipefail

if [ "$#" -lt 11 ]; then
    echo "Usage: $0 loss_scale alpha lambda model_dir sim_dir trial_n input_type seed train opportunity_cost tree_size [expansion_decision_version] [model_variant] [tree_config] [rnn_units] [latent_dim] [extra_evidence_args...]"
    echo "Example: $0 100.0 0.0 0.001 outputs/jax_models_evi/ outputs/jax_simulations_evi/ 2000 evidence 1 train 0.02 2 lstm vae evidence 32 16 --coherence-values 0,0.05,0.1,0.2,0.4,0.8 --observation-noise-std 1.0 --max-observations-before-stop 10 --correct-reward 5"
    echo "The third positional value is the direct paid-KL memory lambda unless --memory-lambda overrides it; the old 1/beta convention is no longer used."
    echo "You can also set MEMORY_LAMBDA to pass --memory-lambda automatically."
    echo "Extra evidence args include --pay-kl-on-stop to pay memory KL on terminal choices; filenames add _stop_paid."
    echo "Extra evidence args include --critic-huber-delta and --advantage-clip for PPO stabilization."
    echo "Correct terminal reward defaults to CORRECT_REWARD=5 unless --correct-reward is passed."
    exit 1
fi

loss_scale_string=$1
alpha_string=$2
memory_lambda_string=$3
model_dir_name=$4
sim_dir_name=$5
trial_n_string=$6
input_type=$7
seed=$8
train=$9
opportunity_cost_string=${10}
tree_size=${11}
expansion_decision_version=${12:-"lstm"}
model_variant=${13:-"vae"}
tree_config=${14:-"evidence"}
rnn_units=${15:-"32"}
latent_dim=${16:-"16"}

shift $(( $# >= 16 ? 16 : $# ))
extra_args=("$@")

has_any_extra_arg() {
    local arg flag
    for arg in "${extra_args[@]:-}"; do
        for flag in "$@"; do
            if [ "$arg" = "$flag" ] || [[ "$arg" == "$flag="* ]]; then
                return 0
            fi
        done
    done
    return 1
}

if ! has_any_extra_arg "--correct-reward"; then
    extra_args+=("--correct-reward" "${CORRECT_REWARD:-5}")
fi

if ! has_any_extra_arg "--memory-lambda" "--kl-lambda"; then
    if [ -n "${MEMORY_LAMBDA:-}" ]; then
        extra_args+=("--memory-lambda" "${MEMORY_LAMBDA}")
    fi
fi

append_xla_flag() {
    case " ${XLA_FLAGS:-} " in
        *" $1 "*) ;;
        *) export XLA_FLAGS="${XLA_FLAGS:-} $1" ;;
    esac
}

append_xla_flag_if_unset() {
    local flag_name="${1%%=*}"
    case " ${XLA_FLAGS:-} " in
        *" ${flag_name}="*|*" ${flag_name} "*) ;;
        *) append_xla_flag "$1" ;;
    esac
}

append_xla_flag_if_unset "--xla_cpu_use_xla_runtime=false"
export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-1}"

case "${JAX_DISABLE_THUNK_RUNTIME:-}" in
    1|true|TRUE|yes|YES)
        append_xla_flag_if_unset "--xla_cpu_use_thunk_runtime=false"
        ;;
esac

if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    jax_thread_count="$SLURM_CPUS_PER_TASK"
else
    jax_thread_count="${JAX_NUM_THREADS:-}"
fi

if [ -n "${jax_thread_count:-}" ]; then
    export OMP_NUM_THREADS="$jax_thread_count"
    export OPENBLAS_NUM_THREADS="$jax_thread_count"
    export MKL_NUM_THREADS="$jax_thread_count"
    export NUMEXPR_NUM_THREADS="$jax_thread_count"
    append_xla_flag "--xla_cpu_multi_thread_eigen=true"
    append_xla_flag "intra_op_parallelism_threads=${jax_thread_count}"
fi

if [ -f vae_env/bin/activate ]; then
    source vae_env/bin/activate
fi

python -m model_jax.evidence_accumulation \
    "$loss_scale_string" \
    "$alpha_string" \
    "$memory_lambda_string" \
    "$model_dir_name" \
    "120" \
    "$input_type" \
    "$seed" \
    "$tree_size" \
    "$train" \
    "$tree_config" \
    "$opportunity_cost_string" \
    "$expansion_decision_version" \
    "$model_variant" \
    "$rnn_units" \
    "$latent_dim" \
    --sim-dir "$sim_dir_name" \
    --n-sim-trials "$trial_n_string" \
    "${extra_args[@]}"
