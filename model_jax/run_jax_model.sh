#!/bin/bash

set -euo pipefail

if [ "$#" -lt 11 ]; then
    echo "Usage: $0 loss_scale alpha lambda model_dir sim_dir trial_n input_type seed train opportunity_cost tree_size [expansion_decision_version] [model_variant] [tree_config] [rnn_units] [latent_dim] [extra_jax_args...]"
    echo "Example: $0 100.0 0.0 10.0 outputs/jax_models/ outputs/jax_simulations/ 2000 uniform 31 train 0.0266666666667 3 lstm vae bandit3 32 16 --num-envs 200 --backend cpu --kl-start-multiplier 5 --kl-annealing-epochs 60 --sigma 0.5 --allow-node-revisit --max-observations-before-stop 10"
    echo "The third positional value is the direct paid-KL memory lambda; the old 1/beta convention is no longer used."
    echo "Extra JAX args include --pay-kl-on-stop to pay pending KL on terminal stop; filenames add _stop_paid."
    echo "Extra JAX args include --observer-only to force terminal choice only at the observation cap; filenames add _observer_endchoice."
    echo "Extra JAX args include --critic-huber-delta and --advantage-clip for PPO stabilization."
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
expansion_decision_version=${12:-"decoder"}
model_variant=${13:-"vae"}
tree_config=${14:-""}
rnn_units=${15:-"64"}
latent_dim=${16:-"32"}

shift $(( $# >= 16 ? 16 : $# ))

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

python -m model_jax.planning \
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
    "$@"
