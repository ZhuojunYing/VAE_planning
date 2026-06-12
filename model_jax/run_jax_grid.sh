#!/bin/bash

set -euo pipefail

is_train_arg() {
    case "$1" in
        train|simulate|simulation|inference|eval|evaluate) return 0 ;;
        *) return 1 ;;
    esac
}

if ! {
    { [ "$#" -ge 16 ] && is_train_arg "${13}"; } ||
    { [ "$#" -ge 14 ] && is_train_arg "${11}"; } ||
    { [ "$#" -ge 12 ] && is_train_arg "${9}"; }
}; then
    echo "Usage legacy: $0 beta_min beta_max beta_steps alpha_min alpha_max alpha_steps opportunity_min opportunity_max opportunity_steps lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim] [num_updates] [num_envs] [extra_jax_args...]"
    echo "Usage opp/lambda lists: $0 beta_min beta_max beta_steps alpha_min alpha_max alpha_steps opportunity_list lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim] [num_updates] [num_envs] [extra_jax_args...]"
    echo "Usage beta/opp/lambda lists: $0 beta_list alpha_min alpha_max alpha_steps opportunity_list lambda_list seed_min seed_max train input_type tree_size expansion_decision_version [model_variant] [tree_config] [rnn_units] [latent_dim] [num_updates] [num_envs] [extra_jax_args...]"
    exit 1
fi

if [ "$#" -ge 16 ] && is_train_arg "${13}"; then
    arg_mode="legacy"
elif [ "$#" -ge 14 ] && is_train_arg "${11}"; then
    arg_mode="opportunity_lambda_lists"
else
    arg_mode="beta_opportunity_lambda_lists"
fi

if [ "$arg_mode" = "legacy" ]; then
    beta_arg=$1; beta_max=$2; beta_steps=$3
    alpha_min=$4; alpha_max=$5; alpha_steps=$6
    opportunity_arg=$7; opportunity_max=$8; opportunity_steps=$9
    lambda_str=${10}; seed_min=${11}; seed_max=${12}
    train=${13}; input_type=${14}; tree_size=${15}; expansion_decision_version=${16}
    model_variant=${17:-"vae"}; tree_config=${18:-""}; rnn_units=${19:-"64"}; latent_dim=${20:-"32"}
    num_updates=${21:-"24000"}; num_envs=${22:-"200"}
    extra_args=("${@:23}")
elif [ "$arg_mode" = "opportunity_lambda_lists" ]; then
    beta_arg=$1; beta_max=$2; beta_steps=$3
    alpha_min=$4; alpha_max=$5; alpha_steps=$6
    opportunity_arg=$7; opportunity_max=""; opportunity_steps=""
    lambda_str=$8; seed_min=$9; seed_max=${10}
    train=${11}; input_type=${12}; tree_size=${13}; expansion_decision_version=${14}
    model_variant=${15:-"vae"}; tree_config=${16:-""}; rnn_units=${17:-"64"}; latent_dim=${18:-"32"}
    num_updates=${19:-"24000"}; num_envs=${20:-"200"}
    extra_args=("${@:21}")
else
    beta_arg=$1; beta_max=""; beta_steps=""
    alpha_min=$2; alpha_max=$3; alpha_steps=$4
    opportunity_arg=$5; opportunity_max=""; opportunity_steps=""
    lambda_str=$6; seed_min=$7; seed_max=$8
    train=$9; input_type=${10}; tree_size=${11}; expansion_decision_version=${12}
    model_variant=${13:-"vae"}; tree_config=${14:-""}; rnn_units=${15:-"64"}; latent_dim=${16:-"32"}
    num_updates=${17:-"24000"}; num_envs=${18:-"200"}
    extra_args=("${@:19}")
fi

mkdir -p logs

jax_cpus_per_task=${JAX_CPUS_PER_TASK:-8}
jax_mem=${JAX_MEM:-16G}
jax_slurm_time=${JAX_SLURM_TIME:-10:00:00}

echo "JAX Slurm resources: cpus-per-task=${jax_cpus_per_task}, mem=${jax_mem}, time=${jax_slurm_time}"

python - "$beta_arg" "$beta_max" "$beta_steps" "$alpha_min" "$alpha_max" "$alpha_steps" \
    "$opportunity_arg" "$opportunity_max" "$opportunity_steps" "$lambda_str" \
    "$seed_min" "$seed_max" "$train" "$input_type" "$tree_size" "$expansion_decision_version" \
    "$model_variant" "$tree_config" "$rnn_units" "$latent_dim" "$num_updates" "$num_envs" \
    "$jax_cpus_per_task" "$jax_mem" "$jax_slurm_time" "${extra_args[@]}" <<'PY' |
import itertools
import shlex
import sys
import numpy as np

(
    beta_arg, beta_max, beta_steps,
    alpha_min, alpha_max, alpha_steps,
    opportunity_arg, opportunity_max, opportunity_steps,
    lambda_str, seed_min, seed_max, train, input_type, tree_size,
    expansion_decision_version, model_variant, tree_config, rnn_units, latent_dim,
    num_updates, num_envs, jax_cpus_per_task, jax_mem, jax_slurm_time, *extra_args
) = sys.argv[1:]

def parse_list(raw):
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected values in {raw!r}")
    return vals

def parse_range_or_list(first, last, steps):
    if str(last).strip() == "" and str(steps).strip() == "":
        return parse_list(first)
    return list(np.linspace(float(first), float(last), int(steps)))

betas = parse_range_or_list(beta_arg, beta_max, beta_steps)
alphas = list(np.linspace(float(alpha_min), float(alpha_max), int(alpha_steps)))
opps = parse_range_or_list(opportunity_arg, opportunity_max, opportunity_steps)
lambdas = parse_list(lambda_str)
seeds = range(int(seed_min), int(seed_max) + 1)
tree_size_i = int(tree_size)
num_steps = tree_size_i
steps_per_epoch = int(num_updates) * int(num_envs) * num_steps // 120
steps_per_epoch = max(steps_per_epoch, int(num_envs) * num_steps)

for seed, beta, alpha, lambda_, opp in itertools.product(seeds, betas, alphas, lambdas, opps):
    cmd = [
        "sbatch",
        "-p", "general",
        "--time", jax_slurm_time,
        "--cpus-per-task", jax_cpus_per_task,
        "--mem", jax_mem,
        "--output=logs/slurm-%j.out",
        "model_jax/run_jax_model.sh",
        str(lambda_), str(alpha), str(beta),
        "outputs/jax_models/", "outputs/jax_simulations/", "2000",
        input_type, str(seed), train, str(opp), tree_size,
        expansion_decision_version, model_variant, tree_config, rnn_units, latent_dim,
        "--num-envs", num_envs,
        "--num-steps", str(num_steps),
        "--steps-per-epoch", str(steps_per_epoch),
        *extra_args,
    ]
    print(" ".join(shlex.quote(part) for part in cmd))
PY
while read -r cmd; do
    echo "Executing: $cmd"
    eval "$cmd"
    sleep 1
done
